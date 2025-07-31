#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced YOLOv5 Validation Script
Provides comprehensive validation metrics including mAP, Precision, Recall, Confusion Matrix
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, non_max_suppression,
                          scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import plot_confusion_matrix
from utils.torch_utils import select_device, smart_inference_mode


class EnhancedValidator:
    def __init__(self, weights, data, imgsz=640, batch_size=32, conf_thres=0.001, 
                 iou_thres=0.6, device='', save_dir='runs/enhanced_val'):
        self.weights = weights
        self.data = data
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'mAP': [],
            'mAP50': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'confusion_matrix': None,
            'class_metrics': {},
            'confidence_distribution': [],
            'iou_distribution': []
        }
        
        # Load model and data
        self.load_model()
        self.load_data()
        
    def load_model(self):
        """Load YOLOv5 model"""
        self.model = DetectMultiBackend(self.weights, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        
        # Warmup
        self.model.warmup(imgsz=(1, 3, *[self.imgsz] * 2))
        
        LOGGER.info(f"Model loaded: {self.weights}")
        LOGGER.info(f"Classes: {self.names}")
        
    def load_data(self):
        """Load validation dataset"""
        self.data_dict = check_dataset(self.data)
        self.nc = int(self.data_dict['nc'])
        self.names = self.data_dict['names']
        
        # Create dataloader
        self.val_loader = create_dataloader(
            self.data_dict['val'],
            self.imgsz,
            self.batch_size,
            self.stride,
            pad=0.5,
            rect=False,
            workers=8,
            prefix='val'
        )[0]
        
        LOGGER.info(f"Validation dataset loaded: {len(self.val_loader.dataset)} images")
        
    def process_batch(self, detections, labels, iouv):
        """Return correct prediction matrix"""
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
                
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
    
    @smart_inference_mode()
    def validate(self):
        """Run comprehensive validation"""
        LOGGER.info("Starting enhanced validation...")
        
        # Initialize
        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  # IoU thresholds
        niou = iouv.numel()
        stats = []
        confusion_matrix = ConfusionMatrix(nc=self.nc)
        
        # Validation loop
        pbar = tqdm(self.val_loader, desc='Validating')
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            im = im.to(self.device, non_blocking=True).float() / 255
            
            # Inference
            preds = self.model(im)
            preds = non_max_suppression(
                preds, self.conf_thres, self.iou_thres, 
                labels=[], multi_label=True, agnostic=False, max_det=300
            )
            
            # Process predictions
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]
                path, shape = Path(paths[si]), shapes[si][0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=self.device)
                seen = 1
                
                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                    continue
                
                # Predictions
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
                
                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct = self.process_batch(predn, labelsn, iouv)
                    confusion_matrix.process_batch(predn, labelsn)
                    
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
                
                # Store confidence and IoU distributions
                if len(pred) > 0:
                    self.metrics['confidence_distribution'].extend(pred[:, 4].cpu().numpy())
                    if nl:
                        iou_values = box_iou(predn[:, :4], labelsn[:, 1:]).max(dim=1)[0]
                        self.metrics['iou_distribution'].extend(iou_values.cpu().numpy())
        
        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(
                *stats, plot=True, save_dir=self.save_dir, names=self.names
            )
            
            # Store metrics
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            
            self.metrics['mAP'].append(map)
            self.metrics['mAP50'].append(map50)
            self.metrics['precision'].append(mp)
            self.metrics['recall'].append(mr)
            self.metrics['f1'].append(f1.mean())
            
            # Store confusion matrix
            self.metrics['confusion_matrix'] = confusion_matrix.matrix
            
            # Store per-class metrics
            for i, c in enumerate(ap_class):
                class_name = self.names[c]
                self.metrics['class_metrics'][class_name] = {
                    'precision': p[i],
                    'recall': r[i],
                    'ap50': ap50[i],
                    'ap': ap[i],
                    'f1': f1[i]
                }
            
            # Print results
            self.print_results(mp, mr, map50, map, ap_class, p, r, ap50, ap)
            
        else:
            LOGGER.warning("No detections found, cannot compute metrics")
            
        # Generate comprehensive report
        self.generate_report()
        
    def print_results(self, mp, mr, map50, map, ap_class, p, r, ap50, ap):
        """Print validation results"""
        LOGGER.info('\n' + '='*60)
        LOGGER.info('ENHANCED VALIDATION RESULTS')
        LOGGER.info('='*60)
        
        # Overall metrics
        pf = '%22s' + '%11s' * 6
        LOGGER.info(pf % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95'))
        
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4
        LOGGER.info(pf % ('all', -1, -1, mp, mr, map50, map))
        
        # Per-class results
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (self.names[c], -1, -1, p[i], r[i], ap50[i], ap[i]))
            
        LOGGER.info('='*60)
        
    def generate_report(self):
        """Generate comprehensive validation report"""
        LOGGER.info("Generating enhanced validation report...")
        
        # Create plots
        self.create_plots()
        
        # Save detailed metrics
        self.save_detailed_metrics()
        
        # Generate summary report
        self.generate_summary_report()
        
    def create_plots(self):
        """Create comprehensive visualization plots"""
        # 1. Confusion Matrix
        if self.metrics['confusion_matrix'] is not None:
            plt.figure(figsize=(10, 8))
            plot_confusion_matrix(
                self.metrics['confusion_matrix'], 
                self.names, 
                save_dir=self.save_dir
            )
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Confidence Distribution
        if self.metrics['confidence_distribution']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.metrics['confidence_distribution'], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Detection Confidence Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.save_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. IoU Distribution
        if self.metrics['iou_distribution']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.metrics['iou_distribution'], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('IoU Score')
            plt.ylabel('Frequency')
            plt.title('IoU Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.save_dir / 'iou_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Per-class Performance
        if self.metrics['class_metrics']:
            classes = list(self.metrics['class_metrics'].keys())
            precisions = [self.metrics['class_metrics'][c]['precision'] for c in classes]
            recalls = [self.metrics['class_metrics'][c]['recall'] for c in classes]
            ap50s = [self.metrics['class_metrics'][c]['ap50'] for c in classes]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Precision
            axes[0].bar(classes, precisions, alpha=0.7)
            axes[0].set_title('Per-class Precision')
            axes[0].set_ylabel('Precision')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Recall
            axes[1].bar(classes, recalls, alpha=0.7)
            axes[1].set_title('Per-class Recall')
            axes[1].set_ylabel('Recall')
            axes[1].tick_params(axis='x', rotation=45)
            
            # mAP@0.5
            axes[2].bar(classes, ap50s, alpha=0.7)
            axes[2].set_title('Per-class mAP@0.5')
            axes[2].set_ylabel('mAP@0.5')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_detailed_metrics(self):
        """Save detailed metrics to JSON"""
        # Prepare metrics for JSON serialization
        metrics_for_json = {
            'overall_metrics': {
                'mAP': self.metrics['mAP'][-1] if self.metrics['mAP'] else 0.0,
                'mAP50': self.metrics['mAP50'][-1] if self.metrics['mAP50'] else 0.0,
                'precision': self.metrics['precision'][-1] if self.metrics['precision'] else 0.0,
                'recall': self.metrics['recall'][-1] if self.metrics['recall'] else 0.0,
                'f1': self.metrics['f1'][-1] if self.metrics['f1'] else 0.0
            },
            'class_metrics': self.metrics['class_metrics'],
            'confidence_stats': {
                'mean': np.mean(self.metrics['confidence_distribution']) if self.metrics['confidence_distribution'] else 0.0,
                'std': np.std(self.metrics['confidence_distribution']) if self.metrics['confidence_distribution'] else 0.0,
                'min': np.min(self.metrics['confidence_distribution']) if self.metrics['confidence_distribution'] else 0.0,
                'max': np.max(self.metrics['confidence_distribution']) if self.metrics['confidence_distribution'] else 0.0
            },
            'iou_stats': {
                'mean': np.mean(self.metrics['iou_distribution']) if self.metrics['iou_distribution'] else 0.0,
                'std': np.std(self.metrics['iou_distribution']) if self.metrics['iou_distribution'] else 0.0,
                'min': np.min(self.metrics['iou_distribution']) if self.metrics['iou_distribution'] else 0.0,
                'max': np.max(self.metrics['iou_distribution']) if self.metrics['iou_distribution'] else 0.0
            }
        }
        
        with open(self.save_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(metrics_for_json, f, indent=2)
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        report = f"""
# 🎯 增強驗證報告

## 📊 整體性能指標

### 主要指標
- **mAP@0.5:0.95**: {self.metrics['mAP'][-1]:.4f}
- **mAP@0.5**: {self.metrics['mAP50'][-1]:.4f}
- **Precision**: {self.metrics['precision'][-1]:.4f}
- **Recall**: {self.metrics['recall'][-1]:.4f}
- **F1-Score**: {self.metrics['f1'][-1]:.4f}

### 檢測統計
- **平均置信度**: {np.mean(self.metrics['confidence_distribution']):.4f}
- **置信度標準差**: {np.std(self.metrics['confidence_distribution']):.4f}
- **平均IoU**: {np.mean(self.metrics['iou_distribution']):.4f}

## 📈 分類別性能

"""
        
        # Add per-class metrics
        for class_name, metrics in self.metrics['class_metrics'].items():
            report += f"""
### {class_name}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **mAP@0.5**: {metrics['ap50']:.4f}
- **mAP@0.5:0.95**: {metrics['ap']:.4f}
- **F1-Score**: {metrics['f1']:.4f}

"""
        
        report += f"""
## 📁 生成的文件
- `confusion_matrix.png`: 混淆矩陣圖
- `confidence_distribution.png`: 置信度分佈圖
- `iou_distribution.png`: IoU分佈圖
- `per_class_performance.png`: 分類別性能圖
- `detailed_metrics.json`: 詳細指標數據

## 🎯 建議
"""
        
        # Add recommendations based on metrics
        if self.metrics['mAP'][-1] < 0.5:
            report += "- ⚠️ mAP偏低，建議增加訓練輪數或改善數據集\n"
        if self.metrics['precision'][-1] < 0.7:
            report += "- ⚠️ Precision偏低，建議調整置信度閾值\n"
        if self.metrics['recall'][-1] < 0.7:
            report += "- ⚠️ Recall偏低，建議增加訓練數據或調整模型\n"
        if np.mean(self.metrics['confidence_distribution']) < 0.5:
            report += "- ⚠️ 平均置信度偏低，建議改善模型訓練\n"
            
        report += "\n---\n*報告生成時間: " + str(pd.Timestamp.now()) + "*"
        
        with open(self.save_dir / 'validation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        LOGGER.info(f"Enhanced validation report saved to {self.save_dir}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='model weights path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size h,w')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='runs/enhanced_val', help='save results to project/name')
    
    return parser.parse_args()


def main(opt):
    # Create validator
    validator = EnhancedValidator(
        weights=opt.weights,
        data=opt.data,
        imgsz=opt.imgsz,
        batch_size=opt.batch_size,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.device,
        save_dir=opt.save_dir
    )
    
    # Run validation
    validator.validate()
    
    LOGGER.info("Enhanced validation completed!")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 