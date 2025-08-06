#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced YOLOv5 Classification Validation Script
提供完整的分類驗證指標，包括混淆矩陣、分類報告、ROC曲線等
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
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import create_classification_dataloader
from utils.general import (LOGGER, check_img_size, Profile)
from utils.torch_utils import select_device, smart_inference_mode


class EnhancedClassificationValidator:
    def __init__(self, weights, data, imgsz=224, batch_size=32, device='', save_dir='runs/enhanced_cls_val'):
        self.weights = weights
        self.data = data
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.device = select_device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'top1_accuracy': 0.0,
            'top5_accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'confusion_matrix': None,
            'classification_report': None,
            'roc_curves': {},
            'pr_curves': {},
            'class_metrics': {},
            'confidence_distribution': [],
            'prediction_confidence': []
        }
        
        # Load model and data
        self.load_model()
        self.load_data()
        
    def load_model(self):
        """Load YOLOv5 classification model"""
        self.model = DetectMultiBackend(self.weights, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        
        # Warmup
        self.model.warmup(imgsz=(1, 3, *[self.imgsz] * 2))
        
        LOGGER.info(f"Classification model loaded: {self.weights}")
        LOGGER.info(f"Classes: {self.names}")
        
    def load_data(self):
        """Load validation dataset"""
        data = Path(self.data)
        test_dir = data / 'test' if (data / 'test').exists() else data / 'val'
        
        if not test_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {test_dir}")
        
        # Create dataloader
        self.val_loader = create_classification_dataloader(
            path=test_dir,
            imgsz=self.imgsz,
            batch_size=self.batch_size,
            augment=False,
            rank=-1,
            workers=8
        )
        
        LOGGER.info(f"Classification dataset loaded: {len(self.val_loader.dataset)} images")
        
    @smart_inference_mode()
    def validate(self):
        """Run comprehensive classification validation"""
        LOGGER.info("Starting enhanced classification validation...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_confidences = []
        
        # Validation loop
        pbar = tqdm(self.val_loader, desc='Validating Classification')
        with torch.cuda.amp.autocast(enabled=self.device.type != 'cpu'):
            for images, labels in pbar:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device)
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(images)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = outputs.argmax(dim=1)
                    confidences = probabilities.max(dim=1)[0]
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        self.calculate_metrics(all_predictions, all_targets, all_probabilities, all_confidences)
        
        # Generate visualizations
        self.create_plots(all_predictions, all_targets, all_probabilities, all_confidences)
        
        # Generate report
        self.generate_report()
        
        # Print results
        self.print_results()
        
    def calculate_metrics(self, predictions, targets, probabilities, confidences):
        """Calculate comprehensive classification metrics"""
        num_classes = len(self.names)
        
        # Basic accuracy metrics
        correct = (predictions == targets).sum()
        total = len(targets)
        self.metrics['top1_accuracy'] = correct / total
        
        # Top-5 accuracy (if applicable)
        if num_classes >= 5:
            top5_correct = 0
            for i in range(len(targets)):
                top5_preds = np.argsort(probabilities[i])[-5:][::-1]
                if targets[i] in top5_preds:
                    top5_correct += 1
            self.metrics['top5_accuracy'] = top5_correct / total
        else:
            self.metrics['top5_accuracy'] = self.metrics['top1_accuracy']
        
        # Precision, Recall, F1
        self.metrics['precision'] = precision_score(targets, predictions, average='weighted', zero_division=0)
        self.metrics['recall'] = recall_score(targets, predictions, average='weighted', zero_division=0)
        self.metrics['f1_score'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(targets, predictions)
        
        # Classification report
        self.metrics['classification_report'] = classification_report(
            targets, predictions, 
            target_names=list(self.names.values()),
            output_dict=True
        )
        
        # Per-class metrics
        for i, class_name in self.names.items():
            if i < num_classes:
                class_mask = targets == i
                if class_mask.sum() > 0:
                    class_precision = precision_score(targets, predictions, pos_label=i, zero_division=0)
                    class_recall = recall_score(targets, predictions, pos_label=i, zero_division=0)
                    class_f1 = f1_score(targets, predictions, pos_label=i, zero_division=0)
                    
                    self.metrics['class_metrics'][class_name] = {
                        'precision': class_precision,
                        'recall': class_recall,
                        'f1_score': class_f1,
                        'support': class_mask.sum()
                    }
        
        # ROC curves (for binary and multiclass)
        if num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            self.metrics['roc_curves']['binary'] = {
                'fpr': fpr, 'tpr': tpr, 'auc': roc_auc
            }
        else:
            # Multiclass classification
            targets_bin = label_binarize(targets, classes=range(num_classes))
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(targets_bin[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                self.metrics['roc_curves'][self.names[i]] = {
                    'fpr': fpr, 'tpr': tpr, 'auc': roc_auc
                }
        
        # Precision-Recall curves
        if num_classes == 2:
            precision, recall, _ = precision_recall_curve(targets, probabilities[:, 1])
            self.metrics['pr_curves']['binary'] = {
                'precision': precision, 'recall': recall
            }
        else:
            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(targets_bin[:, i], probabilities[:, i])
                self.metrics['pr_curves'][self.names[i]] = {
                    'precision': precision, 'recall': recall
                }
        
        # Confidence distribution
        self.metrics['confidence_distribution'] = confidences.tolist()
        self.metrics['prediction_confidence'] = confidences[predictions == targets].tolist()
        
    def create_plots(self, predictions, targets, probabilities, confidences):
        """Create comprehensive visualization plots"""
        num_classes = len(self.names)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = self.metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.names.values()),
                   yticklabels=list(self.names.values()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence Distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Overall Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        correct_conf = confidences[predictions == targets]
        incorrect_conf = confidences[predictions != targets]
        
        plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence by Prediction Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        if num_classes == 2:
            roc_data = self.metrics['roc_curves']['binary']
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f'ROC curve (AUC = {roc_data["auc"]:.3f})')
        else:
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
            for i, (class_name, roc_data) in enumerate(self.metrics['roc_curves'].items()):
                plt.plot(roc_data['fpr'], roc_data['tpr'], color=next(colors),
                        label=f'{class_name} (AUC = {roc_data["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Curves
        plt.subplot(1, 2, 2)
        if num_classes == 2:
            pr_data = self.metrics['pr_curves']['binary']
            plt.plot(pr_data['recall'], pr_data['precision'], 
                    label='Precision-Recall curve')
        else:
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
            for i, (class_name, pr_data) in enumerate(self.metrics['pr_curves'].items()):
                plt.plot(pr_data['recall'], pr_data['precision'], color=next(colors),
                        label=f'{class_name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Per-class Performance
        if num_classes > 2:
            classes = list(self.names.values())
            precisions = [self.metrics['class_metrics'].get(c, {}).get('precision', 0) for c in classes]
            recalls = [self.metrics['class_metrics'].get(c, {}).get('recall', 0) for c in classes]
            f1_scores = [self.metrics['class_metrics'].get(c, {}).get('f1_score', 0) for c in classes]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].bar(classes, precisions, alpha=0.7)
            axes[0].set_title('Per-class Precision')
            axes[0].set_ylabel('Precision')
            axes[0].tick_params(axis='x', rotation=45)
            
            axes[1].bar(classes, recalls, alpha=0.7)
            axes[1].set_title('Per-class Recall')
            axes[1].set_ylabel('Recall')
            axes[1].tick_params(axis='x', rotation=45)
            
            axes[2].bar(classes, f1_scores, alpha=0.7)
            axes[2].set_title('Per-class F1-Score')
            axes[2].set_ylabel('F1-Score')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 分類驗證圖表已保存到: {self.save_dir}")
        
    def save_detailed_metrics(self):
        """Save detailed metrics to JSON"""
        metrics_for_json = {
            'overall_metrics': {
                'top1_accuracy': self.metrics['top1_accuracy'],
                'top5_accuracy': self.metrics['top5_accuracy'],
                'precision': self.metrics['precision'],
                'recall': self.metrics['recall'],
                'f1_score': self.metrics['f1_score']
            },
            'class_metrics': self.metrics['class_metrics'],
            'confidence_stats': {
                'mean': np.mean(self.metrics['confidence_distribution']),
                'std': np.std(self.metrics['confidence_distribution']),
                'min': np.min(self.metrics['confidence_distribution']),
                'max': np.max(self.metrics['confidence_distribution'])
            },
            'classification_report': self.metrics['classification_report']
        }
        
        with open(self.save_dir / 'classification_metrics.json', 'w') as f:
            json.dump(metrics_for_json, f, indent=2)
        
    def generate_report(self):
        """Generate comprehensive classification report"""
        report = f"""
# 🎯 增強分類驗證報告

## 📊 整體性能指標

### 主要指標
- **Top-1 準確率**: {self.metrics['top1_accuracy']:.4f} ({self.metrics['top1_accuracy']*100:.1f}%)
- **Top-5 準確率**: {self.metrics['top5_accuracy']:.4f} ({self.metrics['top5_accuracy']*100:.1f}%)
- **Precision**: {self.metrics['precision']:.4f}
- **Recall**: {self.metrics['recall']:.4f}
- **F1-Score**: {self.metrics['f1_score']:.4f}

### 置信度統計
- **平均置信度**: {np.mean(self.metrics['confidence_distribution']):.4f}
- **置信度標準差**: {np.std(self.metrics['confidence_distribution']):.4f}
- **最低置信度**: {np.min(self.metrics['confidence_distribution']):.4f}
- **最高置信度**: {np.max(self.metrics['confidence_distribution']):.4f}

## 📈 分類別性能

"""
        
        # Add per-class metrics
        for class_name, metrics in self.metrics['class_metrics'].items():
            report += f"""
### {class_name}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
- **Support**: {metrics['support']}

"""
        
        report += f"""
## 📁 生成的文件
- `confusion_matrix.png`: 混淆矩陣圖
- `confidence_analysis.png`: 置信度分析圖
- `roc_pr_curves.png`: ROC和PR曲線圖
- `per_class_performance.png`: 分類別性能圖
- `classification_metrics.json`: 詳細指標數據

## 🎯 建議
"""
        
        # Add recommendations based on metrics
        if self.metrics['top1_accuracy'] < 0.8:
            report += "- ⚠️ 分類準確率偏低，建議增加訓練數據或調整模型\n"
        if self.metrics['precision'] < 0.7:
            report += "- ⚠️ Precision偏低，建議檢查數據標註質量\n"
        if self.metrics['recall'] < 0.7:
            report += "- ⚠️ Recall偏低，建議增加少數類別的樣本\n"
        if np.mean(self.metrics['confidence_distribution']) < 0.7:
            report += "- ⚠️ 平均置信度偏低，建議改善模型訓練\n"
            
        report += "\n---\n*報告生成時間: " + str(pd.Timestamp.now()) + "*"
        
        with open(self.save_dir / 'classification_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed metrics
        self.save_detailed_metrics()
        
        LOGGER.info(f"Enhanced classification report saved to {self.save_dir}")
        
    def print_results(self):
        """Print validation results"""
        LOGGER.info('\n' + '='*60)
        LOGGER.info('ENHANCED CLASSIFICATION VALIDATION RESULTS')
        LOGGER.info('='*60)
        
        LOGGER.info(f"Top-1 Accuracy: {self.metrics['top1_accuracy']:.4f} ({self.metrics['top1_accuracy']*100:.1f}%)")
        LOGGER.info(f"Top-5 Accuracy: {self.metrics['top5_accuracy']:.4f} ({self.metrics['top5_accuracy']*100:.1f}%)")
        LOGGER.info(f"Precision: {self.metrics['precision']:.4f}")
        LOGGER.info(f"Recall: {self.metrics['recall']:.4f}")
        LOGGER.info(f"F1-Score: {self.metrics['f1_score']:.4f}")
        
        LOGGER.info('\nPer-class Performance:')
        for class_name, metrics in self.metrics['class_metrics'].items():
            LOGGER.info(f"  {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        LOGGER.info('='*60)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s-cls.pt', help='model weights path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size h,w')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='runs/enhanced_cls_val', help='save results to project/name')
    
    return parser.parse_args()


def main(opt):
    # Create validator
    validator = EnhancedClassificationValidator(
        weights=opt.weights,
        data=opt.data,
        imgsz=opt.imgsz,
        batch_size=opt.batch_size,
        device=opt.device,
        save_dir=opt.save_dir
    )
    
    # Run validation
    validator.validate()
    
    LOGGER.info("Enhanced classification validation completed!")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 