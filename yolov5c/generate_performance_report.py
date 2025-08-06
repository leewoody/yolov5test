#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Performance Report
生成詳細的性能分析報告，包括mAP、Precision、Recall等指標
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import average_precision_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# 添加項目路徑
sys.path.append(str(Path(__file__).parent))

try:
    from train_joint_ultimate import UltimateJointModel, UltimateDataset
except ImportError:
    print("無法導入UltimateJointModel，使用基礎模型")
    from train_joint_simple import SimpleJointModel, MixedDataset as UltimateDataset


class PerformanceAnalyzer:
    def __init__(self, data_yaml, weights_path, output_dir="performance_analysis"):
        self.data_yaml = Path(data_yaml)
        self.weights_path = Path(weights_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 載入數據配置
        self.load_data_config()
        
        # 載入模型
        self.load_model()
        
        # 設置設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_data_config(self):
        """載入數據配置"""
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        self.detection_classes = config.get('names', [])
        self.classification_classes = ['PSAX', 'PLAX', 'A4C']
        self.num_detection_classes = len(self.detection_classes)
        self.num_classification_classes = len(self.classification_classes)
        
        print(f"✅ 數據配置載入完成:")
        print(f"   - 檢測類別: {self.detection_classes}")
        print(f"   - 分類類別: {self.classification_classes}")
    
    def load_model(self):
        """載入訓練好的模型"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.weights_path}")
        
        # 創建模型實例
        if 'UltimateJointModel' in globals():
            self.model = UltimateJointModel(
                num_classes=self.num_detection_classes
            )
        else:
            self.model = SimpleJointModel(
                num_detection_classes=self.num_detection_classes,
                num_classification_classes=self.num_classification_classes
            )
        
        # 載入權重
        checkpoint = torch.load(self.weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✅ 模型載入完成: {self.weights_path}")
    
    def evaluate_model(self):
        """評估模型性能"""
        print("\n🔍 開始模型評估...")
        
        # 載入測試數據
        test_dataset = UltimateDataset(
            data_dir=str(self.data_yaml),
            split='test',
            transform=None,
            nc=self.num_detection_classes
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        # 收集預測結果
        all_bbox_preds = []
        all_bbox_targets = []
        all_cls_preds = []
        all_cls_targets = []
        all_images = []
        
        with torch.no_grad():
            for batch_idx, (images, bbox_targets, cls_targets) in enumerate(test_loader):
                images = images.to(self.device)
                bbox_targets = bbox_targets.to(self.device)
                cls_targets = cls_targets.to(self.device)
                
                # 前向傳播
                bbox_preds, cls_preds = self.model(images)
                
                # 收集結果
                all_bbox_preds.append(bbox_preds.cpu().numpy())
                all_bbox_targets.append(bbox_targets.cpu().numpy())
                all_cls_preds.append(cls_preds.cpu().numpy())
                all_cls_targets.append(cls_targets.cpu().numpy())
                all_images.append(images.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    print(f"  處理進度: {batch_idx}/{len(test_loader)}")
        
        # 轉換為numpy數組
        self.bbox_preds = np.concatenate(all_bbox_preds, axis=0)
        self.bbox_targets = np.concatenate(all_bbox_targets, axis=0)
        self.cls_preds = np.concatenate(all_cls_preds, axis=0)
        self.cls_targets = np.concatenate(all_cls_targets, axis=0)
        self.images = np.concatenate(all_images, axis=0)
        
        print(f"✅ 評估數據收集完成: {len(self.bbox_preds)} 個樣本")
    
    def calculate_detection_metrics(self):
        """計算檢測任務指標"""
        print("\n📊 計算檢測任務指標...")
        
        # 計算MAE
        bbox_mae = np.mean(np.abs(self.bbox_preds - self.bbox_targets))
        
        # 計算IoU
        ious = self.calculate_iou(self.bbox_preds, self.bbox_targets)
        mean_iou = np.mean(ious)
        
        # 計算mAP (簡化版本)
        map_score = self.calculate_simplified_map()
        
        # 計算Precision和Recall
        precision, recall = self.calculate_detection_pr()
        
        detection_metrics = {
            'bbox_mae': bbox_mae,
            'mean_iou': mean_iou,
            'map_score': map_score,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall + 1e-8)
        }
        
        print(f"   - Bbox MAE: {bbox_mae:.4f}")
        print(f"   - Mean IoU: {mean_iou:.4f}")
        print(f"   - mAP: {map_score:.4f}")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {detection_metrics['f1_score']:.4f}")
        
        return detection_metrics
    
    def calculate_classification_metrics(self):
        """計算分類任務指標"""
        print("\n📊 計算分類任務指標...")
        
        # 轉換為分類標籤
        cls_pred_labels = np.argmax(self.cls_preds, axis=1)
        cls_target_labels = np.argmax(self.cls_targets, axis=1)
        
        # 計算準確率
        accuracy = np.mean(cls_pred_labels == cls_target_labels)
        
        # 計算詳細指標
        precision, recall, f1, support = precision_recall_fscore_support(
            cls_target_labels, cls_pred_labels, average='weighted'
        )
        
        # 計算混淆矩陣
        cm = confusion_matrix(cls_target_labels, cls_pred_labels)
        
        # 計算每個類別的指標
        class_report = classification_report(
            cls_target_labels, cls_pred_labels,
            target_names=self.classification_classes,
            output_dict=True
        )
        
        classification_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'confusion_matrix': cm,
            'class_report': class_report
        }
        
        print(f"   - 整體準確率: {accuracy:.4f}")
        print(f"   - 加權Precision: {precision:.4f}")
        print(f"   - 加權Recall: {recall:.4f}")
        print(f"   - 加權F1-Score: {f1:.4f}")
        
        # 打印每個類別的指標
        for i, class_name in enumerate(self.classification_classes):
            if i < len(class_report):
                class_metrics = class_report[class_name]
                print(f"   - {class_name}: Precision={class_metrics['precision']:.4f}, "
                      f"Recall={class_metrics['recall']:.4f}, F1={class_metrics['f1-score']:.4f}")
        
        return classification_metrics
    
    def calculate_iou(self, preds, targets):
        """計算IoU"""
        ious = []
        for pred, target in zip(preds, targets):
            # 簡化的IoU計算
            pred_area = pred[2] * pred[3]  # width * height
            target_area = target[2] * target[3]
            
            # 計算交集
            x1 = max(pred[0] - pred[2]/2, target[0] - target[2]/2)
            y1 = max(pred[1] - pred[3]/2, target[1] - target[3]/2)
            x2 = min(pred[0] + pred[2]/2, target[0] + target[2]/2)
            y2 = min(pred[1] + pred[3]/2, target[1] + target[3]/2)
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
            else:
                intersection = 0
            
            union = pred_area + target_area - intersection
            iou = intersection / (union + 1e-8)
            ious.append(iou)
        
        return np.array(ious)
    
    def calculate_simplified_map(self):
        """計算簡化版本的mAP"""
        # 使用IoU > 0.5作為正樣本
        ious = self.calculate_iou(self.bbox_preds, self.bbox_targets)
        positive_samples = (ious > 0.5).sum()
        total_samples = len(ious)
        
        # 簡化的mAP計算
        map_score = positive_samples / total_samples if total_samples > 0 else 0
        return map_score
    
    def calculate_detection_pr(self):
        """計算檢測任務的Precision和Recall"""
        ious = self.calculate_iou(self.bbox_preds, self.bbox_targets)
        
        # 使用IoU > 0.5作為正樣本
        tp = (ious > 0.5).sum()
        fp = (ious <= 0.5).sum()
        fn = 0  # 簡化假設
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        return precision, recall
    
    def generate_visualizations(self, detection_metrics, classification_metrics):
        """生成可視化圖表"""
        print("\n📈 生成可視化圖表...")
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 分類混淆矩陣
        plt.figure(figsize=(10, 8))
        cm = classification_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classification_classes,
                   yticklabels=self.classification_classes)
        plt.title('分類混淆矩陣', fontsize=16)
        plt.xlabel('預測標籤', fontsize=12)
        plt.ylabel('真實標籤', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'classification_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 檢測性能指標
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Bbox MAE分布
        bbox_errors = np.abs(self.bbox_preds - self.bbox_targets)
        axes[0, 0].hist(bbox_errors.flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Bbox MAE 分布')
        axes[0, 0].set_xlabel('MAE')
        axes[0, 0].set_ylabel('頻率')
        
        # IoU分布
        ious = self.calculate_iou(self.bbox_preds, self.bbox_targets)
        axes[0, 1].hist(ious, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('IoU 分布')
        axes[0, 1].set_xlabel('IoU')
        axes[0, 1].set_ylabel('頻率')
        
        # 分類準確率
        cls_pred_labels = np.argmax(self.cls_preds, axis=1)
        cls_target_labels = np.argmax(self.cls_targets, axis=1)
        correct_predictions = (cls_pred_labels == cls_target_labels)
        axes[1, 0].bar(['正確', '錯誤'], [correct_predictions.sum(), (~correct_predictions).sum()], 
                      color=['green', 'red'])
        axes[1, 0].set_title('分類預測結果')
        axes[1, 0].set_ylabel('樣本數量')
        
        # 性能指標總結
        metrics_text = f"""
檢測任務:
- Bbox MAE: {detection_metrics['bbox_mae']:.4f}
- Mean IoU: {detection_metrics['mean_iou']:.4f}
- mAP: {detection_metrics['map_score']:.4f}
- Precision: {detection_metrics['precision']:.4f}
- Recall: {detection_metrics['recall']:.4f}

分類任務:
- 準確率: {classification_metrics['accuracy']:.4f}
- Precision: {classification_metrics['precision']:.4f}
- Recall: {classification_metrics['recall']:.4f}
- F1-Score: {classification_metrics['f1_score']:.4f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('性能指標總結')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可視化圖表已保存到: {self.output_dir}")
    
    def generate_report(self, detection_metrics, classification_metrics):
        """生成詳細報告"""
        print("\n📝 生成詳細報告...")
        
        report = f"""# 🤖 多任務模型性能分析報告

## 📋 基本信息
- **模型文件**: {self.weights_path}
- **數據配置**: {self.data_yaml}
- **測試樣本數**: {len(self.bbox_preds)}
- **生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 檢測任務性能

### 主要指標
- **Bbox MAE**: {detection_metrics['bbox_mae']:.4f}
- **Mean IoU**: {detection_metrics['mean_iou']:.4f}
- **mAP**: {detection_metrics['map_score']:.4f}
- **Precision**: {detection_metrics['precision']:.4f}
- **Recall**: {detection_metrics['recall']:.4f}
- **F1-Score**: {detection_metrics['f1_score']:.4f}

### 檢測類別: {self.detection_classes}

## 🎯 分類任務性能

### 主要指標
- **整體準確率**: {classification_metrics['accuracy']:.4f}
- **加權Precision**: {classification_metrics['precision']:.4f}
- **加權Recall**: {classification_metrics['recall']:.4f}
- **加權F1-Score**: {classification_metrics['f1_score']:.4f}

### 分類類別: {self.classification_classes}

### 每個類別的詳細指標
"""
        
        # 添加每個類別的詳細指標
        for i, class_name in enumerate(self.classification_classes):
            if i < len(classification_metrics['class_report']):
                class_metrics = classification_metrics['class_report'][class_name]
                report += f"""
#### {class_name}
- **Precision**: {class_metrics['precision']:.4f}
- **Recall**: {class_metrics['recall']:.4f}
- **F1-Score**: {class_metrics['f1-score']:.4f}
- **Support**: {class_metrics['support']}
"""
        
        report += f"""
## 📊 混淆矩陣

分類混淆矩陣已保存為圖片: `classification_confusion_matrix.png`

## 📈 性能圖表

詳細性能圖表已保存為: `performance_metrics.png`

## 🎯 性能評估

### 檢測任務評估
- **MAE < 0.1**: {'✅ 優秀' if detection_metrics['bbox_mae'] < 0.1 else '⚠️ 需要改進'}
- **IoU > 0.5**: {'✅ 優秀' if detection_metrics['mean_iou'] > 0.5 else '⚠️ 需要改進'}
- **mAP > 0.7**: {'✅ 優秀' if detection_metrics['map_score'] > 0.7 else '⚠️ 需要改進'}

### 分類任務評估
- **準確率 > 0.8**: {'✅ 優秀' if classification_metrics['accuracy'] > 0.8 else '⚠️ 需要改進'}
- **F1-Score > 0.7**: {'✅ 優秀' if classification_metrics['f1_score'] > 0.7 else '⚠️ 需要改進'}

## 🚀 改進建議

### 檢測任務改進
1. **增加訓練數據**: 特別是IoU較低的樣本
2. **調整模型架構**: 使用更深的網絡
3. **優化損失函數**: 調整bbox損失權重
4. **數據增強**: 增加更多樣化的數據增強

### 分類任務改進
1. **類別平衡**: 處理類別不平衡問題
2. **特徵提取**: 改進特徵提取能力
3. **正則化**: 增加Dropout等正則化技術
4. **學習率調度**: 優化學習率策略

---
*報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存報告
        report_file = self.output_dir / 'performance_analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 詳細報告已保存到: {report_file}")
        
        return report
    
    def run_analysis(self):
        """運行完整分析"""
        print("🚀 開始性能分析...")
        
        # 評估模型
        self.evaluate_model()
        
        # 計算指標
        detection_metrics = self.calculate_detection_metrics()
        classification_metrics = self.calculate_classification_metrics()
        
        # 生成可視化
        self.generate_visualizations(detection_metrics, classification_metrics)
        
        # 生成報告
        self.generate_report(detection_metrics, classification_metrics)
        
        print(f"\n✅ 性能分析完成！結果保存在: {self.output_dir}")
        
        return detection_metrics, classification_metrics


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='生成性能分析報告')
    parser.add_argument('--data', type=str, required=True, help='數據配置文件路徑')
    parser.add_argument('--weights', type=str, required=True, help='模型權重文件路徑')
    parser.add_argument('--output-dir', type=str, default='performance_analysis', help='輸出目錄')
    
    args = parser.parse_args()
    
    try:
        analyzer = PerformanceAnalyzer(
            data_yaml=args.data,
            weights_path=args.weights,
            output_dir=args.output_dir
        )
        
        detection_metrics, classification_metrics = analyzer.run_analysis()
        
        print("\n🎉 分析完成！")
        print(f"📁 輸出目錄: {args.output_dir}")
        print(f"📊 檢測MAE: {detection_metrics['bbox_mae']:.4f}")
        print(f"📊 分類準確率: {classification_metrics['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 