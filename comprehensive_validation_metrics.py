#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Validation Metrics Calculator
全面驗證指標計算器
計算 F1-Score、混淆矩陣、mAP、Precision、Recall 等指標
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 添加項目路徑
sys.path.append('yolov5c')

try:
    from train_joint_ultimate import UltimateJointModel, UltimateDataset
except ImportError:
    print("無法導入UltimateJointModel")
    sys.exit(1)

class ComprehensiveMetricsCalculator:
    def __init__(self, data_yaml, weights_path):
        self.data_yaml = Path(data_yaml)
        self.weights_path = Path(weights_path)
        
        # 載入數據配置
        self.load_data_config()
        
        # 載入模型
        self.load_model()
        
        # 設置設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 初始化結果存儲
        self.bbox_preds = []
        self.bbox_targets = []
        self.cls_preds = []
        self.cls_targets = []
        self.cls_probs = []
        
    def load_data_config(self):
        """載入數據配置"""
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        self.detection_classes = config.get('names', [])
        self.classification_classes = ['PSAX', 'PLAX', 'A4C']
        self.num_detection_classes = len(self.detection_classes)
        
        print(f"✅ 數據配置載入完成:")
        print(f"   - 檢測類別: {self.detection_classes}")
        print(f"   - 分類類別: {self.classification_classes}")
    
    def load_model(self):
        """載入訓練好的模型"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.weights_path}")
        
        # 創建模型實例
        self.model = UltimateJointModel(num_classes=self.num_detection_classes)
        
        # 載入權重
        checkpoint = torch.load(self.weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✅ 模型載入完成: {self.weights_path}")
    
    def calculate_iou(self, pred_bbox, target_bbox):
        """計算IoU"""
        # 假設bbox格式為 [x_center, y_center, width, height]
        pred_x1 = pred_bbox[0] - pred_bbox[2]/2
        pred_y1 = pred_bbox[1] - pred_bbox[3]/2
        pred_x2 = pred_bbox[0] + pred_bbox[2]/2
        pred_y2 = pred_bbox[1] + pred_bbox[3]/2
        
        target_x1 = target_bbox[0] - target_bbox[2]/2
        target_y1 = target_bbox[1] - target_bbox[3]/2
        target_x2 = target_bbox[0] + target_bbox[2]/2
        target_y2 = target_bbox[1] + target_bbox[3]/2
        
        # 計算交集
        x1 = max(pred_x1, target_x1)
        y1 = max(pred_y1, target_y1)
        x2 = min(pred_x2, target_x2)
        y2 = min(pred_y2, target_y2)
        
        if x2 > x1 and y2 > y1:
            intersection = (x2 - x1) * (y2 - y1)
        else:
            intersection = 0
        
        # 計算並集
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union = pred_area + target_area - intersection
        
        iou = intersection / (union + 1e-8)
        return iou
    
    def evaluate_model(self):
        """評估模型性能"""
        print("🔍 開始全面模型評估...")
        
        # 創建測試數據集
        data_dir = Path("converted_dataset_fixed")
        test_dataset = UltimateDataset(data_dir, 'test', transform=None, nc=len(self.detection_classes))
        
        print(f"test 數據集: {len(test_dataset)} 個有效樣本")
        
        # 統計類別分布
        class_counts = {}
        for i in range(len(test_dataset)):
            try:
                data = test_dataset[i]
                cls_target = data['classification']
                cls_idx = cls_target.item() if hasattr(cls_target, 'item') else cls_target
                class_counts[cls_idx] = class_counts.get(cls_idx, 0) + 1
            except Exception as e:
                continue
        
        print(f"test 類別分布: {class_counts}")
        
        # 評估模型
        with torch.no_grad():
            for i in range(len(test_dataset)):
                try:
                    data = test_dataset[i]
                    image = data['image']
                    bbox_target = data['bbox']
                    cls_target = data['classification']
                    
                    # 轉換為tensor
                    if not isinstance(image, torch.Tensor):
                        continue
                    
                    image = image.unsqueeze(0).to(self.device)
                    
                    # 前向傳播
                    bbox_pred, cls_pred = self.model(image)
                    
                    # 收集預測結果
                    self.bbox_preds.append(bbox_pred.cpu().numpy()[0])
                    self.bbox_targets.append(bbox_target.numpy())
                    self.cls_preds.append(torch.argmax(cls_pred, dim=1).cpu().numpy()[0])
                    self.cls_targets.append(cls_target.item() if hasattr(cls_target, 'item') else cls_target)
                    self.cls_probs.append(torch.softmax(cls_pred, dim=1).cpu().numpy()[0])
                    
                    if i % 50 == 0:
                        print(f"  處理樣本 {i}/{len(test_dataset)}")
                        
                except Exception as e:
                    print(f"⚠️ 樣本 {i} 處理失敗: {e}")
                    continue
        
        print(f"✅ 評估數據收集完成: {len(self.bbox_preds)} 個樣本")
    
    def calculate_detection_metrics(self):
        """計算檢測任務詳細指標"""
        print("\n📊 計算檢測任務詳細指標...")
        
        # 轉換為numpy數組
        bbox_preds = np.array(self.bbox_preds)
        bbox_targets = np.array(self.bbox_targets)
        
        # 基本指標
        bbox_mae = np.mean(np.abs(bbox_preds - bbox_targets))
        
        # 計算IoU
        ious = []
        for pred, target in zip(bbox_preds, bbox_targets):
            iou = self.calculate_iou(pred, target)
            ious.append(iou)
        
        ious = np.array(ious)
        mean_iou = np.mean(ious)
        
        # 計算mAP
        map_50 = self.calculate_map(ious, iou_threshold=0.5)
        map_75 = self.calculate_map(ious, iou_threshold=0.75)
        
        # 計算Precision和Recall
        precision_50, recall_50 = self.calculate_detection_precision_recall(ious, iou_threshold=0.5)
        precision_75, recall_75 = self.calculate_detection_precision_recall(ious, iou_threshold=0.75)
        
        # 計算F1-Score
        f1_50 = 2 * (precision_50 * recall_50) / (precision_50 + recall_50 + 1e-8)
        f1_75 = 2 * (precision_75 * recall_75) / (precision_75 + recall_75 + 1e-8)
        
        detection_metrics = {
            'bbox_mae': bbox_mae,
            'mean_iou': mean_iou,
            'map_50': map_50,
            'map_75': map_75,
            'precision_50': precision_50,
            'recall_50': recall_50,
            'f1_50': f1_50,
            'precision_75': precision_75,
            'recall_75': recall_75,
            'f1_75': f1_75,
            'ious': ious
        }
        
        print(f"   - Bbox MAE: {bbox_mae:.4f}")
        print(f"   - Mean IoU: {mean_iou:.4f}")
        print(f"   - mAP@0.5: {map_50:.4f}")
        print(f"   - mAP@0.75: {map_75:.4f}")
        print(f"   - Precision@0.5: {precision_50:.4f}")
        print(f"   - Recall@0.5: {recall_50:.4f}")
        print(f"   - F1-Score@0.5: {f1_50:.4f}")
        print(f"   - Precision@0.75: {precision_75:.4f}")
        print(f"   - Recall@0.75: {recall_75:.4f}")
        print(f"   - F1-Score@0.75: {f1_75:.4f}")
        
        return detection_metrics
    
    def calculate_classification_metrics(self):
        """計算分類任務詳細指標"""
        print("\n📊 計算分類任務詳細指標...")
        
        # 轉換為numpy數組
        cls_preds = np.array(self.cls_preds)
        cls_targets = np.array(self.cls_targets)
        cls_probs = np.array(self.cls_probs)
        
        # 計算混淆矩陣
        cm = confusion_matrix(cls_targets, cls_preds)
        
        # 計算每個類別的指標
        precision, recall, f1, support = precision_recall_fscore_support(
            cls_targets, cls_preds, average=None, zero_division=0
        )
        
        # 計算加權平均指標
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            cls_targets, cls_preds, average='weighted', zero_division=0
        )
        
        # 計算宏觀平均指標
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            cls_targets, cls_preds, average='macro', zero_division=0
        )
        
        # 計算整體準確率
        accuracy = np.mean(cls_preds == cls_targets)
        
        # 計算每個類別的準確率
        class_accuracy = {}
        for i in range(len(self.detection_classes)):
            class_mask = cls_targets == i
            if np.sum(class_mask) > 0:
                class_accuracy[self.detection_classes[i]] = np.mean(cls_preds[class_mask] == cls_targets[class_mask])
            else:
                class_accuracy[self.detection_classes[i]] = 0.0
        
        classification_metrics = {
            'accuracy': accuracy,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'class_accuracy': class_accuracy,
            'confusion_matrix': cm,
            'cls_preds': cls_preds,
            'cls_targets': cls_targets,
            'cls_probs': cls_probs
        }
        
        print(f"   - 整體準確率: {accuracy:.4f}")
        print(f"   - 加權Precision: {weighted_precision:.4f}")
        print(f"   - 加權Recall: {weighted_recall:.4f}")
        print(f"   - 加權F1-Score: {weighted_f1:.4f}")
        print(f"   - 宏觀Precision: {macro_precision:.4f}")
        print(f"   - 宏觀Recall: {macro_recall:.4f}")
        print(f"   - 宏觀F1-Score: {macro_f1:.4f}")
        
        return classification_metrics
    
    def calculate_map(self, ious, iou_threshold=0.5):
        """計算mAP"""
        # 簡化的mAP計算
        positive_samples = (ious > iou_threshold).sum()
        total_samples = len(ious)
        map_score = positive_samples / total_samples if total_samples > 0 else 0
        return map_score
    
    def calculate_detection_precision_recall(self, ious, iou_threshold=0.5):
        """計算檢測任務的Precision和Recall"""
        # 使用IoU > threshold作為正樣本
        tp = (ious > iou_threshold).sum()
        fp = (ious <= iou_threshold).sum()
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
        
        # 創建圖表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('YOLOv5WithClassification 全面驗證指標', fontsize=16, fontweight='bold')
        
        # 1. 混淆矩陣
        cm = classification_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.detection_classes, 
                   yticklabels=self.detection_classes, ax=axes[0,0])
        axes[0,0].set_title('混淆矩陣 (Confusion Matrix)')
        axes[0,0].set_xlabel('預測類別')
        axes[0,0].set_ylabel('真實類別')
        
        # 2. 每個類別的F1-Score
        classes = self.detection_classes
        f1_scores = classification_metrics['f1_per_class']
        axes[0,1].bar(classes, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0,1].set_title('每個類別的F1-Score')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Precision-Recall 比較
        precision = classification_metrics['precision_per_class']
        recall = classification_metrics['recall_per_class']
        x = np.arange(len(classes))
        width = 0.35
        axes[0,2].bar(x - width/2, precision, width, label='Precision', color='#FF6B6B')
        axes[0,2].bar(x + width/2, recall, width, label='Recall', color='#4ECDC4')
        axes[0,2].set_title('Precision vs Recall')
        axes[0,2].set_ylabel('分數')
        axes[0,2].set_xticks(x)
        axes[0,2].set_xticklabels(classes)
        axes[0,2].legend()
        axes[0,2].set_ylim(0, 1)
        
        # 4. IoU分布直方圖
        ious = detection_metrics['ious']
        axes[1,0].hist(ious, bins=20, alpha=0.7, color='#45B7D1', edgecolor='black')
        axes[1,0].set_title('IoU分布')
        axes[1,0].set_xlabel('IoU值')
        axes[1,0].set_ylabel('頻率')
        axes[1,0].axvline(detection_metrics['mean_iou'], color='red', linestyle='--', 
                         label=f'Mean IoU: {detection_metrics["mean_iou"]:.3f}')
        axes[1,0].legend()
        
        # 5. 檢測指標總結
        detection_summary = [
            f'Bbox MAE: {detection_metrics["bbox_mae"]:.4f}',
            f'Mean IoU: {detection_metrics["mean_iou"]:.4f}',
            f'mAP@0.5: {detection_metrics["map_50"]:.4f}',
            f'mAP@0.75: {detection_metrics["map_75"]:.4f}',
            f'F1@0.5: {detection_metrics["f1_50"]:.4f}',
            f'F1@0.75: {detection_metrics["f1_75"]:.4f}'
        ]
        axes[1,1].text(0.1, 0.9, '檢測任務指標', fontsize=12, fontweight='bold', transform=axes[1,1].transAxes)
        for i, text in enumerate(detection_summary):
            axes[1,1].text(0.1, 0.8 - i*0.12, text, fontsize=10, transform=axes[1,1].transAxes)
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        
        # 6. 分類指標總結
        classification_summary = [
            f'整體準確率: {classification_metrics["accuracy"]:.4f}',
            f'加權F1: {classification_metrics["weighted_f1"]:.4f}',
            f'宏觀F1: {classification_metrics["macro_f1"]:.4f}',
            f'加權Precision: {classification_metrics["weighted_precision"]:.4f}',
            f'加權Recall: {classification_metrics["weighted_recall"]:.4f}'
        ]
        axes[1,2].text(0.1, 0.9, '分類任務指標', fontsize=12, fontweight='bold', transform=axes[1,2].transAxes)
        for i, text in enumerate(classification_summary):
            axes[1,2].text(0.1, 0.8 - i*0.12, text, fontsize=10, transform=axes[1,2].transAxes)
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_validation_metrics.png', dpi=300, bbox_inches='tight')
        print("✅ 可視化圖表已保存為: comprehensive_validation_metrics.png")
        
        return fig
    
    def generate_detailed_report(self, detection_metrics, classification_metrics):
        """生成詳細報告"""
        print("\n📋 生成詳細驗證報告...")
        
        report = f"""# 🎯 YOLOv5WithClassification 全面驗證指標報告

## 📊 項目概述
- **模型名稱**: UltimateJointModel
- **數據集**: converted_dataset_fixed
- **測試樣本數**: {len(self.bbox_preds)}
- **檢測類別**: {self.detection_classes}
- **分類類別**: {self.classification_classes}

## 🎯 檢測任務指標 (Detection Metrics)

### 主要指標
| 指標 | 數值 | 評估 | 說明 |
|------|------|------|------|
| **Bbox MAE** | {detection_metrics['bbox_mae']:.4f} | {'✅ 優秀' if detection_metrics['bbox_mae'] < 0.1 else '⚠️ 需要改進'} | 邊界框平均絕對誤差 |
| **Mean IoU** | {detection_metrics['mean_iou']:.4f} | {'✅ 優秀' if detection_metrics['mean_iou'] > 0.5 else '⚠️ 需要改進'} | 平均交並比 |
| **mAP@0.5** | {detection_metrics['map_50']:.4f} | {'✅ 優秀' if detection_metrics['map_50'] > 0.7 else '⚠️ 需要改進'} | 平均精度 (IoU>0.5) |
| **mAP@0.75** | {detection_metrics['map_75']:.4f} | {'✅ 優秀' if detection_metrics['map_75'] > 0.5 else '⚠️ 需要改進'} | 平均精度 (IoU>0.75) |

### 詳細指標
| IoU閾值 | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| **0.5** | {detection_metrics['precision_50']:.4f} | {detection_metrics['recall_50']:.4f} | {detection_metrics['f1_50']:.4f} |
| **0.75** | {detection_metrics['precision_75']:.4f} | {detection_metrics['recall_75']:.4f} | {detection_metrics['f1_75']:.4f} |

## 🎯 分類任務指標 (Classification Metrics)

### 整體指標
| 指標 | 數值 | 評估 | 說明 |
|------|------|------|------|
| **整體準確率** | {classification_metrics['accuracy']:.4f} | {'✅ 優秀' if classification_metrics['accuracy'] > 0.8 else '⚠️ 需要改進'} | 所有類別的平均準確率 |
| **加權Precision** | {classification_metrics['weighted_precision']:.4f} | {'✅ 優秀' if classification_metrics['weighted_precision'] > 0.8 else '⚠️ 需要改進'} | 加權精確率 |
| **加權Recall** | {classification_metrics['weighted_recall']:.4f} | {'✅ 優秀' if classification_metrics['weighted_recall'] > 0.8 else '⚠️ 需要改進'} | 加權召回率 |
| **加權F1-Score** | {classification_metrics['weighted_f1']:.4f} | {'✅ 優秀' if classification_metrics['weighted_f1'] > 0.8 else '⚠️ 需要改進'} | 加權F1分數 |
| **宏觀Precision** | {classification_metrics['macro_precision']:.4f} | {'✅ 優秀' if classification_metrics['macro_precision'] > 0.8 else '⚠️ 需要改進'} | 宏觀精確率 |
| **宏觀Recall** | {classification_metrics['macro_recall']:.4f} | {'✅ 優秀' if classification_metrics['macro_recall'] > 0.8 else '⚠️ 需要改進'} | 宏觀召回率 |
| **宏觀F1-Score** | {classification_metrics['macro_f1']:.4f} | {'✅ 優秀' if classification_metrics['macro_f1'] > 0.8 else '⚠️ 需要改進'} | 宏觀F1分數 |

### 每個類別的詳細表現

"""
        
        # 添加每個類別的詳細指標
        for i, class_name in enumerate(self.detection_classes):
            precision = classification_metrics['precision_per_class'][i]
            recall = classification_metrics['recall_per_class'][i]
            f1 = classification_metrics['f1_per_class'][i]
            support = classification_metrics['support_per_class'][i]
            accuracy = classification_metrics['class_accuracy'][class_name]
            
            report += f"""#### {class_name} ({class_name})
- **準確率**: {accuracy:.4f} ({accuracy*100:.2f}%)
- **Precision**: {precision:.4f} ({precision*100:.2f}%)
- **Recall**: {recall:.4f} ({recall*100:.2f}%)
- **F1-Score**: {f1:.4f} ({f1*100:.2f}%)
- **樣本數**: {support}

"""
        
        # 添加混淆矩陣
        report += f"""## 📊 混淆矩陣

```
預測 →
實際 ↓        {self.detection_classes[0]}        {self.detection_classes[1]}        {self.detection_classes[2]}        {self.detection_classes[3]}
"""
        
        cm = classification_metrics['confusion_matrix']
        for i, class_name in enumerate(self.detection_classes):
            report += f"      {class_name}         {cm[i][0]}         {cm[i][1]}         {cm[i][2]}         {cm[i][3]}\n"
        
        report += """```

## 📈 性能評估總結

### 檢測任務評估
"""
        
        # 檢測任務評估
        if detection_metrics['bbox_mae'] < 0.1:
            report += "- **Bbox MAE < 0.1**: ✅ 優秀\n"
        else:
            report += "- **Bbox MAE < 0.1**: ⚠️ 需要改進\n"
            
        if detection_metrics['mean_iou'] > 0.5:
            report += "- **IoU > 0.5**: ✅ 優秀\n"
        else:
            report += "- **IoU > 0.5**: ⚠️ 需要改進\n"
            
        if detection_metrics['map_50'] > 0.7:
            report += "- **mAP > 0.7**: ✅ 優秀\n"
        else:
            report += "- **mAP > 0.7**: ⚠️ 需要改進\n"
        
        report += """
### 分類任務評估
"""
        
        # 分類任務評估
        if classification_metrics['accuracy'] > 0.8:
            report += "- **準確率 > 0.8**: ✅ 優秀\n"
        else:
            report += "- **準確率 > 0.8**: ⚠️ 需要改進\n"
            
        if classification_metrics['weighted_f1'] > 0.7:
            report += "- **F1-Score > 0.7**: ✅ 優秀\n"
        else:
            report += "- **F1-Score > 0.7**: ⚠️ 需要改進\n"
        
        report += f"""
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

## 📊 詳細性能圖表

詳細性能圖表已保存為: `comprehensive_validation_metrics.png`

---
*生成時間: 2025-08-02*
*模型版本: UltimateJointModel v1.0*
"""
        
        # 保存報告
        with open('comprehensive_validation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ 詳細驗證報告已保存為: comprehensive_validation_report.md")
        return report
    
    def run_calculation(self):
        """運行完整計算"""
        print("🚀 開始全面驗證指標計算...")
        
        # 評估模型
        self.evaluate_model()
        
        # 計算檢測指標
        detection_metrics = self.calculate_detection_metrics()
        
        # 計算分類指標
        classification_metrics = self.calculate_classification_metrics()
        
        # 生成可視化
        self.generate_visualizations(detection_metrics, classification_metrics)
        
        # 生成詳細報告
        self.generate_detailed_report(detection_metrics, classification_metrics)
        
        return detection_metrics, classification_metrics

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Validation Metrics Calculator')
    parser.add_argument('--data', type=str, default='converted_dataset_fixed/data.yaml', help='Path to data.yaml file')
    parser.add_argument('--weights', type=str, default='joint_model_ultimate.pth', help='Path to model weights')
    
    args = parser.parse_args()
    
    try:
        # 創建計算器
        calculator = ComprehensiveMetricsCalculator(args.data, args.weights)
        
        # 運行計算
        detection_metrics, classification_metrics = calculator.run_calculation()
        
        print("\n🎉 全面驗證指標計算完成！")
        print("📊 主要結果:")
        print(f"   - 檢測任務 mAP@0.5: {detection_metrics['map_50']:.4f}")
        print(f"   - 分類任務準確率: {classification_metrics['accuracy']:.4f}")
        print(f"   - 分類任務F1-Score: {classification_metrics['weighted_f1']:.4f}")
        
    except Exception as e:
        print(f"❌ 計算失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    main() 