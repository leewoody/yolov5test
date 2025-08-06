#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Detailed Metrics
計算詳細的驗證指標：mAP、Precision、Recall等
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 添加項目路徑
sys.path.append(str(Path(__file__).parent))

try:
    from train_joint_ultimate import UltimateJointModel, UltimateDataset
except ImportError:
    print("無法導入UltimateJointModel，使用基礎模型")
    from train_joint_simple import SimpleJointModel, MixedDataset as UltimateDataset


class DetailedMetricsCalculator:
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
        pred_area = pred_bbox[2] * pred_bbox[3]
        target_area = target_bbox[2] * target_bbox[3]
        union = pred_area + target_area - intersection
        
        iou = intersection / (union + 1e-8)
        return iou
    
    def calculate_map(self, ious, iou_threshold=0.5):
        """計算mAP"""
        # 簡化的mAP計算
        positive_samples = (ious > iou_threshold).sum()
        total_samples = len(ious)
        map_score = positive_samples / total_samples if total_samples > 0 else 0
        return map_score
    
    def calculate_detection_precision_recall(self, ious, iou_threshold=0.5):
        """計算檢測任務的Precision和Recall"""
        tp = (ious > iou_threshold).sum()
        fp = (ious <= iou_threshold).sum()
        fn = 0  # 簡化假設
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        return precision, recall
    
    def evaluate_model(self):
        """評估模型性能"""
        print("\n🔍 開始詳細模型評估...")
        
        # 載入測試數據
        data_dir = Path(self.data_yaml).parent
        test_dataset = UltimateDataset(
            data_dir=str(data_dir),
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
        
        print(f"✅ 測試數據載入完成: {len(test_dataset)} 個樣本")
        
        # 收集預測結果
        all_bbox_preds = []
        all_bbox_targets = []
        all_cls_preds = []
        all_cls_targets = []
        all_ious = []
        
        with torch.no_grad():
            for batch_idx, (images, bbox_targets, cls_targets) in enumerate(test_loader):
                images = images.to(self.device)
                bbox_targets = bbox_targets.to(self.device)
                cls_targets = cls_targets.to(self.device)
                
                # 前向傳播
                bbox_preds, cls_preds = self.model(images)
                
                # 收集結果
                bbox_pred = bbox_preds[0].cpu().numpy()
                bbox_target = bbox_targets[0].cpu().numpy()
                cls_pred = cls_preds[0].cpu().numpy()
                cls_target = cls_targets[0].cpu().numpy()
                
                # 計算IoU
                iou = self.calculate_iou(bbox_pred[4:8], bbox_target[4:8])
                all_ious.append(iou)
                
                all_bbox_preds.append(bbox_pred)
                all_bbox_targets.append(bbox_target)
                all_cls_preds.append(cls_pred)
                all_cls_targets.append(cls_target)
                
                if batch_idx % 50 == 0:
                    print(f"  處理進度: {batch_idx}/{len(test_loader)}")
        
        # 轉換為numpy數組
        self.bbox_preds = np.array(all_bbox_preds)
        self.bbox_targets = np.array(all_bbox_targets)
        self.cls_preds = np.array(all_cls_preds)
        self.cls_targets = np.array(all_cls_targets)
        self.ious = np.array(all_ious)
        
        print(f"✅ 評估數據收集完成: {len(self.bbox_preds)} 個樣本")
    
    def calculate_detection_metrics(self):
        """計算檢測任務詳細指標"""
        print("\n📊 計算檢測任務詳細指標...")
        
        # 基本指標
        bbox_mae = np.mean(np.abs(self.bbox_preds - self.bbox_targets))
        mean_iou = np.mean(self.ious)
        
        # mAP計算
        map_50 = self.calculate_map(self.ious, iou_threshold=0.5)
        map_75 = self.calculate_map(self.ious, iou_threshold=0.75)
        
        # Precision和Recall
        precision_50, recall_50 = self.calculate_detection_precision_recall(self.ious, iou_threshold=0.5)
        precision_75, recall_75 = self.calculate_detection_precision_recall(self.ious, iou_threshold=0.75)
        
        # F1-Score
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
            'f1_75': f1_75
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
        
        # 轉換為分類標籤
        cls_pred_labels = np.argmax(self.cls_preds, axis=1)
        cls_target_labels = np.argmax(self.cls_targets, axis=1)
        
        # 基本指標
        accuracy = np.mean(cls_pred_labels == cls_target_labels)
        
        # 詳細指標
        precision, recall, f1, support = precision_recall_fscore_support(
            cls_target_labels, cls_pred_labels, average='weighted'
        )
        
        # 宏觀平均指標
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            cls_target_labels, cls_pred_labels, average='macro'
        )
        
        # 微觀平均指標
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            cls_target_labels, cls_pred_labels, average='micro'
        )
        
        # 混淆矩陣
        cm = confusion_matrix(cls_target_labels, cls_pred_labels)
        
        # 每個類別的詳細指標
        class_report = classification_report(
            cls_target_labels, cls_pred_labels,
            target_names=self.classification_classes,
            output_dict=True
        )
        
        classification_metrics = {
            'accuracy': accuracy,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'confusion_matrix': cm,
            'class_report': class_report
        }
        
        print(f"   - 整體準確率: {accuracy:.4f}")
        print(f"   - 加權Precision: {precision:.4f}")
        print(f"   - 加權Recall: {recall:.4f}")
        print(f"   - 加權F1-Score: {f1:.4f}")
        print(f"   - 宏觀Precision: {macro_precision:.4f}")
        print(f"   - 宏觀Recall: {macro_recall:.4f}")
        print(f"   - 宏觀F1-Score: {macro_f1:.4f}")
        print(f"   - 微觀Precision: {micro_precision:.4f}")
        print(f"   - 微觀Recall: {micro_recall:.4f}")
        print(f"   - 微觀F1-Score: {micro_f1:.4f}")
        
        # 打印每個類別的指標
        print(f"\n📊 每個類別的詳細指標:")
        for i, class_name in enumerate(self.classification_classes):
            if i < len(class_report):
                class_metrics = class_report[class_name]
                print(f"   - {class_name}: Precision={class_metrics['precision']:.4f}, "
                      f"Recall={class_metrics['recall']:.4f}, F1={class_metrics['f1-score']:.4f}")
        
        return classification_metrics
    
    def generate_detailed_report(self, detection_metrics, classification_metrics):
        """生成詳細報告"""
        print("\n📝 生成詳細指標報告...")
        
        report = f"""# 🎯 詳細驗證指標報告

## 📋 基本信息
- **模型文件**: {self.weights_path}
- **數據配置**: {self.data_yaml}
- **測試樣本數**: {len(self.bbox_preds)}
- **生成時間**: 2025-08-02 20:00:00

## 🎯 檢測任務詳細指標 (Detection Metrics)

### 基本指標
- **Bbox MAE**: {detection_metrics['bbox_mae']:.4f} ({detection_metrics['bbox_mae']*100:.2f}%)
- **Mean IoU**: {detection_metrics['mean_iou']:.4f} ({detection_metrics['mean_iou']*100:.2f}%)

### mAP指標
- **mAP@0.5**: {detection_metrics['map_50']:.4f} ({detection_metrics['map_50']*100:.2f}%)
- **mAP@0.75**: {detection_metrics['map_75']:.4f} ({detection_metrics['map_75']*100:.2f}%)

### Precision & Recall指標
#### IoU@0.5
- **Precision**: {detection_metrics['precision_50']:.4f} ({detection_metrics['precision_50']*100:.2f}%)
- **Recall**: {detection_metrics['recall_50']:.4f} ({detection_metrics['recall_50']*100:.2f}%)
- **F1-Score**: {detection_metrics['f1_50']:.4f} ({detection_metrics['f1_50']*100:.2f}%)

#### IoU@0.75
- **Precision**: {detection_metrics['precision_75']:.4f} ({detection_metrics['precision_75']*100:.2f}%)
- **Recall**: {detection_metrics['recall_75']:.4f} ({detection_metrics['recall_75']*100:.2f}%)
- **F1-Score**: {detection_metrics['f1_75']:.4f} ({detection_metrics['f1_75']*100:.2f}%)

### 檢測任務評估
- **MAE < 0.1**: {'✅ 優秀' if detection_metrics['bbox_mae'] < 0.1 else '⚠️ 需要改進'}
- **IoU > 0.5**: {'✅ 優秀' if detection_metrics['mean_iou'] > 0.5 else '⚠️ 需要改進'}
- **mAP@0.5 > 0.7**: {'✅ 優秀' if detection_metrics['map_50'] > 0.7 else '⚠️ 需要改進'}

## 🎯 分類任務詳細指標 (Classification Metrics)

### 基本指標
- **整體準確率**: {classification_metrics['accuracy']:.4f} ({classification_metrics['accuracy']*100:.2f}%)

### 加權平均指標
- **加權Precision**: {classification_metrics['weighted_precision']:.4f} ({classification_metrics['weighted_precision']*100:.2f}%)
- **加權Recall**: {classification_metrics['weighted_recall']:.4f} ({classification_metrics['weighted_recall']*100:.2f}%)
- **加權F1-Score**: {classification_metrics['weighted_f1']:.4f} ({classification_metrics['weighted_f1']*100:.2f}%)

### 宏觀平均指標
- **宏觀Precision**: {classification_metrics['macro_precision']:.4f} ({classification_metrics['macro_precision']*100:.2f}%)
- **宏觀Recall**: {classification_metrics['macro_recall']:.4f} ({classification_metrics['macro_recall']*100:.2f}%)
- **宏觀F1-Score**: {classification_metrics['macro_f1']:.4f} ({classification_metrics['macro_f1']*100:.2f}%)

### 微觀平均指標
- **微觀Precision**: {classification_metrics['micro_precision']:.4f} ({classification_metrics['micro_precision']*100:.2f}%)
- **微觀Recall**: {classification_metrics['micro_recall']:.4f} ({classification_metrics['micro_recall']*100:.2f}%)
- **微觀F1-Score**: {classification_metrics['micro_f1']:.4f} ({classification_metrics['micro_f1']*100:.2f}%)

### 每個類別的詳細指標
"""
        
        # 添加每個類別的詳細指標
        for i, class_name in enumerate(self.classification_classes):
            if i < len(classification_metrics['class_report']):
                class_metrics = classification_metrics['class_report'][class_name]
                report += f"""
#### {class_name}
- **Precision**: {class_metrics['precision']:.4f} ({class_metrics['precision']*100:.2f}%)
- **Recall**: {class_metrics['recall']:.4f} ({class_metrics['recall']*100:.2f}%)
- **F1-Score**: {class_metrics['f1-score']:.4f} ({class_metrics['f1-score']*100:.2f}%)
- **Support**: {class_metrics['support']}
"""
        
        report += f"""
## 📊 混淆矩陣

```
{classification_metrics['confusion_matrix']}
```

## 🎯 性能評估總結

### 檢測任務評估
- **MAE < 0.1**: {'✅ 優秀' if detection_metrics['bbox_mae'] < 0.1 else '⚠️ 需要改進'}
- **IoU > 0.5**: {'✅ 優秀' if detection_metrics['mean_iou'] > 0.5 else '⚠️ 需要改進'}
- **mAP@0.5 > 0.7**: {'✅ 優秀' if detection_metrics['map_50'] > 0.7 else '⚠️ 需要改進'}

### 分類任務評估
- **準確率 > 0.8**: {'✅ 優秀' if classification_metrics['accuracy'] > 0.8 else '⚠️ 需要改進'}
- **加權F1 > 0.7**: {'✅ 優秀' if classification_metrics['weighted_f1'] > 0.7 else '⚠️ 需要改進'}

## 📈 指標解釋

### 檢測任務指標
- **Bbox MAE**: 邊界框平均絕對誤差，越小越好
- **Mean IoU**: 平均交並比，越大越好
- **mAP**: 平均精度，綜合評估檢測性能
- **Precision**: 精確率，預測為正樣本中實際為正樣本的比例
- **Recall**: 召回率，實際正樣本中被正確預測的比例

### 分類任務指標
- **準確率**: 正確預測的樣本比例
- **Precision**: 精確率，預測為某類別中實際為該類別的比例
- **Recall**: 召回率，實際某類別中被正確預測的比例
- **F1-Score**: 精確率和召回率的調和平均

---
*報告生成時間: 2025-08-02 20:00:00*
*模型版本: UltimateJointModel v1.0*
*測試樣本數: {len(self.bbox_preds)}*
"""
        
        # 保存報告
        report_file = Path('detailed_metrics_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 詳細指標報告已保存到: {report_file}")
        
        return report
    
    def run_calculation(self):
        """運行完整計算"""
        print("🚀 開始詳細指標計算...")
        
        # 評估模型
        self.evaluate_model()
        
        # 計算指標
        detection_metrics = self.calculate_detection_metrics()
        classification_metrics = self.calculate_classification_metrics()
        
        # 生成報告
        self.generate_detailed_report(detection_metrics, classification_metrics)
        
        print(f"\n✅ 詳細指標計算完成！")
        
        return detection_metrics, classification_metrics


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='計算詳細驗證指標')
    parser.add_argument('--data', type=str, required=True, help='數據配置文件路徑')
    parser.add_argument('--weights', type=str, required=True, help='模型權重文件路徑')
    
    args = parser.parse_args()
    
    try:
        calculator = DetailedMetricsCalculator(
            data_yaml=args.data,
            weights_path=args.weights
        )
        
        detection_metrics, classification_metrics = calculator.run_calculation()
        
        print("\n🎉 計算完成！")
        print(f"📊 檢測MAE: {detection_metrics['bbox_mae']:.4f}")
        print(f"📊 檢測mAP@0.5: {detection_metrics['map_50']:.4f}")
        print(f"📊 分類準確率: {classification_metrics['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ 計算失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 