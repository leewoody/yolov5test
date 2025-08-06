#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Model Performance
簡化的模型性能測試腳本
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# 添加項目路徑
sys.path.append(str(Path(__file__).parent))

try:
    from train_joint_ultimate import UltimateJointModel, UltimateDataset
except ImportError:
    print("無法導入UltimateJointModel，使用基礎模型")
    from train_joint_simple import SimpleJointModel, MixedDataset as UltimateDataset


def test_model_performance(data_yaml, weights_path):
    """測試模型性能"""
    print("🚀 開始模型性能測試...")
    
    # 載入數據配置
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    detection_classes = config.get('names', [])
    classification_classes = ['PSAX', 'PLAX', 'A4C']
    num_detection_classes = len(detection_classes)
    
    print(f"✅ 數據配置載入完成:")
    print(f"   - 檢測類別: {detection_classes}")
    print(f"   - 分類類別: {classification_classes}")
    
    # 載入模型
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {weights_path}")
    
    model = UltimateJointModel(num_classes=num_detection_classes)
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"✅ 模型載入完成: {weights_path}")
    
    # 載入測試數據
    data_dir = Path(data_yaml).parent
    test_dataset = UltimateDataset(
        data_dir=str(data_dir),
        split='test',
        transform=None,
        nc=num_detection_classes
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
    
    with torch.no_grad():
        for batch_idx, (images, bbox_targets, cls_targets) in enumerate(test_loader):
            images = images.to(device)
            bbox_targets = bbox_targets.to(device)
            cls_targets = cls_targets.to(device)
            
            # 前向傳播
            bbox_preds, cls_preds = model(images)
            
            # 收集結果
            all_bbox_preds.append(bbox_preds.cpu().numpy())
            all_bbox_targets.append(bbox_targets.cpu().numpy())
            all_cls_preds.append(cls_preds.cpu().numpy())
            all_cls_targets.append(cls_targets.cpu().numpy())
            
            if batch_idx % 50 == 0:
                print(f"  處理進度: {batch_idx}/{len(test_loader)}")
    
    # 轉換為numpy數組
    bbox_preds = np.concatenate(all_bbox_preds, axis=0)
    bbox_targets = np.concatenate(all_bbox_targets, axis=0)
    cls_preds = np.concatenate(all_cls_preds, axis=0)
    cls_targets = np.concatenate(all_cls_targets, axis=0)
    
    print(f"✅ 預測完成: {len(bbox_preds)} 個樣本")
    
    # 計算檢測指標
    print("\n📊 檢測任務指標:")
    bbox_mae = np.mean(np.abs(bbox_preds - bbox_targets))
    print(f"   - Bbox MAE: {bbox_mae:.4f}")
    
    # 計算IoU
    ious = calculate_iou(bbox_preds, bbox_targets)
    mean_iou = np.mean(ious)
    print(f"   - Mean IoU: {mean_iou:.4f}")
    
    # 計算mAP (簡化版本)
    map_score = np.mean(ious > 0.5)
    print(f"   - mAP (IoU>0.5): {map_score:.4f}")
    
    # 計算分類指標
    print("\n📊 分類任務指標:")
    cls_pred_labels = np.argmax(cls_preds, axis=1)
    cls_target_labels = np.argmax(cls_targets, axis=1)
    
    accuracy = np.mean(cls_pred_labels == cls_target_labels)
    print(f"   - 整體準確率: {accuracy:.4f}")
    
    # 計算詳細指標
    precision, recall, f1, support = precision_recall_fscore_support(
        cls_target_labels, cls_pred_labels, average='weighted'
    )
    print(f"   - 加權Precision: {precision:.4f}")
    print(f"   - 加權Recall: {recall:.4f}")
    print(f"   - 加權F1-Score: {f1:.4f}")
    
    # 計算混淆矩陣
    cm = confusion_matrix(cls_target_labels, cls_pred_labels)
    print(f"   - 混淆矩陣:")
    print(cm)
    
    # 每個類別的詳細指標
    class_report = classification_report(
        cls_target_labels, cls_pred_labels,
        target_names=classification_classes,
        output_dict=True
    )
    
    print(f"\n📊 每個類別的詳細指標:")
    for i, class_name in enumerate(classification_classes):
        if i < len(class_report):
            class_metrics = class_report[class_name]
            print(f"   - {class_name}: Precision={class_metrics['precision']:.4f}, "
                  f"Recall={class_metrics['recall']:.4f}, F1={class_metrics['f1-score']:.4f}")
    
    # 生成性能總結
    print(f"\n🎯 性能總結:")
    print(f"   - 檢測任務: MAE={bbox_mae:.4f}, IoU={mean_iou:.4f}, mAP={map_score:.4f}")
    print(f"   - 分類任務: 準確率={accuracy:.4f}, F1={f1:.4f}")
    
    # 性能評估
    print(f"\n📈 性能評估:")
    print(f"   - 檢測MAE < 0.1: {'✅ 優秀' if bbox_mae < 0.1 else '⚠️ 需要改進'}")
    print(f"   - 檢測IoU > 0.5: {'✅ 優秀' if mean_iou > 0.5 else '⚠️ 需要改進'}")
    print(f"   - 分類準確率 > 0.8: {'✅ 優秀' if accuracy > 0.8 else '⚠️ 需要改進'}")
    print(f"   - 分類F1 > 0.7: {'✅ 優秀' if f1 > 0.7 else '⚠️ 需要改進'}")
    
    return {
        'detection': {
            'bbox_mae': bbox_mae,
            'mean_iou': mean_iou,
            'map_score': map_score
        },
        'classification': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'class_report': class_report
        }
    }


def calculate_iou(preds, targets):
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


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='測試模型性能')
    parser.add_argument('--data', type=str, required=True, help='數據配置文件路徑')
    parser.add_argument('--weights', type=str, required=True, help='模型權重文件路徑')
    
    args = parser.parse_args()
    
    try:
        metrics = test_model_performance(args.data, args.weights)
        
        print("\n🎉 性能測試完成！")
        print(f"📊 檢測MAE: {metrics['detection']['bbox_mae']:.4f}")
        print(f"📊 分類準確率: {metrics['classification']['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 