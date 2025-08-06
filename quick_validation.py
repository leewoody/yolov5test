#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Validation Script for Ultimate Joint Model
快速驗證終極聯合模型
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys
import os

# 添加項目路徑
sys.path.append('yolov5c')

try:
    from train_joint_ultimate import UltimateJointModel, UltimateDataset
except ImportError:
    print("無法導入UltimateJointModel")
    sys.exit(1)

def load_model_and_data():
    """載入模型和數據"""
    # 載入數據配置
    data_yaml = "converted_dataset_fixed/data.yaml"
    weights_path = "joint_model_ultimate.pth"
    
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    detection_classes = config.get('names', [])
    classification_classes = ['PSAX', 'PLAX', 'A4C']
    num_detection_classes = len(detection_classes)
    
    print(f"✅ 數據配置載入完成:")
    print(f"   - 檢測類別: {detection_classes}")
    print(f"   - 分類類別: {classification_classes}")
    
    # 載入模型
    model = UltimateJointModel(num_classes=num_detection_classes)
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ 模型載入完成: {weights_path}")
    
    return model, detection_classes, classification_classes

def quick_validation():
    """快速驗證"""
    print("🚀 開始快速驗證...")
    
    # 載入模型和數據
    model, detection_classes, classification_classes = load_model_and_data()
    
    # 創建測試數據集
    data_dir = Path("converted_dataset_fixed")
    test_dataset = UltimateDataset(data_dir, 'test', transform=None, nc=len(detection_classes))
    
    print(f"📊 測試數據集: {len(test_dataset)} 個樣本")
    
    # 統計類別分布
    class_counts = {}
    for i in range(len(test_dataset)):
        try:
            data = test_dataset[i]
            cls_target = data['classification']
            cls_idx = cls_target.item() if hasattr(cls_target, 'item') else cls_target
            class_counts[cls_idx] = class_counts.get(cls_idx, 0) + 1
        except Exception as e:
            print(f"⚠️ 樣本 {i} 載入失敗: {e}")
            continue
    
    print(f"📈 類別分布: {class_counts}")
    
    # 簡單的準確率計算
    correct = 0
    total = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"🔧 使用設備: {device}")
    
    with torch.no_grad():
        for i in range(min(50, len(test_dataset))):  # 只測試前50個樣本
            try:
                data = test_dataset[i]
                image = data['image']
                bbox_target = data['bbox']
                cls_target = data['classification']
                
                # 轉換為tensor
                if not isinstance(image, torch.Tensor):
                    continue
                
                image = image.unsqueeze(0).to(device)
                cls_target = torch.tensor([cls_target]).to(device)
                
                # 前向傳播
                bbox_pred, cls_pred = model(image)
                
                # 計算準確率
                pred_class = torch.argmax(cls_pred, dim=1)
                if pred_class.item() == cls_target.item():
                    correct += 1
                total += 1
                
                if i % 10 == 0:
                    print(f"  樣本 {i}: 預測={pred_class.item()}, 真實={cls_target.item()}")
                    
            except Exception as e:
                print(f"⚠️ 樣本 {i} 處理失敗: {e}")
                continue
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n📊 快速驗證結果:")
    print(f"   - 測試樣本數: {total}")
    print(f"   - 正確預測: {correct}")
    print(f"   - 準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

if __name__ == "__main__":
    try:
        accuracy = quick_validation()
        print(f"\n✅ 快速驗證完成！準確率: {accuracy*100:.2f}%")
    except Exception as e:
        print(f"❌ 驗證失敗: {e}")
        import traceback
        traceback.print_exc() 