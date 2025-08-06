#!/usr/bin/env python3
"""
快速檢測優化測試腳本
驗證檢測任務的性能提升
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse
import time

def load_detection_optimized_model(model_path, num_classes=4):
    """載入檢測優化模型"""
    from yolov5c.train_joint_detection_optimized import DetectionOptimizedJointModel
    
    model = DetectionOptimizedJointModel(num_classes=num_classes)
    
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"成功載入模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return None
    
    return model

def calculate_iou(pred_bbox, target_bbox):
    """計算IoU"""
    # 簡化為矩形IoU計算
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]
    target_x1, target_y1, target_x2, target_y2 = target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]
    
    # 計算交集
    inter_x1 = max(pred_x1, target_x1)
    inter_y1 = max(pred_y1, target_y1)
    inter_x2 = min(pred_x2, target_x2)
    inter_y2 = min(pred_y2, target_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # 計算並集
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def test_detection_performance(model, data_dir, device='cpu', num_samples=50):
    """測試檢測性能"""
    model.eval()
    model.to(device)
    
    # 數據變換
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 載入測試數據
    images_dir = Path(data_dir) / 'val' / 'images'
    labels_dir = Path(data_dir) / 'val' / 'labels'
    
    image_files = sorted(list(images_dir.glob('*.png')))[:num_samples]
    label_files = sorted(list(labels_dir.glob('*.txt')))[:num_samples]
    
    total_iou = 0.0
    total_mae = 0.0
    total_samples = 0
    
    print(f"開始測試 {len(image_files)} 個樣本...")
    
    with torch.no_grad():
        for i, (image_path, label_path) in enumerate(zip(image_files, label_files)):
            # 載入圖像
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # 載入標籤
            bbox_target = torch.zeros(8)
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        parts = lines[0].strip().split()
                        if len(parts) >= 9:
                            bbox_target = torch.tensor([float(x) for x in parts[1:9]])
            
            # 模型預測
            bbox_pred, cls_pred = model(image_tensor)
            
            # 計算指標
            bbox_pred_np = bbox_pred.cpu().numpy()[0]
            bbox_target_np = bbox_target.numpy()
            
            # IoU
            iou = calculate_iou(bbox_pred_np[:4], bbox_target_np[:4])  # 使用前4個座標
            total_iou += iou
            
            # MAE
            mae = np.mean(np.abs(bbox_pred_np - bbox_target_np))
            total_mae += mae
            
            total_samples += 1
            
            if i % 10 == 0:
                print(f"樣本 {i+1}/{len(image_files)}: IoU={iou:.4f}, MAE={mae:.4f}")
    
    avg_iou = total_iou / total_samples
    avg_mae = total_mae / total_samples
    
    print(f"\n檢測性能測試結果:")
    print(f"平均 IoU: {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    print(f"平均 MAE: {avg_mae:.4f}")
    print(f"測試樣本數: {total_samples}")
    
    return avg_iou, avg_mae

def compare_models(data_dir, device='cpu'):
    """比較不同模型的檢測性能"""
    print("=" * 60)
    print("檢測性能模型比較")
    print("=" * 60)
    
    models_to_test = [
        ('joint_model_ultimate.pth', '終極聯合模型'),
        ('joint_model_detection_optimized.pth', '檢測優化模型')
    ]
    
    results = {}
    
    for model_path, model_name in models_to_test:
        print(f"\n測試 {model_name}...")
        
        if Path(model_path).exists():
            model = load_detection_optimized_model(model_path)
            if model is not None:
                iou, mae = test_detection_performance(model, data_dir, device, num_samples=30)
                results[model_name] = {'IoU': iou, 'MAE': mae}
            else:
                print(f"無法載入模型: {model_path}")
        else:
            print(f"模型文件不存在: {model_path}")
    
    # 比較結果
    print("\n" + "=" * 60)
    print("性能比較結果")
    print("=" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  IoU: {metrics['IoU']:.4f} ({metrics['IoU']*100:.2f}%)")
        print(f"  MAE: {metrics['MAE']:.4f}")
    
    # 找出最佳模型
    if results:
        best_iou_model = max(results.items(), key=lambda x: x[1]['IoU'])
        best_mae_model = min(results.items(), key=lambda x: x[1]['MAE'])
        
        print(f"\n最佳 IoU 模型: {best_iou_model[0]} ({best_iou_model[1]['IoU']*100:.2f}%)")
        print(f"最佳 MAE 模型: {best_mae_model[0]} ({best_mae_model[1]['MAE']:.4f})")

def main():
    parser = argparse.ArgumentParser(description='快速檢測性能測試')
    parser.add_argument('--data', type=str, default='converted_dataset_fixed', help='數據集路徑')
    parser.add_argument('--device', type=str, default='auto', help='設備 (auto, cuda, cpu)')
    parser.add_argument('--compare', action='store_true', help='比較多個模型')
    args = parser.parse_args()
    
    # 設備設置
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用設備: {device}")
    print(f"數據集路徑: {args.data}")
    
    if args.compare:
        compare_models(args.data, device)
    else:
        # 測試檢測優化模型
        model_path = 'joint_model_detection_optimized.pth'
        if Path(model_path).exists():
            model = load_detection_optimized_model(model_path)
            if model is not None:
                test_detection_performance(model, args.data, device)
        else:
            print(f"檢測優化模型不存在: {model_path}")
            print("請先運行檢測優化訓練")

if __name__ == '__main__':
    main() 