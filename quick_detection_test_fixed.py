#!/usr/bin/env python3
"""
修正版檢測優化性能測試腳本
驗證修正版模型與YOLO格式標籤的檢測性能
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse

def load_detection_optimized_model_fixed(model_path, num_classes=4):
    """載入修正版檢測優化模型"""
    from yolov5c.train_joint_detection_optimized_fixed import DetectionOptimizedJointModelFixed
    model = DetectionOptimizedJointModelFixed(num_classes=num_classes)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"成功載入模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return None
    return model

def calculate_yolo_iou(pred_bbox, target_bbox):
    """計算YOLO格式IoU"""
    # pred_bbox, target_bbox: [class_id, cx, cy, w, h]
    pred_cx, pred_cy, pred_w, pred_h = pred_bbox[1], pred_bbox[2], pred_bbox[3], pred_bbox[4]
    target_cx, target_cy, target_w, target_h = target_bbox[1], target_bbox[2], target_bbox[3], target_bbox[4]
    pred_x1 = pred_cx - pred_w / 2
    pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2
    pred_y2 = pred_cy + pred_h / 2
    target_x1 = target_cx - target_w / 2
    target_y1 = target_cy - target_h / 2
    target_x2 = target_cx + target_w / 2
    target_y2 = target_cy + target_h / 2
    inter_x1 = max(pred_x1, target_x1)
    inter_y1 = max(pred_y1, target_y1)
    inter_x2 = min(pred_x2, target_x2)
    inter_y2 = min(pred_y2, target_y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def test_detection_performance_fixed(model, data_dir, device='cpu', num_samples=50):
    """測試修正版檢測性能"""
    model.eval()
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
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
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            bbox_target = torch.zeros(5)
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        parts = lines[0].strip().split()
                        if len(parts) >= 5:
                            bbox_target = torch.tensor([float(x) for x in parts[:5]])
            bbox_pred, cls_pred = model(image_tensor)
            bbox_pred_np = bbox_pred.cpu().numpy()[0]
            bbox_target_np = bbox_target.numpy()
            iou = calculate_yolo_iou(bbox_pred_np, bbox_target_np)
            total_iou += iou
            mae = np.mean(np.abs(bbox_pred_np[1:] - bbox_target_np[1:]))
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

def main():
    parser = argparse.ArgumentParser(description='修正版檢測性能測試')
    parser.add_argument('--data', type=str, default='converted_dataset_fixed', help='數據集路徑')
    parser.add_argument('--device', type=str, default='auto', help='設備 (auto, cuda, cpu)')
    args = parser.parse_args()
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"使用設備: {device}")
    print(f"數據集路徑: {args.data}")
    model_path = 'joint_model_detection_optimized_fixed.pth'
    if Path(model_path).exists():
        model = load_detection_optimized_model_fixed(model_path)
        if model is not None:
            test_detection_performance_fixed(model, args.data, device)
    else:
        print(f"修正版檢測優化模型不存在: {model_path}")
        print("請先運行修正版檢測優化訓練")

if __name__ == '__main__':
    main() 