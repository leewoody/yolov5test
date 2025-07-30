#!/usr/bin/env python3
"""
測試聯合訓練的模型
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import numpy as np
from pathlib import Path
import yaml
from train_joint_simple import SimpleJointModel, load_mixed_labels

def test_model(model, test_loader, device='cuda'):
    """Test the joint model"""
    model.eval()
    
    total_bbox_mae = 0
    total_cls_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            bbox_targets = batch['bbox'].to(device)
            cls_targets = batch['classification'].to(device)
            
            # Forward pass
            bbox_pred, cls_pred = model(images)
            
            # Calculate bbox MAE
            bbox_mae = torch.mean(torch.abs(bbox_pred - bbox_targets)).item()
            total_bbox_mae += bbox_mae * images.size(0)
            
            # Calculate classification accuracy
            cls_pred_sigmoid = torch.sigmoid(cls_pred)
            # Convert one-hot targets to class indices
            target_classes = torch.argmax(cls_targets, dim=1)
            pred_classes = torch.argmax(cls_pred_sigmoid, dim=1)
            cls_accuracy = (pred_classes == target_classes).float().mean().item()
            total_cls_accuracy += cls_accuracy * images.size(0)
            
            total_samples += images.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(test_loader)}: '
                      f'Bbox MAE: {bbox_mae:.4f}, Cls Acc: {cls_accuracy:.4f}')
    
    avg_bbox_mae = total_bbox_mae / total_samples
    avg_cls_accuracy = total_cls_accuracy / total_samples
    
    return avg_bbox_mae, avg_cls_accuracy

def test_single_samples(model, data_dir, num_samples=5, device='cuda'):
    """Test model on individual samples"""
    from torch.utils.data import Dataset, DataLoader
    
    class TestDataset(Dataset):
        def __init__(self, data_dir, num_samples, transform=None, nc=4):
            self.data_dir = Path(data_dir)
            self.transform = transform
            self.nc = nc
            
            self.images_dir = self.data_dir / 'test' / 'images'
            self.labels_dir = self.data_dir / 'test' / 'labels'
            
            # Get image files
            self.image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))
            
            # Limit to num_samples
            self.image_files = self.image_files[:num_samples]
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Load labels
            bbox, classification = load_mixed_labels(label_path, self.nc)
            
            return {
                'image': image,
                'bbox': torch.tensor(bbox, dtype=torch.float32),
                'classification': torch.tensor(classification, dtype=torch.float32),
                'filename': img_path.name
            }
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load data configuration
    with open(Path(data_dir) / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    nc = data_config['nc']
    
    # Create test dataset
    test_dataset = TestDataset(data_dir, num_samples, transform, nc)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.eval()
    
    total_bbox_mae = 0
    total_cls_accuracy = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            bbox_targets = batch['bbox'].to(device)
            cls_targets = batch['classification'].to(device)
            filename = batch['filename'][0]
            
            # Forward pass
            bbox_pred, cls_pred = model(images)
            
            # Calculate bbox MAE
            bbox_mae = torch.mean(torch.abs(bbox_pred - bbox_targets)).item()
            
            # Calculate classification accuracy
            cls_pred_sigmoid = torch.sigmoid(cls_pred)
            target_classes = torch.argmax(cls_targets, dim=1)
            pred_classes = torch.argmax(cls_pred_sigmoid, dim=1)
            cls_accuracy = (pred_classes == target_classes).float().item()
            
            # Print results
            print(f'文件: {filename}')
            print(f'  真實 bbox: {bbox_targets[0].cpu().numpy()}')
            print(f'  預測 bbox: {bbox_pred[0].cpu().numpy()}')
            print(f'  真實分類: {cls_targets[0].cpu().numpy()}')
            print(f'  預測分類: {cls_pred_sigmoid[0].cpu().numpy()}')
            print(f'  Bbox MAE: {bbox_mae:.4f}')
            print(f'  分類準確率: {cls_accuracy:.4f}')
            print('-' * 50)
            
            total_bbox_mae += bbox_mae
            total_cls_accuracy += cls_accuracy
    
    avg_bbox_mae = total_bbox_mae / num_samples
    avg_cls_accuracy = total_cls_accuracy / num_samples
    
    print(f'平均 Bbox MAE: {avg_bbox_mae:.4f}')
    print(f'平均分類準確率: {avg_cls_accuracy:.4f}')
    
    return avg_bbox_mae, avg_cls_accuracy

def main():
    parser = argparse.ArgumentParser(description='Test Joint Detection and Classification Model')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='joint_model_simple.pth', help='Path to model weights')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to test')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Load data configuration
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_dir = Path(args.data).parent
    nc = data_config['nc']
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用設備: {device}")
    print(f"類別數量: {nc}")
    
    # Load model
    model = SimpleJointModel(num_classes=nc)
    
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"模型已載入: {args.model}")
    else:
        print(f"錯誤: 模型文件不存在: {args.model}")
        return
    
    model = model.to(device)
    
    # Test on individual samples
    print(f"測試 {args.samples} 個樣本...")
    test_single_samples(model, data_dir, args.samples, device)

if __name__ == '__main__':
    main() 