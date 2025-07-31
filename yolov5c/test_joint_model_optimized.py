import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import numpy as np
from pathlib import Path
import yaml
from train_joint_simple_optimized import OptimizedJointModel, load_mixed_labels

def test_model_optimized(model, test_loader, device='cuda'):
    """Test the optimized joint model with detailed metrics"""
    model.eval()
    
    total_bbox_mae = 0
    total_cls_accuracy = 0
    total_samples = 0
    
    # Per-class accuracy tracking
    class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    class_total = {0: 0, 1: 0, 2: 0, 3: 0}
    
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
            target_classes = torch.argmax(cls_targets, dim=1)
            pred_classes = torch.argmax(cls_pred_sigmoid, dim=1)
            cls_accuracy = (pred_classes == target_classes).float().mean().item()
            total_cls_accuracy += cls_accuracy * images.size(0)
            
            # Track per-class accuracy
            for i in range(images.size(0)):
                true_class = target_classes[i].item()
                pred_class = pred_classes[i].item()
                class_total[true_class] += 1
                if true_class == pred_class:
                    class_correct[true_class] += 1
            
            total_samples += images.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(test_loader)}: '
                      f'Bbox MAE: {bbox_mae:.4f}, Cls Acc: {cls_accuracy:.4f}')
    
    avg_bbox_mae = total_bbox_mae / total_samples
    avg_cls_accuracy = total_cls_accuracy / total_samples
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for class_id in range(4):
        if class_total[class_id] > 0:
            per_class_accuracy[class_id] = class_correct[class_id] / class_total[class_id]
        else:
            per_class_accuracy[class_id] = 0.0
    
    return avg_bbox_mae, avg_cls_accuracy, per_class_accuracy

def test_single_samples_optimized(model, data_dir, num_samples=10, device='cuda'):
    """Test optimized model on individual samples with detailed analysis"""
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
    
    # Data transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load data configuration
    with open(Path(data_dir) / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    nc = data_config['nc']
    class_names = data_config['names']
    
    # Create test dataset
    test_dataset = TestDataset(data_dir, num_samples, transform, nc)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.eval()
    
    total_bbox_mae = 0
    total_cls_accuracy = 0
    class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    class_total = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print(f"測試 {num_samples} 個樣本...")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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
            
            # Track per-class accuracy
            true_class = target_classes[0].item()
            pred_class = pred_classes[0].item()
            class_total[true_class] += 1
            if true_class == pred_class:
                class_correct[true_class] += 1
            
            # Print detailed results
            print(f'樣本 {batch_idx+1}: {filename}')
            print(f'  真實 bbox: {bbox_targets[0].cpu().numpy()}')
            print(f'  預測 bbox: {bbox_pred[0].cpu().numpy()}')
            print(f'  真實分類: {cls_targets[0].cpu().numpy()} ({class_names[true_class]})')
            print(f'  預測分類: {cls_pred_sigmoid[0].cpu().numpy()} ({class_names[pred_class]})')
            print(f'  Bbox MAE: {bbox_mae:.4f}')
            print(f'  分類準確率: {cls_accuracy:.4f}')
            print(f'  預測置信度: {cls_pred_sigmoid[0].max().item():.4f}')
            print('-' * 50)
            
            total_bbox_mae += bbox_mae
            total_cls_accuracy += cls_accuracy
    
    avg_bbox_mae = total_bbox_mae / num_samples
    avg_cls_accuracy = total_cls_accuracy / num_samples
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for class_id in range(4):
        if class_total[class_id] > 0:
            per_class_accuracy[class_id] = class_correct[class_id] / class_total[class_id]
        else:
            per_class_accuracy[class_id] = 0.0
    
    print("=" * 80)
    print("整體性能總結:")
    print(f"平均 Bbox MAE: {avg_bbox_mae:.4f}")
    print(f"平均分類準確率: {avg_cls_accuracy:.4f}")
    print("\n各類別準確率:")
    for class_id in range(4):
        print(f"  {class_names[class_id]}: {per_class_accuracy[class_id]:.4f} ({class_correct[class_id]}/{class_total[class_id]})")
    
    return avg_bbox_mae, avg_cls_accuracy, per_class_accuracy

def main():
    parser = argparse.ArgumentParser(description='Test Optimized Joint Detection and Classification Model')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='joint_model_optimized.pth', help='Path to model weights')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to test')
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
    model = OptimizedJointModel(num_classes=nc)
    
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"優化模型已載入: {args.model}")
    else:
        print(f"錯誤: 模型文件不存在: {args.model}")
        return
    
    model = model.to(device)
    
    # Test on individual samples
    print(f"測試 {args.samples} 個樣本...")
    test_single_samples_optimized(model, data_dir, args.samples, device)

if __name__ == '__main__':
    main() 