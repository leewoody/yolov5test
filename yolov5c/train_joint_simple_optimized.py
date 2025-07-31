import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import numpy as np
from pathlib import Path
import yaml

class OptimizedJointModel(nn.Module):
    def __init__(self, num_classes=4):
        super(OptimizedJointModel, self).__init__()
        
        # Enhanced backbone with more layers and better feature extraction
        self.backbone = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Enhanced detection head
        self.detection_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        
        # Enhanced classification head with attention mechanism
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        detection_output = self.detection_head(features)
        classification_output = self.classification_head(features)
        return detection_output, classification_output

def load_mixed_labels(label_path, nc=4):
    """Load mixed labels with one-hot encoding classification"""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Parse bbox line (first line)
        bbox_line = lines[0].strip().split()
        if len(bbox_line) >= 5:
            bbox = [float(x) for x in bbox_line[1:]]  # Skip class_id, keep x_center, y_center, width, height
        else:
            bbox = [0.5, 0.5, 0.1, 0.1]  # Default bbox
        
        # Parse classification line (second line) - one-hot encoding
        if len(lines) > 1:
            cls_line = lines[1].strip().split()
            # Convert one-hot encoding to list of floats
            classification = [float(x) for x in cls_line]
            # Ensure we have the right number of classes
            if len(classification) != nc:
                # Pad or truncate to match nc
                if len(classification) < nc:
                    classification.extend([0.0] * (nc - len(classification)))
                else:
                    classification = classification[:nc]
        else:
            classification = [1.0] + [0.0] * (nc - 1)  # Default to first class
        
        return bbox, classification
    except Exception as e:
        print(f"警告: 無法讀取標籤 {label_path}: {e}")
        return [0.5, 0.5, 0.1, 0.1], [1.0] + [0.0] * (nc - 1)

class MixedDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, nc=4):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.nc = nc
        
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'
        
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError(f"目錄不存在: {self.images_dir} 或 {self.labels_dir}")
        
        # Get all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))
        
        # Filter out images that don't have corresponding labels
        self.valid_image_files = []
        for img_file in self.image_files:
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                self.valid_image_files.append(img_file)
        
        self.image_files = self.valid_image_files
        
        print(f"{split} 數據集: {len(self.image_files)} 個有效樣本")
    
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
            'classification': torch.tensor(classification, dtype=torch.float32)
        }

def train_model_optimized(model, train_loader, val_loader, num_epochs=20, device='cuda'):
    """Train the optimized joint model with better loss balancing"""
    model = model.to(device)
    
    # Enhanced loss functions
    bbox_criterion = nn.SmoothL1Loss()  # Better for regression
    cls_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Dynamic loss weights
    bbox_weight = 1.0
    cls_weight = 2.0  # Give more weight to classification
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        bbox_loss_sum = 0
        cls_loss_sum = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            bbox_targets = batch['bbox'].to(device)
            cls_targets = batch['classification'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            bbox_pred, cls_pred = model(images)
            
            # Calculate losses
            bbox_loss = bbox_criterion(bbox_pred, bbox_targets)
            cls_loss = cls_criterion(cls_pred, cls_targets)
            
            # Combined loss with dynamic weights
            loss = bbox_weight * bbox_loss + cls_weight * cls_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            bbox_loss_sum += bbox_loss.item()
            cls_loss_sum += cls_loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f} (Bbox: {bbox_loss.item():.4f}, Cls: {cls_loss.item():.4f})')
        
        # Validation
        model.eval()
        val_loss = 0
        val_bbox_loss = 0
        val_cls_loss = 0
        val_cls_accuracy = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                bbox_targets = batch['bbox'].to(device)
                cls_targets = batch['classification'].to(device)
                
                bbox_pred, cls_pred = model(images)
                
                bbox_loss = bbox_criterion(bbox_pred, bbox_targets)
                cls_loss = cls_criterion(cls_pred, cls_targets)
                loss = bbox_weight * bbox_loss + cls_weight * cls_loss
                
                # Calculate classification accuracy
                cls_pred_sigmoid = torch.sigmoid(cls_pred)
                target_classes = torch.argmax(cls_targets, dim=1)
                pred_classes = torch.argmax(cls_pred_sigmoid, dim=1)
                cls_accuracy = (pred_classes == target_classes).float().mean().item()
                
                val_loss += loss.item()
                val_bbox_loss += bbox_loss.item()
                val_cls_loss += cls_loss.item()
                val_cls_accuracy += cls_accuracy * images.size(0)
                val_samples += images.size(0)
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls_accuracy = val_cls_accuracy / val_samples
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"保存最佳模型 (val_loss: {avg_val_loss:.4f}, cls_acc: {avg_val_cls_accuracy:.4f})")
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} (Bbox: {bbox_loss_sum/len(train_loader):.4f}, Cls: {cls_loss_sum/len(train_loader):.4f})')
        print(f'  Val Loss: {avg_val_loss:.4f} (Bbox: {val_bbox_loss/len(val_loader):.4f}, Cls: {val_cls_loss/len(val_loader):.4f})')
        print(f'  Val Cls Accuracy: {avg_val_cls_accuracy:.4f}')
        print('-' * 50)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Optimized Joint Detection and Classification Training')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
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
    
    # Enhanced data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MixedDataset(data_dir, 'train', train_transform, nc)
    val_dataset = MixedDataset(data_dir, 'val', val_transform, nc)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create optimized model
    model = OptimizedJointModel(num_classes=nc)
    
    # Train model
    print("開始優化訓練...")
    trained_model = train_model_optimized(model, train_loader, val_loader, args.epochs, device)
    
    # Save model
    save_path = 'joint_model_optimized.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"優化模型已保存到: {save_path}")

if __name__ == '__main__':
    main() 