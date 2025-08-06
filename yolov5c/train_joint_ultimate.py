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
import random
from collections import Counter
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss for better generalization"""
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class FocalLoss(nn.Module):
    """Enhanced Focal Loss with alpha balancing"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert one-hot to class indices
        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class UltimateJointModel(nn.Module):
    def __init__(self, num_classes=4):
        super(UltimateJointModel, self).__init__()
        
        # Deeper backbone with residual connections
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Enhanced detection head
        self.detection_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4)
        )
        
        # Enhanced classification head with attention mechanism
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Attention mechanism for classification
        self.attention = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Apply attention to features
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        detection_output = self.detection_head(features)
        classification_output = self.classification_head(attended_features)
        
        return detection_output, classification_output

def load_mixed_labels(label_path, nc=4):
    """Load mixed labels with one-hot encoding classification"""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Parse bbox line (first line)
        bbox_line = lines[0].strip().split()
        if len(bbox_line) >= 5:
            bbox = [float(x) for x in bbox_line[1:]]
        else:
            bbox = [0.5, 0.5, 0.1, 0.1]
        
        # Parse classification line (second line) - one-hot encoding
        if len(lines) > 1:
            cls_line = lines[1].strip().split()
            classification = [float(x) for x in cls_line]
            if len(classification) != nc:
                if len(classification) < nc:
                    classification.extend([0.0] * (nc - len(classification)))
                else:
                    classification = classification[:nc]
        else:
            classification = [1.0] + [0.0] * (nc - 1)
        
        return bbox, classification
    except Exception as e:
        print(f"警告: 無法讀取標籤 {label_path}: {e}")
        return [0.5, 0.5, 0.1, 0.1], [1.0] + [0.0] * (nc - 1)

class UltimateDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, nc=4, oversample_minority=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.nc = nc
        self.oversample_minority = oversample_minority
        
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
        self.class_labels = []
        
        for img_file in self.image_files:
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                self.valid_image_files.append(img_file)
                
                try:
                    bbox, classification = load_mixed_labels(label_file, nc)
                    class_label = np.argmax(classification)
                    self.class_labels.append(class_label)
                except:
                    self.class_labels.append(0)
        
        self.image_files = self.valid_image_files
        
        # Smart oversampling for minority classes
        if self.oversample_minority and split == 'train':
            self._smart_oversample_minority_classes()
        
        print(f"{split} 數據集: {len(self.image_files)} 個有效樣本")
        
        if len(self.class_labels) > 0:
            class_counts = Counter(self.class_labels)
            print(f"{split} 類別分布: {dict(class_counts)}")
    
    def _smart_oversample_minority_classes(self):
        """智能過採樣，考慮類別重要性"""
        class_counts = Counter(self.class_labels)
        max_count = max(class_counts.values())
        
        # Calculate target counts with different strategies
        target_counts = {}
        for class_id in range(self.nc):
            current_count = class_counts.get(class_id, 0)
            if current_count == 0:
                # For missing classes, use a reasonable target
                target_counts[class_id] = max_count // 2
            else:
                # For existing classes, balance them
                target_counts[class_id] = max_count
        
        # Create oversampled indices
        oversampled_indices = []
        for class_id in range(self.nc):
            class_indices = [i for i, label in enumerate(self.class_labels) if label == class_id]
            target_count = target_counts[class_id]
            
            if class_indices and target_count > 0:
                repeat_times = target_count // len(class_indices)
                remainder = target_count % len(class_indices)
                
                oversampled_indices.extend(class_indices * repeat_times)
                oversampled_indices.extend(class_indices[:remainder])
        
        # Update dataset
        self.image_files = [self.image_files[i] for i in oversampled_indices]
        self.class_labels = [self.class_labels[i] for i in oversampled_indices]
        
        print(f"智能過採樣後: {len(self.image_files)} 個樣本")
        class_counts = Counter(self.class_labels)
        print(f"智能過採樣後類別分布: {dict(class_counts)}")
    
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

def get_ultimate_transforms(split='train'):
    """Get ultimate data transforms with balanced augmentation"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Smaller size for faster training
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Reduced rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Reduced intensity
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Reduced transformation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_model_ultimate(model, train_loader, val_loader, num_epochs=50, device='cuda', bbox_weight=10.0, cls_weight=8.0, enable_earlystop=False):
    """Train the ultimate joint model with all advanced optimizations"""
    model = model.to(device)
    
    # Multiple loss functions for different scenarios
    bbox_criterion = nn.SmoothL1Loss()
    cls_criterion_focal = FocalLoss(alpha=1, gamma=2)
    cls_criterion_ce = nn.CrossEntropyLoss()
    
    # Advanced optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 0.0001},
        {'params': model.detection_head.parameters(), 'lr': 0.001},
        {'params': model.classification_head.parameters(), 'lr': 0.001},
        {'params': model.attention.parameters(), 'lr': 0.001}
    ], weight_decay=0.01)
    
    # Advanced learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[0.001, 0.01, 0.01, 0.01],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_cls_acc = 0.0
    best_model_state = None
    patience = 15  # Increased patience
    patience_counter = 0
    
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
            cls_loss = cls_criterion_focal(cls_pred, cls_targets)
            
            # Combined loss
            loss = bbox_weight * bbox_loss + cls_weight * cls_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
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
                cls_loss = cls_criterion_focal(cls_pred, cls_targets)
                loss = bbox_weight * bbox_loss + cls_weight * cls_loss
                
                # Calculate classification accuracy
                cls_pred_softmax = F.softmax(cls_pred, dim=1)
                target_classes = torch.argmax(cls_targets, dim=1)
                pred_classes = torch.argmax(cls_pred_softmax, dim=1)
                cls_accuracy = (pred_classes == target_classes).float().mean().item()
                
                val_loss += loss.item()
                val_bbox_loss += bbox_loss.item()
                val_cls_loss += cls_loss.item()
                val_cls_accuracy += cls_accuracy * images.size(0)
                val_samples += images.size(0)
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls_accuracy = val_cls_accuracy / val_samples
        
        # Save best model based on validation accuracy
        if avg_val_cls_accuracy > best_val_cls_acc:
            best_val_cls_acc = avg_val_cls_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"保存最佳模型 (val_cls_acc: {avg_val_cls_accuracy:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping (only if enabled)
        if enable_earlystop and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} (Bbox: {bbox_loss_sum/len(train_loader):.4f}, Cls: {cls_loss_sum/len(train_loader):.4f})')
        print(f'  Val Loss: {avg_val_loss:.4f} (Bbox: {val_bbox_loss/len(val_loader):.4f}, Cls: {val_cls_loss/len(val_loader):.4f})')
        print(f'  Val Cls Accuracy: {avg_val_cls_accuracy:.4f}')
        print(f'  Best Val Cls Accuracy: {best_val_cls_acc:.4f}')
        print('-' * 50)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Ultimate Joint Detection and Classification Training')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--bbox-weight', type=float, default=10.0, help='Bbox loss weight')
    parser.add_argument('--cls-weight', type=float, default=8.0, help='Classification loss weight')
    parser.add_argument('--enable-earlystop', action='store_true', help='Enable early stopping')
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
    
    # Ultimate data transforms
    train_transform = get_ultimate_transforms('train')
    val_transform = get_ultimate_transforms('val')
    
    # Create datasets with smart oversampling
    train_dataset = UltimateDataset(data_dir, 'train', train_transform, nc, oversample_minority=True)
    val_dataset = UltimateDataset(data_dir, 'val', val_transform, nc, oversample_minority=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create ultimate model
    model = UltimateJointModel(num_classes=nc)
    
    # Train model
    print("開始終極優化訓練...")
    print("終極優化措施:")
    print("- 更深層網絡架構")
    print("- 注意力機制")
    print("- 智能過採樣")
    print("- 平衡數據增強")
    print("- 超高分類損失權重 (8.0)")
    print("- OneCycleLR 學習率調度")
    print("- 分層學習率")
    print("- 增強 Early Stopping")
    
    trained_model = train_model_ultimate(model, train_loader, val_loader, args.epochs, device, bbox_weight=args.bbox_weight, cls_weight=args.cls_weight, enable_earlystop=args.enable_earlystop)
    
    # Save model
    save_path = 'joint_model_ultimate.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"終極優化模型已保存到: {save_path}")

if __name__ == '__main__':
    main() 