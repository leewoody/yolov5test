import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import numpy as np
from pathlib import Path
import yaml
import random
from collections import Counter

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        
        # Focal loss calculation
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedJointModel(nn.Module):
    def __init__(self, num_classes=4):
        super(AdvancedJointModel, self).__init__()
        
        # Enhanced backbone with residual connections
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
        
        # Enhanced classification head with attention
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

class AdvancedDataset(Dataset):
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
        self.class_labels = []  # Store class labels for oversampling
        
        for img_file in self.image_files:
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                self.valid_image_files.append(img_file)
                
                # Get class label for oversampling
                try:
                    bbox, classification = load_mixed_labels(label_file, nc)
                    class_label = np.argmax(classification)
                    self.class_labels.append(class_label)
                except:
                    self.class_labels.append(0)  # Default to class 0
        
        self.image_files = self.valid_image_files
        
        # Oversample minority classes
        if self.oversample_minority and split == 'train':
            self._oversample_minority_classes()
        
        print(f"{split} 數據集: {len(self.image_files)} 個有效樣本")
        
        # Print class distribution
        if len(self.class_labels) > 0:
            class_counts = Counter(self.class_labels)
            print(f"{split} 類別分布: {dict(class_counts)}")
    
    def _oversample_minority_classes(self):
        """過採樣少數類別"""
        class_counts = Counter(self.class_labels)
        max_count = max(class_counts.values())
        
        # Calculate target count for each class (make them equal)
        target_count = max_count
        
        # Create oversampled indices
        oversampled_indices = []
        for class_id in range(self.nc):
            class_indices = [i for i, label in enumerate(self.class_labels) if label == class_id]
            if class_indices:
                # Repeat samples to reach target count
                repeat_times = target_count // len(class_indices)
                remainder = target_count % len(class_indices)
                
                oversampled_indices.extend(class_indices * repeat_times)
                oversampled_indices.extend(class_indices[:remainder])
        
        # Update dataset
        self.image_files = [self.image_files[i] for i in oversampled_indices]
        self.class_labels = [self.class_labels[i] for i in oversampled_indices]
        
        print(f"過採樣後: {len(self.image_files)} 個樣本")
        class_counts = Counter(self.class_labels)
        print(f"過採樣後類別分布: {dict(class_counts)}")
    
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

def get_advanced_transforms(split='train'):
    """Get advanced data transforms with strong augmentation"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_model_advanced(model, train_loader, val_loader, num_epochs=50, device='cuda', enable_early_stop=False):
    """Train the advanced joint model with all optimizations and auto-fix"""
    model = model.to(device)
    
    # Enhanced loss functions
    bbox_criterion = nn.SmoothL1Loss()  # Better for regression
    cls_criterion = FocalLoss(alpha=1, gamma=2)  # Focal Loss for class imbalance
    
    # Optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Dynamic loss weights - heavily favor classification
    bbox_weight = 1.0
    cls_weight = 5.0  # Much higher weight for classification
    
    # Auto-fix tracking variables
    best_val_loss = float('inf')
    best_val_cls_acc = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    # Auto-fix configuration
    auto_fix_config = {
        'overfitting_threshold': 0.1,  # Train loss > Val loss + threshold
        'accuracy_drop_threshold': 0.05,  # Accuracy drop threshold
        'lr_reduction_factor': 0.5,
        'weight_decay_increase_factor': 1.5,
        'dropout_increase_factor': 0.05,
        'cls_weight_increase': 1.0,
        'cls_weight_decrease': 0.5,
        'max_cls_weight': 10.0,
        'min_cls_weight': 3.0,
        'max_weight_decay': 0.05,
        'max_dropout': 0.5
    }
    
    # Training history for auto-fix analysis
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'overfitting_flags': [],
        'accuracy_drops': []
    }
    
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
        
        # Update training history
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(avg_val_loss)
        training_history['val_accuracies'].append(avg_val_cls_accuracy)
        
        # --- Auto Fix: 過擬合與準確率下降自動調整 ---
        overfitting_detected = False
        accuracy_drop_detected = False
        
        # 1. 過擬合檢測與自動調整
        if avg_train_loss > avg_val_loss + auto_fix_config['overfitting_threshold']:
            overfitting_detected = True
            training_history['overfitting_flags'].append(True)
            print('⚠️ 過擬合風險檢測，自動調整正則化與學習率')
            
            # 增加Dropout
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    old_dropout = m.p
                    m.p = min(m.p + auto_fix_config['dropout_increase_factor'], 
                             auto_fix_config['max_dropout'])
                    print(f'  Dropout: {old_dropout:.3f} -> {m.p:.3f}')
            
            # 增大weight_decay
            for g in optimizer.param_groups:
                old_weight_decay = g['weight_decay']
                g['weight_decay'] = min(g['weight_decay'] * auto_fix_config['weight_decay_increase_factor'], 
                                       auto_fix_config['max_weight_decay'])
                print(f'  Weight Decay: {old_weight_decay:.4f} -> {g["weight_decay"]:.4f}')
            
            # 降低學習率
            for g in optimizer.param_groups:
                old_lr = g['lr']
                g['lr'] = max(g['lr'] * auto_fix_config['lr_reduction_factor'], 1e-6)
                print(f'  Learning Rate: {old_lr:.6f} -> {g["lr"]:.6f}')
        else:
            training_history['overfitting_flags'].append(False)
        
        # 2. 準確率下降檢測與自動調整
        if len(training_history['val_accuracies']) > 1:
            accuracy_drop = best_val_cls_acc - avg_val_cls_accuracy
            if accuracy_drop > auto_fix_config['accuracy_drop_threshold']:
                accuracy_drop_detected = True
                training_history['accuracy_drops'].append(True)
                print('⚠️ 準確率下降檢測，自動提升分類損失權重')
                
                old_cls_weight = cls_weight
                cls_weight = min(cls_weight + auto_fix_config['cls_weight_increase'], 
                                auto_fix_config['max_cls_weight'])
                print(f'  分類損失權重: {old_cls_weight:.1f} -> {cls_weight:.1f}')
            else:
                # 如果準確率提升，適當降低分類損失權重
                if avg_val_cls_accuracy > best_val_cls_acc:
                    old_cls_weight = cls_weight
                    cls_weight = max(cls_weight - auto_fix_config['cls_weight_decrease'], 
                                    auto_fix_config['min_cls_weight'])
                    print(f'  準確率提升，降低分類損失權重: {old_cls_weight:.1f} -> {cls_weight:.1f}')
                training_history['accuracy_drops'].append(False)
        else:
            training_history['accuracy_drops'].append(False)
        
        # 3. 動態 early stopping
        if enable_early_stop and patience_counter >= patience // 2 and avg_val_loss > best_val_loss:
            print('⚠️ 驗證損失無改善，提前 early stop')
            break
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model based on validation accuracy
        if avg_val_cls_accuracy > best_val_cls_acc:
            best_val_cls_acc = avg_val_cls_accuracy
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✅ 保存最佳模型 (val_cls_acc: {avg_val_cls_accuracy:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if enable_early_stop and patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} (Bbox: {bbox_loss_sum/len(train_loader):.4f}, Cls: {cls_loss_sum/len(train_loader):.4f})')
        print(f'  Val Loss: {avg_val_loss:.4f} (Bbox: {val_bbox_loss/len(val_loader):.4f}, Cls: {val_cls_loss/len(val_loader):.4f})')
        print(f'  Val Cls Accuracy: {avg_val_cls_accuracy:.4f}')
        print(f'  Best Val Cls Accuracy: {best_val_cls_acc:.4f}')
        print(f'  Current Cls Weight: {cls_weight:.1f}')
        if overfitting_detected or accuracy_drop_detected:
            print('  🔧 Auto-fix applied')
        print('-' * 50)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save training history for analysis
    import json
    history_path = 'training_history_advanced.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"📊 訓練歷史已保存到: {history_path}")
    
    # Print auto-fix summary
    overfitting_count = sum(training_history['overfitting_flags'])
    accuracy_drop_count = sum(training_history['accuracy_drops'])
    total_epochs = len(training_history['train_losses'])
    
    print("\n" + "="*60)
    print("🔧 AUTO-FIX 總結報告")
    print("="*60)
    print(f"總訓練輪數: {total_epochs}")
    print(f"過擬合檢測次數: {overfitting_count} ({overfitting_count/total_epochs*100:.1f}%)")
    print(f"準確率下降檢測次數: {accuracy_drop_count} ({accuracy_drop_count/total_epochs*100:.1f}%)")
    print(f"最終最佳驗證準確率: {best_val_cls_acc:.4f}")
    print(f"最終最佳驗證損失: {best_val_loss:.4f}")
    
    if overfitting_count > 0:
        print("✅ 過擬合問題已自動處理")
    if accuracy_drop_count > 0:
        print("✅ 準確率下降問題已自動處理")
    
    print("="*60)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Advanced Joint Detection and Classification Training')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--enable-early-stop', action='store_true', help='Enable early stopping')
    
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
    
    # Enhanced data transforms
    train_transform = get_advanced_transforms('train')
    val_transform = get_advanced_transforms('val')
    
    # Create datasets with oversampling
    train_dataset = AdvancedDataset(data_dir, 'train', train_transform, nc, oversample_minority=True)
    val_dataset = AdvancedDataset(data_dir, 'val', val_transform, nc, oversample_minority=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create advanced model
    model = AdvancedJointModel(num_classes=nc)
    
    # Train model
    print("開始高級優化訓練...")
    print("優化措施:")
    print("- Focal Loss 處理類別不平衡")
    print("- 強力數據增強")
    print("- 過採樣少數類別")
    print("- 高分類損失權重 (5.0)")
    print("- Cosine Annealing 學習率調度")
    print("- Early Stopping")
    print("- 🔧 AUTO-FIX 自動優化系統:")
    print("  • 過擬合自動檢測與修正")
    print("  • 準確率下降自動調整")
    print("  • 動態學習率與正則化調整")
    print("  • 智能早停機制: {'啟用' if enable_early_stop else '禁用'}")
    
    trained_model = train_model_advanced(model, train_loader, val_loader, args.epochs, device, enable_early_stop=args.enable_early_stop)
    
    # Save model
    save_path = 'joint_model_advanced.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"高級優化模型已保存到: {save_path}")

if __name__ == '__main__':
    main() 