#!/usr/bin/env python3
"""
檢測任務優化的 joint detection + classification 訓練腳本
重點提升 IoU 和 mAP 性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import yaml
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random
import os

class DetectionOptimizedJointModel(nn.Module):
    """專門針對檢測任務優化的聯合模型"""
    
    def __init__(self, num_classes=4):
        super(DetectionOptimizedJointModel, self).__init__()
        
        # 更深的backbone網絡 - 專門為檢測優化
        self.backbone = nn.Sequential(
            # 第一層卷積塊 - 增強特徵提取
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二層卷積塊 - 檢測特化
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三層卷積塊 - 高級特徵
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四層卷積塊 - 精細特徵
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第五層卷積塊 - 檢測特化
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 檢測頭 - 專門優化邊界框預測
        self.detection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8)  # x1,y1,x2,y2,x3,y3,x4,y4
        )
        
        # 分類頭 - 保持原有性能
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # 檢測注意力機制 - 增強檢測能力
        self.detection_attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Backbone特徵提取
        features = self.backbone(x)
        
        # 檢測注意力
        attention_weights = self.detection_attention(features)
        attended_features = features * attention_weights
        
        # 檢測預測
        bbox_pred = self.detection_head(attended_features)
        
        # 分類預測
        cls_pred = self.classification_head(features)
        
        return bbox_pred, cls_pred

class DetectionOptimizedDataset(Dataset):
    """檢測優化的數據集"""
    
    def __init__(self, data_dir, split='train', transform=None, nc=4, oversample_minority=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.nc = nc
        self.oversample_minority = oversample_minority
        
        # 載入圖像和標籤
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'
        
        self.image_files = sorted(list(self.images_dir.glob('*.png')))
        self.label_files = sorted(list(self.labels_dir.glob('*.txt')))
        
        # 智能過採樣
        if self.oversample_minority and split == 'train':
            self._smart_oversample_minority_classes()
    
    def _smart_oversample_minority_classes(self):
        """智能過採樣少數類別"""
        class_counts = {i: 0 for i in range(self.nc)}
        
        # 統計每個類別的樣本數
        for label_file in self.label_files:
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        parts = lines[0].strip().split()
                        if len(parts) >= 1:
                            class_id = int(parts[0])
                            if class_id < self.nc:
                                class_counts[class_id] += 1
        
        # 找到最大類別數量
        max_count = max(class_counts.values())
        
        # 為每個類別創建過採樣索引
        oversampled_indices = []
        for i, label_file in enumerate(self.label_files):
            oversampled_indices.append(i)
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        parts = lines[0].strip().split()
                        if len(parts) >= 1:
                            class_id = int(parts[0])
                            if class_id < self.nc:
                                # 計算需要過採樣的次數
                                current_count = class_counts[class_id]
                                if current_count < max_count:
                                    oversample_times = max_count // current_count
                                    for _ in range(oversample_times - 1):
                                        oversampled_indices.append(i)
        
        # 更新文件列表
        self.image_files = [self.image_files[i] for i in oversampled_indices]
        self.label_files = [self.label_files[i] for i in oversampled_indices]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 載入圖像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 載入標籤
        label_path = self.label_files[idx]
        
        # 載入檢測標籤 (第一行)
        bbox_target = torch.zeros(8)  # x1,y1,x2,y2,x3,y3,x4,y4
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 1:
                    parts = lines[0].strip().split()
                    if len(parts) >= 9:  # class_id + 8 coordinates
                        # 跳過class_id，載入8個座標
                        bbox_target = torch.tensor([float(x) for x in parts[1:9]])
        
        # 載入分類標籤 (第二行)
        cls_target = torch.zeros(self.nc)
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    parts = lines[1].strip().split()
                    if len(parts) >= self.nc:
                        cls_target = torch.tensor([float(x) for x in parts[:self.nc]])
        
        # 應用變換
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'bbox': bbox_target,
            'classification': cls_target
        }

def get_detection_optimized_transforms(split='train'):
    """檢測優化的數據變換"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.RandomHorizontalFlip(p=0.3),  # 適度的水平翻轉
            transforms.RandomRotation(degrees=5),     # 輕微旋轉
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class IoULoss(nn.Module):
    """IoU損失函數 - 專門針對檢測優化"""
    
    def __init__(self, reduction='mean'):
        super(IoULoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        # 計算IoU損失
        # pred: (batch_size, 8) - x1,y1,x2,y2,x3,y3,x4,y4
        # target: (batch_size, 8) - x1,y1,x2,y2,x3,y3,x4,y4
        
        # 簡化為矩形IoU計算
        pred_x1, pred_y1, pred_x2, pred_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        target_x1, target_y1, target_x2, target_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
        
        # 計算交集
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 計算並集
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # 計算IoU
        iou = inter_area / (union_area + 1e-6)
        
        # IoU損失 = 1 - IoU
        iou_loss = 1 - iou
        
        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        else:
            return iou_loss

def train_model_detection_optimized(model, train_loader, val_loader, num_epochs=50, device='cuda', 
                                   bbox_weight=15.0, cls_weight=5.0, enable_earlystop=False):
    """檢測優化的聯合訓練"""
    model = model.to(device)
    
    # 檢測優化的損失函數
    bbox_criterion_iou = IoULoss()  # IoU損失
    bbox_criterion_smooth = nn.SmoothL1Loss()  # Smooth L1損失
    cls_criterion = nn.CrossEntropyLoss()  # 分類損失
    
    # 檢測優化的優化器
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 0.0001},
        {'params': model.detection_head.parameters(), 'lr': 0.001},
        {'params': model.classification_head.parameters(), 'lr': 0.0005},
        {'params': model.detection_attention.parameters(), 'lr': 0.001}
    ], weight_decay=0.01)
    
    # 檢測優化的學習率調度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[0.001, 0.01, 0.005, 0.01],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_iou = 0.0
    best_model_state = None
    patience = 20  # 增加耐心值
    patience_counter = 0
    
    # 訓練循環
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        bbox_loss_sum = 0
        cls_loss_sum = 0
        iou_sum = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            bbox_targets = batch['bbox'].to(device)
            cls_targets = batch['classification'].to(device)
            
            optimizer.zero_grad()
            
            # 前向傳播
            bbox_pred, cls_pred = model(images)
            
            # 計算損失
            bbox_loss_iou = bbox_criterion_iou(bbox_pred, bbox_targets)
            bbox_loss_smooth = bbox_criterion_smooth(bbox_pred, bbox_targets)
            
            # 組合檢測損失
            bbox_loss = 0.7 * bbox_loss_iou + 0.3 * bbox_loss_smooth
            
            # 分類損失
            cls_loss = cls_criterion(cls_pred, cls_targets.argmax(dim=1))
            
            # 總損失 - 檢測權重更高
            loss = bbox_weight * bbox_loss + cls_weight * cls_loss
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            bbox_loss_sum += bbox_loss.item()
            cls_loss_sum += cls_loss.item()
            
            # 計算IoU
            with torch.no_grad():
                iou = 1 - bbox_loss_iou.item()
                iou_sum += iou
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f} (Bbox: {bbox_loss.item():.4f}, Cls: {cls_loss.item():.4f}, IoU: {iou:.4f})')
        
        # 驗證
        model.eval()
        val_loss = 0
        val_bbox_loss = 0
        val_cls_loss = 0
        val_cls_accuracy = 0
        val_iou_sum = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                bbox_targets = batch['bbox'].to(device)
                cls_targets = batch['classification'].to(device)
                
                bbox_pred, cls_pred = model(images)
                
                # 計算驗證損失
                bbox_loss_iou = bbox_criterion_iou(bbox_pred, bbox_targets)
                bbox_loss_smooth = bbox_criterion_smooth(bbox_pred, bbox_targets)
                bbox_loss = 0.7 * bbox_loss_iou + 0.3 * bbox_loss_smooth
                cls_loss = cls_criterion(cls_pred, cls_targets.argmax(dim=1))
                loss = bbox_weight * bbox_loss + cls_weight * cls_loss
                
                # 計算準確率
                cls_accuracy = (cls_pred.argmax(dim=1) == cls_targets.argmax(dim=1)).float().mean()
                
                # 計算IoU
                iou = 1 - bbox_loss_iou.item()
                
                val_loss += loss.item()
                val_bbox_loss += bbox_loss.item()
                val_cls_loss += cls_loss.item()
                val_cls_accuracy += cls_accuracy * images.size(0)
                val_iou_sum += iou * images.size(0)
                val_samples += images.size(0)
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls_accuracy = val_cls_accuracy / val_samples
        avg_val_iou = val_iou_sum / val_samples
        
        # 基於IoU保存最佳模型
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"保存最佳模型 (val_iou: {avg_val_iou:.4f})")
        else:
            patience_counter += 1
        
        # 早停
        if enable_earlystop and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} (Bbox: {bbox_loss_sum/len(train_loader):.4f}, Cls: {cls_loss_sum/len(train_loader):.4f})')
        print(f'  Val Loss: {avg_val_loss:.4f} (Bbox: {val_bbox_loss/len(val_loader):.4f}, Cls: {val_cls_loss/len(val_loader):.4f})')
        print(f'  Val Cls Accuracy: {avg_val_cls_accuracy:.4f}')
        print(f'  Val IoU: {avg_val_iou:.4f}')
        print(f'  Best Val IoU: {best_val_iou:.4f}')
        print('-' * 50)
    
    # 載入最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Detection Optimized Joint Training')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--bbox-weight', type=float, default=15.0, help='Bbox loss weight')
    parser.add_argument('--cls-weight', type=float, default=5.0, help='Classification loss weight')
    parser.add_argument('--enable-earlystop', action='store_true', help='Enable early stopping')
    args = parser.parse_args()
    
    # 載入數據配置
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_dir = Path(args.data).parent
    nc = data_config['nc']
    
    # 設備設置
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用設備: {device}")
    print(f"類別數量: {nc}")
    
    # 檢測優化的數據變換
    train_transform = get_detection_optimized_transforms('train')
    val_transform = get_detection_optimized_transforms('val')
    
    # 創建數據集
    train_dataset = DetectionOptimizedDataset(data_dir, 'train', train_transform, nc, oversample_minority=True)
    val_dataset = DetectionOptimizedDataset(data_dir, 'val', val_transform, nc, oversample_minority=False)
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 創建檢測優化模型
    model = DetectionOptimizedJointModel(num_classes=nc)
    
    # 訓練模型
    print("開始檢測優化訓練...")
    print("檢測優化措施:")
    print("- 更深層檢測特化網絡")
    print("- IoU損失函數")
    print("- 檢測注意力機制")
    print("- 高檢測損失權重 (15.0)")
    print("- 檢測特化數據增強")
    print("- 基於IoU的模型選擇")
    
    trained_model = train_model_detection_optimized(
        model, train_loader, val_loader, args.epochs, device, 
        bbox_weight=args.bbox_weight, cls_weight=args.cls_weight, 
        enable_earlystop=args.enable_earlystop
    )
    
    # 保存模型
    save_path = 'joint_model_detection_optimized.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"檢測優化模型已保存到: {save_path}")

if __name__ == '__main__':
    main() 