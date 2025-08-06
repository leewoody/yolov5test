#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization Plan for 70% Performance Improvement
70%性能提升優化計劃
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import yaml
from pathlib import Path
import random
from collections import Counter
import torch.nn.functional as F
import torchvision.models as models

# 添加項目路徑
sys.path.append(str(Path(__file__).parent))

class AdvancedJointModel(nn.Module):
    """高級聯合模型 - 使用預訓練骨幹網絡"""
    def __init__(self, num_detection_classes=4, num_classification_classes=3):
        super(AdvancedJointModel, self).__init__()
        
        # 使用預訓練的ResNet50作為骨幹網絡
        self.backbone = models.resnet50(pretrained=True)
        # 移除最後的分類層
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 特徵維度
        feature_dim = 2048
        
        # 注意力機制
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        # 檢測頭 - 更深的網絡
        self.detection_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_detection_classes * 4)  # 4個檢測類別，每個4個值
        )
        
        # 分類頭 - 更深的網絡
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classification_classes)
        )
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 骨幹網絡特徵提取
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # 展平
        
        # 注意力機制
        features_reshaped = features.unsqueeze(1)  # [batch, 1, features]
        attended_features, _ = self.attention(features_reshaped, features_reshaped, features_reshaped)
        attended_features = attended_features.squeeze(1)  # [batch, features]
        
        # 檢測預測
        detection_output = self.detection_head(attended_features)
        detection_output = detection_output.view(-1, 4, 4)  # [batch, 4, 4]
        
        # 分類預測
        classification_output = self.classification_head(attended_features)
        
        return detection_output, classification_output


class AdvancedFocalLoss(nn.Module):
    """高級Focal Loss - 處理類別不平衡"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(AdvancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # 轉換為分類標籤
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


class AdvancedDataset(Dataset):
    """高級數據集 - 包含強化的數據增強和平衡策略"""
    def __init__(self, data_dir, split='train', transform=None, nc=4, oversample_minority=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.nc = nc
        self.oversample_minority = oversample_minority
        
        # 載入數據配置
        with open(self.data_dir / 'data.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        self.class_names = config.get('names', [])
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'
        
        # 獲取所有圖像文件
        self.image_files = list(self.images_dir.glob('*.png')) + list(self.images_dir.glob('*.jpg'))
        
        # 智能過採樣
        if self.oversample_minority and split == 'train':
            self.image_files = self._advanced_oversample()
        
        print(f"{split} 數據集: {len(self.image_files)} 個有效樣本")
        
        # 設置轉換
        if transform is None:
            self.transform = self._get_advanced_transforms(split)
        else:
            self.transform = transform
    
    def _advanced_oversample(self):
        """高級過採樣策略"""
        # 分析類別分布
        class_counts = Counter()
        for img_file in self.image_files:
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        try:
                            class_id = int(lines[0].split()[0])
                            class_counts[class_id] += 1
                        except:
                            pass
        
        print(f"原始類別分布: {dict(class_counts)}")
        
        # 計算目標樣本數
        max_count = max(class_counts.values()) if class_counts else 1000
        target_count = int(max_count * 1.5)  # 增加50%
        
        # 過採樣
        oversampled_files = []
        for img_file in self.image_files:
            oversampled_files.append(img_file)
            
            # 檢查是否需要過採樣
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        try:
                            class_id = int(lines[0].split()[0])
                            current_count = class_counts[class_id]
                            
                            # 計算過採樣倍數
                            if current_count < target_count:
                                oversample_factor = max(1, int(target_count / current_count))
                                for _ in range(oversample_factor - 1):
                                    oversampled_files.append(img_file)
                        except:
                            pass
        
        print(f"過採樣後: {len(oversampled_files)} 個樣本")
        return oversampled_files
    
    def _get_advanced_transforms(self, split):
        """高級數據增強"""
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        label_file = self.labels_dir / f"{img_file.stem}.txt"
        
        # 載入圖像
        image = Image.open(img_file).convert('RGB')
        image = self.transform(image)
        
        # 載入標籤
        bbox_target = np.zeros(4)
        cls_target = np.zeros(3)
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    # 檢測標籤 (第一行)
                    bbox_parts = lines[0].strip().split()
                    if len(bbox_parts) >= 5:
                        bbox_target = np.array([float(x) for x in bbox_parts[1:5]])
                    
                    # 分類標籤 (第二行)
                    cls_parts = lines[1].strip().split()
                    if len(cls_parts) == 3:
                        cls_target = np.array([float(x) for x in cls_parts])
        
        return image, torch.FloatTensor(bbox_target), torch.FloatTensor(cls_target)


class AdvancedTrainer:
    """高級訓練器"""
    def __init__(self, model, train_loader, val_loader, device, 
                 bbox_weight=1.0, cls_weight=2.0, learning_rate=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight
        
        # 優化器 - 使用AdamW
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 學習率調度器 - 使用CosineAnnealingWarmRestarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 損失函數
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_loss = AdvancedFocalLoss(alpha=1, gamma=2)
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
    
    def train_epoch(self):
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        bbox_loss_sum = 0
        cls_loss_sum = 0
        
        for batch_idx, (images, bbox_targets, cls_targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            bbox_targets = bbox_targets.to(self.device)
            cls_targets = cls_targets.to(self.device)
            
            # 前向傳播
            bbox_preds, cls_preds = self.model(images)
            
            # 計算損失
            bbox_loss = self.bbox_loss(bbox_preds.view(-1, 4), bbox_targets)
            cls_loss = self.cls_loss(cls_preds, cls_targets)
            
            total_loss = self.bbox_weight * bbox_loss + self.cls_weight * cls_loss
            
            # 反向傳播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            bbox_loss_sum += bbox_loss.item()
            cls_loss_sum += cls_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {total_loss.item():.4f} "
                      f"(Bbox: {bbox_loss.item():.4f}, Cls: {cls_loss.item():.4f})")
        
        return bbox_loss_sum / len(self.train_loader), cls_loss_sum / len(self.train_loader)
    
    def validate(self):
        """驗證"""
        self.model.eval()
        total_loss = 0
        bbox_loss_sum = 0
        cls_loss_sum = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for images, bbox_targets, cls_targets in self.val_loader:
                images = images.to(self.device)
                bbox_targets = bbox_targets.to(self.device)
                cls_targets = cls_targets.to(self.device)
                
                # 前向傳播
                bbox_preds, cls_preds = self.model(images)
                
                # 計算損失
                bbox_loss = self.bbox_loss(bbox_preds.view(-1, 4), bbox_targets)
                cls_loss = self.cls_loss(cls_preds, cls_targets)
                
                total_loss = self.bbox_weight * bbox_loss + self.cls_weight * cls_loss
                
                bbox_loss_sum += bbox_loss.item()
                cls_loss_sum += cls_loss.item()
                
                # 計算分類準確率
                pred_labels = torch.argmax(cls_preds, dim=1)
                true_labels = torch.argmax(cls_targets, dim=1)
                correct_predictions += (pred_labels == true_labels).sum().item()
                total_predictions += pred_labels.size(0)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return (bbox_loss_sum / len(self.val_loader), 
                cls_loss_sum / len(self.val_loader), 
                accuracy)
    
    def train(self, num_epochs):
        """完整訓練流程"""
        print("🚀 開始高級優化訓練...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 訓練
            train_bbox_loss, train_cls_loss = self.train_epoch()
            
            # 驗證
            val_bbox_loss, val_cls_loss, val_accuracy = self.validate()
            
            # 學習率調度
            self.scheduler.step()
            
            print(f"Train - Bbox Loss: {train_bbox_loss:.4f}, Cls Loss: {train_cls_loss:.4f}")
            print(f"Val - Bbox Loss: {val_bbox_loss:.4f}, Cls Loss: {val_cls_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # 早停檢查
            val_loss = val_bbox_loss + val_cls_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_optimized_model.pth')
                print(f"✅ 保存最佳模型 (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"🛑 早停觸發 (patience: {self.patience})")
                    break


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='70%性能提升優化計劃')
    parser.add_argument('--data', type=str, required=True, help='數據配置文件路徑')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--bbox-weight', type=float, default=1.0, help='檢測損失權重')
    parser.add_argument('--cls-weight', type=float, default=2.0, help='分類損失權重')
    
    args = parser.parse_args()
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入數據
    data_dir = Path(args.data).parent
    
    train_dataset = AdvancedDataset(data_dir, split='train', nc=4)
    val_dataset = AdvancedDataset(data_dir, split='val', nc=4, oversample_minority=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 創建模型
    model = AdvancedJointModel(num_detection_classes=4, num_classification_classes=3)
    model.to(device)
    
    # 創建訓練器
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        bbox_weight=args.bbox_weight,
        cls_weight=args.cls_weight,
        learning_rate=args.learning_rate
    )
    
    # 開始訓練
    trainer.train(args.epochs)
    
    print("🎉 優化訓練完成！")


if __name__ == "__main__":
    main() 