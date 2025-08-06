#!/usr/bin/env python3
"""
AR類別優化的 joint detection + classification 訓練腳本
重點提升AR類別的分類性能和檢測精度
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

class AROptimizedJointModel(nn.Module):
    """專門針對AR類別優化的聯合模型"""
    
    def __init__(self, num_classes=4):
        super(AROptimizedJointModel, self).__init__()
        
        # 增強的特徵提取backbone
        self.backbone = nn.Sequential(
            # 第一層 - 增強特徵提取
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二層 - AR特化特徵
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三層 - 高級特徵
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
            
            # 第四層 - 精細特徵
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
            
            # 第五層 - AR特化
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
        
        # AR特化的注意力機制
        self.ar_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # 檢測頭 - 優化邊界框預測
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
        
        # AR特化的分類頭
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # PSAX, PLAX, A4C
        )
        
    def forward(self, x):
        # Backbone特徵提取
        features = self.backbone(x)
        
        # AR特化注意力
        attention_weights = self.ar_attention(features.view(features.size(0), -1))
        attended_features = features * attention_weights.view(features.size(0), -1, 1, 1)
        
        # 檢測和分類預測
        bbox_pred = self.detection_head(attended_features)
        cls_pred = self.classification_head(attended_features)
        
        return bbox_pred, cls_pred

class AROptimizedDataset(Dataset):
    """AR類別優化的數據集"""
    
    def __init__(self, data_dir, split='train', transform=None, nc=4, ar_oversample=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.nc = nc
        self.ar_oversample = ar_oversample
        
        # 載入數據集配置
        with open(self.data_dir / 'data.yaml', 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # 獲取圖像和標籤路徑
        self.images_dir = self.data_dir / self.data_config[split]
        self.labels_dir = self.data_dir / self.data_config[split].replace('images', 'labels')
        
        self.image_files = sorted(list(self.images_dir.glob('*.png')))
        
        # AR類別過採樣
        if self.ar_oversample and split == 'train':
            self.image_files = self._ar_oversample()
        
        print(f"📊 {split} 數據集: {len(self.image_files)} 個樣本")
        
    def _ar_oversample(self):
        """AR類別過採樣策略"""
        ar_samples = []
        other_samples = []
        
        for img_file in self.image_files:
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        # 檢查第一行是否為AR類別 (class_id = 0)
                        try:
                            class_id = int(lines[0].split()[0])
                            if class_id == 0:  # AR類別
                                ar_samples.append(img_file)
                            else:
                                other_samples.append(img_file)
                        except:
                            other_samples.append(img_file)
                    else:
                        other_samples.append(img_file)
            else:
                other_samples.append(img_file)
        
        # AR類別過採樣 (增加3倍)
        oversampled_ar = ar_samples * 3
        
        # 組合所有樣本
        all_samples = oversampled_ar + other_samples
        random.shuffle(all_samples)
        
        print(f"🔄 AR類別過採樣: 原始AR樣本 {len(ar_samples)}, 過採樣後 {len(oversampled_ar)}")
        print(f"📈 總樣本數: {len(all_samples)}")
        
        return all_samples
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 載入圖像
        img_file = self.image_files[idx]
        image = Image.open(img_file).convert('RGB')
        
        # 載入標籤
        label_file = self.labels_dir / f"{img_file.stem}.txt"
        
        # 初始化標籤
        bbox_target = torch.zeros(8)  # x1,y1,x2,y2,x3,y3,x4,y4
        classification_target = torch.zeros(3)  # PSAX, PLAX, A4C
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
                if len(lines) >= 1:
                    # 解析檢測標籤 (第一行)
                    detection_line = lines[0].strip().split()
                    if len(detection_line) >= 5:
                        class_id = int(detection_line[0])
                        center_x = float(detection_line[1])
                        center_y = float(detection_line[2])
                        width = float(detection_line[3])
                        height = float(detection_line[4])
                        
                        # 轉換為邊界框座標
                        x1 = center_x - width/2
                        y1 = center_y - height/2
                        x2 = center_x + width/2
                        y2 = center_y + height/2
                        x3 = center_x + width/2
                        y3 = center_y - height/2
                        x4 = center_x - width/2
                        y4 = center_y + height/2
                        
                        bbox_target = torch.tensor([x1, y1, x2, y2, x3, y3, x4, y4], dtype=torch.float32)
                
                if len(lines) >= 2:
                    # 解析分類標籤 (第二行)
                    classification_line = lines[1].strip().split()
                    if len(classification_line) >= 3:
                        classification_target = torch.tensor([
                            float(classification_line[0]),  # PSAX
                            float(classification_line[1]),  # PLAX
                            float(classification_line[2])   # A4C
                        ], dtype=torch.float32)
        
        # 應用變換
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'bbox': bbox_target,
            'classification': classification_target
        }

def get_ar_optimized_transforms(split='train'):
    """AR優化的數據變換"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class FocalLoss(nn.Module):
    """Focal Loss for AR類別優化"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IoULoss(nn.Module):
    """IoU損失函數"""
    
    def __init__(self, reduction='mean'):
        super(IoULoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        # 簡化為矩形IoU計算
        # pred: (batch_size, 8) - x1,y1,x2,y2,x3,y3,x4,y4
        # target: (batch_size, 8) - x1,y1,x2,y2,x3,y3,x4,y4
        
        # 提取邊界框座標
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
        
        # IoU損失
        iou_loss = 1 - iou
        
        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        else:
            return iou_loss

def train_model_ar_optimized(model, train_loader, val_loader, num_epochs=50, device='cuda', 
                           bbox_weight=12.0, cls_weight=8.0, enable_earlystop=False):
    """AR優化的聯合訓練"""
    model = model.to(device)
    
    # AR特化的損失函數
    bbox_criterion_iou = IoULoss()
    bbox_criterion_smooth = nn.SmoothL1Loss()
    cls_criterion_focal = FocalLoss(alpha=1, gamma=2)
    cls_criterion_ce = nn.CrossEntropyLoss()
    
    # AR特化的優化器
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 0.0001},
        {'params': model.ar_attention.parameters(), 'lr': 0.001},
        {'params': model.detection_head.parameters(), 'lr': 0.001},
        {'params': model.classification_head.parameters(), 'lr': 0.001}
    ], weight_decay=0.01)
    
    # AR特化的學習率調度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[0.001, 0.01, 0.01, 0.01],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_ar_acc = 0.0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    # 訓練循環
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        bbox_loss_sum = 0
        cls_loss_sum = 0
        ar_accuracy_sum = 0
        ar_samples = 0
        
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
            bbox_loss = 0.7 * bbox_loss_iou + 0.3 * bbox_loss_smooth
            
            # AR特化的分類損失
            cls_loss_focal = cls_criterion_focal(cls_pred, cls_targets.argmax(dim=1))
            cls_loss_ce = cls_criterion_ce(cls_pred, cls_targets.argmax(dim=1))
            cls_loss = 0.6 * cls_loss_focal + 0.4 * cls_loss_ce
            
            # 總損失
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
            
            # 計算AR類別準確率
            with torch.no_grad():
                cls_pred_softmax = torch.softmax(cls_pred, dim=1)
                cls_pred_class = cls_pred_softmax.argmax(dim=1)
                cls_target_class = cls_targets.argmax(dim=1)
                
                # 只計算AR類別的準確率
                ar_mask = (cls_target_class == 0)  # AR類別
                if ar_mask.sum() > 0:
                    ar_correct = (cls_pred_class[ar_mask] == cls_target_class[ar_mask]).sum()
                    ar_accuracy_sum += ar_correct.item()
                    ar_samples += ar_mask.sum().item()
            
            if batch_idx % 20 == 0:
                ar_acc = ar_accuracy_sum / max(ar_samples, 1)
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f} (Bbox: {bbox_loss.item():.4f}, Cls: {cls_loss.item():.4f}), '
                      f'AR Acc: {ar_acc:.4f}')
        
        # 驗證
        model.eval()
        val_loss = 0
        val_bbox_loss = 0
        val_cls_loss = 0
        val_cls_accuracy = 0
        val_ar_accuracy = 0
        val_samples = 0
        val_ar_samples = 0
        
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
                
                cls_loss_focal = cls_criterion_focal(cls_pred, cls_targets.argmax(dim=1))
                cls_loss_ce = cls_criterion_ce(cls_pred, cls_targets.argmax(dim=1))
                cls_loss = 0.6 * cls_loss_focal + 0.4 * cls_loss_ce
                
                loss = bbox_weight * bbox_loss + cls_weight * cls_loss
                
                val_loss += loss.item()
                val_bbox_loss += bbox_loss.item()
                val_cls_loss += cls_loss.item()
                
                # 計算準確率
                cls_pred_softmax = torch.softmax(cls_pred, dim=1)
                cls_pred_class = cls_pred_softmax.argmax(dim=1)
                cls_target_class = cls_targets.argmax(dim=1)
                
                val_cls_accuracy += (cls_pred_class == cls_target_class).sum().item()
                val_samples += cls_target_class.size(0)
                
                # AR類別準確率
                ar_mask = (cls_target_class == 0)
                if ar_mask.sum() > 0:
                    ar_correct = (cls_pred_class[ar_mask] == cls_target_class[ar_mask]).sum()
                    val_ar_accuracy += ar_correct.item()
                    val_ar_samples += ar_mask.sum().item()
        
        # 計算平均指標
        avg_val_loss = val_loss / len(val_loader)
        avg_val_bbox_loss = val_bbox_loss / len(val_loader)
        avg_val_cls_loss = val_cls_loss / len(val_loader)
        val_cls_acc = val_cls_accuracy / max(val_samples, 1)
        val_ar_acc = val_ar_accuracy / max(val_ar_samples, 1)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Val Loss: {avg_val_loss:.4f} (Bbox: {avg_val_bbox_loss:.4f}, Cls: {avg_val_cls_loss:.4f}), '
              f'Val Cls Acc: {val_cls_acc:.4f}, Val AR Acc: {val_ar_acc:.4f}')
        
        # 早停檢查
        if val_ar_acc > best_val_ar_acc:
            best_val_ar_acc = val_ar_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if enable_earlystop and patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 保存最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, 'joint_model_ar_optimized.pth')
        print(f'✅ 最佳AR優化模型已保存，AR準確率: {best_val_ar_acc:.4f}')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='AR優化的聯合訓練')
    parser.add_argument('--data', type=str, default='converted_dataset_fixed/data.yaml', help='數據集配置文件')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='auto', help='設備')
    parser.add_argument('--bbox-weight', type=float, default=12.0, help='檢測損失權重')
    parser.add_argument('--cls-weight', type=float, default=8.0, help='分類損失權重')
    parser.add_argument('--enable-earlystop', action='store_true', help='啟用早停')
    
    args = parser.parse_args()
    
    # 設備設置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 開始AR優化聯合訓練...")
    print(f"📊 設備: {device}")
    print(f"📈 配置: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"⚖️ 權重: bbox_weight={args.bbox_weight}, cls_weight={args.cls_weight}")
    
    # 創建數據集
    train_dataset = AROptimizedDataset(
        data_dir='converted_dataset_fixed',
        split='train',
        transform=get_ar_optimized_transforms('train'),
        ar_oversample=True
    )
    
    val_dataset = AROptimizedDataset(
        data_dir='converted_dataset_fixed',
        split='val',
        transform=get_ar_optimized_transforms('val'),
        ar_oversample=False
    )
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 創建模型
    model = AROptimizedJointModel(num_classes=4)
    
    # 訓練模型
    trained_model = train_model_ar_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        bbox_weight=args.bbox_weight,
        cls_weight=args.cls_weight,
        enable_earlystop=args.enable_earlystop
    )
    
    print("🎉 AR優化聯合訓練完成！")

if __name__ == "__main__":
    main() 