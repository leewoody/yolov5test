#!/usr/bin/env python3
"""
優化的 joint detection + classification 訓練腳本
解決分類損失壓制檢測的問題
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os

# 添加當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import check_dataset, colorstr
from utils.torch_utils import select_device
from tqdm import tqdm
import yaml

class JointTrainer:
    def __init__(self, data_yaml, weights='yolov5s.pt', epochs=100, batch_size=16, imgsz=640):
        self.data_yaml = data_yaml
        self.weights = weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        
        # 設備
        self.device = select_device('')
        
        # 載入數據配置
        self.data_dict = check_dataset(data_yaml)
        self.nc = self.data_dict['nc']
        
        # 創建模型
        self.model = self.create_model()
        
        # 創建損失函數
        self.compute_loss = ComputeLoss(self.model)
        
        # 創建數據加載器
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        # 優化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0005)
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=0.01, 
            epochs=epochs, 
            steps_per_epoch=len(self.train_loader)
        )
        
        # 動態損失權重
        self.classification_weight = 0.1  # 初始分類權重較小
        self.detection_weight = 1.0
        
    def create_model(self):
        """創建支援 joint detection + classification 的模型"""
        # 使用自定義配置
        model = Model('models/yolov5sc.yaml', ch=3, nc=self.nc)
        
        # 載入預訓練權重（過濾不相容的層）
        if os.path.exists(self.weights):
            ckpt = torch.load(self.weights, map_location=self.device, weights_only=False)
            state_dict = ckpt['model'].float().state_dict()
            
            # 過濾不相容的層
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if '24.' not in k and '25.' not in k:  # 跳過分類頭
                    filtered_state_dict[k] = v
            
            # 載入權重
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            print(f"載入權重: 缺失 {len(missing_keys)} 個鍵，意外 {len(unexpected_keys)} 個鍵")
        
        model = model.to(self.device)
        model.hyp = {
            'box': 0.05,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj': 1.0,
            'obj_pw': 1.0,
            'iou_t': 0.20,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4
        }
        
        return model
    
    def create_dataloaders(self):
        """創建數據加載器"""
        from utils.dataloaders import create_dataloader
        
        # 訓練數據加載器
        train_path = self.data_dict['train']
        train_loader, dataset = create_dataloader(
            train_path,
            imgsz=self.imgsz,
            batch_size=self.batch_size,
            stride=32,
            augment=True,
            hyp=self.model.hyp,
            rect=False,
            rank=-1,
            workers=8,
            image_weights=False,
            quad=False,
            prefix=colorstr('train: ')
        )
        
        # 驗證數據加載器
        val_path = self.data_dict['val']
        val_loader = create_dataloader(
            val_path,
            imgsz=self.imgsz,
            batch_size=self.batch_size * 2,
            stride=32,
            augment=False,
            hyp=self.model.hyp,
            rect=True,
            rank=-1,
            workers=8,
            prefix=colorstr('val: ')
        )[0]
        
        return train_loader, val_loader
    
    def extract_classification_labels(self, targets, paths):
        """從標籤文件中提取分類標籤"""
        cls_targets = []
        
        for i, path in enumerate(paths):
            # 獲取對應的標籤文件路徑
            label_path = str(path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # 查找分類標籤（最後一行，只有一個數字）
                classification_label = 0  # 默認值
                for line in lines:
                    line = line.strip()
                    if line and len(line.split()) == 1:
                        # 單個數字，應該是分類標籤
                        classification_label = int(line)
                        break
                
                cls_targets.append(classification_label)
                
            except Exception as e:
                print(f"警告: 無法讀取分類標籤 {label_path}: {e}")
                cls_targets.append(0)  # 默認值
        
        return torch.tensor(cls_targets, dtype=torch.long, device=self.device)
    
    def train_epoch(self, epoch):
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0
        det_loss_sum = 0
        cls_loss_sum = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for i, (imgs, targets, paths, _) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True).float() / 255
            targets = targets.to(self.device)
            
            # 提取分類標籤
            cls_targets = self.extract_classification_labels(targets, paths)
            
            # 前向傳播
            output = self.model(imgs)
            
            # 處理 joint 輸出
            if isinstance(output, tuple):
                det_pred, cls_pred = output
            else:
                det_pred = output
                cls_pred = None
            
            # 計算檢測損失
            det_loss, det_loss_items = self.compute_loss(det_pred, targets)
            
            # 計算分類損失
            if cls_pred is not None:
                cls_loss = F.cross_entropy(cls_pred, cls_targets)
            else:
                cls_loss = torch.tensor(0.0, device=self.device)
            
            # 總損失（動態權重）
            total_batch_loss = self.detection_weight * det_loss + self.classification_weight * cls_loss
            
            # 反向傳播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # 統計
            total_loss += total_batch_loss.item()
            det_loss_sum += det_loss.item()
            cls_loss_sum += cls_loss.item()
            
            # 更新進度條
            pbar.set_postfix({
                'Total': f'{total_batch_loss.item():.4f}',
                'Det': f'{det_loss.item():.4f}',
                'Cls': f'{cls_loss.item():.4f}',
                'ClsW': f'{self.classification_weight:.3f}'
            })
            
            # 動態調整分類權重
            if i % 100 == 0 and i > 0:
                det_avg = det_loss_sum / (i + 1)
                cls_avg = cls_loss_sum / (i + 1)
                
                # 如果檢測損失過高，降低分類權重
                if det_avg > 2.0:
                    self.classification_weight = max(0.01, self.classification_weight * 0.9)
                # 如果分類損失過高，稍微增加分類權重
                elif cls_avg > 1.5:
                    self.classification_weight = min(0.3, self.classification_weight * 1.1)
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch):
        """驗證"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for imgs, targets, paths, _ in tqdm(self.val_loader, desc='Validating'):
                imgs = imgs.to(self.device, non_blocking=True).float() / 255
                targets = targets.to(self.device)
                
                # 提取分類標籤
                cls_targets = self.extract_classification_labels(targets, paths)
                
                output = self.model(imgs)
                
                # 處理 joint 輸出
                if isinstance(output, tuple):
                    det_pred, cls_pred = output
                else:
                    det_pred = output
                    cls_pred = None
                
                det_loss, _ = self.compute_loss(det_pred, targets)
                
                if cls_pred is not None:
                    cls_loss = F.cross_entropy(cls_pred, cls_targets)
                else:
                    cls_loss = torch.tensor(0.0, device=self.device)
                
                total_loss += (self.detection_weight * det_loss + self.classification_weight * cls_loss).item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """完整訓練流程"""
        print(f"開始訓練 joint detection + classification 模型")
        print(f"數據集: {self.data_yaml}")
        print(f"類別數: {self.nc}")
        print(f"設備: {self.device}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # 訓練
            train_loss = self.train_epoch(epoch)
            
            # 驗證
            val_loss = self.validate(epoch)
            
            print(f"Epoch {epoch+1}/{self.epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'classification_weight': self.classification_weight
                }, 'best_joint_model.pt')
                print(f"保存最佳模型 (val_loss: {val_loss:.4f})")
        
        print("訓練完成！")

if __name__ == "__main__":
    data_yaml = "../regurgitation_joint_20250730/data.yaml"
    trainer = JointTrainer(data_yaml)
    trainer.train()
