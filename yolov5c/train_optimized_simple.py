#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified optimized YOLOv5 training script for Regurgitation dataset
Focus on improving confidence with basic functionality
"""

import argparse
import os
import sys
import time
import yaml
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.general import check_file, check_yaml, colorstr, init_seeds, check_dataset
from utils.loss import ComputeLoss
from utils.torch_utils import select_device


class SimpleOptimizedTrainer:
    def __init__(self, opt, hyp, device):
        self.opt = opt
        self.hyp = hyp
        self.device = device
        self.best_loss = float('inf')
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'confidence': []
        }
        
    def create_model(self, nc):
        """Create optimized model"""
        # Use YOLOv5s for compatibility
        model = Model('models/yolov5s.yaml', ch=3, nc=nc).to(self.device)
        
        # Set hyperparameters for loss computation
        model.hyp = self.hyp
        
        # Load pretrained weights with careful filtering
        if self.opt.weights.endswith('.pt'):
            ckpt = torch.load(self.opt.weights, map_location='cpu', weights_only=False)
            csd = ckpt['model'].float().state_dict()
            
            # Filter out incompatible layers
            model_dict = model.state_dict()
            filtered_csd = {}
            for k, v in csd.items():
                if k in model_dict:
                    if '24.' in k:  # Detection heads
                        if v.shape[0] == nc:  # Class count matches
                            filtered_csd[k] = v
                    else:
                        filtered_csd[k] = v
            
            model.load_state_dict(filtered_csd, strict=False)
            print(f'Loaded {len(filtered_csd)}/{len(model.state_dict())} items from {self.opt.weights}')
        
        return model
    
    def create_optimizer(self, model):
        """Create optimized optimizer and scheduler"""
        # Use AdamW for better convergence
        optimizer = AdamW(model.parameters(), lr=self.hyp['lr0'], weight_decay=0.01)
        
        # Use OneCycleLR for better learning rate scheduling
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=self.hyp['lr0'],
            epochs=self.opt.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return optimizer, scheduler
    
    def create_dataloaders(self, data_dict):
        """Create optimized dataloaders"""
        train_path, val_path = data_dict['train'], data_dict['val']
        
        # Enhanced data augmentation for better generalization
        hyp_enhanced = self.hyp.copy()
        hyp_enhanced.update({
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation augmentation
            'hsv_v': 0.4,    # HSV-Value augmentation
            'degrees': 10.0,  # Image rotation
            'translate': 0.1, # Image translation
            'scale': 0.5,    # Image scaling
            'shear': 2.0,    # Image shear
            'perspective': 0.0, # Perspective transform
            'flipud': 0.0,   # Vertical flip
            'mosaic': 1.0,   # Mosaic augmentation
            'mixup': 0.0,    # Mixup augmentation
        })
        
        self.train_loader, self.dataset = create_dataloader(
            train_path, 
            imgsz=self.opt.imgsz, 
            batch_size=self.opt.batch_size, 
            stride=32,
            augment=True, 
            hyp=hyp_enhanced, 
            rect=False, 
            rank=-1, 
            workers=self.opt.workers, 
            image_weights=False, 
            quad=False, 
            prefix=colorstr('train: ')
        )
        
        self.val_loader = create_dataloader(
            val_path, 
            imgsz=self.opt.imgsz, 
            batch_size=self.opt.batch_size * 2, 
            stride=32,
            augment=False, 
            hyp=self.hyp, 
            rect=True, 
            rank=-1, 
            workers=self.opt.workers, 
            prefix=colorstr('val: ')
        )[0]
        
        return self.train_loader, self.val_loader
    
    def validate(self, model, epoch):
        """Simple validation"""
        model.eval()
        val_loss = torch.zeros(3, device=self.device)
        
        pbar = tqdm(self.val_loader, desc=colorstr('val:'))
        for i, (imgs, targets, paths, _) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True).float() / 255
            
            # Debug: Check shapes in validation
            if i == 0:  # Only print for first batch
                print(f"Validation Debug - imgs.shape: {imgs.shape}")
                print(f"Validation Debug - targets.shape: {targets.shape}")
            
            with torch.no_grad():
                pred = model(imgs)
                
                # Debug: Check prediction shapes in validation
                if i == 0:  # Only print for first batch
                    print(f"Validation Debug - pred type: {type(pred)}")
                    print(f"Validation Debug - pred length: {len(pred)}")
                    if isinstance(pred, list):
                        for j, p in enumerate(pred):
                            print(f"Validation Debug - pred[{j}].shape: {p.shape}")
                    else:
                        print(f"Validation Debug - pred.shape: {pred.shape}")
                
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))
                val_loss += loss_items.cpu()
        
        val_loss /= len(self.val_loader)
        
        # Update metrics history
        self.metrics_history['val_loss'].append(val_loss.sum().item())
        
        print(f'Validation - Loss: {val_loss.sum():.3f}, Box: {val_loss[0]:.3f}, Obj: {val_loss[1]:.3f}, Cls: {val_loss[2]:.3f}')
        
        return val_loss.sum().item()
    
    def train_epoch(self, model, optimizer, scheduler, epoch):
        """Train one epoch with enhanced monitoring"""
        model.train()
        mloss = torch.zeros(3, device=self.device)
        nb = len(self.train_loader)
        
        pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(self.device, non_blocking=True).float() / 255
            
            # Debug: Check shapes
            if i == 0:  # Only print for first batch
                print(f"Debug - imgs.shape: {imgs.shape}")
                print(f"Debug - targets.shape: {targets.shape}")
                print(f"Debug - targets device: {targets.device}")
            
            # Forward pass
            pred = model(imgs)
            
            # Debug: Check prediction shapes
            if i == 0:  # Only print for first batch
                print(f"Debug - pred type: {type(pred)}")
                print(f"Debug - pred length: {len(pred)}")
                for j, p in enumerate(pred):
                    print(f"Debug - pred[{j}].shape: {p.shape}")
            
            # Ensure targets are on correct device
            targets = targets.to(self.device)
            
            # Validate shapes before loss computation
            if i == 0:  # Only print for first batch
                print(f"Debug - Before loss computation:")
                print(f"  pred[0].shape: {pred[0].shape}")
                print(f"  targets.shape: {targets.shape}")
                print(f"  targets device: {targets.device}")
            
            loss, loss_items = self.compute_loss(pred, targets)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            if ni % self.opt.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
            
            # Log
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            
            # Calculate confidence statistics (simplified)
            avg_conf = 0.0
            if len(pred) > 0 and pred[0].shape[1] > 4:
                try:
                    confidences = pred[0][:, 4].cpu().numpy()
                    if len(confidences) > 0:
                        avg_conf = np.mean(confidences)
                except:
                    avg_conf = 0.0
            
            pbar.desc = f'{colorstr("Epoch:")} {epoch}/{self.opt.epochs - 1} {mem} loss:{mloss.sum():.3f} conf:{avg_conf:.3f}'
        
        # Update metrics history
        self.metrics_history['train_loss'].append(mloss.sum().item())
        self.metrics_history['confidence'].append(avg_conf)
        
        return mloss.sum().item()
    
    def save_metrics(self, save_dir):
        """Save training metrics"""
        # Save metrics to JSON
        with open(save_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Metrics saved to {save_dir / 'metrics_history.json'}")
    
    def train(self, data_dict):
        """Main training loop"""
        nc = int(data_dict['nc'])
        names = data_dict['names']
        
        # Create model
        model = self.create_model(nc)
        
        # Create dataloaders
        self.create_dataloaders(data_dict)
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer(model)
        
        # Loss function
        self.compute_loss = ComputeLoss(model)
        
        # Training loop
        print(f'Starting optimized training for {self.opt.epochs} epochs...')
        
        for epoch in range(self.opt.epochs):
            # Train
            train_loss = self.train_epoch(model, optimizer, scheduler, epoch)
            
            # Validate every 5 epochs
            if epoch % 5 == 0 or epoch == self.opt.epochs - 1:
                val_loss = self.validate(model, epoch)
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    
                    ckpt = {
                        'epoch': epoch,
                        'best_loss': self.best_loss,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'hyp': self.hyp,
                        'names': names
                    }
                    
                    save_dir = Path(self.opt.project) / self.opt.name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(ckpt, save_dir / 'best.pt')
                    print(f'Best model saved to {save_dir / "best.pt"} (loss: {val_loss:.3f})')
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                ckpt = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'hyp': self.hyp,
                    'names': names
                }
                save_dir = Path(self.opt.project) / self.opt.name
                torch.save(ckpt, save_dir / f'epoch{epoch}.pt')
        
        # Save final model and metrics
        save_dir = Path(self.opt.project) / self.opt.name
        torch.save(model.state_dict(), save_dir / 'final.pt')
        self.save_metrics(save_dir)
        
        print(f'Training completed. Best loss: {self.best_loss:.3f}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='../Regurgitation-YOLODataset-1new/data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='hyp_optimized.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--project', default='runs/optimized_training', help='save results to project/name')
    parser.add_argument('--name', default='regurgitation_optimized', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--accumulate', type=int, default=1, help='gradient accumulation steps')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    device = select_device('')
    
    # Load hyperparameters
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    else:
        hyp = opt.hyp
    
    # Load data
    data_dict = check_dataset(opt.data)
    
    # Initialize trainer and start training
    trainer = SimpleOptimizedTrainer(opt, hyp, device)
    trainer.train(data_dict) 