#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized YOLOv5 training script for Regurgitation dataset
Focus on improving confidence and adding validation metrics
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
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.general import check_file, check_yaml, colorstr, init_seeds, check_dataset
from utils.loss import ComputeLoss
from utils.torch_utils import select_device
from utils.metrics import ap_per_class, box_iou


class OptimizedTrainer:
    def __init__(self, opt, hyp, device):
        self.opt = opt
        self.hyp = hyp
        self.device = device
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'mAP': [],
            'precision': [],
            'recall': [],
            'confidence': []
        }
        
    def create_model(self, nc, data_dict):
        """Create optimized model"""
        if self.opt.cfg:
            model = Model(self.opt.cfg, ch=3, nc=nc).to(self.device)
        else:
            # Use YOLOv5s for compatibility with pretrained weights
            model = Model('models/yolov5s.yaml', ch=3, nc=nc).to(self.device)
        
        # Set hyperparameters for loss computation
        model.hyp = self.hyp
        
        # Load pretrained weights with better filtering
        if self.opt.weights.endswith('.pt'):
            ckpt = torch.load(self.opt.weights, map_location='cpu', weights_only=False)
            csd = ckpt['model'].float().state_dict()
            
            # Filter out incompatible layers more carefully
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
        """Enhanced validation with metrics"""
        model.eval()
        val_loss = torch.zeros(3, device=self.device)
        stats = []  # For mAP calculation
        
        pbar = tqdm(self.val_loader, desc=colorstr('val:'))
        for imgs, targets, paths, _ in pbar:
            imgs = imgs.to(self.device, non_blocking=True).float() / 255
            
            with torch.no_grad():
                pred = model(imgs)
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))
                val_loss += loss_items.cpu()
                
                # Calculate metrics for mAP
                for i, pred_i in enumerate(pred):
                    pred_i = pred_i.cpu().numpy()
                    targets_i = targets[targets[:, 0] == i].cpu().numpy()
                    
                    if len(targets_i) > 0:
                        # Convert predictions to format for mAP calculation
                        for *xyxy, conf, cls in pred_i:
                            if conf > 0.001:  # Confidence threshold
                                stats.append([*xyxy, conf, cls, 0])  # 0 for TP/FP flag
                
                # Calculate confidence statistics
                if len(pred) > 0 and pred[0].shape[1] > 4:
                    confidences = pred[0][:, 4].cpu().numpy()
                    if len(confidences) > 0:
                        avg_conf = np.mean(confidences)
                        pbar.set_postfix({'avg_conf': f'{avg_conf:.3f}'})
        
        val_loss /= len(self.val_loader)
        
        # Calculate mAP and other metrics
        if len(stats) > 0:
            stats = np.array(stats)
            ap, p, r = ap_per_class(stats, self.nc, iou_thres=0.5)
            mAP = np.mean(ap)
            precision = np.mean(p)
            recall = np.mean(r)
        else:
            mAP = precision = recall = 0.0
        
        # Update metrics history
        self.metrics_history['val_loss'].append(val_loss.sum().item())
        self.metrics_history['mAP'].append(mAP)
        self.metrics_history['precision'].append(precision)
        self.metrics_history['recall'].append(recall)
        
        print(f'Validation - Loss: {val_loss.sum():.3f}, mAP: {mAP:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')
        
        return val_loss.sum().item(), mAP, precision, recall
    
    def train_epoch(self, model, optimizer, scheduler, epoch):
        """Train one epoch with enhanced monitoring"""
        model.train()
        mloss = torch.zeros(3, device=self.device)
        nb = len(self.train_loader)
        
        pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(self.device, non_blocking=True).float() / 255
            
            # Forward pass
            pred = model(imgs)
            loss, loss_items = self.compute_loss(pred, targets.to(self.device))
            
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
            
            # Calculate confidence statistics
            if len(pred) > 0 and pred[0].shape[1] > 4:
                confidences = pred[0][:, 4].cpu().numpy()
                avg_conf = np.mean(confidences) if len(confidences) > 0 else 0.0
            else:
                avg_conf = 0.0
            
            pbar.desc = f'{colorstr("Epoch:")} {epoch}/{self.opt.epochs - 1} {mem} loss:{mloss.sum():.3f} conf:{avg_conf:.3f}'
        
        # Update metrics history
        self.metrics_history['train_loss'].append(mloss.sum().item())
        self.metrics_history['confidence'].append(avg_conf)
        
        return mloss.sum().item()
    
    def save_metrics_plot(self, save_dir):
        """Save training metrics plots"""
        epochs = range(len(self.metrics_history['train_loss']))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mAP
        axes[0, 1].plot(epochs, self.metrics_history['mAP'], label='mAP')
        axes[0, 1].set_title('Mean Average Precision (mAP)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision and Recall
        axes[1, 0].plot(epochs, self.metrics_history['precision'], label='Precision')
        axes[1, 0].plot(epochs, self.metrics_history['recall'], label='Recall')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Confidence
        axes[1, 1].plot(epochs, self.metrics_history['confidence'], label='Avg Confidence')
        axes[1, 1].set_title('Average Detection Confidence')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        with open(save_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def train(self, data_dict):
        """Main training loop"""
        nc = int(data_dict['nc'])
        names = data_dict['names']
        
        # Create model
        model = self.create_model(nc, data_dict)
        
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
                val_loss, mAP, precision, recall = self.validate(model, epoch)
                
                # Save best model
                fitness = mAP  # Use mAP as fitness metric
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_epoch = epoch
                    
                    ckpt = {
                        'epoch': epoch,
                        'best_fitness': self.best_fitness,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'hyp': self.hyp,
                        'names': names
                    }
                    
                    save_dir = Path(self.opt.project) / self.opt.name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(ckpt, save_dir / 'best.pt')
                    print(f'Best model saved to {save_dir / "best.pt"} (mAP: {mAP:.3f})')
            
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
        self.save_metrics_plot(save_dir)
        
        print(f'Training completed. Best mAP: {self.best_fitness:.3f} at epoch {self.best_epoch}')


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
    trainer = OptimizedTrainer(opt, hyp, device)
    trainer.train(data_dict) 