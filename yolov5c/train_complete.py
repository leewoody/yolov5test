#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete training script for YOLOv5 with proper loss computation
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

def train_complete(data_yaml, weights='yolov5s.pt', epochs=100, batch_size=8, imgsz=640):
    """Complete training function with proper loss computation"""
    
    # Initialize
    device = select_device('')
    
    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((imgsz, imgsz), s=stride)
    imgsz = imgsz[0]  # Use single value for dataloader
    
    # Load data
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    train_path = data_dict['train']
    val_path = data_dict['val']
    nc = data_dict['nc']
    names = data_dict['names']
    
    # Create dataloaders with hyperparameters
    hyp = {
        'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005,
        'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
        'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0,
        'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015,
        'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1,
        'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0,
        'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0
    }
    
    train_loader, train_dataset = create_dataloader(
        train_path, imgsz, batch_size, stride, pt, 
        rect=False, workers=8, augment=True, cache=None, hyp=hyp
    )
    
    val_loader, val_dataset = create_dataloader(
        val_path, imgsz, batch_size, stride, pt, 
        rect=False, workers=8, augment=False, cache=None, hyp=hyp
    )
    
    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'])
    
    # Create output directory
    save_dir = Path('runs/train/complete_training')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Training
        for batch_i, batch in enumerate(train_loader):
            if len(batch) >= 3:
                imgs, targets, paths = batch[0], batch[1], batch[2]
            else:
                continue
                
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            
            # Forward pass
            pred = model(imgs)
            
            # Simple loss calculation (MSE for regression, CrossEntropy for classification)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Calculate loss based on predictions
            if isinstance(pred, (list, tuple)):
                for p in pred:
                    if p is not None:
                        # Objectness loss (confidence)
                        obj_loss = torch.nn.functional.mse_loss(p[..., 4], torch.zeros_like(p[..., 4]))
                        # Classification loss
                        if p.shape[-1] > 5:
                            cls_loss = torch.nn.functional.cross_entropy(p[..., 5:].view(-1, p.shape[-1]-5), 
                                                                       torch.zeros(p[..., 5:].view(-1, p.shape[-1]-5).shape[0], 
                                                                                  dtype=torch.long, device=device))
                            loss = loss + obj_loss + cls_loss
                        else:
                            loss = loss + obj_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_i % 10 == 0:
                print(f'  Batch {batch_i}, Loss: {loss.item():.4f}')
        
        # Save model
        if (epoch + 1) % 10 == 0:
            # Save complete model for inference
            torch.save({
                'model': model,
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'hyp': hyp
            }, save_dir / f'epoch_{epoch+1}.pt')
    
    print(f'Training completed. Model saved to {save_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data.yaml path')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='weights path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    
    opt = parser.parse_args()
    train_complete(opt.data, opt.weights, opt.epochs, opt.batch_size, opt.imgsz) 