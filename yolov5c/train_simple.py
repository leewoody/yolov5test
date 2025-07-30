#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple YOLOv5 training script without complex callbacks
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
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


def train_simple(hyp, opt, device):
    """Simple training function without complex callbacks"""
    
    # Initialize
    init_seeds(opt.seed)
    
    # Load hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    
    # Create save directory
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dict = None
    try:
        data_dict = check_dataset(opt.data)
    except:
        # Simple fallback
        with open(opt.data, 'r') as f:
            data_dict = yaml.safe_load(f)
    
    train_path = data_dict['train']
    val_path = data_dict['val']
    nc = int(data_dict['nc'])
    names = data_dict['names']
    
    # Create model
    if opt.cfg:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)
    else:
        # Use default YOLOv5s architecture
        model = Model('models/yolov5s.yaml', ch=3, nc=nc).to(device)
    
    # Set hyperparameters for loss computation
    model.hyp = hyp
    
    # Load pretrained weights
    if opt.weights.endswith('.pt'):
        ckpt = torch.load(opt.weights, map_location='cpu', weights_only=False)
        csd = ckpt['model'].float().state_dict()
        # Filter out incompatible layers (detection heads)
        csd = {k: v for k, v in csd.items() if k in model.state_dict() and '24.' not in k}
        model.load_state_dict(csd, strict=False)
        print(f'Loaded {len(csd)}/{len(model.state_dict())} items from {opt.weights} (excluding detection heads)')
    
    # Create dataloaders
    train_loader, dataset = create_dataloader(
        train_path, opt.imgsz, opt.batch_size, 32, False,
        hyp=hyp, augment=True, cache=None, rect=False,
        rank=-1, workers=opt.workers, image_weights=False,
        quad=False, prefix=colorstr('train: '), shuffle=True
    )
    
    val_loader = create_dataloader(
        val_path, opt.imgsz, opt.batch_size, 32, False,
        hyp=hyp, cache=None, rect=True, rank=-1,
        workers=opt.workers, pad=0.5, prefix=colorstr('val: ')
    )[0]
    
    # Optimizer
    optimizer = SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    
    # Loss function
    compute_loss = ComputeLoss(model)
    
    # Training loop
    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    best_loss = float('inf')
    accumulate = 1
    
    print(f'Starting training for {opt.epochs} epochs...')
    print(f'Image sizes {opt.imgsz} train, {opt.imgsz} val')
    print(f'Using {train_loader.num_workers} dataloader workers')
    
    for epoch in range(opt.epochs):
        model.train()
        
        # Progress bar
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb, desc=f'Epoch {epoch}/{opt.epochs - 1}')
        
        mloss = torch.zeros(3, device=device)  # mean losses
        
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches
            
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, 64 / opt.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    initial_lr = x.get('initial_lr', hyp.get('lr0', 0.01))
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, initial_lr])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            
            # Forward pass
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_postfix({
                'loss': f'{mloss[0]:.4f}',
                'box': f'{mloss[1]:.4f}',
                'obj': f'{mloss[2]:.4f}',
                'mem': mem
            })
        
        # Save model
        if (epoch + 1) % opt.save_period == 0 or epoch == opt.epochs - 1:
            ckpt = {
                'epoch': epoch,
                'model': model.half(),
                'optimizer': optimizer.state_dict(),
                'hyp': hyp,
                'opt': vars(opt),
            }
            torch.save(ckpt, save_dir / f'epoch{epoch}.pt')
        
        # Save best model
        if epoch == 0 or mloss[0] < best_loss:
            best_loss = mloss[0]
            torch.save(ckpt, save_dir / 'best.pt')
    
    print(f'Training completed. Results saved to {save_dir}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='hyp_improved.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='train, val image size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='simple_training', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every x epochs')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    
    return parser.parse_args()


def main(opt):
    # Checks
    opt.data, opt.cfg, opt.hyp, opt.weights = check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights)
    
    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    # Train
    train_simple(opt.hyp, opt, device)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 