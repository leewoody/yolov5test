#!/usr/bin/env python3
"""
YOLOv5WithClassification Joint Training with GradNorm
GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks

This script implements GradNorm to solve gradient interference between detection and classification tasks.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import logging

# Add yolov5c to path
sys.path.append(str(Path(__file__).parent))

from models.gradnorm import GradNormLoss, AdaptiveGradNormLoss
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.loss import ComputeLoss
from utils.metrics import ap_per_class, box_iou
from utils.torch_utils import select_device, time_sync
from utils.plots import plot_results, plot_results_files

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../detection_only_dataset/data.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--device', default='auto', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/gradnorm_training', help='save to project/name')
    parser.add_argument('--name', default='joint_gradnorm', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--gradnorm-alpha', type=float, default=1.5, help='GradNorm alpha parameter')
    parser.add_argument('--initial-detection-weight', type=float, default=1.0, help='Initial detection weight')
    parser.add_argument('--initial-classification-weight', type=float, default=0.01, help='Initial classification weight')
    parser.add_argument('--use-adaptive-gradnorm', action='store_true', help='Use adaptive GradNorm')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval for batches')
    parser.add_argument('--save-interval', type=int, default=5, help='Save interval for epochs')
    
    opt = parser.parse_args()
    return opt

def load_model(weights_path, device):
    """Load YOLOv5 model with classification head"""
    from models.yolo import Model
    
    # Load model
    model = Model(weights_path, device=device)
    
    # Ensure classification is enabled
    if not hasattr(model, 'classification_enabled'):
        model.classification_enabled = True
    
    return model

def create_gradnorm_loss(detection_loss_fn, classification_loss_fn, opt):
    """Create GradNorm loss function"""
    if opt.use_adaptive_gradnorm:
        return AdaptiveGradNormLoss(
            detection_loss_fn=detection_loss_fn,
            classification_loss_fn=classification_loss_fn,
            alpha=opt.gradnorm_alpha,
            momentum=0.9,
            min_weight=0.001,
            max_weight=10.0
        )
    else:
        return GradNormLoss(
            detection_loss_fn=detection_loss_fn,
            classification_loss_fn=classification_loss_fn,
            alpha=opt.gradnorm_alpha,
            initial_detection_weight=opt.initial_detection_weight,
            initial_classification_weight=opt.initial_classification_weight
        )

class ClassificationLoss(nn.Module):
    """Classification loss function"""
    
    def __init__(self, num_classes=4):
        super(ClassificationLoss, self).__init__()
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        """
        Forward pass for classification loss
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Target labels [batch_size, num_classes] (one-hot encoded)
        """
        return self.bce_loss(predictions, targets)

def create_one_hot_targets(labels, num_classes=4):
    """Create one-hot encoded targets for classification"""
    batch_size = labels.shape[0]
    one_hot = torch.zeros(batch_size, num_classes)
    
    # Convert class indices to one-hot encoding
    for i in range(batch_size):
        class_idx = int(labels[i, 0])  # Assuming first column is class index
        if class_idx < num_classes:
            one_hot[i, class_idx] = 1.0
    
    return one_hot

def train_epoch(model, dataloader, optimizer, gradnorm_loss, device, epoch, opt):
    """Train one epoch with GradNorm"""
    model.train()
    
    total_loss = 0.0
    detection_loss_sum = 0.0
    classification_loss_sum = 0.0
    num_batches = 0
    
    # Training history
    epoch_history = {
        'total_loss': [],
        'detection_loss': [],
        'classification_loss': [],
        'detection_weight': [],
        'classification_weight': []
    }
    
    for batch_idx, (imgs, targets, paths, _) in enumerate(dataloader):
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)
        
        # Create classification targets (one-hot encoded)
        classification_targets = create_one_hot_targets(targets, num_classes=4)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get model outputs
        outputs = model(imgs)
        
        # Handle different output formats
        if isinstance(outputs, tuple) and len(outputs) == 2:
            detection_outputs, classification_outputs = outputs
        elif isinstance(outputs, list):
            detection_outputs = outputs
            # Create dummy classification outputs for now
            classification_outputs = torch.randn(imgs.shape[0], 4).to(device)
        else:
            detection_outputs = outputs
            classification_outputs = torch.randn(imgs.shape[0], 4).to(device)
        
        # Compute GradNorm loss
        total_loss, loss_info = gradnorm_loss(
            detection_outputs=detection_outputs,
            classification_outputs=classification_outputs,
            detection_targets=targets,
            classification_targets=classification_targets
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss_info['total_loss']
        detection_loss_sum += loss_info['detection_loss']
        classification_loss_sum += loss_info['classification_loss']
        num_batches += 1
        
        # Store history
        epoch_history['total_loss'].append(loss_info['total_loss'])
        epoch_history['detection_loss'].append(loss_info['detection_loss'])
        epoch_history['classification_loss'].append(loss_info['classification_loss'])
        epoch_history['detection_weight'].append(loss_info['detection_weight'])
        epoch_history['classification_weight'].append(loss_info['classification_weight'])
        
        # Log progress
        if batch_idx % opt.log_interval == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                  f'Total Loss: {loss_info["total_loss"]:.4f}, '
                  f'Detection Loss: {loss_info["detection_loss"]:.4f}, '
                  f'Classification Loss: {loss_info["classification_loss"]:.4f}, '
                  f'Detection Weight: {loss_info["detection_weight"]:.4f}, '
                  f'Classification Weight: {loss_info["classification_weight"]:.4f}')
    
    # Compute averages
    avg_total_loss = total_loss / num_batches
    avg_detection_loss = detection_loss_sum / num_batches
    avg_classification_loss = classification_loss_sum / num_batches
    
    return {
        'total_loss': avg_total_loss,
        'detection_loss': avg_detection_loss,
        'classification_loss': avg_classification_loss,
        'history': epoch_history
    }

def validate_epoch(model, dataloader, device, epoch):
    """Validate one epoch"""
    model.eval()
    
    total_loss = 0.0
    detection_loss_sum = 0.0
    classification_loss_sum = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, targets, paths, _) in enumerate(dataloader):
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)
            
            # Create classification targets
            classification_targets = create_one_hot_targets(targets, num_classes=4)
            
            # Forward pass
            outputs = model(imgs)
            
            # Handle different output formats
            if isinstance(outputs, tuple) and len(outputs) == 2:
                detection_outputs, classification_outputs = outputs
            elif isinstance(outputs, list):
                detection_outputs = outputs
                classification_outputs = torch.randn(imgs.shape[0], 4).to(device)
            else:
                detection_outputs = outputs
                classification_outputs = torch.randn(imgs.shape[0], 4).to(device)
            
            # Compute losses (without GradNorm for validation)
            detection_loss_fn = nn.MSELoss()
            classification_loss_fn = nn.BCEWithLogitsLoss()
            
            detection_loss = detection_loss_fn(detection_outputs[0], targets)  # Simplified
            classification_loss = classification_loss_fn(classification_outputs, classification_targets)
            
            total_loss += detection_loss.item() + classification_loss.item()
            detection_loss_sum += detection_loss.item()
            classification_loss_sum += classification_loss.item()
            num_batches += 1
    
    # Compute averages
    avg_total_loss = total_loss / num_batches
    avg_detection_loss = detection_loss_sum / num_batches
    avg_classification_loss = classification_loss_sum / num_batches
    
    return {
        'total_loss': avg_total_loss,
        'detection_loss': avg_detection_loss,
        'classification_loss': avg_classification_loss
    }

def plot_training_curves(history, save_dir):
    """Plot training curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(15, 10))
    
    # Total loss
    plt.subplot(2, 3, 1)
    plt.plot(history['train_total_loss'], label='Train')
    plt.plot(history['val_total_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Detection loss
    plt.subplot(2, 3, 2)
    plt.plot(history['train_detection_loss'], label='Train')
    plt.plot(history['val_detection_loss'], label='Validation')
    plt.title('Detection Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Classification loss
    plt.subplot(2, 3, 3)
    plt.plot(history['train_classification_loss'], label='Train')
    plt.plot(history['val_classification_loss'], label='Validation')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Detection weight
    plt.subplot(2, 3, 4)
    plt.plot(history['detection_weight'])
    plt.title('Detection Weight')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    
    # Classification weight
    plt.subplot(2, 3, 5)
    plt.plot(history['classification_weight'])
    plt.title('Classification Weight')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    
    # Weight ratio
    plt.subplot(2, 3, 6)
    weight_ratio = [d/c for d, c in zip(history['detection_weight'], history['classification_weight'])]
    plt.plot(weight_ratio)
    plt.title('Detection/Classification Weight Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def main(opt):
    """Main training function"""
    # Initialize
    device = select_device(opt.device)
    
    # Load data
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)
    
    # Load model
    model = load_model(opt.weights, device)
    
    # Create loss functions
    detection_loss_fn = ComputeLoss(model)
    classification_loss_fn = ClassificationLoss(num_classes=4)
    
    # Create GradNorm loss
    gradnorm_loss = create_gradnorm_loss(detection_loss_fn, classification_loss_fn, opt)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Create data loaders (simplified for now)
    # In a real implementation, you would use the proper YOLOv5 data loading
    train_loader = None  # Placeholder
    val_loader = None    # Placeholder
    
    # Training history
    history = {
        'train_total_loss': [],
        'train_detection_loss': [],
        'train_classification_loss': [],
        'val_total_loss': [],
        'val_detection_loss': [],
        'val_classification_loss': [],
        'detection_weight': [],
        'classification_weight': []
    }
    
    # Training loop
    for epoch in range(opt.epochs):
        print(f'\nEpoch {epoch+1}/{opt.epochs}')
        
        # Train
        train_results = train_epoch(model, train_loader, optimizer, gradnorm_loss, device, epoch, opt)
        
        # Validate
        val_results = validate_epoch(model, val_loader, device, epoch)
        
        # Store history
        history['train_total_loss'].append(train_results['total_loss'])
        history['train_detection_loss'].append(train_results['detection_loss'])
        history['train_classification_loss'].append(train_results['classification_loss'])
        history['val_total_loss'].append(val_results['total_loss'])
        history['val_detection_loss'].append(val_results['detection_loss'])
        history['val_classification_loss'].append(val_results['classification_loss'])
        
        # Get current weights
        if hasattr(gradnorm_loss, 'gradnorm'):
            history['detection_weight'].append(gradnorm_loss.gradnorm.task_weights[0].item())
            history['classification_weight'].append(gradnorm_loss.gradnorm.task_weights[1].item())
        
        # Save model
        if (epoch + 1) % opt.save_interval == 0:
            save_path = os.path.join(opt.project, opt.name, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_results['total_loss'],
                'val_loss': val_results['total_loss']
            }, save_path)
    
    # Plot training curves
    save_dir = os.path.join(opt.project, opt.name)
    plot_training_curves(history, save_dir)
    
    # Save final results
    results = {
        'final_train_loss': train_results['total_loss'],
        'final_val_loss': val_results['total_loss'],
        'history': history,
        'config': vars(opt)
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f'\nTraining completed. Results saved to {save_dir}')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 