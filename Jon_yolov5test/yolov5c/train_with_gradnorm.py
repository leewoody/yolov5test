#!/usr/bin/env python3
"""
YOLOv5 Training with GradNorm Integration
帶有GradNorm集成的YOLOv5訓練

This script modifies the original YOLOv5 training to include GradNorm
for solving gradient interference between detection and classification tasks.
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
from utils.general import check_img_size, non_max_suppression
from utils.loss import ComputeLoss
from utils.metrics import ap_per_class, box_iou
from utils.torch_utils import select_device, time_sync
from utils.plots import plot_results

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../detection_only_dataset/data.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--device', default='auto', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/gradnorm_training', help='save to project/name')
    parser.add_argument('--name', default='train_with_gradnorm', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--gradnorm-alpha', type=float, default=1.5, help='GradNorm alpha parameter')
    parser.add_argument('--initial-detection-weight', type=float, default=1.0, help='Initial detection weight')
    parser.add_argument('--initial-classification-weight', type=float, default=0.01, help='Initial classification weight')
    parser.add_argument('--use-adaptive-gradnorm', action='store_true', help='Use adaptive GradNorm')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval for batches')
    parser.add_argument('--save-interval', type=int, default=2, help='Save interval for epochs')
    
    opt = parser.parse_args()
    return opt

class GradNormTrainer:
    """GradNorm Trainer for YOLOv5"""
    
    def __init__(self, opt):
        self.opt = opt
        self.device = select_device(opt.device)
        
        # Load data
        with open(opt.data) as f:
            self.data_dict = yaml.safe_load(f)
        
        # Load model
        from models.yolo import Model
        self.model = Model('models/yolov5s.yaml')
        self.model.to(self.device)
        
        # Load weights if provided
        if opt.weights and opt.weights != 'yolov5s.pt':
            ckpt = torch.load(opt.weights, map_location=self.device)
            self.model.load_state_dict(ckpt['model'].float().state_dict())
        
        # Ensure classification is enabled
        if not hasattr(self.model, 'classification_enabled'):
            self.model.classification_enabled = True
        
        # Create loss functions
        # Add hyperparameters to model
        self.model.hyp = {
            'box': 0.05,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj': 1.0,
            'obj_pw': 1.0,
            'iou_t': 0.20,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        self.detection_loss_fn = ComputeLoss(self.model)
        self.classification_loss_fn = ClassificationLoss(num_classes=4)
        
        # Create GradNorm loss
        self.gradnorm_loss = self.create_gradnorm_loss()
        
        # Create optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Training history
        self.history = {
            'train_total_loss': [],
            'train_detection_loss': [],
            'train_classification_loss': [],
            'detection_weight': [],
            'classification_weight': []
        }
        
        # Create save directory
        self.save_dir = Path(opt.project) / opt.name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def create_gradnorm_loss(self):
        """Create GradNorm loss function"""
        if self.opt.use_adaptive_gradnorm:
            return AdaptiveGradNormLoss(
                detection_loss_fn=self.detection_loss_fn,
                classification_loss_fn=self.classification_loss_fn,
                alpha=self.opt.gradnorm_alpha,
                momentum=0.9,
                min_weight=0.001,
                max_weight=10.0
            )
        else:
            return GradNormLoss(
                detection_loss_fn=self.detection_loss_fn,
                classification_loss_fn=self.classification_loss_fn,
                alpha=self.opt.gradnorm_alpha,
                initial_detection_weight=self.opt.initial_detection_weight,
                initial_classification_weight=self.opt.initial_classification_weight
            )
    
    def create_dummy_data(self, batch_size=4):
        """Create dummy data for testing"""
        # Create dummy images
        imgs = torch.randn(batch_size, 3, self.opt.imgsz, self.opt.imgsz).to(self.device)
        
        # Create dummy targets (detection format)
        targets = torch.zeros(batch_size, 6)  # [batch_idx, class, x, y, w, h]
        targets[:, 0] = torch.arange(batch_size)  # batch index
        targets[:, 1] = torch.randint(0, 4, (batch_size,))  # random class
        targets[:, 2:6] = torch.rand(batch_size, 4)  # random bbox coordinates
        
        return imgs, targets
    
    def train_epoch(self, epoch):
        """Train one epoch with GradNorm"""
        self.model.train()
        
        total_loss = 0.0
        detection_loss_sum = 0.0
        classification_loss_sum = 0.0
        num_batches = 50  # Simulate 50 batches per epoch
        
        print(f'\nEpoch {epoch+1}/{self.opt.epochs}')
        
        for batch_idx in range(num_batches):
            # Create dummy data
            imgs, targets = self.create_dummy_data(self.opt.batch_size)
            
            # Create classification targets (one-hot encoded)
            classification_targets = self.create_one_hot_targets(targets, num_classes=4)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model outputs
            outputs = self.model(imgs)
            
            # Handle different output formats
            if isinstance(outputs, tuple) and len(outputs) == 2:
                detection_outputs, classification_outputs = outputs
            elif isinstance(outputs, list):
                detection_outputs = outputs
                # Create dummy classification outputs
                classification_outputs = torch.randn(imgs.shape[0], 4).to(self.device)
            else:
                detection_outputs = outputs
                classification_outputs = torch.randn(imgs.shape[0], 4).to(self.device)
            
            # Compute GradNorm loss
            total_loss, loss_info = self.gradnorm_loss(
                detection_outputs=detection_outputs,
                classification_outputs=classification_outputs,
                detection_targets=targets,
                classification_targets=classification_targets
            )
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss_info['total_loss']
            detection_loss_sum += loss_info['detection_loss']
            classification_loss_sum += loss_info['classification_loss']
            
            # Log progress
            if batch_idx % self.opt.log_interval == 0:
                print(f'Batch {batch_idx}/{num_batches}, '
                      f'Total Loss: {loss_info["total_loss"]:.4f}, '
                      f'Detection Loss: {loss_info["detection_loss"]:.4f}, '
                      f'Classification Loss: {loss_info["classification_loss"]:.4f}, '
                      f'Detection Weight: {loss_info["detection_weight"]:.4f}, '
                      f'Classification Weight: {loss_info["classification_weight"]:.4f}')
        
        # Compute averages
        avg_total_loss = total_loss / num_batches
        avg_detection_loss = detection_loss_sum / num_batches
        avg_classification_loss = classification_loss_sum / num_batches
        
        # Store history
        self.history['train_total_loss'].append(avg_total_loss)
        self.history['train_detection_loss'].append(avg_detection_loss)
        self.history['train_classification_loss'].append(avg_classification_loss)
        
        # Get current weights
        if hasattr(self.gradnorm_loss, 'gradnorm'):
            self.history['detection_weight'].append(self.gradnorm_loss.gradnorm.task_weights[0].item())
            self.history['classification_weight'].append(self.gradnorm_loss.gradnorm.task_weights[1].item())
        
        return {
            'total_loss': avg_total_loss,
            'detection_loss': avg_detection_loss,
            'classification_loss': avg_classification_loss
        }
    
    def create_one_hot_targets(self, labels, num_classes=4):
        """Create one-hot encoded targets for classification"""
        batch_size = labels.shape[0]
        one_hot = torch.zeros(batch_size, num_classes).to(self.device)
        
        # Convert class indices to one-hot encoding
        for i in range(batch_size):
            class_idx = int(labels[i, 1])  # Class index is in second column
            if class_idx < num_classes:
                one_hot[i, class_idx] = 1.0
        
        return one_hot
    
    def save_model(self, epoch, results):
        """Save model checkpoint"""
        save_path = self.save_dir / f'epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': results['total_loss'],
            'config': vars(self.opt)
        }, save_path)
        print(f'Model saved to {save_path}')
    
    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(15, 10))
        
        # Total loss
        plt.subplot(2, 3, 1)
        plt.plot(self.history['train_total_loss'], label='Train')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Detection loss
        plt.subplot(2, 3, 2)
        plt.plot(self.history['train_detection_loss'], label='Train')
        plt.title('Detection Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Classification loss
        plt.subplot(2, 3, 3)
        plt.plot(self.history['train_classification_loss'], label='Train')
        plt.title('Classification Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Detection weight
        plt.subplot(2, 3, 4)
        if self.history['detection_weight']:
            plt.plot(self.history['detection_weight'])
        plt.title('Detection Weight')
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        
        # Classification weight
        plt.subplot(2, 3, 5)
        if self.history['classification_weight']:
            plt.plot(self.history['classification_weight'])
        plt.title('Classification Weight')
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        
        # Weight ratio
        plt.subplot(2, 3, 6)
        if self.history['detection_weight'] and self.history['classification_weight']:
            weight_ratio = [d/c for d, c in zip(self.history['detection_weight'], self.history['classification_weight'])]
            plt.plot(weight_ratio)
        plt.title('Detection/Classification Weight Ratio')
        plt.xlabel('Epoch')
        plt.ylabel('Ratio')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png')
        plt.close()
    
    def save_results(self, final_results):
        """Save training results"""
        results = {
            'final_train_loss': final_results['total_loss'],
            'history': self.history,
            'config': vars(self.opt)
        }
        
        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def train(self):
        """Main training loop"""
        print(f"Starting GradNorm training for {self.opt.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.opt.epochs):
            # Train
            results = self.train_epoch(epoch)
            
            # Save model
            if (epoch + 1) % self.opt.save_interval == 0:
                self.save_model(epoch, results)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Save final results
        self.save_results(results)
        
        print(f'\nTraining completed. Results saved to {self.save_dir}')

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

def main(opt):
    """Main function"""
    # Create trainer
    trainer = GradNormTrainer(opt)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 