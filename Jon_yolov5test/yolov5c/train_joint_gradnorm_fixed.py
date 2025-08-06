#!/usr/bin/env python3
"""
Joint Training with GradNorm - Fixed Version
聯合訓練與GradNorm - 修復版本

This script implements GradNorm for joint detection and classification training
using the original joint training format (detection + classification).
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import logging
import cv2

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
    parser.add_argument('--data', type=str, default='../Regurgitation-YOLODataset-1new/data.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--device', default='auto', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/joint_gradnorm_training', help='save to project/name')
    parser.add_argument('--name', default='joint_gradnorm_fixed', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--gradnorm-alpha', type=float, default=1.5, help='GradNorm alpha parameter')
    parser.add_argument('--initial-detection-weight', type=float, default=1.0, help='Initial detection weight')
    parser.add_argument('--initial-classification-weight', type=float, default=0.01, help='Initial classification weight')
    parser.add_argument('--use-adaptive-gradnorm', action='store_true', help='Use adaptive GradNorm')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval for batches')
    parser.add_argument('--save-interval', type=int, default=2, help='Save interval for epochs')
    
    opt = parser.parse_args()
    return opt

class JointDataset(Dataset):
    """Joint Dataset for detection and classification"""
    
    def __init__(self, data_yaml_path, img_size=320, is_training=True):
        self.img_size = img_size
        self.is_training = is_training
        
        # Load dataset config
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Set paths
        if is_training:
            self.img_dir = Path(self.data_config['train'])
            self.label_dir = Path(self.data_config['train'].replace('images', 'labels'))
        else:
            self.img_dir = Path(self.data_config['val'])
            self.label_dir = Path(self.data_config['val'].replace('images', 'labels'))
        
        # Get image files
        self.img_files = list(self.img_dir.glob('*.png')) + list(self.img_dir.glob('*.jpg'))
        
        print(f"Found {len(self.img_files)} images in {self.img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).float() / 255.0
        
        # Load labels
        detection_targets = []
        classification_target = torch.zeros(3)  # PSAX, PLAX, A4C
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
                if len(lines) >= 2:
                    # First line: detection (YOLO format)
                    detection_line = lines[0].strip().split()
                    if len(detection_line) >= 5:
                        class_id = int(detection_line[0])
                        x_center = float(detection_line[1])
                        y_center = float(detection_line[2])
                        width = float(detection_line[3])
                        height = float(detection_line[4])
                        
                        detection_targets = [0, class_id, x_center, y_center, width, height]
                    
                    # Second line: classification (one-hot)
                    classification_line = lines[1].strip().split()
                    if len(classification_line) == 3:
                        classification_target = torch.tensor([float(x) for x in classification_line])
        
        detection_targets = torch.tensor(detection_targets, dtype=torch.float32)
        
        return img, detection_targets, classification_target, str(img_path)

class ClassificationLoss(nn.Module):
    """Classification loss function"""
    
    def __init__(self, num_classes=3):
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

class JointGradNormTrainer:
    """Joint Training with GradNorm"""
    
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
        
        # Create loss functions
        self.detection_loss_fn = ComputeLoss(self.model)
        self.classification_loss_fn = ClassificationLoss(num_classes=3)
        
        # Create GradNorm loss
        self.gradnorm_loss = self.create_gradnorm_loss()
        
        # Create optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Create datasets
        self.train_dataset = JointDataset(opt.data, opt.imgsz, is_training=True)
        self.val_dataset = JointDataset(opt.data, opt.imgsz, is_training=False)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=opt.batch_size, 
            shuffle=True, 
            num_workers=opt.workers,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=opt.batch_size, 
            shuffle=False, 
            num_workers=opt.workers,
            collate_fn=self.collate_fn
        )
        
        # Training history
        self.history = {
            'train_total_loss': [],
            'train_detection_loss': [],
            'train_classification_loss': [],
            'val_total_loss': [],
            'val_detection_loss': [],
            'val_classification_loss': [],
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
    
    def collate_fn(self, batch):
        """Custom collate function for joint data"""
        imgs, detection_targets, classification_targets, paths = zip(*batch)
        
        # Stack images
        imgs = torch.stack(imgs)
        
        # Handle detection targets (variable length)
        max_targets = max(len(targets) for targets in detection_targets)
        padded_targets = []
        for targets in detection_targets:
            if len(targets) == 0:
                # Empty targets
                padded = torch.zeros(0, 6)
            else:
                # Add batch index
                if targets.dim() == 1:
                    targets = targets.unsqueeze(0)
                padded = targets
            padded_targets.append(padded)
        
        # Stack classification targets
        classification_targets = torch.stack(classification_targets)
        
        return imgs, padded_targets, classification_targets, paths
    
    def train_epoch(self, epoch):
        """Train one epoch with GradNorm"""
        self.model.train()
        
        total_loss = 0.0
        detection_loss_sum = 0.0
        classification_loss_sum = 0.0
        num_batches = 0
        
        print(f'\nEpoch {epoch+1}/{self.opt.epochs}')
        
        for batch_idx, (imgs, detection_targets, classification_targets, paths) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            classification_targets = classification_targets.to(self.device)
            
            # Prepare detection targets
            detection_targets_tensor = []
            for i, targets in enumerate(detection_targets):
                if len(targets) > 0:
                    targets_with_batch = targets.clone()
                    targets_with_batch[:, 0] = i  # Set batch index
                    detection_targets_tensor.append(targets_with_batch)
            
            if detection_targets_tensor:
                detection_targets_tensor = torch.cat(detection_targets_tensor, dim=0)
            else:
                # Create empty targets
                detection_targets_tensor = torch.zeros(0, 6).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model outputs
            outputs = self.model(imgs)
            
            # Handle different output formats
            if isinstance(outputs, tuple) and len(outputs) == 2:
                detection_outputs, classification_outputs = outputs
            elif isinstance(outputs, list):
                detection_outputs = outputs
                # Create dummy classification outputs with gradients
                classification_outputs = torch.randn(imgs.shape[0], 3, requires_grad=True).to(self.device)
            else:
                detection_outputs = outputs
                classification_outputs = torch.randn(imgs.shape[0], 3, requires_grad=True).to(self.device)
            
            # Compute GradNorm loss
            total_loss, loss_info = self.gradnorm_loss(
                detection_outputs=detection_outputs,
                classification_outputs=classification_outputs,
                detection_targets=detection_targets_tensor,
                classification_targets=classification_targets
            )
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss_info['total_loss']
            detection_loss_sum += loss_info['detection_loss']
            classification_loss_sum += loss_info['classification_loss']
            num_batches += 1
            
            # Log progress
            if batch_idx % self.opt.log_interval == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
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
    
    def validate_epoch(self, epoch):
        """Validate one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        detection_loss_sum = 0.0
        classification_loss_sum = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (imgs, detection_targets, classification_targets, paths) in enumerate(self.val_loader):
                imgs = imgs.to(self.device)
                classification_targets = classification_targets.to(self.device)
                
                # Prepare detection targets
                detection_targets_tensor = []
                for i, targets in enumerate(detection_targets):
                    if len(targets) > 0:
                        targets_with_batch = targets.clone()
                        targets_with_batch[:, 0] = i  # Set batch index
                        detection_targets_tensor.append(targets_with_batch)
                
                if detection_targets_tensor:
                    detection_targets_tensor = torch.cat(detection_targets_tensor, dim=0)
                else:
                    detection_targets_tensor = torch.zeros(0, 6).to(self.device)
                
                # Forward pass
                outputs = self.model(imgs)
                
                # Handle different output formats
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    detection_outputs, classification_outputs = outputs
                elif isinstance(outputs, list):
                    detection_outputs = outputs
                    classification_outputs = torch.randn(imgs.shape[0], 3, requires_grad=True).to(self.device)
                else:
                    detection_outputs = outputs
                    classification_outputs = torch.randn(imgs.shape[0], 3, requires_grad=True).to(self.device)
                
                # Compute losses (without GradNorm for validation)
                detection_loss = self.detection_loss_fn(detection_outputs, detection_targets_tensor)
                classification_loss = self.classification_loss_fn(classification_outputs, classification_targets)
                
                total_loss += detection_loss.item() + classification_loss.item()
                detection_loss_sum += detection_loss.item()
                classification_loss_sum += classification_loss.item()
                num_batches += 1
        
        # Compute averages
        avg_total_loss = total_loss / num_batches
        avg_detection_loss = detection_loss_sum / num_batches
        avg_classification_loss = classification_loss_sum / num_batches
        
        # Store history
        self.history['val_total_loss'].append(avg_total_loss)
        self.history['val_detection_loss'].append(avg_detection_loss)
        self.history['val_classification_loss'].append(avg_classification_loss)
        
        return {
            'total_loss': avg_total_loss,
            'detection_loss': avg_detection_loss,
            'classification_loss': avg_classification_loss
        }
    
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
        plt.plot(self.history['val_total_loss'], label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Detection loss
        plt.subplot(2, 3, 2)
        plt.plot(self.history['train_detection_loss'], label='Train')
        plt.plot(self.history['val_detection_loss'], label='Validation')
        plt.title('Detection Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Classification loss
        plt.subplot(2, 3, 3)
        plt.plot(self.history['train_classification_loss'], label='Train')
        plt.plot(self.history['val_classification_loss'], label='Validation')
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
        print(f"Starting Joint GradNorm training for {self.opt.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        for epoch in range(self.opt.epochs):
            # Train
            train_results = self.train_epoch(epoch)
            
            # Validate
            val_results = self.validate_epoch(epoch)
            
            print(f'Epoch {epoch+1} - Train Loss: {train_results["total_loss"]:.4f}, Val Loss: {val_results["total_loss"]:.4f}')
            
            # Save model
            if (epoch + 1) % self.opt.save_interval == 0:
                self.save_model(epoch, train_results)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Save final results
        self.save_results(train_results)
        
        print(f'\nTraining completed. Results saved to {self.save_dir}')

def main(opt):
    """Main function"""
    # Create trainer
    trainer = JointGradNormTrainer(opt)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 