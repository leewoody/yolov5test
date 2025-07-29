#!/usr/bin/env python3
"""
Medical Image Complete Training Script
Combines YOLOv5 detection with ResNet classification for heart ultrasound analysis
"""

import argparse
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.medical_classifier import create_medical_classifier, get_medical_classifier_loss
from utils.medical_dataloader import create_medical_dataloader
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import LOGGER, colorstr, increment_path, print_args
from utils.torch_utils import select_device, de_parallel


class MedicalCompleteTrainer:
    """
    Complete medical image trainer combining detection and classification
    """
    def __init__(self, config_path, device='cuda'):
        self.device = select_device(device)
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.save_dir = increment_path(Path(self.config.get('project', 'runs/medical_complete')) / 
                                     self.config.get('name', 'exp'), exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.save_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        # Initialize models
        self.setup_models()
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Setup optimizers and schedulers
        self.setup_optimizers()
        
        # Setup loss functions
        self.setup_loss_functions()
        
        # Setup logging
        self.setup_logging()
        
        print(f"Medical Complete Trainer initialized. Save dir: {self.save_dir}")
    
    def setup_models(self):
        """Setup detection and classification models"""
        # Detection model (YOLOv5)
        detection_cfg = self.config.get('detection', {})
        self.detection_model = Model(
            cfg=detection_cfg.get('cfg', 'models/yolov5s.yaml'),
            ch=3,
            nc=self.config['nc'],
            anchors=None
        ).to(self.device)
        
        # Classification model (ResNet)
        classification_cfg = self.config.get('classification', {})
        self.classification_model = create_medical_classifier(
            classifier_type=classification_cfg.get('type', 'resnet'),
            num_classes=self.config.get('classification_classes', 4),
            pretrained=classification_cfg.get('pretrained', True)
        ).to(self.device)
        
        print(f"Detection model: {sum(p.numel() for p in self.detection_model.parameters()):,} parameters")
        print(f"Classification model: {sum(p.numel() for p in self.classification_model.parameters()):,} parameters")
    
    def setup_data_loaders(self):
        """Setup data loaders"""
        data_cfg = self.config.get('data', {})
        self.train_loader = create_medical_dataloader(
            data_yaml_path=self.config['data_yaml_path'],
            batch_size=data_cfg.get('batch_size', 16),
            num_workers=data_cfg.get('num_workers', 4),
            img_size=data_cfg.get('img_size', 640)
        ).get_train_loader()
        
        self.val_loader = create_medical_dataloader(
            data_yaml_path=self.config['data_yaml_path'],
            batch_size=data_cfg.get('batch_size', 16),
            num_workers=data_cfg.get('num_workers', 4),
            img_size=data_cfg.get('img_size', 640)
        ).get_val_loader()
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
    
    def setup_optimizers(self):
        """Setup optimizers and schedulers"""
        opt_cfg = self.config.get('optimizer', {})
        
        # Detection optimizer
        self.detection_optimizer = optim.AdamW(
            self.detection_model.parameters(),
            lr=opt_cfg.get('detection_lr', 0.001),
            weight_decay=opt_cfg.get('weight_decay', 0.0005)
        )
        
        # Classification optimizer
        self.classification_optimizer = optim.AdamW(
            self.classification_model.parameters(),
            lr=opt_cfg.get('classification_lr', 0.001),
            weight_decay=opt_cfg.get('weight_decay', 0.0005)
        )
        
        # Schedulers
        self.detection_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.detection_optimizer, 
            T_max=self.config.get('epochs', 100)
        )
        
        self.classification_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.classification_optimizer, 
            T_max=self.config.get('epochs', 100)
        )
    
    def setup_loss_functions(self):
        """Setup loss functions"""
        # Detection loss
        self.detection_loss_fn = ComputeLoss(self.detection_model)
        
        # Classification loss
        class_weights = self.config.get('classification_weights', None)
        if class_weights:
            class_weights = torch.tensor(class_weights, device=self.device)
        
        self.classification_loss_fn = get_medical_classifier_loss(class_weights)
        
        # Loss weights
        self.detection_weight = self.config.get('detection_weight', 1.0)
        self.classification_weight = self.config.get('classification_weight', 0.1)
    
    def setup_logging(self):
        """Setup logging"""
        self.writer = SummaryWriter(self.save_dir / 'logs')
        self.best_detection_map = 0.0
        self.best_classification_acc = 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.detection_model.train()
        self.classification_model.train()
        
        total_detection_loss = 0.0
        total_classification_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['images'].to(self.device)
            detection_targets = batch['detection_targets'].to(self.device)
            classification_labels = batch['classification_labels'].to(self.device)
            
            # Forward pass - Detection
            detection_outputs = self.detection_model(images)
            detection_loss, detection_loss_items = self.detection_loss_fn(detection_outputs, detection_targets)
            
            # Forward pass - Classification
            classification_outputs = self.classification_model(images)
            classification_loss = self.classification_loss_fn(classification_outputs, classification_labels)
            
            # Combined loss
            combined_loss = (self.detection_weight * detection_loss + 
                           self.classification_weight * classification_loss)
            
            # Backward pass
            self.detection_optimizer.zero_grad()
            self.classification_optimizer.zero_grad()
            combined_loss.backward()
            self.detection_optimizer.step()
            self.classification_optimizer.step()
            
            # Update metrics
            total_detection_loss += detection_loss.item()
            total_classification_loss += classification_loss.item()
            total_loss += combined_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'det_loss': f'{detection_loss.item():.4f}',
                'cls_loss': f'{classification_loss.item():.4f}',
                'total_loss': f'{combined_loss.item():.4f}'
            })
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/Detection', detection_loss.item(), step)
                self.writer.add_scalar('Loss/Classification', classification_loss.item(), step)
                self.writer.add_scalar('Loss/Total', combined_loss.item(), step)
        
        # Update schedulers
        self.detection_scheduler.step()
        self.classification_scheduler.step()
        
        return {
            'detection_loss': total_detection_loss / num_batches,
            'classification_loss': total_classification_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.detection_model.eval()
        self.classification_model.eval()
        
        total_detection_loss = 0.0
        total_classification_loss = 0.0
        total_loss = 0.0
        correct_classifications = 0
        total_classifications = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Validation {epoch}'):
                images = batch['images'].to(self.device)
                detection_targets = batch['detection_targets'].to(self.device)
                classification_labels = batch['classification_labels'].to(self.device)
                
                # Forward pass - Detection
                detection_outputs = self.detection_model(images)
                detection_loss, _ = self.detection_loss_fn(detection_outputs, detection_targets)
                
                # Forward pass - Classification
                classification_outputs = self.classification_model(images)
                classification_loss = self.classification_loss_fn(classification_outputs, classification_labels)
                
                # Combined loss
                combined_loss = (self.detection_weight * detection_loss + 
                               self.classification_weight * classification_loss)
                
                # Classification accuracy
                _, predicted = torch.max(classification_outputs, 1)
                correct_classifications += (predicted == classification_labels).sum().item()
                total_classifications += classification_labels.size(0)
                
                # Update metrics
                total_detection_loss += detection_loss.item()
                total_classification_loss += classification_loss.item()
                total_loss += combined_loss.item()
                num_batches += 1
        
        classification_accuracy = correct_classifications / total_classifications
        
        return {
            'detection_loss': total_detection_loss / num_batches,
            'classification_loss': total_classification_loss / num_batches,
            'total_loss': total_loss / num_batches,
            'classification_accuracy': classification_accuracy
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'detection_model_state_dict': self.detection_model.state_dict(),
            'classification_model_state_dict': self.classification_model.state_dict(),
            'detection_optimizer_state_dict': self.detection_optimizer.state_dict(),
            'classification_optimizer_state_dict': self.classification_optimizer.state_dict(),
            'detection_scheduler_state_dict': self.detection_scheduler.state_dict(),
            'classification_scheduler_state_dict': self.classification_scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            print(f"New best model saved at epoch {epoch}")
    
    def train(self):
        """Main training loop"""
        epochs = self.config.get('epochs', 100)
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Log metrics
            self.writer.add_scalar('Train/Detection_Loss', train_metrics['detection_loss'], epoch)
            self.writer.add_scalar('Train/Classification_Loss', train_metrics['classification_loss'], epoch)
            self.writer.add_scalar('Train/Total_Loss', train_metrics['total_loss'], epoch)
            
            self.writer.add_scalar('Val/Detection_Loss', val_metrics['detection_loss'], epoch)
            self.writer.add_scalar('Val/Classification_Loss', val_metrics['classification_loss'], epoch)
            self.writer.add_scalar('Val/Total_Loss', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('Val/Classification_Accuracy', val_metrics['classification_accuracy'], epoch)
            
            # Print metrics
            print(f"Epoch {epoch}:")
            print(f"  Train - Det: {train_metrics['detection_loss']:.4f}, "
                  f"Cls: {train_metrics['classification_loss']:.4f}, "
                  f"Total: {train_metrics['total_loss']:.4f}")
            print(f"  Val   - Det: {val_metrics['detection_loss']:.4f}, "
                  f"Cls: {val_metrics['classification_loss']:.4f}, "
                  f"Total: {val_metrics['total_loss']:.4f}, "
                  f"Acc: {val_metrics['classification_accuracy']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['classification_accuracy'] > self.best_classification_acc
            if is_best:
                self.best_classification_acc = val_metrics['classification_accuracy']
            
            self.save_checkpoint(epoch, val_metrics, is_best)
        
        print(f"Training completed. Best accuracy: {self.best_classification_acc:.4f}")
        self.writer.close()


def create_training_config():
    """Create default training configuration"""
    config = {
        'data_yaml_path': 'data_medical_complete.yaml',
        'project': 'runs/medical_complete',
        'name': 'heart_ultrasound_complete',
        'epochs': 100,
        'nc': 4,  # Number of detection classes
        'classification_classes': 4,  # Number of classification classes
        
        'data': {
            'batch_size': 16,
            'num_workers': 4,
            'img_size': 640
        },
        
        'detection': {
            'cfg': 'models/yolov5s.yaml',
            'pretrained': True
        },
        
        'classification': {
            'type': 'resnet',
            'pretrained': True
        },
        
        'optimizer': {
            'detection_lr': 0.001,
            'classification_lr': 0.001,
            'weight_decay': 0.0005
        },
        
        'detection_weight': 1.0,
        'classification_weight': 0.1,
        
        'classification_weights': [1.0, 1.0, 1.0, 1.0]  # Class weights for imbalance
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Medical Complete Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Create or load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_training_config()
        # Save default config
        with open('config_medical_complete.yaml', 'w') as f:
            yaml.dump(config, f)
        print("Default config saved to config_medical_complete.yaml")
    
    # Update epochs if specified
    if args.epochs != 100:
        config['epochs'] = args.epochs
    
    # Create trainer and start training
    trainer = MedicalCompleteTrainer(config, device=args.device)
    trainer.train()


if __name__ == '__main__':
    main() 