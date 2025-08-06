#!/usr/bin/env python3
"""
Dual Independent Backbone Solution for YOLOv5WithClassification
雙獨立Backbone解決方案，完全消除梯度干擾

This script implements separate backbones for detection and classification tasks
to completely eliminate gradient interference between tasks.
"""

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
import argparse
import yaml
import sys

# Add yolov5c to path
sys.path.append(str(Path(__file__).parent))

class DetectionBackbone(nn.Module):
    """
    Independent backbone for detection task
    專門用於檢測任務的獨立Backbone
    """
    
    def __init__(self, in_channels=3, width_multiple=0.5, depth_multiple=0.33):
        super(DetectionBackbone, self).__init__()
        
        # Calculate channel numbers based on width multiple
        base_channels = int(64 * width_multiple)
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        
        # Calculate layer numbers based on depth multiple
        depths = [max(round(x * depth_multiple), 1) for x in [1, 3, 6, 9, 3]]
        
        # Detection backbone layers - optimized for localization
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 6, 2, 2),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(inplace=True)
        )
        
        # C3 blocks for detection with residual connections
        self.c3_1 = self._make_c3_block(channels[1], channels[1], depths[1])
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(inplace=True)
        )
        
        self.c3_2 = self._make_c3_block(channels[2], channels[2], depths[2])
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2),
            nn.BatchNorm2d(channels[3]),
            nn.SiLU(inplace=True)
        )
        
        self.c3_3 = self._make_c3_block(channels[3], channels[3], depths[3])
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, 2),
            nn.BatchNorm2d(channels[4]),
            nn.SiLU(inplace=True)
        )
        
        self.c3_4 = self._make_c3_block(channels[4], channels[4], depths[4])
        
        # SPPF for detection
        self.sppf = self._make_sppf(channels[4], channels[4])
        
    def _make_c3_block(self, in_channels, out_channels, depth):
        """Create C3 block for detection backbone"""
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(Bottleneck(in_channels, out_channels, shortcut=True))
            else:
                layers.append(Bottleneck(out_channels, out_channels, shortcut=True))
        return nn.Sequential(*layers)
    
    def _make_sppf(self, in_channels, out_channels):
        """Create SPPF module"""
        return SPPF(in_channels, out_channels, k=5)
    
    def forward(self, x):
        # Detection backbone forward pass - optimized for spatial features
        x1 = self.conv1(x)      # P1/2
        x2 = self.conv2(x1)     # P2/4
        x2 = self.c3_1(x2)
        x3 = self.conv3(x2)     # P3/8
        x3 = self.c3_2(x3)
        x4 = self.conv4(x3)     # P4/16
        x4 = self.c3_3(x4)
        x5 = self.conv5(x4)     # P5/32
        x5 = self.c3_4(x5)
        x5 = self.sppf(x5)
        
        return [x1, x2, x3, x4, x5]

class ClassificationBackbone(nn.Module):
    """
    Independent backbone for classification task
    專門用於分類任務的獨立Backbone
    """
    
    def __init__(self, in_channels=3, width_multiple=0.5, depth_multiple=0.33):
        super(ClassificationBackbone, self).__init__()
        
        # Calculate channel numbers based on width multiple
        base_channels = int(64 * width_multiple)
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        
        # Calculate layer numbers based on depth multiple
        depths = [max(round(x * depth_multiple), 1) for x in [1, 3, 6, 9, 3]]
        
        # Classification backbone layers - optimized for semantic features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 2, 3),  # Larger kernel for global features
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(inplace=True)
        )
        
        # SE blocks for classification with attention
        self.se1 = SE(channels[1])
        self.c3_1 = self._make_c3_block(channels[1], channels[1], depths[1])
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(inplace=True)
        )
        
        self.se2 = SE(channels[2])
        self.c3_2 = self._make_c3_block(channels[2], channels[2], depths[2])
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2),
            nn.BatchNorm2d(channels[3]),
            nn.SiLU(inplace=True)
        )
        
        self.se3 = SE(channels[3])
        self.c3_3 = self._make_c3_block(channels[3], channels[3], depths[3])
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, 2),
            nn.BatchNorm2d(channels[4]),
            nn.SiLU(inplace=True)
        )
        
        self.se4 = SE(channels[4])
        self.c3_4 = self._make_c3_block(channels[4], channels[4], depths[4])
        
        # Global average pooling for classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_c3_block(self, in_channels, out_channels, depth):
        """Create C3 block for classification backbone"""
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(Bottleneck(in_channels, out_channels, shortcut=True))
            else:
                layers.append(Bottleneck(out_channels, out_channels, shortcut=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Classification backbone forward pass - optimized for semantic features
        x1 = self.conv1(x)      # P1/2
        x2 = self.conv2(x1)     # P2/4
        x2 = self.se1(x2)
        x2 = self.c3_1(x2)
        x3 = self.conv3(x2)     # P3/8
        x3 = self.se2(x3)
        x3 = self.c3_2(x3)
        x4 = self.conv4(x3)     # P4/16
        x4 = self.se3(x4)
        x4 = self.c3_3(x4)
        x5 = self.conv5(x4)     # P5/32
        x5 = self.se4(x5)
        x5 = self.c3_4(x5)
        x5 = self.global_pool(x5)
        
        return x5.view(x5.size(0), -1)  # Flatten for classification

class DualBackboneModel(nn.Module):
    """
    Dual Backbone Model with independent backbones
    雙Backbone模型，完全獨立的特徵提取器
    """
    
    def __init__(self, 
                 num_detection_classes=4, 
                 num_classification_classes=4,
                 width_multiple=0.5, 
                 depth_multiple=0.33):
        super(DualBackboneModel, self).__init__()
        
        # Independent backbones
        self.detection_backbone = DetectionBackbone(
            in_channels=3, 
            width_multiple=width_multiple, 
            depth_multiple=depth_multiple
        )
        
        self.classification_backbone = ClassificationBackbone(
            in_channels=3, 
            width_multiple=width_multiple, 
            depth_multiple=depth_multiple
        )
        
        # Detection head
        self.detection_head = DetectionHead(
            in_channels=int(512 * width_multiple), 
            num_classes=num_detection_classes,
            width_multiple=width_multiple
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(int(512 * width_multiple), 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classification_classes)
        )
        
    def forward(self, x):
        # Independent forward passes - no gradient interference
        detection_features = self.detection_backbone(x)
        classification_features = self.classification_backbone(x)
        
        # Generate outputs
        detection_output = self.detection_head(detection_features)
        classification_output = self.classification_head(classification_features)
        
        return detection_output, classification_output
    
    def get_task_parameters(self):
        """Get parameters for each task separately"""
        detection_params = list(self.detection_backbone.parameters()) + list(self.detection_head.parameters())
        classification_params = list(self.classification_backbone.parameters()) + list(self.classification_head.parameters())
        return detection_params, classification_params

class DetectionHead(nn.Module):
    """
    Detection head for dual backbone model
    雙Backbone模型的檢測頭
    """
    
    def __init__(self, in_channels, num_classes, width_multiple=0.5):
        super(DetectionHead, self).__init__()
        
        # Calculate channel numbers
        channels = [int(x * width_multiple) for x in [256, 128, 64]]
        
        # Detection head layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(inplace=True)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification layer
        self.classifier = nn.Linear(channels[2], num_classes)
        
    def forward(self, features):
        # features: [x1, x2, x3, x4, x5]
        # Use the deepest feature map (x5)
        x = features[-1]
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class Bottleneck(nn.Module):
    """Bottleneck block for C3 modules"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c2, 3, 1, padding=1, groups=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SPPF(nn.Module):
    """SPPF module for detection backbone"""
    
    def __init__(self, c1, c2, k=5):
        super(SPPF, self).__init__()
        c_ = c1 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class SE(nn.Module):
    """Squeeze-and-Excitation block for classification backbone"""
    
    def __init__(self, c1, c2=None, ratio=16):
        super(SE, self).__init__()
        c2 = c2 or c1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // ratio, c2, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DualBackboneJointTrainer:
    """
    Joint trainer for dual backbone model
    雙Backbone模型的聯合訓練器
    """
    
    def __init__(self, 
                 model: DualBackboneModel,
                 detection_lr: float = 1e-4,
                 classification_lr: float = 1e-4,
                 detection_weight: float = 1.0,
                 classification_weight: float = 0.01):
        
        self.model = model
        self.detection_weight = detection_weight
        self.classification_weight = classification_weight
        
        # Separate optimizers for each task
        detection_params, classification_params = model.get_task_parameters()
        
        self.detection_optimizer = optim.AdamW(detection_params, lr=detection_lr, weight_decay=1e-4)
        self.classification_optimizer = optim.AdamW(classification_params, lr=classification_lr, weight_decay=1e-4)
        
        # Separate schedulers
        self.detection_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.detection_optimizer, T_max=100)
        self.classification_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.classification_optimizer, T_max=100)
        
        # Loss functions
        self.detection_loss_fn = nn.BCEWithLogitsLoss()
        self.classification_loss_fn = nn.BCEWithLogitsLoss()
    
    def train_step(self, 
                   images: torch.Tensor,
                   detection_targets: torch.Tensor,
                   classification_targets: torch.Tensor) -> dict:
        """
        Single training step with independent optimization
        單步訓練，獨立優化每個任務
        """
        
        # Forward pass
        detection_outputs, classification_outputs = self.model(images)
        
        # Compute losses
        detection_loss = self.detection_loss_fn(detection_outputs, detection_targets)
        classification_loss = self.classification_loss_fn(classification_outputs, classification_targets)
        
        # Independent backward passes
        # Detection optimization
        self.detection_optimizer.zero_grad()
        detection_loss.backward(retain_graph=True)
        self.detection_optimizer.step()
        
        # Classification optimization
        self.classification_optimizer.zero_grad()
        classification_loss.backward()
        self.classification_optimizer.step()
        
        # Compute total loss for monitoring
        total_loss = (self.detection_weight * detection_loss + 
                     self.classification_weight * classification_loss)
        
        return {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'classification_loss': classification_loss.item(),
            'detection_lr': self.detection_optimizer.param_groups[0]['lr'],
            'classification_lr': self.classification_optimizer.param_groups[0]['lr']
        }
    
    def step_schedulers(self):
        """Step learning rate schedulers"""
        self.detection_scheduler.step()
        self.classification_scheduler.step()

def train_dual_backbone_model(model, train_loader, val_loader, num_epochs=50, 
                             device='cuda', detection_lr=1e-4, classification_lr=1e-4):
    """
    Train dual backbone model with independent optimization
    使用獨立優化訓練雙Backbone模型
    """
    
    # Initialize model and move to device
    model = model.to(device)
    
    # Initialize joint trainer
    trainer = DualBackboneJointTrainer(
        model=model,
        detection_lr=detection_lr,
        classification_lr=classification_lr,
        detection_weight=1.0,
        classification_weight=0.01
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'detection_loss': [], 'classification_loss': [],
        'detection_lr': [], 'classification_lr': [],
        'detection_acc': [], 'classification_acc': []
    }
    
    print("Starting dual backbone training with independent optimization...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_detection_loss = 0.0
        epoch_classification_loss = 0.0
        
        for batch_idx, (images, detection_targets, classification_targets) in enumerate(train_loader):
            images = images.to(device)
            detection_targets = detection_targets.to(device)
            classification_targets = classification_targets.to(device)
            
            # Training step
            step_info = trainer.train_step(images, detection_targets, classification_targets)
            
            # Record losses
            epoch_loss += step_info['total_loss']
            epoch_detection_loss += step_info['detection_loss']
            epoch_classification_loss += step_info['classification_loss']
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {step_info['total_loss']:.4f}, "
                      f"D_Loss: {step_info['detection_loss']:.4f}, "
                      f"C_Loss: {step_info['classification_loss']:.4f}")
        
        # Step schedulers
        trainer.step_schedulers()
        
        # Validation
        model.eval()
        val_loss = 0.0
        detection_correct = 0
        classification_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, detection_targets, classification_targets in val_loader:
                images = images.to(device)
                detection_targets = detection_targets.to(device)
                classification_targets = classification_targets.to(device)
                
                detection_outputs, classification_outputs = model(images)
                
                # Compute validation loss
                detection_loss = trainer.detection_loss_fn(detection_outputs, detection_targets)
                classification_loss = trainer.classification_loss_fn(classification_outputs, classification_targets)
                val_total_loss = (trainer.detection_weight * detection_loss + 
                                trainer.classification_weight * classification_loss)
                
                val_loss += val_total_loss.item()
                
                # Compute accuracy
                detection_pred = (torch.sigmoid(detection_outputs) > 0.5).float()
                classification_pred = (torch.sigmoid(classification_outputs) > 0.5).float()
                
                detection_correct += (detection_pred == detection_targets).all(dim=1).sum().item()
                classification_correct += (classification_pred == classification_targets).all(dim=1).sum().item()
                total_samples += images.size(0)
        
        # Record history
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['detection_loss'].append(epoch_detection_loss / len(train_loader))
        history['classification_loss'].append(epoch_classification_loss / len(train_loader))
        history['detection_lr'].append(trainer.detection_optimizer.param_groups[0]['lr'])
        history['classification_lr'].append(trainer.classification_optimizer.param_groups[0]['lr'])
        history['detection_acc'].append(detection_correct / total_samples)
        history['classification_acc'].append(classification_correct / total_samples)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val Loss: {history['val_loss'][-1]:.4f}, "
              f"D_Acc: {history['detection_acc'][-1]:.4f}, "
              f"C_Acc: {history['classification_acc'][-1]:.4f}")
    
    return model, history, trainer

def plot_dual_backbone_training_curves(history, save_dir='dual_backbone_results'):
    """Plot training curves for dual backbone model"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot losses
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot individual losses
    axes[0, 1].plot(history['detection_loss'], label='Detection Loss')
    axes[0, 1].plot(history['classification_loss'], label='Classification Loss')
    axes[0, 1].set_title('Individual Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rates
    axes[0, 2].plot(history['detection_lr'], label='Detection LR')
    axes[0, 2].plot(history['classification_lr'], label='Classification LR')
    axes[0, 2].set_title('Learning Rates')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot accuracies
    axes[1, 0].plot(history['detection_acc'], label='Detection Accuracy')
    axes[1, 0].plot(history['classification_acc'], label='Classification Accuracy')
    axes[1, 0].set_title('Task Accuracies')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot loss ratio
    loss_ratio = [d/c for d, c in zip(history['detection_loss'], history['classification_loss'])]
    axes[1, 1].plot(loss_ratio, label='Detection/Classification Loss Ratio')
    axes[1, 1].set_title('Loss Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot accuracy ratio
    acc_ratio = [d/c for d, c in zip(history['detection_acc'], history['classification_acc'])]
    axes[1, 2].plot(acc_ratio, label='Detection/Classification Accuracy Ratio')
    axes[1, 2].set_title('Accuracy Ratio')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'dual_backbone_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save history
    with open(save_dir / 'dual_backbone_history.json', 'w') as f:
        json.dump(history, f, indent=2)

def main():
    """Main function for dual backbone training"""
    
    parser = argparse.ArgumentParser(description='Dual Backbone Training')
    parser.add_argument('--data', type=str, default='converted_dataset_fixed/data.yaml', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--detection-lr', type=float, default=1e-4, help='Detection learning rate')
    parser.add_argument('--classification-lr', type=float, default=1e-4, help='Classification learning rate')
    parser.add_argument('--device', default='auto', help='Device to use')
    parser.add_argument('--save-dir', type=str, default='dual_backbone_results', help='Save directory')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load dataset (you'll need to implement this based on your dataset structure)
    # train_loader, val_loader = load_dataset(args.data, args.batch_size)
    
    # Initialize model
    model = DualBackboneModel(
        num_detection_classes=4, 
        num_classification_classes=4,
        width_multiple=0.5, 
        depth_multiple=0.33
    )
    
    # Train with dual backbone
    trained_model, history, trainer = train_dual_backbone_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, 
        device=device, 
        detection_lr=args.detection_lr,
        classification_lr=args.classification_lr
    )
    
    # Plot results
    plot_dual_backbone_training_curves(history, args.save_dir)
    
    # Save model
    torch.save(trained_model.state_dict(), f"{args.save_dir}/dual_backbone_model.pth")
    
    print("Dual backbone training completed!")

if __name__ == "__main__":
    main() 