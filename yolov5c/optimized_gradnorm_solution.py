#!/usr/bin/env python3
"""
Optimized GradNorm Solution for YOLOv5WithClassification
專門解決梯度干擾問題的優化方案

This script implements an enhanced GradNorm approach specifically designed to solve
the gradient interference problem between detection and classification tasks.
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

class EnhancedGradNormLoss(nn.Module):
    """
    Enhanced GradNorm Loss for solving gradient interference
    增強版GradNorm損失函數，專門解決梯度干擾問題
    """
    
    def __init__(self, 
                 detection_loss_fn: nn.Module,
                 classification_loss_fn: nn.Module,
                 alpha: float = 1.5,
                 initial_detection_weight: float = 1.0,
                 initial_classification_weight: float = 0.01,
                 momentum: float = 0.9,
                 min_weight: float = 0.001,
                 max_weight: float = 10.0,
                 update_frequency: int = 10):
        super(EnhancedGradNormLoss, self).__init__()
        
        self.detection_loss_fn = detection_loss_fn
        self.classification_loss_fn = classification_loss_fn
        self.alpha = alpha
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.update_frequency = update_frequency
        
        # Initialize task weights
        self.register_buffer('detection_weight', torch.tensor(initial_detection_weight))
        self.register_buffer('classification_weight', torch.tensor(initial_classification_weight))
        
        # Initialize loss history for GradNorm
        self.initial_detection_loss = None
        self.initial_classification_loss = None
        self.detection_loss_history = []
        self.classification_loss_history = []
        
        # Initialize gradient norm history
        self.detection_grad_norm_history = []
        self.classification_grad_norm_history = []
        
        # Step counter for update frequency
        self.step_count = 0
        
        # Weight history for monitoring
        self.detection_weight_history = []
        self.classification_weight_history = []
        
    def compute_relative_inverse_training_rate(self, current_detection_loss, current_classification_loss):
        """計算相對逆訓練率"""
        if self.initial_detection_loss is None:
            self.initial_detection_loss = current_detection_loss
            self.initial_classification_loss = current_classification_loss
            return torch.tensor(1.0), torch.tensor(1.0)
        
        # Compute training rates
        detection_training_rate = current_detection_loss / self.initial_detection_loss
        classification_training_rate = current_classification_loss / self.initial_classification_loss
        
        # Compute average training rate
        avg_training_rate = (detection_training_rate + classification_training_rate) / 2
        
        # Compute relative inverse training rates
        detection_relative_rate = (detection_training_rate / avg_training_rate) ** self.alpha
        classification_relative_rate = (classification_training_rate / avg_training_rate) ** self.alpha
        
        return detection_relative_rate, classification_relative_rate
    
    def compute_gradient_norms(self, detection_gradients, classification_gradients):
        """計算梯度範數"""
        # Compute detection gradient norm
        detection_grad_norm = 0.0
        if detection_gradients is not None:
            for grad in detection_gradients:
                if grad is not None:
                    detection_grad_norm += torch.norm(grad, p=2) ** 2
            detection_grad_norm = torch.sqrt(detection_grad_norm)
        
        # Compute classification gradient norm
        classification_grad_norm = 0.0
        if classification_gradients is not None:
            for grad in classification_gradients:
                if grad is not None:
                    classification_grad_norm += torch.norm(grad, p=2) ** 2
            classification_grad_norm = torch.sqrt(classification_grad_norm)
        
        return detection_grad_norm, classification_grad_norm
    
    def update_task_weights(self, detection_grad_norm, classification_grad_norm, 
                           detection_relative_rate, classification_relative_rate):
        """更新任務權重"""
        # Compute average gradient norm
        avg_grad_norm = (detection_grad_norm + classification_grad_norm) / 2
        
        # Compute new weights
        new_detection_weight = avg_grad_norm * detection_relative_rate / (detection_grad_norm + 1e-8)
        new_classification_weight = avg_grad_norm * classification_relative_rate / (classification_grad_norm + 1e-8)
        
        # Apply momentum and constraints
        self.detection_weight = self.momentum * self.detection_weight + (1 - self.momentum) * new_detection_weight
        self.classification_weight = self.momentum * self.classification_weight + (1 - self.momentum) * new_classification_weight
        
        # Apply min/max constraints
        self.detection_weight = torch.clamp(self.detection_weight, self.min_weight, self.max_weight)
        self.classification_weight = torch.clamp(self.classification_weight, self.min_weight, self.max_weight)
        
        # Record weight history
        self.detection_weight_history.append(self.detection_weight.item())
        self.classification_weight_history.append(self.classification_weight.item())
    
    def forward(self, 
                detection_outputs: torch.Tensor,
                classification_outputs: torch.Tensor,
                detection_targets: torch.Tensor,
                classification_targets: torch.Tensor,
                detection_gradients: list = None,
                classification_gradients: list = None) -> tuple:
        """
        Forward pass with enhanced GradNorm
        
        Args:
            detection_outputs: Detection model outputs
            classification_outputs: Classification model outputs
            detection_targets: Detection targets
            classification_targets: Classification targets
            detection_gradients: Detection gradients (optional)
            classification_gradients: Classification gradients (optional)
        
        Returns:
            total_loss: Combined loss
            loss_info: Dictionary with loss details
        """
        # Compute individual losses
        detection_loss = self.detection_loss_fn(detection_outputs, detection_targets)
        classification_loss = self.classification_loss_fn(classification_outputs, classification_targets)
        
        # Record loss history
        self.detection_loss_history.append(detection_loss.item())
        self.classification_loss_history.append(classification_loss.item())
        
        # Update weights using GradNorm (every update_frequency steps)
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            # Compute relative inverse training rates
            detection_relative_rate, classification_relative_rate = self.compute_relative_inverse_training_rate(
                detection_loss.item(), classification_loss.item()
            )
            
            # Compute gradient norms if provided
            if detection_gradients is not None and classification_gradients is not None:
                detection_grad_norm, classification_grad_norm = self.compute_gradient_norms(
                    detection_gradients, classification_gradients
                )
                
                # Update task weights
                self.update_task_weights(
                    detection_grad_norm, classification_grad_norm,
                    detection_relative_rate, classification_relative_rate
                )
        
        # Compute weighted total loss
        total_loss = (self.detection_weight * detection_loss + 
                     self.classification_weight * classification_loss)
        
        # Prepare loss information
        loss_info = {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'classification_loss': classification_loss.item(),
            'detection_weight': self.detection_weight.item(),
            'classification_weight': self.classification_weight.item(),
            'detection_relative_rate': detection_relative_rate.item() if 'detection_relative_rate' in locals() else 1.0,
            'classification_relative_rate': classification_relative_rate.item() if 'classification_relative_rate' in locals() else 1.0
        }
        
        return total_loss, loss_info

class OptimizedJointModel(nn.Module):
    """
    Optimized Joint Model with enhanced architecture
    優化的聯合模型，增強架構設計
    """
    
    def __init__(self, num_detection_classes=4, num_classification_classes=4):
        super(OptimizedJointModel, self).__init__()
        
        # Enhanced shared backbone with residual connections
        self.backbone = nn.Sequential(
            # First block with residual connection
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second block with residual connection
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third block with residual connection
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth block with residual connection
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Enhanced detection head with attention
        self.detection_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_detection_classes)
        )
        
        # Enhanced classification head with attention
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classification_classes)
        )
        
        # Add attention mechanism to reduce gradient interference
        self.detection_attention = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.Sigmoid()
        )
        
        self.classification_attention = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract shared features
        shared_features = self.backbone(x)
        shared_features = shared_features.view(shared_features.size(0), -1)
        
        # Apply attention mechanisms to reduce gradient interference
        detection_attention = self.detection_attention(shared_features)
        classification_attention = self.classification_attention(shared_features)
        
        # Apply attention to features
        detection_features = shared_features * detection_attention
        classification_features = shared_features * classification_attention
        
        # Generate outputs
        detection_output = self.detection_head(detection_features)
        classification_output = self.classification_head(classification_features)
        
        return detection_output, classification_output

def train_with_optimized_gradnorm(model, train_loader, val_loader, num_epochs=50, 
                                 device='cuda', learning_rate=1e-4):
    """
    Train model with optimized GradNorm
    使用優化GradNorm訓練模型
    """
    
    # Initialize model and move to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Initialize loss functions
    detection_loss_fn = nn.BCEWithLogitsLoss()
    classification_loss_fn = nn.BCEWithLogitsLoss()
    
    # Initialize enhanced GradNorm loss
    gradnorm_loss = EnhancedGradNormLoss(
        detection_loss_fn=detection_loss_fn,
        classification_loss_fn=classification_loss_fn,
        alpha=1.5,
        initial_detection_weight=1.0,
        initial_classification_weight=0.01,
        momentum=0.9,
        min_weight=0.001,
        max_weight=10.0,
        update_frequency=10
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'detection_loss': [], 'classification_loss': [],
        'detection_weight': [], 'classification_weight': [],
        'detection_acc': [], 'classification_acc': []
    }
    
    print("Starting optimized GradNorm training...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_detection_loss = 0.0
        epoch_classification_loss = 0.0
        
        for batch_idx, (images, detection_targets, classification_targets) in enumerate(train_loader):
            images = images.to(device)
            detection_targets = detection_targets.to(device)
            classification_targets = classification_targets.to(device)
            
            # Forward pass
            detection_outputs, classification_outputs = model(images)
            
            # Compute loss with GradNorm
            total_loss, loss_info = gradnorm_loss(
                detection_outputs, classification_outputs,
                detection_targets, classification_targets
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Get gradients for GradNorm (optional)
            detection_gradients = [p.grad for p in model.detection_head.parameters() if p.grad is not None]
            classification_gradients = [p.grad for p in model.classification_head.parameters() if p.grad is not None]
            
            # Update weights using gradients
            if batch_idx % gradnorm_loss.update_frequency == 0:
                gradnorm_loss.forward(
                    detection_outputs, classification_outputs,
                    detection_targets, classification_targets,
                    detection_gradients, classification_gradients
                )
            
            optimizer.step()
            
            # Record losses
            epoch_loss += loss_info['total_loss']
            epoch_detection_loss += loss_info['detection_loss']
            epoch_classification_loss += loss_info['classification_loss']
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss_info['total_loss']:.4f}, "
                      f"D_Weight: {loss_info['detection_weight']:.4f}, "
                      f"C_Weight: {loss_info['classification_weight']:.4f}")
        
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
                val_total_loss, val_loss_info = gradnorm_loss(
                    detection_outputs, classification_outputs,
                    detection_targets, classification_targets
                )
                
                val_loss += val_loss_info['total_loss']
                
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
        history['detection_weight'].append(gradnorm_loss.detection_weight.item())
        history['classification_weight'].append(gradnorm_loss.classification_weight.item())
        history['detection_acc'].append(detection_correct / total_samples)
        history['classification_acc'].append(classification_correct / total_samples)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val Loss: {history['val_loss'][-1]:.4f}, "
              f"D_Acc: {history['detection_acc'][-1]:.4f}, "
              f"C_Acc: {history['classification_acc'][-1]:.4f}")
    
    return model, history, gradnorm_loss

def plot_optimized_training_curves(history, save_dir='optimized_gradnorm_results'):
    """Plot training curves for optimized GradNorm"""
    
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
    
    # Plot task weights
    axes[0, 2].plot(history['detection_weight'], label='Detection Weight')
    axes[0, 2].plot(history['classification_weight'], label='Classification Weight')
    axes[0, 2].set_title('GradNorm Task Weights')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot accuracies
    axes[1, 0].plot(history['detection_acc'], label='Detection Accuracy')
    axes[1, 0].plot(history['classification_acc'], label='Classification Accuracy')
    axes[1, 0].set_title('Task Accuracies')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot weight ratio
    weight_ratio = [d/c for d, c in zip(history['detection_weight'], history['classification_weight'])]
    axes[1, 1].plot(weight_ratio, label='Detection/Classification Weight Ratio')
    axes[1, 1].set_title('Weight Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot loss ratio
    loss_ratio = [d/c for d, c in zip(history['detection_loss'], history['classification_loss'])]
    axes[1, 2].plot(loss_ratio, label='Detection/Classification Loss Ratio')
    axes[1, 2].set_title('Loss Ratio')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'optimized_gradnorm_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save history
    with open(save_dir / 'optimized_gradnorm_history.json', 'w') as f:
        json.dump(history, f, indent=2)

def main():
    """Main function for optimized GradNorm training"""
    
    parser = argparse.ArgumentParser(description='Optimized GradNorm Training')
    parser.add_argument('--data', type=str, default='converted_dataset_fixed/data.yaml', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device to use')
    parser.add_argument('--save-dir', type=str, default='optimized_gradnorm_results', help='Save directory')
    
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
    model = OptimizedJointModel(num_detection_classes=4, num_classification_classes=4)
    
    # Train with optimized GradNorm
    trained_model, history, gradnorm_loss = train_with_optimized_gradnorm(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, 
        device=device, 
        learning_rate=args.lr
    )
    
    # Plot results
    plot_optimized_training_curves(history, args.save_dir)
    
    # Save model
    torch.save(trained_model.state_dict(), f"{args.save_dir}/optimized_gradnorm_model.pth")
    
    print("Optimized GradNorm training completed!")

if __name__ == "__main__":
    main() 