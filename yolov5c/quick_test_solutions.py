#!/usr/bin/env python3
"""
Quick Test Solutions for Gradient Interference Problem
梯度干擾問題的快速測試解決方案

This script provides quick testing for both GradNorm and Dual Backbone solutions
to validate their effectiveness in solving gradient interference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import sys
import time
from datetime import datetime

# Add yolov5c to path
sys.path.append(str(Path(__file__).parent))

class SimpleDataset(Dataset):
    """Simple dataset for quick testing"""
    
    def __init__(self, num_samples=100, num_detection_classes=4, num_classification_classes=4):
        self.num_samples = num_samples
        self.num_detection_classes = num_detection_classes
        self.num_classification_classes = num_classification_classes
        
        # Generate synthetic data
        self.images = torch.randn(num_samples, 3, 64, 64)  # Simplified image size
        self.detection_targets = torch.randint(0, 2, (num_samples, num_detection_classes)).float()
        self.classification_targets = torch.randint(0, 2, (num_samples, num_classification_classes)).float()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.detection_targets[idx], self.classification_targets[idx]

class BaselineModel(nn.Module):
    """Baseline model with gradient interference problem"""
    
    def __init__(self, num_detection_classes=4, num_classification_classes=4):
        super(BaselineModel, self).__init__()
        
        # Shared backbone (problematic)
                self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_detection_classes)
        )
        
        # Classification head (causes gradient interference)
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classification_classes)
        )
            
            def forward(self, x):
        shared_features = self.backbone(x)
        shared_features = shared_features.view(shared_features.size(0), -1)
        
        detection_output = self.detection_head(shared_features)
        classification_output = self.classification_head(shared_features)
        
        return detection_output, classification_output

class GradNormModel(nn.Module):
    """Model with GradNorm solution"""
    
    def __init__(self, num_detection_classes=4, num_classification_classes=4):
        super(GradNormModel, self).__init__()
        
        # Enhanced backbone with attention
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Attention mechanisms to reduce gradient interference
        self.detection_attention = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.Sigmoid()
        )
        
        self.classification_attention = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.Sigmoid()
        )
        
        # Task-specific heads
        self.detection_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_detection_classes)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classification_classes)
        )
    
    def forward(self, x):
        shared_features = self.backbone(x)
        shared_features = shared_features.view(shared_features.size(0), -1)
        
        # Apply attention to reduce gradient interference
        detection_attention = self.detection_attention(shared_features)
        classification_attention = self.classification_attention(shared_features)
        
        detection_features = shared_features * detection_attention
        classification_features = shared_features * classification_attention
        
        detection_output = self.detection_head(detection_features)
        classification_output = self.classification_head(classification_features)
        
        return detection_output, classification_output

class DualBackboneModel(nn.Module):
    """Model with dual backbone solution"""
    
    def __init__(self, num_detection_classes=4, num_classification_classes=4):
        super(DualBackboneModel, self).__init__()
        
        # Independent detection backbone
        self.detection_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Independent classification backbone
        self.classification_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Task-specific heads
        self.detection_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_detection_classes)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classification_classes)
        )
    
    def forward(self, x):
        # Independent forward passes - no gradient interference
        detection_features = self.detection_backbone(x)
        detection_features = detection_features.view(detection_features.size(0), -1)
        
        classification_features = self.classification_backbone(x)
        classification_features = classification_features.view(classification_features.size(0), -1)
        
        detection_output = self.detection_head(detection_features)
        classification_output = self.classification_head(classification_features)
        
        return detection_output, classification_output

class SimpleGradNormLoss(nn.Module):
    """Simplified GradNorm loss for quick testing"""
    
    def __init__(self, alpha=1.5):
        super(SimpleGradNormLoss, self).__init__()
        self.alpha = alpha
        self.detection_weight = 1.0
        self.classification_weight = 0.01
        self.initial_detection_loss = None
        self.initial_classification_loss = None
        
    def forward(self, detection_outputs, classification_outputs, 
                detection_targets, classification_targets):
        
        # Compute losses
        detection_loss = nn.BCEWithLogitsLoss()(detection_outputs, detection_targets)
        classification_loss = nn.BCEWithLogitsLoss()(classification_outputs, classification_targets)
        
        # Initialize initial losses
        if self.initial_detection_loss is None:
            self.initial_detection_loss = detection_loss.item()
            self.initial_classification_loss = classification_loss.item()
        
        # Simple GradNorm weight adjustment
        detection_rate = detection_loss.item() / self.initial_detection_loss
        classification_rate = classification_loss.item() / self.initial_classification_loss
        
        avg_rate = (detection_rate + classification_rate) / 2
        
        # Update weights
        self.detection_weight = (detection_rate / avg_rate) ** self.alpha
        self.classification_weight = (classification_rate / avg_rate) ** self.alpha
        
        # Clamp weights
        self.detection_weight = max(0.1, min(10.0, self.detection_weight))
        self.classification_weight = max(0.001, min(1.0, self.classification_weight))
        
        # Compute weighted total loss
        total_loss = (self.detection_weight * detection_loss + 
                     self.classification_weight * classification_loss)
        
        return total_loss, {
            'detection_loss': detection_loss.item(),
            'classification_loss': classification_loss.item(),
            'detection_weight': self.detection_weight,
            'classification_weight': self.classification_weight
        }

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu', 
                use_gradnorm=False, model_name="Model"):
    """Train model and return training history"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    if use_gradnorm:
        loss_fn = SimpleGradNormLoss(alpha=1.5)
    else:
        detection_loss_fn = nn.BCEWithLogitsLoss()
        classification_loss_fn = nn.BCEWithLogitsLoss()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'detection_loss': [], 'classification_loss': [],
        'detection_acc': [], 'classification_acc': []
    }
    
    if use_gradnorm:
        history['detection_weight'] = []
        history['classification_weight'] = []
    
    print(f"Training {model_name}...")
    
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
        
            # Compute loss
            if use_gradnorm:
                total_loss, loss_info = loss_fn(detection_outputs, classification_outputs,
                                              detection_targets, classification_targets)
                epoch_detection_loss += loss_info['detection_loss']
                epoch_classification_loss += loss_info['classification_loss']
            else:
                detection_loss = detection_loss_fn(detection_outputs, detection_targets)
                classification_loss = classification_loss_fn(classification_outputs, classification_targets)
                total_loss = detection_loss + 0.01 * classification_loss
                epoch_detection_loss += detection_loss.item()
                epoch_classification_loss += classification_loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
            epoch_loss += total_loss.item()
        
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
                if use_gradnorm:
                    val_total_loss, val_loss_info = loss_fn(detection_outputs, classification_outputs,
                                                          detection_targets, classification_targets)
                else:
                    val_detection_loss = detection_loss_fn(detection_outputs, detection_targets)
                    val_classification_loss = classification_loss_fn(classification_outputs, classification_targets)
                    val_total_loss = val_detection_loss + 0.01 * val_classification_loss
                
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
        history['detection_acc'].append(detection_correct / total_samples)
        history['classification_acc'].append(classification_correct / total_samples)
        
        if use_gradnorm:
            history['detection_weight'].append(loss_fn.detection_weight)
            history['classification_weight'].append(loss_fn.classification_weight)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val Loss: {history['val_loss'][-1]:.4f}, "
              f"D_Acc: {history['detection_acc'][-1]:.4f}, "
              f"C_Acc: {history['classification_acc'][-1]:.4f}")
        
    return model, history

def plot_comparison_results(results, save_dir='quick_test_results'):
    """Plot comparison results"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot training losses
    for model_name, history in results.items():
        axes[0, 0].plot(history['train_loss'], label=f'{model_name} Train')
        axes[0, 0].plot(history['val_loss'], label=f'{model_name} Val', linestyle='--')
    
    axes[0, 0].set_title('Training and Validation Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot detection losses
    for model_name, history in results.items():
        axes[0, 1].plot(history['detection_loss'], label=f'{model_name} Detection')
    
    axes[0, 1].set_title('Detection Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot classification losses
    for model_name, history in results.items():
        axes[0, 2].plot(history['classification_loss'], label=f'{model_name} Classification')
    
    axes[0, 2].set_title('Classification Losses')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot detection accuracies
    for model_name, history in results.items():
        axes[1, 0].plot(history['detection_acc'], label=f'{model_name} Detection')
    
    axes[1, 0].set_title('Detection Accuracies')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot classification accuracies
    for model_name, history in results.items():
        axes[1, 1].plot(history['classification_acc'], label=f'{model_name} Classification')
    
    axes[1, 1].set_title('Classification Accuracies')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot GradNorm weights (if available)
    gradnorm_models = [name for name in results.keys() if 'GradNorm' in name]
    if gradnorm_models:
        for model_name in gradnorm_models:
            history = results[model_name]
            if 'detection_weight' in history:
                axes[1, 2].plot(history['detection_weight'], label=f'{model_name} Detection Weight')
                axes[1, 2].plot(history['classification_weight'], label=f'{model_name} Classification Weight', linestyle='--')
        
        axes[1, 2].set_title('GradNorm Task Weights')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'quick_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    with open(save_dir / 'quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

def main():
    """Main function for quick testing"""
    
    parser = argparse.ArgumentParser(description='Quick Test Solutions')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of synthetic samples')
    parser.add_argument('--device', default='auto', help='Device to use')
    parser.add_argument('--save-dir', type=str, default='quick_test_results', help='Save directory')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SimpleDataset(num_samples=args.num_samples)
    val_dataset = SimpleDataset(num_samples=args.num_samples // 4)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Test different solutions
    results = {}
    
    # 1. Baseline model (with gradient interference)
    print("\n" + "="*50)
    print("Testing Baseline Model (with gradient interference)")
    print("="*50)
    
    baseline_model = BaselineModel()
    baseline_model, baseline_history = train_model(
        baseline_model, train_loader, val_loader, 
        num_epochs=args.epochs, device=device, 
        use_gradnorm=False, model_name="Baseline"
    )
    results['Baseline'] = baseline_history
    
    # 2. GradNorm solution
    print("\n" + "="*50)
    print("Testing GradNorm Solution")
    print("="*50)
    
    gradnorm_model = GradNormModel()
    gradnorm_model, gradnorm_history = train_model(
        gradnorm_model, train_loader, val_loader, 
        num_epochs=args.epochs, device=device, 
        use_gradnorm=True, model_name="GradNorm"
    )
    results['GradNorm'] = gradnorm_history
    
    # 3. Dual Backbone solution
    print("\n" + "="*50)
    print("Testing Dual Backbone Solution")
    print("="*50)
    
    dual_backbone_model = DualBackboneModel()
    dual_backbone_model, dual_backbone_history = train_model(
        dual_backbone_model, train_loader, val_loader, 
        num_epochs=args.epochs, device=device, 
        use_gradnorm=False, model_name="DualBackbone"
    )
    results['DualBackbone'] = dual_backbone_history
    
    # Plot comparison results
    plot_comparison_results(results, args.save_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("QUICK TEST SUMMARY")
    print("="*50)
    
    for model_name, history in results.items():
        final_detection_acc = history['detection_acc'][-1]
        final_classification_acc = history['classification_acc'][-1]
        final_train_loss = history['train_loss'][-1]
        
        print(f"\n{model_name}:")
        print(f"  Final Detection Accuracy: {final_detection_acc:.4f}")
        print(f"  Final Classification Accuracy: {final_classification_acc:.4f}")
        print(f"  Final Training Loss: {final_train_loss:.4f}")
        
        if 'detection_weight' in history:
            final_detection_weight = history['detection_weight'][-1]
            final_classification_weight = history['classification_weight'][-1]
            print(f"  Final Detection Weight: {final_detection_weight:.4f}")
            print(f"  Final Classification Weight: {final_classification_weight:.4f}")
    
    print(f"\nResults saved to: {args.save_dir}")
    print("Quick test completed!")

if __name__ == "__main__":
    main() 