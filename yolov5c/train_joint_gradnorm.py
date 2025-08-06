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
from models.common import YOLOv5WithClassification
from utils.datasets import LoadImagesAndLabels
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.loss import ComputeLoss
from utils.metrics import ap_per_class, box_iou
from utils.torch_utils import select_device, time_sync
from utils.plots import plot_results, plot_results_files

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='converted_dataset_fixed/data.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='auto', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='joint_gradnorm', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--gradnorm-alpha', type=float, default=1.5, help='GradNorm alpha parameter')
    parser.add_argument('--initial-detection-weight', type=float, default=1.0, help='Initial detection weight')
    parser.add_argument('--initial-classification-weight', type=float, default=0.01, help='Initial classification weight')
    parser.add_argument('--use-adaptive-gradnorm', action='store_true', help='Use adaptive GradNorm')
    parser.add_argument('--log-interval', type=int, default=20, help='Log interval for batches')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval for epochs')
    
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
        Args:
            predictions: (batch_size, num_classes) - logits
            targets: (batch_size, num_classes) - one-hot encoded targets
        """
        return self.bce_loss(predictions, targets.float())

def train_epoch(model, dataloader, optimizer, gradnorm_loss, device, epoch, opt):
    """Train one epoch with GradNorm"""
    model.train()
    
    # Initialize metrics
    total_loss = 0
    detection_losses = []
    classification_losses = []
    detection_weights = []
    classification_weights = []
    
    # Training loop
    for batch_i, (imgs, targets, paths, _) in enumerate(dataloader):
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get model outputs
        if hasattr(model, 'classification_enabled') and model.classification_enabled:
            detection_outputs, classification_outputs = model(imgs)
        else:
            detection_outputs = model(imgs)
            classification_outputs = None
        
        # Prepare targets
        detection_targets = targets
        classification_targets = torch.zeros(imgs.size(0), 4, device=device)
        
        # Create one-hot classification targets from detection targets
        for i, target in enumerate(targets):
            if len(target) > 0:
                # Get unique classes in this image
                classes = target[:, 0].long()
                for cls in classes:
                    if cls < 4:  # Ensure class index is valid
                        classification_targets[i, cls] = 1.0
        
        # Compute loss with GradNorm
        if classification_outputs is not None:
            total_loss, loss_info = gradnorm_loss(
                detection_outputs, classification_outputs,
                detection_targets, classification_targets
            )
        else:
            # Fallback to detection only
            total_loss = gradnorm_loss.detection_loss_fn(detection_outputs, detection_targets)
            loss_info = {
                'total_loss': total_loss.item(),
                'detection_loss': total_loss.item(),
                'classification_loss': 0.0,
                'detection_weight': 1.0,
                'classification_weight': 0.0
            }
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store metrics
        total_loss += loss_info['total_loss']
        detection_losses.append(loss_info['detection_loss'])
        classification_losses.append(loss_info['classification_loss'])
        detection_weights.append(loss_info['detection_weight'])
        classification_weights.append(loss_info['classification_weight'])
        
        # Log progress
        if batch_i % opt.log_interval == 0:
            print(f'Epoch {epoch}, Batch {batch_i}/{len(dataloader)}, '
                  f'Loss: {loss_info["total_loss"]:.4f} '
                  f'(Det: {loss_info["detection_loss"]:.4f}, '
                  f'Cls: {loss_info["classification_loss"]:.4f}), '
                  f'Weights: Det={loss_info["detection_weight"]:.3f}, '
                  f'Cls={loss_info["classification_weight"]:.3f}')
    
    # Return epoch metrics
    return {
        'total_loss': total_loss / len(dataloader),
        'detection_loss': np.mean(detection_losses),
        'classification_loss': np.mean(classification_losses),
        'detection_weight': np.mean(detection_weights),
        'classification_weight': np.mean(classification_weights)
    }

def validate_epoch(model, dataloader, device, epoch):
    """Validate one epoch"""
    model.eval()
    
    # Initialize metrics
    detection_losses = []
    classification_losses = []
    classification_accuracies = []
    
    with torch.no_grad():
        for batch_i, (imgs, targets, paths, _) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            
            # Forward pass
            if hasattr(model, 'classification_enabled') and model.classification_enabled:
                detection_outputs, classification_outputs = model(imgs)
            else:
                detection_outputs = model(imgs)
                classification_outputs = None
            
            # Prepare targets
            detection_targets = targets
            classification_targets = torch.zeros(imgs.size(0), 4, device=device)
            
            # Create one-hot classification targets
            for i, target in enumerate(targets):
                if len(target) > 0:
                    classes = target[:, 0].long()
                    for cls in classes:
                        if cls < 4:
                            classification_targets[i, cls] = 1.0
            
            # Compute detection loss
            detection_loss_fn = nn.CrossEntropyLoss()
            detection_loss = detection_loss_fn(detection_outputs[0].view(-1, 4), detection_targets[:, 0].long())
            detection_losses.append(detection_loss.item())
            
            # Compute classification metrics
            if classification_outputs is not None:
                classification_loss_fn = nn.BCEWithLogitsLoss()
                classification_loss = classification_loss_fn(classification_outputs, classification_targets)
                classification_losses.append(classification_loss.item())
                
                # Compute accuracy
                predictions = torch.sigmoid(classification_outputs) > 0.5
                accuracy = (predictions == classification_targets.bool()).float().mean()
                classification_accuracies.append(accuracy.item())
    
    return {
        'detection_loss': np.mean(detection_losses),
        'classification_loss': np.mean(classification_losses) if classification_losses else 0.0,
        'classification_accuracy': np.mean(classification_accuracies) if classification_accuracies else 0.0
    }

def plot_training_curves(history, save_dir):
    """Plot training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Total Loss')
    ax1.plot(epochs, history['train_detection_loss'], 'r-', label='Detection Loss')
    ax1.plot(epochs, history['train_classification_loss'], 'g-', label='Classification Loss')
    ax1.set_title('Training Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Validation metrics
    ax2.plot(epochs, history['val_detection_loss'], 'r-', label='Detection Loss')
    ax2.plot(epochs, history['val_classification_loss'], 'g-', label='Classification Loss')
    ax2.plot(epochs, history['val_classification_accuracy'], 'b-', label='Classification Accuracy')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric')
    ax2.legend()
    ax2.grid(True)
    
    # Task weights
    ax3.plot(epochs, history['detection_weight'], 'r-', label='Detection Weight')
    ax3.plot(epochs, history['classification_weight'], 'g-', label='Classification Weight')
    ax3.set_title('GradNorm Task Weights')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Weight')
    ax3.legend()
    ax3.grid(True)
    
    # Loss ratio
    loss_ratio = [cls/det if det > 0 else 0 for det, cls in zip(history['train_detection_loss'], history['train_classification_loss'])]
    ax4.plot(epochs, loss_ratio, 'purple', label='Classification/Detection Loss Ratio')
    ax4.set_title('Loss Ratio')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Ratio')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradnorm_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main(opt):
    # Initialize
    device = select_device(opt.device)
    
    # Create save directory
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=opt.exist_ok)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting GradNorm joint training with parameters: {opt}")
    
    # Load data
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)
    
    train_path = data_dict['train']
    val_path = data_dict['val']
    nc = data_dict['nc']
    
    # Create datasets
    train_dataset = LoadImagesAndLabels(
        train_path, opt.imgsz, opt.batch_size,
        augment=True, hyp=None, rect=False, cache_images=False,
        single_cls=False, stride=32, pad=0.0, prefix='train: '
    )
    
    val_dataset = LoadImagesAndLabels(
        val_path, opt.imgsz, opt.batch_size,
        augment=False, hyp=None, rect=True, cache_images=False,
        single_cls=False, stride=32, pad=0.5, prefix='val: '
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, pin_memory=True, collate_fn=LoadImagesAndLabels.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True, collate_fn=LoadImagesAndLabels.collate_fn
    )
    
    # Load model
    model = load_model(opt.weights, device)
    model.to(device)
    
    # Create loss functions
    detection_loss_fn = ComputeLoss(model)
    classification_loss_fn = ClassificationLoss(num_classes=4)
    
    # Create GradNorm loss
    gradnorm_loss = create_gradnorm_loss(detection_loss_fn, classification_loss_fn, opt)
    gradnorm_loss.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    
    # Training history
    history = {
        'train_loss': [], 'train_detection_loss': [], 'train_classification_loss': [],
        'val_detection_loss': [], 'val_classification_loss': [], 'val_classification_accuracy': [],
        'detection_weight': [], 'classification_weight': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(opt.epochs):
        logging.info(f"Starting epoch {epoch + 1}/{opt.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, gradnorm_loss, device, epoch, opt)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Store history
        history['train_loss'].append(train_metrics['total_loss'])
        history['train_detection_loss'].append(train_metrics['detection_loss'])
        history['train_classification_loss'].append(train_metrics['classification_loss'])
        history['val_detection_loss'].append(val_metrics['detection_loss'])
        history['val_classification_loss'].append(val_metrics['classification_loss'])
        history['val_classification_accuracy'].append(val_metrics['classification_accuracy'])
        history['detection_weight'].append(train_metrics['detection_weight'])
        history['classification_weight'].append(train_metrics['classification_weight'])
        
        # Log epoch results
        logging.info(f"Epoch {epoch + 1} Results:")
        logging.info(f"  Train - Total: {train_metrics['total_loss']:.4f}, "
                    f"Det: {train_metrics['detection_loss']:.4f}, "
                    f"Cls: {train_metrics['classification_loss']:.4f}")
        logging.info(f"  Val - Det: {val_metrics['detection_loss']:.4f}, "
                    f"Cls: {val_metrics['classification_loss']:.4f}, "
                    f"Acc: {val_metrics['classification_accuracy']:.4f}")
        logging.info(f"  Weights - Det: {train_metrics['detection_weight']:.3f}, "
                    f"Cls: {train_metrics['classification_weight']:.3f}")
        
        # Save best model
        if val_metrics['detection_loss'] < best_val_loss:
            best_val_loss = val_metrics['detection_loss']
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            logging.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % opt.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'opt': opt
            }, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), save_dir / 'final_model.pth')
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, save_dir)
    
    # Save training summary
    summary = {
        'final_train_loss': history['train_loss'][-1],
        'final_val_detection_loss': history['val_detection_loss'][-1],
        'final_val_classification_accuracy': history['val_classification_accuracy'][-1],
        'best_val_loss': best_val_loss,
        'final_detection_weight': history['detection_weight'][-1],
        'final_classification_weight': history['classification_weight'][-1],
        'training_parameters': vars(opt)
    }
    
    with open(save_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Training completed. Results saved to {save_dir}")
    logging.info(f"Final validation detection loss: {history['val_detection_loss'][-1]:.4f}")
    logging.info(f"Final validation classification accuracy: {history['val_classification_accuracy'][-1]:.4f}")

if __name__ == '__main__':
    opt = parse_opt()
    main(opt) 