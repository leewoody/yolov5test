#!/usr/bin/env python3
"""
Test script to verify loss function fix
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.yolo import Model
from utils.loss import ComputeLoss
from utils.dataloaders import create_dataloader
from utils.general import check_dataset

def test_loss_function():
    """Test the loss function with a simple case"""
    print("Testing loss function...")
    
    # Load data
    data_yaml = '../regurgitation_detection_only_20250730_100829/data.yaml'
    data_dict = check_dataset(data_yaml)
    
    # Create model
    model = Model('models/yolov5s.yaml', ch=3, nc=4)  # 4 classes for regurgitation
    model.eval()
    
    # Add hyp attribute to model
    model.hyp = {
        'box': 0.05,
        'cls': 0.5,
        'cls_pw': 1.0,
        'obj': 1.0,
        'obj_pw': 1.0,
        'iou_t': 0.20,
        'anchor_t': 4.0,
        'fl_gamma': 0.0
    }
    
    # Create loss function
    compute_loss = ComputeLoss(model)
    
    # Create a simple dataloader
    train_loader, dataset = create_dataloader(
        data_dict['train'],
        imgsz=640,
        batch_size=2,
        stride=32,
        augment=False,
        hyp={},
        rect=False,
        rank=-1,
        workers=0,
        prefix='test: '
    )
    
    print(f"Dataset has {len(dataset)} samples")
    
    # Get one batch
    for batch_i, (imgs, targets, paths, shapes) in enumerate(train_loader):
        print(f"Batch {batch_i}:")
        print(f"  imgs.shape: {imgs.shape}")
        print(f"  targets.shape: {targets.shape}")
        print(f"  targets: {targets}")
        
        # Forward pass
        with torch.no_grad():
            # Ensure correct data type
            imgs = imgs.float()
            pred = model(imgs)
            print(f"  pred type: {type(pred)}")
            print(f"  pred length: {len(pred)}")
            for i, p in enumerate(pred):
                print(f"  pred[{i}].shape: {p.shape}")
            
            # Test loss computation
            try:
                loss, loss_items = compute_loss(pred, targets)
                print(f"  Loss computation successful!")
                print(f"  Total loss: {loss.item():.4f}")
                print(f"  Loss items: {loss_items}")
                break
            except Exception as e:
                print(f"  Loss computation failed: {e}")
                import traceback
                traceback.print_exc()
                break
        
        if batch_i >= 0:  # Only test first batch
            break
    
    print("Test completed!")

if __name__ == '__main__':
    test_loss_function() 