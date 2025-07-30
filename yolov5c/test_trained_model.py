#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for trained YOLOv5 model
"""

import argparse
import os
import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

def test_model(model_path, data_yaml, test_images_path=None, conf_thres=0.25, iou_thres=0.45):
    """Test the trained model"""
    
    # Initialize
    device = select_device('')
    
    # Load model
    if model_path.endswith('.pt'):
        # Load custom trained model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint
    else:
        # Load pre-trained model
        model = DetectMultiBackend(model_path, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)
    imgsz = imgsz[0]  # Use single value for dataloader
    
    # Load data
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    class_names = data_dict['names']
    print(f"Class names: {class_names}")
    
    # Create test dataloader
    test_loader = create_dataloader(
        data_dict['test'], imgsz, batch_size=1, stride=stride, 
        rect=False, workers=8, augment=False, cache=None
    )
    
    # Test the model
    model.eval()
    total_detections = 0
    total_images = 0
    
    print("Testing model...")
    for batch_i, batch in enumerate(test_loader):
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                imgs, targets, paths = batch[0], batch[1], batch[2]
            else:
                continue
        else:
            continue
        
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        
        # Inference
        with torch.no_grad():
            pred = model(imgs)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)
        
        # Process detections
        for i, det in enumerate(pred):
            total_images += 1
            if det is not None and len(det):
                total_detections += len(det)
                print(f"Image {total_images}: {len(det)} detections")
                
                # Print detection details
                for *xyxy, conf, cls in det:
                    class_id = int(cls)
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                    print(f"  - {class_name}: {conf:.3f}")
            else:
                print(f"Image {total_images}: No detections")
        
        # Only test first 10 images for quick evaluation
        if total_images >= 10:
            break
    
    print(f"\nTest Results:")
    print(f"Total images tested: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/total_images:.2f}")
    
    return total_detections, total_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--data', type=str, required=True, help='data.yaml path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    
    opt = parser.parse_args()
    test_model(opt.model, opt.data, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres) 