#!/usr/bin/env python3
"""
Convert segmentation format to YOLOv5 bbox format (FIXED VERSION)
Input format:
- Line 1: segmentation coordinates (class x1 y1 x2 y2 x3 y3 ...)
- Line 2: classification labels (e.g., "0 1 0")

Output format:
- Line 1: YOLOv5 bbox (class x_center y_center width height)
- Line 2: classification label (single digit, e.g., "1")
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path

def convert_segmentation_to_bbox(segmentation_coords):
    """Convert segmentation coordinates to bounding box"""
    if len(segmentation_coords) < 6:  # Need at least 3 points (6 coordinates)
        return None
    
    # Extract x and y coordinates
    x_coords = segmentation_coords[::2]  # Even indices: 0, 2, 4, ...
    y_coords = segmentation_coords[1::2]  # Odd indices: 1, 3, 5, ...
    
    # Calculate bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Convert to YOLO format (center_x, center_y, width, height)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return center_x, center_y, width, height

def convert_label_file(input_file, output_file):
    """Convert a single label file from segmentation to bbox format"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Find non-empty lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 2:
            print(f"警告: 文件 {input_file} 有效行數不足")
            return False
        
        # Parse segmentation line (first non-empty line)
        seg_line = non_empty_lines[0].split()
        if len(seg_line) < 5:  # Need at least class_id + 2 points (4 coordinates)
            print(f"警告: 分割線格式錯誤 {input_file}")
            return False
        
        class_id = int(seg_line[0])
        segmentation_coords = [float(x) for x in seg_line[1:]]
        
        # Convert to bbox
        bbox = convert_segmentation_to_bbox(segmentation_coords)
        if bbox is None:
            print(f"警告: 無法轉換分割座標 {input_file}")
            return False
        
        center_x, center_y, width, height = bbox
        
        # Parse classification line (last non-empty line) - keep original one-hot encoding
        cls_label = non_empty_lines[-1]
        
        # Write output in YOLOv5 format with classification
        with open(output_file, 'w') as f:
            # Write bbox line (class x_center y_center width height)
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            # Write classification line (keep original one-hot encoding)
            f.write(f"{cls_label}\n")
        
        return True
        
    except Exception as e:
        print(f"錯誤: 處理文件 {input_file}: {e}")
        return False

def convert_dataset(input_dir, output_dir):
    """Convert entire dataset"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy images and convert labels
    for split in ['train', 'valid', 'test']:
        input_images_dir = input_path / split / 'images'
        input_labels_dir = input_path / split / 'labels'
        output_images_dir = output_path / split / 'images'
        output_labels_dir = output_path / split / 'labels'
        
        if not input_images_dir.exists() or not input_labels_dir.exists():
            print(f"警告: {split} 目錄不存在，跳過")
            continue
        
        # Copy images
        for img_file in input_images_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                shutil.copy2(img_file, output_images_dir / img_file.name)
        
        # Convert labels
        converted_count = 0
        total_count = 0
        
        for label_file in input_labels_dir.glob('*.txt'):
            output_label_file = output_labels_dir / label_file.name
            if convert_label_file(label_file, output_label_file):
                converted_count += 1
            total_count += 1
        
        print(f"{split}: 轉換了 {converted_count}/{total_count} 個標籤文件")
    
    # Create data.yaml
    create_data_yaml(output_path, input_path)
    
    print(f"數據集轉換完成: {output_dir}")

def create_data_yaml(output_path, input_path):
    """Create data.yaml file"""
    # Read original data.yaml to get class names
    original_yaml = input_path / 'data.yaml'
    class_names = ['AR', 'MR', 'PR', 'TR']  # Default class names
    nc = 4  # Default number of classes
    
    if original_yaml.exists():
        try:
            with open(original_yaml, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip().startswith('names:'):
                        # Extract class names from the original yaml
                        class_names = []
                        for i, l in enumerate(lines[lines.index(line)+1:], 1):
                            if l.strip().startswith('-'):
                                class_names.append(l.strip()[2:])
                            elif l.strip() and not l.strip().startswith('#'):
                                break
                        break
                    elif line.strip().startswith('nc:'):
                        nc = int(line.strip().split(':')[1])
        except Exception as e:
            print(f"警告: 無法讀取原始 data.yaml: {e}")
    
    # Create new data.yaml
    yaml_content = f"""names:
{chr(10).join([f"- {name}" for name in class_names])}
nc: {nc}
path: {output_path.absolute()}
test: test/images
train: train/images
val: val/images
"""
    
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"創建 data.yaml: {output_path / 'data.yaml'}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python convert_segmentation_to_bbox_fixed.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_dir):
        print(f"錯誤: 輸入目錄不存在: {input_dir}")
        sys.exit(1)
    
    convert_dataset(input_dir, output_dir) 