#!/usr/bin/env python3
"""
Convert Joint Training Format to Detection Only Format
將聯合訓練格式轉換為純檢測格式

This script converts the joint training format (detection + classification) 
to detection-only format for baseline training.
"""

import os
import shutil
from pathlib import Path
import argparse
import yaml

def convert_labels_to_detection_only(input_labels_dir, output_labels_dir):
    """Convert joint training labels to detection-only format"""
    
    # Create output directory
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Process each label file
    for label_file in os.listdir(input_labels_dir):
        if not label_file.endswith('.txt'):
            continue
            
        input_path = os.path.join(input_labels_dir, label_file)
        output_path = os.path.join(output_labels_dir, label_file)
        
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        # Extract detection line (first line) and convert to YOLO format
        if len(lines) > 0:
            detection_line = lines[0].strip()
            parts = detection_line.split()
            
            if len(parts) >= 9:  # Segmentation format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Convert polygon to bounding box
                x_coords = coords[::2]  # x coordinates
                y_coords = coords[1::2]  # y coordinates
                
                # Calculate bounding box
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Convert to YOLO format (center_x, center_y, width, height)
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                # Write YOLO format
                yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                with open(output_path, 'w') as f:
                    f.write(yolo_line + '\n')
            else:
                # If not enough coordinates, skip this file
                print(f"Warning: Skipping {label_file} - insufficient coordinates")
                continue

def create_detection_only_dataset(input_dataset_dir, output_dataset_dir):
    """Create detection-only dataset from joint training dataset"""
    
    # Create output directory structure
    for split in ['train', 'valid', 'test']:
        input_split_dir = os.path.join(input_dataset_dir, split)
        output_split_dir = os.path.join(output_dataset_dir, split)
        
        if not os.path.exists(input_split_dir):
            continue
            
        # Create directories
        os.makedirs(os.path.join(output_split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_split_dir, 'labels'), exist_ok=True)
        
        # Copy images
        input_images_dir = os.path.join(input_split_dir, 'images')
        output_images_dir = os.path.join(output_split_dir, 'images')
        
        if os.path.exists(input_images_dir):
            for img_file in os.listdir(input_images_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    src = os.path.join(input_images_dir, img_file)
                    dst = os.path.join(output_images_dir, img_file)
                    shutil.copy2(src, dst)
        
        # Convert labels
        input_labels_dir = os.path.join(input_split_dir, 'labels')
        output_labels_dir = os.path.join(output_split_dir, 'labels')
        
        if os.path.exists(input_labels_dir):
            convert_labels_to_detection_only(input_labels_dir, output_labels_dir)
    
    # Create data.yaml
    input_yaml_path = os.path.join(input_dataset_dir, 'data.yaml')
    output_yaml_path = os.path.join(output_dataset_dir, 'data.yaml')
    
    if os.path.exists(input_yaml_path):
        with open(input_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths for detection-only dataset
        data_config['train'] = os.path.join(output_dataset_dir, 'train/images')
        data_config['val'] = os.path.join(output_dataset_dir, 'valid/images')
        data_config['test'] = os.path.join(output_dataset_dir, 'test/images')
        
        with open(output_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description='Convert joint training dataset to detection-only format')
    parser.add_argument('--input', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output dataset directory')
    
    args = parser.parse_args()
    
    print(f"Converting {args.input} to detection-only format...")
    print(f"Output directory: {args.output}")
    
    create_detection_only_dataset(args.input, args.output)
    
    print("Conversion completed!")
    print(f"Detection-only dataset saved to: {args.output}")

if __name__ == "__main__":
    main() 