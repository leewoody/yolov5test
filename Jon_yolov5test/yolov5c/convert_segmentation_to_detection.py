#!/usr/bin/env python3
"""
Convert segmentation format to YOLO detection format while preserving classification labels
"""

import os
import numpy as np
from pathlib import Path
import cv2

def polygon_to_bbox(polygon_coords):
    """
    Convert polygon coordinates to bounding box (x_center, y_center, width, height)
    
    Args:
        polygon_coords: List of [x1, y1, x2, y2, x3, y3, x4, y4] coordinates
        
    Returns:
        bbox: [x_center, y_center, width, height] in normalized coordinates
    """
    # Reshape coordinates into pairs
    coords = np.array(polygon_coords).reshape(-1, 2)
    
    # Calculate bounding box
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    
    # Convert to center format
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_center, y_center, width, height]

def convert_label_file(input_path, output_path):
    """
    Convert a single label file from segmentation to detection format
    
    Args:
        input_path: Path to input segmentation label file
        output_path: Path to output detection label file
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print(f"Warning: {input_path} has insufficient lines")
        return
    
    # Parse segmentation line (first line)
    seg_line = lines[0].strip()
    if not seg_line:
        print(f"Warning: {input_path} has empty segmentation line")
        return
    
    seg_parts = seg_line.split()
    if len(seg_parts) < 9:  # class_id + 8 coordinates
        print(f"Warning: {input_path} has insufficient coordinates")
        return
    
    # Extract class_id and coordinates
    class_id = int(seg_parts[0])
    coords = [float(x) for x in seg_parts[1:]]
    
    # Convert polygon to bounding box
    bbox = polygon_to_bbox(coords)
    
    # Parse classification line (last line)
    cls_line = lines[-1].strip()
    if not cls_line:
        print(f"Warning: {input_path} has empty classification line")
        return
    
    cls_labels = [int(x) for x in cls_line.split()]
    
    # Write detection format
    with open(output_path, 'w') as f:
        # Write detection line: class_id x_center y_center width height
        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        # Write classification line
        f.write(" ".join(map(str, cls_labels)) + "\n")

def convert_dataset(input_dir, output_dir):
    """
    Convert entire dataset from segmentation to detection format
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        input_labels_dir = input_path / split / 'labels'
        input_images_dir = input_path / split / 'images'
        output_labels_dir = output_path / split / 'labels'
        output_images_dir = output_path / split / 'images'
        
        if not input_labels_dir.exists():
            print(f"Warning: {input_labels_dir} does not exist, skipping...")
            continue
        
        print(f"Processing {split} split...")
        
        # Convert label files
        label_files = list(input_labels_dir.glob('*.txt'))
        converted_count = 0
        
        for label_file in label_files:
            output_label_file = output_labels_dir / label_file.name
            
            try:
                convert_label_file(label_file, output_label_file)
                converted_count += 1
            except Exception as e:
                print(f"Error converting {label_file}: {e}")
        
        print(f"Converted {converted_count} label files for {split} split")
        
        # Copy image files
        if input_images_dir.exists():
            image_files = list(input_images_dir.glob('*.*'))
            copied_count = 0
            
            for image_file in image_files:
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    output_image_file = output_images_dir / image_file.name
                    try:
                        import shutil
                        shutil.copy2(image_file, output_image_file)
                        copied_count += 1
                    except Exception as e:
                        print(f"Error copying {image_file}: {e}")
            
            print(f"Copied {copied_count} image files for {split} split")
    
    # Create data.yaml file
    create_data_yaml(output_path, input_path)

def create_data_yaml(output_dir, input_dir):
    """
    Create data.yaml file for the converted dataset
    
    Args:
        output_dir: Output dataset directory
        input_dir: Input dataset directory (to read original names)
    """
    # Read original data.yaml to get class names
    input_yaml = input_dir / 'data.yaml'
    class_names = ['AR', 'MR', 'PR', 'TR']  # Default names
    
    if input_yaml.exists():
        import yaml
        with open(input_yaml, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                class_names = data['names']
    
    # Create new data.yaml
    output_yaml = output_dir / 'data.yaml'
    yaml_content = f"""# YOLO Detection Dataset
names:
{chr(10).join([f"- {name}" for name in class_names])}
nc: {len(class_names)}
train: {output_dir}/train/images
val: {output_dir}/valid/images
test: {output_dir}/test/images
"""
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created {output_yaml}")

def main():
    """Main function"""
    # Define paths
    input_dataset = "Regurgitation-YOLODataset-1new"
    output_dataset = "Regurgitation-YOLODataset-Detection"
    
    print(f"Converting dataset from {input_dataset} to {output_dataset}")
    print("This will convert segmentation format to detection format while preserving classification labels")
    
    # Check if input dataset exists
    if not Path(input_dataset).exists():
        print(f"Error: Input dataset {input_dataset} not found!")
        return
    
    # Convert dataset
    convert_dataset(input_dataset, output_dataset)
    
    print(f"\nConversion completed!")
    print(f"Output dataset saved to: {output_dataset}")
    print(f"Format: YOLO detection (class_id x_center y_center width height) + classification labels")

if __name__ == "__main__":
    main() 