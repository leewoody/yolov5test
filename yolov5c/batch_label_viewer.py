#!/usr/bin/env python3
"""
Batch Label Viewer Tool for YOLOv5 Dataset
Quickly view multiple samples from the dataset
"""

import os
import sys
import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional

def load_data_yaml(yaml_path: str) -> Dict:
    """Load data.yaml file"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def parse_segmentation_line(line: str) -> List[Tuple[float, float]]:
    """Parse segmentation line to coordinate list"""
    parts = line.strip().split()
    if len(parts) < 3:
        return []
    
    class_id = int(parts[0])
    coordinates = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            x = float(parts[i])
            y = float(parts[i + 1])
            coordinates.append((x, y))
    
    return coordinates

def parse_classification_line(line: str) -> List[int]:
    """Parse classification line to one-hot encoding"""
    parts = line.strip().split()
    return [int(x) for x in parts]

def load_label_file(label_path: str) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Load label file, return segmentation coordinates and classification labels"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter empty lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 2:
            return [], []
        
        # First line is segmentation coordinates
        seg_coordinates = parse_segmentation_line(non_empty_lines[0])
        
        # Last line is classification labels
        cls_labels = parse_classification_line(non_empty_lines[-1])
        
        return seg_coordinates, cls_labels
    
    except Exception as e:
        return [], []

def segmentation_to_bbox(seg_coordinates: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Convert segmentation coordinates to bounding box (x_center, y_center, width, height)"""
    if not seg_coordinates:
        return 0.0, 0.0, 0.0, 0.0
    
    # Convert to numpy array
    coords = np.array(seg_coordinates)
    
    # Calculate bounding box
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    
    # Calculate center point and width/height
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def draw_segmentation(image: np.ndarray, seg_coordinates: List[Tuple[float, float]], 
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw segmentation contour on image"""
    if not seg_coordinates:
        return image
    
    # Convert coordinates to pixel coordinates
    h, w = image.shape[:2]
    pixel_coords = []
    for x, y in seg_coordinates:
        px = int(x * w)
        py = int(y * h)
        pixel_coords.append([px, py])
    
    pixel_coords = np.array(pixel_coords, dtype=np.int32)
    
    # Draw contour
    cv2.polylines(image, [pixel_coords], True, color, thickness)
    
    # Draw vertices
    for px, py in pixel_coords:
        cv2.circle(image, (px, py), 2, color, -1)
    
    return image

def draw_bbox(image: np.ndarray, bbox: Tuple[float, float, float, float], 
              color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    """Draw bounding box on image"""
    x_center, y_center, width, height = bbox
    
    h, w = image.shape[:2]
    
    # Convert to pixel coordinates
    x_center_px = int(x_center * w)
    y_center_px = int(y_center * h)
    width_px = int(width * w)
    height_px = int(height * h)
    
    # Calculate top-left and bottom-right corners
    x1 = x_center_px - width_px // 2
    y1 = y_center_px - height_px // 2
    x2 = x_center_px + width_px // 2
    y2 = y_center_px + height_px // 2
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw center point
    cv2.circle(image, (x_center_px, y_center_px), 2, color, -1)
    
    return image

def visualize_batch_samples(dataset_dir: str, data_yaml_path: str, num_samples: int = 8, 
                           save_path: Optional[str] = None) -> None:
    """Visualize multiple samples in a grid layout"""
    
    # Load data configuration
    data_config = load_data_yaml(data_yaml_path)
    class_names = data_config.get('names', [])
    
    print(f"Dataset: {dataset_dir}")
    print(f"Classes: {class_names}")
    print(f"Visualizing {num_samples} samples...")
    
    # Find all image files
    all_image_files = []
    all_label_files = []
    
    for split in ['train', 'valid', 'test']:
        images_dir = Path(dataset_dir) / split / 'images'
        labels_dir = Path(dataset_dir) / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        # Get image files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        for image_file in image_files:
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            if label_file.exists():
                all_image_files.append(image_file)
                all_label_files.append(label_file)
    
    if not all_image_files:
        print("No valid image-label pairs found!")
        return
    
    # Randomly select samples
    if len(all_image_files) > num_samples:
        indices = np.random.choice(len(all_image_files), num_samples, replace=False)
        selected_images = [all_image_files[i] for i in indices]
        selected_labels = [all_label_files[i] for i in indices]
    else:
        selected_images = all_image_files
        selected_labels = all_label_files
    
    # Calculate grid layout
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each sample
    for idx, (image_file, label_file) in enumerate(zip(selected_images, selected_labels)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label
        seg_coordinates, cls_labels = load_label_file(str(label_file))
        
        # Create image copy for drawing
        image_draw = image_rgb.copy()
        
        # Draw annotations
        if seg_coordinates:
            # Draw segmentation contour
            image_draw = draw_segmentation(image_draw, seg_coordinates, color=(0, 255, 0), thickness=1)
            
            # Draw bounding box
            bbox = segmentation_to_bbox(seg_coordinates)
            image_draw = draw_bbox(image_draw, bbox, color=(255, 0, 0), thickness=1)
        
        # Display image
        ax.imshow(image_draw)
        
        # Add title with class information
        title = f"{image_file.stem[:20]}..."
        if cls_labels and len(cls_labels) <= len(class_names):
            active_classes = [class_names[i] for i, label in enumerate(cls_labels) if label == 1]
            if active_classes:
                title += f"\n{active_classes[0]}"
        
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(selected_images), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Batch visualization saved to: {save_path}")
    
    plt.show()

def analyze_dataset_distribution(dataset_dir: str, data_yaml_path: str) -> None:
    """Analyze and display dataset distribution"""
    
    # Load data configuration
    data_config = load_data_yaml(data_yaml_path)
    class_names = data_config.get('names', [])
    
    print(f"Dataset Analysis: {dataset_dir}")
    print(f"Classes: {class_names}")
    print("-" * 50)
    
    # Analyze each split
    for split in ['train', 'valid', 'test']:
        images_dir = Path(dataset_dir) / split / 'images'
        labels_dir = Path(dataset_dir) / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"{split}: Directory not found")
            continue
        
        # Count files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))
        
        print(f"{split.upper()}:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        
        # Analyze class distribution
        class_counts = {i: 0 for i in range(len(class_names))}
        valid_pairs = 0
        
        for image_file in image_files:
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            if label_file.exists():
                valid_pairs += 1
                seg_coordinates, cls_labels = load_label_file(str(label_file))
                
                if cls_labels:
                    for i, label in enumerate(cls_labels):
                        if i < len(class_names) and label == 1:
                            class_counts[i] += 1
        
        print(f"  Valid pairs: {valid_pairs}")
        print(f"  Class distribution:")
        for i, class_name in enumerate(class_names):
            print(f"    {class_name}: {class_counts[i]}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Batch YOLOv5 Label Viewer')
    parser.add_argument('--data', type=str, required=True, help='data.yaml file path')
    parser.add_argument('--dataset-dir', type=str, help='Dataset directory path')
    parser.add_argument('--samples', type=int, default=8, help='Number of samples to visualize')
    parser.add_argument('--save', type=str, help='Save visualization to file')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset distribution')
    
    args = parser.parse_args()
    
    # Get dataset directory
    data_config = load_data_yaml(args.data)
    dataset_dir = data_config.get('path', '')
    
    if not dataset_dir:
        if args.dataset_dir:
            dataset_dir = args.dataset_dir
        else:
            # Infer from data.yaml file path
            yaml_path = Path(args.data)
            dataset_dir = str(yaml_path.parent)
    
    if not dataset_dir:
        print("Error: Cannot determine dataset path, please use --dataset-dir parameter")
        return
    
    print(f"Using dataset directory: {dataset_dir}")
    
    if args.analyze:
        analyze_dataset_distribution(dataset_dir, args.data)
    else:
        visualize_batch_samples(dataset_dir, args.data, args.samples, args.save)

if __name__ == '__main__':
    main() 