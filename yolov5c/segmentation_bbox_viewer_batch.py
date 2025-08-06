#!/usr/bin/env python3
"""
Batch Segmentation and Bbox Comparison Viewer (No Display)
Generate comparison images without showing windows
"""

import os
import sys
import argparse
import yaml
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

def load_original_label(label_path: str) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Load original label file (segmentation + classification)"""
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

def load_converted_label(label_path: str) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
    """Load converted label file (bbox + classification)"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        bboxes = []
        cls_labels = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                # Bbox format: class_id x_center y_center width height
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                bboxes.append((x_center, y_center, width, height))
            elif len(parts) >= 3:
                # Classification format: one-hot encoding
                cls_labels = [int(x) for x in parts]
        
        return bboxes, cls_labels
    
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
        cv2.circle(image, (px, py), 3, color, -1)
    
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
    cv2.circle(image, (x_center_px, y_center_px), 3, color, -1)
    
    return image

def save_comparison_image(image_path: str, original_label_path: str, converted_label_path: str, 
                         class_names: List[str], save_path: str) -> bool:
    """Save comparison image without displaying"""
    
    try:
        # Load image
        if not os.path.exists(image_path):
            print(f"Error: Image file does not exist {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image {image_path}")
            return False
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load original label (segmentation + classification)
        seg_coordinates, original_cls_labels = load_original_label(original_label_path)
        
        # Load converted label (bbox + classification)
        converted_bboxes, converted_cls_labels = load_converted_label(converted_label_path)
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Original segmentation
        image_seg = image_rgb.copy()
        if seg_coordinates:
            image_seg = draw_segmentation(image_seg, seg_coordinates, color=(0, 255, 0), thickness=2)
        axes[0, 1].imshow(image_seg)
        axes[0, 1].set_title('Original Segmentation (Green)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Converted bbox
        image_bbox = image_rgb.copy()
        if converted_bboxes:
            for bbox in converted_bboxes:
                image_bbox = draw_bbox(image_bbox, bbox, color=(255, 0, 0), thickness=2)
        axes[1, 0].imshow(image_bbox)
        axes[1, 0].set_title('Converted Bounding Box (Red)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Overlay comparison
        image_overlay = image_rgb.copy()
        if seg_coordinates:
            image_overlay = draw_segmentation(image_overlay, seg_coordinates, color=(0, 255, 0), thickness=2)
        if converted_bboxes:
            for bbox in converted_bboxes:
                image_overlay = draw_bbox(image_overlay, bbox, color=(255, 0, 0), thickness=2)
        axes[1, 1].imshow(image_overlay)
        axes[1, 1].set_title('Overlay Comparison (Green: Seg, Red: Bbox)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Add detailed information
        info_text = f"Image: {os.path.basename(image_path)}\n"
        info_text += f"Original Label: {os.path.basename(original_label_path)}\n"
        info_text += f"Converted Label: {os.path.basename(converted_label_path)}\n\n"
        
        if seg_coordinates:
            info_text += f"Segmentation Points: {len(seg_coordinates)}\n"
            calculated_bbox = segmentation_to_bbox(seg_coordinates)
            info_text += f"Calculated Bbox: ({calculated_bbox[0]:.3f}, {calculated_bbox[1]:.3f}, {calculated_bbox[2]:.3f}, {calculated_bbox[3]:.3f})\n"
        
        if converted_bboxes:
            info_text += f"Converted Bboxes: {len(converted_bboxes)}\n"
            for i, bbox in enumerate(converted_bboxes):
                info_text += f"  Bbox {i+1}: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})\n"
        
        if original_cls_labels:
            info_text += f"Original Classification: {original_cls_labels}\n"
            if len(original_cls_labels) <= len(class_names):
                active_classes = [class_names[i] for i, label in enumerate(original_cls_labels) if label == 1]
                info_text += f"Classes: {', '.join(active_classes) if active_classes else 'None'}\n"
        
        if converted_cls_labels:
            info_text += f"Converted Classification: {converted_cls_labels}\n"
            if len(converted_cls_labels) <= len(class_names):
                active_classes = [class_names[i] for i, label in enumerate(converted_cls_labels) if label == 1]
                info_text += f"Classes: {', '.join(active_classes) if active_classes else 'None'}\n"
        
        # Add text information to figure
        fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save image
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def generate_all_comparison_images(original_dataset_dir: str, converted_dataset_dir: str, 
                                 data_yaml_path: str, save_dir: str) -> None:
    """Generate comparison images for all samples"""
    
    # Load data configuration
    data_config = load_data_yaml(data_yaml_path)
    class_names = data_config.get('names', [])
    
    print(f"Original Dataset: {original_dataset_dir}")
    print(f"Converted Dataset: {converted_dataset_dir}")
    print(f"Classes: {class_names}")
    print(f"Save Directory: {save_dir}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Find all image files
    all_image_files = []
    all_original_labels = []
    all_converted_labels = []
    
    for split in ['train', 'valid', 'test']:
        original_images_dir = Path(original_dataset_dir) / split / 'images'
        original_labels_dir = Path(original_dataset_dir) / split / 'labels'
        converted_images_dir = Path(converted_dataset_dir) / split / 'images'
        converted_labels_dir = Path(converted_dataset_dir) / split / 'labels'
        
        if not all([original_images_dir.exists(), original_labels_dir.exists(), 
                   converted_images_dir.exists(), converted_labels_dir.exists()]):
            print(f"Skipping {split}: directories not found")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Get image files
        image_files = list(original_images_dir.glob('*.jpg')) + list(original_images_dir.glob('*.png'))
        
        for image_file in image_files:
            original_label_file = original_labels_dir / f"{image_file.stem}.txt"
            converted_label_file = converted_labels_dir / f"{image_file.stem}.txt"
            
            if original_label_file.exists() and converted_label_file.exists():
                all_image_files.append(image_file)
                all_original_labels.append(original_label_file)
                all_converted_labels.append(converted_label_file)
    
    if not all_image_files:
        print("No valid image-label pairs found!")
        return
    
    print(f"\nFound {len(all_image_files)} valid image-label pairs")
    print("Generating comparison images...")
    
    # Process each sample
    success_count = 0
    for idx, (image_file, original_label, converted_label) in enumerate(
        zip(all_image_files, all_original_labels, all_converted_labels)):
        
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(all_image_files)} ({idx/len(all_image_files)*100:.1f}%)")
        
        # Set save path
        save_path = Path(save_dir) / f"comparison_{image_file.stem}.png"
        
        # Generate comparison image
        if save_comparison_image(
            str(image_file), 
            str(original_label), 
            str(converted_label), 
            class_names, 
            str(save_path)
        ):
            success_count += 1
    
    print(f"\nCompleted! Generated {success_count}/{len(all_image_files)} comparison images")
    print(f"Images saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Batch Segmentation vs Bbox Comparison Generator')
    parser.add_argument('--original-data', type=str, required=True, help='Original data.yaml file path')
    parser.add_argument('--converted-data', type=str, required=True, help='Converted data.yaml file path')
    parser.add_argument('--original-dataset-dir', type=str, help='Original dataset directory path')
    parser.add_argument('--converted-dataset-dir', type=str, help='Converted dataset directory path')
    parser.add_argument('--save-dir', type=str, default='comparison_results', help='Save directory for comparison images')
    
    args = parser.parse_args()
    
    # Get dataset directories
    original_data_config = load_data_yaml(args.original_data)
    converted_data_config = load_data_yaml(args.converted_data)
    
    original_dataset_dir = original_data_config.get('path', '')
    converted_dataset_dir = converted_data_config.get('path', '')
    
    if not original_dataset_dir:
        if args.original_dataset_dir:
            original_dataset_dir = args.original_dataset_dir
        else:
            yaml_path = Path(args.original_data)
            original_dataset_dir = str(yaml_path.parent)
    
    if not converted_dataset_dir:
        if args.converted_dataset_dir:
            converted_dataset_dir = args.converted_dataset_dir
        else:
            yaml_path = Path(args.converted_data)
            converted_dataset_dir = str(yaml_path.parent)
    
    if not original_dataset_dir or not converted_dataset_dir:
        print("Error: Cannot determine dataset paths")
        return
    
    print(f"Original dataset directory: {original_dataset_dir}")
    print(f"Converted dataset directory: {converted_dataset_dir}")
    
    # Generate all comparison images
    generate_all_comparison_images(original_dataset_dir, converted_dataset_dir, 
                                 args.original_data, args.save_dir)

if __name__ == '__main__':
    main() 