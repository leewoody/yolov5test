#!/usr/bin/env python3
"""
Simple Label Viewer Tool for YOLOv5 Dataset
Display images and corresponding label positions
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
            print(f"Warning: File {label_path} has insufficient valid lines")
            return [], []
        
        # First line is segmentation coordinates
        seg_coordinates = parse_segmentation_line(non_empty_lines[0])
        
        # Last line is classification labels
        cls_labels = parse_classification_line(non_empty_lines[-1])
        
        return seg_coordinates, cls_labels
    
    except Exception as e:
        print(f"Error: Reading file {label_path}: {e}")
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

def visualize_label(image_path: str, label_path: str, class_names: List[str], 
                   show_segmentation: bool = True, show_bbox: bool = True,
                   save_path: Optional[str] = None) -> None:
    """Visualize single image and label"""
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load label
    seg_coordinates, cls_labels = load_label_file(label_path)
    
    if not seg_coordinates and not cls_labels:
        print(f"Warning: Cannot parse label file {label_path}")
        return
    
    # Create image copy for drawing
    image_draw = image_rgb.copy()
    
    # Draw segmentation contour
    if show_segmentation and seg_coordinates:
        image_draw = draw_segmentation(image_draw, seg_coordinates, color=(0, 255, 0), thickness=2)
    
    # Draw bounding box
    if show_bbox and seg_coordinates:
        bbox = segmentation_to_bbox(seg_coordinates)
        image_draw = draw_bbox(image_draw, bbox, color=(255, 0, 0), thickness=2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Annotated image
    ax2.imshow(image_draw)
    ax2.set_title('Annotated Image')
    ax2.axis('off')
    
    # Add label information
    info_text = f"Image: {os.path.basename(image_path)}\n"
    info_text += f"Label: {os.path.basename(label_path)}\n"
    
    if seg_coordinates:
        info_text += f"Segmentation points: {len(seg_coordinates)}\n"
        bbox = segmentation_to_bbox(seg_coordinates)
        info_text += f"Bounding box: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})\n"
    
    if cls_labels:
        info_text += f"Classification labels: {cls_labels}\n"
        if len(cls_labels) <= len(class_names):
            active_classes = [class_names[i] for i, label in enumerate(cls_labels) if label == 1]
            info_text += f"Classes: {', '.join(active_classes) if active_classes else 'None'}\n"
    
    # Add text information to figure
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Simple YOLOv5 Label Viewer')
    parser.add_argument('--data', type=str, required=True, help='data.yaml file path')
    parser.add_argument('--image', type=str, required=True, help='Image file path')
    parser.add_argument('--label', type=str, required=True, help='Label file path')
    parser.add_argument('--save', type=str, help='Save visualization to file')
    
    args = parser.parse_args()
    
    # Load data configuration
    data_config = load_data_yaml(args.data)
    class_names = data_config.get('names', [])
    
    print(f"Classes: {class_names}")
    print(f"Image: {args.image}")
    print(f"Label: {args.label}")
    
    # Visualize
    visualize_label(args.image, args.label, class_names, save_path=args.save)

if __name__ == '__main__':
    main() 