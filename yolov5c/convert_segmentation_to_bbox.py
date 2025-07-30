#!/usr/bin/env python3
"""
Convert segmentation format to YOLOv5 bbox format
Input format:
- Line 1: segmentation coordinates (class x1 y1 x2 y2 x3 y3 ...)
- Line 2: classification labels (e.g., "0 1 0")

Output format:
- Line 1: YOLOv5 bbox (class x_center y_center width height)
- Line 2: classification label (single digit, e.g., "1")
"""

import os
import glob
import numpy as np
from pathlib import Path

def parse_segmentation_line(line):
    """Parse segmentation line and extract coordinates"""
    parts = line.strip().split()
    if len(parts) < 7:  # Need at least class + 3 points (6 coordinates)
        return None, None
    
    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        # Group coordinates into (x, y) pairs
        points = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                points.append((coords[i], coords[i + 1]))
        
        return class_id, points
    except (ValueError, IndexError):
        return None, None

def parse_classification_line(line):
    """Parse classification line and return the class index"""
    try:
        parts = line.strip().split()
        # Find the index of "1" in the one-hot encoding
        for i, val in enumerate(parts):
            if val == "1":
                return i
        return 0  # Default to 0 if no "1" found
    except (ValueError, IndexError):
        return 0

def points_to_bbox(points):
    """Convert polygon points to bounding box (x_center, y_center, width, height)"""
    if not points:
        return None
    
    # Convert to numpy array for easier calculation
    points = np.array(points)
    
    # Calculate bounding box
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    
    # Convert to YOLO format (normalized)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def convert_label_file(input_file, output_file):
    """Convert a single label file from segmentation to bbox format"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            print(f"Warning: {input_file} has less than 2 lines, skipping")
            return False
        
        # Parse segmentation line
        class_id, points = parse_segmentation_line(lines[0])
        if class_id is None or not points:
            print(f"Warning: Could not parse segmentation line in {input_file}")
            return False
        
        # Parse classification line
        cls_label = parse_classification_line(lines[1])
        
        # Convert points to bbox
        bbox = points_to_bbox(points)
        if bbox is None:
            print(f"Warning: Could not convert points to bbox in {input_file}")
            return False
        
        x_center, y_center, width, height = bbox
        
        # Write output in YOLOv5 format
        with open(output_file, 'w') as f:
            # Write bbox line (class x_center y_center width height)
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            # Write classification line (single digit)
            f.write(f"{cls_label}\n")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def convert_dataset(input_dir, output_dir):
    """Convert all label files in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files
    label_files = list(input_path.glob("*.txt"))
    
    if not label_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(label_files)} label files")
    
    success_count = 0
    error_count = 0
    
    for label_file in label_files:
        output_file = output_path / label_file.name
        
        if convert_label_file(label_file, output_file):
            success_count += 1
        else:
            error_count += 1
    
    print(f"Conversion completed:")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert segmentation labels to YOLOv5 bbox format")
    parser.add_argument("output_dir", help="Output directory for converted files")
    parser.add_argument("--input-dir", help="Input directory containing label files")
    parser.add_argument("--single-file", help="Convert a single file instead of directory")
    
    args = parser.parse_args()
    
    if args.single_file:
        # Convert single file
        input_file = Path(args.single_file)
        output_file = Path(args.output_dir) / input_file.name
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if convert_label_file(input_file, output_file):
            print(f"Successfully converted {input_file} to {output_file}")
        else:
            print(f"Failed to convert {input_file}")
    elif args.input_dir:
        # Convert entire directory
        convert_dataset(args.input_dir, args.output_dir)
    else:
        print("Error: Please specify either --input-dir or --single-file")
        parser.print_help()

if __name__ == "__main__":
    main() 