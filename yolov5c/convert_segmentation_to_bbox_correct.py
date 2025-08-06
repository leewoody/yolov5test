#!/usr/bin/env python3
"""
Correct Segmentation to Bbox Conversion Script
Maintains original classification labels (PSAX, PLAX, A4C) while converting detection format
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

def parse_segmentation_line(line: str) -> Tuple[int, List[float]]:
    """Parse segmentation line to get class_id and coordinates"""
    parts = line.strip().split()
    if len(parts) < 9:  # class_id + 8 coordinates
        raise ValueError(f"Invalid segmentation line: {line}")
    
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:9]]  # 8 coordinates for polygon
    
    return class_id, coords

def convert_polygon_to_bbox(coords: List[float]) -> Tuple[float, float, float, float]:
    """Convert polygon coordinates to bbox format (center_x, center_y, width, height)"""
    if len(coords) != 8:
        raise ValueError(f"Expected 8 coordinates, got {len(coords)}")
    
    # Extract x and y coordinates
    x_coords = coords[::2]  # Even indices: 0, 2, 4, 6
    y_coords = coords[1::2]  # Odd indices: 1, 3, 5, 7
    
    # Calculate bbox
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Convert to center format
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    return center_x, center_y, width, height

def convert_label_file(input_file: str, output_file: str) -> bool:
    """Convert a single label file from segmentation to bbox format"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find non-empty lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 2:
            print(f"Warning: File {input_file} has insufficient lines")
            return False
        
        # Parse segmentation line (first line)
        seg_line = non_empty_lines[0]
        class_id, coords = parse_segmentation_line(seg_line)
        
        # Convert to bbox format
        center_x, center_y, width, height = convert_polygon_to_bbox(coords)
        
        # Get classification line (second line) - KEEP ORIGINAL
        classification_line = non_empty_lines[1]
        
        # Validate classification line
        cls_parts = classification_line.split()
        if len(cls_parts) != 3:
            print(f"Warning: Invalid classification line in {input_file}: {classification_line}")
            return False
        
        # Write converted label file
        with open(output_file, 'w', encoding='utf-8') as f:
            # First line: bbox format (class_id center_x center_y width height)
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            # Second line: original classification (PSAX, PLAX, A4C)
            f.write(f"{classification_line}\n")
        
        return True
        
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False

def convert_dataset_correct(input_dir: str, output_dir: str) -> Dict:
    """Convert entire dataset with correct logic"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy data.yaml
    if (input_path / 'data.yaml').exists():
        with open(input_path / 'data.yaml', 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths in data.yaml
        data_config['train'] = './train/images'
        data_config['val'] = './valid/images'
        data_config['test'] = './test/images'
        
        with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    
    conversion_stats = {
        'total_files': 0,
        'converted_files': 0,
        'failed_files': 0,
        'detection_classes': {'AR': 0, 'MR': 0, 'PR': 0, 'TR': 0},
        'classification_classes': {'PSAX': 0, 'PLAX': 0, 'A4C': 0}
    }
    
    # Process each split
    for split in ['train', 'test']:
        input_images_dir = input_path / split / 'images'
        input_labels_dir = input_path / split / 'labels'
        output_images_dir = output_path / split / 'images'
        output_labels_dir = output_path / split / 'labels'
        
        if not input_images_dir.exists() or not input_labels_dir.exists():
            print(f"Warning: {split} directory not found")
            continue
        
        # Copy images
        image_files = list(input_images_dir.glob('*.png'))
        for img_file in image_files:
            import shutil
            shutil.copy2(img_file, output_images_dir / img_file.name)
        
        # Convert labels
        label_files = list(input_labels_dir.glob('*.txt'))
        print(f"\nProcessing {split} split: {len(label_files)} label files")
        
        for label_file in label_files:
            conversion_stats['total_files'] += 1
            output_label_file = output_labels_dir / label_file.name
            
            if convert_label_file(str(label_file), str(output_label_file)):
                conversion_stats['converted_files'] += 1
                
                # Count classes
                try:
                    with open(output_label_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Count detection class
                    bbox_line = lines[0].strip().split()
                    if len(bbox_line) >= 1:
                        class_id = int(bbox_line[0])
                        class_names = ['AR', 'MR', 'PR', 'TR']
                        if 0 <= class_id < len(class_names):
                            conversion_stats['detection_classes'][class_names[class_id]] += 1
                    
                    # Count classification class
                    cls_line = lines[1].strip().split()
                    if len(cls_line) == 3:
                        cls_values = [int(x) for x in cls_line]
                        if cls_values == [0, 0, 1]:
                            conversion_stats['classification_classes']['PSAX'] += 1
                        elif cls_values == [0, 1, 0]:
                            conversion_stats['classification_classes']['PLAX'] += 1
                        elif cls_values == [1, 0, 0]:
                            conversion_stats['classification_classes']['A4C'] += 1
                
                except Exception as e:
                    print(f"Error counting classes for {label_file}: {e}")
            else:
                conversion_stats['failed_files'] += 1
    
    return conversion_stats

def verify_conversion(output_dir: str) -> Dict:
    """Verify the converted dataset"""
    
    output_path = Path(output_dir)
    verification_stats = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'detection_classes': {'AR': 0, 'MR': 0, 'PR': 0, 'TR': 0},
        'classification_classes': {'PSAX': 0, 'PLAX': 0, 'A4C': 0},
        'errors': []
    }
    
    for split in ['train', 'test']:
        labels_dir = output_path / split / 'labels'
        images_dir = output_path / split / 'images'
        
        if not labels_dir.exists():
            continue
        
        label_files = list(labels_dir.glob('*.txt'))
        
        for label_file in label_files:
            verification_stats['total_files'] += 1
            
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) < 2:
                    verification_stats['invalid_files'] += 1
                    verification_stats['errors'].append(f"{label_file}: Insufficient lines")
                    continue
                
                # Check bbox line
                bbox_line = lines[0].strip().split()
                if len(bbox_line) != 5:
                    verification_stats['invalid_files'] += 1
                    verification_stats['errors'].append(f"{label_file}: Invalid bbox format")
                    continue
                
                # Check classification line
                cls_line = lines[1].strip().split()
                if len(cls_line) != 3:
                    verification_stats['invalid_files'] += 1
                    verification_stats['errors'].append(f"{label_file}: Invalid classification format")
                    continue
                
                # Validate classification values
                cls_values = [int(x) for x in cls_line]
                if sum(cls_values) != 1 or not all(x in [0, 1] for x in cls_values):
                    verification_stats['invalid_files'] += 1
                    verification_stats['errors'].append(f"{label_file}: Invalid one-hot encoding")
                    continue
                
                # Count classes
                class_id = int(bbox_line[0])
                class_names = ['AR', 'MR', 'PR', 'TR']
                if 0 <= class_id < len(class_names):
                    verification_stats['detection_classes'][class_names[class_id]] += 1
                
                if cls_values == [0, 0, 1]:
                    verification_stats['classification_classes']['PSAX'] += 1
                elif cls_values == [0, 1, 0]:
                    verification_stats['classification_classes']['PLAX'] += 1
                elif cls_values == [1, 0, 0]:
                    verification_stats['classification_classes']['A4C'] += 1
                
                verification_stats['valid_files'] += 1
                
            except Exception as e:
                verification_stats['invalid_files'] += 1
                verification_stats['errors'].append(f"{label_file}: {e}")
    
    return verification_stats

def print_stats(stats: Dict, title: str):
    """Print conversion/verification statistics"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"Total files: {stats['total_files']}")
    if 'converted_files' in stats:
        print(f"Converted files: {stats['converted_files']}")
        print(f"Failed files: {stats['failed_files']}")
    if 'valid_files' in stats:
        print(f"Valid files: {stats['valid_files']}")
        print(f"Invalid files: {stats['invalid_files']}")
    
    print(f"\nDetection Classes:")
    for class_name, count in stats['detection_classes'].items():
        print(f"  {class_name}: {count}")
    
    print(f"\nClassification Classes:")
    for class_name, count in stats['classification_classes'].items():
        print(f"  {class_name}: {count}")
    
    if 'errors' in stats and stats['errors']:
        print(f"\nErrors (first 10):")
        for error in stats['errors'][:10]:
            print(f"  {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")

def main():
    parser = argparse.ArgumentParser(description='Convert segmentation to bbox with correct classification')
    parser.add_argument('input_dir', help='Input dataset directory')
    parser.add_argument('output_dir', help='Output dataset directory')
    parser.add_argument('--verify', action='store_true', help='Verify conversion after completion')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found")
        return
    
    print("Starting correct dataset conversion...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    # Convert dataset
    conversion_stats = convert_dataset_correct(args.input_dir, args.output_dir)
    print_stats(conversion_stats, "CONVERSION STATISTICS")
    
    # Verify conversion
    if args.verify:
        print("\nVerifying conversion...")
        verification_stats = verify_conversion(args.output_dir)
        print_stats(verification_stats, "VERIFICATION STATISTICS")
        
        # Check for issues
        if verification_stats['invalid_files'] > 0:
            print(f"\n⚠️  WARNING: {verification_stats['invalid_files']} invalid files found!")
        else:
            print(f"\n✅ SUCCESS: All {verification_stats['valid_files']} files are valid!")
    
    print(f"\nConversion completed. Output saved to: {args.output_dir}")

if __name__ == '__main__':
    main() 