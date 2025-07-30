#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create detection-only dataset from combined detection+classification dataset
"""

import os
import shutil
from pathlib import Path


def create_detection_only_dataset(source_dir, target_dir):
    """
    Create detection-only dataset by removing classification labels
    
    Args:
        source_dir: Source dataset directory
        target_dir: Target dataset directory
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directories
    for split in ['train', 'valid', 'test']:
        (target_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        source_labels_dir = source_path / split / 'labels'
        source_images_dir = source_path / split / 'images'
        target_labels_dir = target_path / split / 'labels'
        target_images_dir = target_path / split / 'images'
        
        if not source_labels_dir.exists():
            print(f"Warning: {source_labels_dir} does not exist, skipping...")
            continue
            
        print(f"Processing {split} split...")
        
        # Process label files
        label_files = list(source_labels_dir.glob('*.txt'))
        processed_count = 0
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.read().strip().splitlines()
                
                # Keep only detection labels (all lines except the last one)
                detection_lines = lines[:-1] if len(lines) > 1 else lines
                
                if detection_lines:
                    # Write detection-only labels
                    target_label_file = target_labels_dir / label_file.name
                    with open(target_label_file, 'w') as f:
                        f.write('\n'.join(detection_lines) + '\n')
                    
                    # Copy corresponding image
                    image_file = source_images_dir / (label_file.stem + '.jpg')
                    if image_file.exists():
                        target_image_file = target_images_dir / image_file.name
                        shutil.copy2(image_file, target_image_file)
                        processed_count += 1
                    else:
                        print(f"Warning: Image {image_file} not found")
                        
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
        
        print(f"Processed {processed_count} files for {split} split")
    
    # Create data.yaml
    data_yaml_content = f"""train: {target_path}/train/images
val: {target_path}/valid/images
test: {target_path}/test/images

nc: 2
names: ['ASDType2', 'VSDType2']
"""
    
    with open(target_path / 'data.yaml', 'w') as f:
        f.write(data_yaml_content)
    
    print(f"Detection-only dataset created at {target_path}")
    print(f"Data configuration saved to {target_path}/data.yaml")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create detection-only dataset')
    parser.add_argument('--source', required=True, help='Source dataset directory')
    parser.add_argument('--target', required=True, help='Target dataset directory')
    
    args = parser.parse_args()
    
    create_detection_only_dataset(args.source, args.target) 