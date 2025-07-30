#!/usr/bin/env python3
"""
Complete dataset conversion tool for Regurgitation-YOLODataset-1new
Converts segmentation format to YOLOv5 bbox format and updates data.yaml
"""

import os
import shutil
import yaml
from pathlib import Path
from convert_segmentation_to_bbox import convert_dataset

def create_data_yaml(output_dir, class_names):
    """Create data.yaml file for the converted dataset"""
    data_yaml = {
        'path': str(Path(output_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path

def convert_regurgitation_dataset(input_dir, output_dir):
    """Convert the entire Regurgitation dataset"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Class names for regurgitation dataset
    class_names = ['AR', 'MR', 'PR', 'TR']
    
    # Convert each split
    for split in ['train', 'val', 'test']:
        input_split_path = input_path / split
        output_split_path = output_path / split
        
        if not input_split_path.exists():
            print(f"Warning: {input_split_path} does not exist, skipping")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Convert labels
        input_labels_dir = input_split_path / 'labels'
        output_labels_dir = output_split_path / 'labels'
        
        if input_labels_dir.exists():
            convert_dataset(input_labels_dir, output_labels_dir)
        
        # Copy images
        input_images_dir = input_split_path / 'images'
        output_images_dir = output_split_path / 'images'
        
        if input_images_dir.exists():
            image_files = list(input_images_dir.glob("*"))
            print(f"Copying {len(image_files)} images from {input_images_dir} to {output_images_dir}")
            
            for img_file in image_files:
                shutil.copy2(img_file, output_images_dir / img_file.name)
    
    # Create data.yaml
    create_data_yaml(output_dir, class_names)
    
    print(f"\nDataset conversion completed!")
    print(f"Output directory: {output_path}")
    print(f"Class names: {class_names}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Regurgitation dataset to YOLOv5 format")
    parser.add_argument("input_dir", help="Input directory containing the original dataset")
    parser.add_argument("output_dir", help="Output directory for converted dataset")
    
    args = parser.parse_args()
    
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    convert_regurgitation_dataset(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 