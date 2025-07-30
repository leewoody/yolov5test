#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated training script for multiple datasets
"""

import argparse
import os
import sys
import time
import yaml
import json
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def analyze_dataset(dataset_path):
    """Analyze dataset structure and format"""
    dataset_path = Path(dataset_path)
    data_yaml = dataset_path / 'data.yaml'
    
    if not data_yaml.exists():
        return None
    
    # Read data.yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Analyze label format
    train_labels_dir = dataset_path / 'train' / 'labels'
    if train_labels_dir.exists():
        label_files = list(train_labels_dir.glob('*.txt'))
        if label_files:
            sample_label = label_files[0]
            with open(sample_label, 'r') as f:
                lines = f.read().strip().splitlines()
            
            # Determine format
            has_classification = len(lines) > 1 and len(lines[-1].split()) <= 4
            is_segmentation = any(len(line.split()) > 5 for line in lines[:-1] if line.strip())
            
            return {
                'path': str(dataset_path),
                'nc': data_config.get('nc', 0),
                'names': data_config.get('names', []),
                'train_path': data_config.get('train', ''),
                'val_path': data_config.get('val', ''),
                'test_path': data_config.get('test', ''),
                'has_classification': has_classification,
                'is_segmentation': is_segmentation,
                'label_format': 'segmentation' if is_segmentation else 'bbox',
                'sample_labels': len(label_files),
                'data_yaml': str(data_yaml)
            }
    
    return None


def create_detection_only_dataset(source_path, target_path):
    """Create detection-only version of dataset"""
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    # Create directories
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
            
        print(f"Processing {split} split for {source_path.name}...")
        
        # Process label files
        label_files = list(source_labels_dir.glob('*.txt'))
        processed_count = 0
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.read().strip().splitlines()
                
                # Keep only detection labels (all lines except the last one if it's classification)
                detection_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:  # Valid detection label
                        if len(parts) > 5:  # Segmentation format
                            # Convert segmentation to bbox
                            coords = [float(x) for x in parts[1:]]
                            x_coords = coords[::2]
                            y_coords = coords[1::2]
                            x_center = sum(x_coords) / len(x_coords)
                            y_center = sum(y_coords) / len(y_coords)
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            detection_lines.append(f"{parts[0]} {x_center} {y_center} {width} {height}")
                        else:  # Bbox format
                            detection_lines.append(line)
                
                if detection_lines:
                    # Write detection-only labels
                    target_label_file = target_labels_dir / label_file.name
                    with open(target_label_file, 'w') as f:
                        f.write('\n'.join(detection_lines) + '\n')
                    
                    # Copy corresponding image
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                    image_found = False
                    for ext in image_extensions:
                        image_file = source_images_dir / (label_file.stem + ext)
                        if image_file.exists():
                            target_image_file = target_images_dir / image_file.name
                            shutil.copy2(image_file, target_image_file)
                            processed_count += 1
                            image_found = True
                            break
                    
                    if not image_found:
                        print(f"Warning: Image for {label_file} not found")
                        
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
        
        print(f"Processed {processed_count} files for {split} split")
    
    # Create data.yaml
    source_data_yaml = source_path / 'data.yaml'
    if source_data_yaml.exists():
        with open(source_data_yaml, 'r') as f:
            source_config = yaml.safe_load(f)
        
        data_yaml_content = f"""train: {target_path}/train/images
val: {target_path}/valid/images
test: {target_path}/test/images

nc: {source_config.get('nc', 0)}
names: {source_config.get('names', [])}
"""
        
        with open(target_path / 'data.yaml', 'w') as f:
            f.write(data_yaml_content)
    
    print(f"Detection-only dataset created at {target_path}")
    return target_path


def run_training(dataset_path, config):
    """Run training for a specific dataset"""
    dataset_name = Path(dataset_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset_name}_{timestamp}"
    
    # Create training command
    cmd = [
        'python3', 'train_simple.py',
        '--data', str(Path(dataset_path) / 'data.yaml'),
        '--weights', config['weights'],
        '--hyp', config['hyperparameters'],
        '--epochs', str(config['epochs']),
        '--batch-size', str(config['batch_size']),
        '--imgsz', str(config['image_size']),
        '--project', config['project'],
        '--name', run_name,
        '--exist-ok'
    ]
    
    print(f"\n{'='*60}")
    print(f"Training dataset: {dataset_name}")
    print(f"Run name: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.get('timeout', 3600))
        
        # Save results
        results = {
            'dataset': dataset_name,
            'run_name': run_name,
            'timestamp': timestamp,
            'command': ' '.join(cmd),
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
        # Save to file
        results_file = Path(config['project']) / run_name / 'training_results.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if result.returncode == 0:
            print(f"✅ Training completed successfully for {dataset_name}")
        else:
            print(f"❌ Training failed for {dataset_name}")
            print(f"Error: {result.stderr}")
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Training timed out for {dataset_name}")
        return {
            'dataset': dataset_name,
            'run_name': run_name,
            'timestamp': timestamp,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"❌ Training error for {dataset_name}: {e}")
        return {
            'dataset': dataset_name,
            'run_name': run_name,
            'timestamp': timestamp,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Automated training for multiple datasets')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset paths to train on')
    parser.add_argument('--weights', default='yolov5s.pt', help='Initial weights path')
    parser.add_argument('--hyperparameters', default='hyp_improved.yaml', help='Hyperparameters file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--image-size', type=int, default=640, help='Image size')
    parser.add_argument('--project', default='runs/auto_training', help='Project directory')
    parser.add_argument('--timeout', type=int, default=3600, help='Training timeout in seconds')
    parser.add_argument('--create-detection-only', action='store_true', help='Create detection-only versions')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'weights': args.weights,
        'hyperparameters': args.hyperparameters,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'project': args.project,
        'timeout': args.timeout
    }
    
    print(f"🚀 Starting automated training for {len(args.datasets)} datasets")
    print(f"Configuration: {config}")
    
    # Analyze datasets
    datasets_info = []
    for dataset_path in args.datasets:
        info = analyze_dataset(dataset_path)
        if info:
            datasets_info.append(info)
            print(f"\n📊 Dataset: {info['path']}")
            print(f"   Classes: {info['nc']} ({', '.join(info['names'])})")
            print(f"   Format: {info['label_format']}")
            print(f"   Classification: {'Yes' if info['has_classification'] else 'No'}")
            print(f"   Samples: {info['sample_labels']}")
        else:
            print(f"❌ Could not analyze dataset: {dataset_path}")
    
    if not datasets_info:
        print("❌ No valid datasets found!")
        return
    
    # Create detection-only versions if requested
    if args.create_detection_only:
        print("\n🔄 Creating detection-only versions...")
        for info in datasets_info:
            if info['has_classification'] or info['is_segmentation']:
                source_path = Path(info['path'])
                target_path = source_path.parent / f"{source_path.name}_detection_only"
                create_detection_only_dataset(source_path, target_path)
                info['detection_only_path'] = str(target_path)
    
    # Run training for each dataset
    all_results = []
    for info in datasets_info:
        # Use detection-only version if available, otherwise use original
        dataset_path = info.get('detection_only_path', info['path'])
        
        # Check if data.yaml exists
        data_yaml = Path(dataset_path) / 'data.yaml'
        if not data_yaml.exists():
            print(f"❌ data.yaml not found in {dataset_path}")
            continue
        
        result = run_training(dataset_path, config)
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if r.get('success', False)]
    failed = [r for r in all_results if not r.get('success', False)]
    
    print(f"✅ Successful: {len(successful)}")
    for r in successful:
        print(f"   - {r['dataset']} ({r['run_name']})")
    
    print(f"❌ Failed: {len(failed)}")
    for r in failed:
        print(f"   - {r['dataset']}: {r.get('error', 'Unknown error')}")
    
    # Save overall results
    summary_file = Path(config['project']) / 'training_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'datasets_analyzed': datasets_info,
            'results': all_results,
            'summary': {
                'total': len(all_results),
                'successful': len(successful),
                'failed': len(failed)
            }
        }, f, indent=2)
    
    print(f"\n📄 Summary saved to: {summary_file}")
    print("🎉 Automated training completed!")


if __name__ == "__main__":
    main() 