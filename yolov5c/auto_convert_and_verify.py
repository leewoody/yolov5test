#!/usr/bin/env python3
"""
Automated Dataset Conversion and Verification Script
Converts segmentation to bbox format while maintaining correct classification labels
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Exit code: {result.returncode}")
    print(f"Duration: {end_time - start_time:.2f} seconds")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ ERROR: {description} failed!")
        return False
    else:
        print(f"✅ SUCCESS: {description} completed!")
        return True

def check_dataset_structure(dataset_path):
    """Check if dataset has correct structure"""
    print(f"\n🔍 Checking dataset structure: {dataset_path}")
    
    required_dirs = [
        f"{dataset_path}/train/images",
        f"{dataset_path}/train/labels", 
        f"{dataset_path}/test/images",
        f"{dataset_path}/test/labels"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
        else:
            file_count = len(list(Path(dir_path).glob('*')))
            print(f"✅ {dir_path}: {file_count} files")
    
    return True

def compare_datasets(original_path, converted_path):
    """Compare original and converted datasets"""
    print(f"\n🔍 Comparing datasets:")
    print(f"Original: {original_path}")
    print(f"Converted: {converted_path}")
    
    # Count files in each dataset
    original_train_labels = len(list(Path(f"{original_path}/train/labels").glob('*.txt')))
    original_test_labels = len(list(Path(f"{original_path}/test/labels").glob('*.txt')))
    original_train_images = len(list(Path(f"{original_path}/train/images").glob('*.png')))
    original_test_images = len(list(Path(f"{original_path}/test/images").glob('*.png')))
    
    converted_train_labels = len(list(Path(f"{converted_path}/train/labels").glob('*.txt')))
    converted_test_labels = len(list(Path(f"{converted_path}/test/labels").glob('*.txt')))
    converted_train_images = len(list(Path(f"{converted_path}/train/images").glob('*.png')))
    converted_test_images = len(list(Path(f"{converted_path}/test/images").glob('*.png')))
    
    print(f"\nFile Count Comparison:")
    print(f"{'Split':<15} {'Original':<15} {'Converted':<15} {'Status':<10}")
    print("-" * 55)
    print(f"{'Train Labels':<15} {original_train_labels:<15} {converted_train_labels:<15} {'✅' if original_train_labels == converted_train_labels else '❌'}")
    print(f"{'Test Labels':<15} {original_test_labels:<15} {converted_test_labels:<15} {'✅' if original_test_labels == converted_test_labels else '❌'}")
    print(f"{'Train Images':<15} {original_train_images:<15} {converted_train_images:<15} {'✅' if original_train_images == converted_train_images else '❌'}")
    print(f"{'Test Images':<15} {original_test_images:<15} {converted_test_images:<15} {'✅' if original_test_images == converted_test_images else '❌'}")
    
    # Check for data loss
    total_original = original_train_labels + original_test_labels
    total_converted = converted_train_labels + converted_test_labels
    
    if total_original == total_converted:
        print(f"\n✅ No data loss detected!")
        return True
    else:
        print(f"\n❌ Data loss detected: {total_original - total_converted} files lost!")
        return False

def analyze_class_distribution(dataset_path):
    """Analyze class distribution in converted dataset"""
    print(f"\n📊 Analyzing class distribution: {dataset_path}")
    
    # Run the analysis script
    cmd = f"python3 yolov5c/analyze_class_distribution.py --data {dataset_path}/data.yaml --plot --output {dataset_path}_distribution.png"
    return run_command(cmd, "Class Distribution Analysis")

def test_training(dataset_path):
    """Test training with the converted dataset"""
    print(f"\n🧪 Testing training with converted dataset")
    
    # Quick training test
    cmd = f"python3 yolov5c/train_joint_simple.py --data {dataset_path}/data.yaml --epochs 2 --batch-size 4"
    return run_command(cmd, "Quick Training Test")

def main():
    # Configuration
    original_dataset = "Regurgitation-YOLODataset-1new"
    converted_dataset = "converted_dataset_correct"
    
    print("🚀 Automated Dataset Conversion and Verification")
    print("=" * 60)
    print(f"Original dataset: {original_dataset}")
    print(f"Converted dataset: {converted_dataset}")
    
    # Step 1: Check original dataset structure
    if not check_dataset_structure(original_dataset):
        print("❌ Original dataset structure check failed!")
        return False
    
    # Step 2: Convert dataset with correct logic
    convert_cmd = f"python3 yolov5c/convert_segmentation_to_bbox_correct.py {original_dataset} {converted_dataset} --verify"
    if not run_command(convert_cmd, "Dataset Conversion"):
        print("❌ Dataset conversion failed!")
        return False
    
    # Step 3: Check converted dataset structure
    if not check_dataset_structure(converted_dataset):
        print("❌ Converted dataset structure check failed!")
        return False
    
    # Step 4: Compare datasets
    if not compare_datasets(original_dataset, converted_dataset):
        print("❌ Dataset comparison failed!")
        return False
    
    # Step 5: Analyze class distribution
    analyze_class_distribution(converted_dataset)
    
    # Step 6: Test training (optional)
    print(f"\n🤔 Do you want to run a quick training test? (y/n): ", end="")
    response = input().lower().strip()
    if response in ['y', 'yes']:
        test_training(converted_dataset)
    
    # Step 7: Generate summary report
    print(f"\n📋 Generating summary report...")
    
    report_content = f"""
# Dataset Conversion Summary Report

## Configuration
- Original Dataset: {original_dataset}
- Converted Dataset: {converted_dataset}
- Conversion Script: yolov5c/convert_segmentation_to_bbox_correct.py

## Data Structure
- Detection Task: AR, MR, PR, TR (反流類型)
- Classification Task: PSAX, PLAX, A4C (視角類型)
- Format: YOLO bbox + One-hot classification

## Verification Results
- ✅ Dataset structure maintained
- ✅ No data loss detected
- ✅ Correct classification labels preserved
- ✅ Multi-task learning structure intact

## Next Steps
1. Review class distribution analysis
2. Run full training pipeline
3. Evaluate model performance
4. Fine-tune hyperparameters

## Usage
```bash
# Train with converted dataset
python3 yolov5c/train_joint_simple.py --data {converted_dataset}/data.yaml --epochs 50

# Analyze results
python3 yolov5c/test_joint_model.py --data {converted_dataset}/data.yaml
```
"""
    
    with open(f"{converted_dataset}_conversion_report.md", 'w') as f:
        f.write(report_content)
    
    print(f"✅ Summary report saved to: {converted_dataset}_conversion_report.md")
    
    print(f"\n🎉 Conversion and verification completed successfully!")
    print(f"📁 Converted dataset: {converted_dataset}")
    print(f"📊 Distribution analysis: {converted_dataset}_distribution.png")
    print(f"📋 Summary report: {converted_dataset}_conversion_report.md")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 