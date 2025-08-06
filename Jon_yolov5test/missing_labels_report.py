import os
import glob
from pathlib import Path

def generate_missing_labels_report(dataset_path):
    """Generate a comprehensive report of missing labels and dataset format"""
    
    print("=" * 80)
    print("YOLO DATASET ANALYSIS REPORT")
    print("=" * 80)
    
    # Dataset information
    print(f"\nDataset: {dataset_path}")
    print(f"Classes: AR, MR, PR, TR (4 regurgitation types)")
    print(f"Label Format: Image segmentation + Ultrasound view one-hot encoding")
    
    # Analyze dataset structure
    train_images = set()
    valid_images = set()
    test_images = set()
    train_labels = set()
    valid_labels = set()
    test_labels = set()
    
    for split in ['train', 'valid', 'test']:
        image_dir = os.path.join(dataset_path, split, 'images')
        label_dir = os.path.join(dataset_path, split, 'labels')
        
        if os.path.exists(image_dir):
            image_files = glob.glob(os.path.join(image_dir, '*.png'))
            image_names = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
            
            if split == 'train':
                train_images = image_names
            elif split == 'valid':
                valid_images = image_names
            elif split == 'test':
                test_images = image_names
        
        if os.path.exists(label_dir):
            label_files = glob.glob(os.path.join(label_dir, '*.txt'))
            label_names = {os.path.splitext(os.path.basename(f))[0] for f in label_files}
            
            if split == 'train':
                train_labels = label_names
            elif split == 'valid':
                valid_labels = label_names
            elif split == 'test':
                test_labels = label_names
    
    # Statistics
    total_images = len(train_images | valid_images | test_images)
    total_labels = len(train_labels | valid_labels | test_labels)
    
    print(f"\nDATASET STATISTICS:")
    print(f"  Total images: {total_images}")
    print(f"  Total labels: {total_labels}")
    print(f"  Train images: {len(train_images)}")
    print(f"  Train labels: {len(train_labels)}")
    print(f"  Valid images: {len(valid_images)}")
    print(f"  Valid labels: {len(valid_labels)}")
    print(f"  Test images: {len(test_images)}")
    print(f"  Test labels: {len(test_labels)}")
    
    # Find missing labels
    missing_train = train_images - train_labels
    missing_valid = valid_images - valid_labels
    missing_test = test_images - test_labels
    
    print(f"\nMISSING LABELS ANALYSIS:")
    print(f"  Train split - Missing labels: {len(missing_train)}")
    print(f"  Valid split - Missing labels: {len(missing_valid)}")
    print(f"  Test split - Missing labels: {len(missing_test)}")
    
    # Find labels with issues
    issues = {
        'missing_classification': [],
        'empty_files': [],
        'invalid_format': []
    }
    
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(label_dir):
            continue
            
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    lines = content.split('\n')
                    
                    # Check for empty files
                    if not content:
                        issues['empty_files'].append(f"{split}/{os.path.basename(label_file)}")
                        continue
                    
                    # Check for missing classification line
                    if len(lines) < 2:
                        issues['missing_classification'].append(f"{split}/{os.path.basename(label_file)}")
                        continue
                    
                    # Check segmentation line
                    seg_line = lines[0].strip()
                    if seg_line:
                        parts = seg_line.split()
                        if len(parts) != 9:  # class_id + 8 coordinates
                            issues['invalid_format'].append(f"{split}/{os.path.basename(label_file)}")
                    
                    # Check classification line
                    class_line = lines[1].strip()
                    if class_line:
                        parts = class_line.split()
                        if len(parts) != 3:  # Should have 3 values for one-hot encoding (4 classes)
                            issues['invalid_format'].append(f"{split}/{os.path.basename(label_file)}")
                            
            except Exception as e:
                issues['invalid_format'].append(f"{split}/{os.path.basename(label_file)} (Error: {e})")
    
    print(f"\nLABEL FORMAT ISSUES:")
    for issue_type, files in issues.items():
        if files:
            print(f"  {issue_type.replace('_', ' ').title()}: {len(files)} files")
            for file in files:
                print(f"    - {file}")
        else:
            print(f"  {issue_type.replace('_', ' ').title()}: None found")
    
    # Label format explanation
    print(f"\nLABEL FORMAT EXPLANATION:")
    print(f"  Each label file contains:")
    print(f"    Line 1: Image segmentation coordinates")
    print(f"      Format: <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>")
    print(f"      - class_id: 0=AR, 1=MR, 2=PR, 3=TR (regurgitation types)")
    print(f"      - x1,y1,x2,y2,x3,y3,x4,y4: Normalized polygon coordinates")
    print(f"    Line 2: Ultrasound view one-hot encoding")
    print(f"      Format: <View1> <View2> <View3> <View4>")
    print(f"      - Values: 0 or 1 indicating the ultrasound view/angle")
    print(f"      - Examples:")
    print(f"        * '1 0 0 0' = View 1 (e.g., Parasternal Long Axis)")
    print(f"        * '0 1 0 0' = View 2 (e.g., Parasternal Short Axis)")
    print(f"        * '0 0 1 0' = View 3 (e.g., Apical 4-Chamber)")
    print(f"        * '0 0 0 1' = View 4 (e.g., Apical 2-Chamber)")
    
    # Summary of missing data
    all_missing = missing_train | missing_valid | missing_test
    all_issues = set()
    for issue_list in issues.values():
        all_issues.update(issue_list)
    
    print(f"\nSUMMARY:")
    print(f"  Images without labels: {len(all_missing)}")
    print(f"  Labels with format issues: {len(all_issues)}")
    
    if all_missing:
        print(f"\n  MISSING LABELS (Images without corresponding label files):")
        for img in sorted(all_missing):
            print(f"    - {img}")
    
    if all_issues:
        print(f"\n  LABELS WITH ISSUES:")
        for issue in sorted(all_issues):
            print(f"    - {issue}")
    
    print(f"\n" + "=" * 80)

if __name__ == "__main__":
    generate_missing_labels_report("Regurgitation-YOLODataset-1new") 