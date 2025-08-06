import os
import glob
from pathlib import Path

def analyze_dataset(dataset_path):
    """Analyze the dataset to find missing labels"""
    
    # Get all image files
    train_images = set()
    valid_images = set()
    test_images = set()
    
    # Collect image names from all splits
    for split in ['train', 'valid', 'test']:
        image_dir = os.path.join(dataset_path, split, 'images')
        if os.path.exists(image_dir):
            image_files = glob.glob(os.path.join(image_dir, '*.png'))
            image_names = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
            
            if split == 'train':
                train_images = image_names
            elif split == 'valid':
                valid_images = image_names
            elif split == 'test':
                test_images = image_names
    
    # Get all label files
    train_labels = set()
    valid_labels = set()
    test_labels = set()
    
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(label_dir):
            label_files = glob.glob(os.path.join(label_dir, '*.txt'))
            label_names = {os.path.splitext(os.path.basename(f))[0] for f in label_files}
            
            if split == 'train':
                train_labels = label_names
            elif split == 'valid':
                valid_labels = label_names
            elif split == 'test':
                test_labels = label_names
    
    # Find missing labels
    missing_train = train_images - train_labels
    missing_valid = valid_images - valid_labels
    missing_test = test_images - test_labels
    
    # Find images without labels
    images_without_labels = missing_train | missing_valid | missing_test
    
    # Find labels without images
    labels_without_images = (train_labels - train_images) | (valid_labels - valid_images) | (test_labels - test_images)
    
    return {
        'missing_train': missing_train,
        'missing_valid': missing_valid,
        'missing_test': missing_test,
        'images_without_labels': images_without_labels,
        'labels_without_images': labels_without_images,
        'total_images': len(train_images | valid_images | test_images),
        'total_labels': len(train_labels | valid_labels | test_labels)
    }

def analyze_label_format(dataset_path, sample_size=3):
    """Analyze the format of label files"""
    print("\n=== Label Format Analysis ===")
    
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(label_dir):
            continue
            
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        if not label_files:
            continue
            
        print(f"\n{split.upper()} split sample:")
        for i, label_file in enumerate(label_files[:sample_size]):
            print(f"\nFile: {os.path.basename(label_file)}")
            try:
                with open(label_file, 'r') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    print(f"  Total lines: {len(lines)}")
                    print(f"  Raw content:")
                    for j, line in enumerate(lines):
                        if line.strip():  # Only show non-empty lines
                            print(f"    Line {j+1}: '{line}'")
                            
                            if j == 0:  # First line - segmentation
                                parts = line.split()
                                if len(parts) >= 1:
                                    class_id = parts[0]
                                    coords = parts[1:]
                                    print(f"      -> Segmentation: Class {class_id}, {len(coords)} coordinates")
                            elif j == 1:  # Second line - classification
                                parts = line.split()
                                print(f"      -> Classification: {parts}")
                                
            except Exception as e:
                print(f"  Error reading file: {e}")

def check_label_issues(dataset_path):
    """Check for specific issues in label files"""
    print("\n=== Label Issues Analysis ===")
    
    issues = {
        'empty_files': [],
        'missing_classification': [],
        'invalid_format': [],
        'wrong_coordinate_count': []
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
                        if len(parts) < 1:
                            issues['invalid_format'].append(f"{split}/{os.path.basename(label_file)}")
                        elif len(parts) != 9:  # class_id + 8 coordinates
                            issues['wrong_coordinate_count'].append(f"{split}/{os.path.basename(label_file)}")
                    
                    # Check classification line
                    class_line = lines[1].strip()
                    if class_line:
                        parts = class_line.split()
                        if len(parts) != 4:  # Should have 4 values for one-hot encoding
                            issues['invalid_format'].append(f"{split}/{os.path.basename(label_file)}")
                            
            except Exception as e:
                issues['invalid_format'].append(f"{split}/{os.path.basename(label_file)} (Error: {e})")
    
    # Report issues
    for issue_type, files in issues.items():
        if files:
            print(f"\n{issue_type.replace('_', ' ').title()}: {len(files)} files")
            for file in files[:10]:  # Show first 10
                full_path = os.path.join(dataset_path, file)
                print(f"  - {full_path}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
        else:
            print(f"\n{issue_type.replace('_', ' ').title()}: None found")

def main():
    dataset_path = "Regurgitation-YOLODataset-Detection"
    
    print("=== YOLO Dataset Analysis ===")
    print(f"Dataset path: {dataset_path}")
    
    # Analyze missing labels
    results = analyze_dataset(dataset_path)
    
    print(f"\nTotal images: {results['total_images']}")
    print(f"Total labels: {results['total_labels']}")
    
    print(f"\n=== Missing Labels ===")
    print(f"Train split - Missing labels: {len(results['missing_train'])}")
    if results['missing_train']:
        print("Missing train labels:")
        for img in sorted(results['missing_train']):
            print(f"  - {img}")
    
    print(f"\nValid split - Missing labels: {len(results['missing_valid'])}")
    if results['missing_valid']:
        print("Missing valid labels:")
        for img in sorted(results['missing_valid']):
            print(f"  - {img}")
    
    print(f"\nTest split - Missing labels: {len(results['missing_test'])}")
    if results['missing_test']:
        print("Missing test labels:")
        for img in sorted(results['missing_test']):
            print(f"  - {img}")
    
    print(f"\n=== Orphaned Files ===")
    print(f"Labels without images: {len(results['labels_without_images'])}")
    if results['labels_without_images']:
        print("Labels without corresponding images:")
        for label in sorted(results['labels_without_images']):
            print(f"  - {label}")
    
    # Analyze label format
    analyze_label_format(dataset_path)
    
    # Check for label issues
    check_label_issues(dataset_path)

if __name__ == "__main__":
    main() 