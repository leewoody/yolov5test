import os
import glob
from collections import Counter

def analyze_class_distribution(dataset_path):
    """Analyze class distribution from one-hot encoding in label files"""
    
    # Get all label files
    label_files = glob.glob(os.path.join(dataset_path, "*.txt"))
    
    class_counts = Counter()
    total_files = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            # Get the last line (one-hot encoding)
            if lines:
                last_line = lines[-1].strip()
                # Split by space and convert to integers
                one_hot = [int(x) for x in last_line.split()]
                
                # Count which class is active (has value 1)
                for i, value in enumerate(one_hot):
                    if value == 1:
                        class_counts[i] += 1
                        break  # Assuming only one class per image
                
                total_files += 1
                
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    return class_counts, total_files

def main():
    # Analyze training set
    train_path = "Regurgitation-YOLODataset-Detection/train/labels"
    train_counts, train_total = analyze_class_distribution(train_path)
    
    # Analyze validation set
    valid_path = "Regurgitation-YOLODataset-Detection/valid/labels"
    valid_counts, valid_total = analyze_class_distribution(valid_path)
    
    # Analyze test set
    test_path = "Regurgitation-YOLODataset-Detection/test/labels"
    test_counts, test_total = analyze_class_distribution(test_path)
    
    # Class names
    class_names = ['AR', 'MR', 'PR', 'TR']
    
    print("=== CLASS DISTRIBUTION ANALYSIS ===")
    print("\nTraining Set:")
    print(f"Total files: {train_total}")
    for i, class_name in enumerate(class_names):
        count = train_counts.get(i, 0)
        percentage = (count / train_total * 100) if train_total > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print("\nValidation Set:")
    print(f"Total files: {valid_total}")
    for i, class_name in enumerate(class_names):
        count = valid_counts.get(i, 0)
        percentage = (count / valid_total * 100) if valid_total > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print("\nTest Set:")
    print(f"Total files: {test_total}")
    for i, class_name in enumerate(class_names):
        count = test_counts.get(i, 0)
        percentage = (count / test_total * 100) if test_total > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Overall distribution
    print("\nOverall Distribution:")
    total_files = train_total + valid_total + test_total
    overall_counts = Counter()
    for counts in [train_counts, valid_counts, test_counts]:
        overall_counts.update(counts)
    
    for i, class_name in enumerate(class_names):
        count = overall_counts.get(i, 0)
        percentage = (count / total_files * 100) if total_files > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 