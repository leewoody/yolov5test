import os
import glob

def check_all_labels():
    # Check all three directories
    directories = ['train', 'test', 'valid']
    
    for directory in directories:
        label_dir = f"Regurgitation-YOLODataset-Detection/{directory}/labels"
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        
        total_files = len(label_files)
        files_with_detection = 0
        files_with_classification = 0
        files_with_both = 0
        files_with_neither = 0
        
        print(f"\n=== Checking {directory.upper()} directory ===")
        print(f"Total label files: {total_files}")
        
        if total_files == 0:
            print("No label files found!")
            continue
        
        # Check all files in this directory
        for i, label_file in enumerate(label_files):
            with open(label_file, 'r') as f:
                lines = f.read().strip().splitlines()
                
            detection_lines = []
            classification_line = None
            
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                parts = line.split()
                if len(parts) == 5:  # Detection line
                    detection_lines.append(parts)
                elif len(parts) == 3:  # Classification line
                    classification_line = parts
            
            if detection_lines and classification_line:
                files_with_both += 1
            elif detection_lines:
                files_with_detection += 1
            elif classification_line:
                files_with_classification += 1
            else:
                files_with_neither += 1
        
        print(f"Files with both detection and classification: {files_with_both}")
        print(f"Files with only detection: {files_with_detection}")
        print(f"Files with only classification: {files_with_classification}")
        print(f"Files with neither: {files_with_neither}")
        
        # Show sample content from first few files
        print(f"\nSample content (first 3 files):")
        for i, label_file in enumerate(label_files[:3]):
            with open(label_file, 'r') as f:
                lines = f.read().strip().splitlines()
                
            detection_lines = []
            classification_line = None
            
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                parts = line.split()
                if len(parts) == 5:  # Detection line
                    detection_lines.append(parts)
                elif len(parts) == 3:  # Classification line
                    classification_line = parts
            
            print(f"  {os.path.basename(label_file)}:")
            print(f"    Detection: {detection_lines}")
            print(f"    Classification: {classification_line}")

if __name__ == "__main__":
    check_all_labels() 