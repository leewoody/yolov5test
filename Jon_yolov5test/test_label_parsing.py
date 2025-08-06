import os
import numpy as np

def test_label_parsing():
    label_file = "Regurgitation-YOLODataset-Detection/train/labels/a2hiwqVqZ2o=-unnamed_1_1.mp4-0.txt"
    
    with open(label_file) as f:
        lines = f.read().strip().splitlines()
        print(f"Raw lines: {lines}")
        
        # Handle combined detection + classification format
        detection_lines = []
        classification_line = None
        
        for line in lines:
            if len(line.strip()) == 0:
                continue
            parts = line.split()
            print(f"Line parts: {parts}, length: {len(parts)}")
            if len(parts) == 5:  # Detection line: class_id x y width height
                detection_lines.append(parts)
                print(f"Added detection line: {parts}")
            elif len(parts) == 3:  # Classification line: one-hot encoding
                classification_line = parts
                print(f"Found classification line: {parts}")
            else:
                # Skip malformed lines
                print(f"Skipping malformed line: {parts}")
                continue
        
        if detection_lines:
            lb = detection_lines
            print(f"Detection lines before conversion: {lb}")
            lb = np.array(lb, dtype=np.float32)
            print(f"Final labels shape: {lb.shape}")
            print(f"Final labels: {lb}")
        else:
            lb = np.zeros((0, 5), dtype=np.float32)
            print("No detection lines found")

if __name__ == "__main__":
    test_label_parsing() 