#!/usr/bin/env python3
"""
Fix missing labels by copying from source dataset and converting to detection format
"""

import os
import shutil
from pathlib import Path

def polygon_to_bbox(polygon_coords):
    """
    Convert polygon coordinates to bounding box (x_center, y_center, width, height)
    """
    import numpy as np
    
    # Reshape coordinates into pairs
    coords = np.array(polygon_coords).reshape(-1, 2)
    
    # Calculate bounding box
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    
    # Convert to center format
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_center, y_center, width, height]

def convert_label_file(input_path, output_path):
    """
    Convert a single label file from segmentation to detection format
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print(f"Warning: {input_path} has insufficient lines")
        return False
    
    # Parse segmentation line (first line)
    seg_line = lines[0].strip()
    if not seg_line:
        print(f"Warning: {input_path} has empty segmentation line")
        return False
    
    seg_parts = seg_line.split()
    if len(seg_parts) < 9:  # class_id + 8 coordinates
        print(f"Warning: {input_path} has insufficient coordinates")
        return False
    
    # Extract class_id and coordinates
    class_id = int(seg_parts[0])
    coords = [float(x) for x in seg_parts[1:]]
    
    # Convert polygon to bounding box
    bbox = polygon_to_bbox(coords)
    
    # Parse classification line (last line)
    cls_line = lines[-1].strip()
    if not cls_line:
        print(f"Warning: {input_path} has empty classification line")
        return False
    
    cls_labels = [int(x) for x in cls_line.split()]
    
    # Write detection format
    with open(output_path, 'w') as f:
        # Write detection line: class_id x_center y_center width height
        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        # Write classification line
        f.write(" ".join(map(str, cls_labels)) + "\n")
    
    return True

def main():
    """Main function to fix missing labels"""
    
    # Missing files
    missing_files = [
        "aGdjwqtqa8Kb-unnamed_1_6.mp4-40",
        "bmRlwqVuacKY-unnamed_1_3.mp4-28"
    ]
    
    source_dataset = "Regurgitation-YOLODataset-1new"
    target_dataset = "Regurgitation-YOLODataset-Detection"
    
    print("Fixing missing labels...")
    
    for filename in missing_files:
        source_path = Path(source_dataset) / "train" / "labels" / f"{filename}.txt"
        target_path = Path(target_dataset) / "train" / "labels" / f"{filename}.txt"
        
        print(f"\nProcessing: {filename}")
        
        if not source_path.exists():
            print(f"  Error: Source file {source_path} not found!")
            continue
        
        try:
            # Convert and save
            success = convert_label_file(source_path, target_path)
            if success:
                print(f"  ✓ Successfully converted and saved to {target_path}")
            else:
                print(f"  ✗ Failed to convert {filename}")
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
    
    print("\nDone! Run analyze_missing_labels.py again to verify.")

if __name__ == "__main__":
    main() 