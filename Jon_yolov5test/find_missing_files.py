import os
from pathlib import Path

def find_missing_files():
    # Paths
    images_dir = Path("Regurgitation-YOLODataset-Detection/train/images")
    labels_dir = Path("Regurgitation-YOLODataset-Detection/train/labels")
    
    # Get all image files
    image_files = set()
    for img_file in images_dir.glob("*.png"):
        # Convert image filename to expected label filename
        label_filename = img_file.stem + ".txt"
        image_files.add(label_filename)
    
    # Get all label files
    label_files = set()
    for label_file in labels_dir.glob("*.txt"):
        label_files.add(label_file.name)
    
    # Find missing label files
    missing_labels = image_files - label_files
    
    print(f"Total image files: {len(image_files)}")
    print(f"Total label files: {len(label_files)}")
    print(f"Missing label files: {len(missing_labels)}")
    
    if missing_labels:
        print("\nMissing label files:")
        for missing in sorted(missing_labels):
            print(f"  - {missing}")
    
    # Also check for orphaned label files (labels without images)
    orphaned_labels = label_files - image_files
    if orphaned_labels:
        print(f"\nOrphaned label files (no corresponding image): {len(orphaned_labels)}")
        for orphaned in sorted(orphaned_labels):
            print(f"  - {orphaned}")

if __name__ == "__main__":
    find_missing_files() 