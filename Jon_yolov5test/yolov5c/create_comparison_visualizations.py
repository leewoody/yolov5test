#!/usr/bin/env python3
"""
Create Detailed Comparison Visualizations
生成詳細的可視化比較圖像
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
import random
import os
from datetime import datetime

class ComparisonVisualizationGenerator:
    """Generate detailed comparison visualizations"""
    
    def __init__(self, original_dataset_dir, converted_dataset_dir, output_dir):
        self.original_dataset_dir = Path(original_dataset_dir)
        self.converted_dataset_dir = Path(converted_dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset configurations
        with open(self.original_dataset_dir / "data.yaml", 'r') as f:
            self.original_config = yaml.safe_load(f)
        
        with open(self.converted_dataset_dir / "data.yaml", 'r') as f:
            self.converted_config = yaml.safe_load(f)
        
        print(f"🎨 Comparison Visualization Generator initialized")
        print(f"📁 Original dataset: {self.original_dataset_dir}")
        print(f"📁 Converted dataset: {self.converted_dataset_dir}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def parse_segmentation_label(self, label_line):
        """Parse segmentation label format"""
        parts = label_line.strip().split()
        if len(parts) < 9:  # class_id + at least 4 points (8 coordinates)
            return None, []
        
        class_id = int(parts[0])
        # Extract polygon points (x1, y1, x2, y2, ...)
        coords = [float(x) for x in parts[1:]]
        
        # Group coordinates into points
        points = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                points.append((coords[i], coords[i + 1]))
        
        return class_id, points
    
    def parse_bbox_label(self, label_line):
        """Parse YOLO bounding box format"""
        parts = label_line.strip().split()
        if len(parts) != 5:
            return None, None, None, None, None
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        return class_id, x_center, y_center, width, height
    
    def convert_to_image_coords(self, normalized_coords, img_height, img_width):
        """Convert normalized coordinates to image coordinates"""
        image_coords = []
        for x, y in normalized_coords:
            img_x = int(x * img_width)
            img_y = int(y * img_height)
            image_coords.append((img_x, img_y))
        return image_coords
    
    def create_comparison_visualization(self, image_name):
        """Create 4-panel comparison visualization for a single image"""
        try:
            # Find image and labels
            image_base = image_name.replace('.png', '').replace('.jpg', '')
            
            # Search for image in train/val sets
            image_path = None
            original_label_path = None
            converted_label_path = None
            
            for split in ['train', 'valid', 'test']:
                # Check original dataset
                orig_img_path = self.original_dataset_dir / split / 'images' / f"{image_base}.png"
                if not orig_img_path.exists():
                    orig_img_path = self.original_dataset_dir / split / 'images' / f"{image_base}.jpg"
                
                if orig_img_path.exists():
                    image_path = orig_img_path
                    original_label_path = self.original_dataset_dir / split / 'labels' / f"{image_base}.txt"
                    
                    # Find corresponding converted label
                    conv_label_path = self.converted_dataset_dir / split / 'labels' / f"{image_base}.txt"
                    if conv_label_path.exists():
                        converted_label_path = conv_label_path
                    break
            
            if not image_path or not image_path.exists():
                print(f"❌ Image not found: {image_base}")
                return False
            
            if not original_label_path or not original_label_path.exists():
                print(f"❌ Original label not found: {original_label_path}")
                return False
            
            if not converted_label_path or not converted_label_path.exists():
                print(f"❌ Converted label not found: {converted_label_path}")
                return False
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                return False
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = image.shape[:2]
            
            # Parse labels
            with open(original_label_path, 'r') as f:
                original_lines = f.readlines()
            
            with open(converted_label_path, 'r') as f:
                converted_lines = f.readlines()
            
            # Create 4-panel figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Comparison: {image_base}', fontsize=16, fontweight='bold')
            
            # Panel 1: Original Image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Panel 2: Original Segmentation (Green)
            axes[0, 1].imshow(image)
            axes[0, 1].set_title('Original Segmentation (Green)', fontsize=14, fontweight='bold')
            
            seg_info = []
            for line in original_lines:
                if line.strip():
                    class_id, points = self.parse_segmentation_label(line)
                    if class_id is not None and points:
                        # Convert to image coordinates
                        img_points = self.convert_to_image_coords(points, img_height, img_width)
                        
                        # Draw segmentation polygon
                        if len(img_points) >= 3:
                            polygon = patches.Polygon(img_points, linewidth=2, 
                                                    edgecolor='green', facecolor='none')
                            axes[0, 1].add_patch(polygon)
                            
                            # Draw vertices as points
                            for point in img_points:
                                axes[0, 1].plot(point[0], point[1], 'go', markersize=4)
                        
                        seg_info.append(f"Seg points: {len(points)}")
            
            axes[0, 1].axis('off')
            
            # Panel 3: Converted Bounding Box (Red)
            axes[1, 0].imshow(image)
            axes[1, 0].set_title('Converted Bounding Box (Red)', fontsize=14, fontweight='bold')
            
            bbox_info = []
            for line in converted_lines:
                if line.strip():
                    class_id, x_center, y_center, width, height = self.parse_bbox_label(line)
                    if class_id is not None:
                        # Convert to image coordinates
                        x_center_img = x_center * img_width
                        y_center_img = y_center * img_height
                        width_img = width * img_width
                        height_img = height * img_height
                        
                        # Calculate bounding box corners
                        x_min = x_center_img - width_img / 2
                        y_min = y_center_img - height_img / 2
                        
                        # Draw bounding box
                        rect = patches.Rectangle((x_min, y_min), width_img, height_img,
                                               linewidth=2, edgecolor='red', facecolor='none')
                        axes[1, 0].add_patch(rect)
                        
                        # Draw center point
                        axes[1, 0].plot(x_center_img, y_center_img, 'ro', markersize=6)
                        
                        bbox_info.append(f"Bbox: ({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
            
            axes[1, 0].axis('off')
            
            # Panel 4: Overlay Comparison (Green + Red)
            axes[1, 1].imshow(image)
            axes[1, 1].set_title('Overlay Comparison (Green: Seg, Red: Bbox)', fontsize=14, fontweight='bold')
            
            # Draw segmentation polygons
            for line in original_lines:
                if line.strip():
                    class_id, points = self.parse_segmentation_label(line)
                    if class_id is not None and points:
                        img_points = self.convert_to_image_coords(points, img_height, img_width)
                        if len(img_points) >= 3:
                            polygon = patches.Polygon(img_points, linewidth=2, 
                                                    edgecolor='green', facecolor='none', alpha=0.7)
                            axes[1, 1].add_patch(polygon)
            
            # Draw bounding boxes
            for line in converted_lines:
                if line.strip():
                    class_id, x_center, y_center, width, height = self.parse_bbox_label(line)
                    if class_id is not None:
                        x_center_img = x_center * img_width
                        y_center_img = y_center * img_height
                        width_img = width * img_width
                        height_img = height * img_height
                        
                        x_min = x_center_img - width_img / 2
                        y_min = y_center_img - height_img / 2
                        
                        rect = patches.Rectangle((x_min, y_min), width_img, height_img,
                                               linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
                        axes[1, 1].add_patch(rect)
            
            axes[1, 1].axis('off')
            
            # Add information text
            info_text = f"""
Image: {image_path.name}
Original Label: {original_label_path.name}
Converted Label: {converted_label_path.name}

{chr(10).join(seg_info[:3])}
{chr(10).join(bbox_info[:3])}
"""
            
            fig.text(0.02, 0.02, info_text, fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # Save comparison image
            output_path = self.output_dir / f"comparison_{image_base}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Generated: {output_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")
            return False
    
    def generate_batch_comparisons(self, num_samples=50):
        """Generate comparison visualizations for multiple images"""
        print(f"🚀 Generating {num_samples} comparison visualizations...")
        
        # Collect all available images
        all_images = []
        for split in ['train', 'valid', 'test']:
            split_dir = self.original_dataset_dir / split / 'images'
            if split_dir.exists():
                for img_file in split_dir.glob('*.png'):
                    all_images.append(img_file.name)
                for img_file in split_dir.glob('*.jpg'):
                    all_images.append(img_file.name)
        
        print(f"📊 Found {len(all_images)} total images")
        
        # Sample random images
        if len(all_images) > num_samples:
            selected_images = random.sample(all_images, num_samples)
        else:
            selected_images = all_images
        
        # Generate comparisons
        success_count = 0
        for i, image_name in enumerate(selected_images):
            print(f"Processing {i+1}/{len(selected_images)}: {image_name}")
            if self.create_comparison_visualization(image_name):
                success_count += 1
        
        print(f"\n✅ Successfully generated {success_count}/{len(selected_images)} comparison visualizations")
        print(f"📁 Saved to: {self.output_dir}")
        
        return success_count

def main():
    """Main execution function"""
    print("🎨 Comparison Visualization Generator")
    print("=" * 50)
    
    # Configuration
    original_dataset = "/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/Regurgitation-YOLODataset-1new"
    converted_dataset = "/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/detection_only_dataset"
    output_dir = "/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/yolov5c/runs/comparison_results"
    
    # Create generator
    generator = ComparisonVisualizationGenerator(
        original_dataset_dir=original_dataset,
        converted_dataset_dir=converted_dataset,
        output_dir=output_dir
    )
    
    # Generate comparisons
    generator.generate_batch_comparisons(num_samples=100)

if __name__ == "__main__":
    main() 