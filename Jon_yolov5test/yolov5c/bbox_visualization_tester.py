#!/usr/bin/env python3
"""
Bbox Visualization and Model Testing
生成bbox分布圖和模型測試可視化
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
import random
import os
from datetime import datetime
import sys

# Add yolov5 to path for model loading
sys.path.append('/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/yolov5')

class BboxVisualizationTester:
    """Generate bbox distribution plots and model testing visualizations"""
    
    def __init__(self, dataset_dir, model_path, output_dir):
        self.dataset_dir = Path(dataset_dir)
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset configuration
        with open(self.dataset_dir / "data.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config.get('names', ['class_0', 'class_1', 'class_2', 'class_3'])
        
        # Load model if provided
        self.model = None
        if self.model_path and self.model_path.exists():
            try:
                self.model = torch.hub.load('/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/yolov5', 
                                          'custom', path=str(self.model_path), source='local')
                self.model.eval()
                print(f"✅ Model loaded: {self.model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                # Try alternative loading method
                try:
                    from yolov5.models.experimental import attempt_load
                    self.model = attempt_load(str(self.model_path))
                    self.model.eval()
                    print(f"✅ Model loaded with alternative method")
                except Exception as e2:
                    print(f"❌ Model loading failed: {e2}")
                    self.model = None
        
        print(f"🎯 Bbox Visualization Tester initialized")
        print(f"📁 Dataset: {self.dataset_dir}")
        print(f"🤖 Model: {self.model_path if self.model_path else 'None'}")
        print(f"📁 Output: {self.output_dir}")
    
    def parse_yolo_label(self, label_line):
        """Parse YOLO format label"""
        parts = label_line.strip().split()
        if len(parts) != 5:
            return None
        
        return {
            'class_id': int(parts[0]),
            'x_center': float(parts[1]),
            'y_center': float(parts[2]),
            'width': float(parts[3]),
            'height': float(parts[4])
        }
    
    def collect_all_bboxes(self):
        """Collect all bounding boxes from dataset"""
        print("📊 Collecting all bounding boxes...")
        
        all_bboxes = []
        for split in ['train', 'valid', 'test']:
            labels_dir = self.dataset_dir / split / 'labels'
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                bbox = self.parse_yolo_label(line)
                                if bbox:
                                    bbox['split'] = split
                                    bbox['file'] = label_file.name
                                    all_bboxes.append(bbox)
                    except Exception as e:
                        print(f"⚠️ Error reading {label_file}: {e}")
        
        print(f"✅ Collected {len(all_bboxes)} bounding boxes")
        return all_bboxes
    
    def create_bbox_distribution_plot(self, gt_bboxes, pred_bboxes=None):
        """Create bbox distribution plot similar to reference"""
        print("📊 Creating bbox distribution plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Extract GT data
        gt_x_centers = [bbox['x_center'] for bbox in gt_bboxes]
        gt_y_centers = [bbox['y_center'] for bbox in gt_bboxes]
        gt_widths = [bbox['width'] for bbox in gt_bboxes]
        gt_heights = [bbox['height'] for bbox in gt_bboxes]
        
        # Plot 1: Center point distribution
        ax1.scatter(gt_x_centers, gt_y_centers, c='green', alpha=0.6, s=20, label='GT')
        
        if pred_bboxes:
            pred_x_centers = [bbox['x_center'] for bbox in pred_bboxes]
            pred_y_centers = [bbox['y_center'] for bbox in pred_bboxes]
            ax1.scatter(pred_x_centers, pred_y_centers, c='red', alpha=0.6, s=20, label='Pred')
        
        ax1.set_xlabel('x_center')
        ax1.set_ylabel('y_center')
        ax1.set_title('Center Point Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Width-Height distribution
        ax2.scatter(gt_widths, gt_heights, c='green', alpha=0.6, s=20, label='GT')
        
        if pred_bboxes:
            pred_widths = [bbox['width'] for bbox in pred_bboxes]
            pred_heights = [bbox['height'] for bbox in pred_bboxes]
            ax2.scatter(pred_widths, pred_heights, c='red', alpha=0.6, s=20, label='Pred')
        
        ax2.set_xlabel('width')
        ax2.set_ylabel('height')
        ax2.set_title('Width-Height Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'bbox_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
        return output_path
    
    def predict_image(self, image_path):
        """Run inference on an image"""
        if not self.model:
            return None
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Run inference
            results = self.model(image)
            
            # Extract predictions
            predictions = []
            if len(results.xyxy[0]) > 0:
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                    
                    # Convert to YOLO format
                    img_h, img_w = image.shape[:2]
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    predictions.append({
                        'class_id': int(cls),
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': float(conf)
                    })
            
            return predictions
            
        except Exception as e:
            print(f"⚠️ Prediction error for {image_path}: {e}")
            return None
    
    def create_single_bbox_visualization(self, image_path, gt_labels, pred_labels=None, output_name=None):
        """Create visualization for a single image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = image.shape[:2]
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            
            # Draw GT bboxes (green)
            gt_classes = []
            for bbox in gt_labels:
                x_center = bbox['x_center'] * img_width
                y_center = bbox['y_center'] * img_height
                width = bbox['width'] * img_width
                height = bbox['height'] * img_height
                
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                
                rect = patches.Rectangle((x_min, y_min), width, height,
                                       linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
                gt_classes.append(self.class_names[bbox['class_id']])
            
            # Draw prediction bboxes (red)
            pred_classes = []
            if pred_labels:
                for bbox in pred_labels:
                    x_center = bbox['x_center'] * img_width
                    y_center = bbox['y_center'] * img_height
                    width = bbox['width'] * img_width
                    height = bbox['height'] * img_height
                    
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    
                    rect = patches.Rectangle((x_min, y_min), width, height,
                                           linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    pred_classes.append(f"{self.class_names[bbox['class_id']]} ({bbox['confidence']:.2f})")
            
            # Set title
            gt_str = ', '.join(gt_classes) if gt_classes else 'None'
            pred_str = ', '.join(pred_classes) if pred_classes else 'None'
            ax.set_title(f'GT: {gt_str}, Pred: {pred_str}', fontsize=12, fontweight='bold')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='green', lw=2, label='GT'),
                             Line2D([0], [0], color='red', lw=2, label='Pred')]
            ax.legend(handles=legend_elements, loc='upper right')
            
            ax.axis('off')
            plt.tight_layout()
            
            # Save visualization
            if not output_name:
                output_name = f"bbox_vis_{image_path.stem}.png"
            output_path = self.output_dir / output_name
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Generated: {output_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization for {image_path}: {e}")
            return False
    
    def test_model_and_visualize(self, num_samples=20):
        """Test model on random samples and create visualizations"""
        print(f"🚀 Testing model on {num_samples} samples...")
        
        # Collect all test images
        test_images = []
        test_dir = self.dataset_dir / 'test'
        if not test_dir.exists():
            test_dir = self.dataset_dir / 'valid'
        if not test_dir.exists():
            test_dir = self.dataset_dir / 'train'
        
        images_dir = test_dir / 'images'
        labels_dir = test_dir / 'labels'
        
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return
        
        for img_file in images_dir.glob('*.png'):
            test_images.append(img_file)
        for img_file in images_dir.glob('*.jpg'):
            test_images.append(img_file)
        
        print(f"📊 Found {len(test_images)} test images")
        
        # Sample random images
        if len(test_images) > num_samples:
            selected_images = random.sample(test_images, num_samples)
        else:
            selected_images = test_images
        
        # Process each image
        all_gt_bboxes = []
        all_pred_bboxes = []
        
        for i, image_path in enumerate(selected_images):
            print(f"Processing {i+1}/{len(selected_images)}: {image_path.name}")
            
            # Load GT labels
            label_path = labels_dir / f"{image_path.stem}.txt"
            gt_labels = []
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            bbox = self.parse_yolo_label(line)
                            if bbox:
                                gt_labels.append(bbox)
                                all_gt_bboxes.append(bbox)
                except Exception as e:
                    print(f"⚠️ Error reading GT labels: {e}")
            
            # Get predictions
            pred_labels = []
            if self.model:
                pred_labels = self.predict_image(image_path)
                if pred_labels:
                    all_pred_bboxes.extend(pred_labels)
            
            # Create visualization
            output_name = f"bbox_vis_{i}.png"
            self.create_single_bbox_visualization(image_path, gt_labels, pred_labels, output_name)
        
        # Create distribution plot
        self.create_bbox_distribution_plot(all_gt_bboxes, all_pred_bboxes if all_pred_bboxes else None)
        
        print(f"\n✅ Generated {len(selected_images)} bbox visualizations")
        print(f"📁 Saved to: {self.output_dir}")
    
    def generate_gt_only_visualizations(self, num_samples=20):
        """Generate visualizations using GT data only (when no model available)"""
        print(f"📊 Generating GT-only visualizations for {num_samples} samples...")
        
        # Collect all GT bboxes
        all_gt_bboxes = self.collect_all_bboxes()
        
        # Create distribution plot
        self.create_bbox_distribution_plot(all_gt_bboxes)
        
        # Generate sample visualizations
        test_images = []
        for split in ['test', 'valid', 'train']:
            split_dir = self.dataset_dir / split / 'images'
            if split_dir.exists():
                for img_file in split_dir.glob('*.png'):
                    test_images.append((img_file, split))
                for img_file in split_dir.glob('*.jpg'):
                    test_images.append((img_file, split))
                if test_images:
                    break
        
        if len(test_images) > num_samples:
            selected_images = random.sample(test_images, num_samples)
        else:
            selected_images = test_images
        
        for i, (image_path, split) in enumerate(selected_images):
            print(f"Processing {i+1}/{len(selected_images)}: {image_path.name}")
            
            # Load GT labels
            label_path = self.dataset_dir / split / 'labels' / f"{image_path.stem}.txt"
            gt_labels = []
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            bbox = self.parse_yolo_label(line)
                            if bbox:
                                gt_labels.append(bbox)
                except Exception as e:
                    print(f"⚠️ Error reading GT labels: {e}")
            
            # Create visualization (GT only)
            output_name = f"bbox_vis_{i}.png"
            self.create_single_bbox_visualization(image_path, gt_labels, None, output_name)
        
        print(f"\n✅ Generated {len(selected_images)} GT-only visualizations")

def main():
    """Main execution function"""
    print("🎯 Bbox Visualization and Model Tester")
    print("=" * 50)
    
    # Configuration
    dataset_dir = "/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/detection_only_dataset"
    model_path = "/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/yolov5s.pt"  # Default YOLOv5s
    output_dir = "/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/yolov5c/runs/bbox_vis"
    
    # Create tester
    tester = BboxVisualizationTester(
        dataset_dir=dataset_dir,
        model_path=model_path,
        output_dir=output_dir
    )
    
    # Test model and generate visualizations
    if tester.model:
        tester.test_model_and_visualize(num_samples=20)
    else:
        print("⚠️ No model available, generating GT-only visualizations")
        tester.generate_gt_only_visualizations(num_samples=20)

if __name__ == "__main__":
    main() 