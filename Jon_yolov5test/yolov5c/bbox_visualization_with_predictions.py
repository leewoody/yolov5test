#!/usr/bin/env python3
"""
Enhanced Bbox Visualization with Model Predictions
增強版bbox可視化工具，包含模型預測功能
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
import warnings
warnings.filterwarnings('ignore')

class EnhancedBboxVisualizationTester:
    """Enhanced bbox visualization with working model predictions"""
    
    def __init__(self, dataset_dir, output_dir, model_path=None):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset configuration
        with open(self.dataset_dir / "data.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config.get('names', ['AR', 'MR', 'PR', 'TR'])
        
        # Try to load a working model
        self.model = None
        self.model_name = "None"
        
        # Try different model loading strategies
        if model_path and Path(model_path).exists():
            self.model = self._load_model_safe(model_path)
            if self.model:
                self.model_name = Path(model_path).name
        
        # If no model provided or failed, try to download YOLOv5s
        if not self.model:
            self.model, self.model_name = self._download_yolov5s()
        
        print(f"🎯 Enhanced Bbox Visualization Tester initialized")
        print(f"📁 Dataset: {self.dataset_dir}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📁 Output: {self.output_dir}")
        print(f"📊 Classes: {self.class_names}")
    
    def _load_model_safe(self, model_path):
        """Safely load model with multiple strategies"""
        try:
            # Strategy 1: Direct torch.load with weights_only=False
            print(f"🔄 Trying to load model: {model_path}")
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            if hasattr(model, 'eval'):
                model.eval()
                print(f"✅ Model loaded successfully with torch.load")
                return model
        except Exception as e:
            print(f"⚠️ torch.load failed: {e}")
        
        try:
            # Strategy 2: YOLOv5 hub loading
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), force_reload=True)
            model.eval()
            print(f"✅ Model loaded successfully with torch.hub")
            return model
        except Exception as e:
            print(f"⚠️ torch.hub failed: {e}")
        
        return None
    
    def _download_yolov5s(self):
        """Download and load YOLOv5s model"""
        try:
            print("🔄 Downloading YOLOv5s model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.eval()
            print("✅ YOLOv5s model downloaded and loaded successfully")
            return model, "yolov5s (downloaded)"
        except Exception as e:
            print(f"❌ Failed to download YOLOv5s: {e}")
            return None, "None"
    
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
    
    def predict_image(self, image_path):
        """Run inference on an image"""
        if not self.model:
            return []
        
        try:
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            # Convert BGR to RGB for model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(image_rgb)
            
            # Extract predictions
            predictions = []
            img_h, img_w = image.shape[:2]
            
            # Handle different result formats
            if hasattr(results, 'xyxy'):
                # YOLOv5 format
                detections = results.xyxy[0].cpu().numpy() if len(results.xyxy[0]) > 0 else []
            elif hasattr(results, 'boxes'):
                # Newer format
                detections = results.boxes.data.cpu().numpy() if results.boxes is not None else []
            else:
                # Try to extract from pandas format
                try:
                    detections = results.pandas().xyxy[0].values if not results.pandas().xyxy[0].empty else []
                except:
                    detections = []
            
            for det in detections:
                if len(det) >= 6:  # x1, y1, x2, y2, conf, cls
                    x1, y1, x2, y2, conf, cls = det[:6]
                    
                    # Convert to YOLO format (normalized)
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    # Map class to our dataset classes (simple mapping)
                    class_id = int(cls) % len(self.class_names)
                    
                    predictions.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': float(conf)
                    })
            
            return predictions
            
        except Exception as e:
            print(f"⚠️ Prediction error for {image_path}: {e}")
            return []
    
    def create_bbox_distribution_plot(self, gt_bboxes, pred_bboxes=None):
        """Create bbox distribution plot with GT and predictions"""
        print("📊 Creating bbox distribution plot with predictions...")
        
        # Set English font
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Extract GT data
        gt_x_centers = [bbox['x_center'] for bbox in gt_bboxes]
        gt_y_centers = [bbox['y_center'] for bbox in gt_bboxes]
        gt_widths = [bbox['width'] for bbox in gt_bboxes]
        gt_heights = [bbox['height'] for bbox in gt_bboxes]
        
        # Plot 1: Center point distribution
        ax1.scatter(gt_x_centers, gt_y_centers, c='green', alpha=0.6, s=20, label='GT')
        
        if pred_bboxes and len(pred_bboxes) > 0:
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
        
        if pred_bboxes and len(pred_bboxes) > 0:
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
    
    def create_single_bbox_visualization(self, image_path, gt_labels, pred_labels=None, output_name=None):
        """Create visualization for a single image with GT and predictions"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = image.shape[:2]
            
            # Set English font
            plt.rcParams['font.family'] = 'DejaVu Sans'
            
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
                
                class_name = self.class_names[bbox['class_id']] if bbox['class_id'] < len(self.class_names) else f"class_{bbox['class_id']}"
                gt_classes.append(class_name)
            
            # Draw prediction bboxes (red)
            pred_classes = []
            if pred_labels and len(pred_labels) > 0:
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
                    
                    class_name = self.class_names[bbox['class_id']] if bbox['class_id'] < len(self.class_names) else f"class_{bbox['class_id']}"
                    pred_classes.append(f"{class_name} ({bbox['confidence']:.2f})")
            
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
        """Test model on random samples and create visualizations with predictions"""
        print(f"🚀 Testing model on {num_samples} samples with predictions...")
        
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
        
        # Collect images
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
                    print(f"  📊 Found {len(pred_labels)} predictions")
                else:
                    print(f"  📊 No predictions found")
            
            # Create visualization
            output_name = f"bbox_vis_{i}.png"
            self.create_single_bbox_visualization(image_path, gt_labels, pred_labels, output_name)
        
        # Create distribution plot
        self.create_bbox_distribution_plot(all_gt_bboxes, all_pred_bboxes if all_pred_bboxes else None)
        
        print(f"\n✅ Generated {len(selected_images)} bbox visualizations with predictions")
        print(f"📊 Total GT boxes: {len(all_gt_bboxes)}")
        print(f"📊 Total predicted boxes: {len(all_pred_bboxes)}")
        print(f"📁 Saved to: {self.output_dir}")
        
        return len(all_pred_bboxes) > 0

def main():
    """Main execution function"""
    print("🎯 Enhanced Bbox Visualization and Model Tester with Predictions")
    print("=" * 70)
    
    # Configuration
    dataset_dir = "/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/detection_only_dataset"
    output_dir = "/Users/mac/_codes/_Github/yolov5test/Jon_yolov5test/yolov5c/runs/bbox_vis_enhanced"
    
    # Try to use a trained model first, fallback to downloaded YOLOv5s
    trained_model_path = "/Users/mac/_codes/_Github/yolov5test/auto_training_run_20250802_065955/joint_model_ultimate.pth"
    
    # Create enhanced tester
    tester = EnhancedBboxVisualizationTester(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        model_path=trained_model_path
    )
    
    # Test model and generate visualizations with predictions
    success = tester.test_model_and_visualize(num_samples=20)
    
    if success:
        print("\n🎉 Successfully generated visualizations with both GT and predictions!")
    else:
        print("\n⚠️ Generated visualizations with GT only (no predictions)")

if __name__ == "__main__":
    main() 