#!/usr/bin/env python3
"""
Medical Image Complete Inference Script
Combines YOLOv5 detection with ResNet classification for heart ultrasound analysis
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.medical_classifier import create_medical_classifier
from models.yolo import Model
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


class MedicalCompleteInference:
    """
    Complete medical image inference combining detection and classification
    """
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = select_device(device)
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint['config']
        
        # Initialize models
        self.setup_models()
        
        # Load model weights
        self.load_weights()
        
        # Setup class names
        self.setup_class_names()
        
        print(f"Medical Complete Inference initialized with checkpoint: {checkpoint_path}")
    
    def setup_models(self):
        """Setup detection and classification models"""
        # Detection model (YOLOv5)
        detection_cfg = self.config.get('detection', {})
        self.detection_model = Model(
            cfg=detection_cfg.get('cfg', 'models/yolov5s.yaml'),
            ch=3,
            nc=self.config['nc'],
            anchors=None
        ).to(self.device)
        
        # Classification model (ResNet)
        classification_cfg = self.config.get('classification', {})
        self.classification_model = create_medical_classifier(
            classifier_type=classification_cfg.get('type', 'resnet'),
            num_classes=self.config.get('classification_classes', 4),
            pretrained=False  # Don't load pretrained weights since we have our own
        ).to(self.device)
    
    def load_weights(self):
        """Load model weights from checkpoint"""
        # Load detection model weights
        self.detection_model.load_state_dict(self.checkpoint['detection_model_state_dict'])
        
        # Load classification model weights
        self.classification_model.load_state_dict(self.checkpoint['classification_model_state_dict'])
        
        # Set models to evaluation mode
        self.detection_model.eval()
        self.classification_model.eval()
        
        print("Model weights loaded successfully")
    
    def setup_class_names(self):
        """Setup class names for detection and classification"""
        # Detection class names
        self.detection_class_names = self.config.get('detection_class_names', 
                                                    ['AR', 'MR', 'PR', 'TR'])
        
        # Classification class names
        self.classification_class_names = self.config.get('classification_class_names',
                                                         ['Normal', 'Mild', 'Moderate', 'Severe'])
        
        print(f"Detection classes: {self.detection_class_names}")
        print(f"Classification classes: {self.classification_class_names}")
    
    def preprocess_image(self, image_path, img_size=640):
        """Preprocess image for inference"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Store original image
        original_image = image.copy()
        
        # Resize and normalize
        image_resized = cv2.resize(image, (img_size, img_size))
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor, original_image, image_resized
    
    def run_detection(self, image_tensor, conf_thres=0.25, iou_thres=0.45):
        """Run detection inference"""
        with torch.no_grad():
            # Forward pass
            detection_outputs = self.detection_model(image_tensor)
            
            # Apply NMS
            detection_outputs = non_max_suppression(
                detection_outputs,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=1000
            )
        
        return detection_outputs[0]  # Return first (and only) batch
    
    def run_classification(self, image_tensor):
        """Run classification inference"""
        with torch.no_grad():
            # Forward pass
            classification_outputs = self.classification_model(image_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(classification_outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)
        
        return predicted_class.item(), confidence.item(), probabilities[0]
    
    def process_detections(self, detections, original_image, img_size=640):
        """Process detection results"""
        results = []
        
        if detections is not None and len(detections) > 0:
            # Scale coordinates back to original image size
            detections[:, :4] = scale_coords(
                (img_size, img_size), 
                detections[:, :4], 
                original_image.shape
            ).round()
            
            # Convert to list of dictionaries
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                
                result = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.detection_class_names[int(cls)]
                }
                results.append(result)
        
        return results
    
    def visualize_results(self, image, detections, classification_result, save_path=None):
        """Visualize detection and classification results"""
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot original image with detections
        ax1.imshow(image)
        ax1.set_title('Detection Results', fontsize=14, fontweight='bold')
        
        # Draw detection boxes
        colors = ['red', 'blue', 'green', 'orange']
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            color = colors[detection['class_id'] % len(colors)]
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            ax1.text(x1, y1-5, label, color=color, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        ax1.axis('off')
        
        # Plot classification results
        ax2.set_title('Classification Results', fontsize=14, fontweight='bold')
        
        # Create bar chart for classification probabilities
        class_names = self.classification_class_names
        probabilities = classification_result['probabilities'].cpu().numpy()
        
        bars = ax2.bar(class_names, probabilities, color=['green', 'yellow', 'orange', 'red'])
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add classification result text
        predicted_class = classification_result['predicted_class']
        confidence = classification_result['confidence']
        ax2.text(0.5, 0.9, f'Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.3f}',
                transform=ax2.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, detections, classification_result, image_path):
        """Generate medical report"""
        report = {
            'image_path': image_path,
            'detection_summary': {},
            'classification_result': {
                'predicted_class': self.classification_class_names[classification_result['predicted_class']],
                'confidence': classification_result['confidence'],
                'probabilities': {
                    name: float(prob) for name, prob in zip(
                        self.classification_class_names, 
                        classification_result['probabilities']
                    )
                }
            },
            'recommendations': []
        }
        
        # Count detections by class
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in report['detection_summary']:
                report['detection_summary'][class_name] = 0
            report['detection_summary'][class_name] += 1
        
        # Generate recommendations
        predicted_class = classification_result['predicted_class']
        if predicted_class == 0:  # Normal
            report['recommendations'].append("No significant heart valve regurgitation detected.")
            report['recommendations'].append("Continue regular monitoring as recommended by your physician.")
        elif predicted_class == 1:  # Mild
            report['recommendations'].append("Mild heart valve regurgitation detected.")
            report['recommendations'].append("Consider follow-up echocardiography in 6-12 months.")
        elif predicted_class == 2:  # Moderate
            report['recommendations'].append("Moderate heart valve regurgitation detected.")
            report['recommendations'].append("Recommend consultation with cardiologist.")
            report['recommendations'].append("Consider more frequent monitoring.")
        else:  # Severe
            report['recommendations'].append("Severe heart valve regurgitation detected.")
            report['recommendations'].append("Immediate cardiology consultation recommended.")
            report['recommendations'].append("Consider surgical evaluation.")
        
        return report
    
    def inference(self, image_path, conf_thres=0.25, iou_thres=0.45, 
                  save_visualization=True, save_report=True):
        """Run complete inference on image"""
        print(f"Processing image: {image_path}")
        
        # Preprocess image
        image_tensor, original_image, image_resized = self.preprocess_image(image_path)
        
        # Run detection
        detections = self.run_detection(image_tensor, conf_thres, iou_thres)
        detection_results = self.process_detections(detections, original_image)
        
        # Run classification
        predicted_class, confidence, probabilities = self.run_classification(image_tensor)
        classification_result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        }
        
        # Print results
        print(f"\nDetection Results:")
        for detection in detection_results:
            print(f"  {detection['class_name']}: {detection['confidence']:.3f}")
        
        print(f"\nClassification Results:")
        print(f"  Predicted: {self.classification_class_names[predicted_class]}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Probabilities:")
        for i, (name, prob) in enumerate(zip(self.classification_class_names, probabilities)):
            print(f"    {name}: {prob:.3f}")
        
        # Save visualization
        if save_visualization:
            save_path = Path(image_path).parent / f"{Path(image_path).stem}_results.png"
            self.visualize_results(original_image, detection_results, classification_result, save_path)
        
        # Generate and save report
        if save_report:
            report = self.generate_report(detection_results, classification_result, image_path)
            report_path = Path(image_path).parent / f"{Path(image_path).stem}_report.yaml"
            with open(report_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False)
            print(f"Report saved to: {report_path}")
        
        return detection_results, classification_result


def main():
    parser = argparse.ArgumentParser(description='Medical Complete Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='Detection IoU threshold')
    parser.add_argument('--no-visualization', action='store_true', help='Disable visualization')
    parser.add_argument('--no-report', action='store_true', help='Disable report generation')
    
    args = parser.parse_args()
    
    # Create inference engine
    inference_engine = MedicalCompleteInference(args.checkpoint, device=args.device)
    
    # Run inference
    detection_results, classification_result = inference_engine.inference(
        args.image,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        save_visualization=not args.no_visualization,
        save_report=not args.no_report
    )
    
    print("\nInference completed successfully!")


if __name__ == '__main__':
    main() 