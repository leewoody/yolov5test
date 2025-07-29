#!/usr/bin/env python3
"""
Medical Image Data Loader
Specialized for heart ultrasound images with detection and classification labels
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MedicalImageDataset(Dataset):
    """
    Medical image dataset for heart ultrasound detection and classification
    """
    def __init__(self, 
                 data_yaml_path,
                 mode='train',
                 img_size=640,
                 augment=True,
                 medical_mode=True):
        """
        Initialize medical image dataset
        
        Args:
            data_yaml_path: Path to data configuration YAML
            mode: 'train', 'val', or 'test'
            img_size: Input image size
            augment: Whether to apply augmentation
            medical_mode: Medical image specific settings
        """
        self.mode = mode
        self.img_size = img_size
        self.augment = augment
        self.medical_mode = medical_mode
        
        # Load data configuration
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Get image paths
        if mode == 'train':
            self.img_dir = Path(self.data_config['train'])
        elif mode == 'val':
            self.img_dir = Path(self.data_config['val'])
        elif mode == 'test':
            self.img_dir = Path(self.data_config['test'])
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Get label directory
        self.label_dir = self.img_dir.parent / 'labels' / self.img_dir.name
        
        # Get image files
        self.img_files = sorted([f for f in self.img_dir.glob('*') 
                               if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        
        # Load class names
        self.class_names = self.data_config['names']
        self.num_classes = len(self.class_names)
        
        # Classification classes
        self.classification_names = self.data_config.get('classification_names', 
                                                        ['Normal', 'Mild', 'Moderate', 'Severe'])
        self.num_classification_classes = len(self.classification_names)
        
        # Setup augmentation
        self.setup_augmentation()
        
        print(f"Loaded {len(self.img_files)} {mode} images from {self.img_dir}")
    
    def setup_augmentation(self):
        """Setup medical image specific augmentation"""
        if self.medical_mode:
            # Conservative augmentation for medical images
            self.transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.HorizontalFlip(p=0.3),  # Only horizontal flip
                A.RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1),
                A.HueSaturationValue(p=0.1, hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # Standard augmentation
            self.transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        img_path = self.img_files[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load detection labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 5:
                        class_id = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        # Convert to numpy arrays
        bboxes = np.array(bboxes, dtype=np.float32)
        class_labels = np.array(class_labels, dtype=np.int64)
        
        # Apply augmentation
        if self.augment and len(bboxes) > 0:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
            bboxes = np.array(transformed['bboxes'], dtype=np.float32)
            class_labels = np.array(transformed['class_labels'], dtype=np.int64)
        else:
            # Only resize and normalize
            transformed = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])(image=image)
            image = transformed['image']
        
        # Create detection targets
        detection_targets = torch.zeros((len(bboxes), 6))  # [class_id, x_center, y_center, width, height, img_id]
        if len(bboxes) > 0:
            detection_targets[:, 0] = torch.from_numpy(class_labels)  # class_id
            detection_targets[:, 1:5] = torch.from_numpy(bboxes)      # bbox coordinates
            detection_targets[:, 5] = idx                             # image_id
        
        # Generate classification label (simplified - you may need to adapt this)
        # For now, we'll use the presence of any detection as a proxy for severity
        if len(bboxes) == 0:
            classification_label = 0  # Normal
        elif len(bboxes) == 1:
            classification_label = 1  # Mild
        elif len(bboxes) == 2:
            classification_label = 2  # Moderate
        else:
            classification_label = 3  # Severe
        
        return {
            'image': image,
            'detection_targets': detection_targets,
            'classification_label': torch.tensor(classification_label, dtype=torch.long),
            'image_path': str(img_path),
            'bboxes': bboxes,
            'class_labels': class_labels
        }


class MedicalDataLoader:
    """
    Medical data loader factory
    """
    def __init__(self, data_yaml_path, batch_size=16, num_workers=4, img_size=640):
        self.data_yaml_path = data_yaml_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
    
    def get_train_loader(self):
        """Get training data loader"""
        dataset = MedicalImageDataset(
            data_yaml_path=self.data_yaml_path,
            mode='train',
            img_size=self.img_size,
            augment=True,
            medical_mode=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def get_val_loader(self):
        """Get validation data loader"""
        dataset = MedicalImageDataset(
            data_yaml_path=self.data_yaml_path,
            mode='val',
            img_size=self.img_size,
            augment=False,
            medical_mode=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def get_test_loader(self):
        """Get test data loader"""
        dataset = MedicalImageDataset(
            data_yaml_path=self.data_yaml_path,
            mode='test',
            img_size=self.img_size,
            augment=False,
            medical_mode=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        """Custom collate function for medical data"""
        images = torch.stack([item['image'] for item in batch])
        detection_targets = [item['detection_targets'] for item in batch]
        classification_labels = torch.stack([item['classification_label'] for item in batch])
        image_paths = [item['image_path'] for item in batch]
        
        # Concatenate detection targets
        if any(len(targets) > 0 for targets in detection_targets):
            detection_targets = torch.cat(detection_targets, dim=0)
        else:
            detection_targets = torch.empty((0, 6))
        
        return {
            'images': images,
            'detection_targets': detection_targets,
            'classification_labels': classification_labels,
            'image_paths': image_paths
        }


def create_medical_dataloader(data_yaml_path, batch_size=16, num_workers=4, img_size=640):
    """
    Factory function to create medical data loaders
    
    Args:
        data_yaml_path: Path to data configuration YAML
        batch_size: Batch size
        num_workers: Number of workers for data loading
        img_size: Input image size
    
    Returns:
        MedicalDataLoader instance
    """
    return MedicalDataLoader(
        data_yaml_path=data_yaml_path,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size
    ) 