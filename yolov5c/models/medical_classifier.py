#!/usr/bin/env python3
"""
Medical Image Classification Model
Specialized for heart ultrasound severity classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MedicalClassifier(nn.Module):
    """
    Medical image classifier for heart ultrasound severity assessment
    """
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.3):
        super(MedicalClassifier, self).__init__()
        
        # Use ResNet50 as backbone (good for medical images)
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Medical-specific classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class MedicalAttentionClassifier(nn.Module):
    """
    Medical classifier with attention mechanism for better feature focus
    """
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.3):
        super(MedicalAttentionClassifier, self).__init__()
        
        # ResNet backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Keep spatial dimensions
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with attention"""
        features = self.backbone(x)  # [B, 2048, H, W]
        
        # Apply attention
        attention_weights = self.attention(features)  # [B, 1, H, W]
        attended_features = features * attention_weights  # [B, 2048, H, W]
        
        # Classification
        output = self.classifier(attended_features)
        return output, attention_weights


class MedicalMultiScaleClassifier(nn.Module):
    """
    Multi-scale medical classifier for better feature extraction
    """
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.3):
        super(MedicalMultiScaleClassifier, self).__init__()
        
        # ResNet backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Multi-scale feature extraction
        self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])   # 64 channels
        self.layer2 = nn.Sequential(*list(self.backbone.children())[5:6])  # 128 channels
        self.layer3 = nn.Sequential(*list(self.backbone.children())[6:7])  # 256 channels
        self.layer4 = nn.Sequential(*list(self.backbone.children())[7:8])  # 512 channels
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with multi-scale features"""
        # Extract features at different scales
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        # Use final features for classification
        features = self.fusion(f4)
        output = self.classifier(features)
        
        return output


def create_medical_classifier(classifier_type='resnet', num_classes=4, pretrained=True):
    """
    Factory function to create medical classifiers
    
    Args:
        classifier_type: Type of classifier ('resnet', 'attention', 'multiscale')
        num_classes: Number of classification classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Medical classifier model
    """
    if classifier_type == 'resnet':
        return MedicalClassifier(num_classes=num_classes, pretrained=pretrained)
    elif classifier_type == 'attention':
        return MedicalAttentionClassifier(num_classes=num_classes, pretrained=pretrained)
    elif classifier_type == 'multiscale':
        return MedicalMultiScaleClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


class MedicalLoss(nn.Module):
    """
    Custom loss function for medical classification
    """
    def __init__(self, class_weights=None, focal_alpha=1.0, focal_gamma=2.0):
        super(MedicalLoss, self).__init__()
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, predictions, targets):
        """
        Compute medical classification loss
        
        Args:
            predictions: Model predictions [B, num_classes]
            targets: Ground truth labels [B]
        
        Returns:
            Loss value
        """
        # Focal loss for handling class imbalance
        ce_loss = F.cross_entropy(predictions, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss.mean()


def get_medical_classifier_loss(class_weights=None):
    """
    Get medical classifier loss function
    
    Args:
        class_weights: Class weights for handling imbalance
    
    Returns:
        Loss function
    """
    return MedicalLoss(class_weights=class_weights) 