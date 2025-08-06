import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import math

class DetectionBackbone(nn.Module):
    """
    Independent backbone for detection task
    Based on YOLOv5 backbone architecture
    """
    
    def __init__(self, in_channels=3, width_multiple=0.5, depth_multiple=0.33):
        super(DetectionBackbone, self).__init__()
        
        # Calculate channel numbers based on width multiple
        base_channels = int(64 * width_multiple)
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        
        # Calculate layer numbers based on depth multiple
        depths = [max(round(x * depth_multiple), 1) for x in [1, 3, 6, 9, 3]]
        
        # Detection backbone layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 6, 2, 2),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(inplace=True)
        )
        
        # C3 blocks for detection
        self.c3_1 = self._make_c3_block(channels[1], channels[1], depths[1])
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(inplace=True)
        )
        
        self.c3_2 = self._make_c3_block(channels[2], channels[2], depths[2])
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2),
            nn.BatchNorm2d(channels[3]),
            nn.SiLU(inplace=True)
        )
        
        self.c3_3 = self._make_c3_block(channels[3], channels[3], depths[3])
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, 2),
            nn.BatchNorm2d(channels[4]),
            nn.SiLU(inplace=True)
        )
        
        self.c3_4 = self._make_c3_block(channels[4], channels[4], depths[4])
        
        # SPPF for detection
        self.sppf = self._make_sppf(channels[4], channels[4])
        
    def _make_c3_block(self, in_channels, out_channels, depth):
        """Create C3 block for detection backbone"""
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(Bottleneck(in_channels, out_channels, shortcut=True))
            else:
                layers.append(Bottleneck(out_channels, out_channels, shortcut=True))
        return nn.Sequential(*layers)
    
    def _make_sppf(self, in_channels, out_channels):
        """Create SPPF module"""
        return SPPF(in_channels, out_channels, k=5)
    
    def forward(self, x):
        # Detection backbone forward pass
        x1 = self.conv1(x)      # P1/2
        x2 = self.conv2(x1)     # P2/4
        x2 = self.c3_1(x2)
        x3 = self.conv3(x2)     # P3/8
        x3 = self.c3_2(x3)
        x4 = self.conv4(x3)     # P4/16
        x4 = self.c3_3(x4)
        x5 = self.conv5(x4)     # P5/32
        x5 = self.c3_4(x5)
        x5 = self.sppf(x5)
        
        return [x1, x2, x3, x4, x5]

class ClassificationBackbone(nn.Module):
    """
    Independent backbone for classification task
    Optimized for classification with attention mechanisms
    """
    
    def __init__(self, in_channels=3, num_classes=4, width_multiple=0.5):
        super(ClassificationBackbone, self).__init__()
        
        # Calculate channel numbers
        base_channels = int(64 * width_multiple)
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        
        # Classification backbone with attention
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 2, 3),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            SE(channels[1])  # Squeeze-and-Excitation attention
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            CBAM(channels[2])  # CBAM attention
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            SE(channels[3])
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(channels[3], channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(channels[2], channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels[1], num_classes)
        )
        
    def forward(self, x):
        # Classification backbone forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class DualBackboneModel(nn.Module):
    """
    Dual backbone model with independent detection and classification backbones
    This eliminates gradient interference between tasks
    """
    
    def __init__(self, 
                 num_detection_classes=4, 
                 num_classification_classes=4,
                 width_multiple=0.5, 
                 depth_multiple=0.33):
        super(DualBackboneModel, self).__init__()
        
        # Independent backbones
        self.detection_backbone = DetectionBackbone(
            in_channels=3, 
            width_multiple=width_multiple, 
            depth_multiple=depth_multiple
        )
        
        self.classification_backbone = ClassificationBackbone(
            in_channels=3, 
            num_classes=num_classification_classes,
            width_multiple=width_multiple
        )
        
        # Detection head (YOLO head)
        self.detection_head = DetectionHead(
            in_channels=[256, 512, 1024],  # P3, P4, P5 channels
            num_classes=num_detection_classes,
            width_multiple=width_multiple
        )
        
        # Task-specific optimizers can be used
        self.detection_params = list(self.detection_backbone.parameters()) + list(self.detection_head.parameters())
        self.classification_params = list(self.classification_backbone.parameters())
        
    def forward(self, x):
        # Independent forward passes
        detection_features = self.detection_backbone(x)
        classification_output = self.classification_backbone(x)
        
        # Detection head forward pass
        detection_output = self.detection_head(detection_features)
        
        return detection_output, classification_output
    
    def get_task_parameters(self):
        """Get parameters for each task for separate optimization"""
        return {
            'detection': self.detection_params,
            'classification': self.classification_params
        }

class DetectionHead(nn.Module):
    """
    YOLO detection head for dual backbone model
    """
    
    def __init__(self, in_channels, num_classes, width_multiple=0.5):
        super(DetectionHead, self).__init__()
        
        # Calculate channels
        channels = [int(c * width_multiple) for c in in_channels]
        
        # Detection head layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], 1, 1),  # P5 -> P4
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(inplace=True)
        )
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], 1, 1),  # P4 -> P3
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Detection outputs
        self.detect_p3 = nn.Conv2d(channels[0], 3 * (5 + num_classes), 1)  # P3/8
        self.detect_p4 = nn.Conv2d(channels[1], 3 * (5 + num_classes), 1)  # P4/16
        self.detect_p5 = nn.Conv2d(channels[2], 3 * (5 + num_classes), 1)  # P5/32
        
    def forward(self, features):
        # features: [x1, x2, x3, x4, x5]
        x1, x2, x3, x4, x5 = features
        
        # P5 -> P4
        p5 = self.conv1(x5)
        p5_up = self.upsample1(p5)
        p4 = torch.cat([p5_up, x4], dim=1)
        
        # P4 -> P3
        p4 = self.conv2(p4)
        p4_up = self.upsample2(p4)
        p3 = torch.cat([p4_up, x3], dim=1)
        
        # Detection outputs
        out_p3 = self.detect_p3(p3)
        out_p4 = self.detect_p4(p4)
        out_p5 = self.detect_p5(p5)
        
        return [out_p3, out_p4, out_p5]

# Supporting modules
class Bottleneck(nn.Module):
    """Bottleneck module for C3 blocks"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c2, 3, 1, padding=1, groups=g)
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SPPF(nn.Module):
    """SPPF module for detection backbone"""
    
    def __init__(self, c1, c2, k=5):
        super(SPPF, self).__init__()
        c_ = c1 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class SE(nn.Module):
    """Squeeze-and-Excitation attention module"""
    
    def __init__(self, c1, c2=None, ratio=16):
        super(SE, self).__init__()
        c2 = c2 or c1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // ratio, c1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """CBAM attention module"""
    
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class ChannelAttention(nn.Module):
    """Channel attention for CBAM"""
    
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial attention for CBAM"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class DualBackboneJointTrainer:
    """
    Joint trainer for dual backbone model with separate optimizers
    """
    
    def __init__(self, 
                 model: DualBackboneModel,
                 detection_lr: float = 1e-4,
                 classification_lr: float = 1e-4,
                 detection_weight: float = 1.0,
                 classification_weight: float = 0.01):
        
        self.model = model
        self.detection_weight = detection_weight
        self.classification_weight = classification_weight
        
        # Separate optimizers for each task
        task_params = model.get_task_parameters()
        
        self.detection_optimizer = torch.optim.AdamW(
            task_params['detection'], 
            lr=detection_lr,
            weight_decay=1e-4
        )
        
        self.classification_optimizer = torch.optim.AdamW(
            task_params['classification'], 
            lr=classification_lr,
            weight_decay=1e-4
        )
        
        # Separate schedulers
        self.detection_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.detection_optimizer, T_max=100
        )
        
        self.classification_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.classification_optimizer, T_max=100
        )
        
    def train_step(self, 
                   images: torch.Tensor,
                   detection_targets: torch.Tensor,
                   classification_targets: torch.Tensor,
                   detection_loss_fn: nn.Module,
                   classification_loss_fn: nn.Module) -> Dict:
        """
        Training step with separate optimization
        """
        
        # Forward pass
        detection_outputs, classification_outputs = self.model(images)
        
        # Compute losses
        detection_loss = detection_loss_fn(detection_outputs, detection_targets)
        classification_loss = classification_loss_fn(classification_outputs, classification_targets)
        
        # Weighted losses
        weighted_detection_loss = self.detection_weight * detection_loss
        weighted_classification_loss = self.classification_weight * classification_loss
        
        # Separate backward passes
        # Detection backward
        self.detection_optimizer.zero_grad()
        weighted_detection_loss.backward(retain_graph=True)
        self.detection_optimizer.step()
        
        # Classification backward
        self.classification_optimizer.zero_grad()
        weighted_classification_loss.backward()
        self.classification_optimizer.step()
        
        # Update schedulers
        self.detection_scheduler.step()
        self.classification_scheduler.step()
        
        return {
            'detection_loss': detection_loss.item(),
            'classification_loss': classification_loss.item(),
            'weighted_detection_loss': weighted_detection_loss.item(),
            'weighted_classification_loss': weighted_classification_loss.item(),
            'total_loss': (weighted_detection_loss + weighted_classification_loss).item()
        } 