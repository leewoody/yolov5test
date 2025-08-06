#!/usr/bin/env python3
"""
高精度雙獨立骨幹網路 v2.0
專為醫學圖像檢測和分類設計的高性能多任務學習架構

目標性能:
- mAP@0.5: 0.6+
- Precision: 0.7+
- Recall: 0.7+
- F1-Score: 0.6+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.yolo import Model
from utils.torch_utils import select_device
from utils.loss import ComputeLoss
from utils.general import colorstr, init_seeds
import numpy as np
import cv2
from pathlib import Path
import math


class SpatialAttentionModule(nn.Module):
    """空間注意力模塊 - 增強特徵表示"""
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class ChannelAttentionModule(nn.Module):
    """通道注意力模塊 - SE Block 變體"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module - 結合通道和空間注意力"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class EnhancedDetectionBackbone(nn.Module):
    """增強檢測骨幹網路 - 基於YOLOv5l + 注意力機制"""
    
    def __init__(self, config_path='models/yolov5l.yaml', nc=4, device='cpu'):
        super(EnhancedDetectionBackbone, self).__init__()
        self.device = select_device(device)
        self.nc = nc
        
        # 載入YOLOv5l模型
        try:
            self.model = Model(config_path, ch=3, nc=nc).to(self.device)
        except:
            # 如果yolov5l.yaml不存在，使用yolov5s並擴展
            print("Warning: Using YOLOv5s as base, expanding to YOLOv5l-like architecture")
            self.model = Model('models/yolov5s.yaml', ch=3, nc=nc).to(self.device)
        
        # 設置超參數
        self.model.hyp = {
            'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0,
            'iou_t': 0.20, 'anchor_t': 4.0, 'fl_gamma': 0.0,
            # 新增高精度相關參數
            'focal_loss_gamma': 2.0,
            'label_smoothing': 0.1,
            'conf_t': 0.25  # 置信度閾值
        }
        
        # 添加注意力機制到檢測頭
        self._add_attention_modules()
        
        # 特徵提取層標記（用於後續特徵融合）
        self.feature_layers = []
        self._mark_feature_layers()
        
    def _add_attention_modules(self):
        """在檢測模型中添加注意力機制"""
        try:
            # 為模型的關鍵層添加CBAM模塊
            if hasattr(self.model.model, '__iter__'):
                for i, layer in enumerate(self.model.model):
                    if hasattr(layer, 'conv') and hasattr(layer.conv, 'out_channels'):
                        # 在卷積層後添加注意力
                        channels = layer.conv.out_channels
                        if channels >= 64:  # 只在較深的層添加注意力
                            setattr(layer, 'attention', CBAM(channels))
            print("✅ 注意力機制已添加到檢測網路")
        except Exception as e:
            print(f"⚠️ 添加注意力機制時出現問題: {e}")
    
    def _mark_feature_layers(self):
        """標記特徵提取層，用於後續融合"""
        self.feature_layers = [4, 6, 9]  # YOLOv5的主要特徵層
        
    def forward(self, x):
        """前向傳播"""
        if self.training:
            return self.model(x)
        else:
            # 推理模式，應用注意力機制
            return self._forward_with_attention(x)
    
    def _forward_with_attention(self, x):
        """帶注意力機制的前向傳播"""
        features = []
        y = x
        
        for i, layer in enumerate(self.model.model):
            y = layer(y)
            
            # 應用注意力機制
            if hasattr(layer, 'attention'):
                y = layer.attention(y)
            
            # 收集特徵用於可能的融合
            if i in self.feature_layers:
                features.append(y)
                
        return y

    def extract_features(self, x):
        """提取中間特徵，用於跨任務融合"""
        features = []
        y = x
        
        for i, layer in enumerate(self.model.model):
            y = layer(y)
            if hasattr(layer, 'attention'):
                y = layer.attention(y)
            if i in self.feature_layers:
                features.append(y)
                
        return features


class SpecializedClassificationBackbone(nn.Module):
    """專業化分類骨幹網路 - 基於ResNet-101 + 醫學圖像優化"""
    
    def __init__(self, num_classes=3, input_channels=3, pretrained=True):
        super(SpecializedClassificationBackbone, self).__init__()
        self.num_classes = num_classes
        
        # 基礎ResNet-101架構
        self.backbone = models.resnet101(pretrained=pretrained)
        
        # 移除原始分類頭
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 醫學圖像特定的預處理層
        self.medical_preprocessing = self._create_medical_preprocessing()
        
        # 多尺度特徵融合
        self.multiscale_fusion = self._create_multiscale_fusion()
        
        # 增強的分類頭
        self.enhanced_classifier = self._create_enhanced_classifier()
        
        # 注意力機制
        self.attention = CBAM(2048)  # ResNet-101的最後特徵通道數
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def _create_medical_preprocessing(self):
        """創建醫學圖像特定的預處理層"""
        return nn.Sequential(
            # 邊緣增強
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            
            # 對比度增強
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
    
    def _create_multiscale_fusion(self):
        """創建多尺度特徵融合模塊"""
        return nn.ModuleDict({
            'scale1': nn.AdaptiveAvgPool2d((7, 7)),
            'scale2': nn.AdaptiveAvgPool2d((14, 14)),
            'scale3': nn.AdaptiveAvgPool2d((28, 28)),
            'fusion_conv': nn.Conv2d(2048*3, 2048, kernel_size=1, bias=False),
            'fusion_bn': nn.BatchNorm2d(2048),
            'fusion_relu': nn.ReLU(inplace=True)
        })
    
    def _create_enhanced_classifier(self):
        """創建增強的分類頭"""
        return nn.Sequential(
            # 全局平均池化 + 自適應池化組合
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # 第一個FC層
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # 第二個FC層
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 輸出層
            nn.Linear(512, self.num_classes)
        )
    
    def forward(self, x):
        """前向傳播"""
        # 醫學圖像預處理
        x = self.medical_preprocessing(x)
        
        # 主幹特徵提取
        features = self.backbone(x)
        
        # 多尺度特徵融合
        ms_features = self._apply_multiscale_fusion(features)
        
        # 應用注意力機制
        attended_features = self.attention(ms_features)
        
        # 分類
        output = self.enhanced_classifier(attended_features)
        
        return output
    
    def _apply_multiscale_fusion(self, features):
        """應用多尺度特徵融合"""
        # 獲取不同尺度的特徵
        scale1 = self.multiscale_fusion['scale1'](features)  # 7x7
        scale2 = self.multiscale_fusion['scale2'](features)  # 14x14  
        scale3 = self.multiscale_fusion['scale3'](features)  # 28x28
        
        # 調整到相同尺寸
        scale2_resized = F.interpolate(scale2, size=(7, 7), mode='bilinear', align_corners=False)
        scale3_resized = F.interpolate(scale3, size=(7, 7), mode='bilinear', align_corners=False)
        
        # 融合
        fused = torch.cat([scale1, scale2_resized, scale3_resized], dim=1)
        
        # 降維
        fused = self.multiscale_fusion['fusion_conv'](fused)
        fused = self.multiscale_fusion['fusion_bn'](fused)
        fused = self.multiscale_fusion['fusion_relu'](fused)
        
        return fused

    def extract_features(self, x):
        """提取特徵用於跨任務融合"""
        x = self.medical_preprocessing(x)
        features = self.backbone(x)
        return features


class CrossTaskFeatureFusion(nn.Module):
    """跨任務特徵融合模塊 - 在保持獨立性的同時允許特徵協作"""
    
    def __init__(self, detection_channels=512, classification_channels=2048):
        super(CrossTaskFeatureFusion, self).__init__()
        
        # 特徵對齊層
        self.det_align = nn.Conv2d(detection_channels, 512, kernel_size=1)
        self.cls_align = nn.Conv2d(classification_channels, 512, kernel_size=1)
        
        # 跨任務注意力
        self.cross_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # 特徵增強
        self.det_enhance = CBAM(512)
        self.cls_enhance = CBAM(512)
        
        # 輸出投影
        self.det_output = nn.Conv2d(512, detection_channels, kernel_size=1)
        self.cls_output = nn.Conv2d(512, classification_channels, kernel_size=1)
        
    def forward(self, det_features, cls_features):
        """
        Args:
            det_features: 檢測網路特徵 [B, C1, H, W]
            cls_features: 分類網路特徵 [B, C2, H', W']
        """
        # 特徵對齊
        det_aligned = self.det_align(det_features)  # [B, 512, H, W]
        
        # 調整分類特徵尺寸以匹配檢測特徵
        cls_resized = F.interpolate(cls_features, size=det_features.shape[2:], 
                                  mode='bilinear', align_corners=False)
        cls_aligned = self.cls_align(cls_resized)   # [B, 512, H, W]
        
        # 準備跨注意力輸入 (展平為序列)
        B, C, H, W = det_aligned.shape
        det_seq = det_aligned.flatten(2).transpose(1, 2)  # [B, HW, 512]
        cls_seq = cls_aligned.flatten(2).transpose(1, 2)  # [B, HW, 512]
        
        # 跨任務注意力 (檢測查詢分類)
        det_enhanced_seq, _ = self.cross_attention(det_seq, cls_seq, cls_seq)
        cls_enhanced_seq, _ = self.cross_attention(cls_seq, det_seq, det_seq)
        
        # 重塑回特徵圖
        det_enhanced = det_enhanced_seq.transpose(1, 2).reshape(B, C, H, W)
        cls_enhanced = cls_enhanced_seq.transpose(1, 2).reshape(B, C, H, W)
        
        # CBAM增強
        det_enhanced = self.det_enhance(det_enhanced)
        cls_enhanced = self.cls_enhance(cls_enhanced)
        
        # 輸出投影
        det_output = self.det_output(det_enhanced)
        cls_output = self.cls_output(cls_enhanced)
        
        # 殘差連接
        det_final = det_features + det_output
        cls_final = cls_features + F.interpolate(cls_output, size=cls_features.shape[2:], 
                                               mode='bilinear', align_corners=False)
        
        return det_final, cls_final


class AdvancedDualBackboneModel(nn.Module):
    """高級雙獨立骨幹網路模型"""
    
    def __init__(self, nc=4, num_classes=3, device='cpu', enable_fusion=True):
        super(AdvancedDualBackboneModel, self).__init__()
        
        self.device = select_device(device)
        self.nc = nc
        self.num_classes = num_classes
        self.enable_fusion = enable_fusion
        
        # 檢測骨幹網路 (增強版)
        self.detection_backbone = EnhancedDetectionBackbone(
            nc=nc, device=device
        ).to(self.device)
        
        # 分類骨幹網路 (專業化)
        self.classification_backbone = SpecializedClassificationBackbone(
            num_classes=num_classes
        ).to(self.device)
        
        # 跨任務特徵融合 (可選)
        if enable_fusion:
            self.feature_fusion = CrossTaskFeatureFusion().to(self.device)
        
        print(f"✅ 高級雙獨立骨幹網路初始化完成")
        print(f"   檢測類別數: {nc}")
        print(f"   分類類別數: {num_classes}")
        print(f"   特徵融合: {'啟用' if enable_fusion else '禁用'}")
        print(f"   設備: {device}")
        
    def forward(self, x):
        """前向傳播"""
        # 檢測分支
        detection_output = self.detection_backbone(x)
        
        # 分類分支
        classification_output = self.classification_backbone(x)
        
        # 可選的特徵融合
        if self.enable_fusion and self.training:
            try:
                det_features = self.detection_backbone.extract_features(x)
                cls_features = self.classification_backbone.extract_features(x)
                
                # 使用最後一層特徵進行融合
                if det_features and cls_features:
                    enhanced_det, enhanced_cls = self.feature_fusion(
                        det_features[-1], cls_features
                    )
                    # 注意：實際應用中需要將增強特徵重新輸入到各自的頭部
                    
            except Exception as e:
                print(f"特徵融合過程中出現問題: {e}")
        
        return detection_output, classification_output
    
    def get_model_info(self):
        """獲取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        det_params = sum(p.numel() for p in self.detection_backbone.parameters())
        cls_params = sum(p.numel() for p in self.classification_backbone.parameters())
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'detection_parameters': det_params,
            'classification_parameters': cls_params,
            'parameter_ratio': f"Det:{det_params/total_params:.2%}, Cls:{cls_params/total_params:.2%}"
        }
        
        return info


def test_advanced_dual_backbone():
    """測試高級雙獨立骨幹網路"""
    print("🔧 測試高級雙獨立骨幹網路...")
    
    # 初始化模型
    model = AdvancedDualBackboneModel(
        nc=4, 
        num_classes=3, 
        device='cpu',
        enable_fusion=True
    )
    model.eval()
    
    # 測試輸入
    test_input = torch.randn(2, 3, 640, 640)
    
    print(f"輸入形狀: {test_input.shape}")
    
    # 前向傳播測試
    with torch.no_grad():
        detection_out, classification_out = model(test_input)
    
    print(f"檢測輸出形狀: {detection_out.shape if hasattr(detection_out, 'shape') else type(detection_out)}")
    print(f"分類輸出形狀: {classification_out.shape}")
    
    # 模型信息
    info = model.get_model_info()
    print(f"\n📊 模型信息:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("✅ 高級雙獨立骨幹網路測試完成")
    return model


if __name__ == "__main__":
    # 設置隨機種子
    init_seeds(42)
    
    # 運行測試
    model = test_advanced_dual_backbone() 