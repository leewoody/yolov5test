#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Case Inference
測試案例推理腳本 - 用於單個圖像的推理
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加項目路徑
sys.path.append(str(Path(__file__).parent))

try:
    from train_joint_ultimate import UltimateJointModel
except ImportError:
    print("無法導入UltimateJointModel，使用基礎模型")
    from train_joint_simple import SimpleJointModel as UltimateJointModel


class TestCaseInference:
    def __init__(self, data_yaml, weights_path):
        self.data_yaml = Path(data_yaml)
        self.weights_path = Path(weights_path)
        
        # 載入數據配置
        self.load_data_config()
        
        # 載入模型
        self.load_model()
        
        # 設置設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_data_config(self):
        """載入數據配置"""
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        self.detection_classes = config.get('names', [])
        self.classification_classes = ['PSAX', 'PLAX', 'A4C']
        self.num_detection_classes = len(self.detection_classes)
        
        print(f"✅ 數據配置載入完成:")
        print(f"   - 檢測類別: {self.detection_classes}")
        print(f"   - 分類類別: {self.classification_classes}")
    
    def load_model(self):
        """載入訓練好的模型"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.weights_path}")
        
        # 創建模型實例
        self.model = UltimateJointModel(num_classes=self.num_detection_classes)
        
        # 載入權重
        checkpoint = torch.load(self.weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✅ 模型載入完成: {self.weights_path}")
    
    def preprocess_image(self, image_path):
        """預處理圖像"""
        # 載入圖像
        image = Image.open(image_path).convert('RGB')
        
        # 轉換為tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW
        
        # 調整大小到 224x224
        image_tensor = torch.nn.functional.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 正規化
        image_tensor = image_tensor / 255.0
        
        return image_tensor, image
    
    def inference(self, image_path):
        """執行推理"""
        print(f"\n🔍 開始推理: {image_path}")
        
        # 預處理圖像
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            bbox_preds, cls_preds = self.model(image_tensor)
        
        # 後處理結果
        bbox_pred = bbox_preds[0].cpu().numpy()
        cls_pred = cls_preds[0].cpu().numpy()
        
        # 獲取檢測結果
        detection_class_idx = np.argmax(bbox_pred[:4])  # 前4個是檢測類別
        detection_class = self.detection_classes[detection_class_idx]
        detection_confidence = np.max(bbox_pred[:4])
        
        # 獲取分類結果
        classification_class_idx = np.argmax(cls_pred)
        classification_class = self.classification_classes[classification_class_idx]
        classification_confidence = np.max(cls_pred)
        
        # 獲取bbox座標
        bbox_coords = bbox_pred[4:8]  # 假設後4個是bbox座標
        
        results = {
            'detection': {
                'class': detection_class,
                'confidence': detection_confidence,
                'bbox': bbox_coords
            },
            'classification': {
                'class': classification_class,
                'confidence': classification_confidence
            },
            'original_image': original_image
        }
        
        return results
    
    def visualize_results(self, results, output_path=None):
        """可視化推理結果"""
        image = results['original_image']
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始圖像
        ax1.imshow(image)
        ax1.set_title('原始圖像', fontsize=14)
        ax1.axis('off')
        
        # 推理結果
        ax2.imshow(image)
        
        # 繪製bbox
        bbox = results['detection']['bbox']
        if len(bbox) >= 4:
            # 假設bbox格式為 [x_center, y_center, width, height]
            x_center, y_center, width, height = bbox[:4]
            x1 = (x_center - width/2) * image.width
            y1 = (y_center - height/2) * image.height
            rect = patches.Rectangle((x1, y1), width * image.width, height * image.height, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
        
        # 添加文字標籤
        detection_text = f"檢測: {results['detection']['class']} ({results['detection']['confidence']:.3f})"
        classification_text = f"分類: {results['classification']['class']} ({results['classification']['confidence']:.3f})"
        
        ax2.text(10, 30, detection_text, fontsize=12, color='red', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax2.text(10, 60, classification_text, fontsize=12, color='blue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax2.set_title('推理結果', fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ 可視化結果已保存到: {output_path}")
        
        plt.show()
    
    def print_results(self, results):
        """打印推理結果"""
        print(f"\n📊 推理結果:")
        print(f"   - 檢測類別: {results['detection']['class']}")
        print(f"   - 檢測置信度: {results['detection']['confidence']:.4f}")
        print(f"   - 分類類別: {results['classification']['class']}")
        print(f"   - 分類置信度: {results['classification']['confidence']:.4f}")
        print(f"   - Bbox座標: {results['detection']['bbox']}")
        
        # 性能評估
        detection_conf = results['detection']['confidence']
        classification_conf = results['classification']['confidence']
        
        print(f"\n🎯 性能評估:")
        print(f"   - 檢測置信度 > 0.5: {'✅ 高置信度' if detection_conf > 0.5 else '⚠️ 低置信度'}")
        print(f"   - 分類置信度 > 0.5: {'✅ 高置信度' if classification_conf > 0.5 else '⚠️ 低置信度'}")
        
        if detection_conf > 0.7 and classification_conf > 0.7:
            print(f"   - 整體評估: ✅ 優秀")
        elif detection_conf > 0.5 and classification_conf > 0.5:
            print(f"   - 整體評估: ⚠️ 中等")
        else:
            print(f"   - 整體評估: ❌ 需要改進")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='測試案例推理')
    parser.add_argument('--data', type=str, required=True, help='數據配置文件路徑')
    parser.add_argument('--weights', type=str, required=True, help='模型權重文件路徑')
    parser.add_argument('--image', type=str, required=True, help='測試圖像路徑')
    parser.add_argument('--output', type=str, help='輸出可視化文件路徑')
    
    args = parser.parse_args()
    
    try:
        # 創建推理器
        inference = TestCaseInference(args.data, args.weights)
        
        # 執行推理
        results = inference.inference(args.image)
        
        # 打印結果
        inference.print_results(results)
        
        # 可視化結果
        if args.output:
            inference.visualize_results(results, args.output)
        else:
            inference.visualize_results(results)
        
        print(f"\n🎉 推理完成！")
        
    except Exception as e:
        print(f"❌ 推理失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 