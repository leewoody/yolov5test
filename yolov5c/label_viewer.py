#!/usr/bin/env python3
"""
Label Viewer Tool for YOLOv5 Dataset
顯示圖像和對應的標籤位置
"""

import os
import sys
import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Dict, Optional

def load_data_yaml(yaml_path: str) -> Dict:
    """載入 data.yaml 文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def parse_segmentation_line(line: str) -> List[Tuple[float, float]]:
    """解析分割線為座標點列表"""
    parts = line.strip().split()
    if len(parts) < 3:
        return []
    
    class_id = int(parts[0])
    coordinates = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            x = float(parts[i])
            y = float(parts[i + 1])
            coordinates.append((x, y))
    
    return coordinates

def parse_classification_line(line: str) -> List[int]:
    """解析分類線為 one-hot 編碼"""
    parts = line.strip().split()
    return [int(x) for x in parts]

def load_label_file(label_path: str) -> Tuple[List[Tuple[float, float]], List[int]]:
    """載入標籤文件，返回分割座標和分類標籤"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 過濾空行
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 2:
            print(f"警告: 文件 {label_path} 有效行數不足")
            return [], []
        
        # 第一行是分割座標
        seg_coordinates = parse_segmentation_line(non_empty_lines[0])
        
        # 最後一行是分類標籤
        cls_labels = parse_classification_line(non_empty_lines[-1])
        
        return seg_coordinates, cls_labels
    
    except Exception as e:
        print(f"錯誤: 讀取文件 {label_path}: {e}")
        return [], []

def segmentation_to_bbox(seg_coordinates: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """將分割座標轉換為邊界框 (x_center, y_center, width, height)"""
    if not seg_coordinates:
        return 0.0, 0.0, 0.0, 0.0
    
    # 轉換為 numpy 數組
    coords = np.array(seg_coordinates)
    
    # 計算邊界框
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    
    # 計算中心點和寬高
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def draw_segmentation(image: np.ndarray, seg_coordinates: List[Tuple[float, float]], 
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """在圖像上繪製分割輪廓"""
    if not seg_coordinates:
        return image
    
    # 轉換座標為像素座標
    h, w = image.shape[:2]
    pixel_coords = []
    for x, y in seg_coordinates:
        px = int(x * w)
        py = int(y * h)
        pixel_coords.append([px, py])
    
    pixel_coords = np.array(pixel_coords, dtype=np.int32)
    
    # 繪製輪廓
    cv2.polylines(image, [pixel_coords], True, color, thickness)
    
    # 繪製頂點
    for px, py in pixel_coords:
        cv2.circle(image, (px, py), 3, color, -1)
    
    return image

def draw_bbox(image: np.ndarray, bbox: Tuple[float, float, float, float], 
              color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    """在圖像上繪製邊界框"""
    x_center, y_center, width, height = bbox
    
    h, w = image.shape[:2]
    
    # 轉換為像素座標
    x_center_px = int(x_center * w)
    y_center_px = int(y_center * h)
    width_px = int(width * w)
    height_px = int(height * h)
    
    # 計算左上角和右下角
    x1 = x_center_px - width_px // 2
    y1 = y_center_px - height_px // 2
    x2 = x_center_px + width_px // 2
    y2 = y_center_px + height_px // 2
    
    # 繪製矩形
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # 繪製中心點
    cv2.circle(image, (x_center_px, y_center_px), 3, color, -1)
    
    return image

def visualize_label(image_path: str, label_path: str, class_names: List[str], 
                   show_segmentation: bool = True, show_bbox: bool = True,
                   save_path: Optional[str] = None) -> None:
    """可視化單個圖像和標籤"""
    
    # 載入圖像
    if not os.path.exists(image_path):
        print(f"錯誤: 圖像文件不存在 {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤: 無法載入圖像 {image_path}")
        return
    
    # 轉換為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 載入標籤
    seg_coordinates, cls_labels = load_label_file(label_path)
    
    if not seg_coordinates and not cls_labels:
        print(f"警告: 無法解析標籤文件 {label_path}")
        return
    
    # 創建圖像副本用於繪製
    image_draw = image_rgb.copy()
    
    # 繪製分割輪廓
    if show_segmentation and seg_coordinates:
        image_draw = draw_segmentation(image_draw, seg_coordinates, color=(0, 255, 0), thickness=2)
    
    # 繪製邊界框
    if show_bbox and seg_coordinates:
        bbox = segmentation_to_bbox(seg_coordinates)
        image_draw = draw_bbox(image_draw, bbox, color=(255, 0, 0), thickness=2)
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始圖像
    ax1.imshow(image_rgb)
    ax1.set_title('原始圖像')
    ax1.axis('off')
    
    # 標註圖像
    ax2.imshow(image_draw)
    ax2.set_title('標註圖像')
    ax2.axis('off')
    
    # 添加標籤信息
    info_text = f"圖像: {os.path.basename(image_path)}\n"
    info_text += f"標籤: {os.path.basename(label_path)}\n"
    
    if seg_coordinates:
        info_text += f"分割點數: {len(seg_coordinates)}\n"
        bbox = segmentation_to_bbox(seg_coordinates)
        info_text += f"邊界框: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})\n"
    
    if cls_labels:
        info_text += f"分類標籤: {cls_labels}\n"
        if len(cls_labels) <= len(class_names):
            active_classes = [class_names[i] for i, label in enumerate(cls_labels) if label == 1]
            info_text += f"類別: {', '.join(active_classes) if active_classes else '無'}\n"
    
    # 在圖表上添加文本信息
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖像已保存到: {save_path}")
    
    plt.show()

def batch_visualize(dataset_dir: str, data_yaml_path: str, num_samples: int = 5, 
                   save_dir: Optional[str] = None) -> None:
    """批量可視化數據集中的樣本"""
    
    # 載入數據配置
    data_config = load_data_yaml(data_yaml_path)
    class_names = data_config.get('names', [])
    
    print(f"數據集: {dataset_dir}")
    print(f"類別: {class_names}")
    
    # 遍歷訓練、驗證、測試集
    for split in ['train', 'valid', 'test']:
        images_dir = Path(dataset_dir) / split / 'images'
        labels_dir = Path(dataset_dir) / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"跳過 {split}: 目錄不存在")
            continue
        
        print(f"\n=== {split.upper()} 集 ===")
        
        # 獲取圖像文件
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        if not image_files:
            print(f"在 {split} 中未找到圖像文件")
            continue
        
        # 隨機選擇樣本
        selected_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        for i, image_file in enumerate(selected_files):
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            if not label_file.exists():
                print(f"跳過 {image_file.name}: 標籤文件不存在")
                continue
            
            print(f"可視化 {i+1}/{len(selected_files)}: {image_file.name}")
            
            # 設置保存路徑
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f"{split}_{image_file.stem}_visualization.png"
            
            # 可視化
            visualize_label(
                str(image_file), 
                str(label_file), 
                class_names,
                show_segmentation=True,
                show_bbox=True,
                save_path=save_path
            )

def interactive_viewer(dataset_dir: str, data_yaml_path: str) -> None:
    """互動式查看器"""
    
    # 載入數據配置
    data_config = load_data_yaml(data_yaml_path)
    class_names = data_config.get('names', [])
    
    print(f"互動式標籤查看器")
    print(f"數據集: {dataset_dir}")
    print(f"類別: {class_names}")
    print("輸入 'quit' 退出")
    print("-" * 50)
    
    while True:
        # 獲取用戶輸入
        user_input = input("請輸入圖像文件名 (例如: a2hiwqVqZ2o=-unnamed_1_1.mp4-0.png): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        # 查找圖像文件
        image_found = False
        for split in ['train', 'valid', 'test']:
            images_dir = Path(dataset_dir) / split / 'images'
            labels_dir = Path(dataset_dir) / split / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            # 嘗試不同的擴展名
            for ext in ['.png', '.jpg', '.jpeg']:
                image_path = images_dir / f"{user_input}{ext}"
                if image_path.exists():
                    label_path = labels_dir / f"{user_input}.txt"
                    
                    if label_path.exists():
                        print(f"找到文件: {image_path}")
                        visualize_label(str(image_path), str(label_path), class_names)
                        image_found = True
                        break
            
            if image_found:
                break
        
        if not image_found:
            print(f"未找到圖像文件: {user_input}")
            print("請檢查文件名是否正確")

def main():
    parser = argparse.ArgumentParser(description='YOLOv5 標籤可視化工具')
    parser.add_argument('--data', type=str, required=True, help='data.yaml 文件路徑')
    parser.add_argument('--image', type=str, help='單個圖像文件路徑')
    parser.add_argument('--label', type=str, help='單個標籤文件路徑')
    parser.add_argument('--batch', type=int, default=5, help='批量可視化樣本數量')
    parser.add_argument('--save-dir', type=str, help='保存可視化結果的目錄')
    parser.add_argument('--interactive', action='store_true', help='啟動互動式查看器')
    parser.add_argument('--dataset-dir', type=str, help='數據集根目錄路徑')
    
    args = parser.parse_args()
    
    # 獲取數據集目錄
    data_config = load_data_yaml(args.data)
    
    # 嘗試從 data.yaml 獲取數據集路徑，如果沒有則使用命令行參數或推斷
    dataset_dir = data_config.get('path', '')
    if not dataset_dir:
        if args.dataset_dir:
            dataset_dir = args.dataset_dir
        else:
            # 從 data.yaml 文件路徑推斷數據集目錄
            yaml_path = Path(args.data)
            dataset_dir = str(yaml_path.parent)
    
    if not dataset_dir:
        print("錯誤: 無法確定數據集路徑，請使用 --dataset-dir 參數指定")
        return
    
    print(f"使用數據集目錄: {dataset_dir}")
    
    if args.interactive:
        interactive_viewer(dataset_dir, args.data)
    elif args.image and args.label:
        # 單個文件可視化
        class_names = data_config.get('names', [])
        visualize_label(args.image, args.label, class_names)
    else:
        # 批量可視化
        batch_visualize(dataset_dir, args.data, args.batch, args.save_dir)

if __name__ == '__main__':
    main() 