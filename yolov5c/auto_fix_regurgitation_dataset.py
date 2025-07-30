#!/usr/bin/env python3
"""
自動修正 Regurgitation-YOLODataset-1new 數據集
支援 joint detection + classification 訓練
"""

import os
import shutil
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

def fix_label_file(label_file, output_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    detection_lines = []
    classification_label = None
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            detection_lines.append(' '.join(parts))
        elif len(parts) == 1:
            classification_label = parts[0]
    # 若無分類標籤，預設為0
    if classification_label is None:
        classification_label = '0'
    with open(output_file, 'w') as f:
        for det in detection_lines:
            f.write(det + '\n')
        f.write(classification_label + '\n')

def has_bbox(label_file):
    """Return True if label file contains at least one bbox line (5 columns)."""
    if not os.path.exists(label_file):
        return False
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                return True
    return False

def main():
    source_dir = Path("../Regurgitation-YOLODataset-1new")
    output_dir = Path("../regurgitation_joint_20250730")
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        source_images_dir = source_dir / split / 'images'
        source_labels_dir = source_dir / split / 'labels'
        if not source_images_dir.exists():
            continue
        image_files = list(source_images_dir.glob('*.jpg')) + list(source_images_dir.glob('*.png'))
        for img_file in tqdm(image_files, desc=f"fix {split}"):
            label_file = source_labels_dir / f"{img_file.stem}.txt"
            output_label_file = output_dir / split / 'labels' / f"{img_file.stem}.txt"
            if label_file.exists():
                # 只保留有 bbox 的樣本
                if has_bbox(label_file):
                    fix_label_file(label_file, output_label_file)
                    shutil.copy2(img_file, output_dir / split / 'images' / img_file.name)
                else:
                    # 沒有 bbox，直接刪除圖片與標籤
                    if os.path.exists(img_file):
                        os.remove(img_file)
                    if os.path.exists(label_file):
                        os.remove(label_file)
            else:
                # 沒有標籤檔案，直接複製圖片
                shutil.copy2(img_file, output_dir / split / 'images' / img_file.name)
    # 刪除舊cache
    for split in ['train', 'valid', 'test']:
        cache_file = output_dir / split / 'labels.cache'
        if cache_file.exists():
            os.remove(cache_file)
    # 生成data.yaml
    data_yaml = {
        'names': ['AR', 'MR', 'PR', 'TR'],
        'nc': 4,
        'train': str(output_dir / 'train' / 'images'),
        'val': str(output_dir / 'valid' / 'images'),
        'test': str(output_dir / 'test' / 'images')
    }
    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print('Auto fix done!')

if __name__ == "__main__":
    main() 