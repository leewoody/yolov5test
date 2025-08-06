#!/usr/bin/env python3
"""
Class Distribution Analysis for Converted Dataset
Analyze the distribution of classes in the converted dataset
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def load_data_yaml(yaml_path: str) -> Dict:
    """Load data.yaml file"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def parse_classification_line(line: str) -> List[int]:
    """Parse classification line to one-hot encoding"""
    parts = line.strip().split()
    return [int(x) for x in parts]

def analyze_class_distribution(data_yaml_path: str) -> Tuple[Dict, List[str]]:
    """Analyze class distribution in the dataset"""
    
    # Load data configuration
    data_config = load_data_yaml(data_yaml_path)
    class_names = data_config.get('names', [])
    dataset_path = data_config.get('path', '')
    
    if not dataset_path:
        yaml_path = Path(data_yaml_path)
        dataset_path = str(yaml_path.parent)
    
    print(f"Dataset path: {dataset_path}")
    print(f"Classes: {class_names}")
    
    # Initialize counters
    class_counts = defaultdict(int)
    split_counts = defaultdict(lambda: defaultdict(int))
    
    # Analyze each split
    for split in ['train', 'valid', 'test']:
        labels_dir = Path(dataset_path) / split / 'labels'
        
        if not labels_dir.exists():
            print(f"Warning: {split} labels directory not found")
            continue
        
        print(f"\nAnalyzing {split} split...")
        
        # Count files
        label_files = list(labels_dir.glob('*.txt'))
        print(f"Found {len(label_files)} label files")
        
        # Analyze each label file
        for label_file in label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Find classification line (last non-empty line)
                classification_line = None
                for line in reversed(lines):
                    if line.strip():
                        classification_line = line.strip()
                        break
                
                if classification_line:
                    cls_labels = parse_classification_line(classification_line)
                    
                    # Count active classes
                    for i, label in enumerate(cls_labels):
                        if label == 1 and i < len(class_names):
                            class_counts[class_names[i]] += 1
                            split_counts[split][class_names[i]] += 1
                
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
    
    return dict(class_counts), dict(split_counts), class_names

def plot_class_distribution(class_counts: Dict, split_counts: Dict, class_names: List[str], save_path: str = None):
    """Plot class distribution"""
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Overall class distribution
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars1 = ax1.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xlabel('Classes')
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Split-wise distribution
    splits = ['train', 'valid', 'test']
    x = np.arange(len(classes))
    width = 0.25
    
    for i, split in enumerate(splits):
        split_data = [split_counts.get(split, {}).get(cls, 0) for cls in classes]
        ax2.bar(x + i*width, split_data, width, label=split.capitalize(), 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i])
    
    ax2.set_title('Class Distribution by Split', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples')
    ax2.set_xlabel('Classes')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(classes)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    
    plt.show()

def print_detailed_analysis(class_counts: Dict, split_counts: Dict, class_names: List[str]):
    """Print detailed analysis"""
    
    print("\n" + "="*60)
    print("詳細類別分布分析")
    print("="*60)
    
    # Overall statistics
    total_samples = sum(class_counts.values())
    print(f"\n📊 總體統計:")
    print(f"   總樣本數: {total_samples}")
    print(f"   類別數: {len(class_names)}")
    
    # Class-wise analysis
    print(f"\n🏷️ 各類別詳細統計:")
    print("-" * 50)
    
    for i, class_name in enumerate(class_names):
        count = class_counts.get(class_name, 0)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        
        print(f"\n{class_name} (類別 {i}):")
        print(f"   總樣本數: {count}")
        print(f"   佔比: {percentage:.2f}%")
        
        # Split-wise breakdown
        for split in ['train', 'valid', 'test']:
            split_count = split_counts.get(split, {}).get(class_name, 0)
            split_percentage = (split_count / count * 100) if count > 0 else 0
            print(f"   {split.capitalize()}: {split_count} ({split_percentage:.1f}%)")
    
    # Imbalance analysis
    print(f"\n⚖️ 類別不平衡分析:")
    print("-" * 30)
    
    max_count = max(class_counts.values()) if class_counts else 0
    min_count = min(class_counts.values()) if class_counts else 0
    
    if max_count > 0:
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"   最大類別樣本數: {max_count}")
        print(f"   最小類別樣本數: {min_count}")
        print(f"   不平衡比例: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print("   ⚠️  嚴重類別不平衡！")
        elif imbalance_ratio > 5:
            print("   ⚠️  中度類別不平衡")
        else:
            print("   ✅ 類別分布相對平衡")
    
    # Missing classes
    missing_classes = [cls for cls in class_names if class_counts.get(cls, 0) == 0]
    if missing_classes:
        print(f"\n❌ 缺失類別:")
        for cls in missing_classes:
            print(f"   - {cls}: 0 個樣本")
    
    # Recommendations
    print(f"\n💡 建議:")
    print("-" * 20)
    
    if missing_classes:
        print("1. 收集缺失類別的樣本數據")
        print("2. 考慮使用數據增強技術")
        print("3. 調整模型架構以處理缺失類別")
    
    if max_count > 0 and min_count > 0:
        if imbalance_ratio > 5:
            print("4. 使用過採樣技術平衡類別")
            print("5. 調整損失函數權重")
            print("6. 考慮使用 Focal Loss")
    
    print("7. 監控每個類別的驗證準確率")
    print("8. 使用混淆矩陣分析分類錯誤")

def main():
    parser = argparse.ArgumentParser(description='Analyze class distribution in converted dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--plot', action='store_true', help='Generate distribution plot')
    parser.add_argument('--output', type=str, default='class_distribution.png', help='Output plot file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found")
        return
    
    # Analyze class distribution
    class_counts, split_counts, class_names = analyze_class_distribution(args.data)
    
    # Print detailed analysis
    print_detailed_analysis(class_counts, split_counts, class_names)
    
    # Generate plot if requested
    if args.plot:
        plot_class_distribution(class_counts, split_counts, class_names, args.output)

if __name__ == '__main__':
    main() 