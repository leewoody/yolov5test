import os
import yaml
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_class_distribution(data_dir):
    """分析數據集的類別分布"""
    data_dir = Path(data_dir)
    
    # 讀取 data.yaml
    with open(data_dir / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    print(f"類別名稱: {class_names}")
    
    # 分析每個分割的類別分布
    splits = ['train', 'val', 'test']
    all_distributions = {}
    
    for split in splits:
        labels_dir = data_dir / split / 'labels'
        if not labels_dir.exists():
            print(f"警告: {split} 標籤目錄不存在")
            continue
        
        class_counts = Counter()
        total_files = 0
        
        # 統計每個標籤文件的類別
        for label_file in labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) >= 2:
                    # 讀取分類標籤（第二行）
                    cls_line = lines[1].strip().split()
                    if cls_line:
                        # 找到 one-hot encoding 中的 1 的位置
                        for i, val in enumerate(cls_line):
                            if float(val) == 1.0:
                                class_counts[i] += 1
                                break
                        total_files += 1
            except Exception as e:
                print(f"警告: 無法讀取 {label_file}: {e}")
        
        all_distributions[split] = class_counts
        print(f"\n{split.upper()} 分割:")
        print(f"總樣本數: {total_files}")
        print("類別分布:")
        for i, class_name in enumerate(class_names):
            count = class_counts[i]
            percentage = (count / total_files * 100) if total_files > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    return all_distributions, class_names

def plot_class_distribution(all_distributions, class_names):
    """繪製類別分布圖"""
    splits = list(all_distributions.keys())
    num_classes = len(class_names)
    
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 5))
    if len(splits) == 1:
        axes = [axes]
    
    for i, split in enumerate(splits):
        counts = [all_distributions[split][j] for j in range(num_classes)]
        axes[i].bar(class_names, counts)
        axes[i].set_title(f'{split.upper()} 分割')
        axes[i].set_ylabel('樣本數量')
        axes[i].tick_params(axis='x', rotation=45)
        
        # 添加數值標籤
        for j, count in enumerate(counts):
            axes[i].text(j, count + max(counts) * 0.01, str(count), 
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("類別分布圖已保存為: class_distribution.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析數據集類別分布')
    parser.add_argument('--data', type=str, required=True, help='數據集目錄路徑')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("數據集類別分布分析")
    print("=" * 60)
    
    all_distributions, class_names = analyze_class_distribution(args.data)
    
    # 計算整體分布
    print("\n" + "=" * 60)
    print("整體統計")
    print("=" * 60)
    
    total_counts = Counter()
    for split_dist in all_distributions.values():
        total_counts.update(split_dist)
    
    total_samples = sum(total_counts.values())
    print(f"總樣本數: {total_samples}")
    print("整體類別分布:")
    
    for i, class_name in enumerate(class_names):
        count = total_counts[i]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # 檢查類別不平衡
    print("\n類別不平衡分析:")
    max_count = max(total_counts.values())
    min_count = min(total_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"最多樣本類別: {max_count}")
    print(f"最少樣本類別: {min_count}")
    print(f"不平衡比例: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2:
        print("⚠️  檢測到明顯的類別不平衡，建議進行過採樣或使用 Focal Loss")
    else:
        print("✅ 類別分布相對平衡")
    
    # 繪製分布圖
    try:
        plot_class_distribution(all_distributions, class_names)
    except Exception as e:
        print(f"警告: 無法繪製分布圖: {e}")

if __name__ == '__main__':
    main() 