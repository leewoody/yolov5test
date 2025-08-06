#!/usr/bin/env python3
"""
Quick Dataset Summary
快速顯示數據集摘要統計
"""

import json
import os

def load_statistics(stats_file):
    """載入統計數據"""
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_summary(stats_data):
    """打印摘要統計"""
    print("=" * 80)
    print("📊 數據集統計摘要")
    print("=" * 80)
    
    # 檢測類別統計
    print("\n🔍 檢測類別統計 (總計):")
    print("-" * 40)
    detection_stats = stats_data['detection_stats']['total']
    total_detections = sum(detection_stats.values())
    for class_name, count in detection_stats.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name:>3}: {count:>4} ({percentage:>5.1f}%)")
    
    # 分類類別統計
    print("\n🏷️ 分類類別統計 (總計):")
    print("-" * 40)
    classification_stats = stats_data['classification_stats']['total']
    total_classifications = sum(classification_stats.values())
    for class_name, count in classification_stats.items():
        percentage = (count / total_classifications * 100) if total_classifications > 0 else 0
        print(f"  {class_name:>4}: {count:>4} ({percentage:>5.1f}%)")
    
    # 數據集分割統計
    print("\n📁 數據集分割統計:")
    print("-" * 40)
    splits = ['train', 'val', 'test']
    for split in splits:
        train_count = len(stats_data['detection_stats'][split])
        val_count = len(stats_data['classification_stats'][split])
        print(f"  {split.upper():>5}: {train_count:>4} 樣本")
    
    # 關鍵發現
    print("\n🔍 關鍵發現:")
    print("-" * 40)
    
    # 檢測類別不平衡
    detection_counts = list(detection_stats.values())
    max_det = max(detection_counts)
    min_det = min(detection_counts)
    det_imbalance = max_det / min_det if min_det > 0 else 0
    print(f"  • 檢測類別不平衡比例: {det_imbalance:.1f}:1")
    
    # 分類類別不平衡
    classification_counts = list(classification_stats.values())
    max_cls = max(classification_counts)
    min_cls = min(classification_counts)
    cls_imbalance = max_cls / min_cls if min_cls > 0 else 0
    print(f"  • 分類類別不平衡比例: {cls_imbalance:.1f}:1")
    
    # 最常見組合
    combined_stats = stats_data['combined_stats']['total']
    if combined_stats:
        most_common = max(combined_stats.items(), key=lambda x: x[1])
        print(f"  • 最常見組合: {most_common[0]} ({most_common[1]} 樣本)")
    
    # 最少見組合
    if combined_stats:
        least_common = min(combined_stats.items(), key=lambda x: x[1])
        print(f"  • 最少見組合: {least_common[0]} ({least_common[1]} 樣本)")
    
    print("\n" + "=" * 80)

def main():
    stats_file = "dataset_statistics/dataset_statistics.json"
    
    if not os.path.exists(stats_file):
        print("❌ 統計文件不存在，請先運行 analyze_dataset_statistics.py")
        return
    
    stats_data = load_statistics(stats_file)
    print_summary(stats_data)

if __name__ == "__main__":
    main() 