#!/usr/bin/env python3
"""
Dataset Statistics Analyzer
分析數據集中的檢測類別和分類標籤數量統計
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetStatisticsAnalyzer:
    def __init__(self, data_yaml_path):
        """初始化數據集統計分析器"""
        self.data_yaml_path = data_yaml_path
        self.data_config = self.load_data_config()
        self.detection_classes = self.data_config.get('names', [])
        self.classification_classes = ['PSAX', 'PLAX', 'A4C']  # 根據標籤格式
        
        # 統計數據
        self.detection_stats = defaultdict(lambda: defaultdict(int))
        self.classification_stats = defaultdict(lambda: defaultdict(int))
        self.combined_stats = defaultdict(lambda: defaultdict(int))
        
    def load_data_config(self):
        """載入數據配置文件"""
        try:
            with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 成功載入數據配置: {self.data_yaml_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 載入數據配置失敗: {e}")
            return {}
    
    def parse_label_file(self, label_path):
        """解析標籤文件"""
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None, None
            
            # 第一行：檢測標籤 (YOLO格式)
            detection_line = lines[0].strip()
            detection_labels = []
            
            if detection_line:
                parts = detection_line.split()
                if len(parts) >= 5:  # 至少包含 class_id center_x center_y width height
                    class_id = int(parts[0])
                    detection_labels.append(class_id)
            
            # 第二行：分類標籤 (one-hot編碼)
            classification_line = lines[1].strip()
            classification_label = None
            
            if classification_line:
                parts = classification_line.split()
                if len(parts) == 3:  # PSAX PLAX A4C
                    # 找到值為1的索引
                    for i, val in enumerate(parts):
                        if val == '1':
                            classification_label = i
                            break
            
            return detection_labels, classification_label
            
        except Exception as e:
            logger.warning(f"⚠️ 解析標籤文件失敗 {label_path}: {e}")
            return None, None
    
    def analyze_dataset_split(self, split_name, labels_dir):
        """分析數據集分割"""
        logger.info(f"📊 分析 {split_name} 分割...")
        
        if not os.path.exists(labels_dir):
            logger.warning(f"⚠️ 標籤目錄不存在: {labels_dir}")
            return
        
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        logger.info(f"📁 找到 {len(label_files)} 個標籤文件")
        
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            detection_labels, classification_label = self.parse_label_file(label_path)
            
            if detection_labels is not None:
                # 統計檢測類別
                for class_id in detection_labels:
                    if 0 <= class_id < len(self.detection_classes):
                        class_name = self.detection_classes[class_id]
                        self.detection_stats[split_name][class_name] += 1
                        self.detection_stats['total'][class_name] += 1
            
            if classification_label is not None:
                # 統計分類類別
                if 0 <= classification_label < len(self.classification_classes):
                    class_name = self.classification_classes[classification_label]
                    self.classification_stats[split_name][class_name] += 1
                    self.classification_stats['total'][class_name] += 1
                    
                    # 統計組合情況
                    if detection_labels:
                        for det_class_id in detection_labels:
                            if 0 <= det_class_id < len(self.detection_classes):
                                det_class_name = self.detection_classes[det_class_id]
                                key = f"{det_class_name}_{class_name}"
                                self.combined_stats[split_name][key] += 1
                                self.combined_stats['total'][key] += 1
    
    def analyze_full_dataset(self):
        """分析完整數據集"""
        logger.info("🚀 開始分析完整數據集...")
        
        # 分析各個分割
        for split_name in ['train', 'val', 'test']:
            labels_dir = os.path.join(os.path.dirname(self.data_yaml_path), split_name, 'labels')
            self.analyze_dataset_split(split_name, labels_dir)
        
        logger.info("✅ 數據集分析完成")
    
    def generate_statistics_report(self):
        """生成統計報告"""
        report = []
        report.append("# 數據集統計分析報告")
        report.append("=" * 50)
        report.append("")
        
        # 檢測類別統計
        report.append("## 1. 檢測類別統計")
        report.append("-" * 30)
        for split_name in ['train', 'val', 'test', 'total']:
            report.append(f"\n### {split_name.upper()} 分割")
            total_detections = sum(self.detection_stats[split_name].values())
            for class_name, count in self.detection_stats[split_name].items():
                percentage = (count / total_detections * 100) if total_detections > 0 else 0
                report.append(f"  - {class_name}: {count} ({percentage:.1f}%)")
        
        # 分類類別統計
        report.append("\n\n## 2. 分類類別統計")
        report.append("-" * 30)
        for split_name in ['train', 'val', 'test', 'total']:
            report.append(f"\n### {split_name.upper()} 分割")
            total_classifications = sum(self.classification_stats[split_name].values())
            for class_name, count in self.classification_stats[split_name].items():
                percentage = (count / total_classifications * 100) if total_classifications > 0 else 0
                report.append(f"  - {class_name}: {count} ({percentage:.1f}%)")
        
        # 組合統計
        report.append("\n\n## 3. 檢測+分類組合統計")
        report.append("-" * 30)
        for split_name in ['train', 'val', 'test', 'total']:
            report.append(f"\n### {split_name.upper()} 分割")
            total_combined = sum(self.combined_stats[split_name].values())
            for combo_name, count in self.combined_stats[split_name].items():
                percentage = (count / total_combined * 100) if total_combined > 0 else 0
                report.append(f"  - {combo_name}: {count} ({percentage:.1f}%)")
        
        # 數據集平衡性分析
        report.append("\n\n## 4. 數據集平衡性分析")
        report.append("-" * 30)
        
        # 檢測類別平衡性
        detection_counts = list(self.detection_stats['total'].values())
        if detection_counts:
            detection_std = np.std(detection_counts)
            detection_mean = np.mean(detection_counts)
            detection_cv = detection_std / detection_mean if detection_mean > 0 else 0
            report.append(f"\n### 檢測類別平衡性")
            report.append(f"  - 標準差: {detection_std:.2f}")
            report.append(f"  - 變異係數: {detection_cv:.3f}")
            report.append(f"  - 平衡性評估: {'良好' if detection_cv < 0.5 else '需要改進'}")
        
        # 分類類別平衡性
        classification_counts = list(self.classification_stats['total'].values())
        if classification_counts:
            classification_std = np.std(classification_counts)
            classification_mean = np.mean(classification_counts)
            classification_cv = classification_std / classification_mean if classification_mean > 0 else 0
            report.append(f"\n### 分類類別平衡性")
            report.append(f"  - 標準差: {classification_std:.2f}")
            report.append(f"  - 變異係數: {classification_cv:.3f}")
            report.append(f"  - 平衡性評估: {'良好' if classification_cv < 0.5 else '需要改進'}")
        
        return "\n".join(report)
    
    def create_visualizations(self, output_dir):
        """創建可視化圖表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 檢測類別分佈圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('數據集統計分析', fontsize=16, fontweight='bold')
        
        # 檢測類別 - 總體分佈
        detection_data = self.detection_stats['total']
        if detection_data:
            axes[0, 0].pie(detection_data.values(), labels=detection_data.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('檢測類別總體分佈')
        
        # 分類類別 - 總體分佈
        classification_data = self.classification_stats['total']
        if classification_data:
            axes[0, 1].pie(classification_data.values(), labels=classification_data.keys(), autopct='%1.1f%%')
            axes[0, 1].set_title('分類類別總體分佈')
        
        # 檢測類別 - 各分割分佈
        splits = ['train', 'val', 'test']
        detection_splits = {split: self.detection_stats[split] for split in splits}
        if detection_splits:
            x = np.arange(len(self.detection_classes))
            width = 0.25
            
            for i, (split, data) in enumerate(detection_splits.items()):
                values = [data.get(cls, 0) for cls in self.detection_classes]
                axes[1, 0].bar(x + i*width, values, width, label=split)
            
            axes[1, 0].set_xlabel('檢測類別')
            axes[1, 0].set_ylabel('數量')
            axes[1, 0].set_title('檢測類別各分割分佈')
            axes[1, 0].set_xticks(x + width)
            axes[1, 0].set_xticklabels(self.detection_classes)
            axes[1, 0].legend()
        
        # 分類類別 - 各分割分佈
        classification_splits = {split: self.classification_stats[split] for split in splits}
        if classification_splits:
            x = np.arange(len(self.classification_classes))
            width = 0.25
            
            for i, (split, data) in enumerate(classification_splits.items()):
                values = [data.get(cls, 0) for cls in self.classification_classes]
                axes[1, 1].bar(x + i*width, values, width, label=split)
            
            axes[1, 1].set_xlabel('分類類別')
            axes[1, 1].set_ylabel('數量')
            axes[1, 1].set_title('分類類別各分割分佈')
            axes[1, 1].set_xticks(x + width)
            axes[1, 1].set_xticklabels(self.classification_classes)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 組合統計熱力圖
        if self.combined_stats['total']:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 創建組合矩陣
            combo_matrix = np.zeros((len(self.detection_classes), len(self.classification_classes)))
            for combo_name, count in self.combined_stats['total'].items():
                det_class, cls_class = combo_name.split('_', 1)
                det_idx = self.detection_classes.index(det_class)
                cls_idx = self.classification_classes.index(cls_class)
                combo_matrix[det_idx, cls_idx] = count
            
            sns.heatmap(combo_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                       xticklabels=self.classification_classes,
                       yticklabels=self.detection_classes,
                       ax=ax)
            ax.set_title('檢測+分類組合統計熱力圖')
            ax.set_xlabel('分類類別')
            ax.set_ylabel('檢測類別')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'combined_statistics_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"📊 可視化圖表已保存到: {output_dir}")
    
    def save_statistics(self, output_dir):
        """保存統計結果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存報告
        report = self.generate_statistics_report()
        report_path = os.path.join(output_dir, 'dataset_statistics_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存原始數據
        import json
        stats_data = {
            'detection_stats': dict(self.detection_stats),
            'classification_stats': dict(self.classification_stats),
            'combined_stats': dict(self.combined_stats),
            'detection_classes': self.detection_classes,
            'classification_classes': self.classification_classes
        }
        
        stats_path = os.path.join(output_dir, 'dataset_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 統計報告已保存到: {output_dir}")
        return report_path

def main():
    parser = argparse.ArgumentParser(description='分析數據集統計信息')
    parser.add_argument('--data', type=str, default='converted_dataset_fixed/data.yaml',
                       help='數據配置文件路徑')
    parser.add_argument('--output', type=str, default='dataset_analysis',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 檢查數據配置文件
    if not os.path.exists(args.data):
        logger.error(f"❌ 數據配置文件不存在: {args.data}")
        return
    
    # 創建分析器
    analyzer = DatasetStatisticsAnalyzer(args.data)
    
    # 分析數據集
    analyzer.analyze_full_dataset()
    
    # 生成報告和可視化
    output_dir = args.output
    report_path = analyzer.save_statistics(output_dir)
    analyzer.create_visualizations(output_dir)
    
    # 顯示報告摘要
    print("\n" + "="*60)
    print("📊 數據集統計分析完成")
    print("="*60)
    
    # 顯示檢測類別統計
    print("\n🔍 檢測類別統計 (總計):")
    total_detections = sum(analyzer.detection_stats['total'].values())
    for class_name, count in analyzer.detection_stats['total'].items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # 顯示分類類別統計
    print("\n🏷️ 分類類別統計 (總計):")
    total_classifications = sum(analyzer.classification_stats['total'].values())
    for class_name, count in analyzer.classification_stats['total'].items():
        percentage = (count / total_classifications * 100) if total_classifications > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\n📄 詳細報告: {report_path}")
    print(f"📊 可視化圖表: {output_dir}/")

if __name__ == "__main__":
    main() 