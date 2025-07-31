#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Validation Analysis Script
快速分析訓練日誌並提供驗證指標建議
"""

import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime


class TrainingLogAnalyzer:
    def __init__(self):
        self.epoch_data = []
        self.batch_data = []
        
    def parse_training_log(self, log_text):
        """解析訓練日誌文本"""
        lines = log_text.strip().split('\n')
        
        for line in lines:
            # 解析Epoch行
            epoch_match = re.search(r'Epoch (\d+)/(\d+), Batch (\d+)/(\d+), Loss: ([\d.]+) \(Bbox: ([\d.]+), Cls: ([\d.]+)\)', line)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                batch_num = int(epoch_match.group(3))
                total_batches = int(epoch_match.group(4))
                total_loss = float(epoch_match.group(5))
                bbox_loss = float(epoch_match.group(6))
                cls_loss = float(epoch_match.group(7))
                
                self.batch_data.append({
                    'epoch': epoch_num,
                    'batch': batch_num,
                    'total_batches': total_batches,
                    'total_loss': total_loss,
                    'bbox_loss': bbox_loss,
                    'cls_loss': cls_loss
                })
            
            # 解析Epoch總結行
            epoch_summary_match = re.search(r'Epoch (\d+):\s+Train Loss: ([\d.]+) \(Bbox: ([\d.]+), Cls: ([\d.]+)\)\s+Val Loss: ([\d.]+) \(Bbox: ([\d.]+), Cls: ([\d.]+)\)\s+Val Cls Accuracy: ([\d.]+)\s+Best Val Cls Accuracy: ([\d.]+)', line)
            if not epoch_summary_match:
                # 嘗試更寬鬆的匹配模式
                epoch_summary_match = re.search(r'Epoch (\d+):\s+Train Loss: ([\d.]+).*Val Loss: ([\d.]+).*Val Cls Accuracy: ([\d.]+).*Best Val Cls Accuracy: ([\d.]+)', line)
            if epoch_summary_match:
                epoch_num = int(epoch_summary_match.group(1))
                train_loss = float(epoch_summary_match.group(2))
                
                # 檢查是否有詳細的bbox和cls損失信息
                if len(epoch_summary_match.groups()) >= 9:
                    train_bbox_loss = float(epoch_summary_match.group(3))
                    train_cls_loss = float(epoch_summary_match.group(4))
                    val_loss = float(epoch_summary_match.group(5))
                    val_bbox_loss = float(epoch_summary_match.group(6))
                    val_cls_loss = float(epoch_summary_match.group(7))
                    val_cls_accuracy = float(epoch_summary_match.group(8))
                    best_val_cls_accuracy = float(epoch_summary_match.group(9))
                else:
                    # 簡化模式，只有主要損失和準確率
                    val_loss = float(epoch_summary_match.group(3))
                    val_cls_accuracy = float(epoch_summary_match.group(4))
                    best_val_cls_accuracy = float(epoch_summary_match.group(5))
                    # 設置默認值
                    train_bbox_loss = train_cls_loss = val_bbox_loss = val_cls_loss = 0.0
                
                self.epoch_data.append({
                    'epoch': epoch_num,
                    'train_loss': train_loss,
                    'train_bbox_loss': train_bbox_loss,
                    'train_cls_loss': train_cls_loss,
                    'val_loss': val_loss,
                    'val_bbox_loss': val_bbox_loss,
                    'val_cls_loss': val_cls_loss,
                    'val_cls_accuracy': val_cls_accuracy,
                    'best_val_cls_accuracy': best_val_cls_accuracy
                })
    
    def analyze_training_status(self):
        """分析訓練狀態"""
        if not self.epoch_data:
            return "無法解析訓練日誌數據"
        
        latest_epoch = self.epoch_data[-1]
        
        analysis = {
            'current_epoch': latest_epoch['epoch'],
            'total_epochs': 40,  # 從日誌中推斷
            'progress': f"{latest_epoch['epoch']}/40 ({latest_epoch['epoch']/40*100:.1f}%)",
            
            # 損失分析
            'train_loss': latest_epoch['train_loss'],
            'val_loss': latest_epoch['val_loss'],
            'loss_gap': latest_epoch['train_loss'] - latest_epoch['val_loss'],
            
            # 分類準確率
            'current_accuracy': latest_epoch['val_cls_accuracy'],
            'best_accuracy': latest_epoch['best_val_cls_accuracy'],
            'accuracy_trend': 'improving' if latest_epoch['val_cls_accuracy'] >= latest_epoch['best_val_cls_accuracy'] else 'declining',
            
            # 過擬合分析
            'overfitting_risk': 'high' if latest_epoch['train_loss'] > latest_epoch['val_loss'] else 'low',
            
            # 建議
            'recommendations': []
        }
        
        # 生成建議
        if latest_epoch['train_loss'] > latest_epoch['val_loss']:
            analysis['recommendations'].append("⚠️ 檢測到過擬合風險：訓練損失 > 驗證損失")
            analysis['recommendations'].append("建議：減少訓練輪數或增加正則化")
        
        if latest_epoch['val_cls_accuracy'] < latest_epoch['best_val_cls_accuracy']:
            analysis['recommendations'].append("⚠️ 分類準確率下降：當前 < 最佳")
            analysis['recommendations'].append("建議：檢查學習率或數據質量")
        
        if latest_epoch['val_cls_accuracy'] < 0.7:
            analysis['recommendations'].append("⚠️ 分類準確率偏低 (< 70%)")
            analysis['recommendations'].append("建議：增加訓練數據或調整模型架構")
        
        if latest_epoch['epoch'] < 20:
            analysis['recommendations'].append("ℹ️ 訓練輪數較少，建議繼續訓練")
        
        return analysis
    
    def generate_plots(self, save_dir='runs/analysis'):
        """生成分析圖表"""
        if not self.epoch_data:
            return
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = [d['epoch'] for d in self.epoch_data]
        
        # 創建子圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 損失曲線
        train_losses = [d['train_loss'] for d in self.epoch_data]
        val_losses = [d['val_loss'] for d in self.epoch_data]
        
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 分類準確率
        accuracies = [d['val_cls_accuracy'] for d in self.epoch_data]
        best_accuracies = [d['best_val_cls_accuracy'] for d in self.epoch_data]
        
        axes[0, 1].plot(epochs, accuracies, 'g-', label='Current Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, best_accuracies, 'g--', label='Best Accuracy', linewidth=2)
        axes[0, 1].set_title('Classification Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 邊界框和分類損失
        train_bbox_losses = [d['train_bbox_loss'] for d in self.epoch_data]
        train_cls_losses = [d['train_cls_loss'] for d in self.epoch_data]
        
        axes[1, 0].plot(epochs, train_bbox_losses, 'orange', label='Bbox Loss', linewidth=2)
        axes[1, 0].plot(epochs, train_cls_losses, 'purple', label='Cls Loss', linewidth=2)
        axes[1, 0].set_title('Training Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 損失差距（過擬合指標）
        loss_gaps = [d['train_loss'] - d['val_loss'] for d in self.epoch_data]
        
        axes[1, 1].plot(epochs, loss_gaps, 'red', label='Train-Val Loss Gap', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Indicator (Train-Val Loss Gap)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Gap')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 分析圖表已保存到: {save_dir / 'training_analysis.png'}")
    
    def generate_report(self, save_dir='runs/analysis'):
        """生成分析報告"""
        analysis = self.analyze_training_status()
        
        if isinstance(analysis, str):
            print(f"❌ {analysis}")
            return
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        report = f"""
# 📊 訓練狀態分析報告

## 🎯 當前狀態
- **訓練進度**: {analysis['progress']}
- **當前Epoch**: {analysis['current_epoch']}/40

## 📈 性能指標

### 損失函數
- **訓練損失**: {analysis['train_loss']:.4f}
- **驗證損失**: {analysis['val_loss']:.4f}
- **損失差距**: {analysis['loss_gap']:.4f} ({'過擬合風險' if analysis['loss_gap'] > 0 else '正常'})

### 分類性能
- **當前準確率**: {analysis['current_accuracy']:.4f} ({analysis['current_accuracy']*100:.1f}%)
- **最佳準確率**: {analysis['best_accuracy']:.4f} ({analysis['best_accuracy']*100:.1f}%)
- **準確率趨勢**: {analysis['accuracy_trend']}

### 風險評估
- **過擬合風險**: {analysis['overfitting_risk']}

## 🎯 建議

"""
        
        for rec in analysis['recommendations']:
            report += f"{rec}\n"
        
        report += f"""
## 📊 詳細數據

### Epoch歷史記錄
"""
        
        # 添加詳細的epoch數據
        for epoch_data in self.epoch_data:
            report += f"""
**Epoch {epoch_data['epoch']}**:
- 訓練損失: {epoch_data['train_loss']:.4f} (Bbox: {epoch_data['train_bbox_loss']:.4f}, Cls: {epoch_data['train_cls_loss']:.4f})
- 驗證損失: {epoch_data['val_loss']:.4f} (Bbox: {epoch_data['val_bbox_loss']:.4f}, Cls: {epoch_data['val_cls_loss']:.4f})
- 分類準確率: {epoch_data['val_cls_accuracy']:.4f} (最佳: {epoch_data['best_val_cls_accuracy']:.4f})
"""
        
        report += f"""
---
*報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存報告
        with open(save_dir / 'training_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存JSON數據
        with open(save_dir / 'training_data.json', 'w') as f:
            json.dump({
                'epoch_data': self.epoch_data,
                'batch_data': self.batch_data,
                'analysis': analysis
            }, f, indent=2)
        
        print(f"📄 分析報告已保存到: {save_dir / 'training_analysis_report.md'}")
        print(f"📊 數據文件已保存到: {save_dir / 'training_data.json'}")


def analyze_log_from_text(log_text):
    """從文本分析訓練日誌"""
    analyzer = TrainingLogAnalyzer()
    analyzer.parse_training_log(log_text)
    
    # 生成分析
    analysis = analyzer.analyze_training_status()
    
    # 生成圖表和報告
    analyzer.generate_plots()
    analyzer.generate_report()
    
    return analysis


if __name__ == "__main__":
    # 示例使用
    sample_log = """
Epoch 23/40, Batch 0/83, Loss: 3.1771 (Bbox: 0.0026, Cls: 0.3968)
Epoch 23/40, Batch 20/83, Loss: 1.2542 (Bbox: 0.0036, Cls: 0.1563)
Epoch 23/40, Batch 40/83, Loss: 2.5399 (Bbox: 0.0033, Cls: 0.3171)
Epoch 23/40, Batch 60/83, Loss: 1.6488 (Bbox: 0.0037, Cls: 0.2056)
Epoch 23/40, Batch 80/83, Loss: 1.0560 (Bbox: 0.0029, Cls: 0.1316)
Epoch 23:
  Train Loss: 2.0978 (Bbox: 0.0029, Cls: 0.2619)
  Val Loss: 1.5627 (Bbox: 0.0030, Cls: 0.1950)
  Val Cls Accuracy: 0.6490
  Best Val Cls Accuracy: 0.8125
"""
    
    analysis = analyze_log_from_text(sample_log)
    print("✅ 分析完成！") 