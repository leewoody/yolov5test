#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Training Analysis Script
分析高級訓練歷史，包括 auto-fix 效果評估
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def analyze_advanced_training(history_file='training_history_advanced.json'):
    """分析高級訓練歷史"""
    
    # 讀取訓練歷史
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到訓練歷史文件: {history_file}")
        return None
    
    # 提取數據
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    val_accuracies = history['val_accuracies']
    overfitting_flags = history['overfitting_flags']
    accuracy_drops = history['accuracy_drops']
    
    epochs = range(1, len(train_losses) + 1)
    
    # 分析結果
    analysis = {
        'total_epochs': len(train_losses),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_val_accuracy': val_accuracies[-1],
        'best_val_accuracy': max(val_accuracies),
        'best_val_loss': min(val_losses),
        'overfitting_count': sum(overfitting_flags),
        'accuracy_drop_count': sum(accuracy_drops),
        'overfitting_rate': sum(overfitting_flags) / len(overfitting_flags) * 100,
        'accuracy_drop_rate': sum(accuracy_drops) / len(accuracy_drops) * 100
    }
    
    # 生成圖表
    create_analysis_plots(epochs, train_losses, val_losses, val_accuracies, 
                         overfitting_flags, accuracy_drops, analysis)
    
    # 生成報告
    generate_analysis_report(analysis, history)
    
    return analysis


def create_analysis_plots(epochs, train_losses, val_losses, val_accuracies, 
                         overfitting_flags, accuracy_drops, analysis):
    """創建分析圖表"""
    
    # 創建子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 損失曲線
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    # 標記過擬合點
    overfitting_epochs = [i+1 for i, flag in enumerate(overfitting_flags) if flag]
    if overfitting_epochs:
        overfitting_losses = [val_losses[i] for i, flag in enumerate(overfitting_flags) if flag]
        axes[0, 0].scatter(overfitting_epochs, overfitting_losses, 
                          color='red', s=100, marker='x', label='Overfitting Detected')
    
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 準確率曲線
    axes[0, 1].plot(epochs, val_accuracies, 'g-', label='Val Accuracy', linewidth=2)
    
    # 標記準確率下降點
    accuracy_drop_epochs = [i+1 for i, flag in enumerate(accuracy_drops) if flag]
    if accuracy_drop_epochs:
        accuracy_drop_values = [val_accuracies[i] for i, flag in enumerate(accuracy_drops) if flag]
        axes[0, 1].scatter(accuracy_drop_epochs, accuracy_drop_values, 
                          color='orange', s=100, marker='s', label='Accuracy Drop Detected')
    
    axes[0, 1].axhline(y=analysis['best_val_accuracy'], color='g', linestyle='--', 
                       label=f'Best Accuracy: {analysis["best_val_accuracy"]:.4f}')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 過擬合檢測統計
    overfitting_percentages = []
    accuracy_drop_percentages = []
    
    for i in range(len(epochs)):
        overfitting_percentages.append(sum(overfitting_flags[:i+1]) / (i+1) * 100)
        accuracy_drop_percentages.append(sum(accuracy_drops[:i+1]) / (i+1) * 100)
    
    axes[1, 0].plot(epochs, overfitting_percentages, 'red', label='Overfitting Rate', linewidth=2)
    axes[1, 0].plot(epochs, accuracy_drop_percentages, 'orange', label='Accuracy Drop Rate', linewidth=2)
    axes[1, 0].set_title('Auto-Fix Detection Rates')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Detection Rate (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 損失差距分析
    loss_gaps = [train_losses[i] - val_losses[i] for i in range(len(train_losses))]
    axes[1, 1].plot(epochs, loss_gaps, 'purple', label='Train-Val Loss Gap', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].fill_between(epochs, loss_gaps, 0, where=(np.array(loss_gaps) > 0), 
                           alpha=0.3, color='red', label='Overfitting Region')
    axes[1, 1].set_title('Overfitting Indicator (Train-Val Loss Gap)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 分析圖表已保存到: advanced_training_analysis.png")


def generate_analysis_report(analysis, history):
    """生成分析報告"""
    
    report = f"""
# 🔧 高級訓練分析報告

## 📊 訓練概況

### 基本統計
- **總訓練輪數**: {analysis['total_epochs']}
- **最終訓練損失**: {analysis['final_train_loss']:.4f}
- **最終驗證損失**: {analysis['final_val_loss']:.4f}
- **最終驗證準確率**: {analysis['final_val_accuracy']:.4f} ({analysis['final_val_accuracy']*100:.1f}%)
- **最佳驗證準確率**: {analysis['best_val_accuracy']:.4f} ({analysis['best_val_accuracy']*100:.1f}%)
- **最佳驗證損失**: {analysis['best_val_loss']:.4f}

### Auto-Fix 效果分析
- **過擬合檢測次數**: {analysis['overfitting_count']} ({analysis['overfitting_rate']:.1f}%)
- **準確率下降檢測次數**: {analysis['accuracy_drop_count']} ({analysis['accuracy_drop_rate']:.1f}%)

## 🎯 性能評估

### 訓練穩定性
"""
    
    if analysis['overfitting_rate'] < 20:
        report += "- ✅ **優秀**: 過擬合檢測率低，訓練穩定\n"
    elif analysis['overfitting_rate'] < 40:
        report += "- ⚠️ **良好**: 偶爾出現過擬合，Auto-Fix 有效處理\n"
    else:
        report += "- ❌ **需要改進**: 過擬合頻繁，建議調整模型架構\n"
    
    if analysis['accuracy_drop_rate'] < 20:
        report += "- ✅ **優秀**: 準確率穩定，分類性能良好\n"
    elif analysis['accuracy_drop_rate'] < 40:
        report += "- ⚠️ **良好**: 偶爾準確率下降，Auto-Fix 有效調整\n"
    else:
        report += "- ❌ **需要改進**: 準確率不穩定，建議檢查數據質量\n"
    
    # 計算改進幅度
    if len(history['val_accuracies']) > 1:
        initial_accuracy = history['val_accuracies'][0]
        final_accuracy = history['val_accuracies'][-1]
        improvement = final_accuracy - initial_accuracy
        
        report += f"""
### 訓練改進
- **初始準確率**: {initial_accuracy:.4f} ({initial_accuracy*100:.1f}%)
- **最終準確率**: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)
- **改進幅度**: {improvement:.4f} ({improvement*100:.1f}%)
"""
        
        if improvement > 0.1:
            report += "- ✅ **顯著改進**: 訓練效果優秀\n"
        elif improvement > 0.05:
            report += "- ⚠️ **適度改進**: 訓練效果良好\n"
        else:
            report += "- ❌ **改進有限**: 需要調整訓練策略\n"
    
    report += f"""
## 🔍 詳細分析

### 過擬合分析
- **檢測頻率**: {analysis['overfitting_rate']:.1f}%
- **處理效果**: {'優秀' if analysis['overfitting_rate'] < 20 else '良好' if analysis['overfitting_rate'] < 40 else '需要改進'}

### 準確率穩定性
- **下降頻率**: {analysis['accuracy_drop_rate']:.1f}%
- **調整效果**: {'優秀' if analysis['accuracy_drop_rate'] < 20 else '良好' if analysis['accuracy_drop_rate'] < 40 else '需要改進'}

## 🚀 建議

"""
    
    # 根據分析結果給出建議
    if analysis['overfitting_rate'] > 30:
        report += "1. **降低模型複雜度**: 考慮減少網絡層數或參數\n"
        report += "2. **增強正則化**: 增加 Dropout 或 Weight Decay\n"
        report += "3. **減少訓練輪數**: 避免過度訓練\n"
    
    if analysis['accuracy_drop_rate'] > 30:
        report += "4. **檢查數據質量**: 確保標註準確性\n"
        report += "5. **調整學習率**: 使用更小的學習率\n"
        report += "6. **數據增強**: 增加訓練數據多樣性\n"
    
    if analysis['final_val_accuracy'] < 0.8:
        report += "7. **模型架構優化**: 考慮使用更強大的backbone\n"
        report += "8. **超參數調優**: 使用網格搜索或貝葉斯優化\n"
    
    report += f"""
## 📈 結論

Auto-Fix 系統在本次訓練中:
- 檢測並處理了 {analysis['overfitting_count']} 次過擬合問題
- 檢測並調整了 {analysis['accuracy_drop_count']} 次準確率下降
- 最終達到 {analysis['final_val_accuracy']:.4f} 的驗證準確率

整體訓練效果: {'優秀' if analysis['final_val_accuracy'] > 0.9 else '良好' if analysis['final_val_accuracy'] > 0.8 else '需要改進'}

---
*分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存報告
    with open('advanced_training_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 分析報告已保存到: advanced_training_analysis_report.md")


def main():
    """主函數"""
    print("🔍 開始分析高級訓練歷史...")
    
    # 分析訓練歷史
    analysis = analyze_advanced_training()
    
    if analysis:
        print("\n" + "="*60)
        print("📊 分析結果摘要")
        print("="*60)
        print(f"總訓練輪數: {analysis['total_epochs']}")
        print(f"最終驗證準確率: {analysis['final_val_accuracy']:.4f} ({analysis['final_val_accuracy']*100:.1f}%)")
        print(f"最佳驗證準確率: {analysis['best_val_accuracy']:.4f} ({analysis['best_val_accuracy']*100:.1f}%)")
        print(f"過擬合檢測率: {analysis['overfitting_rate']:.1f}%")
        print(f"準確率下降檢測率: {analysis['accuracy_drop_rate']:.1f}%")
        
        print("\n✅ 分析完成！詳細報告已保存到:")
        print("  - advanced_training_analysis.png")
        print("  - advanced_training_analysis_report.md")
    else:
        print("❌ 分析失敗")


if __name__ == "__main__":
    main() 