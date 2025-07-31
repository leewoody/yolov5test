#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析您提供的訓練日誌
"""

import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_training_log():
    """分析您提供的訓練日誌"""
    
    # 您提供的訓練日誌
    log_text = """
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
    
    # 解析數據
    batch_data = []
    epoch_data = None
    
    lines = log_text.strip().split('\n')
    for line in lines:
        # 解析batch數據
        batch_match = re.search(r'Epoch (\d+)/(\d+), Batch (\d+)/(\d+), Loss: ([\d.]+) \(Bbox: ([\d.]+), Cls: ([\d.]+)\)', line)
        if batch_match:
            batch_data.append({
                'epoch': int(batch_match.group(1)),
                'batch': int(batch_match.group(3)),
                'total_batches': int(batch_match.group(4)),
                'total_loss': float(batch_match.group(5)),
                'bbox_loss': float(batch_match.group(6)),
                'cls_loss': float(batch_match.group(7))
            })
        
        # 解析epoch總結
        if line.strip().startswith('Epoch 23:'):
            # 解析接下來的幾行
            train_loss_match = re.search(r'Train Loss: ([\d.]+) \(Bbox: ([\d.]+), Cls: ([\d.]+)\)', lines[lines.index(line) + 1])
            val_loss_match = re.search(r'Val Loss: ([\d.]+) \(Bbox: ([\d.]+), Cls: ([\d.]+)\)', lines[lines.index(line) + 2])
            val_acc_match = re.search(r'Val Cls Accuracy: ([\d.]+)', lines[lines.index(line) + 3])
            best_acc_match = re.search(r'Best Val Cls Accuracy: ([\d.]+)', lines[lines.index(line) + 4])
            
            if train_loss_match and val_loss_match and val_acc_match and best_acc_match:
                epoch_data = {
                    'epoch': 23,
                    'train_loss': float(train_loss_match.group(1)),
                    'train_bbox_loss': float(train_loss_match.group(2)),
                    'train_cls_loss': float(train_loss_match.group(3)),
                    'val_loss': float(val_loss_match.group(1)),
                    'val_bbox_loss': float(val_loss_match.group(2)),
                    'val_cls_loss': float(val_loss_match.group(3)),
                    'val_cls_accuracy': float(val_acc_match.group(1)),
                    'best_val_cls_accuracy': float(best_acc_match.group(1))
                }
    
    # 分析結果
    if epoch_data:
        analysis = {
            'current_epoch': epoch_data['epoch'],
            'total_epochs': 40,
            'progress': f"{epoch_data['epoch']}/40 ({epoch_data['epoch']/40*100:.1f}%)",
            
            # 損失分析
            'train_loss': epoch_data['train_loss'],
            'val_loss': epoch_data['val_loss'],
            'loss_gap': epoch_data['train_loss'] - epoch_data['val_loss'],
            
            # 分類準確率
            'current_accuracy': epoch_data['val_cls_accuracy'],
            'best_accuracy': epoch_data['best_val_cls_accuracy'],
            'accuracy_trend': 'declining' if epoch_data['val_cls_accuracy'] < epoch_data['best_val_cls_accuracy'] else 'improving',
            
            # 過擬合分析
            'overfitting_risk': 'high' if epoch_data['train_loss'] > epoch_data['val_loss'] else 'low',
            
            # 建議
            'recommendations': []
        }
        
        # 生成建議
        if epoch_data['train_loss'] > epoch_data['val_loss']:
            analysis['recommendations'].append("⚠️ 檢測到過擬合風險：訓練損失 > 驗證損失")
            analysis['recommendations'].append("建議：減少訓練輪數或增加正則化")
        
        if epoch_data['val_cls_accuracy'] < epoch_data['best_val_cls_accuracy']:
            analysis['recommendations'].append("⚠️ 分類準確率下降：當前 < 最佳")
            analysis['recommendations'].append("建議：檢查學習率或數據質量")
        
        if epoch_data['val_cls_accuracy'] < 0.7:
            analysis['recommendations'].append("⚠️ 分類準確率偏低 (< 70%)")
            analysis['recommendations'].append("建議：增加訓練數據或調整模型架構")
        
        if epoch_data['epoch'] < 20:
            analysis['recommendations'].append("ℹ️ 訓練輪數較少，建議繼續訓練")
        
        # 生成報告
        generate_report(analysis, batch_data, epoch_data)
        
        return analysis
    else:
        print("❌ 無法解析訓練日誌數據")
        return None

def generate_report(analysis, batch_data, epoch_data):
    """生成分析報告"""
    save_dir = Path('runs/analysis')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成圖表
    generate_plots(batch_data, epoch_data, save_dir)
    
    # 生成報告
    report = f"""
# 📊 訓練日誌分析報告

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

## 📊 Batch級別分析

### 損失變化趨勢
"""
    
    if batch_data:
        losses = [b['total_loss'] for b in batch_data]
        bbox_losses = [b['bbox_loss'] for b in batch_data]
        cls_losses = [b['cls_loss'] for b in batch_data]
        
        report += f"""
- **總損失範圍**: {min(losses):.4f} - {max(losses):.4f}
- **邊界框損失範圍**: {min(bbox_losses):.4f} - {max(bbox_losses):.4f}
- **分類損失範圍**: {min(cls_losses):.4f} - {max(cls_losses):.4f}
- **損失穩定性**: {'穩定' if max(losses) - min(losses) < 1.0 else '不穩定'}
"""

    report += f"""
## 🎯 建議

"""
    
    for rec in analysis['recommendations']:
        report += f"{rec}\n"
    
    report += f"""
## 📈 詳細數據

### Epoch 23 詳細指標
- **訓練損失**: {epoch_data['train_loss']:.4f} (Bbox: {epoch_data['train_bbox_loss']:.4f}, Cls: {epoch_data['train_cls_loss']:.4f})
- **驗證損失**: {epoch_data['val_loss']:.4f} (Bbox: {epoch_data['val_bbox_loss']:.4f}, Cls: {epoch_data['val_cls_loss']:.4f})
- **分類準確率**: {epoch_data['val_cls_accuracy']:.4f} (最佳: {epoch_data['best_val_cls_accuracy']:.4f})

### Batch級別數據
"""
    
    for i, batch in enumerate(batch_data):
        report += f"""
**Batch {batch['batch']}**:
- 總損失: {batch['total_loss']:.4f}
- 邊界框損失: {batch['bbox_loss']:.4f}
- 分類損失: {batch['cls_loss']:.4f}
"""
    
    report += f"""
## 🔍 關鍵發現

1. **過擬合警告**: 訓練損失 ({epoch_data['train_loss']:.4f}) > 驗證損失 ({epoch_data['val_loss']:.4f})
2. **準確率下降**: 當前準確率 ({epoch_data['val_cls_accuracy']:.4f}) < 最佳準確率 ({epoch_data['best_val_cls_accuracy']:.4f})
3. **邊界框性能**: 邊界框損失很低 ({epoch_data['train_bbox_loss']:.4f})，表示定位準確
4. **分類挑戰**: 分類損失 ({epoch_data['train_cls_loss']:.4f}) 相對較高，需要改善

## 🚀 立即行動建議

1. **調整學習率**: 考慮降低學習率以穩定訓練
2. **增加正則化**: 添加Dropout或Weight Decay
3. **早停機制**: 監控驗證損失，避免過擬合
4. **數據增強**: 增加訓練數據的多樣性
5. **模型檢查**: 考慮使用更小的模型或調整架構

---
*報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存報告
    with open(save_dir / 'training_log_analysis.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存JSON數據
    with open(save_dir / 'training_log_data.json', 'w') as f:
        json.dump({
            'analysis': analysis,
            'batch_data': batch_data,
            'epoch_data': epoch_data
        }, f, indent=2)
    
    print(f"📄 分析報告已保存到: {save_dir / 'training_log_analysis.md'}")
    print(f"📊 數據文件已保存到: {save_dir / 'training_log_data.json'}")

def generate_plots(batch_data, epoch_data, save_dir):
    """生成分析圖表"""
    if not batch_data or not epoch_data:
        return
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Batch損失變化
    batch_nums = [b['batch'] for b in batch_data]
    total_losses = [b['total_loss'] for b in batch_data]
    bbox_losses = [b['bbox_loss'] for b in batch_data]
    cls_losses = [b['cls_loss'] for b in batch_data]
    
    axes[0, 0].plot(batch_nums, total_losses, 'b-', label='Total Loss', linewidth=2, marker='o')
    axes[0, 0].plot(batch_nums, bbox_losses, 'orange', label='Bbox Loss', linewidth=2, marker='s')
    axes[0, 0].plot(batch_nums, cls_losses, 'purple', label='Cls Loss', linewidth=2, marker='^')
    axes[0, 0].set_title('Batch Loss Progression (Epoch 23)')
    axes[0, 0].set_xlabel('Batch Number')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 損失組件比較
    components = ['Total', 'Bbox', 'Cls']
    train_values = [epoch_data['train_loss'], epoch_data['train_bbox_loss'], epoch_data['train_cls_loss']]
    val_values = [epoch_data['val_loss'], epoch_data['val_bbox_loss'], epoch_data['val_cls_loss']]
    
    x = np.arange(len(components))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    axes[0, 1].bar(x + width/2, val_values, width, label='Validation', alpha=0.8)
    axes[0, 1].set_title('Loss Components Comparison')
    axes[0, 1].set_xlabel('Loss Type')
    axes[0, 1].set_ylabel('Loss Value')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(components)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 準確率分析
    accuracies = [epoch_data['val_cls_accuracy'], epoch_data['best_val_cls_accuracy']]
    labels = ['Current', 'Best']
    colors = ['red' if epoch_data['val_cls_accuracy'] < epoch_data['best_val_cls_accuracy'] else 'green', 'blue']
    
    axes[1, 0].bar(labels, accuracies, color=colors, alpha=0.7)
    axes[1, 0].set_title('Classification Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 過擬合指標
    train_val_gap = epoch_data['train_loss'] - epoch_data['val_loss']
    colors = ['red' if train_val_gap > 0 else 'green']
    
    axes[1, 1].bar(['Train-Val Gap'], [train_val_gap], color=colors, alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Indicator')
    axes[1, 1].set_ylabel('Loss Gap')
    axes[1, 1].text(0, train_val_gap + (0.1 if train_val_gap > 0 else -0.1), 
                   f'{train_val_gap:.3f}', ha='center', va='bottom' if train_val_gap > 0 else 'top')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_log_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 分析圖表已保存到: {save_dir / 'training_log_analysis.png'}")

if __name__ == "__main__":
    print("🔍 開始分析您的訓練日誌...")
    analysis = analyze_training_log()
    
    if analysis:
        print("\n" + "="*60)
        print("📊 分析結果摘要")
        print("="*60)
        print(f"🎯 訓練進度: {analysis['progress']}")
        print(f"📈 訓練損失: {analysis['train_loss']:.4f}")
        print(f"📉 驗證損失: {analysis['val_loss']:.4f}")
        print(f"🎯 當前準確率: {analysis['current_accuracy']:.4f} ({analysis['current_accuracy']*100:.1f}%)")
        print(f"🏆 最佳準確率: {analysis['best_accuracy']:.4f} ({analysis['best_accuracy']*100:.1f}%)")
        print(f"⚠️ 過擬合風險: {analysis['overfitting_risk']}")
        
        print("\n🎯 建議:")
        for rec in analysis['recommendations']:
            print(f"  {rec}")
        
        print("\n✅ 分析完成！詳細報告已保存到 runs/analysis/ 目錄")
    else:
        print("❌ 分析失敗") 