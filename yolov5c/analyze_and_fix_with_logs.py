#!/usr/bin/env python3
"""
🔧 訓練分析與修復工具 (含詳細日誌)
分析訓練歷史並提供修復建議
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime
import argparse
from pathlib import Path

# 設置日誌
def setup_logging(log_file="training_analysis.log"):
    """設置詳細的日誌記錄"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def analyze_training_history(history_file, logger):
    """分析訓練歷史數據"""
    logger.info("🔍 開始分析訓練歷史數據...")
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        logger.info(f"✅ 成功載入訓練歷史文件: {history_file}")
    except Exception as e:
        logger.error(f"❌ 載入訓練歷史文件失敗: {e}")
        return None
    
    # 提取數據
    train_losses = history.get('train_losses', [])
    val_losses = history.get('val_losses', [])
    val_accuracies = history.get('val_accuracies', [])
    overfitting_flags = history.get('overfitting_flags', [])
    accuracy_drops = history.get('accuracy_drops', [])
    
    logger.info(f"📊 數據統計:")
    logger.info(f"  - 訓練輪數: {len(train_losses)}")
    logger.info(f"  - 訓練損失範圍: {min(train_losses):.4f} - {max(train_losses):.4f}")
    logger.info(f"  - 驗證損失範圍: {min(val_losses):.4f} - {max(val_losses):.4f}")
    logger.info(f"  - 驗證準確率範圍: {min(val_accuracies):.4f} - {max(val_accuracies):.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'overfitting_flags': overfitting_flags,
        'accuracy_drops': accuracy_drops
    }

def detect_issues(data, logger):
    """檢測訓練問題"""
    logger.info("🔍 開始檢測訓練問題...")
    
    issues = []
    
    # 1. 過擬合檢測
    overfitting_count = sum(data['overfitting_flags'])
    if overfitting_count > 0:
        logger.warning(f"⚠️ 檢測到 {overfitting_count} 次過擬合問題")
        issues.append({
            'type': 'overfitting',
            'count': overfitting_count,
            'severity': 'high' if overfitting_count > 5 else 'medium'
        })
    else:
        logger.info("✅ 未檢測到過擬合問題")
    
    # 2. 準確率下降檢測
    accuracy_drop_count = sum(data['accuracy_drops'])
    if accuracy_drop_count > 0:
        logger.warning(f"⚠️ 檢測到 {accuracy_drop_count} 次準確率下降問題")
        issues.append({
            'type': 'accuracy_drop',
            'count': accuracy_drop_count,
            'severity': 'high' if accuracy_drop_count > 20 else 'medium'
        })
    else:
        logger.info("✅ 未檢測到準確率下降問題")
    
    # 3. 訓練不穩定檢測
    train_loss_std = np.std(data['train_losses'])
    val_loss_std = np.std(data['val_losses'])
    
    if train_loss_std > 0.2:
        logger.warning(f"⚠️ 訓練損失不穩定 (標準差: {train_loss_std:.4f})")
        issues.append({
            'type': 'unstable_training',
            'std': train_loss_std,
            'severity': 'medium'
        })
    
    if val_loss_std > 0.3:
        logger.warning(f"⚠️ 驗證損失不穩定 (標準差: {val_loss_std:.4f})")
        issues.append({
            'type': 'unstable_validation',
            'std': val_loss_std,
            'severity': 'high'
        })
    
    # 4. 最終性能分析
    final_accuracy = data['val_accuracies'][-1]
    best_accuracy = max(data['val_accuracies'])
    
    logger.info(f"📈 性能分析:")
    logger.info(f"  - 最終準確率: {final_accuracy:.4f}")
    logger.info(f"  - 最佳準確率: {best_accuracy:.4f}")
    logger.info(f"  - 性能差距: {best_accuracy - final_accuracy:.4f}")
    
    if final_accuracy < 0.7:
        logger.warning("⚠️ 最終準確率偏低 (< 70%)")
        issues.append({
            'type': 'low_accuracy',
            'final_accuracy': final_accuracy,
            'severity': 'high'
        })
    
    return issues

def generate_fix_recommendations(issues, data, logger):
    """生成修復建議"""
    logger.info("🔧 生成修復建議...")
    
    recommendations = []
    
    for issue in issues:
        if issue['type'] == 'overfitting':
            logger.info("🔧 過擬合修復建議:")
            logger.info("  - 增加 Dropout 率 (0.3 -> 0.5)")
            logger.info("  - 增加 Weight Decay (1e-4 -> 5e-4)")
            logger.info("  - 減少學習率 (當前 * 0.8)")
            logger.info("  - 增加數據增強")
            recommendations.append({
                'issue': 'overfitting',
                'solutions': [
                    'increase_dropout',
                    'increase_weight_decay', 
                    'reduce_learning_rate',
                    'increase_data_augmentation'
                ]
            })
        
        elif issue['type'] == 'accuracy_drop':
            logger.info("🔧 準確率下降修復建議:")
            logger.info("  - 調整分類損失權重")
            logger.info("  - 使用 Focal Loss")
            logger.info("  - 實施學習率調度")
            logger.info("  - 檢查數據平衡")
            recommendations.append({
                'issue': 'accuracy_drop',
                'solutions': [
                    'adjust_classification_weight',
                    'use_focal_loss',
                    'implement_lr_scheduling',
                    'check_data_balance'
                ]
            })
        
        elif issue['type'] == 'unstable_training':
            logger.info("🔧 訓練不穩定修復建議:")
            logger.info("  - 降低學習率")
            logger.info("  - 增加批次大小")
            logger.info("  - 使用梯度裁剪")
            logger.info("  - 檢查數據預處理")
            recommendations.append({
                'issue': 'unstable_training',
                'solutions': [
                    'reduce_learning_rate',
                    'increase_batch_size',
                    'use_gradient_clipping',
                    'check_data_preprocessing'
                ]
            })
        
        elif issue['type'] == 'low_accuracy':
            logger.info("🔧 低準確率修復建議:")
            logger.info("  - 增加訓練輪數")
            logger.info("  - 調整模型架構")
            logger.info("  - 優化數據增強")
            logger.info("  - 檢查標籤質量")
            recommendations.append({
                'issue': 'low_accuracy',
                'solutions': [
                    'increase_epochs',
                    'adjust_model_architecture',
                    'optimize_data_augmentation',
                    'check_label_quality'
                ]
            })
    
    return recommendations

def create_analysis_plots(data, output_dir, logger):
    """創建分析圖表"""
    logger.info("📊 創建分析圖表...")
    
    try:
        # 創建圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('訓練分析報告', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(data['train_losses']) + 1)
        
        # 1. 損失曲線
        axes[0, 0].plot(epochs, data['train_losses'], 'b-', label='訓練損失', linewidth=2)
        axes[0, 0].plot(epochs, data['val_losses'], 'r-', label='驗證損失', linewidth=2)
        axes[0, 0].set_title('損失曲線')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 準確率曲線
        axes[0, 1].plot(epochs, data['val_accuracies'], 'g-', label='驗證準確率', linewidth=2)
        axes[0, 1].set_title('準確率曲線')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 問題檢測
        overfitting_epochs = [i+1 for i, flag in enumerate(data['overfitting_flags']) if flag]
        accuracy_drop_epochs = [i+1 for i, flag in enumerate(data['accuracy_drops']) if flag]
        
        axes[1, 0].plot(epochs, data['train_losses'], 'b-', label='訓練損失', alpha=0.7)
        axes[1, 0].plot(epochs, data['val_losses'], 'r-', label='驗證損失', alpha=0.7)
        if overfitting_epochs:
            axes[1, 0].scatter(overfitting_epochs, [data['val_losses'][i-1] for i in overfitting_epochs], 
                              color='red', s=50, label='過擬合', zorder=5)
        axes[1, 0].set_title('問題檢測')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 準確率問題
        axes[1, 1].plot(epochs, data['val_accuracies'], 'g-', label='驗證準確率', alpha=0.7)
        if accuracy_drop_epochs:
            axes[1, 1].scatter(accuracy_drop_epochs, [data['val_accuracies'][i-1] for i in accuracy_drop_epochs], 
                              color='orange', s=50, label='準確率下降', zorder=5)
        axes[1, 1].set_title('準確率問題檢測')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        plot_file = os.path.join(output_dir, 'training_analysis_plots.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"✅ 分析圖表已保存: {plot_file}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"❌ 創建圖表失敗: {e}")

def generate_fix_script(recommendations, output_dir, logger):
    """生成修復腳本"""
    logger.info("🔧 生成修復腳本...")
    
    script_content = """#!/usr/bin/env python3
\"\"\"
🔧 自動修復腳本
基於分析結果生成的修復建議
\"\"\"

import argparse
import os

def apply_fixes():
    \"\"\"應用修復建議\"\"\"
    print("🔧 開始應用修復建議...")
    
"""
    
    for rec in recommendations:
        if rec['issue'] == 'overfitting':
            script_content += """
    # 過擬合修復
    print("🔧 應用過擬合修復...")
    # 1. 增加 Dropout
    dropout_rate = 0.5  # 從 0.3 增加到 0.5
    
    # 2. 增加 Weight Decay
    weight_decay = 5e-4  # 從 1e-4 增加到 5e-4
    
    # 3. 減少學習率
    learning_rate = 0.0008  # 從 0.001 減少到 0.0008
    
    print(f"  - Dropout 率: {dropout_rate}")
    print(f"  - Weight Decay: {weight_decay}")
    print(f"  - 學習率: {learning_rate}")
"""
        
        elif rec['issue'] == 'accuracy_drop':
            script_content += """
    # 準確率下降修復
    print("🔧 應用準確率下降修復...")
    # 1. 調整分類損失權重
    cls_weight = 8.0  # 適中權重
    
    # 2. 使用 Focal Loss
    use_focal_loss = True
    
    # 3. 學習率調度
    use_lr_scheduler = True
    
    print(f"  - 分類損失權重: {cls_weight}")
    print(f"  - 使用 Focal Loss: {use_focal_loss}")
    print(f"  - 學習率調度: {use_lr_scheduler}")
"""
        
        elif rec['issue'] == 'unstable_training':
            script_content += """
    # 訓練不穩定修復
    print("🔧 應用訓練不穩定修復...")
    # 1. 降低學習率
    learning_rate = 0.0005  # 進一步降低
    
    # 2. 增加批次大小
    batch_size = 32  # 從 16 增加到 32
    
    # 3. 梯度裁剪
    gradient_clip = 1.0
    
    print(f"  - 學習率: {learning_rate}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 梯度裁剪: {gradient_clip}")
"""
    
    script_content += """
    print("✅ 修復建議已生成")
    print("📝 請手動更新訓練腳本中的相應參數")

if __name__ == "__main__":
    apply_fixes()
"""
    
    # 保存腳本
    script_file = os.path.join(output_dir, 'apply_fixes.py')
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    logger.info(f"✅ 修復腳本已生成: {script_file}")

def generate_report(data, issues, recommendations, output_dir, logger):
    """生成詳細報告"""
    logger.info("📝 生成詳細報告...")
    
    report_content = f"""# 🔧 訓練分析與修復報告

## 📊 訓練概況
- **訓練輪數**: {len(data['train_losses'])}
- **最終訓練損失**: {data['train_losses'][-1]:.4f}
- **最終驗證損失**: {data['val_losses'][-1]:.4f}
- **最終驗證準確率**: {data['val_accuracies'][-1]:.4f}
- **最佳驗證準確率**: {max(data['val_accuracies']):.4f}

## 🚨 檢測到的問題

"""
    
    if issues:
        for i, issue in enumerate(issues, 1):
            report_content += f"### {i}. {issue['type'].replace('_', ' ').title()}\n"
            report_content += f"- **嚴重程度**: {issue['severity']}\n"
            if 'count' in issue:
                report_content += f"- **發生次數**: {issue['count']}\n"
            if 'std' in issue:
                report_content += f"- **標準差**: {issue['std']:.4f}\n"
            if 'final_accuracy' in issue:
                report_content += f"- **最終準確率**: {issue['final_accuracy']:.4f}\n"
            report_content += "\n"
    else:
        report_content += "✅ 未檢測到嚴重問題\n\n"
    
    report_content += "## 🔧 修復建議\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        report_content += f"### {i}. {rec['issue'].replace('_', ' ').title()}\n"
        for solution in rec['solutions']:
            report_content += f"- {solution.replace('_', ' ').title()}\n"
        report_content += "\n"
    
    report_content += f"""
## 📈 性能統計
- **訓練損失範圍**: {min(data['train_losses']):.4f} - {max(data['train_losses']):.4f}
- **驗證損失範圍**: {min(data['val_losses']):.4f} - {max(data['val_losses']):.4f}
- **驗證準確率範圍**: {min(data['val_accuracies']):.4f} - {max(data['val_accuracies']):.4f}
- **過擬合檢測次數**: {sum(data['overfitting_flags'])}
- **準確率下降次數**: {sum(data['accuracy_drops'])}

## 🎯 建議行動
1. 查看生成的修復腳本: `apply_fixes.py`
2. 根據建議調整訓練參數
3. 重新訓練模型
4. 監控新的訓練過程

---
*報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存報告
    report_file = os.path.join(output_dir, 'analysis_and_fix_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"✅ 詳細報告已生成: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='訓練分析與修復工具')
    parser.add_argument('--history', type=str, required=True, help='訓練歷史文件路徑')
    parser.add_argument('--output', type=str, default='analysis_output', help='輸出目錄')
    parser.add_argument('--log', type=str, default='training_analysis.log', help='日誌文件')
    
    args = parser.parse_args()
    
    # 設置日誌
    logger = setup_logging(args.log)
    logger.info("🚀 開始訓練分析與修復流程...")
    
    # 創建輸出目錄
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"📁 輸出目錄: {args.output}")
    
    try:
        # 1. 分析訓練歷史
        data = analyze_training_history(args.history, logger)
        if data is None:
            return
        
        # 2. 檢測問題
        issues = detect_issues(data, logger)
        
        # 3. 生成修復建議
        recommendations = generate_fix_recommendations(issues, data, logger)
        
        # 4. 創建圖表
        create_analysis_plots(data, args.output, logger)
        
        # 5. 生成修復腳本
        generate_fix_script(recommendations, args.output, logger)
        
        # 6. 生成報告
        generate_report(data, issues, recommendations, args.output, logger)
        
        logger.info("🎉 分析與修復流程完成!")
        logger.info(f"📁 輸出文件:")
        logger.info(f"  - 分析圖表: {args.output}/training_analysis_plots.png")
        logger.info(f"  - 修復腳本: {args.output}/apply_fixes.py")
        logger.info(f"  - 詳細報告: {args.output}/analysis_and_fix_report.md")
        logger.info(f"  - 日誌文件: {args.log}")
        
    except Exception as e:
        logger.error(f"❌ 分析過程出現錯誤: {e}")
        raise

if __name__ == "__main__":
    main() 