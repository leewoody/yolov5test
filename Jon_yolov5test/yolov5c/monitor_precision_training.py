#!/usr/bin/env python3
"""
高精度訓練監控腳本
監控 mAP 0.6+, Precision 0.7+, Recall 0.7+, F1-Score 0.6+ 目標達成情況
"""

import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime
import os


class PrecisionTrainingMonitor:
    """高精度訓練監控器"""
    
    def __init__(self):
        self.output_dir = Path('runs/advanced_dual_precision')
        self.history_file = self.output_dir / 'training_history.json'
        self.target_metrics = {
            'mAP': 0.6,
            'precision': 0.7,
            'recall': 0.7,
            'f1_score': 0.6
        }
        
    def load_training_history(self):
        """載入訓練歷史"""
        if not self.history_file.exists():
            return None
            
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"載入訓練歷史失敗: {e}")
            return None
    
    def check_target_achievement(self, history):
        """檢查目標達成情況"""
        if not history or not history.get('mAP'):
            return False, {}
            
        # 獲取最新指標
        latest_metrics = {
            'mAP': history['mAP'][-1] if history['mAP'] else 0,
            'precision': history['precision'][-1] if history['precision'] else 0,
            'recall': history['recall'][-1] if history['recall'] else 0,
            'f1_score': history['f1_score'][-1] if history['f1_score'] else 0
        }
        
        # 檢查每個目標
        achievements = {}
        all_achieved = True
        
        for metric, target in self.target_metrics.items():
            current_value = latest_metrics[metric]
            achieved = current_value >= target
            achievements[metric] = {
                'current': current_value,
                'target': target,
                'achieved': achieved,
                'progress': (current_value / target * 100) if target > 0 else 0
            }
            if not achieved:
                all_achieved = False
                
        return all_achieved, achievements
    
    def create_progress_plots(self, history):
        """創建進度圖表"""
        if not history or not history.get('epochs'):
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = history['epochs']
        
        # 1. 損失曲線
        if history.get('total_losses'):
            ax1.plot(epochs[:len(history['total_losses'])], history['total_losses'], 'b-', label='總損失', linewidth=2)
            if history.get('detection_losses'):
                ax1.plot(epochs[:len(history['detection_losses'])], history['detection_losses'], 'r--', label='檢測損失', alpha=0.7)
            if history.get('classification_losses'):
                ax1.plot(epochs[:len(history['classification_losses'])], history['classification_losses'], 'g--', label='分類損失', alpha=0.7)
            ax1.set_title('📉 損失曲線', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. mAP 進度
        if history.get('mAP'):
            validation_epochs = [epochs[i] for i in range(0, len(epochs), 5) if i < len(history['mAP'])][:len(history['mAP'])]
            ax2.plot(validation_epochs, history['mAP'], 'bo-', label='mAP@0.5', linewidth=2, markersize=6)
            ax2.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='目標 (0.6)')
            ax2.set_title('🎯 mAP@0.5 進度', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mAP@0.5')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        # 3. Precision & Recall
        if history.get('precision') and history.get('recall'):
            validation_epochs = [epochs[i] for i in range(0, len(epochs), 5) if i < len(history['precision'])][:len(history['precision'])]
            ax3.plot(validation_epochs, history['precision'], 'go-', label='Precision', linewidth=2, markersize=6)
            ax3.plot(validation_epochs, history['recall'], 'mo-', label='Recall', linewidth=2, markersize=6)
            ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='目標 (0.7)')
            ax3.set_title('⚖️ Precision & Recall', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
        
        # 4. F1-Score
        if history.get('f1_score'):
            validation_epochs = [epochs[i] for i in range(0, len(epochs), 5) if i < len(history['f1_score'])][:len(history['f1_score'])]
            ax4.plot(validation_epochs, history['f1_score'], 'co-', label='F1-Score', linewidth=2, markersize=6)
            ax4.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='目標 (0.6)')
            ax4.set_title('🎯 F1-Score 進度', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('F1-Score')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def display_status(self, history, achievements):
        """顯示訓練狀態"""
        print("\n" + "="*80)
        print(f"🎯 高精度雙獨立骨幹網路訓練監控")
        print(f"📅 監控時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if not history:
            print("❌ 未找到訓練歷史文件")
            return
        
        # 基本訓練信息
        current_epoch = len(history.get('epochs', []))
        total_losses = len(history.get('total_losses', []))
        validations = len(history.get('mAP', []))
        
        print(f"📊 訓練進度:")
        print(f"   當前 Epoch: {current_epoch}")
        print(f"   已完成訓練批次: {total_losses}")
        print(f"   已完成驗證: {validations}")
        
        if not achievements:
            print("⚠️ 尚未開始驗證")
            return
        
        print(f"\n🎯 目標達成情況:")
        all_achieved = True
        
        for metric, info in achievements.items():
            status = "✅" if info['achieved'] else "❌"
            progress_bar = "█" * int(info['progress'] // 10) + "░" * (10 - int(info['progress'] // 10))
            
            print(f"   {metric.upper():12} {status} {info['current']:.4f} / {info['target']:.1f} "
                  f"[{progress_bar}] {info['progress']:.1f}%")
            
            if not info['achieved']:
                all_achieved = False
        
        if all_achieved:
            print(f"\n🎉 恭喜！所有目標已達成！")
        else:
            # 計算離目標最近的指標
            gaps = [(metric, info['target'] - info['current']) 
                   for metric, info in achievements.items() if not info['achieved']]
            if gaps:
                closest = min(gaps, key=lambda x: x[1])
                print(f"\n📈 最接近達成的目標: {closest[0].upper()} (還需提升 {closest[1]:.4f})")
        
        # 訓練趨勢分析
        if len(history.get('mAP', [])) >= 2:
            recent_map = history['mAP'][-2:]
            trend = "📈 上升" if recent_map[-1] > recent_map[-2] else "📉 下降"
            print(f"\n📊 最近 mAP 趨勢: {trend}")
    
    def monitor_continuously(self, interval=30):
        """持續監控"""
        print("🔍 開始持續監控高精度訓練...")
        print(f"⏰ 監控間隔: {interval} 秒")
        print("按 Ctrl+C 停止監控")
        
        try:
            while True:
                history = self.load_training_history()
                all_achieved, achievements = self.check_target_achievement(history)
                
                # 清屏
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # 顯示狀態
                self.display_status(history, achievements)
                
                # 創建進度圖表
                if history:
                    self.create_progress_plots(history)
                    print(f"\n📈 進度圖表已更新: {self.output_dir}/precision_training_progress.png")
                
                # 如果目標達成，發出通知
                if all_achieved:
                    print("\n🎊 所有目標已達成！可以考慮停止訓練。")
                    break
                
                # 等待下次檢查
                print(f"\n⏰ {interval} 秒後再次檢查...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 監控已停止")
    
    def generate_summary_report(self):
        """生成總結報告"""
        history = self.load_training_history()
        if not history:
            print("❌ 無法生成報告：未找到訓練歷史")
            return
        
        all_achieved, achievements = self.check_target_achievement(history)
        
        report = f"""# 🎯 高精度訓練總結報告

**生成時間**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 📊 目標達成情況

| 指標 | 當前值 | 目標值 | 達成率 | 狀態 |
|------|--------|--------|--------|------|
"""
        
        for metric, info in achievements.items():
            status = "✅ 達成" if info['achieved'] else "❌ 未達成"
            report += f"| {metric.upper()} | {info['current']:.4f} | {info['target']:.1f} | {info['progress']:.1f}% | {status} |\n"
        
        report += f"""
## 📈 訓練統計

- **總訓練輪數**: {len(history.get('epochs', []))}
- **驗證次數**: {len(history.get('mAP', []))}
- **最終 mAP**: {history.get('mAP', [0])[-1] if history.get('mAP') else 'N/A'}
- **最終損失**: {history.get('total_losses', [0])[-1] if history.get('total_losses') else 'N/A'}

## 🎯 總體狀態

{"🎉 **所有目標已達成！**" if all_achieved else "⚠️ **仍有目標未達成，建議繼續訓練**"}

## 📁 相關文件

- 訓練歷史: `{self.history_file}`
- 進度圖表: `{self.output_dir}/precision_training_progress.png`
- 最佳模型: `{self.output_dir}/best_model.pt`

---
**報告生成器**: 高精度訓練監控系統
"""
        
        report_path = self.output_dir / 'precision_training_summary.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 總結報告已生成: {report_path}")


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='高精度訓練監控')
    parser.add_argument('--mode', type=str, choices=['once', 'continuous', 'report'], 
                       default='continuous', help='監控模式')
    parser.add_argument('--interval', type=int, default=30, help='持續監控間隔(秒)')
    
    args = parser.parse_args()
    
    monitor = PrecisionTrainingMonitor()
    
    if args.mode == 'once':
        # 單次檢查
        history = monitor.load_training_history()
        all_achieved, achievements = monitor.check_target_achievement(history)
        monitor.display_status(history, achievements)
        if history:
            monitor.create_progress_plots(history)
            
    elif args.mode == 'continuous':
        # 持續監控
        monitor.monitor_continuously(interval=args.interval)
        
    elif args.mode == 'report':
        # 生成報告
        monitor.generate_summary_report()


if __name__ == "__main__":
    main() 