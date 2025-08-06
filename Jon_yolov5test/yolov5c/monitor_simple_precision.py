#!/usr/bin/env python3
"""
簡化版高精度訓練監控腳本
監控 mAP 0.6+, Precision 0.7+, Recall 0.7+, F1-Score 0.6+ 目標達成情況
"""

import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime
import os


class SimplePrecisionMonitor:
    """簡化版高精度訓練監控器"""
    
    def __init__(self):
        self.output_dir = Path('runs/simple_precision_training')
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
    
    def create_live_plots(self, history):
        """創建實時進度圖表"""
        if not history or not history.get('epochs'):
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = history['epochs']
        
        # 1. 損失曲線
        if history.get('total_losses'):
            ax1.plot(epochs, history['total_losses'], 'b-', label='總損失', linewidth=2, marker='o')
            if history.get('detection_losses'):
                ax1.plot(epochs, history['detection_losses'], 'r--', label='檢測損失', alpha=0.8, linewidth=1.5)
            if history.get('classification_losses'):
                ax1.plot(epochs, history['classification_losses'], 'g--', label='分類損失', alpha=0.8, linewidth=1.5)
            ax1.set_title('📉 損失曲線變化', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. mAP 進度追蹤
        if history.get('mAP'):
            validation_epochs = [epochs[i] for i in range(4, len(epochs), 5)][:len(history['mAP'])]
            ax2.plot(validation_epochs, history['mAP'], 'bo-', label='mAP@0.5', linewidth=3, markersize=8)
            ax2.axhline(y=0.6, color='red', linestyle='--', linewidth=2, alpha=0.8, label='目標 (0.6)')
            
            # 添加當前值標註
            if history['mAP']:
                current_map = history['mAP'][-1]
                ax2.text(validation_epochs[-1], current_map + 0.02, f'{current_map:.3f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            ax2.set_title('🎯 mAP@0.5 進度追蹤', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mAP@0.5')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        # 3. Precision & Recall 雙軌追蹤
        if history.get('precision') and history.get('recall'):
            validation_epochs = [epochs[i] for i in range(4, len(epochs), 5)][:len(history['precision'])]
            ax3.plot(validation_epochs, history['precision'], 'go-', label='Precision', linewidth=3, markersize=8)
            ax3.plot(validation_epochs, history['recall'], 'mo-', label='Recall', linewidth=3, markersize=8)
            ax3.axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.8, label='目標 (0.7)')
            
            # 添加最新值標註
            if history['precision'] and history['recall']:
                current_prec = history['precision'][-1]
                current_rec = history['recall'][-1]
                ax3.text(validation_epochs[-1], current_prec + 0.02, f'P:{current_prec:.3f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10, color='green')
                ax3.text(validation_epochs[-1], current_rec - 0.05, f'R:{current_rec:.3f}', 
                        ha='center', va='top', fontweight='bold', fontsize=10, color='purple')
            
            ax3.set_title('⚖️ Precision & Recall 平衡', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
        
        # 4. 目標達成儀表板
        if history.get('mAP') and history.get('precision') and history.get('recall') and history.get('f1_score'):
            latest_metrics = {
                'mAP': history['mAP'][-1],
                'Precision': history['precision'][-1],
                'Recall': history['recall'][-1],
                'F1-Score': history['f1_score'][-1]
            }
            
            targets = [0.6, 0.7, 0.7, 0.6]
            labels = list(latest_metrics.keys())
            values = list(latest_metrics.values())
            
            # 創建雷達圖
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]  # 閉合圖形
            angles += angles[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=3, label='當前值', color='blue')
            ax4.fill(angles, values, alpha=0.25, color='blue')
            
            # 目標線
            target_values = targets + targets[:1]
            ax4.plot(angles, target_values, 's-', linewidth=2, label='目標值', color='red', alpha=0.7)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(labels)
            ax4.set_ylim(0, 1)
            ax4.set_title('🏆 目標達成儀表板', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'live_precision_progress.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def display_live_status(self, history, achievements):
        """顯示實時狀態"""
        print("\n" + "="*90)
        print(f"🎯 簡化版高精度雙獨立骨幹網路 - 實時監控")
        print(f"📅 更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*90)
        
        if not history:
            print("❌ 訓練尚未開始或歷史文件不存在")
            return
        
        # 基本訓練信息
        current_epoch = len(history.get('epochs', []))
        total_losses = len(history.get('total_losses', []))
        validations = len(history.get('mAP', []))
        
        print(f"📊 訓練狀態:")
        print(f"   🔄 當前 Epoch: {current_epoch}/30")
        print(f"   📈 完成批次: {total_losses}")
        print(f"   🔍 驗證次數: {validations}")
        
        # 最新損失信息
        if history.get('total_losses'):
            latest_loss = history['total_losses'][-1]
            latest_det_loss = history['detection_losses'][-1] if history.get('detection_losses') else 0
            latest_cls_loss = history['classification_losses'][-1] if history.get('classification_losses') else 0
            
            print(f"\n🔥 最新損失:")
            print(f"   總損失: {latest_loss:.4f}")
            print(f"   檢測損失: {latest_det_loss:.4f}")
            print(f"   分類損失: {latest_cls_loss:.4f}")
        
        if not achievements:
            print("\n⚠️ 尚未開始驗證 (每5個epoch驗證一次)")
            return
        
        print(f"\n🎯 目標達成情況:")
        all_achieved = True
        
        for metric, info in achievements.items():
            status = "✅" if info['achieved'] else "❌"
            progress_blocks = int(info['progress'] // 5)  # 每5%一個方塊
            progress_bar = "█" * progress_blocks + "░" * (20 - progress_blocks)
            
            print(f"   {metric.upper():12} {status} {info['current']:.4f}/{info['target']:.1f} "
                  f"[{progress_bar}] {info['progress']:.1f}%")
            
            if not info['achieved']:
                all_achieved = False
                gap = info['target'] - info['current']
                print(f"                     └─ 還需提升 {gap:.4f}")
        
        if all_achieved:
            print(f"\n🎉 🎊 恭喜！所有目標已達成！🎊 🎉")
            print(f"🏆 可以考慮停止訓練或繼續優化以獲得更高精度")
        else:
            # 預測達成時間
            if len(history.get('mAP', [])) >= 2:
                recent_progress = []
                for metric, info in achievements.items():
                    if len(history.get(metric.replace('map', 'mAP'), [])) >= 2:
                        recent_values = history[metric.replace('map', 'mAP').replace('MAP', 'mAP')][-2:]
                        if len(recent_values) == 2:
                            improvement = recent_values[1] - recent_values[0]
                            if improvement > 0 and not info['achieved']:
                                gap = info['target'] - info['current']
                                epochs_needed = gap / improvement * 5  # 每5epoch驗證一次
                                recent_progress.append(epochs_needed)
                
                if recent_progress:
                    avg_epochs_needed = np.mean(recent_progress)
                    print(f"\n📈 預測: 約需 {avg_epochs_needed:.0f} 個epoch可達成剩餘目標")
        
        # 訓練趨勢分析
        if len(history.get('mAP', [])) >= 2:
            recent_map = history['mAP'][-2:]
            trend = "📈 上升" if recent_map[-1] > recent_map[-2] else "📉 下降"
            change = abs(recent_map[-1] - recent_map[-2])
            print(f"\n📊 mAP 趨勢: {trend} (變化: {change:.4f})")
    
    def monitor_with_alerts(self, interval=20, max_duration=3600):
        """帶警報的持續監控"""
        print("🔍 開始實時監控簡化版高精度訓練...")
        print(f"⏰ 監控間隔: {interval} 秒")
        print(f"⏱️ 最大監控時長: {max_duration/60:.0f} 分鐘")
        print("按 Ctrl+C 停止監控")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < max_duration:
                history = self.load_training_history()
                all_achieved, achievements = self.check_target_achievement(history)
                
                # 清屏
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # 顯示狀態
                self.display_live_status(history, achievements)
                
                # 創建實時圖表
                if history:
                    self.create_live_plots(history)
                    print(f"\n📈 實時圖表已更新: {self.output_dir}/live_precision_progress.png")
                
                # 目標達成警報
                if all_achieved:
                    print("\n🚨 🎉 警報：所有目標已達成！🎉 🚨")
                    print("建議檢查訓練狀態並考慮停止訓練")
                    
                    # 連續警報
                    for i in range(3):
                        print(f"🔔 目標達成警報 ({i+1}/3)")
                        time.sleep(1)
                    break
                
                # 等待下次檢查
                elapsed = time.time() - start_time
                remaining = max_duration - elapsed
                print(f"\n⏰ {interval} 秒後再次檢查... (剩餘監控時間: {remaining/60:.0f}分鐘)")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 監控已手動停止")
        
        print(f"\n📊 監控結束，總時長: {(time.time() - start_time)/60:.1f} 分鐘")
    
    def generate_progress_report(self):
        """生成進度報告"""
        history = self.load_training_history()
        if not history:
            print("❌ 無法生成報告：未找到訓練歷史")
            return
        
        all_achieved, achievements = self.check_target_achievement(history)
        
        report = f"""# 📊 簡化版高精度訓練進度報告

**生成時間**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}  
**訓練狀態**: {'🎉 目標達成' if all_achieved else '🔄 進行中'}

## 🎯 目標達成情況

| 指標 | 當前值 | 目標值 | 進度 | 狀態 |
|------|--------|--------|------|------|
"""
        
        for metric, info in achievements.items():
            status = "✅ 達成" if info['achieved'] else "🔄 進行中"
            progress_bar = "█" * int(info['progress'] // 5) + "░" * (20 - int(info['progress'] // 5))
            report += f"| **{metric.upper()}** | {info['current']:.4f} | {info['target']:.1f} | {progress_bar} {info['progress']:.1f}% | {status} |\n"
        
        # 訓練統計
        current_epoch = len(history.get('epochs', []))
        latest_loss = history.get('total_losses', [0])[-1] if history.get('total_losses') else 0
        
        report += f"""

## 📈 訓練統計

- **當前輪數**: {current_epoch}/30
- **完成批次**: {len(history.get('total_losses', []))}
- **驗證次數**: {len(history.get('mAP', []))}
- **最新損失**: {latest_loss:.4f}
- **最佳 mAP**: {max(history.get('mAP', [0])):.4f}

## 🔄 訓練趨勢

"""
        
        if len(history.get('mAP', [])) >= 2:
            recent_map = history['mAP'][-2:]
            trend = "上升 📈" if recent_map[-1] > recent_map[-2] else "下降 📉"
            report += f"- **mAP 趨勢**: {trend}\n"
        
        if achievements:
            gaps = [(metric, info['target'] - info['current']) 
                   for metric, info in achievements.items() if not info['achieved']]
            if gaps:
                closest = min(gaps, key=lambda x: x[1])
                report += f"- **最接近目標**: {closest[0].upper()} (還需 {closest[1]:.4f})\n"
        
        report += f"""

## 📁 相關文件

- 訓練歷史: `{self.history_file}`
- 實時圖表: `{self.output_dir}/live_precision_progress.png`
- 最佳模型: `{self.output_dir}/best_precision_model.pt`

---
**監控系統**: 簡化版高精度訓練監控器
"""
        
        report_path = self.output_dir / 'progress_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 進度報告已生成: {report_path}")


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='簡化版高精度訓練監控')
    parser.add_argument('--mode', type=str, choices=['once', 'live', 'report'], 
                       default='live', help='監控模式')
    parser.add_argument('--interval', type=int, default=20, help='監控間隔(秒)')
    parser.add_argument('--duration', type=int, default=3600, help='最大監控時長(秒)')
    
    args = parser.parse_args()
    
    monitor = SimplePrecisionMonitor()
    
    if args.mode == 'once':
        # 單次檢查
        history = monitor.load_training_history()
        all_achieved, achievements = monitor.check_target_achievement(history)
        monitor.display_live_status(history, achievements)
        if history:
            monitor.create_live_plots(history)
            
    elif args.mode == 'live':
        # 實時監控
        monitor.monitor_with_alerts(interval=args.interval, max_duration=args.duration)
        
    elif args.mode == 'report':
        # 生成報告
        monitor.generate_progress_report()


if __name__ == "__main__":
    main() 