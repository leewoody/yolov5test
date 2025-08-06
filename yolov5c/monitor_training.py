#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Log Monitor
實時監控訓練日誌
"""

import os
import sys
import time
import re
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import queue


class TrainingMonitor:
    def __init__(self, log_file=None, auto_detect=True, update_interval=2):
        self.log_file = log_file
        self.auto_detect = auto_detect
        self.update_interval = update_interval
        
        # 訓練數據存儲
        self.epoch_data = deque(maxlen=100)
        self.batch_data = deque(maxlen=500)
        self.auto_fix_events = deque(maxlen=50)
        
        # 監控狀態
        self.is_monitoring = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.start_time = None
        
        # 圖表設置
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Monitor - Real-time Log Analysis', fontsize=16)
        
    def auto_detect_log_file(self):
        """自動檢測訓練日誌文件"""
        possible_log_files = [
            "training.log",
            "train.log",
            "output.log",
            "*.log"
        ]
        
        for pattern in possible_log_files:
            if pattern.endswith('*'):
                # 搜索所有日誌文件
                for log_file in Path('.').glob(pattern):
                    if self.is_training_log(log_file):
                        return log_file
            else:
                log_file = Path(pattern)
                if log_file.exists() and self.is_training_log(log_file):
                    return log_file
        
        return None
    
    def is_training_log(self, log_file):
        """判斷是否為訓練日誌文件"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 檢查是否包含訓練相關的關鍵詞
                training_keywords = ['epoch', 'loss', 'train', 'val', 'accuracy']
                return any(keyword in content.lower() for keyword in training_keywords)
        except:
            return False
    
    def parse_log_line(self, line):
        """解析日誌行"""
        line = line.strip()
        
        # 解析epoch信息
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            self.total_epochs = int(epoch_match.group(2))
            return {'type': 'epoch', 'epoch': self.current_epoch, 'total': self.total_epochs}
        
        # 解析batch信息
        batch_match = re.search(r'Batch (\d+)/(\d+), Loss: ([\d.]+)', line)
        if batch_match:
            batch_num = int(batch_match.group(1))
            total_batches = int(batch_match.group(2))
            loss = float(batch_match.group(3))
            
            # 解析bbox和cls損失
            bbox_match = re.search(r'Bbox: ([\d.]+)', line)
            cls_match = re.search(r'Cls: ([\d.]+)', line)
            
            bbox_loss = float(bbox_match.group(1)) if bbox_match else 0.0
            cls_loss = float(cls_match.group(1)) if cls_match else 0.0
            
            return {
                'type': 'batch',
                'epoch': self.current_epoch,
                'batch': batch_num,
                'total_batches': total_batches,
                'loss': loss,
                'bbox_loss': bbox_loss,
                'cls_loss': cls_loss
            }
        
        # 解析epoch總結
        epoch_summary_match = re.search(r'Epoch (\d+):', line)
        if epoch_summary_match:
            # 解析訓練損失
            train_loss_match = re.search(r'Train Loss: ([\d.]+)', line)
            val_loss_match = re.search(r'Val Loss: ([\d.]+)', line)
            val_acc_match = re.search(r'Val Cls Accuracy: ([\d.]+)', line)
            best_acc_match = re.search(r'Best Val Cls Accuracy: ([\d.]+)', line)
            
            if train_loss_match and val_loss_match:
                return {
                    'type': 'epoch_summary',
                    'epoch': int(epoch_summary_match.group(1)),
                    'train_loss': float(train_loss_match.group(1)),
                    'val_loss': float(val_loss_match.group(1)),
                    'val_accuracy': float(val_acc_match.group(1)) if val_acc_match else 0.0,
                    'best_accuracy': float(best_acc_match.group(1)) if best_acc_match else 0.0
                }
        
        # 解析Auto-Fix事件
        if '過擬合風險檢測' in line:
            return {'type': 'auto_fix', 'event': 'overfitting_detected', 'line': line}
        elif '準確率下降檢測' in line:
            return {'type': 'auto_fix', 'event': 'accuracy_drop_detected', 'line': line}
        elif 'Auto-fix applied' in line:
            return {'type': 'auto_fix', 'event': 'fix_applied', 'line': line}
        
        return None
    
    def update_plots(self, frame):
        """更新圖表"""
        if not self.is_monitoring:
            return
        
        # 清空圖表
        for ax in self.axes.flat:
            ax.clear()
        
        # 1. 損失曲線
        if self.epoch_data:
            epochs = [d['epoch'] for d in self.epoch_data if 'train_loss' in d]
            train_losses = [d['train_loss'] for d in self.epoch_data if 'train_loss' in d]
            val_losses = [d['val_loss'] for d in self.epoch_data if 'val_loss' in d]
            
            if epochs and train_losses and val_losses:
                self.axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
                self.axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
                self.axes[0, 0].set_title('Training and Validation Loss')
                self.axes[0, 0].set_xlabel('Epoch')
                self.axes[0, 0].set_ylabel('Loss')
                self.axes[0, 0].legend()
                self.axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 準確率曲線
        if self.epoch_data:
            epochs = [d['epoch'] for d in self.epoch_data if 'val_accuracy' in d]
            accuracies = [d['val_accuracy'] for d in self.epoch_data if 'val_accuracy' in d]
            best_accuracies = [d['best_accuracy'] for d in self.epoch_data if 'best_accuracy' in d]
            
            if epochs and accuracies:
                self.axes[0, 1].plot(epochs, accuracies, 'g-', label='Val Accuracy', linewidth=2)
                if best_accuracies:
                    self.axes[0, 1].plot(epochs, best_accuracies, 'g--', label='Best Accuracy', linewidth=2)
                self.axes[0, 1].set_title('Validation Accuracy')
                self.axes[0, 1].set_xlabel('Epoch')
                self.axes[0, 1].set_ylabel('Accuracy')
                self.axes[0, 1].legend()
                self.axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Batch損失變化
        if self.batch_data:
            batch_indices = list(range(len(self.batch_data)))
            losses = [d['loss'] for d in self.batch_data]
            bbox_losses = [d['bbox_loss'] for d in self.batch_data]
            cls_losses = [d['cls_loss'] for d in self.batch_data]
            
            self.axes[1, 0].plot(batch_indices, losses, 'b-', label='Total Loss', alpha=0.7)
            self.axes[1, 0].plot(batch_indices, bbox_losses, 'r-', label='Bbox Loss', alpha=0.7)
            self.axes[1, 0].plot(batch_indices, cls_losses, 'g-', label='Cls Loss', alpha=0.7)
            self.axes[1, 0].set_title('Batch Loss Progression')
            self.axes[1, 0].set_xlabel('Batch')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Auto-Fix事件統計
        if self.auto_fix_events:
            event_types = [d['event'] for d in self.auto_fix_events]
            overfitting_count = event_types.count('overfitting_detected')
            accuracy_drop_count = event_types.count('accuracy_drop_detected')
            fix_applied_count = event_types.count('fix_applied')
            
            events = ['Overfitting', 'Accuracy Drop', 'Fix Applied']
            counts = [overfitting_count, accuracy_drop_count, fix_applied_count]
            
            bars = self.axes[1, 1].bar(events, counts, color=['red', 'orange', 'green'], alpha=0.7)
            self.axes[1, 1].set_title('Auto-Fix Events')
            self.axes[1, 1].set_ylabel('Count')
            
            # 添加數值標籤
            for bar, count in zip(bars, counts):
                self.axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
    
    def monitor_log_file(self):
        """監控日誌文件"""
        if not self.log_file:
            if self.auto_detect:
                self.log_file = self.auto_detect_log_file()
            
            if not self.log_file:
                print("❌ 未找到訓練日誌文件")
                return
        
        print(f"🔍 開始監控日誌文件: {self.log_file}")
        print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.is_monitoring = True
        self.start_time = time.time()
        
        # 讀取已存在的內容
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.process_log_line(line)
        except FileNotFoundError:
            print(f"⚠️ 日誌文件不存在，等待創建: {self.log_file}")
        
        # 實時監控新內容
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                # 移動到文件末尾
                f.seek(0, 2)
                
                while self.is_monitoring:
                    line = f.readline()
                    if line:
                        self.process_log_line(line)
                    else:
                        time.sleep(self.update_interval)
                        
        except KeyboardInterrupt:
            print("\n⏹️ 監控已停止")
        except Exception as e:
            print(f"❌ 監控錯誤: {e}")
        finally:
            self.is_monitoring = False
    
    def process_log_line(self, line):
        """處理日誌行"""
        parsed = self.parse_log_line(line)
        if not parsed:
            return
        
        # 根據類型處理數據
        if parsed['type'] == 'epoch_summary':
            self.epoch_data.append(parsed)
            self.print_epoch_summary(parsed)
        elif parsed['type'] == 'batch':
            self.batch_data.append(parsed)
        elif parsed['type'] == 'auto_fix':
            self.auto_fix_events.append(parsed)
            self.print_auto_fix_event(parsed)
    
    def print_epoch_summary(self, data):
        """打印epoch總結"""
        epoch = data['epoch']
        train_loss = data['train_loss']
        val_loss = data['val_loss']
        val_acc = data['val_accuracy']
        best_acc = data['best_accuracy']
        
        # 計算運行時間
        if self.start_time:
            elapsed = time.time() - self.start_time
            elapsed_str = f"{elapsed/60:.1f}分鐘"
        else:
            elapsed_str = "未知"
        
        # 計算進度
        if self.total_epochs > 0:
            progress = (epoch / self.total_epochs) * 100
            eta = (elapsed / epoch) * (self.total_epochs - epoch) if epoch > 0 else 0
            eta_str = f"{eta/60:.1f}分鐘" if eta > 0 else "未知"
        else:
            progress = 0
            eta_str = "未知"
        
        print(f"\n{'='*60}")
        print(f"📊 Epoch {epoch} 總結")
        print(f"{'='*60}")
        print(f"訓練損失: {train_loss:.4f}")
        print(f"驗證損失: {val_loss:.4f}")
        print(f"驗證準確率: {val_acc:.4f} ({val_acc*100:.1f}%)")
        print(f"最佳準確率: {best_acc:.4f} ({best_acc*100:.1f}%)")
        print(f"進度: {progress:.1f}% ({epoch}/{self.total_epochs})")
        print(f"運行時間: {elapsed_str}")
        print(f"預計剩餘: {eta_str}")
        
        # 分析趨勢
        if len(self.epoch_data) > 1:
            prev_data = list(self.epoch_data)[-2]
            train_loss_change = train_loss - prev_data['train_loss']
            val_loss_change = val_loss - prev_data['val_loss']
            acc_change = val_acc - prev_data['val_accuracy']
            
            print(f"\n📈 趨勢分析:")
            print(f"訓練損失變化: {train_loss_change:+.4f}")
            print(f"驗證損失變化: {val_loss_change:+.4f}")
            print(f"準確率變化: {acc_change:+.4f}")
            
            # 過擬合檢測
            if train_loss < val_loss:
                print("⚠️ 過擬合風險: 訓練損失 < 驗證損失")
            elif val_loss > prev_data['val_loss']:
                print("⚠️ 驗證損失上升")
            
            # 準確率分析
            if val_acc < prev_data['val_accuracy']:
                print("⚠️ 準確率下降")
            elif val_acc > best_acc:
                print("✅ 新的最佳準確率！")
    
    def print_auto_fix_event(self, data):
        """打印Auto-Fix事件"""
        event = data['event']
        line = data['line']
        
        print(f"\n🔧 Auto-Fix 事件: {event}")
        print(f"詳細信息: {line.strip()}")
        
        if event == 'overfitting_detected':
            print("🎯 系統正在自動調整正則化參數...")
        elif event == 'accuracy_drop_detected':
            print("🎯 系統正在自動調整分類損失權重...")
        elif event == 'fix_applied':
            print("✅ Auto-Fix 已應用")
    
    def start_monitoring(self):
        """開始監控"""
        # 啟動圖表更新
        ani = animation.FuncAnimation(self.fig, self.update_plots, interval=1000)
        
        # 啟動日誌監控
        monitor_thread = threading.Thread(target=self.monitor_log_file)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 顯示圖表
        plt.show()
        
        # 等待監控結束
        try:
            while self.is_monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ 監控已停止")
            self.is_monitoring = False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Training Log Monitor')
    parser.add_argument('--log-file', type=str, help='日誌文件路徑')
    parser.add_argument('--no-auto-detect', action='store_true', help='禁用自動檢測日誌文件')
    parser.add_argument('--update-interval', type=float, default=2.0, help='更新間隔（秒）')
    
    args = parser.parse_args()
    
    # 創建監控器
    monitor = TrainingMonitor(
        log_file=args.log_file,
        auto_detect=not args.no_auto_detect,
        update_interval=args.update_interval
    )
    
    print("🔍 訓練日誌監控器")
    print("="*50)
    print("功能:")
    print("- 實時監控訓練進度")
    print("- 自動檢測過擬合和準確率下降")
    print("- 顯示Auto-Fix事件")
    print("- 實時圖表更新")
    print("- 趨勢分析和預測")
    
    # 開始監控
    monitor.start_monitoring()


if __name__ == "__main__":
    main() 