#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Training Monitor
簡化訓練監控器
"""

import os
import sys
import time
import re
import subprocess
from pathlib import Path
from datetime import datetime


class SimpleMonitor:
    def __init__(self):
        self.current_epoch = 0
        self.total_epochs = 0
        self.start_time = time.time()
        self.last_epoch_time = None
        
    def monitor_training_process(self):
        """監控訓練進程"""
        print("🔍 簡化訓練監控器")
        print("="*50)
        print("正在監控訓練進程...")
        print("按 Ctrl+C 停止監控")
        print("="*50)
        
        try:
            while True:
                # 檢查是否有Python訓練進程
                training_processes = self.find_training_processes()
                
                if training_processes:
                    print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - 發現 {len(training_processes)} 個訓練進程")
                    
                    for proc in training_processes:
                        self.monitor_process(proc)
                else:
                    print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - 未發現訓練進程")
                
                time.sleep(5)  # 每5秒檢查一次
                
        except KeyboardInterrupt:
            print("\n⏹️ 監控已停止")
    
    def find_training_processes(self):
        """查找訓練進程"""
        try:
            # 使用ps命令查找Python進程
            result = subprocess.run(
                ['ps', 'aux'], 
                capture_output=True, 
                text=True
            )
            
            processes = []
            for line in result.stdout.split('\n'):
                if 'python' in line.lower() and any(keyword in line.lower() for keyword in ['train', 'yolov5']):
                    # 解析進程信息
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        cmd = ' '.join(parts[10:])
                        processes.append({
                            'pid': pid,
                            'cmd': cmd,
                            'line': line
                        })
            
            return processes
            
        except Exception as e:
            print(f"❌ 查找進程失敗: {e}")
            return []
    
    def monitor_process(self, proc):
        """監控單個進程"""
        pid = proc['pid']
        cmd = proc['cmd']
        
        print(f"\n📊 進程 {pid}: {cmd[:80]}...")
        
        # 獲取進程詳細信息
        try:
            # 檢查進程是否還在運行
            result = subprocess.run(
                ['ps', '-p', pid, '-o', 'pid,ppid,etime,pcpu,pmem,comm'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    info = lines[1].split()
                    if len(info) >= 5:
                        etime = info[2]  # 運行時間
                        cpu = info[3]    # CPU使用率
                        mem = info[4]    # 內存使用率
                        
                        print(f"  運行時間: {etime}")
                        print(f"  CPU使用率: {cpu}%")
                        print(f"  內存使用率: {mem}%")
                        
                        # 檢查是否為GPU進程
                        if 'cuda' in cmd.lower() or 'gpu' in cmd.lower():
                            print(f"  🚀 GPU訓練進程")
                        
                        # 檢查是否為Auto-Fix進程
                        if 'advanced' in cmd.lower():
                            print(f"  🔧 Auto-Fix 訓練進程")
                        
                        # 估算進度
                        self.estimate_progress(etime)
            
        except Exception as e:
            print(f"  ❌ 獲取進程信息失敗: {e}")
    
    def estimate_progress(self, etime):
        """估算訓練進度"""
        try:
            # 解析運行時間
            if ':' in etime:
                parts = etime.split(':')
                if len(parts) == 2:  # MM:SS
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    total_seconds = minutes * 60 + seconds
                elif len(parts) == 3:  # HH:MM:SS
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2])
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                else:
                    total_seconds = 0
            else:
                total_seconds = 0
            
            # 基於時間估算進度
            if total_seconds > 0:
                # 假設每個epoch大約需要2-5分鐘
                avg_epoch_time = 180  # 3分鐘
                estimated_epochs = total_seconds / avg_epoch_time
                
                if self.total_epochs > 0:
                    progress = min(estimated_epochs / self.total_epochs * 100, 100)
                    print(f"  估算進度: {progress:.1f}% (約 {estimated_epochs:.1f} epochs)")
                else:
                    print(f"  估算已運行: {estimated_epochs:.1f} epochs")
                    
        except Exception as e:
            pass
    
    def monitor_log_file(self):
        """監控日誌文件"""
        print("🔍 監控日誌文件...")
        
        # 查找可能的日誌文件
        log_files = []
        for pattern in ['*.log', 'training.log', 'train.log', 'output.log']:
            for log_file in Path('.').glob(pattern):
                if log_file.is_file():
                    log_files.append(log_file)
        
        if not log_files:
            print("❌ 未找到日誌文件")
            return
        
        print(f"✅ 找到 {len(log_files)} 個日誌文件:")
        for log_file in log_files:
            print(f"  - {log_file}")
        
        # 監控最新的日誌文件
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print(f"\n📊 監控最新日誌: {latest_log}")
        
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                # 移動到文件末尾
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        self.process_log_line(line.strip())
                    else:
                        time.sleep(1)
                        
        except KeyboardInterrupt:
            print("\n⏹️ 日誌監控已停止")
        except Exception as e:
            print(f"❌ 日誌監控錯誤: {e}")
    
    def process_log_line(self, line):
        """處理日誌行"""
        if not line:
            return
        
        # 解析epoch信息
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            self.total_epochs = int(epoch_match.group(2))
            self.last_epoch_time = time.time()
            
            print(f"\n📈 Epoch {self.current_epoch}/{self.total_epochs}")
        
        # 解析epoch總結
        if 'Epoch' in line and 'Train Loss:' in line:
            self.parse_epoch_summary(line)
        
        # 解析Auto-Fix事件
        if '過擬合風險檢測' in line or '準確率下降檢測' in line:
            print(f"\n🔧 Auto-Fix: {line}")
    
    def parse_epoch_summary(self, line):
        """解析epoch總結"""
        try:
            # 提取損失信息
            train_loss_match = re.search(r'Train Loss: ([\d.]+)', line)
            val_loss_match = re.search(r'Val Loss: ([\d.]+)', line)
            val_acc_match = re.search(r'Val Cls Accuracy: ([\d.]+)', line)
            
            if train_loss_match and val_loss_match:
                train_loss = float(train_loss_match.group(1))
                val_loss = float(val_loss_match.group(1))
                val_acc = float(val_acc_match.group(1)) if val_acc_match else 0.0
                
                print(f"  訓練損失: {train_loss:.4f}")
                print(f"  驗證損失: {val_loss:.4f}")
                print(f"  驗證準確率: {val_acc:.4f} ({val_acc*100:.1f}%)")
                
                # 分析趨勢
                if train_loss < val_loss:
                    print("  ⚠️ 過擬合風險")
                elif val_loss > 2.0:
                    print("  ⚠️ 驗證損失較高")
                
                if val_acc > 0.8:
                    print("  ✅ 準確率良好")
                elif val_acc < 0.6:
                    print("  ⚠️ 準確率較低")
                
        except Exception as e:
            pass


def main():
    """主函數"""
    print("🔍 簡化訓練監控器")
    print("="*50)
    print("選擇監控模式:")
    print("1. 監控進程 (推薦)")
    print("2. 監控日誌文件")
    print("3. 同時監控")
    
    choice = input("\n請選擇 (1/2/3): ").strip()
    
    monitor = SimpleMonitor()
    
    if choice == '1':
        monitor.monitor_training_process()
    elif choice == '2':
        monitor.monitor_log_file()
    elif choice == '3':
        # 同時監控
        import threading
        
        def monitor_process():
            monitor.monitor_training_process()
        
        def monitor_log():
            monitor.monitor_log_file()
        
        thread1 = threading.Thread(target=monitor_process)
        thread2 = threading.Thread(target=monitor_log)
        
        thread1.daemon = True
        thread2.daemon = True
        
        thread1.start()
        thread2.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ 監控已停止")
    else:
        print("❌ 無效選擇")


if __name__ == "__main__":
    main() 