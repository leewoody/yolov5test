#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Monitor
快速監控啟動器
"""

import subprocess
import sys
from pathlib import Path


def quick_monitor():
    """快速監控"""
    print("🔍 快速訓練監控")
    print("="*40)
    
    # 檢查是否有訓練進程
    training_processes = check_training_processes()
    
    if training_processes:
        print(f"✅ 發現 {len(training_processes)} 個訓練進程")
        print("🚀 啟動進程監控...")
        
        # 啟動進程監控
        try:
            subprocess.run([
                "python3", "yolov5c/simple_monitor.py"
            ], check=True)
        except KeyboardInterrupt:
            print("\n⏹️ 監控已停止")
        except subprocess.CalledProcessError as e:
            print(f"❌ 監控失敗: {e}")
    
    else:
        print("⚠️ 未發現訓練進程")
        print("🔍 檢查日誌文件...")
        
        # 檢查日誌文件
        log_files = find_log_files()
        
        if log_files:
            print(f"✅ 發現 {len(log_files)} 個日誌文件")
            print("📊 啟動日誌監控...")
            
            # 啟動日誌監控
            try:
                subprocess.run([
                    "python3", "yolov5c/simple_monitor.py"
                ], check=True)
            except KeyboardInterrupt:
                print("\n⏹️ 監控已停止")
            except subprocess.CalledProcessError as e:
                print(f"❌ 監控失敗: {e}")
        else:
            print("❌ 未發現訓練進程或日誌文件")
            print("\n💡 建議:")
            print("1. 先啟動訓練: python3 yolov5c/quick_auto_run.py")
            print("2. 然後運行監控: python3 yolov5c/quick_monitor.py")


def check_training_processes():
    """檢查訓練進程"""
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True
        )
        
        processes = []
        for line in result.stdout.split('\n'):
            if 'python' in line.lower() and any(keyword in line.lower() for keyword in ['train', 'yolov5']):
                parts = line.split()
                if len(parts) >= 2:
                    processes.append({
                        'pid': parts[1],
                        'cmd': ' '.join(parts[10:])
                    })
        
        return processes
        
    except Exception as e:
        print(f"❌ 檢查進程失敗: {e}")
        return []


def find_log_files():
    """查找日誌文件"""
    log_files = []
    for pattern in ['*.log', 'training.log', 'train.log', 'output.log']:
        for log_file in Path('.').glob(pattern):
            if log_file.is_file():
                log_files.append(log_file)
    
    return log_files


def main():
    """主函數"""
    print("🚀 快速監控啟動器")
    print("="*40)
    print("自動檢測並啟動最佳監控模式")
    print("="*40)
    
    quick_monitor()


if __name__ == "__main__":
    main() 