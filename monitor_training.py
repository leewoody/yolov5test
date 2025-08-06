#!/usr/bin/env python3
"""
訓練監控腳本
監控綜合優化訓練的進度
"""

import time
import os
import subprocess
from pathlib import Path

def check_training_progress():
    """檢查訓練進度"""
    print("🔍 檢查訓練進度...")
    
    # 檢查是否有訓練進程
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'train_joint_comprehensive_optimized' in result.stdout:
            print("✅ 訓練進程正在運行")
            return True
        else:
            print("❌ 未找到訓練進程")
            return False
    except:
        print("❌ 無法檢查進程")
        return False

def check_model_files():
    """檢查模型文件"""
    print("📁 檢查模型文件...")
    
    model_files = [
        'joint_model_comprehensive_optimized.pth',
        'joint_model_ar_optimized.pth',
        'joint_model_detection_optimized.pth'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"✅ {model_file}: {size:.1f} MB")
        else:
            print(f"❌ {model_file}: 不存在")

def check_logs():
    """檢查日誌文件"""
    print("📋 檢查日誌文件...")
    
    log_files = [
        'training.log',
        'optimization.log'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file) / 1024  # KB
            print(f"✅ {log_file}: {size:.1f} KB")
            
            # 顯示最後幾行
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"   最後更新: {lines[-1].strip()}")
            except:
                pass
        else:
            print(f"❌ {log_file}: 不存在")

def main():
    print("🚀 訓練監控開始...")
    print("=" * 50)
    
    while True:
        print(f"\n⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 30)
        
        # 檢查訓練進度
        is_running = check_training_progress()
        
        # 檢查模型文件
        check_model_files()
        
        # 檢查日誌
        check_logs()
        
        if not is_running:
            print("\n🎉 訓練可能已完成！")
            break
        
        print("\n⏳ 等待30秒後再次檢查...")
        time.sleep(30)

if __name__ == "__main__":
    main() 