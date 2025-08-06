#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Auto Run Training
快速自動運行訓練系統
"""

import sys
import subprocess
from pathlib import Path


def find_data_yaml():
    """自動查找 data.yaml 文件"""
    possible_paths = [
        "data.yaml",
        "converted_dataset_fixed/data.yaml",
        "regurgitation_converted/data.yaml",
        "regurgitation_converted_onehot/data.yaml",
        "demo/data.yaml",
        "demo2/data.yaml"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    return None


def main():
    """主函數"""
    print("🚀 快速自動訓練系統")
    print("="*50)
    
    # 自動查找數據配置文件
    data_yaml = find_data_yaml()
    
    if not data_yaml:
        print("❌ 未找到 data.yaml 文件")
        print("請確保以下路徑之一存在 data.yaml 文件:")
        print("  - data.yaml")
        print("  - converted_dataset_fixed/data.yaml")
        print("  - regurgitation_converted/data.yaml")
        print("  - demo/data.yaml")
        return
    
    print(f"✅ 找到數據配置文件: {data_yaml}")
    
    # 構建自動運行命令
    auto_run_cmd = [
        "python3", "yolov5c/auto_run_training.py",
        "--data", data_yaml,
        "--epochs", "50",
        "--batch-size", "16",
        "--device", "auto"
    ]
    
    # 詢問是否啟用早停
    early_stop_response = input(f"\n是否啟用早停機制? (y/N): ").strip().lower()
    if early_stop_response in ['y', 'yes']:
        auto_run_cmd.append("--enable-early-stop")
        print("✅ 早停機制已啟用")
    else:
        print("✅ 早停機制已禁用")
    
    print(f"\n🎯 開始自動訓練流程...")
    print(f"命令: {' '.join(auto_run_cmd)}")
    print(f"這將執行:")
    print(f"  1. 高級訓練 (Auto-Fix 啟用)")
    print(f"  2. 統一驗證")
    print(f"  3. 訓練分析")
    print(f"  4. 結果整理")
    print(f"  5. 生成總結報告")
    
    # 確認執行
    response = input(f"\n是否繼續執行? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 取消執行")
        return
    
    try:
        # 執行自動訓練系統
        result = subprocess.run(auto_run_cmd, check=True)
        print("\n✅ 自動訓練系統執行完成！")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 執行失敗: {e}")
        return
    except KeyboardInterrupt:
        print("\n❌ 用戶中斷執行")
        return


if __name__ == "__main__":
    main() 