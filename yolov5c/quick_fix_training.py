#!/usr/bin/env python3
"""
🚀 快速修復版訓練啟動器
基於分析結果自動運行修復版訓練
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def find_data_yaml():
    """自動尋找數據配置文件"""
    possible_paths = [
        "converted_dataset_fixed/data.yaml",
        "data.yaml",
        "dataset/data.yaml",
        "yolov5/data/coco.yaml"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def run_fixed_training(data_yaml, epochs=50, batch_size=16, device='auto'):
    """運行修復版訓練"""
    print("🚀 啟動修復版訓練...")
    print(f"📁 數據配置: {data_yaml}")
    print(f"🔄 訓練輪數: {epochs}")
    print(f"📦 批次大小: {batch_size}")
    print(f"💻 設備: {device}")
    
    # 構建命令
    cmd = [
        "python3", "yolov5c/train_joint_fixed.py",
        "--data", data_yaml,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", device,
        "--log", "training_fixed_detailed.log"
    ]
    
    print(f"🔧 執行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # 執行訓練
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 修復版訓練完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 訓練失敗: {e}")
        print(f"錯誤輸出: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='快速修復版訓練啟動器')
    parser.add_argument('--data', type=str, help='數據配置文件路徑 (自動尋找如果未指定)')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='auto', help='設備')
    
    args = parser.parse_args()
    
    print("🔧 快速修復版訓練啟動器")
    print("=" * 50)
    
    # 尋找數據配置文件
    if args.data:
        data_yaml = args.data
    else:
        data_yaml = find_data_yaml()
    
    if not data_yaml:
        print("❌ 找不到數據配置文件!")
        print("請指定 --data 參數或確保以下文件存在:")
        print("  - converted_dataset_fixed/data.yaml")
        print("  - data.yaml")
        print("  - dataset/data.yaml")
        return
    
    if not os.path.exists(data_yaml):
        print(f"❌ 數據配置文件不存在: {data_yaml}")
        return
    
    print(f"✅ 找到數據配置文件: {data_yaml}")
    
    # 檢查修復版訓練腳本
    if not os.path.exists("yolov5c/train_joint_fixed.py"):
        print("❌ 修復版訓練腳本不存在: yolov5c/train_joint_fixed.py")
        return
    
    print("✅ 修復版訓練腳本已就緒")
    
    # 確認運行
    print("\n🎯 修復版訓練配置:")
    print(f"  - 數據配置: {data_yaml}")
    print(f"  - 訓練輪數: {args.epochs}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 設備: {args.device}")
    print(f"  - 日誌文件: training_fixed_detailed.log")
    
    confirm = input("\n是否開始修復版訓練? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 訓練已取消")
        return
    
    # 運行訓練
    success = run_fixed_training(
        data_yaml, 
        args.epochs, 
        args.batch_size, 
        args.device
    )
    
    if success:
        print("\n🎉 修復版訓練完成!")
        print("📁 生成的文件:")
        print("  - joint_model_fixed.pth (修復版模型)")
        print("  - training_history_fixed.json (訓練歷史)")
        print("  - training_fixed_detailed.log (詳細日誌)")
        
        print("\n🔧 修復措施已應用:")
        print("  - Focal Loss 處理類別不平衡")
        print("  - 適中分類損失權重 (8.0)")
        print("  - 降低學習率 (0.0008)")
        print("  - 增加 Weight Decay (5e-4)")
        print("  - 增加 Dropout (0.5)")
        print("  - 梯度裁剪")
        print("  - 學習率調度")
        print("  - 智能早停機制")
        
        print("\n📊 下一步:")
        print("  1. 查看訓練日誌: cat training_fixed_detailed.log")
        print("  2. 分析訓練歷史: python3 yolov5c/analyze_and_fix_with_logs.py --history training_history_fixed.json")
        print("  3. 使用修復版模型進行推理")
    else:
        print("\n❌ 修復版訓練失敗，請檢查錯誤信息")

if __name__ == "__main__":
    main() 