#!/usr/bin/env python3
"""
自動轉換 Regurgitation 數據集並執行聯合訓練
保持 one-hot encoding 格式不變
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """執行命令並顯示進度"""
    print(f"\n{'='*50}")
    print(f"執行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} 成功完成")
        if result.stdout:
            print("輸出:", result.stdout[-500:])  # 顯示最後500字符
    else:
        print(f"❌ {description} 失敗")
        print("錯誤:", result.stderr)
        return False
    
    return True

def main():
    """主函數"""
    print("🚀 開始自動轉換和訓練流程")
    
    # 1. 轉換數據集
    input_dataset = "Regurgitation-YOLODataset-1new"
    output_dataset = "regurgitation_converted_onehot"
    
    if not os.path.exists(input_dataset):
        print(f"❌ 找不到輸入數據集: {input_dataset}")
        return False
    
    # 清理舊的轉換結果
    if os.path.exists(output_dataset):
        print(f"清理舊的轉換結果: {output_dataset}")
        shutil.rmtree(output_dataset)
    
    # 執行轉換
    convert_cmd = f"python3 yolov5c/convert_regurgitation_dataset.py \"{input_dataset}\" \"{output_dataset}\""
    if not run_command(convert_cmd, "轉換數據集"):
        return False
    
    # 2. 檢查轉換結果
    print("\n📊 檢查轉換結果:")
    for split in ['train', 'val', 'test']:
        labels_dir = f"{output_dataset}/{split}/labels"
        if os.path.exists(labels_dir):
            label_files = list(Path(labels_dir).glob("*.txt"))
            print(f"  {split}: {len(label_files)} 個標籤文件")
            
            # 檢查前幾個文件的格式
            if label_files:
                sample_file = label_files[0]
                with open(sample_file, 'r') as f:
                    lines = f.readlines()
                if len(lines) >= 1:
                    print(f"    範例格式: {lines[0].strip()}")
                else:
                    print(f"    範例格式: 空文件")
    
    # 3. 執行訓練
    print("\n🎯 開始聯合訓練")
    
    # 設置訓練參數
    data_yaml = f"{output_dataset}/data.yaml"
    epochs = 100
    batch_size = 16
    imgsz = 640
    
    train_cmd = f"python3 yolov5c/train_joint_optimized.py --data {data_yaml} --epochs {epochs} --batch-size {batch_size} --imgsz {imgsz}"
    
    if not run_command(train_cmd, "聯合訓練"):
        return False
    
    print("\n🎉 自動轉換和訓練流程完成！")
    print(f"📁 轉換後的數據集: {output_dataset}")
    print(f"📊 訓練結果保存在: runs/train/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 