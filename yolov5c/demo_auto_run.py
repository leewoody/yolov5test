#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Auto Run Training System
演示自動運行訓練系統
"""

import os
import sys
import time
from pathlib import Path


def demo_auto_run_system():
    """演示自動運行系統"""
    
    print("🤖 Auto-Fix 自動訓練系統演示")
    print("="*60)
    
    # 檢查必要文件
    required_files = [
        "yolov5c/train_joint_advanced.py",
        "yolov5c/unified_validation.py", 
        "yolov5c/analyze_advanced_training.py",
        "yolov5c/auto_run_training.py",
        "yolov5c/quick_auto_run.py"
    ]
    
    print("📋 檢查必要文件...")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            return False
    
    # 檢查數據配置文件
    data_yaml_paths = [
        "converted_dataset_fixed/data.yaml",
        "regurgitation_converted/data.yaml", 
        "demo/data.yaml",
        "data.yaml"
    ]
    
    data_yaml = None
    for path in data_yaml_paths:
        if Path(path).exists():
            data_yaml = path
            break
    
    if not data_yaml:
        print("❌ 未找到數據配置文件")
        return False
    
    print(f"✅ 找到數據配置: {data_yaml}")
    
    print("\n🎯 自動運行系統功能:")
    print("1. 🔧 Auto-Fix 自動優化")
    print("   - 過擬合自動檢測與修正")
    print("   - 準確率下降自動調整")
    print("   - 動態學習率與正則化調整")
    
    print("\n2. 🚀 完整訓練流程")
    print("   - 高級訓練 (Auto-Fix 啟用)")
    print("   - 統一驗證 (檢測/分類)")
    print("   - 訓練分析與可視化")
    print("   - 結果整理與報告生成")
    
    print("\n3. 📊 智能監控")
    print("   - 實時訓練狀態監控")
    print("   - 自動問題檢測")
    print("   - 性能指標分析")
    
    print("\n🚀 使用方法:")
    print("\n方法1: 快速自動運行 (推薦)")
    print("```bash")
    print("python3 yolov5c/quick_auto_run.py")
    print("```")
    
    print("\n方法2: 完整自動運行")
    print("```bash")
    print("python3 yolov5c/auto_run_training.py \\")
    print("    --data converted_dataset_fixed/data.yaml \\")
    print("    --epochs 50 \\")
    print("    --batch-size 16")
    print("```")
    
    print("\n方法3: 手動分步執行")
    print("```bash")
    print("# 步驟1: 高級訓練")
    print("python3 yolov5c/train_joint_advanced.py \\")
    print("    --data converted_dataset_fixed/data.yaml \\")
    print("    --epochs 50")
    print("")
    print("# 步驟2: 統一驗證")
    print("python3 yolov5c/unified_validation.py \\")
    print("    --weights joint_model_advanced.pth \\")
    print("    --data converted_dataset_fixed/data.yaml")
    print("")
    print("# 步驟3: 訓練分析")
    print("python3 yolov5c/analyze_advanced_training.py")
    print("```")
    
    print("\n📈 預期結果:")
    print("- 自動生成的模型文件")
    print("- 完整的驗證指標")
    print("- 詳細的分析報告")
    print("- 可視化圖表")
    print("- 訓練總結報告")
    
    print("\n🎯 特色功能:")
    print("✅ 一鍵式自動化訓練")
    print("✅ 智能問題檢測與修正")
    print("✅ 完整的性能分析")
    print("✅ 自動結果整理")
    print("✅ 詳細的使用指南")
    
    return True


def main():
    """主函數"""
    print("🎬 開始演示...")
    
    if demo_auto_run_system():
        print("\n✅ 演示完成！")
        print("\n🚀 現在您可以運行:")
        print("python3 yolov5c/quick_auto_run.py")
    else:
        print("\n❌ 演示失敗，請檢查文件完整性")


if __name__ == "__main__":
    main() 