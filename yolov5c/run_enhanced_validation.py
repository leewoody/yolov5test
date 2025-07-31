#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Validation Runner Script
簡化運行增強驗證的腳本
"""

import os
import sys
from pathlib import Path
import argparse

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from enhanced_validation import EnhancedValidator


def run_validation_for_dataset(weights_path, data_yaml, dataset_name, save_dir=None):
    """
    為指定數據集運行增強驗證
    
    Args:
        weights_path: 模型權重文件路徑
        data_yaml: 數據集配置文件路徑
        dataset_name: 數據集名稱（用於保存結果）
        save_dir: 保存目錄（可選）
    """
    if save_dir is None:
        save_dir = f"runs/enhanced_val_{dataset_name}"
    
    print(f"🚀 開始為 {dataset_name} 數據集運行增強驗證...")
    print(f"📁 模型權重: {weights_path}")
    print(f"📁 數據集配置: {data_yaml}")
    print(f"📁 結果保存: {save_dir}")
    print("-" * 60)
    
    try:
        # 創建驗證器
        validator = EnhancedValidator(
            weights=weights_path,
            data=data_yaml,
            imgsz=640,
            batch_size=16,  # 較小的batch size以節省記憶體
            conf_thres=0.001,
            iou_thres=0.6,
            device='',  # 自動選擇設備
            save_dir=save_dir
        )
        
        # 運行驗證
        validator.validate()
        
        print(f"✅ {dataset_name} 數據集驗證完成！")
        print(f"📊 詳細報告已保存到: {save_dir}")
        
    except Exception as e:
        print(f"❌ 驗證過程中發生錯誤: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='運行增強驗證')
    parser.add_argument('--weights', type=str, default='best.pt', 
                       help='模型權重文件路徑')
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='數據集配置文件路徑')
    parser.add_argument('--dataset-name', type=str, default='dataset',
                       help='數據集名稱')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='保存目錄（可選）')
    
    args = parser.parse_args()
    
    # 檢查文件是否存在
    if not os.path.exists(args.weights):
        print(f"❌ 錯誤: 模型權重文件不存在: {args.weights}")
        return
    
    if not os.path.exists(args.data):
        print(f"❌ 錯誤: 數據集配置文件不存在: {args.data}")
        return
    
    # 運行驗證
    run_validation_for_dataset(
        weights_path=args.weights,
        data_yaml=args.data,
        dataset_name=args.dataset_name,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main() 