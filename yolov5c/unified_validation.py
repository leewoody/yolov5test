#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified YOLOv5 Validation Script
統一驗證腳本，自動判斷檢測或分類任務並調用相應驗證功能
"""

import argparse
import os
import sys
from pathlib import Path
import torch

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from enhanced_validation import EnhancedValidator
from enhanced_classification_validation import EnhancedClassificationValidator


def detect_task_type(weights_path):
    """
    自動檢測模型類型（檢測或分類）
    
    Args:
        weights_path: 模型權重文件路徑
    
    Returns:
        str: 'detection' 或 'classification'
    """
    try:
        # 嘗試加載模型
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(weights_path, device='cpu')
        
        # 檢查模型類型
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            # 檢查最後一層的輸出
            last_layer = list(model.model.model.children())[-1]
            if hasattr(last_layer, 'nc'):
                # 如果有nc屬性，通常是檢測模型
                return 'detection'
            else:
                # 否則可能是分類模型
                return 'classification'
        else:
            # 默認認為是檢測模型
            return 'detection'
            
    except Exception as e:
        print(f"⚠️ 無法自動檢測模型類型: {e}")
        print("🔍 請手動指定任務類型")
        return None


def run_unified_validation(weights_path, data_path, task_type=None, **kwargs):
    """
    運行統一驗證
    
    Args:
        weights_path: 模型權重文件路徑
        data_path: 數據集配置文件路徑
        task_type: 任務類型 ('detection' 或 'classification')
        **kwargs: 其他參數
    """
    
    # 自動檢測任務類型
    if task_type is None:
        print("🔍 自動檢測模型類型...")
        task_type = detect_task_type(weights_path)
    
    if task_type is None:
        print("❌ 無法確定任務類型，請手動指定 --task-type")
        return
    
    print(f"🎯 檢測到任務類型: {task_type}")
    
    # 根據任務類型選擇驗證器
    if task_type == 'detection':
        print("🚀 啟動檢測驗證...")
        validator = EnhancedValidator(
            weights=weights_path,
            data=data_path,
            **kwargs
        )
        validator.validate()
        
    elif task_type == 'classification':
        print("🚀 啟動分類驗證...")
        validator = EnhancedClassificationValidator(
            weights=weights_path,
            data=data_path,
            **kwargs
        )
        validator.validate()
        
    else:
        print(f"❌ 不支持的任務類型: {task_type}")
        return
    
    print(f"✅ {task_type} 驗證完成！")


def main():
    parser = argparse.ArgumentParser(description='統一YOLOv5驗證腳本')
    parser.add_argument('--weights', type=str, required=True, help='模型權重文件路徑')
    parser.add_argument('--data', type=str, required=True, help='數據集配置文件路徑')
    parser.add_argument('--task-type', type=str, choices=['detection', 'classification'], 
                       help='任務類型（自動檢測）')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, 
                       help='推理尺寸')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='置信度閾值')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU閾值')
    parser.add_argument('--device', default='', help='設備')
    parser.add_argument('--save-dir', type=str, default='runs/unified_val', help='保存目錄')
    
    args = parser.parse_args()
    
    # 檢查文件是否存在
    if not os.path.exists(args.weights):
        print(f"❌ 錯誤: 模型權重文件不存在: {args.weights}")
        return
    
    if not os.path.exists(args.data):
        print(f"❌ 錯誤: 數據集配置文件不存在: {args.data}")
        return
    
    # 運行統一驗證
    run_unified_validation(
        weights_path=args.weights,
        data_path=args.data,
        task_type=args.task_type,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main() 