#!/usr/bin/env python3
"""
簡化的訓練監控腳本
監控 GradNorm 和雙獨立骨幹網路的 50 epoch 訓練進度
"""

import os
import time
import subprocess
from datetime import datetime

def get_log_tail(log_file, lines=10):
    """獲取日誌文件的最後幾行"""
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()[-lines:]
        return []
    except:
        return []

def parse_gradnorm_progress(log_file):
    """解析 GradNorm 訓練進度"""
    lines = get_log_tail(log_file, 30)
    
    current_epoch = None
    current_batch = None
    latest_loss = None
    detection_weight = None
    classification_weight = None
    has_error = False
    
    for line in lines:
        line = line.strip()
        
        # 檢查錯誤
        if any(keyword in line.lower() for keyword in ['error', 'traceback', 'exception']):
            has_error = True
        
        # 解析進度
        if "Epoch" in line and "/" in line:
            try:
                epoch_part = line.split("Epoch ")[1].split("/")[0]
                current_epoch = int(epoch_part)
            except:
                pass
        
        if "Batch" in line and "Total Loss:" in line:
            try:
                parts = line.split(",")
                # 解析batch
                batch_part = [p for p in parts if "Batch" in p][0]
                current_batch = batch_part.split("/")[0].split()[-1]
                
                # 解析loss
                loss_part = [p for p in parts if "Total Loss:" in p][0]
                latest_loss = float(loss_part.split("Total Loss:")[1].strip())
                
                # 解析權重
                det_part = [p for p in parts if "Detection Weight:" in p][0]
                detection_weight = float(det_part.split("Detection Weight:")[1].strip())
                
                cls_part = [p for p in parts if "Classification Weight:" in p][0]
                classification_weight = float(cls_part.split("Classification Weight:")[1].strip())
            except:
                pass
    
    return {
        "epoch": current_epoch,
        "batch": current_batch,
        "loss": latest_loss,
        "detection_weight": detection_weight,
        "classification_weight": classification_weight,
        "has_error": has_error
    }

def parse_dual_backbone_progress(log_file):
    """解析雙獨立骨幹網路訓練進度"""
    lines = get_log_tail(log_file, 30)
    
    current_epoch = None
    status = "初始化中"
    has_error = False
    
    for line in lines:
        line = line.strip()
        
        # 檢查錯誤
        if any(keyword in line.lower() for keyword in ['error', 'traceback', 'exception']):
            has_error = True
        
        # 檢查狀態
        if "訓練樣本" in line or "training samples" in line:
            status = "數據加載完成"
        elif "Epoch" in line and "/" in line:
            status = "訓練進行中"
            try:
                epoch_part = line.split("Epoch ")[1].split("/")[0]
                current_epoch = int(epoch_part)
            except:
                pass
    
    return {
        "epoch": current_epoch,
        "status": status,
        "has_error": has_error
    }

def check_jobs():
    """檢查背景作業狀態"""
    try:
        result = subprocess.run(['jobs'], shell=True, capture_output=True, text=True)
        output = result.stdout
        
        gradnorm_running = "train_joint_gradnorm" in output
        dual_backbone_running = "train_dual_backbone" in output
        
        return gradnorm_running, dual_backbone_running
    except:
        return False, False

def monitor_training():
    """監控訓練狀態"""
    gradnorm_log = "gradnorm_complete_50.log"
    dual_backbone_log = "dual_backbone_complete_50.log"
    
    # 檢查作業狀態
    gradnorm_running, dual_backbone_running = check_jobs()
    
    # 解析進度
    gradnorm_progress = parse_gradnorm_progress(gradnorm_log)
    dual_backbone_progress = parse_dual_backbone_progress(dual_backbone_log)
    
    # 生成報告
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*70}")
    print(f"🔍 訓練監控報告 - {timestamp}")
    print(f"{'='*70}")
    
    # GradNorm 狀態
    status_icon = "🟢" if gradnorm_running else "🔴"
    print(f"GradNorm 訓練: {status_icon} {'運行中' if gradnorm_running else '已停止'}")
    
    if gradnorm_progress["epoch"]:
        print(f"  📈 進度: Epoch {gradnorm_progress['epoch']}/50, Batch {gradnorm_progress['batch']}")
        if gradnorm_progress["loss"]:
            print(f"  📉 總損失: {gradnorm_progress['loss']:.4f}")
        if gradnorm_progress["detection_weight"] and gradnorm_progress["classification_weight"]:
            print(f"  ⚖️  檢測權重: {gradnorm_progress['detection_weight']:.4f}")
            print(f"  ⚖️  分類權重: {gradnorm_progress['classification_weight']:.4f}")
            ratio = gradnorm_progress["detection_weight"] / gradnorm_progress["classification_weight"]
            print(f"  📐 權重比例: {ratio:.1f}:1")
    
    if gradnorm_progress["has_error"]:
        print(f"  ⚠️  檢測到錯誤")
    
    # 雙獨立骨幹網路狀態
    status_icon = "🟢" if dual_backbone_running else "🔴"
    print(f"雙獨立骨幹網路: {status_icon} {'運行中' if dual_backbone_running else '已停止'}")
    print(f"  📋 狀態: {dual_backbone_progress['status']}")
    
    if dual_backbone_progress["epoch"]:
        print(f"  📈 進度: Epoch {dual_backbone_progress['epoch']}/50")
    
    if dual_backbone_progress["has_error"]:
        print(f"  ⚠️  檢測到錯誤")
    
    # 完成狀態檢查
    gradnorm_completed = gradnorm_progress.get("epoch") == 50
    dual_backbone_completed = dual_backbone_progress.get("epoch") == 50
    
    if gradnorm_completed and dual_backbone_completed:
        print("\n🎉 兩個訓練都已完成 50 epochs!")
        return True
    elif not gradnorm_running and not dual_backbone_running:
        print("\n⚠️  兩個訓練都已停止")
        
    return False

def main():
    """主函數"""
    print("🚀 開始監控 GradNorm 和雙獨立骨幹網路訓練...")
    print("📋 每2分鐘檢查一次進度")
    print("💡 按 Ctrl+C 停止監控")
    
    try:
        while True:
            completed = monitor_training()
            if completed:
                break
            time.sleep(120)  # 每2分鐘檢查一次
    except KeyboardInterrupt:
        print("\n⏹️  監控已停止")

if __name__ == "__main__":
    main() 