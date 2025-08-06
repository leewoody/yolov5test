#!/usr/bin/env python3
"""
訓練進度監控腳本
監控GradNorm和雙獨立骨幹網路的50 epoch訓練進度
"""

import os
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path

class TrainingMonitor:
    def __init__(self):
        self.gradnorm_log = "gradnorm_50_training.log"
        self.dual_backbone_log = "dual_backbone_50_training_fixed.log"
        self.monitor_log = "training_monitor.log"
        
    def check_process_status(self):
        """檢查訓練進程狀態"""
        try:
            # 檢查背景作業
            result = subprocess.run(['jobs'], shell=True, capture_output=True, text=True)
            jobs_output = result.stdout
            
            gradnorm_running = "train_joint_gradnorm" in jobs_output
            dual_backbone_running = "train_dual_backbone" in jobs_output
            
            return gradnorm_running, dual_backbone_running
        except:
            return False, False
    
    def get_log_tail(self, log_file, lines=10):
        """獲取日誌文件的最後幾行"""
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.readlines()[-lines:]
            return []
        except:
            return []
    
    def parse_gradnorm_progress(self):
        """解析GradNorm訓練進度"""
        log_lines = self.get_log_tail(self.gradnorm_log, 50)
        
        current_epoch = None
        current_batch = None
        latest_loss = None
        detection_weight = None
        classification_weight = None
        
        for line in log_lines:
            if "Epoch" in line and "/" in line:
                try:
                    # Extract epoch number: "Epoch 1/50"
                    epoch_part = line.split("Epoch ")[1].split("/")[0]
                    current_epoch = int(epoch_part)
                except:
                    pass
            
            if "Batch" in line and "Total Loss:" in line:
                try:
                    # Extract batch and loss info
                    parts = line.split(",")
                    batch_part = [p for p in parts if "Batch" in p][0]
                    current_batch = batch_part.split()[1].split("/")[0]
                    
                    loss_part = [p for p in parts if "Total Loss:" in p][0]
                    latest_loss = float(loss_part.split("Total Loss:")[1].strip())
                    
                    det_weight_part = [p for p in parts if "Detection Weight:" in p][0]
                    detection_weight = float(det_weight_part.split("Detection Weight:")[1].strip())
                    
                    cls_weight_part = [p for p in parts if "Classification Weight:" in p][0]
                    classification_weight = float(cls_weight_part.split("Classification Weight:")[1].strip())
                except:
                    pass
        
        return {
            "epoch": current_epoch,
            "batch": current_batch,
            "total_loss": latest_loss,
            "detection_weight": detection_weight,
            "classification_weight": classification_weight
        }
    
    def parse_dual_backbone_progress(self):
        """解析雙獨立骨幹網路訓練進度"""
        log_lines = self.get_log_tail(self.dual_backbone_log, 50)
        
        status = "初始化中"
        if any("訓練樣本" in line for line in log_lines):
            status = "數據加載完成"
        if any("Epoch" in line for line in log_lines):
            status = "訓練進行中"
        
        return {
            "status": status,
            "recent_activity": log_lines[-3:] if log_lines else []
        }
    
    def generate_status_report(self):
        """生成狀態報告"""
        gradnorm_running, dual_backbone_running = self.check_process_status()
        gradnorm_progress = self.parse_gradnorm_progress()
        dual_backbone_progress = self.parse_dual_backbone_progress()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            "timestamp": timestamp,
            "gradnorm": {
                "running": gradnorm_running,
                "progress": gradnorm_progress
            },
            "dual_backbone": {
                "running": dual_backbone_running,
                "progress": dual_backbone_progress
            }
        }
        
        return report
    
    def log_status(self, report):
        """記錄狀態到日誌文件"""
        with open(self.monitor_log, 'a', encoding='utf-8') as f:
            f.write(f"\n=== {report['timestamp']} ===\n")
            
            # GradNorm狀態
            grad_status = "🟢 運行中" if report['gradnorm']['running'] else "🔴 已停止"
            f.write(f"GradNorm訓練: {grad_status}\n")
            
            if report['gradnorm']['progress']['epoch']:
                f.write(f"  當前進度: Epoch {report['gradnorm']['progress']['epoch']}, ")
                f.write(f"Batch {report['gradnorm']['progress']['batch']}\n")
                f.write(f"  總損失: {report['gradnorm']['progress']['total_loss']:.4f}\n")
                f.write(f"  檢測權重: {report['gradnorm']['progress']['detection_weight']:.4f}\n")
                f.write(f"  分類權重: {report['gradnorm']['progress']['classification_weight']:.4f}\n")
            
            # 雙獨立骨幹網路狀態
            dual_status = "🟢 運行中" if report['dual_backbone']['running'] else "🔴 已停止"
            f.write(f"雙獨立骨幹網路: {dual_status}\n")
            f.write(f"  狀態: {report['dual_backbone']['progress']['status']}\n")
    
    def print_status(self, report):
        """打印狀態到控制台"""
        print(f"\n{'='*60}")
        print(f"🔍 訓練狀態監控 - {report['timestamp']}")
        print(f"{'='*60}")
        
        # GradNorm狀態
        grad_status = "🟢 運行中" if report['gradnorm']['running'] else "🔴 已停止"
        print(f"GradNorm訓練: {grad_status}")
        
        if report['gradnorm']['progress']['epoch']:
            print(f"  📊 當前進度: Epoch {report['gradnorm']['progress']['epoch']}, Batch {report['gradnorm']['progress']['batch']}")
            print(f"  📉 總損失: {report['gradnorm']['progress']['total_loss']:.4f}")
            print(f"  ⚖️  檢測權重: {report['gradnorm']['progress']['detection_weight']:.4f}")
            print(f"  ⚖️  分類權重: {report['gradnorm']['progress']['classification_weight']:.4f}")
            
            # 計算權重比例
            if report['gradnorm']['progress']['detection_weight'] and report['gradnorm']['progress']['classification_weight']:
                ratio = report['gradnorm']['progress']['detection_weight'] / report['gradnorm']['progress']['classification_weight']
                print(f"  📐 權重比例: {ratio:.1f}:1 (Detection:Classification)")
        
        # 雙獨立骨幹網路狀態
        dual_status = "🟢 運行中" if report['dual_backbone']['running'] else "🔴 已停止"
        print(f"雙獨立骨幹網路: {dual_status}")
        print(f"  📋 狀態: {report['dual_backbone']['progress']['status']}")
        
    def monitor_loop(self, interval=60):
        """持續監控循環"""
        print("🚀 開始監控訓練進度...")
        print(f"⏰ 監控間隔: {interval}秒")
        
        try:
            while True:
                report = self.generate_status_report()
                self.print_status(report)
                self.log_status(report)
                
                # 檢查是否兩個訓練都已停止
                if not report['gradnorm']['running'] and not report['dual_backbone']['running']:
                    print("\n🏁 所有訓練已完成，停止監控")
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  監控已手動停止")

def main():
    monitor = TrainingMonitor()
    
    # 生成一次性狀態報告
    report = monitor.generate_status_report()
    monitor.print_status(report)
    monitor.log_status(report)
    
    # 詢問是否開始持續監控
    print("\n是否開始持續監控? (y/n): ", end="")
    
    # 對於腳本執行，默認不開始持續監控
    # 只生成當前狀態報告
    return report

if __name__ == "__main__":
    main() 