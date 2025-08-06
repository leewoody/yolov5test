#!/usr/bin/env python3
"""
自動訓練監控和修復系統
確保 GradNorm 和雙獨立骨幹網路的 50 epoch 訓練成功完成
"""

import os
import time
import subprocess
import signal
import json
import logging
from datetime import datetime
from pathlib import Path
import threading
import psutil

class AutoTrainingMonitor:
    def __init__(self):
        self.setup_logging()
        self.gradnorm_cmd = [
            "python3", "train_joint_gradnorm_fixed.py",
            "--epochs", "50",
            "--batch-size", "8", 
            "--device", "cpu",
            "--gradnorm-alpha", "1.5",
            "--project", "runs/gradnorm_50_epochs",
            "--name", "complete_training"
        ]
        
        self.dual_backbone_cmd = [
            "python3", "train_dual_backbone_complete.py",
            "--data", "/Users/mac/_codes/_Github/yolov5test/converted_dataset_fixed/data.yaml",
            "--epochs", "50",
            "--batch-size", "8",
            "--device", "cpu", 
            "--project", "runs/dual_backbone_50_epochs",
            "--name", "complete_training",
            "--exist-ok"
        ]
        
        self.gradnorm_process = None
        self.dual_backbone_process = None
        self.gradnorm_log = "auto_gradnorm_50.log"
        self.dual_backbone_log = "auto_dual_backbone_50.log"
        self.monitor_log = "auto_monitor.log"
        
        self.max_restarts = 3
        self.gradnorm_restarts = 0
        self.dual_backbone_restarts = 0
        
    def setup_logging(self):
        """設置日誌系統"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('auto_training_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_gradnorm_training(self):
        """啟動 GradNorm 訓練"""
        try:
            self.logger.info("🚀 啟動 GradNorm 50 epoch 訓練...")
            with open(self.gradnorm_log, 'w') as f:
                self.gradnorm_process = subprocess.Popen(
                    self.gradnorm_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd()
                )
            self.logger.info(f"✅ GradNorm 訓練已啟動，PID: {self.gradnorm_process.pid}")
            return True
        except Exception as e:
            self.logger.error(f"❌ GradNorm 訓練啟動失敗: {e}")
            return False
    
    def start_dual_backbone_training(self):
        """啟動雙獨立骨幹網路訓練"""
        try:
            self.logger.info("🚀 啟動雙獨立骨幹網路 50 epoch 訓練...")
            with open(self.dual_backbone_log, 'w') as f:
                self.dual_backbone_process = subprocess.Popen(
                    self.dual_backbone_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd()
                )
            self.logger.info(f"✅ 雙獨立骨幹網路訓練已啟動，PID: {self.dual_backbone_process.pid}")
            return True
        except Exception as e:
            self.logger.error(f"❌ 雙獨立骨幹網路訓練啟動失敗: {e}")
            return False
    
    def check_process_health(self, process, name):
        """檢查進程健康狀態"""
        if process is None:
            return False, "進程未啟動"
        
        if process.poll() is not None:
            return False, f"進程已退出，退出碼: {process.returncode}"
        
        try:
            # 檢查進程是否還在運行
            psutil_process = psutil.Process(process.pid)
            if psutil_process.is_running():
                return True, "正常運行"
            else:
                return False, "進程不在運行"
        except psutil.NoSuchProcess:
            return False, "進程不存在"
        except Exception as e:
            return False, f"檢查進程時出錯: {e}"
    
    def get_log_tail(self, log_file, lines=20):
        """獲取日誌文件的最後幾行"""
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.readlines()[-lines:]
            return []
        except:
            return []
    
    def analyze_gradnorm_progress(self):
        """分析 GradNorm 訓練進度"""
        log_lines = self.get_log_tail(self.gradnorm_log, 50)
        
        current_epoch = None
        current_batch = None
        latest_loss = None
        has_error = False
        error_msg = ""
        
        for line in log_lines:
            line = line.strip()
            
            # 檢查是否有錯誤
            if any(error_keyword in line.lower() for error_keyword in ['error', 'traceback', 'exception', 'failed']):
                has_error = True
                error_msg = line
            
            # 解析進度
            if "Epoch" in line and "/" in line:
                try:
                    epoch_part = line.split("Epoch ")[1].split("/")[0]
                    current_epoch = int(epoch_part)
                except:
                    pass
            
            if "Batch" in line and "Total Loss:" in line:
                try:
                    current_batch = line.split("Batch ")[1].split("/")[0]
                    loss_part = line.split("Total Loss:")[1].split(",")[0]
                    latest_loss = float(loss_part.strip())
                except:
                    pass
        
        return {
            "epoch": current_epoch,
            "batch": current_batch,
            "loss": latest_loss,
            "has_error": has_error,
            "error_msg": error_msg,
            "recent_lines": log_lines[-5:] if log_lines else []
        }
    
    def analyze_dual_backbone_progress(self):
        """分析雙獨立骨幹網路訓練進度"""
        log_lines = self.get_log_tail(self.dual_backbone_log, 50)
        
        current_epoch = None
        has_error = False
        error_msg = ""
        status = "初始化中"
        
        for line in log_lines:
            line = line.strip()
            
            # 檢查是否有錯誤
            if any(error_keyword in line.lower() for error_keyword in ['error', 'traceback', 'exception', 'failed']):
                has_error = True
                error_msg = line
            
            # 檢查狀態
            if "訓練樣本" in line:
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
            "has_error": has_error,
            "error_msg": error_msg,
            "recent_lines": log_lines[-5:] if log_lines else []
        }
    
    def restart_gradnorm_if_needed(self, analysis):
        """如果需要則重啟 GradNorm 訓練"""
        if analysis["has_error"] and self.gradnorm_restarts < self.max_restarts:
            self.logger.warning(f"🔄 GradNorm 訓練出現錯誤，嘗試重啟 (第 {self.gradnorm_restarts + 1} 次)")
            self.logger.warning(f"錯誤信息: {analysis['error_msg']}")
            
            # 終止現有進程
            if self.gradnorm_process:
                try:
                    self.gradnorm_process.terminate()
                    self.gradnorm_process.wait(timeout=10)
                except:
                    self.gradnorm_process.kill()
                
            # 重新啟動
            time.sleep(5)
            if self.start_gradnorm_training():
                self.gradnorm_restarts += 1
                return True
        
        return False
    
    def restart_dual_backbone_if_needed(self, analysis):
        """如果需要則重啟雙獨立骨幹網路訓練"""
        if analysis["has_error"] and self.dual_backbone_restarts < self.max_restarts:
            self.logger.warning(f"🔄 雙獨立骨幹網路訓練出現錯誤，嘗試重啟 (第 {self.dual_backbone_restarts + 1} 次)")
            self.logger.warning(f"錯誤信息: {analysis['error_msg']}")
            
            # 終止現有進程
            if self.dual_backbone_process:
                try:
                    self.dual_backbone_process.terminate()
                    self.dual_backbone_process.wait(timeout=10)
                except:
                    self.dual_backbone_process.kill()
            
            # 重新啟動
            time.sleep(5)
            if self.start_dual_backbone_training():
                self.dual_backbone_restarts += 1
                return True
        
        return False
    
    def monitor_and_report(self):
        """監控並報告訓練狀態"""
        # 檢查進程健康狀態
        gradnorm_healthy, gradnorm_status = self.check_process_health(self.gradnorm_process, "GradNorm")
        dual_backbone_healthy, dual_backbone_status = self.check_process_health(self.dual_backbone_process, "DualBackbone")
        
        # 分析訓練進度
        gradnorm_analysis = self.analyze_gradnorm_progress()
        dual_backbone_analysis = self.analyze_dual_backbone_progress()
        
        # 生成報告
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📊 訓練監控報告 - {timestamp}")
        self.logger.info(f"{'='*80}")
        
        # GradNorm 狀態
        status_icon = "🟢" if gradnorm_healthy else "🔴"
        self.logger.info(f"GradNorm 訓練: {status_icon} {gradnorm_status}")
        if gradnorm_analysis["epoch"]:
            self.logger.info(f"  📈 進度: Epoch {gradnorm_analysis['epoch']}, Batch {gradnorm_analysis['batch']}")
            if gradnorm_analysis["loss"]:
                self.logger.info(f"  📉 損失: {gradnorm_analysis['loss']:.4f}")
        if gradnorm_analysis["has_error"]:
            self.logger.warning(f"  ⚠️  錯誤: {gradnorm_analysis['error_msg']}")
        
        # 雙獨立骨幹網路狀態
        status_icon = "🟢" if dual_backbone_healthy else "🔴"
        self.logger.info(f"雙獨立骨幹網路: {status_icon} {dual_backbone_status}")
        self.logger.info(f"  📋 狀態: {dual_backbone_analysis['status']}")
        if dual_backbone_analysis["epoch"]:
            self.logger.info(f"  📈 進度: Epoch {dual_backbone_analysis['epoch']}")
        if dual_backbone_analysis["has_error"]:
            self.logger.warning(f"  ⚠️  錯誤: {dual_backbone_analysis['error_msg']}")
        
        return gradnorm_healthy, dual_backbone_healthy, gradnorm_analysis, dual_backbone_analysis
    
    def run_monitoring_loop(self, check_interval=60):
        """運行監控循環"""
        self.logger.info("🔍 開始自動訓練監控系統...")
        
        # 啟動兩個訓練
        gradnorm_started = self.start_gradnorm_training()
        time.sleep(10)  # 稍等片刻再啟動第二個
        dual_backbone_started = self.start_dual_backbone_training()
        
        if not gradnorm_started and not dual_backbone_started:
            self.logger.error("❌ 兩個訓練都無法啟動，退出監控")
            return
        
        try:
            while True:
                # 監控並報告
                gradnorm_healthy, dual_backbone_healthy, gradnorm_analysis, dual_backbone_analysis = self.monitor_and_report()
                
                # 自動修復
                if not gradnorm_healthy:
                    self.restart_gradnorm_if_needed(gradnorm_analysis)
                
                if not dual_backbone_healthy:
                    self.restart_dual_backbone_if_needed(dual_backbone_analysis)
                
                # 檢查是否完成
                gradnorm_completed = gradnorm_analysis.get("epoch") == 50
                dual_backbone_completed = dual_backbone_analysis.get("epoch") == 50
                
                if gradnorm_completed and dual_backbone_completed:
                    self.logger.info("🎉 兩個訓練都已完成 50 epochs！")
                    break
                elif not gradnorm_healthy and not dual_backbone_healthy:
                    if self.gradnorm_restarts >= self.max_restarts and self.dual_backbone_restarts >= self.max_restarts:
                        self.logger.error("❌ 兩個訓練都達到最大重啟次數，停止監控")
                        break
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("⏹️  監控被手動停止")
        finally:
            # 清理進程
            if self.gradnorm_process:
                try:
                    self.gradnorm_process.terminate()
                except:
                    pass
            if self.dual_backbone_process:
                try:
                    self.dual_backbone_process.terminate()
                except:
                    pass

def main():
    monitor = AutoTrainingMonitor()
    monitor.run_monitoring_loop(check_interval=120)  # 每2分鐘檢查一次

if __name__ == "__main__":
    main() 