#!/usr/bin/env python3
"""
Auto Train Test Summary
自動訓練、測試並生成完整摘要報告
"""

import os
import sys
import subprocess
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoTrainTestSummary:
    def __init__(self, data_yaml_path, epochs=20, batch_size=16):
        """初始化自動訓練測試系統"""
        self.data_yaml_path = data_yaml_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"auto_train_test_{self.timestamp}"
        
        # 創建輸出目錄
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 設置日誌文件
        self.log_file = os.path.join(self.output_dir, "auto_train_test.log")
        self.setup_file_logging()
        
    def setup_file_logging(self):
        """設置文件日誌"""
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def run_command(self, cmd, description):
        """執行命令並記錄結果"""
        logger.info(f"🚀 {description}")
        logger.info(f"📝 執行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                logger.info(f"✅ {description} 成功完成")
                if result.stdout:
                    logger.info(f"📤 輸出: {result.stdout[:500]}...")
            else:
                logger.error(f"❌ {description} 失敗")
                logger.error(f"📤 錯誤輸出: {result.stderr}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"❌ {description} 執行異常: {e}")
            return False
    
    def step1_dataset_analysis(self):
        """步驟1: 數據集分析"""
        logger.info("=" * 60)
        logger.info("📊 步驟1: 數據集統計分析")
        logger.info("=" * 60)
        
        cmd = [
            "python3", "yolov5c/analyze_dataset_statistics.py",
            "--data", self.data_yaml_path,
            "--output", os.path.join(self.output_dir, "dataset_analysis")
        ]
        
        return self.run_command(cmd, "數據集統計分析")
    
    def step2_training(self):
        """步驟2: 訓練模型"""
        logger.info("=" * 60)
        logger.info("🏋️ 步驟2: 模型訓練")
        logger.info("=" * 60)
        
        # 使用修復版訓練腳本
        cmd = [
            "python3", "yolov5c/train_joint_fixed.py",
            "--data", self.data_yaml_path,
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--device", "auto",
            "--log", os.path.join(self.output_dir, "training.log")
        ]
        
        return self.run_command(cmd, "聯合訓練")
    
    def step3_validation(self):
        """步驟3: 模型驗證"""
        logger.info("=" * 60)
        logger.info("🔍 步驟3: 模型驗證")
        logger.info("=" * 60)
        
        # 查找最新的模型文件
        model_files = [
            "joint_model_fixed.pth",
            "best_simple_joint_model.pt",
            "joint_model_advanced.pth"
        ]
        
        model_path = None
        for model_file in model_files:
            if os.path.exists(model_file):
                model_path = model_file
                break
        
        if not model_path:
            logger.warning("⚠️ 未找到訓練好的模型文件")
            return False
        
        # 運行統一驗證
        cmd = [
            "python3", "yolov5c/unified_validation.py",
            "--model", model_path,
            "--data", self.data_yaml_path,
            "--output", os.path.join(self.output_dir, "validation_results")
        ]
        
        return self.run_command(cmd, "模型驗證")
    
    def step4_performance_analysis(self):
        """步驟4: 性能分析"""
        logger.info("=" * 60)
        logger.info("📈 步驟4: 性能分析")
        logger.info("=" * 60)
        
        # 分析訓練歷史
        history_files = [
            "training_history_fixed.json",
            "training_history_advanced.json"
        ]
        
        for history_file in history_files:
            if os.path.exists(history_file):
                cmd = [
                    "python3", "yolov5c/analyze_advanced_training.py",
                    "--history", history_file,
                    "--output", os.path.join(self.output_dir, f"analysis_{history_file.replace('.json', '')}")
                ]
                self.run_command(cmd, f"訓練歷史分析 ({history_file})")
        
        return True
    
    def step5_generate_summary_report(self):
        """步驟5: 生成摘要報告"""
        logger.info("=" * 60)
        logger.info("📋 步驟5: 生成摘要報告")
        logger.info("=" * 60)
        
        report_path = os.path.join(self.output_dir, "summary_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_summary_content())
        
        logger.info(f"✅ 摘要報告已生成: {report_path}")
        return True
    
    def generate_summary_content(self):
        """生成摘要報告內容"""
        content = []
        content.append("# 自動訓練測試摘要報告")
        content.append("=" * 60)
        content.append(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**數據配置**: {self.data_yaml_path}")
        content.append(f"**訓練輪數**: {self.epochs}")
        content.append(f"**批次大小**: {self.batch_size}")
        content.append("")
        
        # 數據集統計摘要
        content.append("## 📊 數據集統計摘要")
        content.append("-" * 40)
        
        dataset_stats_file = os.path.join(self.output_dir, "dataset_analysis", "dataset_statistics.json")
        if os.path.exists(dataset_stats_file):
            try:
                with open(dataset_stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                # 檢測類別統計
                detection_stats = stats['detection_stats']['total']
                total_detections = sum(detection_stats.values())
                content.append("### 檢測類別統計")
                for class_name, count in detection_stats.items():
                    percentage = (count / total_detections * 100) if total_detections > 0 else 0
                    content.append(f"- {class_name}: {count} ({percentage:.1f}%)")
                
                # 分類類別統計
                classification_stats = stats['classification_stats']['total']
                total_classifications = sum(classification_stats.values())
                content.append("\n### 分類類別統計")
                for class_name, count in classification_stats.items():
                    percentage = (count / total_classifications * 100) if total_classifications > 0 else 0
                    content.append(f"- {class_name}: {count} ({percentage:.1f}%)")
                    
            except Exception as e:
                content.append(f"⚠️ 無法載入數據集統計: {e}")
        
        # 訓練結果摘要
        content.append("\n## 🏋️ 訓練結果摘要")
        content.append("-" * 40)
        
        # 檢查訓練日誌
        training_log = os.path.join(self.output_dir, "training.log")
        if os.path.exists(training_log):
            try:
                with open(training_log, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()
                
                # 提取最後的訓練結果
                final_results = []
                for line in reversed(log_lines[-50:]):  # 檢查最後50行
                    if "Best Val Cls Accuracy" in line or "mAP" in line or "Precision" in line or "Recall" in line:
                        final_results.append(line.strip())
                        if len(final_results) >= 5:  # 最多顯示5行
                            break
                
                if final_results:
                    content.append("### 最終訓練指標")
                    for result in reversed(final_results):
                        content.append(f"- {result}")
                else:
                    content.append("⚠️ 未找到最終訓練指標")
                    
            except Exception as e:
                content.append(f"⚠️ 無法讀取訓練日誌: {e}")
        
        # 驗證結果摘要
        content.append("\n## 🔍 驗證結果摘要")
        content.append("-" * 40)
        
        validation_dir = os.path.join(self.output_dir, "validation_results")
        if os.path.exists(validation_dir):
            # 查找驗證報告
            for file in os.listdir(validation_dir):
                if file.endswith('.md') and 'report' in file.lower():
                    content.append(f"📄 詳細驗證報告: `{validation_dir}/{file}`")
                    break
        
        # 性能分析摘要
        content.append("\n## 📈 性能分析摘要")
        content.append("-" * 40)
        
        analysis_dirs = [d for d in os.listdir(self.output_dir) if d.startswith('analysis_')]
        for analysis_dir in analysis_dirs:
            analysis_path = os.path.join(self.output_dir, analysis_dir)
            report_files = [f for f in os.listdir(analysis_path) if f.endswith('.md')]
            for report_file in report_files:
                content.append(f"📊 性能分析報告: `{analysis_path}/{report_file}`")
        
        # 文件列表
        content.append("\n## 📁 生成文件列表")
        content.append("-" * 40)
        
        def list_files_recursive(directory, prefix=""):
            files = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    files.append(f"{prefix}📄 {item}")
                elif os.path.isdir(item_path):
                    files.append(f"{prefix}📁 {item}/")
                    files.extend(list_files_recursive(item_path, prefix + "  "))
            return files
        
        files = list_files_recursive(self.output_dir)
        for file in files:
            content.append(file)
        
        # 總結
        content.append("\n## 🎯 總結")
        content.append("-" * 40)
        content.append("✅ 自動訓練測試流程已完成")
        content.append(f"📁 所有結果保存在: `{self.output_dir}/`")
        content.append("📋 詳細報告請查看上述文件")
        
        return "\n".join(content)
    
    def run_full_pipeline(self):
        """運行完整流程"""
        logger.info("🚀 開始自動訓練測試流程")
        logger.info(f"📁 輸出目錄: {self.output_dir}")
        
        start_time = time.time()
        
        # 執行各個步驟
        steps = [
            ("數據集分析", self.step1_dataset_analysis),
            ("模型訓練", self.step2_training),
            ("模型驗證", self.step3_validation),
            ("性能分析", self.step4_performance_analysis),
            ("生成摘要", self.step5_generate_summary_report)
        ]
        
        results = {}
        for step_name, step_func in steps:
            logger.info(f"\n{'='*20} {step_name} {'='*20}")
            success = step_func()
            results[step_name] = success
            
            if not success:
                logger.warning(f"⚠️ {step_name} 步驟失敗，但繼續執行後續步驟")
        
        # 計算總時間
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        # 生成最終摘要
        logger.info("\n" + "="*60)
        logger.info("🎉 自動訓練測試流程完成")
        logger.info("="*60)
        logger.info(f"⏱️ 總耗時: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logger.info(f"📁 結果目錄: {self.output_dir}")
        
        # 顯示步驟結果
        logger.info("\n📋 步驟執行結果:")
        for step_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失敗"
            logger.info(f"  {step_name}: {status}")
        
        # 顯示摘要報告位置
        summary_report = os.path.join(self.output_dir, "summary_report.md")
        if os.path.exists(summary_report):
            logger.info(f"\n📄 摘要報告: {summary_report}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='自動訓練測試並生成摘要報告')
    parser.add_argument('--data', type=str, default='converted_dataset_fixed/data.yaml',
                       help='數據配置文件路徑')
    parser.add_argument('--epochs', type=int, default=20,
                       help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 檢查數據配置文件
    if not os.path.exists(args.data):
        logger.error(f"❌ 數據配置文件不存在: {args.data}")
        return
    
    # 創建自動訓練測試系統
    auto_system = AutoTrainTestSummary(args.data, args.epochs, args.batch_size)
    
    # 運行完整流程
    results = auto_system.run_full_pipeline()
    
    # 顯示摘要報告
    summary_report = os.path.join(auto_system.output_dir, "summary_report.md")
    if os.path.exists(summary_report):
        print("\n" + "="*80)
        print("📋 摘要報告預覽")
        print("="*80)
        
        with open(summary_report, 'r', encoding='utf-8') as f:
            content = f.read()
            # 顯示前30行
            lines = content.split('\n')[:30]
            print('\n'.join(lines))
            if len(content.split('\n')) > 30:
                print("...")
                print(f"📄 完整報告請查看: {summary_report}")

if __name__ == "__main__":
    main() 