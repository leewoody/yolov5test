#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Run Training System
自動運行完整的訓練、驗證和分析流程
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import yaml


class AutoTrainingSystem:
    def __init__(self, data_yaml, epochs=50, batch_size=16, device='auto', enable_early_stop=False):
        self.data_yaml = Path(data_yaml)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.enable_early_stop = enable_early_stop
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"auto_training_run_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        # 檢查數據配置
        self.check_data_config()
        
    def check_data_config(self):
        """檢查數據配置文件"""
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"數據配置文件不存在: {self.data_yaml}")
        
        try:
            with open(self.data_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            # 檢測任務類別 (反流類型)
            self.detection_classes = config.get('names', [])
            self.num_detection_classes = len(self.detection_classes)
            
            # 分類任務類別 (視角類型)
            self.classification_classes = ['PSAX', 'PLAX', 'A4C']
            self.num_classification_classes = len(self.classification_classes)
            
            print(f"✅ 多任務數據配置檢查通過:")
            print(f"   - 檢測任務類別: {self.detection_classes} (反流類型)")
            print(f"   - 檢測類別數量: {self.num_detection_classes}")
            print(f"   - 分類任務類別: {self.classification_classes} (視角類型)")
            print(f"   - 分類類別數量: {self.num_classification_classes}")
            
        except Exception as e:
            raise ValueError(f"數據配置文件格式錯誤: {e}")
    
    def run_command(self, command, description, check=True):
        """運行命令並處理結果"""
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"{'='*60}")
        print(f"執行命令: {command}")
        print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=check,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ {description} 完成")
            print(f"執行時間: {duration:.2f} 秒")
            
            if result.stdout:
                print("輸出:")
                print(result.stdout)
            
            return result
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} 失敗")
            print(f"錯誤代碼: {e.returncode}")
            print(f"錯誤輸出: {e.stderr}")
            
            if check:
                raise e
            return e
    
    def step1_training(self):
        """步驟1: 執行多任務聯合訓練"""
        print("\n" + "🎯 步驟 1: 開始多任務聯合訓練")
        
        # 構建訓練命令 - 使用終極訓練腳本
        train_cmd = f"python3 yolov5c/train_joint_ultimate.py " \
                   f"--data {self.data_yaml} " \
                   f"--epochs {self.epochs} " \
                   f"--batch-size {self.batch_size} " \
                   f"--device {self.device} " \
                   f"--bbox-weight 10.0 " \
                   f"--cls-weight 8.0"
        
        # 添加早停選項
        if self.enable_early_stop:
            train_cmd += " --enable-earlystop"
        
        # 執行訓練
        result = self.run_command(train_cmd, "終極聯合訓練 (多任務學習)")
        
        # 檢查訓練結果
        if result.returncode == 0:
            print("✅ 訓練成功完成")
            
            # 檢查生成的模型文件
            model_files = ['joint_model_ultimate.pth', 'training_history_ultimate.json']
            for model_file in model_files:
                if Path(model_file).exists():
                    print(f"✅ 生成文件: {model_file}")
                else:
                    print(f"⚠️ 警告: 未找到 {model_file}")
        else:
            print("❌ 訓練失敗")
            return False
        
        return True
    
    def step2_validation(self):
        """步驟2: 執行多任務模型驗證"""
        print("\n" + "🎯 步驟 2: 開始多任務模型驗證")
        
        # 檢查模型文件
        model_path = "joint_model_ultimate.pth"
        if not Path(model_path).exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        # 構建驗證命令 - 使用終極測試腳本
        val_cmd = f"python3 yolov5c/test_ultimate_model.py " \
                 f"--data {self.data_yaml} " \
                 f"--weights {model_path} " \
                 f"--output-dir validation_results"
        
        # 執行驗證
        result = self.run_command(val_cmd, "終極模型驗證", check=False)
        
        if result.returncode == 0:
            print("✅ 驗證成功完成")
        else:
            print("⚠️ 驗證過程中出現問題，但繼續執行")
        
        return True
    
    def step3_analysis(self):
        """步驟3: 執行多任務性能分析"""
        print("\n" + "🎯 步驟 3: 開始多任務性能分析")
        
        # 檢查訓練歷史文件
        history_file = "training_history_ultimate.json"
        if not Path(history_file).exists():
            print(f"❌ 訓練歷史文件不存在: {history_file}")
            return False
        
        # 構建分析命令 - 生成性能報告
        analysis_cmd = f"python3 yolov5c/generate_performance_report.py " \
                      f"--data {self.data_yaml} " \
                      f"--history {history_file} " \
                      f"--output-dir performance_analysis"
        
        # 執行分析
        result = self.run_command(analysis_cmd, "多任務性能分析", check=False)
        
        if result.returncode == 0:
            print("✅ 分析成功完成")
        else:
            print("⚠️ 分析過程中出現問題")
        
        return True
    
    def step4_organize_results(self):
        """步驟4: 整理結果文件"""
        print("\n" + "🎯 步驟 4: 整理結果文件")
        
        # 要移動的文件列表
        files_to_move = [
            "joint_model_ultimate.pth",
            "training_history_ultimate.json",
            "performance_analysis/",
            "validation_results/"
        ]
        
        # 移動文件到輸出目錄
        for file_path in files_to_move:
            src_path = Path(file_path)
            if src_path.exists():
                if src_path.is_dir():
                    # 移動目錄
                    dst_path = self.output_dir / src_path.name
                    if dst_path.exists():
                        import shutil
                        shutil.rmtree(dst_path)
                    src_path.rename(dst_path)
                    print(f"✅ 移動目錄: {file_path} -> {dst_path}")
                else:
                    # 移動文件
                    dst_path = self.output_dir / file_path
                    src_path.rename(dst_path)
                    print(f"✅ 移動文件: {file_path} -> {dst_path}")
            else:
                print(f"⚠️ 文件不存在: {file_path}")
        
        return True
    
    def step5_generate_summary(self):
        """步驟5: 生成總結報告"""
        print("\n" + "🎯 步驟 5: 生成總結報告")
        
        summary = f"""
# 🤖 自動訓練系統總結報告

## 📋 訓練配置
- **數據配置文件**: {self.data_yaml}
- **訓練輪數**: {self.epochs}
- **批次大小**: {self.batch_size}
- **設備**: {self.device}
- **檢測任務類別**: {self.detection_classes} (反流類型)
- **檢測類別數量**: {self.num_detection_classes}
- **分類任務類別**: {self.classification_classes} (視角類型)
- **分類類別數量**: {self.num_classification_classes}

## 📊 生成文件
- **模型文件**: joint_model_advanced.pth
- **訓練歷史**: training_history_advanced.json
- **分析圖表**: advanced_training_analysis.png
- **分析報告**: advanced_training_analysis_report.md
- **驗證結果**: validation_results/

## 🎯 使用建議

### 1. 模型部署
```bash
# 使用訓練好的模型進行推理
python yolov5c/unified_validation.py \\
    --weights {self.output_dir}/joint_model_advanced.pth \\
    --data {self.data_yaml} \\
    --source your_test_images/
```

### 2. 進一步分析
```bash
# 查看詳細分析報告
cat {self.output_dir}/advanced_training_analysis_report.md

# 查看驗證結果
ls {self.output_dir}/validation_results/
```

### 3. 模型優化
如果性能不滿意，可以：
- 調整 Auto-Fix 參數
- 增加訓練輪數
- 調整批次大小
- 使用更大的模型架構

## 📈 性能指標
請查看 `advanced_training_analysis_report.md` 獲取詳細的性能指標。

---
*自動生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*輸出目錄: {self.output_dir}*
"""
        
        # 保存總結報告
        summary_file = self.output_dir / "training_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"✅ 總結報告已保存到: {summary_file}")
        
        return True
    
    def run_complete_pipeline(self):
        """運行完整的訓練流程"""
        print("🤖 自動訓練系統啟動")
        print(f"📁 輸出目錄: {self.output_dir}")
        print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 記錄開始時間
        start_time = time.time()
        
        try:
            # 步驟1: 訓練
            if not self.step1_training():
                print("❌ 訓練失敗，停止執行")
                return False
            
            # 步驟2: 驗證
            self.step2_validation()
            
            # 步驟3: 分析
            self.step3_analysis()
            
            # 步驟4: 整理結果
            self.step4_organize_results()
            
            # 步驟5: 生成總結
            self.step5_generate_summary()
            
            # 計算總時間
            end_time = time.time()
            total_duration = end_time - start_time
            
            print("\n" + "="*60)
            print("🎉 自動訓練系統執行完成！")
            print("="*60)
            print(f"總執行時間: {total_duration:.2f} 秒 ({total_duration/60:.1f} 分鐘)")
            print(f"輸出目錄: {self.output_dir}")
            print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("\n📋 生成的文件:")
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    print(f"  - {file_path.relative_to(self.output_dir)}")
            
            print(f"\n🚀 下一步:")
            print(f"  1. 查看總結報告: cat {self.output_dir}/training_summary.md")
            print(f"  2. 查看詳細分析: cat {self.output_dir}/advanced_training_analysis_report.md")
            print(f"  3. 使用模型進行推理")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 執行過程中出現錯誤: {e}")
            return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='自動訓練系統')
    parser.add_argument('--data', type=str, required=True, help='數據配置文件路徑 (data.yaml)')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數 (默認: 50)')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小 (默認: 16)')
    parser.add_argument('--device', type=str, default='auto', help='設備 (auto, cuda, cpu)')
    parser.add_argument('--enable-early-stop', action='store_true', help='啟用早停機制')
    
    args = parser.parse_args()
    
    try:
        # 創建自動訓練系統
        auto_system = AutoTrainingSystem(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            enable_early_stop=args.enable_early_stop
        )
        
        # 運行完整流程
        success = auto_system.run_complete_pipeline()
        
        if success:
            print("\n✅ 自動訓練系統執行成功！")
            sys.exit(0)
        else:
            print("\n❌ 自動訓練系統執行失敗！")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 系統錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 