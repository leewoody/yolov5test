#!/usr/bin/env python3
"""
檢測優化訓練執行腳本
自動運行檢測優化訓練並生成報告
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse
import os

def run_command(command, description):
    """執行命令並顯示進度"""
    print(f"\n{'='*60}")
    print(f"執行: {description}")
    print(f"命令: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"✅ {description} 完成")
        print(f"執行時間: {end_time - start_time:.2f} 秒")
        
        if result.stdout:
            print("輸出:")
            print(result.stdout)
        
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"❌ {description} 失敗")
        print(f"執行時間: {end_time - start_time:.2f} 秒")
        print(f"錯誤代碼: {e.returncode}")
        
        if e.stdout:
            print("標準輸出:")
            print(e.stdout)
        
        if e.stderr:
            print("錯誤輸出:")
            print(e.stderr)
        
        return False, e.stderr

def check_prerequisites():
    """檢查前置條件"""
    print("檢查前置條件...")
    
    # 檢查數據集
    data_yaml = Path("converted_dataset_fixed/data.yaml")
    if not data_yaml.exists():
        print("❌ 數據集文件不存在: converted_dataset_fixed/data.yaml")
        return False
    
    # 檢查訓練腳本
    train_script = Path("yolov5c/train_joint_detection_optimized.py")
    if not train_script.exists():
        print("❌ 訓練腳本不存在: yolov5c/train_joint_detection_optimized.py")
        return False
    
    print("✅ 前置條件檢查通過")
    return True

def run_detection_optimization_training(epochs=30, batch_size=16, device='auto'):
    """運行檢測優化訓練"""
    
    # 構建訓練命令
    command = f"""python3 yolov5c/train_joint_detection_optimized.py \\
        --data converted_dataset_fixed/data.yaml \\
        --epochs {epochs} \\
        --batch-size {batch_size} \\
        --device {device} \\
        --bbox-weight 15.0 \\
        --cls-weight 5.0"""
    
    success, output = run_command(command, "檢測優化聯合訓練")
    
    if success:
        print("\n🎉 檢測優化訓練完成!")
        print("模型已保存為: joint_model_detection_optimized.pth")
    else:
        print("\n❌ 檢測優化訓練失敗!")
    
    return success

def run_performance_test():
    """運行性能測試"""
    command = "python3 quick_detection_test.py --compare"
    
    success, output = run_command(command, "檢測性能測試")
    
    if success:
        print("\n🎉 性能測試完成!")
    else:
        print("\n❌ 性能測試失敗!")
    
    return success

def generate_training_report():
    """生成訓練報告"""
    report_content = f"""# 🔍 檢測優化訓練報告

## 📋 訓練概述

### 訓練配置
- **模型**: DetectionOptimizedJointModel
- **數據集**: converted_dataset_fixed
- **訓練輪數**: 30
- **批次大小**: 16
- **檢測損失權重**: 15.0
- **分類損失權重**: 5.0

### 優化措施
1. **更深層檢測特化網絡** - 5層卷積塊專門為檢測優化
2. **IoU損失函數** - 直接優化IoU指標
3. **檢測注意力機制** - 增強檢測能力
4. **高檢測損失權重** - 優先優化檢測任務
5. **檢測特化數據增強** - 適度的數據增強
6. **基於IoU的模型選擇** - 直接優化IoU指標

## 🎯 預期改進

### 檢測任務
- **IoU提升**: 從 4.21% 目標提升至 >20%
- **mAP提升**: 從 4.21% 目標提升至 >15%
- **Bbox MAE**: 保持 <0.1 的優秀水平

### 分類任務
- **準確率**: 保持 >80% 的優秀水平
- **穩定性**: 保持訓練穩定性

## 📊 模型文件

- **檢測優化模型**: `joint_model_detection_optimized.pth`
- **終極聯合模型**: `joint_model_ultimate.pth` (對比基準)

## 🚀 使用指南

### 重新訓練
```bash
python yolov5c/train_joint_detection_optimized.py \\
    --data converted_dataset_fixed/data.yaml \\
    --epochs 30 \\
    --batch-size 16 \\
    --device auto \\
    --bbox-weight 15.0 \\
    --cls-weight 5.0
```

### 性能測試
```bash
python quick_detection_test.py --compare
```

### 快速測試
```bash
python quick_detection_test.py
```

## 📈 性能對比

| 模型 | IoU | mAP@0.5 | Bbox MAE | 分類準確率 |
|------|-----|---------|----------|------------|
| 終極聯合模型 | 4.21% | 4.21% | 0.0590 | 85.58% |
| 檢測優化模型 | 目標>20% | 目標>15% | <0.1 | >80% |

## 🎯 下一步計劃

1. **驗證檢測性能提升**
2. **進一步優化IoU損失函數**
3. **嘗試更深的檢測網絡**
4. **集成多尺度特徵**
5. **優化數據增強策略**

---
*生成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}*
*訓練狀態: 進行中*
"""
    
    with open("detection_optimization_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 訓練報告已生成: detection_optimization_report.md")

def main():
    parser = argparse.ArgumentParser(description='檢測優化訓練執行腳本')
    parser.add_argument('--epochs', type=int, default=30, help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='auto', help='設備')
    parser.add_argument('--skip-training', action='store_true', help='跳過訓練，只運行測試')
    parser.add_argument('--skip-test', action='store_true', help='跳過測試，只運行訓練')
    args = parser.parse_args()
    
    print("🚀 檢測優化訓練執行腳本")
    print("=" * 60)
    
    # 檢查前置條件
    if not check_prerequisites():
        print("❌ 前置條件檢查失敗，退出")
        sys.exit(1)
    
    # 生成訓練報告
    generate_training_report()
    
    # 運行檢測優化訓練
    if not args.skip_training:
        training_success = run_detection_optimization_training(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        if not training_success:
            print("❌ 訓練失敗，跳過性能測試")
            sys.exit(1)
    else:
        print("⏭️ 跳過訓練階段")
    
    # 運行性能測試
    if not args.skip_test:
        test_success = run_performance_test()
        
        if test_success:
            print("\n🎉 所有任務完成!")
        else:
            print("\n⚠️ 性能測試失敗，但訓練可能成功")
    else:
        print("⏭️ 跳過性能測試")
    
    print("\n📋 總結:")
    print("- 檢測優化模型: joint_model_detection_optimized.pth")
    print("- 訓練報告: detection_optimization_report.md")
    print("- 性能測試: 使用 quick_detection_test.py")

if __name__ == '__main__':
    main() 