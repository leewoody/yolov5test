# 🔍 訓練監控工具使用指南

## 📋 概述

訓練監控工具提供實時監控訓練進程、分析訓練日誌、檢測問題和優化建議的功能。

## 🚀 監控工具

### 1. 快速監控啟動器 (`quick_monitor.py`)
- **自動檢測**: 自動檢測訓練進程和日誌文件
- **智能選擇**: 根據當前狀態選擇最佳監控模式
- **一鍵啟動**: 最簡單的使用方式

### 2. 簡化監控器 (`simple_monitor.py`)
- **進程監控**: 監控訓練進程的CPU、內存使用率
- **日誌監控**: 實時分析訓練日誌
- **趨勢分析**: 自動檢測過擬合和準確率下降

### 3. 高級監控器 (`monitor_training.py`)
- **實時圖表**: 動態更新訓練曲線
- **詳細分析**: 完整的訓練數據分析
- **Auto-Fix監控**: 專門監控Auto-Fix事件

## 🎯 使用方法

### 快速開始
```bash
# 自動檢測並啟動最佳監控模式
python3 yolov5c/quick_monitor.py
```

### 進程監控
```bash
# 監控訓練進程
python3 yolov5c/simple_monitor.py
# 選擇模式 1: 監控進程
```

### 日誌監控
```bash
# 監控日誌文件
python3 yolov5c/simple_monitor.py
# 選擇模式 2: 監控日誌文件
```

### 實時圖表監控
```bash
# 高級監控（需要matplotlib）
python3 yolov5c/monitor_training.py
```

## 📊 監控功能

### 進程監控
- **CPU使用率**: 實時監控CPU使用情況
- **內存使用率**: 監控內存佔用
- **運行時間**: 計算訓練運行時間
- **進度估算**: 基於時間估算訓練進度
- **進程識別**: 自動識別Auto-Fix訓練進程

### 日誌分析
- **Epoch進度**: 實時顯示訓練輪數
- **損失變化**: 監控訓練和驗證損失
- **準確率變化**: 追蹤分類準確率
- **Auto-Fix事件**: 檢測自動優化事件
- **趨勢分析**: 自動分析訓練趨勢

### 問題檢測
- **過擬合檢測**: 當訓練損失 < 驗證損失時警告
- **準確率下降**: 檢測準確率下降趨勢
- **高損失警告**: 當驗證損失過高時提醒
- **性能評估**: 自動評估訓練效果

## 🎯 監控輸出示例

### 進程監控輸出
```
⏰ 22:41:56 - 發現 4 個訓練進程

📊 進程 67555: python3 yolov5c/train_joint_advanced.py
  運行時間: 13:23
  CPU使用率: 276.2%
  內存使用率: 9.3%
  🔧 Auto-Fix 訓練進程
  估算已運行: 4.5 epochs
```

### 日誌監控輸出
```
📈 Epoch 23/50
  訓練損失: 2.0978
  驗證損失: 1.5627
  驗證準確率: 0.6490 (64.9%)
  ⚠️ 過擬合風險

🔧 Auto-Fix: 過擬合風險檢測，自動調整正則化與學習率
```

## 🔧 配置選項

### 更新間隔
```bash
# 設置更新間隔為1秒
python3 yolov5c/monitor_training.py --update-interval 1.0
```

### 指定日誌文件
```bash
# 監控指定日誌文件
python3 yolov5c/monitor_training.py --log-file training.log
```

### 禁用自動檢測
```bash
# 禁用自動檢測日誌文件
python3 yolov5c/monitor_training.py --no-auto-detect
```

## 📈 監控指標

### 性能指標
- **CPU使用率**: 正常範圍 50-300%
- **內存使用率**: 正常範圍 5-20%
- **訓練損失**: 應該逐漸下降
- **驗證損失**: 應該穩定或下降
- **準確率**: 應該逐漸提升

### 警告條件
- **過擬合**: 訓練損失 < 驗證損失
- **準確率下降**: 連續epoch準確率下降
- **高損失**: 驗證損失 > 2.0
- **低準確率**: 準確率 < 60%

## 🎯 最佳實踐

### 1. 監控時機
- 訓練開始後立即啟動監控
- 長時間訓練時持續監控
- 出現問題時重點監控

### 2. 監控重點
- 關注損失變化趨勢
- 監控Auto-Fix觸發頻率
- 檢查資源使用情況

### 3. 問題處理
- 過擬合時觀察Auto-Fix效果
- 準確率下降時檢查數據質量
- 資源不足時調整批次大小

## 🚀 進階使用

### 同時監控多個進程
```bash
# 使用簡化監控器的同時監控模式
python3 yolov5c/simple_monitor.py
# 選擇模式 3: 同時監控
```

### 自定義監控腳本
```python
from yolov5c.monitor_training import TrainingMonitor

# 創建自定義監控器
monitor = TrainingMonitor(
    log_file='custom_training.log',
    update_interval=1.0
)

# 開始監控
monitor.start_monitoring()
```

### 批量監控
```bash
# 監控多個訓練任務
for i in {1..3}; do
    python3 yolov5c/quick_monitor.py &
done
```

## ⚠️ 注意事項

### 1. 系統要求
- **matplotlib**: 高級監控需要matplotlib
- **psutil**: 進程監控需要psutil
- **Python 3.6+**: 需要Python 3.6或更高版本

### 2. 性能影響
- 監控工具本身會消耗少量資源
- 建議在訓練機器上運行
- 長時間監控時注意磁盤空間

### 3. 日誌文件
- 確保日誌文件有讀取權限
- 大型日誌文件可能影響監控性能
- 建議定期清理舊日誌文件

## 🔍 故障排除

### 常見問題

1. **找不到訓練進程**
   ```bash
   # 檢查進程是否正在運行
   ps aux | grep python
   ```

2. **日誌文件無法讀取**
   ```bash
   # 檢查文件權限
   ls -la *.log
   ```

3. **圖表無法顯示**
   ```bash
   # 安裝matplotlib
   pip install matplotlib
   ```

4. **監控工具無響應**
   ```bash
   # 檢查系統資源
   top
   ```

### 調試建議
- 使用 `--no-auto-detect` 手動指定文件
- 降低更新間隔減少資源消耗
- 檢查日誌文件格式是否正確

---

*最後更新: 2025年7月31日* 