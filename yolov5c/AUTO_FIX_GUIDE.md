# 🔧 Auto-Fix 自動優化系統使用指南

## 📋 概述

Auto-Fix 是一個智能訓練優化系統，能夠自動檢測並修正訓練過程中的常見問題，包括過擬合、準確率下降、學習率調整等。

## 🚀 核心功能

### 1. 過擬合自動檢測與修正
- **檢測條件**: 訓練損失 > 驗證損失 + 閾值
- **自動修正措施**:
  - 增加 Dropout 比例
  - 增大 Weight Decay
  - 降低學習率

### 2. 準確率下降自動調整
- **檢測條件**: 驗證準確率下降超過閾值
- **自動修正措施**:
  - 提升分類損失權重
  - 動態調整學習率

### 3. 智能早停機制
- **動態早停**: 根據驗證損失變化提前停止
- **最佳模型保存**: 自動保存最佳性能模型
- **可選功能**: 默認禁用，可通過參數啟用

## 🎯 使用方法

### 基本使用

```bash
# 使用 Auto-Fix 訓練
python yolov5c/train_joint_advanced.py \
    --data data.yaml \
    --epochs 50 \
    --batch-size 16
```

### 高級配置

```bash
# 自定義 Auto-Fix 參數
python3 yolov5c/train_joint_advanced.py \
    --data data.yaml \
    --epochs 100 \
    --batch-size 32 \
    --device cuda

# 啟用早停機制
python3 yolov5c/train_joint_advanced.py \
    --data data.yaml \
    --epochs 100 \
    --batch-size 32 \
    --enable-early-stop
```

## 📊 訓練監控

### 實時監控輸出

訓練過程中會顯示詳細的 Auto-Fix 信息：

```
Epoch 23/50:
  Train Loss: 2.0978 (Bbox: 0.0029, Cls: 0.2619)
  Val Loss: 1.5627 (Bbox: 0.0030, Cls: 0.1950)
  Val Cls Accuracy: 0.6490
  Best Val Cls Accuracy: 0.8125
  Current Cls Weight: 5.0
  🔧 Auto-fix applied

⚠️ 過擬合風險檢測，自動調整正則化與學習率
  Dropout: 0.300 -> 0.350
  Weight Decay: 0.0100 -> 0.0150
  Learning Rate: 0.001000 -> 0.000500
```

### 訓練完成後的分析

```bash
# 分析 Auto-Fix 效果
python yolov5c/analyze_advanced_training.py
```

## 🔧 Auto-Fix 配置參數

### 過擬合檢測配置

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `overfitting_threshold` | 過擬合檢測閾值 | 0.1 |
| `lr_reduction_factor` | 學習率減少因子 | 0.5 |
| `weight_decay_increase_factor` | Weight Decay 增加因子 | 1.5 |
| `dropout_increase_factor` | Dropout 增加因子 | 0.05 |
| `max_weight_decay` | 最大 Weight Decay | 0.05 |
| `max_dropout` | 最大 Dropout | 0.5 |

### 準確率下降檢測配置

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `accuracy_drop_threshold` | 準確率下降閾值 | 0.05 |
| `cls_weight_increase` | 分類權重增加量 | 1.0 |
| `cls_weight_decrease` | 分類權重減少量 | 0.5 |
| `max_cls_weight` | 最大分類權重 | 10.0 |
| `min_cls_weight` | 最小分類權重 | 3.0 |

## 📈 分析報告

### 訓練歷史文件

訓練完成後會生成 `training_history_advanced.json` 文件，包含：

```json
{
  "train_losses": [...],
  "val_losses": [...],
  "val_accuracies": [...],
  "overfitting_flags": [...],
  "accuracy_drops": [...]
}
```

### 分析圖表

運行分析腳本後會生成：

1. **advanced_training_analysis.png** - 完整的訓練分析圖表
2. **advanced_training_analysis_report.md** - 詳細的分析報告

## 🎯 最佳實踐

### 1. 監控訓練過程
- 觀察 Auto-Fix 觸發頻率
- 關注學習率和權重變化
- 檢查最終模型性能

### 2. 調整配置參數
- 如果過擬合頻繁，降低 `overfitting_threshold`
- 如果準確率不穩定，調整 `accuracy_drop_threshold`
- 根據數據集特點調整權重範圍

### 3. 結合驗證工具
```bash
# 訓練完成後驗證模型
python3 yolov5c/unified_validation.py \
    --weights joint_model_advanced.pth \
    --data data.yaml

# 分析訓練效果
python3 yolov5c/analyze_advanced_training.py
```

## ⚠️ 注意事項

### 1. 計算資源
- Auto-Fix 會增加少量計算開銷
- 建議使用 GPU 加速訓練

### 2. 參數調整
- 過於激進的 Auto-Fix 可能影響訓練穩定性
- 建議先使用默認參數，再根據需要調整

### 3. 數據質量
- Auto-Fix 無法解決數據質量問題
- 確保訓練數據標註準確

## 🔍 故障排除

### 常見問題

1. **Auto-Fix 頻繁觸發**
   ```bash
   # 檢查數據集大小和質量
   # 考慮減少模型複雜度
   ```

2. **訓練不穩定**
   ```bash
   # 降低學習率
   # 增加正則化
   ```

3. **準確率無法提升**
   ```bash
   # 檢查數據標註
   # 調整模型架構
   ```

## 📊 效果評估

### 優秀指標
- 過擬合檢測率 < 20%
- 準確率下降檢測率 < 20%
- 最終驗證準確率 > 90%

### 良好指標
- 過擬合檢測率 < 40%
- 準確率下降檢測率 < 40%
- 最終驗證準確率 > 80%

### 需要改進
- 過擬合檢測率 > 40%
- 準確率下降檢測率 > 40%
- 最終驗證準確率 < 80%

## 🚀 進階使用

### 早停控制

```bash
# 禁用早停 (默認)
python3 yolov5c/train_joint_advanced.py --data data.yaml --epochs 50

# 啟用早停
python3 yolov5c/train_joint_advanced.py --data data.yaml --epochs 50 --enable-early-stop

# 自動運行系統中的早停選項
python3 yolov5c/auto_run_training.py --data data.yaml --enable-early-stop
```

### 自定義 Auto-Fix 邏輯

可以在 `train_joint_advanced.py` 中修改 `auto_fix_config` 字典：

```python
auto_fix_config = {
    'overfitting_threshold': 0.05,  # 更敏感的過擬合檢測
    'accuracy_drop_threshold': 0.02,  # 更敏感的準確率下降檢測
    'lr_reduction_factor': 0.7,  # 更溫和的學習率調整
    # ... 其他參數
}
```

### 集成到其他訓練腳本

可以將 Auto-Fix 邏輯集成到其他訓練腳本中：

1. 複製 Auto-Fix 配置和邏輯
2. 在訓練循環中添加檢測代碼
3. 根據需要調整觸發條件

---

*最後更新: 2025年7月31日* 