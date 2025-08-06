# 原始代碼訓練驗證指南

## 概述

本指南提供使用原始YOLOv5代碼進行訓練驗證的完整流程，用於建立基準性能，為後續的梯度干擾解決方案提供對比基準。

## 快速開始

### 方法1: 使用Shell腳本（推薦）

```bash
# 進入yolov5c目錄
cd Jon_yolov5test/yolov5c

# 執行驗證腳本
./run_original_validation.sh
```

腳本會提供三個選項：
1. **快速測試** (5 epochs, 小batch size) - 約10-15分鐘
2. **完整訓練** (50 epochs, 正常batch size) - 約2-3小時
3. **兩者都執行** - 先快速測試，再完整訓練

### 方法2: 直接執行Python腳本

```bash
# 進入yolov5c目錄
cd Jon_yolov5test/yolov5c

# 快速測試
python quick_original_test.py

# 或者完整訓練驗證
python original_training_validation.py
```

### 方法3: 直接使用原始命令

```bash
# 進入yolov5c目錄
cd Jon_yolov5test/yolov5c

# 訓練
python train.py \
    --data ../Regurgitation-YOLODataset-1new/data.yaml \
    --weights yolov5s.pt \
    --epochs 50 \
    --batch-size 16 \
    --imgsz 640 \
    --device auto \
    --project runs/original_validation \
    --name original_baseline \
    --exist-ok

# 驗證
python val.py \
    --data ../Regurgitation-YOLODataset-1new/data.yaml \
    --weights runs/original_validation/original_baseline/weights/best.pt \
    --imgsz 640 \
    --device auto \
    --project runs/original_validation \
    --name original_baseline_validation \
    --exist-ok
```

## 數據集配置

### 數據集結構
```
Regurgitation-YOLODataset-1new/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### data.yaml配置
```yaml
names:
- AR
- MR
- PR
- TR
nc: 4
train: ../Regurgitation-YOLODataset-1new/train/images
val: ../Regurgitation-YOLODataset-1new/valid/images
test: ../Regurgitation-YOLODataset-1new/test/images
```

## 訓練參數

### 快速測試參數
- **Epochs**: 5
- **Batch Size**: 8
- **Image Size**: 320
- **Device**: Auto (GPU/CPU)
- **Early Stopping**: 10 epochs patience

### 完整訓練參數
- **Epochs**: 50
- **Batch Size**: 16
- **Image Size**: 640
- **Device**: Auto (GPU/CPU)
- **Early Stopping**: 50 epochs patience
- **Save Period**: 10 epochs

## 預期結果

### 基準性能（原始代碼）
基於您的分析，預期會看到以下性能：

- **mAP@0.5**: ~0.07 (7%)
- **mAP@0.5:0.95**: ~0.03 (3%)
- **Precision**: ~0.07 (7%)
- **Recall**: ~0.23 (23%)
- **F1-Score**: ~0.10 (10%)

### 性能問題分析
這些低性能指標證實了您提到的梯度干擾問題：

1. **分類任務干擾檢測任務**
2. **共享特徵層無法同時優化兩種不同的特徵表示**
3. **即使分類權重很低，梯度干擾仍然存在**

## 結果分析

### 訓練曲線分析
觀察以下指標的變化趨勢：

1. **檢測損失 vs 分類損失**
   - 檢測損失應該下降緩慢
   - 分類損失可能下降較快
   - 兩者可能存在競爭關係

2. **驗證準確率**
   - mAP值應該相對較低
   - 可能存在過擬合現象
   - 檢測和分類性能不平衡

3. **梯度範數**
   - 檢測和分類梯度的相對大小
   - 梯度干擾的視覺化證據

### 關鍵觀察點

1. **梯度干擾證據**
   - 檢測性能遠低於預期
   - 分類任務主導訓練過程
   - 損失曲線不穩定

2. **特徵表示衝突**
   - 檢測需要定位特徵
   - 分類需要語義特徵
   - 共享特徵層無法同時滿足

3. **權重設置問題**
   - 即使分類權重很低，干擾仍然存在
   - 靜態權重無法動態平衡任務

## 輸出文件

### 訓練輸出
```
runs/original_validation/
├── original_baseline_YYYYMMDD_HHMMSS/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── hyp.yaml
│   └── opt.yaml
└── original_baseline_YYYYMMDD_HHMMSS_validation/
    ├── confusion_matrix.png
    └── results.txt
```

### 結果文件
- **weights/**: 訓練好的模型權重
- **results.png**: 訓練曲線圖
- **confusion_matrix.png**: 混淆矩陣
- **hyp.yaml**: 超參數配置
- **opt.yaml**: 訓練選項配置

## 下一步

完成原始代碼驗證後，您可以：

1. **實施GradNorm解決方案**
   ```bash
   python optimized_gradnorm_solution.py \
       --data ../Regurgitation-YOLODataset-1new/data.yaml \
       --epochs 50 \
       --batch-size 16
   ```

2. **實施雙Backbone解決方案**
   ```bash
   python dual_backbone_solution.py \
       --data ../Regurgitation-YOLODataset-1new/data.yaml \
       --epochs 50 \
       --batch-size 16
   ```

3. **對比分析**
   - 比較三種方案的性能
   - 分析梯度干擾的解決效果
   - 驗證您的理論分析

## 故障排除

### 常見問題

1. **CUDA內存不足**
   ```bash
   # 減少batch size
   --batch-size 8
   
   # 減少image size
   --imgsz 320
   ```

2. **數據集路徑錯誤**
   ```bash
   # 檢查數據集路徑
   ls -la ../Regurgitation-YOLODataset-1new/
   ```

3. **權重文件不存在**
   ```bash
   # 檢查權重文件
   ls -la yolov5s.pt
   ```

### 調試模式

```bash
# 啟用詳細日誌
python train.py --data ../Regurgitation-YOLODataset-1new/data.yaml \
    --weights yolov5s.pt \
    --epochs 5 \
    --batch-size 4 \
    --verbose
```

## 結論

原始代碼驗證將為您提供：

1. **基準性能數據** - 用於後續方案對比
2. **梯度干擾證據** - 驗證您的理論分析
3. **問題量化** - 具體的性能差距數值
4. **改進空間** - 明確的優化目標

這些結果將為您的論文提供重要的實驗基礎和對比基準。 