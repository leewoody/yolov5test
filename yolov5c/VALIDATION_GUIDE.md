# 🎯 YOLOv5 增強驗證使用指南

## 📋 概述

本指南介紹如何使用 YOLOv5 的增強驗證功能，包括檢測驗證、分類驗證和統一驗證。

## 🚀 快速開始

### 1. 統一驗證（推薦）

統一驗證腳本會自動判斷您的模型是檢測還是分類任務，並調用相應的驗證功能。

```bash
# 自動檢測任務類型
python yolov5c/unified_validation.py --weights best.pt --data data.yaml

# 手動指定任務類型
python yolov5c/unified_validation.py --weights best.pt --data data.yaml --task-type detection
python yolov5c/unified_validation.py --weights best-cls.pt --data data.yaml --task-type classification
```

### 2. 專用驗證

如果您知道確切的任務類型，可以直接使用專用驗證腳本：

#### 檢測驗證
```bash
python yolov5c/enhanced_validation.py \
    --weights best.pt \
    --data data.yaml \
    --batch-size 16 \
    --conf-thres 0.001 \
    --iou-thres 0.6
```

#### 分類驗證
```bash
python yolov5c/enhanced_classification_validation.py \
    --weights best-cls.pt \
    --data data.yaml \
    --batch-size 32 \
    --imgsz 224
```

## 📊 驗證指標說明

### 檢測驗證指標

| 指標 | 說明 | 理想值 |
|------|------|--------|
| **mAP@0.5** | 平均精度（IoU=0.5） | > 0.8 |
| **mAP@0.5:0.95** | 平均精度（IoU=0.5-0.95） | > 0.6 |
| **Precision** | 精確率 | > 0.8 |
| **Recall** | 召回率 | > 0.8 |
| **F1-Score** | F1分數 | > 0.8 |

### 分類驗證指標

| 指標 | 說明 | 理想值 |
|------|------|--------|
| **Top-1 Accuracy** | 前1準確率 | > 0.9 |
| **Top-5 Accuracy** | 前5準確率 | > 0.95 |
| **Precision** | 精確率 | > 0.8 |
| **Recall** | 召回率 | > 0.8 |
| **F1-Score** | F1分數 | > 0.8 |

## 📁 輸出文件

### 檢測驗證輸出

```
runs/enhanced_val/
├── confusion_matrix.png          # 混淆矩陣
├── confidence_distribution.png   # 置信度分佈
├── iou_distribution.png          # IoU分佈
├── per_class_performance.png     # 分類別性能
├── detailed_metrics.json         # 詳細指標（JSON）
└── validation_report.md          # 驗證報告
```

### 分類驗證輸出

```
runs/enhanced_cls_val/
├── confusion_matrix.png          # 混淆矩陣
├── confidence_analysis.png       # 置信度分析
├── roc_pr_curves.png            # ROC和PR曲線
├── per_class_performance.png     # 分類別性能
├── classification_metrics.json   # 詳細指標（JSON）
└── classification_report.md      # 分類報告
```

## 🎯 使用範例

### 範例1：醫學圖像檢測驗證

```bash
# 驗證心臟超音波檢測模型
python yolov5c/unified_validation.py \
    --weights runs/train/heart_detection/weights/best.pt \
    --data Regurgitation-YOLODataset-1new/data.yaml \
    --batch-size 8 \
    --conf-thres 0.25 \
    --save-dir runs/val_heart_detection
```

### 範例2：醫學圖像分類驗證

```bash
# 驗證心臟疾病分類模型
python yolov5c/enhanced_classification_validation.py \
    --weights runs/train/heart_classification/weights/best.pt \
    --data heart_classification/data.yaml \
    --batch-size 16 \
    --imgsz 224 \
    --save-dir runs/val_heart_classification
```

### 範例3：批次驗證多個模型

```bash
# 創建批次驗證腳本
cat > validate_all.sh << 'EOF'
#!/bin/bash

# 驗證檢測模型
python yolov5c/enhanced_validation.py \
    --weights models/detection_best.pt \
    --data data/detection.yaml \
    --save-dir runs/val_detection

# 驗證分類模型
python yolov5c/enhanced_classification_validation.py \
    --weights models/classification_best.pt \
    --data data/classification.yaml \
    --save-dir runs/val_classification

echo "所有驗證完成！"
EOF

chmod +x validate_all.sh
./validate_all.sh
```

## 🔧 參數說明

### 通用參數

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--weights` | 模型權重文件路徑 | 必需 |
| `--data` | 數據集配置文件路徑 | 必需 |
| `--batch-size` | 批次大小 | 32 |
| `--device` | 設備（CPU/GPU） | 自動 |
| `--save-dir` | 結果保存目錄 | runs/val |

### 檢測驗證參數

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--imgsz` | 推理尺寸 | 640 |
| `--conf-thres` | 置信度閾值 | 0.001 |
| `--iou-thres` | NMS IoU閾值 | 0.6 |

### 分類驗證參數

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--imgsz` | 推理尺寸 | 224 |

## 📈 結果解讀

### 檢測驗證結果解讀

1. **mAP@0.5 > 0.8**: 優秀的檢測性能
2. **Precision > 0.8**: 低誤檢率
3. **Recall > 0.8**: 高檢出率
4. **平均置信度 > 0.7**: 模型對預測有信心

### 分類驗證結果解讀

1. **Top-1 Accuracy > 0.9**: 優秀的分類性能
2. **F1-Score > 0.8**: 平衡的精確率和召回率
3. **ROC AUC > 0.9**: 優秀的區分能力
4. **混淆矩陣對角線 > 0.8**: 各類別分類準確

## ⚠️ 常見問題

### 1. 內存不足
```bash
# 減少批次大小
--batch-size 8

# 使用CPU
--device cpu
```

### 2. 驗證速度慢
```bash
# 使用GPU
--device 0

# 減少批次大小
--batch-size 16
```

### 3. 指標異常
```bash
# 檢查數據集配置
cat data.yaml

# 檢查模型權重
python -c "import torch; print(torch.load('best.pt', map_location='cpu').keys())"
```

## 🎯 最佳實踐

1. **使用統一驗證**: 自動判斷任務類型，減少錯誤
2. **調整置信度閾值**: 根據應用需求調整
3. **監控置信度分佈**: 確保模型對預測有信心
4. **分析混淆矩陣**: 識別分類錯誤模式
5. **比較多個模型**: 使用相同參數比較性能

## 📞 支援

如果遇到問題，請：

1. 檢查錯誤日誌
2. 確認文件路徑正確
3. 驗證數據集格式
4. 查看本指南的常見問題部分

---

*最後更新: 2025年7月31日* 