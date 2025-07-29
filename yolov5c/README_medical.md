# YOLOv5 醫學圖像檢測專案

## 📋 專案概述

本專案針對醫學圖像（特別是心臟超音波）進行了優化，專門用於檢測心臟瓣膜反流（Regurgitation）問題。

### 🏥 檢測目標
- **AR (Aortic Regurgitation)** - 主動脈瓣反流
- **MR (Mitral Regurgitation)** - 二尖瓣反流  
- **PR (Pulmonary Regurgitation)** - 肺動脈瓣反流
- **TR (Tricuspid Regurgitation)** - 三尖瓣反流

## 🚀 快速開始

### 1. 環境設置
```bash
# 安裝依賴
pip install -r requirements.txt

# 確保CUDA可用（推薦）
nvidia-smi
```

### 2. 數據準備
```bash
# 數據集結構
Regurgitation-YOLODataset-1new/
├── train/
│   ├── images/     # 訓練圖像
│   └── labels/     # 訓練標籤
├── valid/
│   ├── images/     # 驗證圖像
│   └── labels/     # 驗證標籤
└── test/
    ├── images/     # 測試圖像
    └── labels/     # 測試標籤
```

### 3. 訓練模型

#### 方法一：使用醫學專用腳本（推薦）
```bash
# 使用醫學圖像優化的訓練腳本
chmod +x train_medical.sh
./train_medical.sh
```

#### 方法二：手動訓練
```bash
python train_medical.py \
    --data data_regurgitation.yaml \
    --hyp hyp_medical.yaml \
    --weights yolov5s.pt \
    --epochs 150 \
    --batch-size 16 \
    --imgsz 640 \
    --medical-mode \
    --project runs/medical_train \
    --name heart_valve_regurgitation
```

### 4. 檢測推理
```bash
# 使用醫學圖像優化的檢測
python detect_medical.py \
    --weights runs/medical_train/heart_valve_regurgitation/weights/best.pt \
    --source path/to/your/images \
    --conf-thres 0.3 \
    --iou-thres 0.4 \
    --medical-mode
```

## ⚙️ 醫學圖像專用配置

### 超參數優化 (`hyp_medical.yaml`)
- **保守的學習率**: `lr0: 0.008` (降低初始學習率)
- **醫學圖像增強**: 禁用旋轉、剪切、透視變換
- **高精度閾值**: `iou_t: 0.25` (提高IoU閾值)
- **分類損失**: 默認禁用，避免干擾檢測

### 數據配置 (`data_regurgitation.yaml`)
```yaml
# 心臟瓣膜反流檢測配置
train: ../Regurgitation-YOLODataset-1new/train/images
val: ../Regurgitation-YOLODataset-1new/valid/images
test: ../Regurgitation-YOLODataset-1new/test/images

nc: 4  # 4個類別
names:
  - AR    # 主動脈瓣反流
  - MR    # 二尖瓣反流
  - PR    # 肺動脈瓣反流
  - TR    # 三尖瓣反流
```

## 📊 性能表現

### 當前模型性能
- **mAP**: 89-91%
- **AP50**: 88-91%
- **各類別AP**: 71-100%

### 醫學圖像優化效果
- ✅ 提高檢測精度
- ✅ 減少假陽性
- ✅ 增強醫學圖像適應性
- ✅ 保守的數據增強策略

## 🔧 醫學圖像專用功能

### 1. 醫學模式 (`--medical-mode`)
- 自動調整置信度閾值
- 增強標籤顯示
- 醫學圖像特定的後處理

### 2. 保守的數據增強
- 禁用旋轉（保持醫學方向）
- 禁用剪切（保持解剖結構）
- 禁用透視變換（保持醫學準確性）

### 3. 增強的可視化
- 置信度等級標示
- 醫學圖像優化的標註
- 更粗的邊界框線條

## 📁 文件結構

```
yolov5c/
├── data_regurgitation.yaml      # 醫學數據配置
├── hyp_medical.yaml             # 醫學圖像超參數
├── train_medical.py             # 醫學圖像訓練腳本
├── detect_medical.py            # 醫學圖像檢測腳本
├── train_medical.sh             # 醫學圖像訓練批處理
├── README_medical.md            # 醫學圖像專用文檔
├── train_improved.py            # 改進版訓練腳本
├── train_fixed.py               # 修復版訓練腳本
└── requirements.txt             # 依賴包列表
```

## 🎯 使用建議

### 訓練階段
1. **純檢測訓練**: 使用 `train_medical.py` 進行純檢測訓練
2. **保守增強**: 醫學圖像使用保守的數據增強策略
3. **耐心訓練**: 醫學圖像需要更長的訓練時間

### 推理階段
1. **高置信度**: 使用較高的置信度閾值 (0.3+)
2. **醫學模式**: 啟用醫學模式獲得最佳結果
3. **後處理**: 根據醫學需求進行額外的後處理

## 🚨 注意事項

### 醫學圖像特殊性
- **方向敏感**: 醫學圖像的方向很重要，避免旋轉增強
- **結構保持**: 保持解剖結構的完整性
- **高精度要求**: 醫學檢測需要更高的精度

### 模型限制
- 僅適用於心臟超音波圖像
- 需要專業醫學驗證
- 不應作為唯一診斷依據

## 🔄 更新日誌

### v1.0.0 (當前版本)
- ✅ 醫學圖像專用訓練腳本
- ✅ 保守的數據增強策略
- ✅ 醫學模式檢測
- ✅ 心臟瓣膜反流檢測優化
- ✅ 89-91% mAP性能

## 📞 技術支持

如有問題，請檢查：
1. 數據集格式是否正確
2. 依賴包是否完整安裝
3. GPU內存是否充足
4. 醫學圖像質量是否達標

## 📄 許可證

本專案基於 YOLOv5 AGPL-3.0 許可證，請遵守相關條款。 