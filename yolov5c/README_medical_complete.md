# 🏥 醫學圖像檢測分類完整系統

## 📋 系統概述

本系統整合了 YOLOv5 目標檢測和 ResNet 分類功能，專門針對心臟超音波圖像進行分析，能夠同時檢測心臟瓣膜反流類型並評估其嚴重程度。

### 🎯 系統功能

#### **檢測功能 (Detection)**
- **AR (Aortic Regurgitation)** - 主動脈瓣反流
- **MR (Mitral Regurgitation)** - 二尖瓣反流  
- **PR (Pulmonary Regurgitation)** - 肺動脈瓣反流
- **TR (Tricuspid Regurgitation)** - 三尖瓣反流

#### **分類功能 (Classification)**
- **Normal** - 正常
- **Mild** - 輕度
- **Moderate** - 中度
- **Severe** - 重度

## 🚀 快速開始

### 1. 環境設置
```bash
# 安裝依賴
pip install -r requirements.txt

# 額外安裝醫學圖像處理依賴
pip install albumentations matplotlib tensorboard

# 確保CUDA可用（推薦）
nvidia-smi
```

### 2. 數據準備
```bash
# 數據集結構
Regurgitation-YOLODataset-1new/
├── train/
│   ├── images/     # 訓練圖像
│   └── labels/     # 訓練標籤 (YOLO格式)
├── valid/
│   ├── images/     # 驗證圖像
│   └── labels/     # 驗證標籤
└── test/
    ├── images/     # 測試圖像
    └── labels/     # 測試標籤
```

### 3. 訓練模型

#### 方法一：一鍵訓練（推薦）
```bash
# 設置執行權限
chmod +x train_medical_complete.sh

# 開始訓練
./train_medical_complete.sh
```

#### 方法二：手動訓練
```bash
# 創建配置文件
python train_medical_complete.py

# 開始訓練
python train_medical_complete.py \
    --config config_medical_complete.yaml \
    --epochs 150 \
    --device cuda
```

### 4. 推理檢測
```bash
# 使用訓練好的模型進行推理
python inference_medical_complete.py \
    --checkpoint runs/medical_complete/heart_ultrasound_complete/best.pt \
    --image path/to/your/heart_ultrasound.jpg \
    --conf-thres 0.3 \
    --iou-thres 0.4
```

## 📁 系統架構

```
yolov5c/
├── data_medical_complete.yaml          # 完整數據配置
├── hyp_medical_complete.yaml           # 醫學圖像超參數
├── config_medical_complete.yaml        # 訓練配置文件
├── train_medical_complete.py           # 完整訓練腳本
├── inference_medical_complete.py       # 完整推理腳本
├── train_medical_complete.sh           # 訓練批處理腳本
├── models/
│   └── medical_classifier.py           # 醫學分類模型
├── utils/
│   └── medical_dataloader.py           # 醫學數據加載器
└── README_medical_complete.md          # 本文檔
```

## ⚙️ 配置說明

### 數據配置 (`data_medical_complete.yaml`)
```yaml
# 數據集路徑
train: ../Regurgitation-YOLODataset-1new/train/images
val: ../Regurgitation-YOLODataset-1new/valid/images
test: ../Regurgitation-YOLODataset-1new/test/images

# 檢測類別
nc: 4
names: [AR, MR, PR, TR]

# 分類類別
classification_classes: 4
classification_names: [Normal, Mild, Moderate, Severe]

# 醫學圖像特定設置
medical_mode: true
confidence_threshold: 0.3
iou_threshold: 0.4
```

### 超參數配置 (`hyp_medical_complete.yaml`)
```yaml
# 優化器參數
lr0: 0.006  # 較低的學習率
warmup_epochs: 6.0  # 更長的預熱

# 損失係數
box: 0.035  # 檢測框損失
cls: 0.2    # 分類損失
classification_weight: 0.008  # 分類權重

# 數據增強（保守）
degrees: 0.0  # 禁用旋轉
shear: 0.0    # 禁用剪切
perspective: 0.0  # 禁用透視變換
```

## 📊 性能表現

### 預期性能
- **檢測 mAP**: 90%+
- **分類準確率**: 85%+
- **推理速度**: 30-50 FPS (GPU)

### 醫學圖像優化效果
- ✅ 提高檢測精度
- ✅ 減少假陽性
- ✅ 增強醫學圖像適應性
- ✅ 保守的數據增強策略
- ✅ 醫學診斷建議

## 🔧 系統功能

### 1. 聯合訓練
- **檢測優先**: 前70%訓練專注於檢測
- **分類後加**: 後30%加入分類任務
- **權重平衡**: 動態調整檢測和分類權重

### 2. 醫學圖像優化
- **保守增強**: 保持醫學圖像完整性
- **方向保持**: 避免旋轉和翻轉
- **結構保護**: 保持解剖結構

### 3. 智能推理
- **雙重分析**: 同時進行檢測和分類
- **置信度評估**: 醫學級別的置信度閾值
- **報告生成**: 自動生成醫學報告

### 4. 可視化結果
- **檢測框**: 彩色標註不同瓣膜類型
- **分類圖表**: 嚴重程度概率分布
- **醫學建議**: 基於結果的診斷建議

## 🎯 使用建議

### 訓練階段
1. **數據質量**: 確保醫學圖像質量高
2. **標註準確**: 專業醫學標註
3. **平衡數據**: 各類別樣本平衡
4. **驗證頻繁**: 醫學應用需要高可靠性

### 推理階段
1. **高置信度**: 使用較高的置信度閾值 (0.3+)
2. **醫學驗證**: 結果需要醫學專家驗證
3. **連續監控**: 定期更新模型性能

## 🚨 重要注意事項

### 醫學應用限制
- **非診斷工具**: 僅作為輔助分析工具
- **專家驗證**: 所有結果需要醫學專家確認
- **責任聲明**: 不承擔醫療診斷責任

### 技術限制
- **圖像質量**: 需要高質量超音波圖像
- **標準化**: 圖像需要標準化處理
- **更新維護**: 需要定期更新和維護

## 📈 監控和日誌

### TensorBoard 監控
```bash
# 查看訓練日誌
tensorboard --logdir runs/medical_complete/heart_ultrasound_complete/logs
```

### 日誌內容
- 檢測損失變化
- 分類損失變化
- 驗證準確率
- 學習率變化
- 模型性能指標

## 🔄 更新日誌

### v1.0.0 (當前版本)
- ✅ 整合 YOLOv5 檢測和 ResNet 分類
- ✅ 醫學圖像專用優化
- ✅ 聯合訓練策略
- ✅ 智能推理系統
- ✅ 醫學報告生成
- ✅ 完整可視化
- ✅ 一鍵式訓練腳本

## 📞 技術支持

### 常見問題
1. **GPU內存不足**: 減少 batch_size
2. **訓練不穩定**: 降低學習率
3. **檢測效果差**: 檢查數據標註質量
4. **分類不準確**: 平衡各類別數據

### 聯繫方式
- 檢查配置文件格式
- 驗證數據集結構
- 確認依賴包版本
- 查看錯誤日誌

## 📄 許可證

本系統基於 YOLOv5 AGPL-3.0 許可證，請遵守相關條款。

---

**⚠️ 醫學聲明**: 本系統僅供研究和輔助分析使用，不應作為醫療診斷的唯一依據。所有醫學決策應由合格的醫療專業人員做出。 