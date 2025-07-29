# YOLOv5 Test Project

## 專案概述

這是一個基於 YOLOv5 的目標檢測和分類測試專案，專注於醫學圖像分析，特別是心臟超音波圖像的檢測和分類任務。專案包含多個數據集和改進的訓練方案。

## 專案結構

```
yolov5test/
├── datasets/                    # 標準數據集
│   └── coco128/                # COCO128 數據集
├── demo/                       # 演示數據集
├── demo2/                      # 演示數據集 2
├── Regurgitation-YOLODataset-1new/  # 心臟反流數據集
├── small/                      # 小型數據集
├── small2/                     # 小型數據集 2
├── yolov5/                     # 原始 YOLOv5 代碼
└── yolov5c/                    # 自定義 YOLOv5 版本（主要開發版本）
```

## 主要功能

### 1. 目標檢測
- 基於 YOLOv5 的標準目標檢測
- 支持多種預訓練模型 (YOLOv5n, YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x)
- 自定義數據集訓練

### 2. 聯合檢測和分類
- 同時進行目標檢測和圖像分類
- 改進的損失函數設計
- 動態權重調整策略

### 3. 醫學圖像分析
- 心臟超音波圖像檢測
- 心臟疾病分類（正常、輕度、重度）
- 專業醫學數據集支持

## 最近更新 (2024年7月)

### 主要改進

1. **分類損失修復** - 解決了分類損失導致檢測精度下降的問題
2. **改進的訓練方案** - 新增多個訓練腳本和超參數配置
3. **數據加載器增強** - 支持聯合檢測和分類任務
4. **工具和文檔** - 新增多個實用工具和詳細文檔

### 新增文件

- `yolov5c/train_fixed.py` - 修復版訓練腳本
- `yolov5c/train_improved.py` - 改進版訓練腳本
- `yolov5c/hyp_fixed.yaml` - 修復版超參數
- `yolov5c/hyp_improved.yaml` - 改進版超參數
- `yolov5c/hyp_medical.yaml` - 醫學圖像專用超參數
- `yolov5c/utils/format_converter.py` - 數據格式轉換工具
- `yolov5c/utils/label_analyzer.py` - 標籤分析工具
- 多個 README 文檔

## 快速開始

### 環境要求

```bash
# Python 3.8+
# PyTorch 1.8+
# CUDA (可選，用於 GPU 加速)
```

### 安裝

```bash
# 克隆專案
git clone <repository-url>
cd yolov5test

# 安裝依賴
pip install -r yolov5c/requirements.txt
```

### 基本使用

#### 1. 純檢測訓練（推薦）

```bash
cd yolov5c

# 使用改進版訓練腳本
python train_improved.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --hyp hyp_improved.yaml \
    --epochs 100
```

#### 2. 聯合檢測和分類訓練

```bash
# 謹慎使用，建議先進行純檢測訓練
python train_improved.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --hyp hyp_improved.yaml \
    --classification-enabled \
    --classification-weight 0.01 \
    --epochs 100
```

#### 3. 使用批處理腳本

```bash
# Windows
train_improved.bat

# Linux/Mac
chmod +x train_improved.sh
./train_improved.sh
```

## 數據集

### 可用數據集

1. **COCO128** - 標準測試數據集
2. **Demo/Demo2** - 演示數據集
3. **Regurgitation-YOLODataset** - 心臟反流數據集
4. **Small/Small2** - 小型測試數據集

### 數據格式

#### 檢測標籤格式
```
class_id x_center y_center width height
```

#### 分類標籤格式
```
# 空格分隔格式（推薦）
0 0 1

# Python 列表格式
[0, 0, 1]
```

### 數據集配置

```yaml
# data.yaml
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 2  # 檢測類別數量
names: ['ASDType2', 'VSDType2']  # 檢測類別名稱

# 可選：分類配置
classification:
  nc: 3  # 分類類別數量
  names: ['Normal', 'Mild', 'Severe']  # 分類類別名稱
```

## 訓練策略

### 推薦的兩階段訓練

#### 階段1：純檢測訓練（必需）
```bash
python train_improved.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --hyp hyp_improved.yaml \
    --epochs 80 \
    --name detection_only
```

#### 階段2：微調分類（可選）
```bash
python train_improved.py \
    --data data.yaml \
    --weights runs/train/detection_only/weights/best.pt \
    --hyp hyp_improved.yaml \
    --classification-enabled \
    --classification-weight 0.01 \
    --epochs 20 \
    --name with_classification
```

### 超參數配置

#### 醫學圖像專用配置 (`hyp_medical.yaml`)
```yaml
# 保守的數據增強
degrees: 0.0      # 禁用旋轉
shear: 0.0        # 禁用剪切
perspective: 0.0  # 禁用透視變換
mixup: 0.0        # 禁用混合

# 優化的損失權重
box: 0.05
cls: 0.3
classification_weight: 0.01
```

## 工具和腳本

### 數據處理工具

#### 格式轉換器
```bash
# 驗證數據格式
python utils/format_converter.py --input train/labels --validate

# 轉換格式
python utils/format_converter.py --input train/labels --format space --output converted/labels
```

#### 標籤分析器
```bash
# 分析標籤分佈
python utils/label_analyzer.py --input train/labels --output analysis_report.md
```

### 訓練腳本

- `train.py` - 原始訓練腳本
- `train_fixed.py` - 修復版訓練腳本
- `train_improved.py` - 改進版訓練腳本
- `train_combined.py` - 聯合訓練腳本

## 性能對比

### 修復前 vs 修復後

| 指標 | 修復前 | 修復後 |
|------|--------|--------|
| mAP | 0.43-0.45 | 0.89-0.91 |
| 檢測精度 | 無法使用 | 優秀 |
| 分類精度 | 不穩定 | 可選功能 |

## 故障排除

### 常見問題

1. **分類損失為 NaN**
   ```bash
   # 降低分類權重
   --classification-weight 0.005
   ```

2. **檢測精度下降**
   ```bash
   # 禁用分類損失
   # 移除 --classification-enabled 參數
   ```

3. **訓練不穩定**
   ```bash
   # 使用更保守的超參數
   --hyp hyp_improved.yaml
   ```

### 調試建議

```bash
# 檢查數據格式
python utils/format_converter.py --input train/labels --validate

# 測試小批量訓練
python train_improved.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --batch-size 4 \
    --epochs 5 \
    --classification-enabled \
    --classification-weight 0.01
```

## 文檔

### 詳細文檔

- [改進訓練方案](yolov5c/README_improved_training.md)
- [分類損失修復方案](yolov5c/README_classification_fix.md)
- [數據加載器說明](yolov5c/utils/README_dataloaders.md)
- [標籤分析報告](yolov5c/utils/label_analysis_report.md)

### 醫學文檔

- [淡江心臟超音波簡報](yolov5c/docs/淡江心臟超音波簡報.pdf)

## 最佳實踐

1. **優先檢測性能** - 確保檢測精度穩定後再考慮分類
2. **分階段訓練** - 先純檢測，再微調分類
3. **保守權重** - 分類權重不超過 0.02
4. **監控損失** - 密切關注檢測和分類損失變化
5. **備份模型** - 在啟用分類前備份純檢測模型

## 貢獻

歡迎提交 Issue 和 Pull Request 來改進這個專案。

## 授權

本專案基於 YOLOv5 的 AGPL-3.0 授權。

## 更新日誌

### 2024年7月29日
- 修復分類損失導致的檢測精度下降問題
- 新增改進的訓練腳本和超參數配置
- 新增數據處理工具和文檔
- 優化醫學圖像訓練策略

### 2024年5月15日
- 初始版本發布
- 基本檢測和分類功能
- 醫學數據集支持 