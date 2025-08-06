# YOLOv5 Test Project

## 專案概述

這是一個基於 YOLOv5 的目標檢測和分類測試專案，專注於醫學圖像分析，特別是心臟超音波圖像的檢測和分類任務。專案包含多個數據集和改進的訓練方案。

## 🚨 重要規則 (Cursor Rules)

### YOLOv5WithClassification 聯合訓練規則

#### 1. 聯合訓練必須啟用分類功能
- **專案重點是 YOLOv5WithClassification 聯合檢測和分類**
- **必須啟用分類功能，不能禁用聯合訓練**
- **所有超參數文件都應配置為支持聯合訓練**

#### 2. 超參數配置規則
- **`hyp_improved.yaml`** - 改進版聯合訓練超參數
  - `classification_enabled: true`
  - `classification_weight: 0.01`
- **`hyp_medical.yaml`** - 醫學圖像聯合訓練超參數
  - `classification_enabled: true`
  - `classification_weight: 0.005`
- **`hyp_medical_complete.yaml`** - 醫學圖像完整聯合訓練超參數
  - `classification_enabled: true`
  - `classification_weight: 0.008`
  - `progressive_training: true`
  - `classification_schedule: true`
- **`hyp_fixed.yaml`** - 修復版聯合訓練超參數
  - `classification_enabled: true`
  - `classification_weight: 0.05`

#### 3. 聯合訓練命令規則

##### 高級聯合訓練 (train_joint_advanced.py)
```bash
# 基本高級聯合訓練
python train_joint_advanced.py \
    --data converted_dataset_fixed/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto

# 關閉早停，獲得完整訓練圖表
python train_joint_advanced.py \
    --data converted_dataset_fixed/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto
    # 移除 --enable-early-stop 參數
```

##### 終極聯合訓練 (train_joint_ultimate.py)
```bash
# 終極聯合訓練
python train_joint_ultimate.py \
    --data converted_dataset_fixed/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --bbox-weight 10.0 \
    --cls-weight 8.0

# 關閉早停，獲得完整訓練圖表
python train_joint_ultimate.py \
    --data converted_dataset_fixed/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --bbox-weight 10.0 \
    --cls-weight 8.0
    # 移除 --enable-earlystop 參數
```

##### 修復版聯合訓練 (train_joint_fixed.py)
```bash
# 修復版聯合訓練
python train_joint_fixed.py \
    --data converted_dataset_fixed/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --log training_fixed.log
```

##### 優化修復版聯合訓練 (train_joint_optimized_fixed.py)
```bash
# 優化修復版聯合訓練
python train_joint_optimized_fixed.py \
    --data converted_dataset_fixed/data.yaml \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640
```

#### 4. 早停機制規則

##### 重要：關閉早停機制
- **必須關閉早停機制** 以獲得完整的訓練圖表
- 移除所有 `--enable-early-stop` 和 `--enable-earlystop` 參數
- 這樣可以獲得：
  - 完整訓練曲線
  - 豐富的圖表數據
  - AUTO-FIX 效果觀察
  - 訓練穩定性評估

##### 關閉早停的原因
1. **完整訓練曲線** - 可以看到整個訓練過程的趨勢
2. **豐富的圖表數據** - 包括過擬合、準確率變化等
3. **更好的分析** - 可以分析模型在不同階段的表現
4. **AUTO-FIX 效果觀察** - 可以看到自動優化系統的完整效果
5. **訓練穩定性評估** - 觀察模型在後期的表現

#### 5. 數據擴增功能建議關閉
- **醫學圖像訓練建議關閉數據擴增**
- 保持醫學圖像的原始特徵和準確性
- 避免數據擴增對醫學診斷的干擾

#### 6. 推薦訓練策略

##### 階段 1：基礎聯合訓練
```bash
python train_joint_fixed.py \
    --data converted_dataset_fixed/data.yaml \
    --epochs 20 \
    --batch-size 16 \
    --device auto
```

##### 階段 2：高級優化訓練
```bash
python train_joint_advanced.py \
    --data converted_dataset_fixed/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto
```

##### 階段 3：終極聯合訓練
```bash
python train_joint_ultimate.py \
    --data converted_dataset_fixed/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --bbox-weight 10.0 \
    --cls-weight 8.0
```

#### 7. 快速測試命令

```bash
# 快速測試（小數據集，少輪數）
python train_joint_advanced.py \
    --data demo/data.yaml \
    --epochs 5 \
    --batch-size 8 \
    --device auto

# 快速測試終極版本
python train_joint_ultimate.py \
    --data demo/data.yaml \
    --epochs 3 \
    --batch-size 8 \
    --device auto \
    --bbox-weight 5.0 \
    --cls-weight 5.0
```

#### 8. 預期輸出文件

每個訓練腳本會生成：
- `joint_model_advanced.pth` / `joint_model_fixed.pth` / `joint_model_ultimate.pth`
- `training_history_advanced.json` / `training_history_fixed.json`
- 訓練日誌文件

#### 9. 預期獲得的圖表

關閉早停後，將獲得：
1. **訓練損失曲線** - 完整的訓練和驗證損失變化
2. **分類準確率曲線** - 驗證準確率的完整變化趨勢
3. **AUTO-FIX 效果圖** - 自動優化系統的干預次數和效果
4. **學習率變化圖** - 動態學習率調整的完整過程
5. **損失權重變化圖** - 分類損失權重的動態調整
6. **過擬合檢測圖** - 過擬合風險的完整監控

#### 10. 注意事項

1. **數據集路徑**：確保使用正確的 `data.yaml` 文件
2. **GPU 記憶體**：如果記憶體不足，減少 `batch-size`
3. **訓練時間**：終極版本訓練時間較長，建議先用小數據集測試
4. **早停機制**：必須關閉以獲得完整圖表
5. **聯合訓練**：必須啟用分類功能，不能禁用

#### 11. 禁止事項

- 禁止禁用分類功能 (`classification_enabled: false`)
- 禁止使用 `--enable-early-stop` 或 `--enable-earlystop` 參數
- 禁止使用純檢測訓練模式
- 禁止修改超參數文件中的分類設置為禁用狀態

#### 12. 語言規則

- 回應使用繁體中文
- 腳本註釋使用英文
- 保持專業和技術準確性

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

### 4. 增強驗證與訓練日誌分析（2025年7月）
- 新增 `enhanced_validation.py`：一鍵產生 mAP、Precision、Recall、F1、混淆矩陣、置信度分佈等完整驗證指標與圖表
- 新增 `enhanced_classification_validation.py`：完整分類驗證指標，包括混淆矩陣、ROC曲線、PR曲線等
- 新增 `unified_validation.py`：統一驗證腳本，自動判斷檢測或分類任務
- 新增 `run_enhanced_validation.py`：簡化批次驗證流程
- 新增 `quick_validation.py`、`analyze_your_log.py`：快速分析訓練日誌，產生訓練狀態報告與可視化

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
- `enhanced_validation.py` - 增強版檢測驗證腳本，產生完整驗證指標與圖表
- `enhanced_classification_validation.py` - 增強版分類驗證腳本，產生完整分類指標與圖表
- `unified_validation.py` - 統一驗證腳本，自動判斷檢測或分類任務
- `run_enhanced_validation.py` - 批次驗證運行器
- `quick_validation.py` - 快速訓練日誌分析
- `analyze_your_log.py` - 專用訓練日誌分析與可視化
- `analyze_advanced_training.py` - 高級訓練分析，評估Auto-Fix效果
- `auto_run_training.py` - 自動運行完整訓練流程
- `quick_auto_run.py` - 快速自動運行（推薦）
- `monitor_training.py` - 實時訓練日誌監控（帶圖表）
- `simple_monitor.py` - 簡化訓練監控器
- `quick_monitor.py` - 快速監控啟動器

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

### 2025年7月31日
- 新增增強驗證腳本與訓練日誌分析工具
- 新增分類驗證腳本，支援混淆矩陣、ROC曲線、PR曲線等完整分類指標
- 新增統一驗證腳本，自動判斷檢測或分類任務
- 新增自動優化訓練系統（Auto-Fix），自動處理過擬合和準確率下降問題
- 新增高級訓練分析腳本，分析Auto-Fix效果
- 新增實時訓練監控工具，支援進程監控和日誌分析
- 支援一鍵產生完整驗證指標、混淆矩陣、置信度分佈、訓練狀態報告
- 優化驗證流程與可視化 