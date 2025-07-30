# Segmentation to YOLOv5 Bbox Conversion Tools

這個工具集用於將包含 segmentation 格式的標籤文件轉換為標準的 YOLOv5 bbox 格式。

## 原始格式

原始標籤文件包含兩行：
```
2 0.44642857142857145 0.4450501104848485 0.5222854341714286 0.4450501104848485 0.5222854341714286 0.5804040404040405 0.44642857142857145 0.5804040404040405
0 1 0
```

- **第一行**：segmentation 座標（class x1 y1 x2 y2 x3 y3 ...）
- **第二行**：分類標籤（one-hot encoding，如 "0 1 0"）

## 轉換後格式

轉換後的標籤文件包含兩行：
```
2 0.484357 0.512727 0.075857 0.135354
0
```

- **第一行**：YOLOv5 bbox 格式（class x_center y_center width height）
- **第二行**：分類標籤（單一數字，如 "0"）

## 工具說明

### 1. convert_segmentation_to_bbox.py

單一文件或目錄轉換工具。

#### 使用方法：

```bash
# 轉換單一文件
python3 convert_segmentation_to_bbox.py --single-file "input.txt" output_dir

# 轉換整個目錄
python3 convert_segmentation_to_bbox.py --input-dir "input_labels" output_dir
```

#### 功能：
- 將 segmentation 多邊形轉換為邊界框
- 將 one-hot encoding 分類標籤轉換為單一數字
- 支援批量處理

### 2. convert_regurgitation_dataset.py

完整的數據集轉換工具，專門用於 Regurgitation-YOLODataset-1new。

#### 使用方法：

```bash
python3 convert_regurgitation_dataset.py "input_dataset" "output_dataset"
```

#### 功能：
- 轉換整個數據集（train/val/test）
- 複製圖片文件
- 生成 data.yaml 配置文件
- 自動處理目錄結構

## 轉換邏輯

### Segmentation 到 Bbox 轉換

1. 解析 segmentation 座標點
2. 計算所有點的最小/最大 x, y 值
3. 計算邊界框的中心點和寬高
4. 輸出 YOLOv5 格式：`class x_center y_center width height`

### 分類標籤轉換

1. 解析 one-hot encoding（如 "0 1 0"）
2. 找到值為 "1" 的索引位置
3. 輸出單一數字（如 "1"）

## 數據集統計

轉換結果：
- **訓練集**：1040 個成功轉換，2 個錯誤
- **測試集**：306 個成功轉換，0 個錯誤
- **類別**：AR, MR, PR, TR（4個類別）

## 使用建議

1. **備份原始數據**：轉換前請備份原始數據集
2. **檢查轉換結果**：轉換後檢查幾個樣本確認格式正確
3. **更新訓練腳本**：使用轉換後的 data.yaml 路徑

## 範例

```bash
# 轉換整個 Regurgitation 數據集
python3 convert_regurgitation_dataset.py "Regurgitation-YOLODataset-1new" "regurgitation_converted"

# 檢查轉換結果
ls regurgitation_converted/
cat regurgitation_converted/data.yaml
head regurgitation_converted/train/labels/*.txt
``` 