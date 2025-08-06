# 分割 vs 邊界框比較圖像說明

## 📊 概述

本目錄包含了 **1,139 個** 詳細的比較圖像，展示了原始分割格式與轉換後邊界框格式的對比。

## 🖼️ 圖像格式

每個比較圖像包含 **4 個子圖**：

### 1. 原始圖像 (Original Image)
- 顯示未標註的原始超聲波圖像
- 用於參考和對比

### 2. 原始分割標註 (Original Segmentation - Green)
- 顯示原始分割格式的標註
- **綠色輪廓線**: 分割多邊形的邊界
- **綠色圓點**: 分割頂點
- 格式: `class_id x1 y1 x2 y2 x3 y3 x4 y4`

### 3. 轉換邊界框 (Converted Bounding Box - Red)
- 顯示轉換後的 YOLOv5 邊界框格式
- **紅色矩形**: 邊界框
- **紅色圓點**: 邊界框中心點
- 格式: `class_id x_center y_center width height`

### 4. 重疊比較 (Overlay Comparison)
- 同時顯示分割和邊界框
- **綠色**: 原始分割
- **紅色**: 轉換邊界框
- 用於驗證轉換的準確性

## 📋 圖像信息

每個圖像底部包含詳細的標籤信息：

```
Image: [圖像文件名]
Original Label: [原始標籤文件名]
Converted Label: [轉換標籤文件名]

Segmentation Points: [分割點數量]
Calculated Bbox: (x_center, y_center, width, height)

Converted Bboxes: [邊界框數量]
  Bbox 1: (x_center, y_center, width, height)

Original Classification: [原始分類標籤]
Classes: [對應的類別名稱]

Converted Classification: [轉換後分類標籤]
Classes: [對應的類別名稱]
```

## 🏷️ 類別說明

數據集包含 4 個心臟瓣膜反流類別：

- **AR**: Aortic Regurgitation (主動脈反流)
- **MR**: Mitral Regurgitation (二尖瓣反流)  
- **PR**: Pulmonary Regurgitation (肺動脈反流)
- **TR**: Tricuspid Regurgitation (三尖瓣反流) - ⚠️ 數據集中無此類別樣本

## 📁 文件命名

比較圖像的命名格式：
```
comparison_[原始圖像文件名].png
```

例如：
- `comparison_a2hiwqVqZ2o=-unnamed_1_1.mp4-0.png`
- `comparison_ZmZiwqZua8KawqA=-unnamed_1_2.mp4-12.png`

## 🔍 驗證要點

查看比較圖像時，請注意以下幾點：

### 1. 轉換準確性
- 紅色邊界框應該完全包含綠色分割區域
- 邊界框不應過大或過小

### 2. 分類標籤
- 原始和轉換後的分類標籤應該完全相同
- 確認 one-hot 編碼格式正確

### 3. 座標精度
- 檢查邊界框座標是否合理
- 確認中心點位置準確

## 📊 統計信息

- **總圖像數量**: 1,139 個
- **轉換成功率**: 100%
- **覆蓋數據集**: 訓練集 + 測試集
- **圖像分辨率**: 高分辨率 (300 DPI)
- **文件格式**: PNG

## 🛠️ 生成工具

這些比較圖像是使用以下工具生成的：
```bash
python3 yolov5c/segmentation_bbox_viewer_batch.py \
  --original-data Regurgitation-YOLODataset-1new/data.yaml \
  --converted-data converted_dataset_visualization/data.yaml \
  --original-dataset-dir Regurgitation-YOLODataset-1new \
  --converted-dataset-dir converted_dataset_visualization \
  --save-dir comparison_results
```

## ✅ 質量保證

所有比較圖像都經過以下驗證：
- ✅ 圖像載入成功
- ✅ 標籤解析正確
- ✅ 座標轉換準確
- ✅ 分類信息保留
- ✅ 視覺效果清晰

## 🚀 使用建議

1. **批量檢查**: 使用文件管理器快速瀏覽多個圖像
2. **重點驗證**: 關注邊界框與分割的重疊程度
3. **類別分析**: 統計不同類別的樣本分布
4. **質量評估**: 識別可能的轉換錯誤或異常樣本

這些比較圖像為數據集轉換的質量提供了完整的視覺驗證，確保轉換過程的準確性和可靠性。 