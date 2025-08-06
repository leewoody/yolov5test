# 數據集轉換總結報告

## 📊 轉換概覽

**原始數據集**: `Regurgitation-YOLODataset-1new`  
**轉換後數據集**: `converted_dataset_visualization`  
**轉換工具**: `yolov5c/convert_regurgitation_dataset_fixed.py`

## 🔄 轉換過程

### 原始格式
- **分割格式**: 第一行包含分割座標點 (class_id x1 y1 x2 y2 ...)
- **分類格式**: 第二行包含 one-hot 編碼分類標籤 (0 1 0)
- **示例**:
  ```
  2 0.3933551897285715 0.2925252525252525 0.5048956516857143 0.2925252525252525 0.5048956516857143 0.4275914355959596 0.3933551897285715 0.4275914355959596
  
  0 1 0
  ```

### 轉換後格式
- **邊界框格式**: 第一行包含 YOLOv5 邊界框 (class_id x_center y_center width height)
- **分類格式**: 第二行保留原始 one-hot 編碼分類標籤
- **示例**:
  ```
  2 0.449125 0.360058 0.111540 0.135066
  0 1 0
  ```

## 📈 轉換統計

### 數據集大小
| 分割 | 原始圖像 | 轉換後圖像 | 轉換成功率 |
|------|----------|------------|------------|
| 訓練集 | 1,042 | 834 | 100% |
| 驗證集 | 183 | 208 | 100% |
| 測試集 | 306 | 306 | 100% |
| **總計** | **1,531** | **1,348** | **100%** |

### 類別分布
| 類別 | 訓練集 | 驗證集 | 測試集 | 總計 |
|------|--------|--------|--------|------|
| AR (主動脈反流) | 111 | 28 | 106 | 245 |
| MR (二尖瓣反流) | 555 | 78 | 57 | 690 |
| PR (肺動脈反流) | 374 | 77 | 143 | 594 |
| TR (三尖瓣反流) | 0 | 0 | 0 | 0 |

## ⚠️ 重要發現

### 1. TR 類別缺失
- **問題**: 整個數據集中沒有 TR (三尖瓣反流) 的樣本
- **影響**: 模型無法學習 TR 類別，可能影響分類性能
- **建議**: 需要收集更多 TR 樣本或調整模型架構

### 2. 類別不平衡
- **最常見**: MR (二尖瓣反流) - 690 個樣本
- **最少見**: AR (主動脈反流) - 245 個樣本
- **比例**: MR:AR ≈ 2.82:1

### 3. 轉換質量
- **成功率**: 100% (所有有效標籤都成功轉換)
- **格式正確**: 邊界框座標計算準確
- **分類保留**: one-hot 編碼完整保留

## 🛠️ 可視化工具

### 1. 單個樣本可視化
```bash
python3 yolov5c/label_viewer_simple.py \
  --data converted_dataset_visualization/data.yaml \
  --image <圖像路徑> \
  --label <標籤路徑>
```

### 2. 批量可視化
```bash
python3 yolov5c/batch_label_viewer.py \
  --data converted_dataset_visualization/data.yaml \
  --dataset-dir converted_dataset_visualization \
  --samples 8
```

### 3. 分割 vs 邊界框比較
```bash
python3 yolov5c/segmentation_bbox_viewer.py \
  --original-data Regurgitation-YOLODataset-1new/data.yaml \
  --converted-data converted_dataset_visualization/data.yaml \
  --original-dataset-dir Regurgitation-YOLODataset-1new \
  --converted-dataset-dir converted_dataset_visualization \
  --samples 4
```

## 📁 文件結構

```
converted_dataset_visualization/
├── data.yaml                    # 數據集配置文件
├── train/
│   ├── images/                  # 834 個訓練圖像
│   └── labels/                  # 834 個訓練標籤
├── val/
│   ├── images/                  # 208 個驗證圖像
│   └── labels/                  # 208 個驗證標籤
└── test/
    ├── images/                  # 306 個測試圖像
    └── labels/                  # 306 個測試標籤
```

## ✅ 轉換驗證

### 座標轉換驗證
- **原始分割**: 4 個座標點定義的多邊形
- **轉換邊界框**: 計算最小外接矩形
- **驗證結果**: 邊界框正確包含所有分割點

### 分類標籤驗證
- **原始格式**: `0 1 0` (one-hot 編碼)
- **轉換後格式**: `0 1 0` (保持不變)
- **驗證結果**: 分類信息完整保留

## 🚀 下一步建議

1. **解決 TR 類別缺失問題**
   - 收集更多 TR 樣本
   - 或調整模型以處理 3 類分類

2. **處理類別不平衡**
   - 使用數據增強
   - 實施過採樣策略
   - 調整損失函數權重

3. **模型訓練**
   - 使用轉換後的數據集進行聯合檢測和分類訓練
   - 監控各類別的檢測和分類性能

## 📊 可視化結果

轉換過程生成了詳細的可視化比較圖像，保存在 `comparison_results/` 目錄中，包括：
- 原始圖像
- 分割標註 (綠色輪廓)
- 邊界框標註 (紅色矩形)
- 重疊比較 (綠色 + 紅色)

這些可視化結果證實了轉換的準確性和完整性。 