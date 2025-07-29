# YOLOv5c DataLoaders 說明文檔

## 概述

`yolov5c` 包含兩個主要的數據加載器文件：
- `dataloaders.py`: 標準的 YOLOv5 數據加載器
- `dataloaders3.py`: 增強版數據加載器，支持聯合檢測和分類任務

## 主要差異

### dataloaders.py (標準版本)
- 支持標準的 YOLOv5 目標檢測任務
- 處理標準的 YOLO 格式標籤文件
- 不包含分類功能

### dataloaders3.py (增強版本)
- 繼承了標準版本的所有功能
- 新增 `LoadImagesLabelsAndCl` 類，支持聯合檢測和分類
- 新增 `create_cl_cache` 方法，用於創建分類標籤緩存
- 支持多種分類標籤格式

## 分類標籤格式

### 支持的格式

1. **空格分隔格式** (推薦):
   ```
   0 0.4765625 0.5875 0.22734375 0.3734375
   0 0 1
   ```

2. **Python 列表格式**:
   ```
   0 0.4765625 0.5875 0.22734375 0.3734375
   [0, 0, 1]
   ```

### 格式說明
- 第一行：標準 YOLO 檢測標籤 (class x_center y_center width height)
- 第二行：分類標籤 (多個數值表示不同類別的概率或標籤)

## 使用方法

### 1. 標準檢測任務
```python
from utils.dataloaders import create_dataloader

# 使用標準數據加載器
dataloader, dataset = create_dataloader(
    path='path/to/data.yaml',
    imgsz=640,
    batch_size=16,
    stride=32,
    augment=True
)
```

### 2. 聯合檢測和分類任務
```python
from utils.dataloaders3 import create_dataloader

# 使用增強版數據加載器
dataloader, dataset = create_dataloader(
    path='path/to/data.yaml',
    imgsz=640,
    batch_size=16,
    stride=32,
    augment=True
)
```

### 3. 自定義分類標籤路徑
```python
from utils.dataloaders3 import LoadImagesLabelsAndCl

# 創建包含分類標籤的數據集
dataset = LoadImagesLabelsAndCl(
    labelcl_path='path/to/classification_labels.npy',
    path='path/to/images',
    img_size=640,
    batch_size=16,
    augment=True
)
```

## 數據格式轉換

### 使用 format_converter.py

1. **驗證數據格式**:
   ```bash
   python utils/format_converter.py --input demo/train/labels --validate
   ```

2. **轉換為空格分隔格式**:
   ```bash
   python utils/format_converter.py --input demo2/train/labels --format space --output demo2_converted/train/labels
   ```

3. **轉換為 Python 列表格式**:
   ```bash
   python utils/format_converter.py --input demo/train/labels --format python --output demo_converted/train/labels
   ```

## 數據集配置

### data.yaml 文件格式
```yaml
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

## 注意事項

1. **格式一致性**: 確保整個數據集使用相同的分類標籤格式
2. **文件結構**: 標籤文件必須與圖像文件對應
3. **緩存**: 首次運行會創建緩存文件，後續運行會更快
4. **內存使用**: 大數據集建議使用磁盤緩存而非內存緩存

## 故障排除

### 常見問題

1. **分類標籤解析錯誤**:
   - 檢查標籤文件格式是否一致
   - 使用 `format_converter.py` 統一格式

2. **緩存文件損壞**:
   - 刪除 `.cache` 文件重新生成
   - 檢查文件權限

3. **內存不足**:
   - 減少 batch_size
   - 使用 `cache='disk'` 而非 `cache='ram'`

### 調試建議

1. 使用 `--validate` 參數檢查數據格式
2. 檢查標籤文件的數量和格式
3. 確認圖像文件路徑正確
4. 驗證 data.yaml 配置

## 更新日誌

### v1.1 (最新)
- 支持多種分類標籤格式
- 新增格式轉換工具
- 改進錯誤處理
- 添加詳細文檔

### v1.0
- 初始版本
- 基本檢測和分類功能
- 標準 YOLO 格式支持 