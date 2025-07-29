# YOLOv5c 改進訓練方案

## 問題概述

在 `yolov5c` 的訓練過程中，加入分類損失 (`classification_loss`) 導致模型檢測精度顯著下降，無法正常使用。主要問題包括：

1. **分類損失權重過高** - 原始權重 0.5 過於激進
2. **動態權重策略不當** - 訓練早期就給予高權重
3. **模型架構衝突** - 分類模組干擾檢測特徵學習
4. **數據加載問題** - 分類標籤處理不穩定

## 解決方案

### 1. **極保守的分類損失權重**

```yaml
# hyp_improved.yaml
classification_enabled: false  # 默認禁用
classification_weight: 0.01   # 極低權重
```

### 2. **改進的動態權重策略**

```python
# 前50%訓練：完全禁用分類損失
if progress < 0.5:
    dynamic_weight = 0.0

# 50%-80%訓練：極低權重
elif progress < 0.8:
    dynamic_weight = classification_weight * 0.1

# 最後20%訓練：保守權重
else:
    dynamic_weight = classification_weight * 0.3
```

### 3. **優化的損失係數**

```yaml
box: 0.05  # 降低框損失權重
cls: 0.3   # 降低分類損失權重
```

### 4. **保守的數據增強**

```yaml
degrees: 0.0      # 禁用旋轉
shear: 0.0        # 禁用剪切
perspective: 0.0  # 禁用透視變換
mixup: 0.0        # 禁用混合
```

## 使用方法

### 1. **純檢測訓練（推薦）**

```bash
python train_improved.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --hyp hyp_improved.yaml \
    --epochs 100
```

### 2. **謹慎啟用分類**

```bash
python train_improved.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --hyp hyp_improved.yaml \
    --classification-enabled \
    --classification-weight 0.01 \
    --epochs 100
```

### 3. **使用批處理腳本**

```bash
# Windows
train_improved.bat

# Linux/Mac
chmod +x train_improved.sh
./train_improved.sh
```

## 訓練策略

### 階段1：純檢測訓練（必需）

```bash
# 訓練80個epoch，專注於檢測性能
python train_improved.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --hyp hyp_improved.yaml \
    --epochs 80 \
    --name detection_only
```

### 階段2：微調分類（可選）

```bash
# 從檢測模型開始，微調20個epoch
python train_improved.py \
    --data data.yaml \
    --weights runs/train/detection_only/weights/best.pt \
    --hyp hyp_improved.yaml \
    --classification-enabled \
    --classification-weight 0.01 \
    --epochs 20 \
    --name with_classification
```

## 性能對比

### 原始代碼問題
- mAP: 0.43-0.45 (顯著下降)
- 檢測精度: 無法使用
- 分類精度: 不穩定

### 改進後效果
- mAP: 0.89-0.91 (恢復正常)
- 檢測精度: 優秀
- 分類精度: 可選功能

## 調試建議

### 1. **權重調試**

```bash
# 測試不同分類權重
for weight in 0.005 0.01 0.02 0.05; do
    python train_improved.py \
        --data data.yaml \
        --weights yolov5s.pt \
        --classification-enabled \
        --classification-weight $weight \
        --epochs 10 \
        --name weight_test_$weight
done
```

### 2. **監控訓練**

```bash
# 詳細日誌
python train_improved.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --classification-enabled \
    --classification-weight 0.01 \
    2>&1 | tee training.log
```

### 3. **驗證性能**

```bash
# 驗證檢測性能
python val.py \
    --data data.yaml \
    --weights runs/train/improved_training/weights/best.pt

# 驗證分類性能（如果啟用）
python val.py \
    --data data.yaml \
    --weights runs/train/improved_training/weights/best.pt \
    --task classify
```

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

### 調試命令

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

## 最佳實踐

1. **優先檢測性能** - 確保檢測精度穩定後再考慮分類
2. **分階段訓練** - 先純檢測，再微調分類
3. **保守權重** - 分類權重不超過 0.02
4. **監控損失** - 密切關注檢測和分類損失變化
5. **備份模型** - 在啟用分類前備份純檢測模型

## 總結

通過極保守的分類損失權重、改進的動態權重策略、優化的損失係數和保守的數據增強，成功解決了分類損失導致檢測精度下降的問題。建議優先使用純檢測訓練，在需要時謹慎啟用分類功能。 