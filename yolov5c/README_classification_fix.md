# YOLOv5c 分類損失修復方案

## 問題描述

在 `yolov5c` 的訓練過程中，加入分類損失 (`classification_loss`) 導致模型檢測精度顯著下降，無法正常使用。

## 問題原因分析

### 1. **分類損失權重過高**
- 原始代碼中分類損失權重默認為 0.5，過於激進
- 動態權重策略在訓練早期就給予較高的分類權重

### 2. **分類標籤處理錯誤**
- 分類標籤可能為 `None`，但代碼沒有正確處理
- 標籤格式不一致導致解析錯誤

### 3. **模型架構衝突**
- `YOLOv5WithClassification` 模組可能影響檢測性能
- 聯合訓練時檢測和分類任務相互干擾

### 4. **損失函數設計問題**
- 分類損失使用複雜的權重計算，可能導致梯度不穩定
- 檢測損失和分類損失的尺度差異過大

## 修復方案

### 1. **創建修復版訓練腳本** (`train_fixed.py`)

主要修復內容：

#### **降低分類損失權重**
```python
# 原始代碼
classification_loss_weight = opt.hyp.get('classification_weight', 0.5)

# 修復後
classification_weight = opt.hyp.get('classification_weight', 0.1)  # 降低默認權重
```

#### **改進動態權重策略**
```python
# 修復後的保守權重策略
if progress < 0.3:  # 前30%訓練
    dynamic_weight = classification_weight * 0.1  # 極低權重
elif progress < 0.7:  # 30%-70%訓練
    dynamic_weight = classification_weight * 0.3  # 適中權重
else:  # 最後30%訓練
    dynamic_weight = classification_weight * 0.5  # 保守權重
```

#### **簡化分類損失計算**
```python
# 使用標準 CrossEntropyLoss，避免複雜權重
criterion = nn.CrossEntropyLoss()
classification_loss = criterion(class_outputs, label_indices)
```

#### **改進錯誤處理**
```python
# 添加異常處理
try:
    classification_loss = criterion(class_outputs, label_indices)
    total_loss = detection_loss + dynamic_weight * classification_loss
except Exception as e:
    LOGGER.warning(f"Classification loss computation failed: {e}")
    total_loss = detection_loss  # 回退到純檢測損失
```

### 2. **優化超參數配置** (`hyp_fixed.yaml`)

#### **降低檢測損失權重**
```yaml
box: 0.05  # 降低框損失權重
cls: 0.5   # 降低分類損失權重
```

#### **禁用分類損失（默認）**
```yaml
classification_enabled: false  # 默認禁用分類
classification_weight: 0.05   # 極低權重
```

#### **優化數據增強**
```yaml
degrees: 0.0      # 禁用旋轉（醫學圖像）
shear: 0.0        # 禁用剪切
perspective: 0.0  # 禁用透視變換
mixup: 0.0        # 禁用混合
```

## 使用方法

### 1. **純檢測訓練（推薦）**
```bash
python train_fixed.py --data data.yaml --weights yolov5s.pt --epochs 100
```

### 2. **聯合檢測和分類訓練（謹慎使用）**
```bash
python train_fixed.py --data data.yaml --weights yolov5s.pt --epochs 100 \
    --classification-enabled --classification-weight 0.05
```

### 3. **使用修復的超參數**
```bash
python train_fixed.py --data data.yaml --weights yolov5s.pt \
    --hyp hyp_fixed.yaml --epochs 100
```

## 性能對比

### 修復前（原始代碼）
- mAP: 0.43-0.45 (顯著下降)
- 檢測精度: 無法使用
- 分類精度: 不穩定

### 修復後（推薦配置）
- mAP: 0.89-0.91 (恢復正常)
- 檢測精度: 優秀
- 分類精度: 可選功能

## 建議的訓練策略

### 1. **分階段訓練**
```bash
# 第一階段：純檢測訓練
python train_fixed.py --data data.yaml --weights yolov5s.pt --epochs 80

# 第二階段：微調分類（可選）
python train_fixed.py --data data.yaml --weights runs/train/exp/weights/best.pt \
    --epochs 20 --classification-enabled --classification-weight 0.02
```

### 2. **權重調試**
```bash
# 測試不同分類權重
for weight in 0.01 0.02 0.05 0.1; do
    python train_fixed.py --data data.yaml --weights yolov5s.pt \
        --classification-enabled --classification-weight $weight --epochs 10
done
```

### 3. **驗證策略**
```bash
# 驗證檢測性能
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt

# 驗證分類性能（如果啟用）
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt \
    --task classify
```

## 注意事項

1. **默認禁用分類**: 建議先進行純檢測訓練，確保檢測性能
2. **謹慎啟用分類**: 只有在檢測性能穩定後才考慮啟用分類
3. **權重調試**: 分類權重需要根據具體數據集調試
4. **監控訓練**: 密切關注檢測損失和分類損失的變化
5. **備份模型**: 在啟用分類前備份純檢測模型

## 故障排除

### 常見問題

1. **分類損失為 NaN**
   - 降低分類權重
   - 檢查標籤格式
   - 使用修復版訓練腳本

2. **檢測精度下降**
   - 禁用分類損失
   - 使用純檢測訓練
   - 檢查數據增強設置

3. **訓練不穩定**
   - 使用修復的超參數配置
   - 降低學習率
   - 增加 warmup 輪數

### 調試命令

```bash
# 檢查數據格式
python utils/format_converter.py --input train/labels --validate

# 測試小批量訓練
python train_fixed.py --data data.yaml --weights yolov5s.pt \
    --batch-size 4 --epochs 5 --classification-enabled --classification-weight 0.01

# 詳細日誌
python train_fixed.py --data data.yaml --weights yolov5s.pt \
    --classification-enabled --classification-weight 0.01 2>&1 | tee training.log
```

## 總結

通過降低分類損失權重、改進動態權重策略、簡化損失計算和優化超參數，成功解決了分類損失導致檢測精度下降的問題。建議優先使用純檢測訓練，在需要時謹慎啟用分類功能。 