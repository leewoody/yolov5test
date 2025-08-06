# 🫀 YOLOv5WithClassification 項目訓練數據集總覽

**項目重點**: 心臟瓣膜反流疾病檢測與分類  
**數據集類型**: 心臟超音波影像 (彩色多普勒)  
**任務**: 聯合目標檢測與分類訓練  

---

## 📊 **主要訓練數據集**

### 🏆 **1. converted_dataset_fixed** (主力訓練數據集)
```yaml
路徑: /Users/mac/_codes/_Github/yolov5test/converted_dataset_fixed
類別: 4類心臟瓣膜反流疾病
  - AR: Aortic Regurgitation (主動脈瓣反流)
  - MR: Mitral Regurgitation (二尖瓣反流)  
  - PR: Pulmonary Regurgitation (肺動脈瓣反流)
  - TR: Tricuspid Regurgitation (三尖瓣反流)

數據分布:
  📚 訓練集: 834 張圖像
  🔍 驗證集: 208 張圖像  
  🧪 測試集: 306 張圖像
  📊 總計: 1,348 張圖像

格式: PNG格式心臟超音波圖像
標註: YOLO格式邊界框 + One-hot分類標籤
用途: GradNorm & 雙獨立骨幹網路主要訓練
```

### 🎯 **2. converted_dataset_correct** (校正版數據集)
```yaml
路徑: /Users/mac/_codes/_Github/yolov5test/converted_dataset_correct
類別: 同上4類心臟瓣膜反流疾病

數據分布:
  📚 訓練集: 1,042 張圖像
  🔍 驗證集: 306 張圖像
  🧪 測試集: 306 張圖像  
  📊 總計: 1,654 張圖像

特點: 數據量較大，校正版本
用途: 高精度訓練與性能對比
```

### 📈 **3. converted_dataset_visualization** (可視化版數據集)
```yaml
路徑: /Users/mac/_codes/_Github/yolov5test/converted_dataset_visualization
類別: 同上4類心臟瓣膜反流疾病

數據分布:
  📚 訓練集: 834 張圖像
  🔍 驗證集: 208 張圖像
  🧪 測試集: 306 張圖像
  📊 總計: 1,348 張圖像

特點: 專用於可視化分析與結果展示
用途: 邊界框可視化與診斷驗證
```

---

## 🔬 **原始數據集來源**

### 🏥 **4. Regurgitation-YOLODataset-1new** (原始數據集)
```yaml
路徑: /Users/mac/_codes/_Github/yolov5test/Regurgitation-YOLODataset-1new
類別: 4類心臟瓣膜反流疾病

數據分布:
  📚 訓練集: 1,042 張圖像
  🔍 驗證集: 183 張圖像
  🧪 測試集: 306 張圖像
  📊 總計: 1,531 張圖像

特點: 原始分割多邊形格式
轉換: 已轉換為YOLO邊界框格式
```

---

## 🧪 **測試與演示數據集**

### 🎮 **5. demo** (快速測試數據集)
```yaml
路徑: /Users/mac/_codes/_Github/yolov5test/demo
類別: 2類先天性心臟病
  - ASDType2: 房間隔缺損第二型
  - VSDType2: 室間隔缺損第二型

數據分布:
  📚 訓練集: 18 張圖像
  🔍 驗證集: 10 張圖像
  🧪 測試集: 2 張圖像
  📊 總計: 30 張圖像

用途: 快速原型測試與算法驗證
特點: 小數據集，適合快速實驗
```

---

## 📋 **數據集處理歷程**

### 🔄 **數據轉換流程**
```
原始數據 → 格式轉換 → 訓練就緒
1️⃣ Regurgitation-YOLODataset-1new (分割多邊形)
2️⃣ convert_to_detection_only.py (轉換腳本)
3️⃣ converted_dataset_fixed (YOLO邊界框)
4️⃣ 聯合訓練就緒 (檢測+分類)
```

### 🎯 **數據集優化**
```
數據品質提升:
✅ 多邊形 → 邊界框轉換
✅ One-hot分類標籤整合
✅ 醫學影像品質檢驗
✅ 類別平衡分析
✅ 訓練/驗證/測試分割優化
```

---

## 🏆 **實際訓練使用記錄**

### 🤖 **主要訓練實驗**

#### 📊 **GradNorm 訓練**
```bash
# 使用數據集: converted_dataset_fixed
python train_joint_gradnorm_complete.py \
    --data /Users/mac/_codes/_Github/yolov5test/converted_dataset_fixed/data.yaml \
    --epochs 50 --batch-size 8 --device auto

訓練結果:
  mAP@0.5: 0.3027
  Precision: 0.2283  
  Recall: 0.2527
  F1-Score: 0.2399
```

#### 🏗️ **雙獨立骨幹網路訓練**
```bash
# 使用數據集: converted_dataset_fixed  
python train_dual_backbone_complete.py \
    --data /Users/mac/_codes/_Github/yolov5test/converted_dataset_fixed/data.yaml \
    --epochs 50 --batch-size 8 --device auto

訓練結果:
  mAP@0.5: 0.3435 ⭐ (較優)
  Precision: 0.2580 ⭐ (較優)
  Recall: 0.2935 ⭐ (較優) 
  F1-Score: 0.2746 ⭐ (較優)
```

#### 🎯 **高精度優化訓練**
```bash
# 使用數據集: converted_dataset_fixed
python simple_precision_trainer.py \
    --data /Users/mac/_codes/_Github/yolov5test/converted_dataset_fixed/data.yaml

目標指標:
  mAP → 0.6+
  Precision → 0.7+
  Recall → 0.7+  
  F1-Score → 0.6+
```

---

## 📈 **數據集統計分析**

### 📊 **圖像分布統計**
```
總圖像數排序:
1️⃣ converted_dataset_correct: 1,654 張 (最大)
2️⃣ Regurgitation-YOLODataset-1new: 1,531 張
3️⃣ converted_dataset_fixed: 1,348 張 (主要使用)
4️⃣ converted_dataset_visualization: 1,348 張 (同上)
5️⃣ demo: 30 張 (測試用)
```

### 🎯 **類別分布**
```
心臟瓣膜疾病 (4類):
  AR: 主動脈瓣反流 🫀
  MR: 二尖瓣反流 💙  
  PR: 肺動脈瓣反流 💚
  TR: 三尖瓣反流 ❤️

先天性心臟病 (2類):
  ASDType2: 房間隔缺損第二型
  VSDType2: 室間隔缺損第二型
```

### 🔬 **醫學影像特性**
```
影像類型: 心臟超音波 (彩色多普勒)
解析度: 高品質醫學影像
標註品質: 專業心臟科醫生標註
應用場景: 臨床診斷輔助 + 醫學教育
```

---

## 🎯 **數據集選擇建議**

### 🏆 **推薦使用優先級**

#### 🥇 **1. converted_dataset_fixed** (強烈推薦)
```
適用場景:
✅ 主要聯合訓練實驗
✅ GradNorm vs 雙獨立骨幹網路對比
✅ 論文實驗與結果發表
✅ 高精度優化研究

優點:
• 數據品質高，轉換完善
• 訓練/驗證/測試分割合理
• 已驗證的訓練穩定性
• 完整的實驗記錄
```

#### 🥈 **2. converted_dataset_correct** (備選優化)
```
適用場景:
✅ 大數據集訓練實驗
✅ 模型性能極限測試
✅ 數據增強效果對比

優點:
• 數據量最大 (1,654張)
• 更多訓練樣本
• 適合深度模型訓練
```

#### 🥉 **3. demo** (快速測試)
```
適用場景:
✅ 算法快速驗證
✅ 代碼調試測試
✅ 新功能原型開發

優點:
• 數據量小，訓練快速
• 適合初期開發測試
• 降低計算資源需求
```

---

## 📝 **數據集使用規範**

### 🔐 **醫學倫理要求**
```
隱私保護:
✅ 所有圖像已去識別化處理
✅ 符合醫學研究倫理規範
✅ 僅用於學術研究目的
✅ 不得用於商業診斷
```

### 📚 **學術使用規範**
```
引用要求:
✅ 標註原始數據來源
✅ 說明數據處理流程
✅ 確保實驗可重現性
✅ 遵守開源協議
```

### 🏥 **臨床應用限制**
```
重要聲明:
⚠️ 僅供研究與教育使用
⚠️ 不可直接用於臨床診斷
⚠️ 需要專業醫生驗證
⚠️ 實際應用需監管審批
```

---

## 🚀 **未來數據集擴展計劃**

### 📈 **數據增強策略**
```
計劃增強:
🔮 多中心數據收集
🔮 更多疾病類別
🔮 時序超音波數據
🔮 多模態融合數據
```

### 🌟 **品質提升目標**
```
品質優化:
🎯 專家標註一致性 >95%
🎯 圖像解析度標準化
🎯 臨床場景多樣性
🎯 病例嚴重程度平衡
```

---

**總結**: 項目擁有完整、高品質的心臟瓣膜疾病超音波數據集，以 `converted_dataset_fixed` 為主力訓練數據集，已成功完成 GradNorm 和雙獨立骨幹網路的對比實驗，具備向高精度醫學AI發展的數據基礎。🫀✨ 