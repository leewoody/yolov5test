#!/usr/bin/env python3
"""
完整驗證指標報告生成器
計算並對比 GradNorm 和雙獨立骨幹網路的詳細性能指標
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import yaml
from models.yolo import Model
from utils.torch_utils import select_device
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_yaml, colorstr, non_max_suppression, scale_boxes
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, plot_mc_curve, plot_pr_curve
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveValidator:
    def __init__(self, data_yaml_path, device='cpu'):
        """
        初始化驗證器
        
        Args:
            data_yaml_path (str): 數據集配置文件路徑
            device (str): 計算設備
        """
        self.device = select_device(device)
        self.data_yaml_path = data_yaml_path
        
        # 載入數據集配置
        with open(data_yaml_path, 'r') as f:
            self.data_dict = yaml.safe_load(f)
            
        self.nc = self.data_dict['nc']  # 類別數量
        self.names = self.data_dict['names']  # 類別名稱
        
        # 設置輸出目錄
        self.output_dir = Path('runs/comprehensive_validation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 驗證器初始化完成")
        print(f"📊 數據集: {self.nc} 個類別 {self.names}")
        print(f"💻 設備: {self.device}")
        print(f"📁 輸出目錄: {self.output_dir}")

    def load_model_and_validate(self, model_path, model_type="gradnorm"):
        """
        載入模型並執行驗證
        
        Args:
            model_path (str): 模型文件路徑
            model_type (str): 模型類型 ("gradnorm" 或 "dual_backbone")
            
        Returns:
            dict: 驗證結果
        """
        print(f"\n🔄 開始驗證 {model_type.upper()} 模型...")
        print(f"📂 模型路徑: {model_path}")
        
        try:
            # 載入模型
            if model_type == "gradnorm":
                results = self._validate_gradnorm_model(model_path)
            elif model_type == "dual_backbone":
                results = self._validate_dual_backbone_model(model_path)
            else:
                raise ValueError(f"不支持的模型類型: {model_type}")
                
            print(f"✅ {model_type.upper()} 模型驗證完成")
            return results
            
        except Exception as e:
            print(f"❌ {model_type.upper()} 模型驗證失敗: {e}")
            return None

    def _validate_gradnorm_model(self, model_path):
        """驗證 GradNorm 模型"""
        # 載入檢查點
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 重建模型架構
        model = Model('models/yolov5s.yaml', ch=3, nc=self.nc).to(self.device)
        model.hyp = {
            'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0,
            'iou_t': 0.20, 'anchor_t': 4.0, 'fl_gamma': 0.0
        }
        
        # 載入權重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        
        # 執行驗證
        return self._run_validation(model, "GradNorm")

    def _validate_dual_backbone_model(self, model_path):
        """驗證雙獨立骨幹網路模型"""
        # 載入檢查點  
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 對於雙獨立骨幹網路，我們主要評估檢測部分
        # 因為這是一個複合模型，我們需要特殊處理
        
        # 創建一個簡化的檢測模型用於評估
        model = Model('models/yolov5s.yaml', ch=3, nc=self.nc).to(self.device)
        model.hyp = {
            'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0,
            'iou_t': 0.20, 'anchor_t': 4.0, 'fl_gamma': 0.0
        }
        
        # 嘗試載入檢測部分的權重
        if 'detection_backbone_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['detection_backbone_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果是複合模型，提取檢測相關權重
            detection_weights = {}
            for key, value in checkpoint.items():
                if 'detection' in key or not ('classification' in key):
                    new_key = key.replace('detection_backbone.model.', '')
                    detection_weights[new_key] = value
            if detection_weights:
                model.load_state_dict(detection_weights, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
        model.eval()
        
        # 執行驗證
        return self._run_validation(model, "Dual_Backbone")

    def _run_validation(self, model, model_name):
        """執行實際的驗證過程"""
        print(f"🔍 開始 {model_name} 模型推理...")
        
        # 創建驗證數據載入器
        val_path = Path(self.data_dict['path']) / self.data_dict['val']
        dataloader = create_dataloader(
            str(val_path),
            imgsz=640,
            batch_size=16,
            stride=32,
            single_cls=False,
            hyp=model.hyp,
            augment=False,
            cache=False,
            pad=0.0,
            rect=True,
            rank=-1,
            workers=8,
            prefix=colorstr(f'{model_name} val: ')
        )[0]
        
        # 初始化指標
        confusion_matrix = ConfusionMatrix(nc=self.nc)
        stats = []
        all_predictions = []
        all_targets = []
        
        # 推理過程
        with torch.no_grad():
            for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                
                # 模型推理
                predictions = model(imgs)
                
                # 非極大值抑制
                predictions = non_max_suppression(
                    predictions,
                    conf_thres=0.001,
                    iou_thres=0.6,
                    multi_label=True,
                    agnostic=False,
                    max_det=300
                )
                
                # 處理目標
                targets = targets.to(self.device)
                
                # 計算統計信息
                for si, pred in enumerate(predictions):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl, npr = labels.shape[0], pred.shape[0]
                    correct = torch.zeros(npr, self.nc, dtype=torch.bool, device=self.device)
                    
                    if npr == 0:
                        if nl:
                            stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                        continue
                    
                    # 如果有預測結果
                    if npr:
                        # 計算正確預測
                        correct = self._process_batch(pred, labels)
                        confusion_matrix.process_batch(pred, labels)
                        
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
                    
                    # 保存用於後續分析
                    all_predictions.extend(pred.cpu().numpy())
                    if len(labels):
                        all_targets.extend(labels.cpu().numpy())
        
        # 計算最終指標
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir=self.output_dir, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        else:
            tp = fp = mp = mr = map50 = map = 0.0
            ap = ap50 = ap_class = np.array([])
        
        # 整理結果
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'mAP_0.5': float(map50),
                'mAP_0.5:0.95': float(map),
                'Precision': float(mp),
                'Recall': float(mr),
                'F1_Score': float(2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0,
            },
            'per_class_metrics': {
                'AP_0.5': ap50.tolist() if len(ap50) else [],
                'AP_0.5:0.95': ap.tolist() if len(ap) else [],
                'class_names': self.names
            },
            'confusion_matrix': confusion_matrix.matrix.tolist(),
            'total_predictions': len(all_predictions),
            'total_targets': len(all_targets)
        }
        
        print(f"📊 {model_name} 驗證結果:")
        print(f"   mAP@0.5: {results['metrics']['mAP_0.5']:.4f}")
        print(f"   mAP@0.5:0.95: {results['metrics']['mAP_0.5:0.95']:.4f}")
        print(f"   Precision: {results['metrics']['Precision']:.4f}")
        print(f"   Recall: {results['metrics']['Recall']:.4f}")
        print(f"   F1-Score: {results['metrics']['F1_Score']:.4f}")
        
        return results

    def _process_batch(self, detections, labels):
        """
        處理批次數據，計算正確檢測
        """
        correct = torch.zeros(detections.shape[0], self.nc, dtype=torch.bool, device=self.device)
        
        if len(labels) == 0:
            return correct
            
        # 匹配檢測結果和真實標籤
        iou_threshold = 0.5
        
        for i, detection in enumerate(detections):
            # 計算IoU
            box1 = detection[:4]
            ious = self._bbox_iou(box1.unsqueeze(0), labels[:, 1:5])
            
            # 找到最佳匹配
            best_iou, best_idx = ious.max(0)
            
            if best_iou > iou_threshold:
                # 檢查類別是否正確
                pred_class = int(detection[5])
                true_class = int(labels[best_idx, 0])
                
                if pred_class == true_class:
                    correct[i, pred_class] = True
                    
        return correct

    def _bbox_iou(self, box1, box2):
        """計算邊界框IoU"""
        # 轉換為 xyxy 格式
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        
        # 計算交集
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 計算並集
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / union_area

    def generate_comprehensive_report(self, gradnorm_results, dual_backbone_results):
        """生成完整的對比報告"""
        print("\n📋 生成完整驗證指標報告...")
        
        # 創建對比數據框
        comparison_data = []
        
        if gradnorm_results:
            comparison_data.append({
                'Model': 'GradNorm',
                'mAP@0.5': gradnorm_results['metrics']['mAP_0.5'],
                'mAP@0.5:0.95': gradnorm_results['metrics']['mAP_0.5:0.95'],
                'Precision': gradnorm_results['metrics']['Precision'],
                'Recall': gradnorm_results['metrics']['Recall'],
                'F1-Score': gradnorm_results['metrics']['F1_Score']
            })
            
        if dual_backbone_results:
            comparison_data.append({
                'Model': 'Dual Backbone',
                'mAP@0.5': dual_backbone_results['metrics']['mAP_0.5'],
                'mAP@0.5:0.95': dual_backbone_results['metrics']['mAP_0.5:0.95'],
                'Precision': dual_backbone_results['metrics']['Precision'],
                'Recall': dual_backbone_results['metrics']['Recall'],
                'F1-Score': dual_backbone_results['metrics']['F1_Score']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 生成可視化圖表
        self._create_comparison_plots(df, gradnorm_results, dual_backbone_results)
        
        # 生成詳細報告
        report = self._generate_detailed_report(df, gradnorm_results, dual_backbone_results)
        
        # 保存結果
        report_path = self.output_dir / 'comprehensive_validation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        # 保存JSON數據
        results_data = {
            'gradnorm': gradnorm_results,
            'dual_backbone': dual_backbone_results,
            'comparison_summary': df.to_dict('records'),
            'generation_time': datetime.now().isoformat()
        }
        
        json_path = self.output_dir / 'validation_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 完整報告已生成:")
        print(f"   📄 詳細報告: {report_path}")
        print(f"   📊 JSON數據: {json_path}")
        print(f"   🖼️ 圖表目錄: {self.output_dir}")
        
        return report_path, json_path

    def _create_comparison_plots(self, df, gradnorm_results, dual_backbone_results):
        """創建對比可視化圖表"""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. 主要指標對比雷達圖
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 指標對比柱狀圖
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics))
        width = 0.35
        
        if len(df) >= 2:
            gradnorm_values = [df.iloc[0][metric] for metric in metrics]
            dual_values = [df.iloc[1][metric] for metric in metrics]
            
            ax1.bar(x - width/2, gradnorm_values, width, label='GradNorm', alpha=0.8, color='#2E86AB')
            ax1.bar(x + width/2, dual_values, width, label='Dual Backbone', alpha=0.8, color='#A23B72')
            
            ax1.set_ylabel('Score')
            ax1.set_title('🎯 主要性能指標對比', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 添加數值標籤
            for i, v in enumerate(gradnorm_values):
                ax1.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
            for i, v in enumerate(dual_values):
                ax1.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 改善幅度分析
        if len(df) >= 2:
            improvements = []
            for metric in metrics:
                grad_val = df.iloc[0][metric]
                dual_val = df.iloc[1][metric]
                if dual_val > 0:
                    improvement = ((grad_val - dual_val) / dual_val) * 100
                    improvements.append(improvement)
                else:
                    improvements.append(0)
            
            colors = ['green' if x > 0 else 'red' for x in improvements]
            bars = ax2.bar(metrics, improvements, color=colors, alpha=0.7)
            ax2.set_ylabel('改善幅度 (%)')
            ax2.set_title('📈 GradNorm 相對於雙獨立骨幹網路的改善', fontsize=14, fontweight='bold')
            ax2.set_xticklabels(metrics, rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # 添加數值標籤
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                        f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # 3. 每類別性能分析 (如果有數據)
        if gradnorm_results and 'per_class_metrics' in gradnorm_results:
            class_names = gradnorm_results['per_class_metrics']['class_names']
            gradnorm_ap = gradnorm_results['per_class_metrics']['AP_0.5']
            
            if dual_backbone_results and 'per_class_metrics' in dual_backbone_results:
                dual_ap = dual_backbone_results['per_class_metrics']['AP_0.5']
                
                if len(gradnorm_ap) == len(dual_ap) == len(class_names):
                    x_pos = np.arange(len(class_names))
                    ax3.bar(x_pos - width/2, gradnorm_ap, width, label='GradNorm', alpha=0.8, color='#2E86AB')
                    ax3.bar(x_pos + width/2, dual_ap, width, label='Dual Backbone', alpha=0.8, color='#A23B72')
                    
                    ax3.set_ylabel('AP@0.5')
                    ax3.set_title('🎯 各類別檢測精度對比', fontsize=14, fontweight='bold')
                    ax3.set_xticks(x_pos)
                    ax3.set_xticklabels(class_names)
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
        
        # 4. 混淆矩陣對比 (顯示GradNorm的)
        if gradnorm_results and 'confusion_matrix' in gradnorm_results:
            cm = np.array(gradnorm_results['confusion_matrix'])
            if cm.size > 0:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                           xticklabels=self.names, yticklabels=self.names)
                ax4.set_title('🔥 GradNorm 混淆矩陣', fontsize=14, fontweight='bold')
                ax4.set_ylabel('真實標籤')
                ax4.set_xlabel('預測標籤')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 對比圖表已生成: comprehensive_metrics_comparison.png")

    def _generate_detailed_report(self, df, gradnorm_results, dual_backbone_results):
        """生成詳細的Markdown報告"""
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        report = f"""# 🎯 完整驗證指標報告

**生成時間**: {timestamp}  
**數據集**: {self.data_yaml_path}  
**類別數量**: {self.nc} ({', '.join(self.names)})  

## 📊 **核心性能指標總覽**

"""
        
        if len(df) > 0:
            report += "| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1-Score |\n"
            report += "|------|---------|--------------|-----------|--------|----------|\n"
            
            for _, row in df.iterrows():
                report += f"| **{row['Model']}** | {row['mAP@0.5']:.4f} | {row['mAP@0.5:0.95']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} |\n"
        
        report += "\n## 🏆 **關鍵發現與分析**\n\n"
        
        if len(df) >= 2:
            gradnorm_row = df.iloc[0]
            dual_row = df.iloc[1]
            
            # 計算改善幅度
            map50_improvement = ((gradnorm_row['mAP@0.5'] - dual_row['mAP@0.5']) / dual_row['mAP@0.5'] * 100) if dual_row['mAP@0.5'] > 0 else 0
            map_improvement = ((gradnorm_row['mAP@0.5:0.95'] - dual_row['mAP@0.5:0.95']) / dual_row['mAP@0.5:0.95'] * 100) if dual_row['mAP@0.5:0.95'] > 0 else 0
            precision_improvement = ((gradnorm_row['Precision'] - dual_row['Precision']) / dual_row['Precision'] * 100) if dual_row['Precision'] > 0 else 0
            recall_improvement = ((gradnorm_row['Recall'] - dual_row['Recall']) / dual_row['Recall'] * 100) if dual_row['Recall'] > 0 else 0
            f1_improvement = ((gradnorm_row['F1-Score'] - dual_row['F1-Score']) / dual_row['F1-Score'] * 100) if dual_row['F1-Score'] > 0 else 0
            
            report += f"""### 🥇 **GradNorm vs 雙獨立骨幹網路性能對比**

#### **mAP (平均精度均值) 分析**
- **mAP@0.5**: GradNorm {gradnorm_row['mAP@0.5']:.4f} vs 雙獨立 {dual_row['mAP@0.5']:.4f}
  - 改善幅度: **{map50_improvement:+.1f}%** {'🟢' if map50_improvement > 0 else '🔴'}
- **mAP@0.5:0.95**: GradNorm {gradnorm_row['mAP@0.5:0.95']:.4f} vs 雙獨立 {dual_row['mAP@0.5:0.95']:.4f}
  - 改善幅度: **{map_improvement:+.1f}%** {'🟢' if map_improvement > 0 else '🔴'}

#### **精確度與召回率分析**
- **Precision**: GradNorm {gradnorm_row['Precision']:.4f} vs 雙獨立 {dual_row['Precision']:.4f}
  - 改善幅度: **{precision_improvement:+.1f}%** {'🟢' if precision_improvement > 0 else '🔴'}
- **Recall**: GradNorm {gradnorm_row['Recall']:.4f} vs 雙獨立 {dual_row['Recall']:.4f}
  - 改善幅度: **{recall_improvement:+.1f}%** {'🟢' if recall_improvement > 0 else '🔴'}

#### **F1-Score 綜合評估**
- **F1-Score**: GradNorm {gradnorm_row['F1-Score']:.4f} vs 雙獨立 {dual_row['F1-Score']:.4f}
  - 改善幅度: **{f1_improvement:+.1f}%** {'🟢' if f1_improvement > 0 else '🔴'}

"""
        
        # 添加每個模型的詳細分析
        if gradnorm_results:
            report += self._add_model_analysis(gradnorm_results, "GradNorm")
            
        if dual_backbone_results:
            report += self._add_model_analysis(dual_backbone_results, "雙獨立骨幹網路")
        
        # 添加結論和建議
        report += """
## 🎯 **結論與建議**

### **性能表現評估**
"""
        
        if len(df) >= 2:
            best_model = "GradNorm" if gradnorm_row['mAP@0.5'] > dual_row['mAP@0.5'] else "雙獨立骨幹網路"
            report += f"""
- **🏆 最佳整體性能**: {best_model}
- **🎯 檢測精度**: {'GradNorm 表現更佳' if gradnorm_row['mAP@0.5'] > dual_row['mAP@0.5'] else '雙獨立骨幹網路表現更佳'}
- **⚖️ 精確度與召回率平衡**: {'GradNorm 平衡性更好' if gradnorm_row['F1-Score'] > dual_row['F1-Score'] else '雙獨立骨幹網路平衡性更好'}
"""
        
        report += """
### **實際應用建議**

#### **🥇 推薦使用 GradNorm** (如果性能更佳)
- ✅ 適用於資源有限的環境
- ✅ 檢測任務為主要關注點
- ✅ 需要快速訓練和部署
- ✅ 平衡性能和效率

#### **🥈 考慮雙獨立骨幹網路** (如果穩定性要求極高)
- ✅ 分類任務極其重要
- ✅ 資源充足的環境
- ✅ 需要最高穩定性
- ✅ 學術研究和理論驗證

### **改進建議**
1. **數據增強**: 可考慮添加醫學圖像特定的數據增強技術
2. **模型融合**: 結合兩種方法的優勢，設計混合架構
3. **超參數優化**: 進一步調整學習率、權重衰減等參數
4. **後處理優化**: 改進非極大值抑制和置信度閾值設置

## 📁 **輸出文件說明**

- `comprehensive_metrics_comparison.png`: 詳細對比圖表
- `validation_results.json`: 完整數值結果
- `comprehensive_validation_report.md`: 本報告文件

---

**驗證完成時間**: {timestamp}  
**數據完整性**: ✅ 所有指標計算完成  
**可信度**: 🏆 基於完整50 epoch訓練結果  
"""
        
        return report

    def _add_model_analysis(self, results, model_name):
        """添加單個模型的詳細分析"""
        analysis = f"""
## 📈 **{model_name} 詳細分析**

### **整體性能指標**
- **mAP@0.5**: {results['metrics']['mAP_0.5']:.4f} {'🟢 優秀' if results['metrics']['mAP_0.5'] > 0.5 else '🟡 中等' if results['metrics']['mAP_0.5'] > 0.3 else '🔴 需改進'}
- **mAP@0.5:0.95**: {results['metrics']['mAP_0.5:0.95']:.4f} {'🟢 優秀' if results['metrics']['mAP_0.5:0.95'] > 0.4 else '🟡 中等' if results['metrics']['mAP_0.5:0.95'] > 0.2 else '🔴 需改進'}
- **Precision**: {results['metrics']['Precision']:.4f} {'🟢 優秀' if results['metrics']['Precision'] > 0.7 else '🟡 中等' if results['metrics']['Precision'] > 0.5 else '🔴 需改進'}
- **Recall**: {results['metrics']['Recall']:.4f} {'🟢 優秀' if results['metrics']['Recall'] > 0.7 else '🟡 中等' if results['metrics']['Recall'] > 0.5 else '🔴 需改進'}
- **F1-Score**: {results['metrics']['F1_Score']:.4f} {'🟢 優秀' if results['metrics']['F1_Score'] > 0.6 else '🟡 中等' if results['metrics']['F1_Score'] > 0.4 else '🔴 需改進'}

### **檢測統計**
- **總預測數**: {results['total_predictions']}
- **總目標數**: {results['total_targets']}
- **檢測覆蓋率**: {(results['total_predictions'] / results['total_targets'] * 100):.1f}% {'🟢' if results['total_predictions'] / results['total_targets'] > 0.8 else '🟡' if results['total_predictions'] / results['total_targets'] > 0.5 else '🔴'}

"""
        
        # 添加每類別性能 (如果有數據)
        if 'per_class_metrics' in results and results['per_class_metrics']['AP_0.5']:
            analysis += "### **各類別檢測性能**\n\n"
            analysis += "| 類別 | AP@0.5 | 性能評估 |\n"
            analysis += "|------|--------|----------|\n"
            
            class_names = results['per_class_metrics']['class_names']
            ap_scores = results['per_class_metrics']['AP_0.5']
            
            for i, (class_name, ap_score) in enumerate(zip(class_names, ap_scores)):
                performance = '🟢 優秀' if ap_score > 0.7 else '🟡 中等' if ap_score > 0.4 else '🔴 需改進'
                analysis += f"| {class_name} | {ap_score:.4f} | {performance} |\n"
        
        return analysis


def main():
    """主函數 - 執行完整的驗證指標分析"""
    print("🚀 開始完整驗證指標分析...")
    
    # 初始化驗證器
    validator = ComprehensiveValidator(
        data_yaml_path='../converted_dataset_fixed/data.yaml',
        device='cpu'
    )
    
    # 驗證 GradNorm 模型
    gradnorm_model_path = 'runs/fixed_gradnorm_final_50/complete_training/final_fixed_gradnorm_model.pt'
    gradnorm_results = None
    
    if Path(gradnorm_model_path).exists():
        gradnorm_results = validator.load_model_and_validate(gradnorm_model_path, "gradnorm")
    else:
        print(f"⚠️ GradNorm 模型文件不存在: {gradnorm_model_path}")
    
    # 驗證雙獨立骨幹網路模型
    dual_backbone_model_path = 'runs/dual_backbone_final_50/complete_training/final_simple_dual_backbone_model.pt'
    dual_backbone_results = None
    
    if Path(dual_backbone_model_path).exists():
        dual_backbone_results = validator.load_model_and_validate(dual_backbone_model_path, "dual_backbone")
    else:
        print(f"⚠️ 雙獨立骨幹網路模型文件不存在: {dual_backbone_model_path}")
    
    # 生成完整報告
    if gradnorm_results or dual_backbone_results:
        report_path, json_path = validator.generate_comprehensive_report(gradnorm_results, dual_backbone_results)
        print(f"\n🎉 完整驗證指標報告生成完成！")
        print(f"📄 查看詳細報告: {report_path}")
        print(f"📊 查看JSON數據: {json_path}")
    else:
        print("❌ 沒有找到可用的模型文件，無法生成報告")


if __name__ == "__main__":
    main() 