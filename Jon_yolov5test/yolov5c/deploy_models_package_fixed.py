#!/usr/bin/env python3
"""
🚀 YOLOv5WithClassification 模型部署打包工具 (修復版)
Purpose: Package final trained models for deployment
Author: YOLOv5WithClassification Project
Date: 2025-08-06
"""

import os
import shutil
import json
import zipfile
from pathlib import Path
from datetime import datetime

class ModelDeploymentPackager:
    def __init__(self):
        self.project_root = Path("/Users/mac/_codes/_Github/yolov5test")
        self.output_dir = Path("runs/deploy_package")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🚀 YOLOv5WithClassification 模型部署打包工具")
        print(f"📁 輸出目錄: {self.output_dir.absolute()}")
        
    def find_final_models(self):
        """查找最終模型"""
        print("\n🔍 搜索最終訓練模型...")
        
        final_models = {
            "gradnorm_final": None,
            "dual_backbone_final": None,
            "other_models": []
        }
        
        # 查找GradNorm最終模型
        gradnorm_path = self.project_root / "Jon_yolov5test/yolov5c/runs/fixed_gradnorm_final_50/complete_training/final_fixed_gradnorm_model.pt"
        if gradnorm_path.exists():
            final_models["gradnorm_final"] = gradnorm_path
            print(f"  ✅ 找到GradNorm最終模型: {gradnorm_path.name}")
        
        # 查找雙獨立骨幹最終模型
        dual_backbone_path = self.project_root / "Jon_yolov5test/yolov5c/runs/dual_backbone_final_50/complete_training/final_simple_dual_backbone_model.pt"
        if dual_backbone_path.exists():
            final_models["dual_backbone_final"] = dual_backbone_path
            print(f"  ✅ 找到雙獨立骨幹最終模型: {dual_backbone_path.name}")
        
        # 查找其他重要模型
        for pattern in ["**/joint_model_*.pth"]:
            for model_file in self.project_root.glob(pattern):
                if model_file.is_file() and model_file.stat().st_size > 1024 * 1024:  # > 1MB
                    final_models["other_models"].append(model_file)
        
        return final_models
    
    def create_deployment_structure(self):
        """創建部署目錄結構"""
        dirs = {
            "models": self.output_dir / "models",
            "configs": self.output_dir / "configs",
            "scripts": self.output_dir / "scripts",
            "docs": self.output_dir / "documentation",
            "samples": self.output_dir / "test_samples"
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return dirs
    
    def package_models(self, final_models, dirs):
        """打包模型文件"""
        print("\n📦 打包模型文件...")
        
        deployment_info = {
            "package_date": datetime.now().isoformat(),
            "project": "YOLOv5WithClassification",
            "version": "1.0.0",
            "models": {}
        }
        
        # GradNorm模型
        if final_models["gradnorm_final"]:
            gradnorm_dest = dirs["models"] / "gradnorm_final_model.pt"
            shutil.copy2(final_models["gradnorm_final"], gradnorm_dest)
            
            deployment_info["models"]["gradnorm"] = {
                "file": "models/gradnorm_final_model.pt",
                "type": "GradNorm Joint Training",
                "size_mb": round(gradnorm_dest.stat().st_size / 1024 / 1024, 2),
                "description": "Dynamic gradient balancing for joint detection and classification"
            }
            print(f"  ✅ GradNorm模型: {gradnorm_dest.name}")
        
        # 雙獨立骨幹模型
        if final_models["dual_backbone_final"]:
            dual_dest = dirs["models"] / "dual_backbone_final_model.pt"
            shutil.copy2(final_models["dual_backbone_final"], dual_dest)
            
            deployment_info["models"]["dual_backbone"] = {
                "file": "models/dual_backbone_final_model.pt",
                "type": "Dual Independent Backbone",
                "size_mb": round(dual_dest.stat().st_size / 1024 / 1024, 2),
                "description": "Completely separate networks for detection and classification (RECOMMENDED)"
            }
            print(f"  ✅ 雙獨立骨幹模型: {dual_dest.name}")
        
        # 其他重要模型 (只選最大的3個)
        other_models = sorted(final_models["other_models"], 
                            key=lambda x: x.stat().st_size, reverse=True)[:3]
        
        for i, model in enumerate(other_models):
            dest = dirs["models"] / f"additional_model_{i+1}.pth"
            shutil.copy2(model, dest)
            
            deployment_info["models"][f"additional_{i+1}"] = {
                "file": f"models/additional_model_{i+1}.pth",
                "type": "Additional Training Result",
                "size_mb": round(dest.stat().st_size / 1024 / 1024, 2),
                "original": str(model)
            }
            print(f"  ✅ 額外模型{i+1}: {dest.name}")
        
        return deployment_info
    
    def copy_configs(self, dirs):
        """複製配置文件"""
        print("\n⚙️ 複製配置文件...")
        
        config_files = [
            (self.project_root / "converted_dataset_fixed/data.yaml", "dataset_config.yaml"),
            (self.project_root / "Jon_yolov5test/yolov5c/models/yolov5s.yaml", "model_config.yaml")
        ]
        
        for src, dest_name in config_files:
            if src.exists():
                dest = dirs["configs"] / dest_name
                shutil.copy2(src, dest)
                print(f"  ✅ {dest_name}")
    
    def copy_scripts(self, dirs):
        """複製推理腳本"""
        print("\n🔧 複製推理腳本...")
        
        script_files = [
            "complete_test_visualization.py",
            "cardiac_bbox_visualization.py"
        ]
        
        for script_name in script_files:
            src = Path(script_name)
            if src.exists():
                dest = dirs["scripts"] / script_name
                shutil.copy2(src, dest)
                print(f"  ✅ {script_name}")
    
    def copy_samples(self, dirs):
        """複製測試樣本"""
        print("\n🧪 複製測試樣本...")
        
        test_images_dir = self.project_root / "converted_dataset_fixed/test/images"
        if test_images_dir.exists():
            samples = list(test_images_dir.glob("*.png"))[:5]
            for sample in samples:
                dest = dirs["samples"] / sample.name
                shutil.copy2(sample, dest)
            print(f"  ✅ 複製了 {len(samples)} 個測試樣本")
    
    def create_documentation(self, dirs, deployment_info):
        """創建文檔"""
        print("\n📚 創建部署文檔...")
        
        readme_content = f"""# 🫀 YOLOv5WithClassification 部署包

## 📋 項目概述
- **項目**: 心臟瓣膜反流疾病檢測與分類
- **技術**: 聯合目標檢測與圖像分類
- **打包時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 模型說明

### 🏗️ 雙獨立骨幹網路模型 ⭐ (推薦)
- **文件**: `models/dual_backbone_final_model.pt`
- **技術**: 完全獨立的檢測與分類網路
- **優勢**: 完全消除梯度干擾
- **性能**: mAP@0.5: 0.3435, Precision: 0.2580

### 🤖 GradNorm模型
- **文件**: `models/gradnorm_final_model.pt`
- **技術**: 動態梯度平衡訓練
- **優勢**: 智能權重調整
- **性能**: mAP@0.5: 0.3027, Precision: 0.2283

## 🫀 支援疾病類型
- **AR**: 主動脈瓣反流
- **MR**: 二尖瓣反流
- **PR**: 肺動脈瓣反流
- **TR**: 三尖瓣反流

## 🚀 快速開始

```python
# 使用雙獨立骨幹模型進行推理
python scripts/complete_test_visualization.py \\
    --model models/dual_backbone_final_model.pt \\
    --dataset path/to/test/data \\
    --output results/
```

## ⚠️ 重要說明
- 僅供研究與教育使用
- 不可直接用於臨床診斷
- 需要專業醫生驗證

---
**版本**: {deployment_info['version']}
**生成時間**: {deployment_info['package_date']}
"""
        
        readme_path = dirs["docs"] / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        print(f"  ✅ README.md")
        
        # 保存部署信息
        info_path = self.output_dir / "deployment_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(deployment_info, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ deployment_info.json")
    
    def create_zip_package(self):
        """創建ZIP部署包"""
        print("\n📦 創建ZIP部署包...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"YOLOv5WithClassification_Deploy_{timestamp}.zip"
        zip_path = self.output_dir.parent / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.output_dir.parent)
                    zipf.write(file_path, arcname)
        
        size_mb = round(zip_path.stat().st_size / 1024 / 1024, 2)
        print(f"✅ ZIP包創建完成: {zip_name} ({size_mb} MB)")
        
        return zip_path
    
    def run_packaging(self):
        """執行完整打包流程"""
        print("🫀 開始模型部署打包流程...")
        
        # 1. 查找模型
        final_models = self.find_final_models()
        
        # 2. 創建目錄結構
        dirs = self.create_deployment_structure()
        
        # 3. 打包模型
        deployment_info = self.package_models(final_models, dirs)
        
        # 4. 複製配置
        self.copy_configs(dirs)
        
        # 5. 複製腳本
        self.copy_scripts(dirs)
        
        # 6. 複製樣本
        self.copy_samples(dirs)
        
        # 7. 創建文檔
        self.create_documentation(dirs, deployment_info)
        
        # 8. 創建ZIP包
        zip_path = self.create_zip_package()
        
        # 9. 生成總結
        self.generate_summary(deployment_info, zip_path)
        
        return deployment_info, zip_path
    
    def generate_summary(self, deployment_info, zip_path):
        """生成部署總結"""
        total_models = len(deployment_info["models"])
        total_size = sum(model.get("size_mb", 0) for model in deployment_info["models"].values())
        
        summary = f"""
🚀 YOLOv5WithClassification 模型部署完成！

📦 部署包信息:
├── 📁 模型數量: {total_models} 個
├── 💾 總大小: {total_size:.2f} MB
├── 📂 部署目錄: {self.output_dir}
└── 📦 ZIP包: {zip_path}

🎯 主要模型:
├── 🏗️ 雙獨立骨幹網路 ⭐ (推薦使用)
├── 🤖 GradNorm動態梯度平衡
└── 📋 其他訓練結果

🏥 醫學應用: 心臟瓣膜疾病檢測 (AR, MR, PR, TR)
✅ 部署就緒: 可用於研究與教育用途

⚠️ 重要提醒: 請遵循醫學倫理，僅供研究使用
"""
        
        print(summary)
        
        # 保存到文件
        summary_path = self.output_dir / "deployment_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        return summary

def main():
    """主函數"""
    packager = ModelDeploymentPackager()
    
    try:
        deployment_info, zip_path = packager.run_packaging()
        print("\n🎉 模型部署打包成功完成！")
        return deployment_info, zip_path
    except Exception as e:
        print(f"\n❌ 打包失敗: {e}")
        raise

if __name__ == "__main__":
    main() 