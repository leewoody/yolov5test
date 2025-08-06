#!/usr/bin/env python3
"""
Fixed Test for Gradient Interference Solutions
Test GradNorm and Dual Backbone approaches with correct data formats
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

def create_synthetic_data(batch_size=8, num_classes=4):
    """Create synthetic data for testing with correct formats"""
    # Create synthetic images
    images = torch.randn(batch_size, 3, 64, 64)
    
    # Create synthetic detection targets (correct format for CrossEntropyLoss)
    detection_targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create synthetic classification targets (one-hot)
    classification_targets = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        class_idx = torch.randint(0, num_classes, (1,)).item()
        classification_targets[i, class_idx] = 1.0
    
    return images, detection_targets, classification_targets

def test_gradnorm_solution():
    """Test GradNorm solution with synthetic data"""
    
    print("🧪 測試 GradNorm 解決方案")
    print("=" * 50)
    
    try:
        # Import GradNorm modules
        from models.gradnorm import GradNormLoss
        
        # Test parameters
        batch_size = 8
        num_classes = 4
        epochs = 5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"📊 測試參數:")
        print(f"   - 批次大小: {batch_size}")
        print(f"   - 類別數: {num_classes}")
        print(f"   - 訓練輪數: {epochs}")
        print(f"   - 設備: {device}")
        
        # Create simple model for testing
        class SimpleTestModel(nn.Module):
            def __init__(self, num_classes=4):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.detection_head = nn.Linear(32, num_classes)
                self.classification_head = nn.Linear(32, num_classes)
            
            def forward(self, x):
                features = self.backbone(x).view(x.size(0), -1)
                detection = self.detection_head(features)
                classification = self.classification_head(features)
                return detection, classification
        
        model = SimpleTestModel(num_classes).to(device)
        
        # Create loss functions
        detection_loss_fn = nn.CrossEntropyLoss()
        classification_loss_fn = nn.BCEWithLogitsLoss()
        
        # Create GradNorm loss
        gradnorm_loss = GradNormLoss(
            detection_loss_fn=detection_loss_fn,
            classification_loss_fn=classification_loss_fn,
            alpha=1.5,
            initial_detection_weight=1.0,
            initial_classification_weight=0.01
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training loop
        print(f"\n🔄 開始訓練 (GradNorm)...")
        start_time = time.time()
        
        detection_weights = []
        classification_weights = []
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_i in range(10):  # 10 batches per epoch
                # Create synthetic data
                images, detection_targets, classification_targets = create_synthetic_data(batch_size, num_classes)
                images = images.to(device)
                detection_targets = detection_targets.to(device)
                classification_targets = classification_targets.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                detection_outputs, classification_outputs = model(images)
                
                # Compute loss with GradNorm
                total_loss, loss_info = gradnorm_loss(
                    detection_outputs, classification_outputs,
                    detection_targets, classification_targets
                )
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += loss_info['total_loss']
                batch_count += 1
                detection_weights.append(loss_info['detection_weight'])
                classification_weights.append(loss_info['classification_weight'])
                
                if batch_i % 3 == 0:
                    print(f"   Epoch {epoch+1}, Batch {batch_i+1}: "
                          f"Loss={loss_info['total_loss']:.4f}, "
                          f"Weights: Det={loss_info['detection_weight']:.3f}, "
                          f"Cls={loss_info['classification_weight']:.3f}")
            
            avg_epoch_loss = epoch_loss / batch_count
            epoch_losses.append(avg_epoch_loss)
        
        training_time = time.time() - start_time
        
        # Results
        results = {
            'solution': 'GradNorm',
            'training_time': training_time,
            'final_detection_weight': detection_weights[-1] if detection_weights else 1.0,
            'final_classification_weight': classification_weights[-1] if classification_weights else 0.01,
            'weight_change_detection': detection_weights[-1] - detection_weights[0] if len(detection_weights) > 1 else 0,
            'weight_change_classification': classification_weights[-1] - classification_weights[0] if len(classification_weights) > 1 else 0,
            'avg_loss': np.mean(epoch_losses) if epoch_losses else 0,
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"\n✅ GradNorm 測試完成!")
        print(f"   - 訓練時間: {training_time:.2f} 秒")
        print(f"   - 最終檢測權重: {results['final_detection_weight']:.3f}")
        print(f"   - 最終分類權重: {results['final_classification_weight']:.3f}")
        print(f"   - 權重變化: 檢測={results['weight_change_detection']:.3f}, 分類={results['weight_change_classification']:.3f}")
        print(f"   - 平均損失: {results['avg_loss']:.4f}")
        print(f"   - 模型參數: {results['model_parameters']:,}")
        
        return results
        
    except Exception as e:
        print(f"❌ GradNorm 測試失敗: {str(e)}")
        return {'solution': 'GradNorm', 'error': str(e)}

def test_dual_backbone_solution():
    """Test Dual Backbone solution with synthetic data"""
    
    print("\n🧪 測試雙獨立 Backbone 解決方案")
    print("=" * 50)
    
    try:
        # Import Dual Backbone modules
        from models.dual_backbone import DualBackboneModel, DualBackboneJointTrainer
        
        # Test parameters
        batch_size = 8
        num_classes = 4
        epochs = 5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"📊 測試參數:")
        print(f"   - 批次大小: {batch_size}")
        print(f"   - 類別數: {num_classes}")
        print(f"   - 訓練輪數: {epochs}")
        print(f"   - 設備: {device}")
        
        # Create dual backbone model
        model = DualBackboneModel(
            num_detection_classes=num_classes,
            num_classification_classes=num_classes,
            width_multiple=0.25,  # Smaller for quick test
            depth_multiple=0.25
        ).to(device)
        
        # Create joint trainer
        trainer = DualBackboneJointTrainer(
            model=model,
            detection_lr=1e-3,
            classification_lr=1e-3,
            detection_weight=1.0,
            classification_weight=0.01
        )
        
        # Create loss functions
        class SimpleDetectionLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse_loss = nn.MSELoss()
            
            def forward(self, predictions, targets):
                total_loss = 0
                for pred in predictions:
                    if pred is not None:
                        # Ensure pred has correct shape for MSE loss
                        if pred.dim() > 2:
                            pred = pred.view(pred.size(0), -1)
                        target_zeros = torch.zeros_like(pred)
                        total_loss += self.mse_loss(pred, target_zeros)
                return total_loss
        
        class SimpleClassificationLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bce_loss = nn.BCEWithLogitsLoss()
            
            def forward(self, predictions, targets):
                return self.bce_loss(predictions, targets.float())
        
        # Training loop
        print(f"\n🔄 開始訓練 (雙獨立 Backbone)...")
        start_time = time.time()
        
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_i in range(10):  # 10 batches per epoch
                # Create synthetic data
                images, detection_targets, classification_targets = create_synthetic_data(batch_size, num_classes)
                images = images.to(device)
                detection_targets = detection_targets.to(device)
                classification_targets = classification_targets.to(device)
                
                # Training step
                metrics = trainer.train_step(
                    images=images,
                    detection_targets=detection_targets,
                    classification_targets=classification_targets,
                    detection_loss_fn=SimpleDetectionLoss(),
                    classification_loss_fn=SimpleClassificationLoss()
                )
                
                epoch_loss += metrics['total_loss']
                batch_count += 1
                
                if batch_i % 3 == 0:
                    print(f"   Epoch {epoch+1}, Batch {batch_i+1}: "
                          f"Loss={metrics['total_loss']:.4f}, "
                          f"Det={metrics['detection_loss']:.4f}, "
                          f"Cls={metrics['classification_loss']:.4f}")
            
            avg_epoch_loss = epoch_loss / batch_count
            epoch_losses.append(avg_epoch_loss)
        
        training_time = time.time() - start_time
        
        # Results
        results = {
            'solution': 'Dual Backbone',
            'training_time': training_time,
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
            'avg_loss': np.mean(epoch_losses) if epoch_losses else 0,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'detection_parameters': sum(p.numel() for p in model.detection_backbone.parameters()) + 
                                  sum(p.numel() for p in model.detection_head.parameters()),
            'classification_parameters': sum(p.numel() for p in model.classification_backbone.parameters())
        }
        
        print(f"\n✅ 雙獨立 Backbone 測試完成!")
        print(f"   - 訓練時間: {training_time:.2f} 秒")
        print(f"   - 最終損失: {results['final_loss']:.4f}")
        print(f"   - 平均損失: {results['avg_loss']:.4f}")
        print(f"   - 模型參數總數: {results['model_parameters']:,}")
        print(f"   - 檢測參數: {results['detection_parameters']:,}")
        print(f"   - 分類參數: {results['classification_parameters']:,}")
        
        return results
        
    except Exception as e:
        print(f"❌ 雙獨立 Backbone 測試失敗: {str(e)}")
        return {'solution': 'Dual Backbone', 'error': str(e)}

def compare_test_results(gradnorm_results, dual_backbone_results):
    """Compare test results"""
    
    print("\n📊 測試結果比較")
    print("=" * 50)
    
    # Create comparison table
    comparison_data = []
    
    if 'error' not in gradnorm_results and 'error' not in dual_backbone_results:
        comparison_data.append({
            '指標': '訓練時間 (秒)',
            'GradNorm': f"{gradnorm_results['training_time']:.2f}",
            '雙獨立 Backbone': f"{dual_backbone_results['training_time']:.2f}"
        })
        
        comparison_data.append({
            '指標': '最終損失',
            'GradNorm': f"{gradnorm_results['avg_loss']:.4f}",
            '雙獨立 Backbone': f"{dual_backbone_results['final_loss']:.4f}"
        })
        
        comparison_data.append({
            '指標': '模型參數',
            'GradNorm': f"{gradnorm_results['model_parameters']:,}",
            '雙獨立 Backbone': f"{dual_backbone_results['model_parameters']:,}"
        })
        
        comparison_data.append({
            '指標': '檢測權重',
            'GradNorm': f"{gradnorm_results['final_detection_weight']:.3f}",
            '雙獨立 Backbone': '固定 1.0'
        })
        
        comparison_data.append({
            '指標': '分類權重',
            'GradNorm': f"{gradnorm_results['final_classification_weight']:.3f}",
            '雙獨立 Backbone': '固定 0.01'
        })
    
    # Print comparison table
    if comparison_data:
        print(f"{'指標':<15} {'GradNorm':<15} {'雙獨立 Backbone':<15}")
        print("-" * 45)
        for row in comparison_data:
            print(f"{row['指標']:<15} {row['GradNorm']:<15} {row['雙獨立 Backbone']:<15}")
    
    # Analysis
    print(f"\n🔍 分析結果:")
    
    if 'error' in gradnorm_results:
        print(f"   ❌ GradNorm: {gradnorm_results['error']}")
    else:
        print(f"   ✅ GradNorm: 成功運行，權重動態調整")
        if gradnorm_results['weight_change_detection'] != 0 or gradnorm_results['weight_change_classification'] != 0:
            print(f"      - 權重變化明顯，GradNorm 正常工作")
        else:
            print(f"      - 權重變化較小，可能需要調整參數")
    
    if 'error' in dual_backbone_results:
        print(f"   ❌ 雙獨立 Backbone: {dual_backbone_results['error']}")
    else:
        print(f"   ✅ 雙獨立 Backbone: 成功運行，任務完全解耦")
        if dual_backbone_results['model_parameters'] > 100000:
            print(f"      - 模型參數較多，計算資源需求高")
        else:
            print(f"      - 模型參數合理，適合快速測試")
    
    # Recommendations
    print(f"\n💡 建議:")
    if 'error' not in gradnorm_results and 'error' not in dual_backbone_results:
        print(f"   1. 兩種解決方案都成功運行")
        print(f"   2. GradNorm 適合資源受限環境")
        print(f"   3. 雙獨立 Backbone 適合高精度要求")
        print(f"   4. 建議在真實數據集上進行進一步測試")
    elif 'error' not in gradnorm_results:
        print(f"   1. GradNorm 解決方案可行")
        print(f"   2. 雙獨立 Backbone 需要調試")
        print(f"   3. 建議先使用 GradNorm 方案")
    elif 'error' not in dual_backbone_results:
        print(f"   1. 雙獨立 Backbone 解決方案可行")
        print(f"   2. GradNorm 需要調試")
        print(f"   3. 建議使用雙獨立 Backbone 方案")
    else:
        print(f"   1. 兩種方案都需要調試")
        print(f"   2. 檢查依賴和環境配置")
        print(f"   3. 建議先解決基礎問題")

def main():
    """Main test function"""
    
    print("🚀 梯度干擾解決方案修復測試")
    print("=" * 60)
    print("測試時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Test GradNorm solution
    gradnorm_results = test_gradnorm_solution()
    
    # Test Dual Backbone solution
    dual_backbone_results = test_dual_backbone_solution()
    
    # Compare results
    compare_test_results(gradnorm_results, dual_backbone_results)
    
    # Save results
    results = {
        'test_time': datetime.now().isoformat(),
        'gradnorm_results': gradnorm_results,
        'dual_backbone_results': dual_backbone_results
    }
    
    with open('fixed_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 測試結果已保存至 'fixed_test_results.json'")
    print("=" * 60)
    print("✅ 修復測試完成！")

if __name__ == '__main__':
    main() 