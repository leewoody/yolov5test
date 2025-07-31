import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
from collections import Counter
import argparse

# Import the ultimate model and dataset
from train_joint_ultimate import UltimateJointModel, UltimateDataset, get_ultimate_transforms

def test_ultimate_model(model, test_loader, device, class_names):
    """Test the ultimate joint model"""
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    bbox_mae_sum = 0
    
    # Per-class accuracy tracking
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}
    
    # Confusion matrix
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            bbox_targets = batch['bbox'].to(device)
            cls_targets = batch['classification'].to(device)
            
            # Forward pass
            bbox_pred, cls_pred = model(images)
            
            # Calculate bbox MAE
            bbox_mae = torch.mean(torch.abs(bbox_pred - bbox_targets)).item()
            bbox_mae_sum += bbox_mae * images.size(0)
            
            # Calculate classification accuracy
            cls_pred_softmax = F.softmax(cls_pred, dim=1)
            target_classes = torch.argmax(cls_targets, dim=1)
            pred_classes = torch.argmax(cls_pred_softmax, dim=1)
            
            # Overall accuracy
            correct = (pred_classes == target_classes).sum().item()
            correct_predictions += correct
            total_samples += images.size(0)
            
            # Per-class accuracy
            for i in range(images.size(0)):
                true_class = target_classes[i].item()
                pred_class = pred_classes[i].item()
                
                class_total[true_class] += 1
                if true_class == pred_class:
                    class_correct[true_class] += 1
                
                confusion_matrix[true_class][pred_class] += 1
    
    # Calculate metrics
    overall_accuracy = correct_predictions / total_samples
    bbox_mae = bbox_mae_sum / total_samples
    
    # Per-class accuracy
    per_class_accuracy = {}
    for class_id in range(len(class_names)):
        if class_total[class_id] > 0:
            per_class_accuracy[class_names[class_id]] = class_correct[class_id] / class_total[class_id]
        else:
            per_class_accuracy[class_names[class_id]] = 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'bbox_mae': bbox_mae,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': confusion_matrix,
        'total_samples': total_samples
    }

def print_detailed_results(results, class_names):
    """Print detailed test results"""
    print("\n" + "=" * 60)
    print("終極模型測試結果")
    print("=" * 60)
    
    print(f"總樣本數: {results['total_samples']}")
    print(f"整體分類準確率: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    print(f"檢測 MAE: {results['bbox_mae']:.6f}")
    
    print("\n各類別準確率:")
    for class_name, accuracy in results['per_class_accuracy'].items():
        print(f"  {class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n混淆矩陣:")
    print("預測 →")
    print("實際 ↓", end="")
    for class_name in class_names:
        print(f"  {class_name:>8}", end="")
    print()
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>8}", end="")
        for j in range(len(class_names)):
            print(f"  {results['confusion_matrix'][i][j]:>8}", end="")
        print()
    
    # Calculate precision, recall, F1-score
    print("\n詳細指標:")
    for i, class_name in enumerate(class_names):
        tp = results['confusion_matrix'][i][i]
        fp = results['confusion_matrix'][:, i].sum() - tp
        fn = results['confusion_matrix'][i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {class_name}:")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Test Ultimate Joint Model')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='joint_model_ultimate.pth', help='Path to model weights')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Load data configuration
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_dir = Path(args.data).parent
    nc = data_config['nc']
    class_names = data_config['names']
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用設備: {device}")
    print(f"類別數量: {nc}")
    print(f"類別名稱: {class_names}")
    
    # Create test dataset and loader
    test_transform = get_ultimate_transforms('val')
    test_dataset = UltimateDataset(data_dir, 'test', test_transform, nc, oversample_minority=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"測試樣本數: {len(test_dataset)}")
    
    # Create and load model
    model = UltimateJointModel(num_classes=nc)
    
    if Path(args.model).exists():
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"已載入模型: {args.model}")
    else:
        print(f"警告: 模型文件不存在 {args.model}")
        return
    
    model = model.to(device)
    
    # Test model
    print("開始測試終極模型...")
    results = test_ultimate_model(model, test_loader, device, class_names)
    
    # Print results
    print_detailed_results(results, class_names)
    
    # Save results
    import json
    results_dict = {
        'overall_accuracy': results['overall_accuracy'],
        'bbox_mae': results['bbox_mae'],
        'per_class_accuracy': results['per_class_accuracy'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'total_samples': results['total_samples']
    }
    
    with open('ultimate_test_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n測試結果已保存到: ultimate_test_results.json")
    
    # Check if we reached 80% accuracy
    if results['overall_accuracy'] >= 0.8:
        print(f"\n🎉 恭喜！已達到目標準確率 80%+ (實際: {results['overall_accuracy']*100:.2f}%)")
    else:
        print(f"\n⚠️  尚未達到目標準確率 80% (實際: {results['overall_accuracy']*100:.2f}%)")
        print("建議進一步優化措施:")
        print("- 增加訓練數據")
        print("- 調整模型架構")
        print("- 使用預訓練模型")
        print("- 嘗試集成學習")

if __name__ == '__main__':
    main() 