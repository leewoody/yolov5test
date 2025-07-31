import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from train_joint_ultimate import UltimateJointModel, UltimateDataset, get_ultimate_transforms
import argparse

def xywh2xyxy(box, img_w=1, img_h=1):
    x, y, w, h = box
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]

def analyze_bbox_iou(model, loader, device, out_path):
    model.eval()
    ious = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Calculating IoU'):
            images = batch['image'].to(device)
            bbox_targets = batch['bbox'].cpu().numpy()
            bbox_pred, _ = model(images)
            bbox_pred = bbox_pred.cpu().numpy()
            for i in range(images.size(0)):
                gt_box = xywh2xyxy(bbox_targets[i])
                pred_box = xywh2xyxy(bbox_pred[i])
                gt_box_tensor = torch.tensor([gt_box])
                pred_box_tensor = torch.tensor([pred_box])
                iou = box_iou(pred_box_tensor, gt_box_tensor).item()
                ious.append(iou)
    ious = np.array(ious)
    # 繪製直方圖
    plt.figure(figsize=(6,4))
    plt.hist(ious, bins=20, color='skyblue', edgecolor='k', alpha=0.8)
    plt.xlabel('IoU')
    plt.ylabel('樣本數')
    plt.title('bbox IoU 分布')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'已儲存 bbox IoU 分布圖到 {out_path}')
    print(f'IoU 統計: mean={ious.mean():.4f}, std={ious.std():.4f}, <0.5比例={(ious<0.5).mean()*100:.2f}%')

def main():
    parser = argparse.ArgumentParser(description='分析 bbox IoU 分布')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='joint_model_ultimate.pth', help='Path to model weights')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--out', type=str, default='bbox_iou_hist.png', help='Output image path')
    args = parser.parse_args()

    # Load data config
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    data_dir = Path(args.data).parent
    nc = data_config['nc']

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Dataset
    test_transform = get_ultimate_transforms('val')
    test_dataset = UltimateDataset(data_dir, 'test', test_transform, nc, oversample_minority=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = UltimateJointModel(num_classes=nc)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)

    # 分析 IoU
    analyze_bbox_iou(model, test_loader, device, args.out)

if __name__ == '__main__':
    main() 