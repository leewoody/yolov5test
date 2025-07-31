import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from train_joint_ultimate import UltimateJointModel, UltimateDataset, get_ultimate_transforms

import argparse

def xywh2xyxy(box, img_w, img_h):
    x, y, w, h = box
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]

def analyze_and_visualize_bbox(model, loader, device, class_names, out_dir, num_samples=20):
    model.eval()
    pred_centers = []
    gt_centers = []
    pred_wh = []
    gt_wh = []
    vis_indices = random.sample(range(len(loader.dataset)), min(num_samples, len(loader.dataset)))
    vis_count = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc='Analyzing')):
            images = batch['image'].to(device)
            bbox_targets = batch['bbox'].cpu().numpy()
            cls_targets = batch['classification'].cpu().numpy()
            img_batch = batch['image'].cpu().numpy()
            
            bbox_pred, cls_pred = model(images)
            bbox_pred = bbox_pred.cpu().numpy()
            cls_pred = torch.softmax(cls_pred, dim=1).cpu().numpy()
            
            for i in range(images.size(0)):
                gt_box = bbox_targets[i]
                pred_box = bbox_pred[i]
                pred_centers.append(pred_box[:2])
                gt_centers.append(gt_box[:2])
                pred_wh.append(pred_box[2:])
                gt_wh.append(gt_box[2:])
                
                # 可視化部分樣本
                global_idx = idx * loader.batch_size + i
                if global_idx in vis_indices and vis_count < num_samples:
                    img = img_batch[i].transpose(1,2,0)
                    img = (img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
                    img = np.clip(img, 0, 1)
                    h, w = img.shape[:2]
                    gt_xyxy = xywh2xyxy(gt_box, w, h)
                    pred_xyxy = xywh2xyxy(pred_box, w, h)
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(img)
                    # GT bbox (green)
                    ax.add_patch(plt.Rectangle((gt_xyxy[0], gt_xyxy[1]), gt_xyxy[2]-gt_xyxy[0], gt_xyxy[3]-gt_xyxy[1],
                                               fill=False, edgecolor='g', linewidth=2, label='GT'))
                    # Pred bbox (red)
                    ax.add_patch(plt.Rectangle((pred_xyxy[0], pred_xyxy[1]), pred_xyxy[2]-pred_xyxy[0], pred_xyxy[3]-pred_xyxy[1],
                                               fill=False, edgecolor='r', linewidth=2, label='Pred'))
                    ax.set_title(f"GT: {class_names[np.argmax(cls_targets[i])]}, Pred: {class_names[np.argmax(cls_pred[i])]}")
                    ax.legend(handles=[plt.Line2D([0],[0],color='g',lw=2,label='GT'), plt.Line2D([0],[0],color='r',lw=2,label='Pred')])
                    plt.axis('off')
                    plt.tight_layout()
                    save_path = Path(out_dir) / f'bbox_vis_{vis_count}.png'
                    plt.savefig(save_path, dpi=150)
                    plt.close()
                    vis_count += 1
    pred_centers = np.array(pred_centers)
    gt_centers = np.array(gt_centers)
    pred_wh = np.array(pred_wh)
    gt_wh = np.array(gt_wh)
    # 分布圖
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.scatter(gt_centers[:,0], gt_centers[:,1], c='g', alpha=0.5, label='GT')
    plt.scatter(pred_centers[:,0], pred_centers[:,1], c='r', alpha=0.5, label='Pred')
    plt.xlabel('x_center')
    plt.ylabel('y_center')
    plt.title('中心點分布')
    plt.legend()
    plt.subplot(1,2,2)
    plt.scatter(gt_wh[:,0], gt_wh[:,1], c='g', alpha=0.5, label='GT')
    plt.scatter(pred_wh[:,0], pred_wh[:,1], c='r', alpha=0.5, label='Pred')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.title('寬高分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(out_dir)/'bbox_distribution.png'), dpi=150)
    plt.close()
    print(f'已儲存 bbox 分布圖與可視化到 {out_dir}')

def main():
    parser = argparse.ArgumentParser(description='分析並可視化 bbox 預測與標註分布')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='joint_model_ultimate.pth', help='Path to model weights')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--out', type=str, default='bbox_vis', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of images to visualize')
    args = parser.parse_args()

    # Load data config
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    data_dir = Path(args.data).parent
    nc = data_config['nc']
    class_names = data_config['names']

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

    # Output dir
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 分析與可視化
    analyze_and_visualize_bbox(model, test_loader, device, class_names, out_dir, args.num_samples)

if __name__ == '__main__':
    main() 