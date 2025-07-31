import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from torchvision.ops import box_iou
from train_joint_ultimate import UltimateJointModel, UltimateDataset, get_ultimate_transforms
import argparse

IOU_THRESH = 0.5

# 將 [x_center, y_center, w, h] 轉為 [x1, y1, x2, y2]
def xywh2xyxy(box):
    x, y, w, h = box
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]

def get_all_predictions(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting'):
            images = batch['image'].to(device)
            bbox_targets = batch['bbox'].cpu().numpy()
            cls_targets = batch['classification'].cpu().numpy()
            
            bbox_pred, cls_pred = model(images)
            bbox_pred = bbox_pred.cpu().numpy()
            cls_pred = torch.softmax(cls_pred, dim=1).cpu().numpy()
            
            for i in range(images.size(0)):
                # 預測框
                pred_box = bbox_pred[i]
                pred_cls = np.argmax(cls_pred[i])
                pred_score = np.max(cls_pred[i])
                all_preds.append({
                    'bbox': xywh2xyxy(pred_box),
                    'class': pred_cls,
                    'score': pred_score
                })
                # 真實框
                gt_box = bbox_targets[i]
                gt_cls = np.argmax(cls_targets[i])
                all_gts.append({
                    'bbox': xywh2xyxy(gt_box),
                    'class': gt_cls
                })
    return all_preds, all_gts

def compute_ap(recall, precision):
    # VOC 2010 11-point interpolation
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.
    return ap

def eval_map(model, loader, device, class_names, iou_thresh=0.5):
    all_preds, all_gts = get_all_predictions(model, loader, device, class_names)
    aps = []
    report_lines = []
    for cls_id, cls_name in enumerate(class_names):
        # 收集該類別的預測與真實框
        preds = [p for p in all_preds if p['class'] == cls_id]
        gts = [g for g in all_gts if g['class'] == cls_id]
        npos = len(gts)
        if npos == 0:
            aps.append(0.0)
            report_lines.append(f"- {cls_name}: AP=0.0 (無真實樣本)")
            continue
        # 按分數排序
        preds = sorted(preds, key=lambda x: -x['score'])
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        detected = np.zeros(npos)
        gt_boxes = np.array([g['bbox'] for g in gts])
        for i, pred in enumerate(preds):
            pred_box = torch.tensor([pred['bbox']])
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            ious = box_iou(pred_box, torch.tensor(gt_boxes)).numpy()[0]
            max_iou = np.max(ious)
            max_idx = np.argmax(ious)
            if max_iou >= iou_thresh and detected[max_idx] == 0:
                tp[i] = 1
                detected[max_idx] = 1
            else:
                fp[i] = 1
        # 累積
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / npos
        precision = tp_cum / (tp_cum + fp_cum + 1e-16)
        ap = compute_ap(recall, precision)
        aps.append(ap)
        report_lines.append(f"- {cls_name}: AP={ap:.4f}, 樣本數={npos}")
    mAP = np.mean(aps)
    report_lines.append(f"\n**mAP@{iou_thresh}: {mAP:.4f}**")
    return mAP, aps, report_lines

def main():
    parser = argparse.ArgumentParser(description='Evaluate mAP and generate markdown report')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='joint_model_ultimate.pth', help='Path to model weights')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--output', type=str, default='detection_map_report.md', help='Markdown report output')
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

    # Eval
    print('Evaluating mAP...')
    mAP, aps, report_lines = eval_map(model, test_loader, device, class_names, IOU_THRESH)

    # Markdown report
    with open(args.output, 'w') as f:
        f.write(f"# 檢測任務 mAP 報告\n\n")
        f.write(f"- 測試集樣本數: {len(test_dataset)}\n")
        f.write(f"- IoU 門檻: {IOU_THRESH}\n\n")
        for line in report_lines:
            f.write(line + '\n')
    print(f"mAP 報告已儲存到: {args.output}")
    print('\n'.join(report_lines))

if __name__ == '__main__':
    main() 