#!/usr/bin/env python3
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Medical Image Training Script for YOLOv5
Optimized for heart ultrasound and medical imaging applications
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

try:
    import comet_ml
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import val as validate
from models.experimental import attempt_load
from models.yolo import Model, ClassificationModel2, Detect
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first, smartCrossEntropyLoss)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):
    """
    Medical image training function with enhanced stability and monitoring
    """
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('Medical Image Hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if loggers.clearml:
            data_dict = loggers.clearml.data_dict
        if loggers.wb and loggers.wb.wandb_run_id is None:
            loggers.wb.log_artifact(str(weights)) if weights else None

    # Register actions
    for k in methods(loggers):
        callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve and not opt.noplots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data, autodownload=True)

    # Model
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=data_dict['nc'], anchors=None).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not opt.resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
    else:
        model = Model(cfg, ch=3, nc=data_dict['nc'], anchors=hyp.get('anchors')).to(device)

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Batch size
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz)

    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                            imgsz,
                                            batch_size // WORLD_SIZE,
                                            gs,
                                            single_cls,
                                            hyp=hyp,
                                            augment=True,
                                            cache=None if opt.cache == 'val' else opt.cache,
                                            rect=opt.rect,
                                            rank=LOCAL_RANK,
                                            workers=workers,
                                            image_weights=opt.image_weights,
                                            quad=opt.quad,
                                            prefix=colorstr('train: '),
                                            shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                     imgsz,
                                     batch_size // WORLD_SIZE * 2,
                                     gs,
                                     single_cls,
                                     hyp=hyp,
                                     cache=None if noval else opt.cache,
                                     rect=True,
                                     rank=-1,
                                     workers=workers * 2,
                                     pad=0.5,
                                     prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=4.0, imgsz=imgsz)
            model.half().float()

        # Model attributes
        model.nc = nc
        model.hyp = hyp
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
        model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    # Medical image specific: Conservative classification loss handling
    classification_enabled = opt.hyp.get('classification_enabled', False)
    classification_weight = opt.hyp.get('classification_weight', 0.005)  # Very low default weight
    
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, commented out for medical images)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 + maps) / (1 + maps)
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)

        # Update mosaic border (optional)
        if epoch - start_epoch >= hyp['mosaic']:
            if hasattr(dataset, 'mosaic'):
                dataset.mosaic = False
            if hasattr(dataset, 'close_mosaic'):
                dataset.close_mosaic(hyp=hyp)

        pbar = enumerate(train_loader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)
                
                # Medical image specific: Simplified output handling
                detections = pred
                class_outputs = None
                
                if isinstance(pred, tuple) and len(pred) == 2:
                    detections, class_outputs = pred

                # Compute detection loss
                targets = targets.to(detections[0].device)
                detection_loss, loss_items = compute_loss(detections, targets)

                # Medical image specific: Conservative classification loss computation
                classification_loss = torch.tensor(0.0, device=device)
                if classification_enabled and class_outputs is not None and classification_labels is not None:
                    try:
                        # Use standard CrossEntropyLoss for medical images
                        criterion = nn.CrossEntropyLoss()
                        label_indices = classification_labels.argmax(dim=1)
                        classification_loss = criterion(class_outputs, label_indices)
                        
                        # Medical image specific: Very conservative weight scheduling
                        progress = epoch / epochs
                        if progress < 0.6:  # First 60% of training
                            dynamic_weight = 0.0  # No classification loss initially
                        elif progress < 0.85:  # 60%-85% of training
                            dynamic_weight = classification_weight * 0.05  # Very low weight
                        else:  # Last 15% of training
                            dynamic_weight = classification_weight * 0.2  # Still conservative
                        
                        total_loss = detection_loss + dynamic_weight * classification_loss
                        
                        # Medical image specific: Enhanced logging
                        if i % 50 == 0:
                            LOGGER.info(f"Medical Batch {i}: detection_loss = {detection_loss.item():.4f}, "
                                       f"classification_loss = {classification_loss.item():.4f}, "
                                       f"weight = {dynamic_weight:.4f}")
                    except Exception as e:
                        LOGGER.warning(f"Medical classification loss computation failed: {e}")
                        total_loss = detection_loss
                else:
                    total_loss = detection_loss

            # Backward
            scaler.scale(total_loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1) if mloss is not None else loss_items
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                   (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return

            # Medical image specific: Enhanced validation frequency
            if RANK in {-1, 0} and (epoch % 5 == 0 or epoch == epochs - 1):
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=True,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss)

            callbacks.run('on_train_epoch_end', epoch=epoch)

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_fit_epoch_end', epoch=epoch, fitness=fitness(results))

            # Save best
            if (best_fitness == 0) or (fitness(results) > best_fitness):
                best_fitness = fitness(results)
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_train_end', logs=log_vals, epoch=epoch)

            # Save last
            if save and (epoch + 1) % save_period == 0:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,
                    'date': datetime.now().isoformat()}

                torch.save(ckpt, last)
                if best_fitness == fitness(results):
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch + 1) % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch + 1}.pt')
                del ckpt

        if RANK in {-1, 0} and stopper(epoch=epoch, fitness=fitness(results)):
            break

        if RANK in {-1, 0} and stop:
            break

    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,
                        save_dir=save_dir,
                        save_json=is_coco,
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_txt=opt.save_txt | save_hybrid,
                        save_hybrid=save_hybrid,
                        save_conf=opt.save_conf,
                        plots=False,
                        callbacks=callbacks,
                        compute_loss=compute_loss)
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', epoch=epoch, fitness=fitness(results))

        callbacks.run('on_train_end', logs=log_vals, epoch=epoch)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data_regurgitation.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'hyp_medical.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume from last checkpoint')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images for faster training (ram/disk)')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine learning rate scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Medical image specific arguments
    parser.add_argument('--medical-mode', action='store_true', help='Enable medical image specific optimizations')
    parser.add_argument('--classification-enabled', action='store_true', help='Enable classification loss (use with caution)')
    parser.add_argument('--classification-weight', type=float, default=0.005, help='Classification loss weight')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / 'requirements.txt')

    # Resume
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'
        opt_data = opt_yaml.read_text() if opt_yaml.exists() else ''
        opt = argparse.Namespace(**(yaml.safe_load(opt_data) if opt_data else {}))
        opt.cfg, opt.weights, opt.resume = '', str(last), True
        LOGGER.info(f'Resuming training from {last}')
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),
            'lrf': (1, 0.01, 1.0),
            'momentum': (0.3, 0.6, 0.98),
            'weight_decay': (1, 0.0, 0.001),
            'warmup_epochs': (1, 0.0, 5.0),
            'warmup_momentum': (1, 0.0, 0.95),
            'warmup_bias_lr': (1, 0.0, 0.2),
            'box': (1, 0.02, 0.2),
            'cls': (1, 0.2, 4.0),
            'cls_pw': (1, 0.5, 2.0),
            'obj': (1, 0.2, 4.0),
            'obj_pw': (1, 0.5, 2.0),
            'iou_t': (0, 0.1, 0.7),
            'anchor_t': (1, 2.0, 8.0),
            'fl_gamma': (0, 0.0, 2.0),
            'hsv_h': (1, 0.0, 0.1),
            'hsv_s': (1, 0.0, 0.9),
            'hsv_v': (1, 0.0, 0.9),
            'degrees': (1, 0.0, 45.0),
            'translate': (1, 0.0, 0.9),
            'scale': (1, 0.0, 0.9),
            'shear': (1, 0.0, 10.0),
            'perspective': (0, 0.0, 0.001),
            'flipud': (1, 0.0, 1.0),
            'fliplr': (1, 0.0, 1.0),
            'mosaic': (1, 0.0, 1.0),
            'mixup': (1, 0.0, 1.0),
            'copy_paste': (1, 0.0, 1.0)}
        opt.hyp = meta.copy()
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)
        # ei = [isinstance(x, (int, float)) for x in opt.hyp.values()]
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')

        for _ in range(opt.evolve):
            if evolve_csv.exists():
                parent = 'single'  # parent solver
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                for p, v in zip(parents, x):
                    p.copy_attr(v)
                for mp, v in zip(meta, x[0]):
                    mp[1] = v
            for mp, v in meta.items():
                mp[1] = mutate(v[0], v[1], v[2], 0.9)

            results = train(opt.hyp, opt, device, callbacks)
            callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch=best_epoch, fitness=fitness(results))

            if (not opt.nosave) or (final_epoch and not opt.evolve):
                ckpt = {
                    'epoch': best_epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,
                    'date': datetime.now().isoformat()}

                torch.save(ckpt, last)
                if best_fitness == fitness(results):
                    torch.save(ckpt, best)
                if opt.bucket:
                    os.system(f'gsutil cp {best} gs://{opt.bucket}/evolve.pt')
                if plots:
                    plot_evolve(evolve_csv)
                log_vals = list(mloss) + list(results) + lr
                callbacks.run('on_fit_epoch_end', log_vals, epoch=best_epoch, fitness=fitness(results))

            callbacks.run('on_train_end', logs=log_vals, epoch=best_epoch)

        if RANK in {-1, 0}:
            LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                       f'Results saved to {colorstr("bold", save_dir)}\n'
                       f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt) 