"""
Training script for MFF-YOLOv5s-4EL
Underground Coal Mine Dataset - 4500 images, 8 classes
Hardware: RTX 3090 (24GB), Ubuntu 18.04
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Set paths
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.yolo import Model
from models.loss import SIoULoss, FocalLoss
from utils.dataloader import create_dataloader
from utils.metrics import compute_loss, ap_per_class


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(hyp, opt, device):
    """Main training function"""
    set_seed(42)
    
    # TensorBoard writer
    log_dir = Path(opt.save_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Load data config
    with open(opt.data, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    nc = data_dict['nc']
    names = data_dict['names']
    
    # Initialize model
    model = Model(opt.cfg, ch=3, nc=nc).to(device)
    
    # Load pretrained weights if specified
    if opt.weights and opt.weights.endswith('.pt'):
        ckpt = torch.load(opt.weights, map_location=device)
        state_dict = ckpt['model'].float().state_dict()
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {opt.weights}")
    
    # Initialize optimizer
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    
    optimizer = torch.optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    
    # Learning rate scheduler
    lf = lambda x: (1 - x / hyp['epochs']) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # Loss functions
    siou_loss = SIoULoss()
    focal_loss = FocalLoss(alpha=0.25, gamma=1.5)
    
    # Create dataloaders
    train_loader = create_dataloader(
        data_dict['train'],
        opt.img_size,
        opt.batch_size,
        opt.workers,
        shuffle=True,
        hyp=hyp,
        augment=True,
        rect=False
    )
    
    val_loader = create_dataloader(
        data_dict['val'],
        opt.img_size,
        opt.batch_size,
        opt.workers,
        shuffle=False,
        hyp=hyp,
        augment=False,
        rect=True
    )
    
    # Training loop
    best_mAP = 0.0
    for epoch in range(hyp['epochs']):
        model.train()
        mloss = torch.zeros(4, device=device)
        
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            pred = model(imgs)
            
            # Compute losses
            loss, loss_items = compute_loss(pred, targets, model, siou_loss, focal_loss)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            mloss = (mloss * i + loss_items) / (i + 1)
            
            # Logging
            if i % 50 == 0:
                print(f"Epoch {epoch}/{hyp['epochs']} | Batch {i} | Loss: {loss.item():.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Validation every epoch
        results = validate(model, val_loader, device, siou_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', mloss[0], epoch)
        writer.add_scalar('Loss/val', results[0], epoch)
        writer.add_scalar('Metrics/mAP', results[2], epoch)
        
        # Save checkpoint
        if results[2] > best_mAP:
            best_mAP = results[2]
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mAP': best_mAP,
                'hyp': hyp,
            }, Path(opt.save_dir) / 'best.pt')
        
        print(f"Epoch {epoch} complete | mAP: {results[2]:.3f} | Best: {best_mAP:.3f}")
    
    writer.close()
    print(f"Training complete! Best mAP: {best_mAP:.3f}")
    return model


def validate(model, dataloader, device, loss_fn):
    """Validation function"""
    model.eval()
    stats = []
    
    with torch.no_grad():
        for imgs, targets, paths, _ in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            pred = model(imgs)
            
            # Evaluate
            for si in range(len(pred)):
                # Process predictions
                pass  # Add your evaluation logic here
    
    return [0.0, 0.0, 0.0]  # Return [loss, precision, mAP]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='config/coal_mine.yaml', help='dataset config')
    parser.add_argument('--cfg', type=str, default='models/yaml/yolov5s_4el.yaml', help='model config')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='pretrained weights')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=301, help='number of epochs')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--workers', type=int, default=8, help='data workers')
    parser.add_argument('--device', type=str, default='0', help='cuda device')
    parser.add_argument('--save-dir', type=str, default='runs/train/exp', help='save directory')
    args = parser.parse_args()
    
    # Hyperparameters (from your specifications)
    hyp = {
        'lr0': 0.011,
        'lrf': 0.01,
        'momentum': 0.9371,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 0.05,
        'cls': 0.5,
        'cls_pw': 1.0,
        'obj': 1.0,
        'obj_pw': 1.0,
        'iou_t': 0.20,
        'anchor_t': 4.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'epochs': args.epochs,
    }
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    train(hyp, args, device)


if __name__ == '__main__':
    main()