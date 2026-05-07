"""
Data loading utilities for underground coal mine dataset
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm


class CoalMineDataset(Dataset):
    """Custom dataset for underground coal mine object detection"""
    
    def __init__(self, path, img_size=640, stride=32, augment=False, hyp=None):
        self.path = path
        self.img_size = img_size
        self.stride = stride
        self.augment = augment
        self.hyp = hyp or {}
        
        # Collect image paths
        self.im_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.im_files.extend(Path(path).rglob(ext))
        
        self.n = len(self.im_files)
        print(f"Found {self.n} images in {path}")
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        img_path = self.im_files[index]
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = str(img_path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = [float(p) for p in parts[:5]]
                        labels.append([cls, x, y, w, h])
        
        labels = np.array(labels) if labels else np.zeros((0, 5))
        
        # Resize and pad
        img, labels = self._resize_and_pad(img, labels, self.img_size)
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(labels)
        
        return img, labels, str(img_path), index
    
    def _resize_and_pad(self, img, labels, target_size):
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad to target size
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Adjust labels
        if len(labels) > 0:
            labels[:, 1] = labels[:, 1] * new_w / target_size
            labels[:, 2] = labels[:, 2] * new_h / target_size
            labels[:, 3] = labels[:, 3] * new_w / target_size
            labels[:, 4] = labels[:, 4] * new_h / target_size
        
        return img, labels


def create_dataloader(path, img_size, batch_size, workers, shuffle=True, augment=False, hyp=None, rect=False):
    """Create dataloader for training or validation"""
    dataset = CoalMineDataset(path, img_size, augment=augment, hyp=hyp)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=lambda x: x  # Custom collate for variable batch
    )
    return dataloader