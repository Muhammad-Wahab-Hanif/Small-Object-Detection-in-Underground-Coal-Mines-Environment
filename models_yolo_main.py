"""
MFF-YOLOv5s-4EL: Main Model Definition
Integrates all 5 enhancements:
1. SIoU Loss
2. Decoupled Detection Head
3. Small Object Detection Layer (P2)
4. SimAM Attention
5. C3Ghost Module
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml

from .common import Conv, C3Ghost, SPPF, SimAM
from .head import DecoupledDetect


class Model(nn.Module):
    """MFF-YOLOv5s-4EL Model"""
    def __init__(self, cfg='yolov5s_4el.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, 'r') as f:
                self.yaml = yaml.safe_load(f)
        
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc
        if anchors:
            self.yaml['anchors'] = anchors
        
        # Build model from YAML
        self.model, self.save = self._parse_model(self.yaml, ch=[ch])
        
        # Defaults
        self.stride = torch.tensor([32., 16., 8., 4.])  # P5, P4, P3, P2 strides
        
    def forward(self, x, augment=False, profile=False):
        """Forward pass"""
        return self._forward_once(x, profile)
    
    def _forward_once(self, x, profile=False):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
    
    def _parse_model(self, d, ch):
        """Parse YAML configuration to build model"""
        from .common import SimAM, C3Ghost, GhostConv
        
        nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
        na = len(d['anchors'][0]) // 2 if d.get('anchors', False) else 3
        
        layers, save, c2 = [], [], ch[-1]
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            m = eval(m) if isinstance(m, str) else m
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a
                except:
                    pass
            
            n = max(round(n * gd), 1) if n > 1 else n
            if m in [Conv, C3Ghost, SPPF, SimAM]:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(c2 * gw, 8)
                
                args = [c1, c2, *args[1:]]
                if m in [C3Ghost]:
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is Detect or m is DecoupledDetect:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                c2 = ch[f]
            
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')
            m_.i, m_.f, m_.type = i, f, t
            layers.append(m_)
            ch = [c2]
        
        return nn.Sequential(*layers), save


def make_divisible(x, divisor):
    """Return x made divisible by divisor"""
    return int(np.ceil(x / divisor) * divisor)


def autopad(k, p=None):
    """Auto-pad for convolution"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# Import here to avoid circular imports
import numpy as np
from .common import Concat
from .head import DecoupledDetect as Detect