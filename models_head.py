"""
Decoupled Detection Head for YOLOv5
Separates classification and regression branches for better convergence
"""

import torch
import torch.nn as nn
from .common import Conv, SimAM


class DecoupledDetect(nn.Module):
    """Decoupled Detection Head with separate classification and regression branches
    
    Standard YOLO head couples classification and regression in the same conv layer.
    Decoupled heads (as used in YOLOX) improve performance by separating these tasks.
    """
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (reg:4 + obj:1 + cls:nc)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl
        
        # SimAM applied before each detection layer (as per ablation: row 5)
        self.simam = SimAM()
        
        # Decoupled heads for each detection layer
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        
        for i in range(self.nl):
            # Classification branch
            cls_conv = nn.Sequential(
                Conv(ch[i] * 2, ch[i], 3, 1),  # after concat, channels double
                nn.Conv2d(ch[i], self.na * nc, 1)
            )
            
            # Regression branch (4 values: x, y, w, h)
            reg_conv = nn.Sequential(
                Conv(ch[i], ch[i], 3, 1),
                nn.Conv2d(ch[i], self.na * 4, 1)
            )
            
            self.cls_convs.append(cls_conv)
            self.reg_convs.append(reg_conv)
        
        # Objectness branch (simple conv)
        self.obj_convs = nn.ModuleList(
            [nn.Conv2d(ch[i], self.na * 1, 1) for i in range(self.nl)]
        )
        
        self.stride = torch.zeros(self.nl)
        
    def forward(self, x):
        """Forward pass with decoupled heads"""
        z = []
        for i in range(self.nl):
            # Apply SimAM attention (from ablation study row 5)
            x[i] = self.simam(x[i])
            
            # Separate branches
            cls_out = self.cls_convs[i](x[i])
            reg_out = self.reg_convs[i](x[i])
            obj_out = self.obj_convs[i](x[i])
            
            # Concatenate: [reg (4), obj (1), cls (nc)]
            x_out = torch.cat([reg_out, obj_out, cls_out], 1)
            
            bs, _, ny, nx = x_out.shape
            x_out = x_out.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:
                # Inference mode - decode predictions
                y = x_out.sigmoid()
                z.append(y)
            else:
                z.append(x_out)
        
        return z if self.training else (torch.cat(z, 1), x)


class DecoupledDetect_P2(nn.Module):
    """Decoupled Detection Head with P2 small object detection layer"""
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        
        # SimAM for each layer
        self.simam = SimAM()
        
        # Decoupled heads (same as above but with P2 support)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.obj_convs = nn.ModuleList()
        
        for i in range(self.nl):
            # Channel dimension adapts to feature map size
            # P2: 128 channels, P3: 256, P4: 512, P5: 512
            c = ch[i]
            
            self.cls_convs.append(
                nn.Sequential(Conv(c, c, 3, 1), nn.Conv2d(c, self.na * nc, 1))
            )
            self.reg_convs.append(
                nn.Sequential(Conv(c, c, 3, 1), nn.Conv2d(c, self.na * 4, 1))
            )
            self.obj_convs.append(nn.Conv2d(c, self.na * 1, 1))
        
        self.stride = torch.zeros(self.nl)