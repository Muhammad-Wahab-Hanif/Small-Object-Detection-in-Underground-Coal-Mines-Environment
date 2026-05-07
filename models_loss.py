"""
SIoU Loss Implementation (SCYLLA-IoU)
Paper: SIoU Loss: More Powerful Learning for Bounding Box Regression
"""

import torch
import torch.nn as nn
import math


class SIoULoss(nn.Module):
    """SIoU Loss with angle, distance, and shape costs
    Improves upon CIoU by incorporating the angle of the gradient
    """
    def __init__(self, x1y1x2y2=True, eps=1e-7):
        super().__init__()
        self.x1y1x2y2 = x1y1x2y2
        self.eps = eps
        
    def forward(self, pred, target):
        """
        pred: predicted bboxes [x, y, w, h] (center format)
        target: target bboxes [x, y, w, h] (center format)
        """
        # Convert to [x1, y1, x2, y2] if needed
        if not self.x1y1x2y2:
            pred = self._cxcywh_to_xyxy(pred)
            target = self._cxcywh_to_xyxy(target)
        
        # Compute intersection area
        inter = self._intersection_area(pred, target)
        
        # Compute union area
        area_pred = self._bbox_area(pred)
        area_target = self._bbox_area(target)
        union = area_pred + area_target - inter + self.eps
        
        # IoU
        iou = inter / union
        
        # SIoU components
        iou = self._siou_calculation(pred, target, iou, area_pred, area_target)
        
        return (1 - iou).mean()
    
    def _cxcywh_to_xyxy(self, boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        x1 = boxes[..., 0] - boxes[..., 2] / 2
        y1 = boxes[..., 1] - boxes[..., 3] / 2
        x2 = boxes[..., 0] + boxes[..., 2] / 2
        y2 = boxes[..., 1] + boxes[..., 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _intersection_area(self, pred, target):
        x1 = torch.max(pred[..., 0], target[..., 0])
        y1 = torch.max(pred[..., 1], target[..., 1])
        x2 = torch.min(pred[..., 2], target[..., 2])
        y2 = torch.min(pred[..., 3], target[..., 3])
        w = (x2 - x1).clamp(0)
        h = (y2 - y1).clamp(0)
        return w * h
    
    def _bbox_area(self, boxes):
        w = boxes[..., 2] - boxes[..., 0]
        h = boxes[..., 3] - boxes[..., 1]
        return w * h
    
    def _siou_calculation(self, pred, target, iou, area_pred, area_target):
        """Calculate SIoU with angle, distance, and shape costs"""
        
        # Center points
        pred_cx = (pred[..., 0] + pred[..., 2]) / 2
        pred_cy = (pred[..., 1] + pred[..., 3]) / 2
        target_cx = (target[..., 0] + target[..., 2]) / 2
        target_cy = (target[..., 1] + target[..., 3]) / 2
        
        # Distance between centers
        rho_x = pred_cx - target_cx
        rho_y = pred_cy - target_cy
        sigma = torch.sqrt(torch.pow(rho_x, 2) + torch.pow(rho_y, 2) + self.eps)
        
        # Enclosing box dimensions
        enclose_x1 = torch.min(pred[..., 0], target[..., 0])
        enclose_y1 = torch.min(pred[..., 1], target[..., 1])
        enclose_x2 = torch.max(pred[..., 2], target[..., 2])
        enclose_y2 = torch.max(pred[..., 3], target[..., 3])
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        
        # Angle cost
        sin_alpha = torch.abs(rho_y) / (sigma + self.eps)
        sin_alpha = torch.clamp(sin_alpha, min=0, max=1)
        angle_cost = torch.cos(2 * torch.arcsin(sin_alpha) - math.pi / 2)
        
        # Distance cost
        rho_x_norm = torch.pow(rho_x / (enclose_w + self.eps), 2)
        rho_y_norm = torch.pow(rho_y / (enclose_h + self.eps), 2)
        gamma = 2 - angle_cost
        distance_cost = 2 - torch.exp(-gamma * rho_x_norm) - torch.exp(-gamma * rho_y_norm)
        
        # Shape cost
        w_pred = pred[..., 2] - pred[..., 0]
        h_pred = pred[..., 3] - pred[..., 1]
        w_target = target[..., 2] - target[..., 0]
        h_target = target[..., 3] - target[..., 1]
        
        ww = torch.abs(w_pred - w_target) / torch.max(w_pred, w_target)
        hh = torch.abs(h_pred - h_target) / torch.max(h_pred, h_target)
        theta = 4
        shape_cost = torch.pow(1 - torch.exp(-ww), theta) + torch.pow(1 - torch.exp(-hh), theta)
        
        # SIoU
        siou = iou - (distance_cost + shape_cost) / 2
        
        return siou


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=0.25, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()