from __future__ import annotations

import torch


def box_iou_plus1(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU with the +1 pixel-inclusive convention from boxoverlap.m.

    boxes_a: [N, 4] (x1, y1, x2, y2)
    boxes_b: [M, 4]
    Returns: [N, M]
    Invalid intersections (w <= 0 or h <= 0) → IoU = 0.
    """
    a = boxes_a.unsqueeze(1)  # [N, 1, 4]
    b = boxes_b.unsqueeze(0)  # [1, M, 4]
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    w = (x2 - x1 + 1).clamp(min=0)
    h = (y2 - y1 + 1).clamp(min=0)
    inter = w * h
    area_a = (boxes_a[:, 2] - boxes_a[:, 0] + 1) * (boxes_a[:, 3] - boxes_a[:, 1] + 1)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0] + 1) * (boxes_b[:, 3] - boxes_b[:, 1] + 1)
    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou


def nms_plus1(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy NMS with +1 convention. Port of functions/nms.m.

    Returns indices of kept boxes, sorted by descending score.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
