from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_short_side(image: Image.Image, target: int, mode: str = "short") -> Tuple[Image.Image, float]:
    """Resize so the short (or long) side equals `target`, preserving aspect ratio.

    Returns (resized_image, scale) where scale = new_short / old_short (or new_long / old_long).
    Port of functions/myresize.m.
    """
    w, h = image.size
    short, long = min(w, h), max(w, h)
    if mode == "short":
        scale = target / short
    elif mode == "long":
        scale = target / long
    else:
        raise ValueError(f"mode must be 'short' or 'long', got {mode!r}")
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return image.resize((new_w, new_h), Image.BILINEAR), scale


def normalize(image_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] HWC or CHW → float32 [0, 1] CHW, mean/std normalized.

    Accepts a tensor with values in [0, 255]. Returns NCHW unsqueezed when input is CHW.
    """
    if image_uint8.dim() == 3 and image_uint8.shape[-1] == 3:
        image_uint8 = image_uint8.permute(2, 0, 1)
    x = image_uint8.float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0) if x.dim() == 3 else x


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize(). NCHW or CHW float → uint8-range CHW float in [0, 255]."""
    squeeze = x.dim() == 4
    if squeeze:
        x = x.squeeze(0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x * std + mean) * 255.0


def clip_image(x_unnorm: torch.Tensor) -> torch.Tensor:
    """Clip un-normalized image to [0, 255]. Port of functions/image_clip.m."""
    return x_unnorm.clamp(0.0, 255.0)
