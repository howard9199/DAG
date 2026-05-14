from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# Pascal VOC 21-class colormap matching data/pascal_seg_colormap.mat conventions.
def _voc_colormap(n: int = 256) -> np.ndarray:
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = (r, g, b)
    return cmap


VOC_COLORMAP = _voc_colormap()


def save_image_uint8(arr: np.ndarray, path: Path) -> None:
    """arr: HxWx3 uint8 or HxW uint8. Saves as PNG."""
    Image.fromarray(arr).save(path)


def tensor_to_uint8(image_chw: torch.Tensor) -> np.ndarray:
    """CHW float [0, 255] → HWC uint8 numpy."""
    x = image_chw.detach().clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    return x.transpose(1, 2, 0)


def seg_overlay(image_uint8: np.ndarray, seg_pred: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a per-pixel class label map onto an image using the VOC colormap."""
    color = VOC_COLORMAP[seg_pred]
    blend = (image_uint8.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha)
    return blend.clip(0, 255).astype(np.uint8)


def draw_detections(
    image_uint8: np.ndarray,
    boxes: torch.Tensor,
    labels: Sequence[int],
    scores: Sequence[float],
    class_names: Sequence[str],
) -> np.ndarray:
    """Draw detection boxes on an image and label them. boxes: [N, 4] in xyxy."""
    img = Image.fromarray(image_uint8).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    for box, lbl, sc in zip(boxes.tolist(), labels, scores):
        x1, y1, x2, y2 = box
        # Use saturated palette colors keyed on label so different classes are visually distinct.
        color = tuple(int(c) for c in VOC_COLORMAP[(lbl + 1) % len(VOC_COLORMAP)])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        text = f"{class_names[lbl]} {sc:.2f}"
        # Filled background behind text for legibility. PIL's default bitmap font
        # doesn't support textbbox, so we estimate the box from the text length.
        try:
            tw = draw.textlength(text, font=font)
        except AttributeError:  # very old PIL
            tw = 6 * len(text)
        th = 11
        draw.rectangle([x1 + 4, y1 + 4, x1 + 4 + tw + 4, y1 + 4 + th + 2], fill=color)
        draw.text((x1 + 6, y1 + 5), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def save_triptych(
    original_uint8: np.ndarray,
    adv_uint8: np.ndarray,
    perturbation: np.ndarray,
    path: Path,
) -> None:
    """Save side-by-side: original | adversarial | perturbation (rescaled to visible)."""
    import matplotlib.pyplot as plt

    pert_vis = perturbation - perturbation.min()
    if pert_vis.max() > 0:
        pert_vis = pert_vis / pert_vis.max()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_uint8)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(adv_uint8)
    axes[1].set_title("adversarial")
    axes[1].axis("off")
    axes[2].imshow(pert_vis)
    axes[2].set_title("perturbation (rescaled)")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
