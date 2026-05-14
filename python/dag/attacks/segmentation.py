from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.imaging import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class SegAttackResult:
    perturbation: torch.Tensor          # [1, 3, H, W] in normalized image space
    iterations: int
    succeeded: bool
    active_history: List[int] = field(default_factory=list)
    final_pred: torch.Tensor | None = None  # [H, W] int64 class indices


def _clip_perturbation_to_pixel_range(x_norm: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Clip r so that (x_norm + r), when un-normalized, lies in [0, 255].

    Equivalent to functions/image_clip.m applied in pixel space.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=x_norm.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x_norm.device).view(1, 3, 1, 1)
    x_pixel = (x_norm + r) * std + mean  # in [0, 1]
    x_pixel = x_pixel.clamp(0.0, 1.0)
    return (x_pixel - mean) / std - x_norm


def fool_seg(
    model: nn.Module,
    x_norm: torch.Tensor,
    seg_target: torch.Tensor,
    seg_original: torch.Tensor,
    max_iter: int = 200,
    step_length: float = 0.5,
    success_ratio: float = 0.01,
    verbose: bool = False,
) -> SegAttackResult:
    """DAG segmentation attack — port of code/fooling_seg_net.m + forward_and_back_propogation_seg.m.

    Args:
        model: torchvision segmentation model returning {'out': logits [1, C, H, W]}.
        x_norm: [1, 3, H, W] image tensor, ImageNet-normalized, on the model's device.
        seg_target: [H, W] int64 adversarial target labels (0 = ignore).
        seg_original: [H, W] int64 ground-truth labels (0 = bg, 255 = ignore).
        max_iter: max iterations (default 200 matching generate_config.m).
        step_length: pixel-space step length (default 0.5).
        success_ratio: stop when active pixels < ratio * |original foreground|.
    Returns:
        SegAttackResult with the optimal perturbation in *normalized* space.
    """
    device = x_norm.device
    seg_target = seg_target.to(device)
    seg_original = seg_original.to(device)

    # Active = pixels where target != 0 (we want to push the prediction toward target there).
    # Original DAG terminates when the per-pixel prediction matches the (target-or-original)
    # set; we mirror that by counting pixels where target!=0 AND pred != target.
    target_active_mask = (seg_target != 0)
    n_target_pixels = int(target_active_mask.sum().item())
    n_original_fg = int(((seg_original > 0) & (seg_original != 255)).sum().item())
    threshold = max(1, int(success_ratio * max(n_original_fg, n_target_pixels)))

    # Step length is defined in *pixel* units (matching the original .m where r is in pixel space).
    # Translate to normalized space: 1 normalized unit ≈ 1 / (255 * std) pixels per channel.
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    step_norm = step_length / (255.0 * std)  # average per-channel normalized step

    r = torch.zeros_like(x_norm)
    history: List[int] = []
    succeeded = False
    final_pred: torch.Tensor | None = None

    for itr in range(max_iter):
        r_req = r.detach().clone().requires_grad_(True)
        logits = model(x_norm + r_req)["out"]  # [1, C, H, W]
        with torch.no_grad():
            pred = logits.argmax(1).squeeze(0)  # [H, W]
        final_pred = pred.detach()

        # Active set: target pixels still not flipped to the target class.
        still_wrong = target_active_mask & (pred != seg_target)
        n_active = int(still_wrong.sum().item())
        history.append(n_active)
        if verbose:
            print(f"[seg] iter {itr}: active={n_active}")

        if n_active <= threshold:
            succeeded = True
            break

        # Construct scatter masks over [1, C, H, W].
        C = logits.shape[1]
        flat_mask = still_wrong.view(-1)
        target_flat = seg_target.view(-1).clamp(min=0, max=C - 1)
        pred_flat = pred.view(-1).clamp(min=0, max=C - 1)

        # Gather logit at target class minus logit at predicted class, summed over active pixels.
        logits_flat = logits.squeeze(0).view(C, -1)  # [C, H*W]
        idx = torch.arange(logits_flat.shape[1], device=device)
        idx_active = idx[flat_mask]
        target_active = target_flat[flat_mask]
        pred_active = pred_flat[flat_mask]

        loss_target = logits_flat[target_active, idx_active].sum()
        loss_pred = logits_flat[pred_active, idx_active].sum()
        loss = loss_target - loss_pred

        dr = torch.autograd.grad(loss, r_req)[0].detach()
        max_abs = dr.abs().max()
        if float(max_abs) == 0.0:
            if verbose:
                print("[seg] gradient is zero; stopping early")
            break

        # Same step rule as fooling_seg_net.m: r += step / max(|dr|) * dr (in pixel space).
        r = r + step_norm * (dr / max_abs)
        r = _clip_perturbation_to_pixel_range(x_norm, r).detach()

    return SegAttackResult(
        perturbation=r.detach(),
        iterations=itr + 1,
        succeeded=succeeded,
        active_history=history,
        final_pred=final_pred,
    )
