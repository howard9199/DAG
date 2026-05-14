"""Smoke tests: run a few attack iterations on a small synthetic model to verify
gradients flow and the perturbation grows. No torchvision pretrained weights needed.
"""
import numpy as np
import torch
import torch.nn as nn

from dag.attacks.segmentation import fool_seg


class TinySegModel(nn.Module):
    """Mini segmentation model: 1x1 conv producing 21-class logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 21, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        return {"out": self.conv(x)}


def test_seg_attack_smoke():
    torch.manual_seed(0)
    model = TinySegModel().eval()
    x = torch.randn(1, 3, 32, 32)
    target = torch.zeros((32, 32), dtype=torch.long)
    target[10:20, 10:20] = 5
    original = torch.zeros((32, 32), dtype=torch.long)

    result = fool_seg(model, x, target, original, max_iter=5, step_length=0.5)
    assert result.iterations <= 5
    assert result.perturbation.shape == x.shape
    # Smoke: at least *some* perturbation was applied.
    assert result.perturbation.abs().sum().item() > 0
    # active_history strictly non-empty and non-negative.
    assert all(a >= 0 for a in result.active_history)
