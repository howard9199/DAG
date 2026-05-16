from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
SHAPES_DIR = REPO_ROOT / "data" / "shapes"


def generate_mapping(gt_idx: np.ndarray, num_object_classes: int = 20, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Port of code/generate_mapping.m.

    Given the set of ground-truth object class indices (1-indexed, excluding background),
    return:
      - mapping: int array of length (num_object_classes + 1) where index i maps GT class i to
        a randomly chosen non-present class. Background (0) maps to 0. Indices not in gt_idx
        map to 0 (sentinel, not used by the attack).
      - shuffled_targets: the full shuffled pool of candidate targets (the
        target_idx_candidate_shuffle output of the original .m).

    Class indexing follows MATLAB (1..20 object classes), preserved here so the algorithm
    matches faithfully. Background is 0 (Python-friendly) — the attack treats 0 as 'background'.
    """
    rng = rng or np.random.default_rng()
    gt_idx = np.asarray(gt_idx, dtype=np.int64)
    gt_idx = gt_idx[gt_idx > 0]
    pool = np.array([c for c in range(1, num_object_classes + 1) if c not in gt_idx], dtype=np.int64)
    shuffled = rng.permutation(pool)
    mapping = np.zeros(num_object_classes + 1, dtype=np.int64)
    n = min(len(gt_idx), len(shuffled))
    mapping[gt_idx[:n]] = shuffled[:n]
    return mapping, shuffled


def load_shape_mask(shape: str) -> np.ndarray:
    """Load a 500x500 int mask from data/shapes/. circle/square are binary {0,1}; strip is {0..4}."""
    if shape not in {"circle", "square", "strip"}:
        raise ValueError(f"shape must be circle|square|strip, got {shape!r}")
    return np.load(SHAPES_DIR / f"{shape}.npy")


def random_shape_mask(
    target_size: Tuple[int, int],
    rng: np.random.Generator,
    num_blobs_range: Tuple[int, int] = (1, 3),
    scale_range: Tuple[float, float] = (0.05, 0.30),
    num_vertices_range: Tuple[int, int] = (5, 12),
) -> np.ndarray:
    """Generate an int64 HxW mask with K random irregular polygon blobs (0 = background, blobs = 1..K).

    Each blob is an irregular polygon: N vertices sampled around a center at uniform angles,
    with jittered radii so the boundary is non-circular. Center and scale (area fraction of
    image) are sampled per blob; placement is constrained so each blob fits inside the image.
    Later blobs overwrite earlier ones where they overlap.
    """
    H, W = target_size
    img_area = float(H * W)
    canvas = Image.new("I", (W, H), 0)
    draw = ImageDraw.Draw(canvas)

    k_blobs = int(rng.integers(num_blobs_range[0], num_blobs_range[1] + 1))
    for blob_id in range(1, k_blobs + 1):
        scale = float(rng.uniform(scale_range[0], scale_range[1]))
        target_area = scale * img_area
        base_r = float(np.sqrt(target_area / np.pi))
        # Keep the blob inside the image by leaving a margin equal to its max possible radius.
        margin = min(base_r * 1.5, min(H, W) * 0.45)
        if margin >= min(H, W) / 2:
            cx = W / 2.0
            cy = H / 2.0
        else:
            cx = float(rng.uniform(margin, W - margin))
            cy = float(rng.uniform(margin, H - margin))

        n_v = int(rng.integers(num_vertices_range[0], num_vertices_range[1] + 1))
        angles = np.sort(rng.uniform(0.0, 2 * np.pi, size=n_v))
        radii = base_r * rng.uniform(0.55, 1.45, size=n_v)
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)
        verts = list(zip(xs.tolist(), ys.tolist()))
        draw.polygon(verts, fill=blob_id, outline=blob_id)

    return np.asarray(canvas, dtype=np.int64)


def build_target_mask(
    shape: str,
    gt_mask: np.ndarray,
    target_size: Tuple[int, int],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build adversarial target segmentation mask.

    Loads the geometric shape from data/shapes/, resizes to target_size (H, W), and replaces
    each nonzero region's value with a randomly assigned non-present VOC class.

    Args:
        shape: 'circle' | 'square' | 'strip' | 'random'
        gt_mask: HxW int array of the original GT segmentation (used to compute present classes).
        target_size: (H, W) of the desired output mask.
    Returns:
        HxW int64 array with adversarial target labels per pixel (0 = background / ignore).
    """
    rng = rng or np.random.default_rng()
    if shape == "random":
        resized = random_shape_mask(target_size, rng)
    else:
        raw = load_shape_mask(shape).astype(np.int64)
        raw_t = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(raw_t, size=target_size, mode="nearest").squeeze().long().numpy()

    gt_classes = np.unique(gt_mask)
    gt_classes = gt_classes[(gt_classes != 0) & (gt_classes != 255)]
    _, shuffled = generate_mapping(gt_classes, rng=rng)

    region_ids = np.unique(resized)
    region_ids = region_ids[region_ids != 0]
    out = np.zeros_like(resized, dtype=np.int64)
    if len(shuffled) == 0:
        return out
    for k, rid in enumerate(region_ids):
        target_cls = int(shuffled[k % len(shuffled)])
        out[resized == rid] = target_cls
    return out
