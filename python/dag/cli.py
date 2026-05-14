"""Command-line entry point for DAG attacks on Pascal VOC.

Usage:
    python -m dag.cli seg --config python/configs/seg_voc.yaml
    python -m dag.cli det --config python/configs/det_voc.yaml

Flags override config values.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from PIL import Image
import torch.nn.functional as F

from .attacks.detection import fool_det
from .attacks.segmentation import fool_seg
from .data.targets import build_target_mask
from .data.voc import (
    VOC_DET_CLASSES,
    VOC_SEG_CLASSES,
    coco_to_voc_label,
    load_voc_det_sample,
    load_voc_seg_sample,
)
from .models.detection import build_det_model
from .models.segmentation import build_seg_model
from .utils.imaging import denormalize, normalize, resize_short_side
from .utils.viz import (
    draw_detections,
    save_image_uint8,
    save_triptych,
    seg_overlay,
    tensor_to_uint8,
)


def _load_config(path: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    cfg["voc_root"] = os.path.expanduser(cfg.get("voc_root", ""))
    return cfg


def _seed_everything(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _pil_to_tensor_normalized(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"))  # HWC uint8
    t = torch.from_numpy(arr).to(device)  # HWC uint8
    return normalize(t)  # [1, 3, H, W] normalized


def run_seg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = cfg["device"]
    rng = _seed_everything(int(cfg.get("seed", 0)))

    sample = load_voc_seg_sample(cfg["image_id"], Path(cfg["voc_root"]), cfg["year"])
    img, scale = resize_short_side(sample.image, int(cfg["short_side"]), mode="short")
    seg_mask = sample.seg_mask
    # Resize the GT mask consistently (nearest neighbour, no antialias).
    seg_t = torch.from_numpy(seg_mask)
    seg_resized = F.interpolate(
        seg_t.float().unsqueeze(0).unsqueeze(0), size=img.size[::-1], mode="nearest"
    ).long().squeeze().numpy()

    x_norm = _pil_to_tensor_normalized(img, device)
    model = build_seg_model(cfg["model"], device=device, weights_path=cfg.get("weights"))

    target_mask = build_target_mask(cfg["shape"], seg_resized, target_size=x_norm.shape[-2:], rng=rng)
    target_t = torch.from_numpy(target_mask).long()
    orig_t = torch.from_numpy(seg_resized).long()

    result = fool_seg(
        model,
        x_norm,
        target_t,
        orig_t,
        max_iter=int(cfg["max_iter"]),
        step_length=float(cfg["step_length"]),
        success_ratio=float(cfg.get("success_ratio", 0.01)),
        verbose=cfg.get("verbose", False),
    )

    out_dir = Path(cfg["output_dir"]).expanduser() / cfg["image_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_pixel = denormalize(x_norm).squeeze(0)
    adv_pixel = denormalize(x_norm + result.perturbation).squeeze(0)
    pert_pixel = (adv_pixel - orig_pixel).cpu().numpy().transpose(1, 2, 0)

    orig_u8 = tensor_to_uint8(orig_pixel)
    adv_u8 = tensor_to_uint8(adv_pixel)
    save_image_uint8(orig_u8, out_dir / "orig.png")
    save_image_uint8(adv_u8, out_dir / "adv.png")
    save_triptych(orig_u8, adv_u8, pert_pixel, out_dir / "triptych.png")

    if result.final_pred is not None:
        save_image_uint8(
            seg_overlay(adv_u8, result.final_pred.cpu().numpy()),
            out_dir / "adv_seg_overlay.png",
        )
        save_image_uint8(
            seg_overlay(orig_u8, target_mask),
            out_dir / "target_overlay.png",
        )

    with torch.no_grad():
        orig_pred = model(x_norm)["out"].argmax(1).squeeze(0).cpu().numpy()
    save_image_uint8(seg_overlay(orig_u8, orig_pred), out_dir / "orig_seg_overlay.png")

    metrics = {
        "image_id": cfg["image_id"],
        "shape": cfg["shape"],
        "iterations": result.iterations,
        "succeeded": result.succeeded,
        "active_history": result.active_history,
        "max_perturbation_pixel": float(np.abs(pert_pixel).max()),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_det(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = cfg["device"]
    rng = _seed_everything(int(cfg.get("seed", 0)))

    sample = load_voc_det_sample(cfg["image_id"], Path(cfg["voc_root"]), cfg["year"])
    img, scale = resize_short_side(sample.image, int(cfg["short_side"]), mode="short")
    gt_boxes = sample.boxes * scale
    gt_labels = sample.labels

    x_norm = _pil_to_tensor_normalized(img, device)
    det = build_det_model(
        cfg["model"],
        device=device,
        weights_path=cfg.get("weights"),
        num_classes=int(cfg.get("num_classes", 91)),
        rpn_post_nms_top_n=int(cfg.get("rpn_post_nms_top_n", 3000)),
        rpn_nms_thresh=float(cfg.get("rpn_nms_thresh", 0.9)),
    )

    result = fool_det(
        det,
        x_norm,
        gt_boxes.to(device),
        gt_labels.to(device),
        max_iter=int(cfg["max_iter"]),
        step_length=float(cfg["step_length"]),
        strong_nms_thresh=float(cfg.get("strong_nms_thresh", 0.35)),
        strong_score_thresh=float(cfg.get("strong_score_thresh", 0.8)),
        rng=rng,
        verbose=cfg.get("verbose", False),
    )

    out_dir = Path(cfg["output_dir"]).expanduser() / cfg["image_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_pixel = denormalize(x_norm).squeeze(0)
    adv_pixel = denormalize(x_norm + result.perturbation).squeeze(0)
    pert_pixel = (adv_pixel - orig_pixel).cpu().numpy().transpose(1, 2, 0)

    orig_u8 = tensor_to_uint8(orig_pixel)
    adv_u8 = tensor_to_uint8(adv_pixel)
    save_image_uint8(orig_u8, out_dir / "orig.png")
    save_image_uint8(adv_u8, out_dir / "adv.png")
    save_triptych(orig_u8, adv_u8, pert_pixel, out_dir / "triptych.png")

    # Run the full detection pipeline on both the original and adversarial image and
    # draw boxes. torchvision's full forward applies its own GeneralizedRCNNTransform
    # (which normalizes by ImageNet mean/std), so we hand it an un-normalized [0, 1]
    # tensor — not the already-normalized x_norm used by the attack's box_head_forward.
    # Evaluate against the SAME fixed proposals the attack saw. Running torchvision's full
    # pipeline regenerates proposals via the RPN on the adv image, which the attack never
    # touched — those new proposals can produce predictions the attack didn't suppress,
    # masking the attack's true success. The original DAG paper evaluates against the
    # attack's fixed proposal set; we do the same.
    voc_native = det.num_coco_classes == 21
    from torchvision.ops import nms as tv_nms
    proposals = result.proposals
    score_thresh = float(cfg.get("viz_score_thresh", 0.5))
    nms_thresh = float(cfg.get("viz_nms_thresh", 0.5))

    def _draw_fixed_proposals(image_u8: np.ndarray, x_normalized: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = det.box_head_forward(x_normalized, proposals)
        if not voc_native:
            from .attacks.detection import _coco_logits_to_voc
            logits = _coco_logits_to_voc(logits)
        probs = logits.softmax(-1)  # [N, 21]
        # Per-class score thresh + NMS, matching torchvision's postprocess_detections.
        kept_boxes, kept_labels, kept_scores = [], [], []
        for c in range(1, probs.shape[1]):
            sc = probs[:, c]
            mask = sc > score_thresh
            if not mask.any():
                continue
            b = proposals[mask]
            s = sc[mask]
            keep = tv_nms(b, s, nms_thresh)
            kept_boxes.append(b[keep])
            kept_labels.extend([c] * len(keep))
            kept_scores.extend(s[keep].tolist())
        if not kept_boxes:
            return image_u8
        boxes_all = torch.cat(kept_boxes, 0).cpu()
        return draw_detections(image_u8, boxes_all, kept_labels, kept_scores, VOC_DET_CLASSES)

    save_image_uint8(_draw_fixed_proposals(orig_u8, x_norm), out_dir / "orig_det_overlay.png")
    save_image_uint8(_draw_fixed_proposals(adv_u8, x_norm + result.perturbation), out_dir / "adv_det_overlay.png")

    metrics = {
        "image_id": cfg["image_id"],
        "iterations": result.iterations,
        "succeeded": result.succeeded,
        "active_history": result.active_history,
        "n_proposals": int(result.proposals.shape[0]) if result.proposals is not None else 0,
        "max_perturbation_pixel": float(np.abs(pert_pixel).max()),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DAG adversarial attacks on Pascal VOC.")
    sub = p.add_subparsers(dest="task", required=True)
    for task in ("seg", "det"):
        sp = sub.add_parser(task)
        sp.add_argument("--config", required=True, type=Path)
        sp.add_argument("--image-id", default=None)
        sp.add_argument("--voc-root", default=None)
        sp.add_argument("--max-iter", type=int, default=None)
        sp.add_argument("--device", default=None)
        sp.add_argument("--output-dir", default=None)
        sp.add_argument("--verbose", action="store_true")
        if task == "seg":
            sp.add_argument("--shape", default=None, choices=[None, "circle", "square", "strip"])
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    overrides = {
        "image_id": args.image_id,
        "voc_root": args.voc_root,
        "max_iter": args.max_iter,
        "device": args.device,
        "output_dir": args.output_dir,
        "verbose": args.verbose if args.verbose else None,
    }
    if args.task == "seg":
        overrides["shape"] = args.shape
    cfg = _load_config(args.config, overrides)
    if args.task == "seg":
        metrics = run_seg(cfg)
    else:
        metrics = run_det(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
