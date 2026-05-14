"""Run DAG-Seg + DAG-Det on a configurable set of VOC images.

Reads `configs/experiment.yaml` (path overridable via --config). Each image gets both
attacks; results land at `output_dir/{seg,det}/{image_id}/`.

Example:
    uv run python -m scripts.run_experiment --config configs/experiment.yaml
    uv run python -m scripts.run_experiment --config configs/experiment.yaml --num-item 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag.cli import run_det, run_seg  # noqa: E402


def _pick_ids(voc_root: Path, year: str, split: str, num_item: Optional[int]) -> List[str]:
    """Use the segmentation split — these are images with BOTH .png mask and .xml annotations,
    so they can be attacked by both DAG-Seg and DAG-Det."""
    ids_path = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    ids = ids_path.read_text().split()
    if num_item is not None:
        ids = ids[:num_item]
    return ids


def _seg_cfg(cfg: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    return {
        "model": cfg["seg_model"],
        "weights": cfg.get("seg_weights"),
        "voc_root": cfg["voc_root"],
        "year": cfg["year"],
        "image_id": image_id,
        "shape": cfg.get("seg_shape", "square"),
        "short_side": cfg["seg_short_side"],
        "max_iter": cfg["seg_iter"],
        "step_length": cfg.get("seg_step_length", 0.5),
        "success_ratio": cfg.get("seg_success_ratio", 0.01),
        "device": cfg["device"],
        "output_dir": str(Path(cfg["output_dir"]) / "seg"),
        "seed": cfg.get("seed", 0),
        "verbose": False,
    }


def _det_cfg(cfg: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    return {
        "model": cfg["det_model"],
        "weights": cfg.get("det_weights"),
        "num_classes": cfg.get("det_num_classes", 21),
        "voc_root": cfg["voc_root"],
        "year": cfg["year"],
        "image_id": image_id,
        "short_side": cfg["det_short_side"],
        "rpn_post_nms_top_n": cfg.get("det_rpn_post_nms_top_n", 3000),
        "rpn_nms_thresh": cfg.get("det_rpn_nms_thresh", 0.9),
        "max_iter": cfg["det_iter"],
        "step_length": cfg.get("det_step_length", 0.5),
        "strong_nms_thresh": cfg.get("det_strong_nms_thresh", 0.35),
        "strong_score_thresh": cfg.get("det_strong_score_thresh", 0.8),
        "device": cfg["device"],
        "output_dir": str(Path(cfg["output_dir"]) / "det"),
        "seed": cfg.get("seed", 0),
        "verbose": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--num-item", type=int, default=None, help="override num_item from config")
    p.add_argument("--skip-seg", action="store_true")
    p.add_argument("--skip-det", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.num_item is not None:
        cfg["num_item"] = args.num_item

    ids = _pick_ids(Path(cfg["voc_root"]), cfg["year"], cfg.get("split", "trainval"), cfg.get("num_item"))
    print(f"running on {len(ids)} images: {ids[:5]}{'...' if len(ids) > 5 else ''}")
    print(f"seg_model={cfg['seg_model']}  det_model={cfg['det_model']}")
    print(f"seg_iter={cfg['seg_iter']}  det_iter={cfg['det_iter']}  device={cfg['device']}")
    print()

    summary = {"seg": [], "det": []}
    t0 = time.time()
    for i, image_id in enumerate(ids):
        print(f"==== [{i+1}/{len(ids)}] {image_id} ====")
        if not args.skip_seg:
            t = time.time()
            try:
                m = run_seg(_seg_cfg(cfg, image_id))
                summary["seg"].append(m)
                h = m["active_history"]
                drop = 100 * (h[0] - h[-1]) / max(h[0], 1)
                print(f"  seg: iters={m['iterations']:3d}  drop={drop:5.1f}%  "
                      f"succ={m['succeeded']}  pert={m['max_perturbation_pixel']:5.2f}  "
                      f"({time.time()-t:.1f}s)")
            except Exception as e:
                print(f"  seg ERROR on {image_id}: {e}")
        if not args.skip_det:
            t = time.time()
            try:
                m = run_det(_det_cfg(cfg, image_id))
                summary["det"].append(m)
                h = m["active_history"]
                if h[0] > 0:
                    drop = 100 * (h[0] - h[-1]) / h[0]
                    print(f"  det: iters={m['iterations']:3d}  drop={drop:5.1f}%  "
                          f"succ={m['succeeded']}  pert={m['max_perturbation_pixel']:5.2f}  "
                          f"({time.time()-t:.1f}s)")
                else:
                    print(f"  det: no ground-truth proposals matched")
            except Exception as e:
                print(f"  det ERROR on {image_id}: {e}")

    dt = time.time() - t0
    print(f"\n==== DONE in {dt/60:.1f} min ====")

    # Aggregate summary.
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary → {out_dir / 'summary.json'}")

    for task in ("seg", "det"):
        items = summary[task]
        if not items:
            continue
        succ = sum(1 for m in items if m.get("succeeded"))
        avg_pert = sum(m["max_perturbation_pixel"] for m in items) / len(items)
        avg_iter = sum(m["iterations"] for m in items) / len(items)
        print(f"  {task}: {succ}/{len(items)} succeeded  avg_iter={avg_iter:.0f}  avg_pert={avg_pert:.2f}")


if __name__ == "__main__":
    main()
