"""Run DAG-Seg + DAG-Det across multiple models on a set of VOC images.

Reads `configs/experiment.yaml` (path overridable via --config). The config specifies
lists of `seg_models` and `det_models`; each model runs on the same image set.

Output layout: `{output_dir}/{task}/{model_name}/{image_id}/`.

With `skip_existing: true` the script skips an (image, model) pair if its metrics.json
already exists, so the run is resumable.

Examples:
    uv run python -m scripts.run_experiment --config configs/experiment.yaml
    uv run python -m scripts.run_experiment --config configs/experiment.yaml --num-item 5
    uv run python -m scripts.run_experiment --config configs/experiment.yaml --skip-det
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
    """Use the seg trainval split — these have both .png mask and .xml annotations."""
    ids_path = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    ids = ids_path.read_text().split()
    if num_item is not None:
        ids = ids[:num_item]
    return ids


def _seg_cfg(cfg: Dict[str, Any], model_name: str, image_id: str) -> Dict[str, Any]:
    return {
        "model": model_name,
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
        "output_dir": str(Path(cfg["output_dir"]) / "seg" / model_name),
        "seed": cfg.get("seed", 0),
        "verbose": False,
    }


def _det_cfg(cfg: Dict[str, Any], model_name: str, image_id: str) -> Dict[str, Any]:
    return {
        "model": model_name,
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
        "output_dir": str(Path(cfg["output_dir"]) / "det" / model_name),
        "seed": cfg.get("seed", 0),
        "verbose": False,
    }


def _already_done(out_root: Path, task: str, model_name: str, image_id: str) -> bool:
    return (out_root / task / model_name / image_id / "metrics.json").exists()


def _run_model_pass(
    task: str,
    model_name: str,
    ids: List[str],
    cfg: Dict[str, Any],
    runner,
    cfg_builder,
    skip_existing: bool,
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    out_root = Path(cfg["output_dir"])
    print(f"\n[{task.upper()}] model = {model_name}")
    t_model = time.time()
    n_done, n_skip, n_err = 0, 0, 0
    for i, image_id in enumerate(ids):
        if skip_existing and _already_done(out_root, task, model_name, image_id):
            n_skip += 1
            continue
        t = time.time()
        try:
            m = runner(cfg_builder(cfg, model_name, image_id))
            summary.append({**m, "model": model_name})
            h = m["active_history"]
            drop = 100 * (h[0] - h[-1]) / max(h[0], 1) if h else 0.0
            n_done += 1
            if (i + 1) % 10 == 0 or i < 3 or i + 1 == len(ids):
                print(f"  [{i+1:4d}/{len(ids)}] {image_id}: iters={m['iterations']:3d}  "
                      f"drop={drop:5.1f}%  succ={m['succeeded']}  "
                      f"pert={m['max_perturbation_pixel']:5.2f}  ({time.time()-t:.1f}s)")
        except Exception as e:
            n_err += 1
            print(f"  [{i+1:4d}/{len(ids)}] {image_id}: ERROR — {type(e).__name__}: {e}")
    print(f"  → {task}/{model_name}: done={n_done} skipped={n_skip} errors={n_err}  "
          f"elapsed={(time.time()-t_model)/60:.1f} min")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--num-item", type=int, default=None, help="override num_item")
    p.add_argument("--skip-seg", action="store_true")
    p.add_argument("--skip-det", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="ignore skip_existing in the config")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.num_item is not None:
        cfg["num_item"] = args.num_item

    seg_models = list(cfg.get("seg_models") or [])
    det_models = list(cfg.get("det_models") or [])
    if not seg_models and not det_models:
        raise SystemExit("config must specify at least one of seg_models or det_models")

    skip_existing = (not args.force) and bool(cfg.get("skip_existing", True))

    ids = _pick_ids(Path(cfg["voc_root"]), cfg["year"], cfg.get("split", "trainval"), cfg.get("num_item"))
    print(f"dataset: {len(ids)} images (split={cfg.get('split','trainval')})")
    print(f"seg_models: {seg_models}  det_models: {det_models}")
    print(f"seg_iter={cfg['seg_iter']}  det_iter={cfg['det_iter']}  device={cfg['device']}  "
          f"skip_existing={skip_existing}")

    all_summary: Dict[str, List[Dict[str, Any]]] = {"seg": [], "det": []}
    t_total = time.time()

    if not args.skip_seg:
        for model_name in seg_models:
            all_summary["seg"].extend(
                _run_model_pass("seg", model_name, ids, cfg, run_seg, _seg_cfg, skip_existing)
            )

    if not args.skip_det:
        for model_name in det_models:
            all_summary["det"].extend(
                _run_model_pass("det", model_name, ids, cfg, run_det, _det_cfg, skip_existing)
            )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\n==== DONE in {(time.time()-t_total)/60:.1f} min ====")
    print(f"summary → {out_dir / 'summary.json'}")

    for task in ("seg", "det"):
        items = all_summary[task]
        if not items:
            continue
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            by_model.setdefault(it["model"], []).append(it)
        for model_name, ms in by_model.items():
            succ = sum(1 for m in ms if m.get("succeeded"))
            avg_pert = sum(m["max_perturbation_pixel"] for m in ms) / len(ms)
            avg_iter = sum(m["iterations"] for m in ms) / len(ms)
            print(f"  {task}/{model_name}: {succ}/{len(ms)} succeeded  "
                  f"avg_iter={avg_iter:.0f}  avg_pert={avg_pert:.2f}")


if __name__ == "__main__":
    main()
