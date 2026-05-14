# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo implements **Dense Adversary Generation (DAG)** (Xie et al., ICCV 2017, https://arxiv.org/abs/1703.08603) — an iterative algorithm that crafts adversarial perturbations to fool semantic segmentation networks and object detection networks on Pascal VOC. Two implementations live side-by-side:
- The **original MATLAB on matcaffe** code under `code/`, `prototxt/`, `functions/`, `fetch_data/`, and the `caffe` submodule. Models: FCN-8s, FCN-AlexNet (seg) and Faster R-CNN VGG/ZF (det).
- A **PyTorch port** under `python/` using torchvision models (FCN-ResNet50 / Faster R-CNN ResNet50-FPN). Same algorithm, modern stack, no Caffe dependency.

## Environment / Build

The `caffe` directory is a git submodule pointing at `https://github.com/Microsoft/caffe` (Microsoft's fork is required because Faster R-CNN needs `roi_pooling_layer`). Before running anything:

1. `git submodule update --init --recursive` to populate `caffe/`.
2. Build Caffe with **matcaffe** enabled (see `caffe/Makefile.config.example` after submodule init). GPU build is assumed — `demo.m` calls `caffe.set_mode_gpu()` / `caffe.set_device(0)`.
3. The `weight/` directory is gitignored; running `demo.m` will auto-invoke `fetch_data/fetch_all_models.m` if a selected model's weights are missing. That script downloads the Faster R-CNN weights from OneDrive and the FCN weights from `dl.caffe.berkeleyvision.org` into `../weight/`.

## Running the Demo

From MATLAB, `cd code/` then run `demo`. Configuration is done by editing `code/generate_config.m` — there are no CLI flags:

- `model_select`: one of `seg_fcn_8s`, `seg_fcn_alexnet`, `det_VGG`, `det_ZF`. The string is parsed with `strfind(..., 'det')` / `'seg'` in `demo.m` and `generate_config.m` to branch between the two pipelines, so keep the `det_` / `seg_` prefix when adding new models.
- `MAX_ITER`: 150 for detection, 200 for segmentation by default.
- `step_length`: max pixel value change per iteration (default 0.5).
- For segmentation, `shape` picks the adversarial target mask from `data/{circle,square,strip}.mat`.
- `im_name` selects the input image from `data/` (`2007_000925` for det, `2011_003271` for seg).

`demo.m` adds `../caffe/matlab/` to the MATLAB path itself, so the only required setup is that the submodule is built.

## Code Architecture

Two parallel adversarial pipelines share a common iterative structure: forward pass → identify still-misclassified targets → accumulate gradient `dr` scaled by `step_length / max(|dr|)` into perturbation `r` → repeat until target set is empty or `MAX_ITER` is hit.

### Segmentation pipeline (`fooling_seg_net.m`)
- Input: image `x`, target mask `seg_mask_target` (random label remapping over a geometric shape from `data/{shape}.mat`), original mask `seg_mask_ori`.
- `forward_and_back_propogation_seg.m` does both passes in one call: forward gives current per-pixel argmax `seg_result`; backward computes `res_fool - res_pred` — gradient of the loss pushing toward target labels minus gradient pushing away from currently-predicted labels, restricted to still-wrong pixels. The loop terminates when fewer than 1% of original foreground pixels remain mispredicted.

### Detection pipeline (`fooling_det_net.m`)
- Uses pre-computed RPN proposals from `data/{im_name}_box_3000_{det_VGG|det_ZF}.mat` (top 3000 boxes, NMS 0.9). It does **not** re-run the RPN.
- `assign_target_det.m`: for each proposal, picks a GT class label (via IoU > 0.1 and softmax confidence > 0.1) and a target adversarial label from `mapping` (built by `generate_mapping.m` to permute the set of present classes; background class 1 is kept fixed). Returns `box_label` rows = [original, target, current_prediction].
- `back_propogation_det.m` / `forward_propogation_det.m`: per-iteration backward (gradient toward target − away from current) and forward (refresh current predictions). The loop continues while any box still has `current == original ≠ background`.

### Shared conventions
- Images are stored as `W × H × C` with **BGR** channel order and mean-subtracted (`mean_data = [103.939, 116.779, 123.680]` BGR). Channel swap from RGB and `permute([2 1 3])` for the W/H flip happen in `demo.m` before passing to any pipeline; results are unflipped/un-meaned only at visualization time.
- `softmax_dim.m` is used because matcaffe's outputs come from logit layers (the deploy prototxts in `prototxt/` have the loss layer stripped to allow backward passes).
- `functions/` holds Pascal VOC helpers (`VOCreadrecxml`, `VOCxml2struct`), NMS/IoU (`nms`, `boxoverlap`), and the image utilities (`myresize`, `image_clip`). The detection pipeline resizes so the short side is 600px; segmentation uses native size.

### Prototxt files
The `prototxt/` deploy nets are modified versions of the standard FCN / Faster R-CNN networks with the final loss layer removed so that `net.backward()` can be called from matcaffe. Don't add a loss layer back when editing them.

## Python port (`python/`)

Entry point: `python -m dag.cli {seg|det} --config python/configs/{seg,det}_voc.yaml`.

Layout:
- `python/dag/attacks/{segmentation,detection}.py` — `fool_seg()` and `fool_det()` implement the iterative DAG loop. Each one ports a specific MATLAB file pair:
  - `attacks/segmentation.py` ↔ `code/fooling_seg_net.m` + `code/forward_and_back_propogation_seg.m`
  - `attacks/detection.py` ↔ `code/fooling_det_net.m` + `code/assign_target_det.m` + `back_propogation_det.m` + `forward_propogation_det.m`
- `python/dag/models/{segmentation,detection}.py` — torchvision model builders. `DetWrapper.box_head_forward` lets the attack run **only** the backbone + box head + classifier per iteration (proposals are fixed once at attack start, matching the original's precomputed top-3000 NMS-0.9 RPN proposals).
- `python/dag/utils/{boxes,imaging,viz}.py` — `+1`-convention IoU/NMS (matching `boxoverlap.m` / `nms.m`), normalize/denormalize, pixel-range clipping, matplotlib visualization.
- `python/dag/data/{voc,targets}.py` — VOC XML/PNG loaders, COCO↔VOC label mapping (because torchvision Faster R-CNN is pretrained on COCO), `generate_mapping`, shape-mask loader.
- `python/configs/*.yaml` — defaults; CLI flags override.
- `python/tests/` — unit tests + 5-iter gradient-flow smoke test.

Critical numerics preserved from the MATLAB:
- `step_length / max(|dr|) * dr` update rule. `step_length=0.5` is defined in **pixel** units; we convert to normalized-tensor units via the ImageNet std at the top of each attack so the per-iteration step magnitude matches the original.
- `+1` in IoU/NMS area formulas — the strong-adversarial revert thresholds (`0.35`, `0.8`) were tuned against this convention.
- COCO logits are projected onto VOC class indices via `index_select` so the attack reasons entirely in VOC class space. Background = 0, object classes 1..20.

`data/shapes/{circle,square,strip}.npy` were converted once from `data/{circle,square,strip}.mat` via scipy and committed.

## Notes for Future Edits

- When adding a new model, the `det_` / `seg_` prefix convention in `model_select` is load-bearing — `strfind` branches in `demo.m` and `generate_config.m` depend on it.
- Pre-computed proposals are required for the detection path; there is no code in this repo to regenerate them. If you change the detection backbone, you also need a matching `*_box_3000_*.mat` in `data/`.
- The `try/catch eval(config); catch keyboard; end` pattern in `demo.m`, `fooling_seg_net.m`, and `fooling_det_net.m` drops into MATLAB's interactive debugger on config-load failure — keep that in mind if running non-interactively.
