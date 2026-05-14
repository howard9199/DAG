# DAG — PyTorch port (Pascal VOC)

PyTorch reimplementation of **Dense Adversary Generation** (Xie et al., ICCV 2017,
[arXiv:1703.08603](https://arxiv.org/abs/1703.08603)). The original MATLAB/matcaffe
code lives under `../code/`, `../prototxt/`, etc. — this directory is a clean port
that runs on a modern stack (PyTorch + torchvision) and targets Pascal VOC for both
the segmentation and detection attacks.

## Installation

```bash
pip install -r requirements.txt
```

The CPU-only PyTorch wheels are sufficient for the smoke tests; for production attacks use a
GPU build (`pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`).

## Datasets

Download the Pascal VOC tar from [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/) and point `voc_root` in the config (or `--voc-root` on the CLI) at the resulting `VOCdevkit/VOCYYYY` directory. Either of these layouts is auto-detected:

```
VOCdevkit/VOC2012/JPEGImages/...
VOCdevkit/VOC2012/SegmentationClass/...
VOCdevkit/VOC2012/Annotations/...
```

## Usage

```bash
# Segmentation attack (defaults to FCN-ResNet50 / VOC2012)
python -m dag.cli seg --config configs/seg_voc.yaml --image-id 2011_003271 --shape square

# Detection attack (defaults to Faster R-CNN ResNet50 FPN / VOC2007)
python -m dag.cli det --config configs/det_voc.yaml --image-id 2007_000925
```

Outputs are written to `{output_dir}/{image_id}/`:
- `orig.png`, `adv.png` — pre/post-attack image.
- `triptych.png` — original / adversarial / rescaled perturbation.
- `adv_seg_overlay.png` (seg) — predicted segmentation overlaid on the adversarial image.
- `adv_det_overlay.png` (det) — predicted boxes (with VOC class labels) on the adversarial image.
- `metrics.json` — `iterations`, `succeeded`, full `active_history`, and max pixel-space perturbation.

## Algorithm

Both pipelines share the same outer loop:
```
r = 0
for it in range(MAX_ITER):
    logits = model(x + r)
    if termination_met:
        break
    loss = (logits at target).sum() - (logits at current_prediction).sum()
    dr = grad(loss, r)
    r += step_length / max(|dr|) * dr
    r = clip_to_pixel_range(x + r)
```

**Segmentation.** Per-iteration the active set is `(target != 0) & (pred != target)`
(pixels still needing flipping). The loss gathers logits at `(target_class, h, w)` and
subtracts logits at `(pred_class, h, w)` over those pixels. Termination: active fraction
< 1% of original foreground.

**Detection.** Proposals are produced once by the RPN (top-3000 with NMS 0.9, matching the
original DAG paper) and held fixed across iterations. `assign_target_det` assigns each ROI
an `(orig_class, target_class, current_pred)` triple. The loss gathers logits at the target
and original class over still-active ROIs. After each step, `forward_propogation_det.m`'s
**strong-adversarial revert** runs: ROIs whose prediction matches the adversarial target are
NMS'd at 0.35; survivors with confidence < 0.8 have their prediction reverted to the original
class, which prevents the loop from terminating on weak adversarial detections.

### COCO → VOC class projection

torchvision's Faster R-CNN is pretrained on COCO (91 classes). We project the COCO class
logits onto the 21 VOC classes by `index_select`-ing the COCO columns that correspond
to the VOC class names. This lets the attack reason entirely in VOC class space
(matching the original algorithm's class indexing) while reusing the pretrained
COCO model. If you have a VOC-finetuned Faster R-CNN checkpoint, pass it via `weights:`
in the config and the attack will use those weights directly.

## Differences from the original

- **Backbones:** FCN-ResNet50 instead of FCN-8s/FCN-AlexNet; Faster R-CNN ResNet50-FPN
  instead of Faster R-CNN VGG/ZF. The original Caffe checkpoints aren't easily portable.
- **RoI pool:** torchvision's `MultiScaleRoIAlign` replaces Caffe's `roi_pooling_layer`.
  The attack itself is unchanged.
- **No BGR/permute pipeline:** torchvision models consume RGB normalized with ImageNet
  mean/std. The original used BGR + mean-subtract in pixel units; we pull `step_length`
  from pixel space into normalized space exactly once at the start of each attack.

## Tests

```bash
pytest tests/
```

Covers `+1`-convention IoU/NMS, target-mapping invariants (no GT class maps to itself or
to background), shape-mask loading, and a 5-iter end-to-end gradient flow check.
