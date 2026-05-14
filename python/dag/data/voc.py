from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch
from PIL import Image


# Pascal VOC 20 object classes + background as class 0.
VOC_OBJECT_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
VOC_SEG_CLASSES = ("background",) + VOC_OBJECT_CLASSES  # 21
VOC_DET_CLASSES = VOC_SEG_CLASSES  # same indexing in the attack


# COCO 91-class indexing as used by torchvision Faster R-CNN (1-indexed; index 0 = __background__).
# Mapping VOC class name → COCO class index in torchvision's labels.
# Reference: torchvision.models.detection.faster_rcnn (COCO_INSTANCE_CATEGORY_NAMES).
_COCO_NAMES = (
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)
# VOC name → equivalent COCO name (some are different words).
_VOC_TO_COCO_NAME = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "sofa": "couch",
    "train": "train",
    "tvmonitor": "tv",
}


def coco_to_voc_label(coco_label: int) -> int:
    """Map a torchvision COCO class index to VOC class index (0..20). Returns 0 if not in VOC."""
    if coco_label < 0 or coco_label >= len(_COCO_NAMES):
        return 0
    coco_name = _COCO_NAMES[coco_label]
    for voc_idx, voc_name in enumerate(VOC_SEG_CLASSES):
        if voc_idx == 0:
            continue
        if _VOC_TO_COCO_NAME.get(voc_name) == coco_name:
            return voc_idx
    return 0


def voc_to_coco_label(voc_label: int) -> int:
    """Map VOC class index (0..20) to torchvision COCO class index. Background → 0."""
    if voc_label == 0:
        return 0
    voc_name = VOC_SEG_CLASSES[voc_label]
    coco_name = _VOC_TO_COCO_NAME.get(voc_name)
    if coco_name is None:
        return 0
    try:
        return _COCO_NAMES.index(coco_name)
    except ValueError:
        return 0


class DetSample(NamedTuple):
    image: Image.Image
    boxes: torch.Tensor       # [N, 4] xyxy in image pixels
    labels: torch.Tensor      # [N] VOC class indices (1..20)


class SegSample(NamedTuple):
    image: Image.Image
    seg_mask: np.ndarray      # HxW int, VOC class indices (0..20, 255 = ignore)


def _resolve_voc_root(voc_root: Optional[Path], year: str) -> Path:
    """Find VOCdevkit/VOCYYYY directory. Accepts either VOCdevkit/, VOCdevkit/VOCYYYY/, or a parent."""
    if voc_root is None:
        raise FileNotFoundError("VOC dataset root not provided (--voc-root or config).")
    voc_root = Path(voc_root)
    for candidate in (voc_root, voc_root / f"VOC{year}", voc_root / "VOCdevkit" / f"VOC{year}"):
        if (candidate / "JPEGImages").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate VOC{year} at {voc_root}. Expected JPEGImages/ inside one of: "
        f"{voc_root}, {voc_root}/VOC{year}, or {voc_root}/VOCdevkit/VOC{year}."
    )


def load_voc_det_sample(image_id: str, voc_root: Optional[Path], year: str = "2007") -> DetSample:
    """Load a VOCDetection sample from disk by image_id (e.g. '2007_000925')."""
    root = _resolve_voc_root(voc_root, year)
    img = Image.open(root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
    ann_path = root / "Annotations" / f"{image_id}.xml"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")
    tree = ET.parse(ann_path)
    boxes, labels = [], []
    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(VOC_SEG_CLASSES)}
    for obj in tree.iter("object"):
        difficult = obj.find("difficult")
        if difficult is not None and difficult.text == "1":
            continue
        name = obj.find("name").text
        if name not in name_to_idx:
            continue
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        boxes.append([x1, y1, x2, y2])
        labels.append(name_to_idx[name])
    return DetSample(
        image=img,
        boxes=torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
        labels=torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
    )


def load_voc_seg_sample(image_id: str, voc_root: Optional[Path], year: str = "2012") -> SegSample:
    """Load a VOCSegmentation sample by image_id (e.g. '2011_003271')."""
    root = _resolve_voc_root(voc_root, year)
    img = Image.open(root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
    mask_path = root / "SegmentationClass" / f"{image_id}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Segmentation mask not found: {mask_path}")
    mask = np.array(Image.open(mask_path))
    return SegSample(image=img, seg_mask=mask.astype(np.int64))
