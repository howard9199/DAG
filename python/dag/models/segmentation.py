from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision


def build_seg_model(name: str = "fcn_resnet50", device: str = "cuda", weights_path: str | None = None) -> nn.Module:
    """Build a torchvision segmentation model with VOC-compatible 21-class output.

    Default: FCN-ResNet50 with COCO_WITH_VOC_LABELS_V1 weights — directly outputs the 21 VOC
    classes used by the original DAG paper.
    """
    if name == "fcn_resnet50":
        weights = torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
    elif name == "fcn_resnet101":
        weights = torchvision.models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        model = torchvision.models.segmentation.fcn_resnet101(weights=weights)
    elif name == "deeplabv3_resnet50":
        weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
    elif name == "deeplabv3_resnet101":
        weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
    else:
        raise ValueError(f"Unknown segmentation model: {name}")

    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)

    model.eval()
    return model.to(device)
