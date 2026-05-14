from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList


@dataclass
class DetWrapper:
    """Convenience wrapper around a torchvision Faster R-CNN exposing:

    - `get_proposals(image)`: run backbone+RPN once, return top-N proposals (post-NMS).
    - `box_head_forward(image, proposals)`: run backbone + box head + classifier on the
       supplied (image, proposals) pair. Returns class_logits [N, num_coco_classes].
    """
    model: nn.Module
    device: str

    @property
    def num_coco_classes(self) -> int:
        return self.model.roi_heads.box_predictor.cls_score.out_features

    def _images_to_imagelist(self, image: torch.Tensor) -> Tuple[ImageList, List[Tuple[int, int]]]:
        # image: [1, 3, H, W] already normalized for torchvision's transforms.
        # We bypass GeneralizedRCNNTransform because we want direct access in image coords.
        h, w = image.shape[-2:]
        image_sizes = [(h, w)]
        return ImageList(image, image_sizes), image_sizes

    @torch.no_grad()
    def get_proposals(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run backbone + RPN. Returns (proposals [N, 4], objectness scores [N])."""
        image_list, image_sizes = self._images_to_imagelist(image)
        features = self.model.backbone(image)
        proposals, _ = self.model.rpn(image_list, features, targets=None)
        proposals = proposals[0]
        # RPN doesn't return scores in the same call; recompute objectness by re-running last NMS
        # is overkill — use uniform scores since DAG only needs the proposal *set*.
        scores = torch.ones(proposals.shape[0], device=proposals.device)
        return proposals, scores

    def box_head_forward(
        self,
        image: torch.Tensor,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        """Run backbone + box head + box predictor with supplied proposals.

        Gradients flow through `image`. `proposals` are treated as constants (fixed at attack
        start, matching the original DAG which uses a precomputed proposal set).
        """
        h, w = image.shape[-2:]
        features = self.model.backbone(image)
        box_features = self.model.roi_heads.box_roi_pool(
            features, [proposals], [(h, w)]
        )
        box_features = self.model.roi_heads.box_head(box_features)
        class_logits, _ = self.model.roi_heads.box_predictor(box_features)
        return class_logits

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> List[dict]:
        """Run full detection pipeline (for visualization)."""
        outputs = self.model(image)
        return outputs


def build_det_model(
    name: str = "fasterrcnn_resnet50_fpn",
    device: str = "cuda",
    weights_path: str | None = None,
    num_classes: int = 91,
    rpn_pre_nms_top_n: int = 6000,
    rpn_post_nms_top_n: int = 3000,
    rpn_nms_thresh: float = 0.9,
) -> DetWrapper:
    """Build a Faster R-CNN matching the original DAG's proposal regime.

    Args:
        num_classes: 91 (COCO native, default) or 21 (VOC native, including background).
            When 21, the box_predictor head is replaced with a fresh 21-class predictor.
            Backbone + RPN keep their COCO-pretrained weights; train the head on VOC
            via scripts/finetune_det_voc.py, then load via `weights_path`.
        weights_path: optional .pt of a full model state_dict to load on top.
    """
    if name == "fasterrcnn_resnet50_fpn":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n,
            rpn_nms_thresh=rpn_nms_thresh,
        )
    elif name == "fasterrcnn_resnet50_fpn_v2":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n,
            rpn_nms_thresh=rpn_nms_thresh,
        )
    else:
        raise ValueError(f"Unknown detection model: {name}")

    if num_classes != 91:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        old_cls = model.roi_heads.box_predictor.cls_score
        old_reg = model.roi_heads.box_predictor.bbox_pred
        new_pred = FastRCNNPredictor(in_features, num_classes)
        # Warm-start: copy the COCO weight rows corresponding to each VOC class into the
        # new head. Background (idx 0) maps to COCO __background__ (idx 0). This skips
        # most of the head-training cost — the head starts already class-aware for the
        # 20 VOC categories that have COCO equivalents.
        if num_classes == 21:
            from ..data.voc import voc_to_coco_label
            with torch.no_grad():
                for voc_idx in range(num_classes):
                    coco_idx = voc_to_coco_label(voc_idx)
                    new_pred.cls_score.weight[voc_idx] = old_cls.weight[coco_idx]
                    new_pred.cls_score.bias[voc_idx] = old_cls.bias[coco_idx]
                    # bbox_pred is 4 logits per class (laid out as [c0_x, c0_y, c0_w, c0_h, c1_x, ...])
                    new_pred.bbox_pred.weight[4 * voc_idx:4 * (voc_idx + 1)] = (
                        old_reg.weight[4 * coco_idx:4 * (coco_idx + 1)]
                    )
                    new_pred.bbox_pred.bias[4 * voc_idx:4 * (voc_idx + 1)] = (
                        old_reg.bias[4 * coco_idx:4 * (coco_idx + 1)]
                    )
        model.roi_heads.box_predictor = new_pred

    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)

    model.eval()
    model.to(device)
    return DetWrapper(model=model, device=device)
