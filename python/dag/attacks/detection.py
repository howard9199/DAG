from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..data.targets import generate_mapping
from ..data.voc import coco_to_voc_label, voc_to_coco_label
from ..models.detection import DetWrapper
from ..utils.boxes import box_iou_plus1, nms_plus1
from ..utils.imaging import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class DetAttackResult:
    perturbation: torch.Tensor              # [1, 3, H, W] normalized space
    iterations: int
    succeeded: bool
    active_history: List[int] = field(default_factory=list)
    proposals: torch.Tensor | None = None   # [N, 4] xyxy
    final_box_labels: torch.Tensor | None = None  # [3, N] (orig, target, pred) in VOC indices


def assign_target_det(
    det: DetWrapper,
    x_norm: torch.Tensor,
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    rng: np.random.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Port of code/assign_target_det.m.

    Returns:
        box_label: [3, N] long tensor where
                   row 0 = original VOC class (0 = background),
                   row 1 = adversarial target VOC class,
                   row 2 = current predicted VOC class.
        mapping_t: [num_object_classes+1] long tensor mapping VOC class → adversarial target.
    """
    device = x_norm.device
    N = proposals.shape[0]

    with torch.no_grad():
        cls_logits = det.box_head_forward(x_norm, proposals)  # [N, C_coco]
        # Translate COCO logits → VOC logits by summing over duplicates / picking the matching idx.
        cls_logits_voc = _coco_logits_to_voc(cls_logits)  # [N, 21]
        probs = cls_logits_voc.softmax(-1)  # [N, 21]

    if gt_boxes.numel() == 0:
        box_label = torch.zeros((3, N), dtype=torch.long, device=device)
        mapping, _ = generate_mapping(np.array([], dtype=np.int64), rng=rng)
        return box_label, torch.from_numpy(mapping).to(device)

    # IoU between each GT and each proposal: [num_gt, N]
    iou = box_iou_plus1(gt_boxes.to(device), proposals)

    num_gt = gt_boxes.shape[0]
    assignment_mask = torch.zeros((num_gt, N), dtype=torch.bool, device=device)
    for g in range(num_gt):
        gt_cls = int(gt_labels[g].item())
        positive_iou = iou[g] > 0.1
        positive_score = probs[:, gt_cls] > 0.1
        assignment_mask[g] = positive_iou & positive_score

    assignment_count = assignment_mask.sum(0)  # [N]
    # Default assignment = argmax along GT dim, with tiebreak by IoU × assignment.
    assignment = assignment_mask.long().argmax(0)
    multi = (assignment_count > 1).nonzero(as_tuple=True)[0]
    if multi.numel() > 0:
        scored = iou[:, multi] * assignment_mask[:, multi].float()
        assignment[multi] = scored.argmax(0)

    box_label = torch.zeros((3, N), dtype=torch.long, device=device)
    box_label[0] = gt_labels[assignment].to(device)
    box_label[0, assignment_count == 0] = 0  # background

    gt_classes_present = np.unique(gt_labels.cpu().numpy())
    gt_classes_present = gt_classes_present[gt_classes_present > 0]
    mapping_np, _ = generate_mapping(gt_classes_present, rng=rng)
    mapping_t = torch.from_numpy(mapping_np).to(device)

    box_label[1] = mapping_t[box_label[0]]
    box_label[2] = box_label[0].clone()
    return box_label, mapping_t


def _coco_logits_to_voc(cls_logits: torch.Tensor) -> torch.Tensor:
    """Project Faster R-CNN class logits onto the 21 VOC classes.

    - If the model is already VOC-native (`cls_logits.shape[1] == 21`), returns as-is.
    - Otherwise (COCO-native, 91 classes), picks the COCO column matching each VOC class.
      Index 0 (background) maps to COCO __background__.
    """
    if cls_logits.shape[1] == 21:
        return cls_logits
    device = cls_logits.device
    voc_to_coco = [voc_to_coco_label(k) for k in range(21)]
    indices = torch.tensor(voc_to_coco, dtype=torch.long, device=device)
    return cls_logits.index_select(1, indices)


def fool_det(
    det: DetWrapper,
    x_norm: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    max_iter: int = 150,
    step_length: float = 0.5,
    strong_nms_thresh: float = 0.35,
    strong_score_thresh: float = 0.8,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> DetAttackResult:
    """DAG detection attack — port of code/fooling_det_net.m + back/forward_propogation_det.m.

    Proposals are computed once on the unperturbed image (top-3000 RPN, NMS 0.9 — set on the
    DetWrapper) and held fixed across iterations.
    """
    device = x_norm.device

    proposals, _ = det.get_proposals(x_norm)
    if proposals.numel() == 0:
        return DetAttackResult(
            perturbation=torch.zeros_like(x_norm),
            iterations=0,
            succeeded=True,
            active_history=[],
            proposals=proposals,
            final_box_labels=None,
        )

    box_label, _ = assign_target_det(det, x_norm, proposals, gt_boxes, gt_labels, rng=rng)

    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    step_norm = step_length / (255.0 * std)

    r = torch.zeros_like(x_norm)
    history: List[int] = []
    succeeded = False

    def active_mask() -> torch.Tensor:
        # Original counts ROIs where pred == orig AND orig != background.
        return (box_label[0] == box_label[2]) & (box_label[0] != 0)

    history.append(int(active_mask().sum().item()))
    if verbose:
        print(f"[det] iter 0: active={history[-1]} / {box_label.shape[1]}")

    for itr in range(max_iter):
        if int(active_mask().sum().item()) == 0:
            succeeded = True
            break

        r_req = r.detach().clone().requires_grad_(True)
        cls_logits = det.box_head_forward(x_norm + r_req, proposals)
        cls_logits_voc = _coco_logits_to_voc(cls_logits)  # [N, 21]

        active = active_mask()
        idx_active = active.nonzero(as_tuple=True)[0]
        target_cls = box_label[1, idx_active]
        orig_cls = box_label[0, idx_active]

        loss_target = cls_logits_voc[idx_active, target_cls].sum()
        loss_orig = cls_logits_voc[idx_active, orig_cls].sum()
        loss = loss_target - loss_orig

        dr = torch.autograd.grad(loss, r_req)[0].detach()
        max_abs = dr.abs().max()
        if float(max_abs) == 0.0:
            if verbose:
                print("[det] gradient is zero; stopping early")
            break

        r = r + step_norm * (dr / max_abs)
        # Clip x+r to valid pixel range.
        x_pixel = (x_norm + r) * std + torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        x_pixel = x_pixel.clamp(0.0, 1.0)
        r = (x_pixel - torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)) / std - x_norm
        r = r.detach()

        # Refresh predictions (port of forward_propogation_det.m).
        with torch.no_grad():
            cls_logits_new = det.box_head_forward(x_norm + r, proposals)
            cls_probs_voc = _coco_logits_to_voc(cls_logits_new).softmax(-1)
            new_pred = cls_probs_voc.argmax(-1)
            box_label[2] = new_pred

            # Strong-adversarial revert: if pred == target AND orig != background,
            # run NMS on those boxes, then any post-NMS box with max-prob < strong_score_thresh
            # gets its prediction reverted to the original class.
            strong = (box_label[2] == box_label[1]) & (box_label[0] != 0)
            strong_idx = strong.nonzero(as_tuple=True)[0]
            if strong_idx.numel() > 0:
                strong_boxes = proposals[strong_idx]
                strong_scores = cls_probs_voc[strong_idx].gather(
                    1, box_label[2, strong_idx].unsqueeze(1)
                ).squeeze(1)
                keep_local = nms_plus1(strong_boxes, strong_scores, strong_nms_thresh)
                kept_global = strong_idx[keep_local]
                low_score = strong_scores[keep_local] < strong_score_thresh
                revert_idx = kept_global[low_score]
                box_label[2, revert_idx] = box_label[0, revert_idx]

        n_active = int(active_mask().sum().item())
        history.append(n_active)
        if verbose:
            print(f"[det] iter {itr + 1}: active={n_active}")

    if int(active_mask().sum().item()) == 0:
        succeeded = True

    return DetAttackResult(
        perturbation=r.detach(),
        iterations=len(history) - 1,
        succeeded=succeeded,
        active_history=history,
        proposals=proposals.detach(),
        final_box_labels=box_label.detach(),
    )
