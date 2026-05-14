import torch

from dag.utils.boxes import box_iou_plus1, nms_plus1


def test_iou_plus1_identical_boxes():
    a = torch.tensor([[0.0, 0.0, 9.0, 9.0]])  # 10x10 pixel box (inclusive)
    iou = box_iou_plus1(a, a)
    assert iou.shape == (1, 1)
    assert abs(iou.item() - 1.0) < 1e-6


def test_iou_plus1_disjoint():
    a = torch.tensor([[0.0, 0.0, 9.0, 9.0]])
    b = torch.tensor([[20.0, 20.0, 29.0, 29.0]])
    iou = box_iou_plus1(a, b)
    assert iou.item() == 0.0


def test_iou_plus1_half_overlap():
    # Two 10x10 boxes overlapping in a 10x5 strip.
    a = torch.tensor([[0.0, 0.0, 9.0, 9.0]])      # area = 100
    b = torch.tensor([[0.0, 5.0, 9.0, 14.0]])     # area = 100
    # Intersection: x ∈ [0, 9], y ∈ [5, 9] → width 10, height 5 → area 50
    # Union = 100 + 100 - 50 = 150 → IoU = 1/3
    iou = box_iou_plus1(a, b).item()
    assert abs(iou - (50 / 150)) < 1e-6


def test_nms_plus1_basic():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 9.0, 9.0],      # A: high score, area 100
            [1.0, 1.0, 10.0, 10.0],    # B: overlaps A, lower score → suppressed
            [50.0, 50.0, 59.0, 59.0],  # C: disjoint, kept
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms_plus1(boxes, scores, 0.5)
    assert set(keep.tolist()) == {0, 2}
