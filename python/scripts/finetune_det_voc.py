"""Fine-tune Faster R-CNN's box head on Pascal VOC.

Backbone + RPN keep their COCO_V1 pretrained weights and are frozen — we only train the
21-class box head. This is enough to give the attack a model whose softmax operates in
VOC class space, removing the COCO→VOC projection gap.

Default: train on VOC2007 trainval (5012 images). On CPU, expect ~1.5 hr per epoch with
default settings. For a faster smoke check, set --max-images to a small number.

Usage:
    python -m scripts.finetune_det_voc \
        --voc-root /tmp/VOCdevkit/VOC2007 --year 2007 \
        --epochs 2 --batch-size 2 \
        --out python/outputs/weights/faster_rcnn_voc.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag.data.voc import load_voc_det_sample, VOC_SEG_CLASSES  # noqa: E402
from dag.models.detection import build_det_model  # noqa: E402


class VOCDetTrainset(data.Dataset):
    def __init__(self, voc_root: Path, year: str, split: str, max_images: int | None = None):
        ids_path = voc_root / "ImageSets" / "Main" / f"{split}.txt"
        ids = ids_path.read_text().split()
        if max_images:
            ids = ids[:max_images]
        self.ids = ids
        self.voc_root = voc_root
        self.year = year

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        sample = load_voc_det_sample(image_id, self.voc_root, self.year)
        # Skip images with no GT (set target to a single zero-area box assigned to BG so DataLoader
        # doesn't blow up; torchvision Faster R-CNN tolerates empty targets via {boxes: [0,4]}).
        boxes = sample.boxes
        labels = sample.labels
        img = torchvision.transforms.functional.to_tensor(sample.image.convert("RGB"))
        target = {
            "boxes": boxes if boxes.numel() else torch.zeros((0, 4), dtype=torch.float32),
            "labels": labels if labels.numel() else torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor([idx]),
        }
        return img, target


def collate(batch):
    imgs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    return imgs, tgts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--voc-root", required=True, type=Path)
    p.add_argument("--year", default="2007")
    p.add_argument("--split", default="trainval")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--max-images", type=int, default=None, help="cap dataset size for smoke runs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    print(f"Loading model with 21-class VOC head (COCO backbone+RPN frozen)...")
    det = build_det_model(
        "fasterrcnn_resnet50_fpn",
        device=args.device,
        num_classes=21,
        rpn_post_nms_top_n=2000,
        rpn_nms_thresh=0.7,
    )
    model = det.model
    model.train()

    # Freeze everything except the box head + box predictor.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("roi_heads.box_head") or name.startswith("roi_heads.box_predictor")
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    train_n = sum(p.numel() for p in trainable)
    print(f"  trainable params: {train_n:,} / {total:,} ({100*train_n/total:.1f}%)")

    ds = VOCDetTrainset(args.voc_root, args.year, args.split, args.max_images)
    print(f"  dataset: {len(ds)} images ({args.split})")
    loader = data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
    )

    optim = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t_start = time.time()
        running = 0.0
        steps = 0
        for step, (imgs, tgts) in enumerate(loader):
            imgs = [im.to(args.device) for im in imgs]
            tgts = [{k: v.to(args.device) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
            steps += 1
            if (step + 1) % args.log_every == 0:
                rate = (step + 1) / (time.time() - t_start)
                print(f"  ep {epoch} step {step + 1}/{len(loader)}: loss={running/steps:.4f}  "
                      f"rate={rate:.2f} batch/s")
        dt = time.time() - t_start
        print(f"epoch {epoch} done in {dt/60:.1f} min — avg loss {running/max(steps,1):.4f}")
        torch.save(model.state_dict(), args.out)
        print(f"saved → {args.out}")

    print("training complete.")


if __name__ == "__main__":
    main()
