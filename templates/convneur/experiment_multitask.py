"""
Option 2: Multi-Task Evaluation for ConvNeur (Pipeline Sanity Check)
Evaluates ConvNeur multi-scale features across vision tasks:
1. Image Classification  (CIFAR-10)
2. Object Detection      (Penn-Fudan Detection — small-scale surrogate sanity check, not MS COCO)
3. Semantic Segmentation (Penn-Fudan Segmentation — small-scale surrogate sanity check, not ADE20K)

NOTE:
Penn-Fudan is solely a lightweight surrogate sanity check (~170 images) for validating pipeline execution,
not a direct replacement or benchmark for ADE20K or MS COCO.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import build_model
from model.convneur import ConvNeur
from tiny_imagenet import TinyImageNet
from pennfudan import PennFudanDataset
from downstream_heads import ConvNeurForSegmentation, ConvNeurForDetection


def compute_box_iou(box1, box2):
    """
    box: [cx, cy, w, h] normalized
    Returns IoU
    """
    x1_min = box1[..., 0] - box1[..., 2] / 2
    y1_min = box1[..., 1] - box1[..., 3] / 2
    x1_max = box1[..., 0] + box1[..., 2] / 2
    y1_max = box1[..., 1] + box1[..., 3] / 2

    x2_min = box2[..., 0] - box2[..., 2] / 2
    y2_min = box2[..., 1] - box2[..., 3] / 2
    x2_max = box2[..., 0] + box2[..., 2] / 2
    y2_max = box2[..., 1] + box2[..., 3] / 2

    inter_xmin = torch.max(x1_min, x2_min)
    inter_ymin = torch.max(y1_min, y2_min)
    inter_xmax = torch.min(x1_max, x2_max)
    inter_ymax = torch.min(y1_max, y2_max)

    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area + 1e-6
    return inter_area / union_area


def compute_miou(preds, targets, num_classes=2):
    """
    preds: (B, H, W)
    targets: (B, H, W)
    """
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return np.mean(ious)


def run_classification(device="mps", epochs=1, max_steps=10):
    print("\n" + "="*60)
    print("TASK 1: IMAGE CLASSIFICATION (Tiny-ImageNet-200)")
    print("="*60)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    aug_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    aug_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = TinyImageNet(root="./data", split="train", transform=aug_train, download=False)
    test_ds = TinyImageNet(root="./data", split="val", transform=aug_test, download=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = build_model(scale="m1", num_classes=200, img_size=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    start_time = time.time()
    for step, (x, y) in enumerate(train_loader):
        if step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(test_loader):
            if b_idx >= 15:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += logits.argmax(1).eq(y).sum().item()
            total += y.size(0)

    val_acc = correct / max(total, 1)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"-> Classification Top-1 Acc: {val_acc*100:.2f}% | Params: {params:.2f}M")
    return {"val_acc": val_acc, "params_M": params, "loss": loss.item()}


def run_segmentation(device="mps", epochs=1, max_steps=10):
    print("\n" + "="*60)
    print("TASK 2: SEMANTIC SEGMENTATION (Penn-Fudan Pedestrian Masks)")
    print("="*60)
    train_ds = PennFudanDataset(root="./data", task="segmentation", split="train", img_size=128)
    test_ds = PennFudanDataset(root="./data", task="segmentation", split="test", img_size=128)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    backbone = build_model(scale="m1", num_classes=10, img_size=128)
    model = ConvNeurForSegmentation(backbone, num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step, (x, mask) in enumerate(train_loader):
        if step >= max_steps:
            break
        x, mask = x.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()

    model.eval()
    mious = []
    pixel_accs = []
    with torch.no_grad():
        for b_idx, (x, mask) in enumerate(test_loader):
            if b_idx >= 5:
                break
            x, mask = x.to(device), mask.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            miou = compute_miou(preds.cpu(), mask.cpu(), num_classes=2)
            mious.append(miou)
            acc = preds.eq(mask).float().mean().item()
            pixel_accs.append(acc)

    mean_miou = float(np.mean(mious))
    mean_pixel_acc = float(np.mean(pixel_accs))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"-> Segmentation mIoU: {mean_miou*100:.2f}% | Pixel Acc: {mean_pixel_acc*100:.2f}% | Params: {params:.2f}M")
    return {"mIoU": mean_miou, "pixel_acc": mean_pixel_acc, "params_M": params, "loss": loss.item()}


def run_detection(device="mps", epochs=1, max_steps=10):
    print("\n" + "="*60)
    print("TASK 3: OBJECT DETECTION (Penn-Fudan Bounding Boxes)")
    print("="*60)
    train_ds = PennFudanDataset(root="./data", task="detection", split="train", img_size=128)
    test_ds = PennFudanDataset(root="./data", task="detection", split="test", img_size=128)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    backbone = build_model(scale="m1", num_classes=10, img_size=128)
    model = ConvNeurForDetection(backbone, num_queries=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    model.train()
    for step, (x, target_box) in enumerate(train_loader):
        if step >= max_steps:
            break
        x, target_box = x.to(device), target_box.to(device)
        optimizer.zero_grad()
        pred_boxes, _ = model(x)
        # Primary box prediction
        loss = criterion(pred_boxes.squeeze(1), target_box)
        loss.backward()
        optimizer.step()

    model.eval()
    ious = []
    with torch.no_grad():
        for b_idx, (x, target_box) in enumerate(test_loader):
            if b_idx >= 5:
                break
            x, target_box = x.to(device), target_box.to(device)
            pred_boxes, _ = model(x)
            iou = compute_box_iou(pred_boxes.squeeze(1).cpu(), target_box.cpu()).mean().item()
            ious.append(iou)

    mean_iou = float(np.mean(ious))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"-> Detection Box Mean IoU: {mean_iou*100:.2f}% | Smooth L1 Loss: {loss.item():.4f} | Params: {params:.2f}M")
    return {"mean_box_iou": mean_iou, "loss": loss.item(), "params_M": params}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="baseline_multitask_results")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--max_steps", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "#"*60)
    print("RUNNING OPTION 2: FULL 3-TASK BASELINE EVALUATION (CVPR 2026)")
    print("#"*60)

    # 1. Classification
    cls_res = run_classification(device=args.device, max_steps=args.max_steps)

    # 2. Segmentation
    seg_res = run_segmentation(device=args.device, max_steps=args.max_steps)

    # 3. Detection
    det_res = run_detection(device=args.device, max_steps=args.max_steps)

    summary = {
        "model": "ConvNeur-m1 (Official Backbone)",
        "venue_reference": "CVPR 2026 (arXiv:2603.19567)",
        "tasks": {
            "image_classification": {
                "dataset": "Tiny-ImageNet-200 (Stanford CS231N)",
                "surrogate_of": "ImageNet-1K",
                "top1_acc": cls_res["val_acc"],
                "params_M": cls_res["params_M"],
            },
            "semantic_segmentation": {
                "dataset": "Penn-Fudan Pedestrian Segmentation",
                "surrogate_of": "ADE20K (UPerNet)",
                "mIoU": seg_res["mIoU"],
                "pixel_accuracy": seg_res["pixel_acc"],
                "params_M": seg_res["params_M"],
            },
            "object_detection": {
                "dataset": "Penn-Fudan Bounding Boxes",
                "surrogate_of": "MS COCO (Mask R-CNN)",
                "mean_box_iou": det_res["mean_box_iou"],
                "params_M": det_res["params_M"],
            }
        }
    }

    out_file = os.path.join(args.out_dir, "baseline_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "#"*60)
    print(f"ALL 3 TASKS COMPLETED SUCCESSFULLY!")
    print(f"Summary saved to: {out_file}")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
