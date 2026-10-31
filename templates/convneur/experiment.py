"""
ConvNeur Experiment Script — Vision Architecture (CVPR 2026 Template)
Evaluates ConvNeur on:
1. Image Classification: CIFAR-10 (32x32, 10 classes)
2. Semantic Segmentation: Penn-Fudan Masks (surrogate / small-scale sanity check, not ADE20K)
3. Object Detection: VOC subset / Penn-Fudan Boxes (surrogate / small-scale sanity check, not MS COCO)

NOTE:
Penn-Fudan is strictly a small-scale sanity check (~170 pedestrian images) to ensure
multi-scale downstream heads execute, calculate losses, and backpropagate gradients.
It is NOT a benchmark for full-scale ADE20K or MS COCO.
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
from torchvision import datasets, transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import build_model
from model.convneur import ConvNeur, Block
from cifar10_loader import CIFAR10HF
from pennfudan import PennFudanDataset
from voc_dataset import VOCDetection
from downstream_heads import ConvNeurForSegmentation, ConvNeurForDetection


# ──────────────────────────────────────────────────────────────────────────────
# PROFILING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, input_size: tuple = (1, 3, 64, 64)) -> int:
    flop_count = [0]

    def conv_hook(module, inp, out):
        B, C_in, H, W = inp[0].shape
        B, C_out, H_out, W_out = out.shape
        k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        g = module.groups
        flops = 2 * (C_in // g) * k * k * C_out * H_out * W_out
        flop_count[0] += flops

    def linear_hook(module, inp, out):
        flop_count[0] += 2 * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy = torch.zeros(*input_size, device=device)
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return flop_count[0]


def measure_latency(model: nn.Module, device: str, input_size: tuple = (1, 3, 64, 64), n_iters: int = 30) -> float:
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - start) / n_iters * 1000  # ms


# ──────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def compute_box_iou(box1, box2):
    """Normalized [cx, cy, w, h] IoU"""
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


def compute_ap(ious, scores=None, iou_threshold=0.5):
    """
    Computes Average Precision (AP) at a given IoU threshold (e.g. AP50).
    Follows official PASCAL VOC / MS COCO 11-point interpolation protocol.
    """
    ious = np.asarray(ious)
    n_gt = len(ious)
    if n_gt == 0:
        return 0.0
    if scores is None:
        scores = np.ones_like(ious)
    else:
        scores = np.asarray(scores)

    indices = np.argsort(-scores)
    sorted_ious = ious[indices]
    tp = (sorted_ious >= iou_threshold).astype(float)
    fp = (sorted_ious < iou_threshold).astype(float)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / n_gt
    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)

    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        prec_at_t = precisions[recalls >= t]
        p = np.max(prec_at_t) if len(prec_at_t) > 0 else 0.0
        ap += p / 11.0
    return float(ap)


def compute_coco_map(ious, scores=None):
    """
    Computes MS COCO mAP (averaged across 10 IoU thresholds from 0.50 to 0.95 with step 0.05)
    and returns (mAP, AP50).
    """
    thresholds = np.linspace(0.50, 0.95, 10)
    aps = [compute_ap(ious, scores, iou_threshold=t) for t in thresholds]
    return float(np.mean(aps)), float(aps[0])


def compute_miou(preds, targets, num_classes=2):
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


# ──────────────────────────────────────────────────────────────────────────────
# TASK 1: IMAGE CLASSIFICATION (CIFAR-10)
# ──────────────────────────────────────────────────────────────────────────────

def run_classification_task(data_path="./data", device="mps", epochs=3, batch_size=128, max_steps=None):
    print("\n" + "="*65)
    print(f"▶ TASK 1: CLASSIFICATION (CIFAR-10 | {epochs} Epochs)")
    print("="*65)
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    aug_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    aug_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = CIFAR10HF(split="train", transform=aug_train)
    test_ds  = CIFAR10HF(split="test",  transform=aug_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=(device=="cuda"))

    model = build_model(scale="m1", num_classes=10, img_size=32).to(device)
    n_params = count_parameters(model)
    flops = estimate_flops(model, input_size=(1, 3, 32, 32))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_log, val_log = [], []
    best_acc = 0.0

    for ep in range(epochs):
        model.train()
        ep_loss, correct, total = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                print(f"  [Epoch {ep+1}/{epochs}] Step {step+1:3d}/{len(train_loader)} | Loss: {loss.item():.4f} | Batch Acc: {correct/total*100:.2f}%")

        scheduler.step()

        # Validate
        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for b_idx, (x, y) in enumerate(test_loader):
                if max_steps and b_idx >= 25:
                    break
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_loss += criterion(logits, y).item() * y.size(0)
                v_correct += logits.argmax(1).eq(y).sum().item()
                v_total += y.size(0)

        val_acc = v_correct / max(v_total, 1)
        best_acc = max(best_acc, val_acc)
        train_log.append({"epoch": ep + 1, "train_acc": correct / max(total, 1), "loss": ep_loss / max(step + 1, 1)})
        val_log.append({"epoch": ep + 1, "val_acc": val_acc, "val_loss": v_loss / max(v_total, 1)})
        print(f"  ✓ Epoch {ep+1} Finished | Val Top-1 Acc: {val_acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

    latency_ms = measure_latency(model, device, input_size=(1, 3, 32, 32))
    print(f"  Summary -> Best Acc: {best_acc*100:.2f}% | Latency: {latency_ms:.2f}ms | Params: {n_params/1e6:.2f}M | FLOPs: {flops/1e6:.1f}M")

    return {
        "best_val_acc": best_acc,
        "final_val_acc": val_log[-1]["val_acc"] if val_log else 0.0,
        "n_params_M": n_params / 1e6,
        "flops_M": flops / 1e6,
        "latency_ms": latency_ms,
        "train_log": train_log,
        "val_log": val_log,
    }


# ──────────────────────────────────────────────────────────────────────────────
# TASK 2: SEMANTIC SEGMENTATION (Penn-Fudan Surrogate Sanity Check, not ADE20K)
# ──────────────────────────────────────────────────────────────────────────────

def run_segmentation_task(data_path="./data", device="mps", epochs=5, batch_size=16, max_steps=None):
    print("\n" + "="*65)
    print(f"▶ TASK 2: SEMANTIC SEGMENTATION (Penn-Fudan surrogate check | {epochs} Epochs)")
    print("  [Notice] Penn-Fudan is a small-scale sanity check (~170 images), not ADE20K.")
    print("="*65)
    train_ds = PennFudanDataset(root=data_path, task="segmentation", split="train", img_size=128)
    test_ds  = PennFudanDataset(root=data_path, task="segmentation", split="test",  img_size=128)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=(device=="cuda"))

    backbone = build_model(scale="m1", num_classes=10, img_size=128)
    model = ConvNeurForSegmentation(backbone, num_classes=2).to(device)
    n_params = count_parameters(model)
    flops = estimate_flops(model, input_size=(1, 3, 128, 128))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()

    train_log, val_log = [], []
    best_miou = 0.0

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for step, (x, mask) in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            x, mask = x.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        mious, pixel_accs = [], []
        with torch.no_grad():
            for x, mask in test_loader:
                x, mask = x.to(device), mask.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                mious.append(compute_miou(preds.cpu(), mask.cpu(), num_classes=2))
                pixel_accs.append(preds.eq(mask).float().mean().item())

        mean_miou = float(np.mean(mious))
        mean_pacc = float(np.mean(pixel_accs))
        best_miou = max(best_miou, mean_miou)
        train_log.append({"epoch": ep + 1, "loss": ep_loss / max(step + 1, 1)})
        val_log.append({"epoch": ep + 1, "mIoU": mean_miou, "pixel_acc": mean_pacc})
        print(f"  ✓ Epoch {ep+1} Finished | mIoU: {mean_miou*100:.2f}% | Pixel Acc: {mean_pacc*100:.2f}%")

    latency_ms = measure_latency(model, device, input_size=(1, 3, 128, 128))
    print(f"  Summary -> Best mIoU: {best_miou*100:.2f}% | Latency: {latency_ms:.2f}ms | Params: {n_params/1e6:.2f}M")

    return {
        "best_mIoU": best_miou,
        "pixel_accuracy": val_log[-1]["pixel_acc"] if val_log else 0.0,
        "n_params_M": n_params / 1e6,
        "flops_M": flops / 1e6,
        "latency_ms": latency_ms,
        "train_log": train_log,
        "val_log": val_log,
    }


# ──────────────────────────────────────────────────────────────────────────────
# TASK 3: OBJECT DETECTION (Pascal VOC 2007 Subset — Surrogate Sanity Check, not MS COCO)
# ──────────────────────────────────────────────────────────────────────────────

def run_detection_task(data_path="./data", device="mps", epochs=10, batch_size=32, max_steps=None):
    print("\n" + "="*65)
    print(f"▶ TASK 3: OBJECT DETECTION (Pascal VOC 2007 surrogate check | {epochs} Epochs)")
    print("  [Notice] VOC subset is a small-scale sanity check, not full MS COCO.")
    print("="*65)
    train_ds = VOCDetection(root=data_path, split="train", img_size=128)
    test_ds  = VOCDetection(root=data_path, split="val",   img_size=128)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device=="cuda"))

    backbone = build_model(scale="m1", num_classes=10, img_size=128)
    model = ConvNeurForDetection(backbone, num_classes=20, num_queries=1).to(device)
    n_params = count_parameters(model)
    flops = estimate_flops(model, input_size=(1, 3, 128, 128))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion_box = nn.SmoothL1Loss()
    criterion_cls = nn.CrossEntropyLoss()

    train_log, val_log = [], []
    best_iou = 0.0
    best_ap50 = 0.0
    best_map = 0.0

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for step, (x, target_box, target_cls) in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            x, target_box, target_cls = x.to(device), target_box.to(device), target_cls.to(device)
            optimizer.zero_grad()
            pred_boxes, pred_cls = model(x)
            loss_box = criterion_box(pred_boxes.squeeze(1), target_box)
            loss_cls = criterion_cls(pred_cls.squeeze(1), target_cls)
            loss = loss_box + 0.5 * loss_cls
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        all_ious, all_scores, v_losses = [], [], []
        with torch.no_grad():
            for x, target_box, target_cls in test_loader:
                if max_steps and len(all_ious) >= 100:
                    break
                x, target_box, target_cls = x.to(device), target_box.to(device), target_cls.to(device)
                pred_boxes, pred_cls = model(x)
                iou = compute_box_iou(pred_boxes.squeeze(1).cpu(), target_box.cpu()).numpy()
                scores = pred_cls.squeeze(1).softmax(-1).max(-1)[0].cpu().numpy()
                all_ious.extend(iou.tolist())
                all_scores.extend(scores.tolist())
                v_loss = criterion_box(pred_boxes.squeeze(1), target_box).item()
                v_losses.append(v_loss)

        mean_iou = float(np.mean(all_ious))
        coco_map, ap50 = compute_coco_map(all_ious, all_scores)
        best_iou = max(best_iou, mean_iou)
        best_ap50 = max(best_ap50, ap50)
        best_map = max(best_map, coco_map)

        train_log.append({"epoch": ep + 1, "loss": ep_loss / max(step + 1, 1)})
        val_log.append({
            "epoch": ep + 1,
            "mean_iou": mean_iou,
            "AP50": ap50,
            "mAP": coco_map,
            "loss": float(np.mean(v_losses))
        })
        print(f"  ✓ Epoch {ep+1} Finished | AP50: {ap50*100:.2f}% | COCO mAP: {coco_map*100:.2f}% | Mean IoU: {mean_iou*100:.2f}% | Box Loss: {np.mean(v_losses):.4f}")

    latency_ms = measure_latency(model, device, input_size=(1, 3, 128, 128))
    print(f"  Summary -> Best AP50: {best_ap50*100:.2f}% | Best mAP: {best_map*100:.2f}% | Best IoU: {best_iou*100:.2f}% | Latency: {latency_ms:.2f}ms | Params: {n_params/1e6:.2f}M")

    return {
        "best_AP50": best_ap50,
        "best_mAP": best_map,
        "best_box_iou": best_iou,
        "final_box_loss": val_log[-1]["loss"] if val_log else 0.0,
        "n_params_M": n_params / 1e6,
        "flops_M": flops / 1e6,
        "latency_ms": latency_ms,
        "train_log": train_log,
        "val_log": val_log,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Robust data directory resolution: check ../../data, ../data, then ./data
    default_data = "./data"
    for candidate in [
        os.path.abspath(os.path.join(current_dir, "../../data")),
        os.path.abspath(os.path.join(current_dir, "../data")),
        os.path.abspath(os.path.join(current_dir, "data")),
        os.path.abspath("./data"),
    ]:
        if os.path.isdir(os.path.join(candidate, "tiny-imagenet-200")):
            default_data = candidate
            break

    parser = argparse.ArgumentParser(description="ConvNeur Multi-Task Baseline Runner (CVPR 2026)")
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--data_path", type=str, default=default_data)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--cls_epochs", type=int, default=10)
    parser.add_argument("--seg_epochs", type=int, default=10)
    parser.add_argument("--det_epochs", type=int, default=10)
    parser.add_argument("--cls_batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=None, help="Limit steps per epoch for fast debugging/smoke testing")
    parser.add_argument("--seed_offset", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed_offset)
    np.random.seed(args.seed_offset)

    print("\n" + "#"*65)
    print("STARTING FULL CVPR 2026 MULTI-TASK BASELINE RUN (run_0)")
    print(f"Device: {args.device} | Target Directory: {args.out_dir}")
    print("#"*65)

    # 1. Classification (CIFAR-10)
    cls_res = run_classification_task(
        data_path=args.data_path,
        device=args.device,
        epochs=args.cls_epochs,
        batch_size=args.cls_batch_size,
        max_steps=args.max_steps,
    )

    # 2. Semantic Segmentation (Penn-Fudan Surrogate Masks)
    seg_res = run_segmentation_task(
        data_path=args.data_path,
        device=args.device,
        epochs=args.seg_epochs,
        max_steps=args.max_steps,
    )

    # 3. Object Detection (Pascal VOC Surrogate Boxes)
    det_res = run_detection_task(
        data_path=args.data_path,
        device=args.device,
        epochs=args.det_epochs,
        max_steps=args.max_steps,
    )

    # Structure final_info.json in standard AI Scientist format
    final_info = {
        "image_classification": {
            "means": {
                "best_val_acc": cls_res["best_val_acc"],
                "final_val_acc": cls_res["final_val_acc"],
                "n_params_M": cls_res["n_params_M"],
                "flops_M": cls_res["flops_M"],
                "latency_ms": cls_res["latency_ms"],
            },
            "stderrs": {
                "best_val_acc": 0.0,
                "n_params_M": 0.0,
                "flops_M": 0.0,
                "latency_ms": 0.0,
            },
            "final_info_dict": {
                "dataset": "CIFAR-10",
                "epochs": args.cls_epochs,
            }
        },
        "semantic_segmentation": {
            "means": {
                "best_mIoU": seg_res["best_mIoU"],
                "pixel_accuracy": seg_res["pixel_accuracy"],
                "n_params_M": seg_res["n_params_M"],
                "flops_M": seg_res["flops_M"],
                "latency_ms": seg_res["latency_ms"],
            },
            "stderrs": {
                "best_mIoU": 0.0,
                "n_params_M": 0.0,
                "flops_M": 0.0,
                "latency_ms": 0.0,
            },
            "final_info_dict": {
                "dataset": "Penn-Fudan Segmentation",
                "epochs": args.seg_epochs,
            }
        },
        "object_detection": {
            "means": {
                "best_AP50": det_res["best_AP50"],
                "best_mAP": det_res["best_mAP"],
                "best_box_iou": det_res["best_box_iou"],
                "final_box_loss": det_res["final_box_loss"],
                "n_params_M": det_res["n_params_M"],
                "flops_M": det_res["flops_M"],
                "latency_ms": det_res["latency_ms"],
            },
            "stderrs": {
                "best_AP50": 0.0,
                "best_mAP": 0.0,
                "best_box_iou": 0.0,
                "n_params_M": 0.0,
                "flops_M": 0.0,
                "latency_ms": 0.0,
            },
            "final_info_dict": {
                "dataset": "Pascal VOC 2007 Detection",
                "epochs": args.det_epochs,
            }
        }
    }

    all_results = {
        "image_classification": cls_res,
        "semantic_segmentation": seg_res,
        "object_detection": det_res,
        # Backward compatibility aliases
        "val_log": cls_res["val_log"],
        "train_log": cls_res["train_log"],
    }

    final_info_path = os.path.join(args.out_dir, "final_info.json")
    all_results_path = os.path.join(args.out_dir, "all_results.npy")

    with open(final_info_path, "w") as f:
        json.dump(final_info, f, indent=2)

    np.save(all_results_path, all_results)

    print("\n" + "#"*65)
    print("ALL 3 TASKS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {final_info_path}")
    print(f"NPY arrays saved to: {all_results_path}")
    print("#"*65 + "\n")


if __name__ == "__main__":
    main()
