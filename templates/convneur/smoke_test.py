#!/usr/bin/env python3
"""
Smoke Test for ConvNeur Architecture & Pipeline
Verifies:
1. Module imports (ConvNeur, Block, build_model, downstream heads, CIFAR-10 loader)
2. Model forward pass on CIFAR-10 resolution (32x32)
3. Autograd backward pass & non-zero gradients on active parameters
4. Memory state propagation across stages (validating neural memory gradient flow)
5. One full training iteration on an actual CIFAR-10 batch
6. Downstream heads (segmentation & detection) sanity check
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Ensure template root is in sys.path
convneur_dir = os.path.dirname(os.path.abspath(__file__))
if convneur_dir not in sys.path:
    sys.path.insert(0, convneur_dir)

from model import build_model
from model.convneur import ConvNeur, Block
from downstream_heads import ConvNeurForSegmentation, ConvNeurForDetection
from cifar10_loader import CIFAR10HF
from pennfudan import PennFudanDataset


def run_smoke_test():
    print("=" * 65)
    print("RUNNING CONVNEUR SMOKE TEST SUITE")
    print("=" * 65)

    # 1. Test Import & Instantiation
    print("[1/6] Testing model instantiation (build_model 'm1' for CIFAR-10)...")
    model = build_model(scale="m1", num_classes=10, img_size=32)
    assert isinstance(model, ConvNeur), "model is not an instance of ConvNeur"
    print("  ✓ Instantiation successful.")

    # 2. Test Forward Pass
    print("[2/6] Testing forward pass with random batch (B=2, C=3, H=32, W=32)...")
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 10), f"Expected out shape (2, 10), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN found in forward output"
    assert not torch.isinf(out).any(), "Inf found in forward output"
    print(f"  ✓ Forward pass succeeded. Output shape: {out.shape}")

    # 3. Test Backward Pass & Gradients
    print("[3/6] Testing autograd backward pass...")
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([1, 7], dtype=torch.long)
    loss = criterion(out, target)
    loss.backward()

    assert x.grad is not None, "Input x received no gradients"
    assert torch.isfinite(x.grad).all(), "Non-finite gradients in input"
    print(f"  ✓ Backward pass succeeded. Input gradient norm: {x.grad.norm().item():.4f}")

    # 4. Test Memory State & Gradient Flow in Neural Memory
    print("[4/6] Verifying neural memory states and gradients across stages...")
    active_memory_stages = [0, 1, 2]
    for stage_idx in active_memory_stages:
        block0 = model.stages[stage_idx][0]
        assert block0.use_mem, f"Stage {stage_idx} block 0 should have use_mem=True"
        mem = block0.mem
        assert hasattr(mem, "to_keys") and mem.to_keys.weight.grad is not None, (
            f"Stage {stage_idx} neural memory (to_keys) received no gradients!"
        )
        grad_norm = mem.to_keys.weight.grad.norm().item()
        assert grad_norm > 0, f"Stage {stage_idx} memory gradient norm must be > 0, got {grad_norm}"
        print(f"  ✓ Stage {stage_idx} NeuralMemory gradient active: to_keys grad norm = {grad_norm:.4f}")

    # 5. Test 1 Batch of CIFAR-10
    print("[5/6] Testing 1 full training batch on CIFAR-10 dataset...")
    aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    ds = CIFAR10HF(split="train", transform=aug, fast_smoke=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch_x, batch_y = next(iter(loader))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    batch_loss.backward()
    optimizer.step()

    assert not torch.isnan(batch_loss), "Loss on CIFAR-10 batch was NaN"
    print(f"  ✓ CIFAR-10 batch step complete. Batch loss: {batch_loss.item():.4f}, Batch size: {batch_x.size(0)}")

    # 6. Test Downstream Heads Sanity Check
    print("[6/6] Testing downstream segmentation and detection heads...")
    # Segmentation head
    seg_backbone = build_model(scale="m1", num_classes=10, img_size=128)
    seg_model = ConvNeurForSegmentation(seg_backbone, num_classes=2)
    x_seg = torch.randn(2, 3, 128, 128, requires_grad=True)
    seg_out = seg_model(x_seg)
    assert seg_out.shape == (2, 2, 128, 128), f"Unexpected seg out shape: {seg_out.shape}"
    seg_out.sum().backward()
    assert x_seg.grad is not None, "Segmentation head input received no grad"
    print("  ✓ Segmentation head (FPN-lite) forward + backward verified.")

    # Detection head
    det_backbone = build_model(scale="m1", num_classes=10, img_size=128)
    det_model = ConvNeurForDetection(det_backbone, num_classes=20, num_queries=1)
    x_det = torch.randn(2, 3, 128, 128, requires_grad=True)
    det_boxes, det_scores = det_model(x_det)
    assert det_boxes.shape == (2, 1, 4), f"Unexpected det boxes shape: {det_boxes.shape}"
    assert det_scores.shape == (2, 1, 20), f"Unexpected det scores shape: {det_scores.shape}"
    (det_boxes.sum() + det_scores.sum()).backward()
    assert x_det.grad is not None, "Detection head input received no grad"
    print("  ✓ Detection head forward + backward verified.")

    print("\n" + "=" * 65)
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY! ✓")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
