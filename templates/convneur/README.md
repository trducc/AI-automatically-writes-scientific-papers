# ConvNeur Template — AI Scientist

Implementation of **ConvNeur** (Global-Local Decoupled vision architecture) for automated scientific experimentation within The AI Scientist framework.

> **"Efficiency Follows Global-Local Decoupling"** — CVPR 2026  
> Paper: [CVPR 2026 OpenAccess](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Efficiency_Follows_Global-Local_Decoupling_CVPR_2026_paper.html)  
> Original Codebase: [NUST-Machine-Intelligence-Laboratory/ConvNeur](https://github.com/NUST-Machine-Intelligence-Laboratory/ConvNeur)  
> Authors: Zhenyu Yang et al. (NUST Machine Intelligence Laboratory)

---

## 1. Architecture Overview

ConvNeur decouples visual processing into two specialized branches per block:

```text
Input Feature Map (B, C, H, W)
        │
  ┌─────┴────────────────┐
  ▼                      ▼
Locality Branch        Neural Memory Branch
(ConvNeXt 7x7 DW       (Titans-style NeuralMemory
 + GlobalNorm + PW)     Long-term token memory)
  │                      │
  │                      ▼
  │               x_mem = sigmoid(proj_out(x_mem))
  └──────────┬───────────┘
             ▼
        x = x_mem * x   (Multiplicative Gated Modulation)
             ▼
     + DropPath + Residual
```

- **Locality Branch**: Captures fine-grained inductive spatial structures via depthwise 7x7 convolution, LayerNorm, and Global Response Normalization (GRN).
- **Neural Memory Branch**: Dynamically models long-range multi-scale dependencies via test-time associative memory updates.
- **Cross-Stage Memory Recurrence**: Memory update residuals are passed sequentially across hierarchy stages ($Stage_0 \rightarrow Stage_1 \rightarrow Stage_2$).

---

## 2. Supported Datasets & Benchmarking Scope

| Task | Dataset | Resolution | Scope & Purpose |
|---|---|---|---|
| **Image Classification** | **CIFAR-10** | 32x32 (10 classes) | **Primary Active Baseline Benchmark** (Coarse object recognition) |
| **Semantic Segmentation** | **Penn-Fudan Pedestrians** | 128x128 (binary mask) | **Surrogate Sanity Check Only** (pipeline verification) |
| **Object Detection** | **Pascal VOC subset / Penn-Fudan** | 128x128 (bounding boxes) | **Surrogate Sanity Check Only** (pipeline verification) |

> [!IMPORTANT]
> **Surrogate Sanity Check Disclaimer:**  
> Penn-Fudan is a small-scale dataset (~170 pedestrian images) used solely for fast local smoke testing and pipeline verification of multi-scale downstream heads (FPN-lite segmentation and box regression).  
> **Penn-Fudan does NOT directly represent, replace, or benchmark large-scale datasets such as ADE20K (semantic segmentation, 25k+ scenes, 150 classes) or MS COCO (object detection, 120k+ scenes, 80 classes).**  
> Downstream metrics on Penn-Fudan must not be claimed or cited as ADE20K or MS COCO performance.  
> *(Note: CIFAR-100 and Tiny-ImageNet are not implemented in the current baseline execution; the baseline evaluates exclusively on CIFAR-10).*

---

## 3. Installation & Dependencies

Ensure you have a Python 3.10+ or 3.11 environment with PyTorch installed.

```bash
# 1. Create and activate conda environment
conda create -n ai-scientist python=3.11 -y
conda activate ai-scientist

# 2. Install dependencies
pip install -r requirements.txt
# Or install template-specific dependencies directly:
pip install -r templates/convneur/requirements.txt
```

### Core Requirements
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `timm>=0.9.0`
- `einops>=0.7.0`
- `einx>=0.2.0`
- `tensordict>=0.2.0`
- `datasets>=2.14.0`
- `matplotlib>=3.7.0`
- `numpy>=1.24.0`
- `Pillow>=9.5.0`

---

## 4. Verification & Smoke Testing

Run the automated smoke test suite to verify module imports, forward pass, autograd backward pass, memory state propagation, and a full CIFAR-10 training batch:

```bash
# From repository root
python templates/convneur/smoke_test.py

# Or via pytest
pytest tests/test_convneur_smoke.py
```

---

## 5. Quick Start & Baseline Execution

```bash
cd templates/convneur

# 1. Run full baseline experiment on CIFAR-10 (and downstream sanity checks)
python experiment.py --out_dir run_0 --cls_epochs 10

# 2. Fast debugging / verification run (1 epoch, 5 steps)
python experiment.py --out_dir run_0 --cls_epochs 1 --seg_epochs 1 --det_epochs 1 --max_steps 5

# 3. Generate publication figures and plots
python plot.py
```

### Running with AI Scientist

```bash
python launch_scientist.py \
    --model "claude-3-5-sonnet-20241022" \
    --experiment convneur \
    --num-ideas 3
```

---

## 6. Licenses & Attribution

This template integrates code and concepts from multiple open-source works:

1. **ConvNeur Architecture & Code**:
   - Adapted from [ZhenyuYang01/ConvNeur](https://github.com/ZhenyuYang01/ConvNeur) / NUST Machine Intelligence Laboratory.
   - Paper: *"Efficiency Follows Global-Local Decoupling"*, CVPR 2026.
   - License: **Apache License 2.0**.
2. **Titans Neural Memory**:
   - Neural memory mechanics adapted from [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) by Phil Wang (2025).
   - License: **MIT License**.
3. **Normalization (LayerNorm, GlobalNorm/GRN)**:
   - Adapted from ConvNeXt V2 ([facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)) / `timm`.
   - License: **Apache License 2.0**.
4. **Datasets**:
   - **CIFAR-10**: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton (2009).
   - **Penn-Fudan Pedestrian Database**: Liming Wang, Jianbo Shi, Gang Song, and I-fan Shen (2007), University of Pennsylvania & Fudan University.
   - **PASCAL VOC**: Everingham et al. (2007).
