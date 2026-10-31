# Open Source Attribution & Third-Party Licenses

This template integrates code and datasets adapted from multiple open-source repositories.
The original licenses, copyrights, and citations are documented below.

---

## 1. ConvNeur Model Architecture

- **Project**: ConvNeur ("Efficiency Follows Global-Local Decoupling", CVPR 2026)
- **Original Repository**: https://github.com/NUST-Machine-Intelligence-Laboratory/ConvNeur (also mirrored at https://github.com/ZhenyuYang01/ConvNeur)
- **Authors**: Zhenyu Yang et al., NUST Machine Intelligence Laboratory (Nanjing University of Science and Technology)
- **License**: Apache License 2.0
- **Modifications**:
  - Adapted hierarchical multi-stage model factory for arbitrary input resolutions (including 32x32 CIFAR-10).
  - Encapsulated stage blocks into `nn.ModuleList` for safe programmatic traversal and inspection.
  - Adjusted chunk-size resolution routing to ensure test-time memory state backpropagation is active across all valid spatial levels.

---

## 2. Neural Memory (Titans PyTorch)

- **Project**: Titans PyTorch
- **Repository**: https://github.com/lucidrains/titans-pytorch
- **Author**: Phil Wang (lucidrains)
- **Copyright**: (c) 2025 Phil Wang
- **License**: MIT License
- **Reference**: *"Titans: Learning to Memorize at Test Time"*, Behrouz et al., 2024.

---

## 3. ConvNeXt V2 Normalization

- **Project**: ConvNeXt V2 ("Co-designing and Scaling ConvNets with Masked Autoencoders", CVPR 2023)
- **Repository**: https://github.com/facebookresearch/ConvNeXt-V2
- **Authors**: Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie (Meta FAIR & KAIST)
- **License**: Apache License 2.0
- **Components Used**: Global Response Normalization (GlobalNorm / GRN) and 2D LayerNorm.

---

## 4. Benchmark Datasets & Surrogates

- **CIFAR-10**:
  - Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton (2009). *"Learning Multiple Layers of Features from Tiny Images."*
  - URL: https://www.cs.toronto.edu/~kriz/cifar.html
- **Penn-Fudan Pedestrian Database**:
  - Liming Wang, Jianbo Shi, Gang Song, and I-fan Shen (2007). *"Object Detection with Contour Segment Networks."*
  - University of Pennsylvania & Fudan University.
  - URL: https://www.cis.upenn.edu/~jshi/ped_html/
  - **Surrogate Scope**: Used strictly as a small-scale sanity check (~170 images) for downstream segmentation and detection heads. It is NOT an empirical substitute or benchmark for ADE20K or MS COCO.
- **PASCAL VOC 2007**:
  - Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. *"The PASCAL Visual Object Classes Challenge 2007."*
  - URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
