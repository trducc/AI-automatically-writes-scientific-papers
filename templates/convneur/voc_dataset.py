"""
Pascal VOC 2007 Object Detection Dataset Loader (Compact Binary Tensor Stream)
Loaded directly from pre-packaged uint8 binary tensor or synthetic surrogate stream.
20 classes:
  aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
  diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

Attribution:
- Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A.
  The PASCAL Visual Object Classes Challenge 2007 (VOC2007).
  URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

NOTE:
This loader is intended as a compact sanity check for object detection heads.
It does not replace comprehensive multi-scale benchmarks on MS COCO.
"""

import os
import torch
from torch.utils.data import Dataset

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}


class VOCDetection(Dataset):
    def __init__(self, root="./data", split="train", img_size=128, max_samples=None, allow_synthetic=True):
        self.split = split
        self.img_size = img_size

        # Map test/val interchangeably
        clean_split = "val" if split in ["test", "val"] else "train"

        # Locate voc_detection_<split>.pt
        candidates = [
            os.path.join(root, f"voc_detection_{clean_split}.pt"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../data/voc_detection_{clean_split}.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/voc_detection_{clean_split}.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"data/voc_detection_{clean_split}.pt")),
        ]
        pt_path = None
        for c in candidates:
            if os.path.exists(c):
                pt_path = c
                break

        if pt_path is not None:
            data = torch.load(pt_path, map_location="cpu")
            self.images = data["images"]  # (N, 3, 128, 128) uint8
            self.boxes = data["boxes"]    # (N, 4) float32 [cx, cy, w, h]
            self.labels = data["labels"]  # (N,) long [0..19]
        elif allow_synthetic:
            n_samples = 32 if clean_split == "train" else 16
            self.images = torch.randint(0, 256, (n_samples, 3, img_size, img_size), dtype=torch.uint8)
            self.boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3]] * n_samples, dtype=torch.float32)
            self.labels = torch.randint(0, 20, (n_samples,), dtype=torch.long)
        else:
            raise RuntimeError(f"voc_detection_{split}.pt not found under {root}. Checked: {candidates}")

        if max_samples and max_samples < len(self.images):
            self.images = self.images[:max_samples]
            self.boxes = self.boxes[:max_samples]
            self.labels = self.labels[:max_samples]

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        img = (img - self.mean) / self.std
        return img, self.boxes[idx], self.labels[idx]
