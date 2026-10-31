"""
Penn-Fudan Pedestrian Dataset Loader (Surrogate Sanity Check)
Dataset: Penn-Fudan Database for Pedestrian Detection and Segmentation
Authors: Liming Wang, Jianbo Shi, Gang Song, I-fan Shen (2007)
URL: https://www.cis.upenn.edu/~jshi/ped_html/

NOTE:
Penn-Fudan is used here strictly as a lightweight surrogate / small-scale sanity check
(~170 images) for quick local verification and smoke testing of multi-scale downstream heads
(segmentation mask prediction and bounding box regression).
It DOES NOT directly represent, replace, or benchmark large-scale datasets such as:
- ADE20K (150 semantic categories, 25k+ high-resolution scenes)
- MS COCO (80 object categories, 120k+ complex multi-instance scenes)
Performance metrics on Penn-Fudan should NOT be cited as ADE20K or COCO results.
"""

import os
import torch
from torch.utils.data import Dataset


class PennFudanDataset(Dataset):
    def __init__(self, root="./data", task="segmentation", split="train", img_size=128, allow_synthetic=True):
        self.split = split
        self.task = task
        self.img_size = img_size

        # Map test/val interchangeably
        clean_split = "val" if split in ["test", "val"] else "train"

        # Locate pennfudan_<split>.pt
        candidates = [
            os.path.join(root, f"pennfudan_{clean_split}.pt"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../data/pennfudan_{clean_split}.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/pennfudan_{clean_split}.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"data/pennfudan_{clean_split}.pt")),
        ]
        pt_path = None
        for c in candidates:
            if os.path.exists(c):
                pt_path = c
                break

        if pt_path is not None:
            data = torch.load(pt_path, map_location="cpu")
            self.images = data["images"]  # (N, 3, 128, 128) uint8
            self.masks = data.get("masks", torch.zeros((len(self.images), img_size, img_size), dtype=torch.long))
            self.boxes = data.get("boxes", torch.tensor([[0.5, 0.5, 0.4, 0.6]] * len(self.images), dtype=torch.float32))
        elif allow_synthetic:
            # Generate lightweight synthetic surrogate samples for pipeline sanity checking
            n_samples = 32 if clean_split == "train" else 16
            self.images = torch.randint(0, 256, (n_samples, 3, img_size, img_size), dtype=torch.uint8)
            self.masks = torch.randint(0, 2, (n_samples, img_size, img_size), dtype=torch.long)
            # Default normalized boxes [cx, cy, w, h]
            self.boxes = torch.tensor([[0.5, 0.5, 0.3, 0.5]] * n_samples, dtype=torch.float32)
        else:
            raise RuntimeError(f"pennfudan_{split}.pt not found under {root}. Checked: {candidates}")

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        img = (img - self.mean) / self.std

        if self.task == "detection":
            return img, self.boxes[idx]
        return img, self.masks[idx]
