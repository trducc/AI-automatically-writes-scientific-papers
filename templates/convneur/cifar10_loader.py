"""
CIFAR-10 Dataset Loader (Multi-backend: Direct Cached Arrow / HuggingFace / torchvision / Synthetic Fallback)
50,000 train images, 10,000 test images (10 classes).
Image size: 32x32x3.

Attribution:
- CIFAR-10 dataset collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton (2009).
  URL: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class CIFAR10HF(Dataset):
    """
    Robust CIFAR-10 Loader:
    1. Loads instantly from local arrow files if cached (<0.05s, zero network latency).
    2. Falls back to HuggingFace datasets library.
    3. Falls back to torchvision.datasets.CIFAR10.
    4. Falls back to synthetic dataset for offline CI / smoke testing.
    """
    def __init__(self, root="./data", split="train", transform=None, fast_smoke=False):
        self.split = split
        self.transform = transform
        self.classes = CIFAR10_CLASSES
        self.backend = None
        self.ds = None

        if fast_smoke:
            self.backend = "synthetic"
            torch.manual_seed(42)
            num_samples = 64
            self.ds = [
                (
                    Image.fromarray(torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8).numpy()),
                    i % 10
                )
                for i in range(num_samples)
            ]
            return

        clean_split = "train" if split == "train" else "test"

        # 1. Fast path: check for existing cached arrow file in huggingface cache
        arrow_pattern = os.path.expanduser(f"~/.cache/huggingface/datasets/uoft-cs___cifar10/**/cifar10-{clean_split}.arrow")
        matching_arrows = glob.glob(arrow_pattern, recursive=True)
        if matching_arrows and os.path.exists(matching_arrows[0]):
            try:
                from datasets import Dataset as HFDataset
                self.ds = HFDataset.from_file(matching_arrows[0])
                self.backend = "hf"
            except Exception:
                pass

        # 2. Try standard Hugging Face datasets loader
        if self.ds is None:
            try:
                from datasets import load_dataset
                self.ds = load_dataset("uoft-cs/cifar10", split=split)
                self.backend = "hf"
            except Exception:
                pass

        # 3. Try torchvision CIFAR10
        if self.ds is None:
            try:
                from torchvision.datasets import CIFAR10
                is_train = (clean_split == "train")
                self.ds = CIFAR10(root=root, train=is_train, download=True)
                self.backend = "torchvision"
            except Exception:
                pass

        # 4. Fallback to synthetic dataset for completely offline CI
        if self.ds is None:
            self.backend = "synthetic"
            num_samples = 64 if clean_split != "train" else 256
            self.ds = [
                (
                    Image.fromarray(torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8).numpy()),
                    i % 10
                )
                for i in range(num_samples)
            ]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if self.backend == "hf":
            item = self.ds[idx]
            img = item["img"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            label = int(item["label"])
        elif self.backend == "torchvision":
            img, label = self.ds[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            label = int(label)
        else:
            img, label = self.ds[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
