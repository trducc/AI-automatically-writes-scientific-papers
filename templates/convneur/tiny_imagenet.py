"""
Tiny-ImageNet-200 Dataset Loader for PyTorch (Optional / Extension)
Downloads and prepares Tiny-ImageNet-200 (Stanford CS231N).
200 classes, 100,000 train images (500 per class), 10,000 val images (50 per class).
Image size: 64x64x3.

Attribution:
- Stanford CS231N Tiny ImageNet Visual Recognition Challenge.
  URL: https://cs231n.stanford.edu/

NOTE:
The default ConvNeur baseline in this repository evaluates on CIFAR-10.
Tiny-ImageNet is an optional experimental loader for higher-resolution scaling studies.
"""

import os
import urllib.request
import zipfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


class TinyImageNet(Dataset):
    def __init__(self, root="./data", split="train", transform=None, download=True, max_samples=None):
        self.root = os.path.abspath(root)
        self.split = split
        self.transform = transform
        self.dataset_dir = os.path.join(self.root, "tiny-imagenet-200")
        self.zip_path = os.path.join(self.root, "tiny-imagenet-200.zip")

        if download and not os.path.exists(self.dataset_dir):
            self._download()

        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(f"Dataset not found at {self.dataset_dir}. Pass download=True to download.")

        # Read class IDs (wnids)
        wnids_file = os.path.join(self.dataset_dir, "wnids.txt")
        with open(wnids_file, "r") as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {cls_id: i for i, cls_id in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.samples = []
        if self.split == "train":
            train_dir = os.path.join(self.dataset_dir, "train")
            for cls_id in self.classes:
                cls_idx = self.class_to_idx[cls_id]
                img_dir = os.path.join(train_dir, cls_id, "images")
                if not os.path.exists(img_dir):
                    continue
                for fname in os.listdir(img_dir):
                    if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((os.path.join(img_dir, fname), cls_idx))
        elif self.split == "val":
            val_dir = os.path.join(self.dataset_dir, "val")
            val_annot = os.path.join(val_dir, "val_annotations.txt")
            val_img_dir = os.path.join(val_dir, "images")
            with open(val_annot, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        fname, cls_id = parts[0], parts[1]
                        if cls_id in self.class_to_idx:
                            self.samples.append((os.path.join(val_img_dir, fname), self.class_to_idx[cls_id]))
        else:
            raise ValueError(f"Split {split} not supported. Use 'train' or 'val'.")

        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(self.zip_path):
            print(f"Downloading Tiny-ImageNet-200 from {TINY_IMAGENET_URL}...")
            # Use urllib with progress
            def reporthook(blocknum, blocksize, totalsize):
                readsofar = blocknum * blocksize
                if totalsize > 0:
                    percent = readsofar * 1e2 / totalsize
                    if blocknum % 500 == 0:
                        print(f"  {readsofar / 1e6:.1f} MB / {totalsize / 1e6:.1f} MB ({percent:.1f}%)")

            urllib.request.urlretrieve(TINY_IMAGENET_URL, self.zip_path, reporthook)
            print("Download completed.")

        print(f"Extracting {self.zip_path}...")
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        print("Extraction completed.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target
