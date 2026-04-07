"""
dataset.py — Dataset loading and augmentation transforms
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T


class MicroplasticsDataset(Dataset):
    """
    Loads images + bounding boxes from a Roboflow-style CSV annotation file.
    CSV columns: filename, width, height, class, xmin, ymin, xmax, ymax
    """
    LABEL_MAP = {"background": 0, "Microplastic": 1}

    def __init__(self, img_dir, csv_path, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        df = pd.read_csv(csv_path)
        self.image_ids = df["filename"].unique().tolist()
        self.records = {}
        for fname, grp in df.groupby("filename"):
            boxes  = grp[["xmin", "ymin", "xmax", "ymax"]].values.astype(float)
            labels = [self.LABEL_MAP.get(c, 1) for c in grp["class"].tolist()]
            self.records[fname] = {"boxes": boxes, "labels": labels}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fname    = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, fname)

        img    = Image.open(img_path).convert("RGB")
        record = self.records[fname]

        boxes  = torch.as_tensor(record["boxes"],  dtype=torch.float32)
        labels = torch.as_tensor(record["labels"], dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


def get_transforms(train=True):
    """
    Build the image transform pipeline.
    SSDLite320 expects 320×320 float tensors in [0, 1].
    """
    ops = []
    if train:
        ops.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2))
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops.append(T.Resize((320, 320)))
    ops.append(T.ToTensor())
    return T.Compose(ops)


def collate_fn(batch):
    """Custom collate so DataLoader handles variable-length target dicts."""
    return tuple(zip(*batch))
