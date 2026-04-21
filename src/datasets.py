from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .cifar100_groups import GROUP_TO_FINE_INDICES, GROUP_TO_FINE_LABELS, canonical_group_name

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


@dataclass
class GroupInfo:
    name: str
    fine_indices: List[int]
    fine_names: List[str]
    num_classes: int


class CIFAR100GroupDataset(Dataset):
    """CIFAR-100 subset dataset for one superclass.

    It remaps the 5 fine labels inside the selected superclass to local labels 0..4.
    """

    def __init__(self, root: str, group_name: str, train: bool, transform=None, download: bool = True):
        super().__init__()
        self.group_name = canonical_group_name(group_name)
        self.fine_indices = GROUP_TO_FINE_INDICES[self.group_name]
        self.fine_names = GROUP_TO_FINE_LABELS[self.group_name]
        self.local_map = {fine_idx: local_idx for local_idx, fine_idx in enumerate(self.fine_indices)}
        self.transform = transform

        base = datasets.CIFAR100(root=root, train=train, transform=None, download=download)
        self.samples: List[Tuple[torch.Tensor, int]] = []

        for img, target in zip(base.data, base.targets):
            if target in self.local_map:
                self.samples.append((img, self.local_map[target]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, target = self.samples[idx]
        img = transforms.functional.to_pil_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class TransformDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int], transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, target = self.base_dataset.samples[real_idx]
        img = transforms.functional.to_pil_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def build_transforms(image_size: int = 32):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_transform, eval_transform


def split_indices(indices: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    copied = list(indices)
    rng.shuffle(copied)
    val_count = max(1, int(len(copied) * val_ratio)) if val_ratio > 0 else 0
    val_indices = copied[:val_count]
    train_indices = copied[val_count:]
    return train_indices, val_indices


def get_group_datasets(
    root: str,
    group_name: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    download: bool = True,
):
    train_transform, eval_transform = build_transforms()

    raw_train = CIFAR100GroupDataset(root=root, group_name=group_name, train=True, transform=None, download=download)
    raw_test = CIFAR100GroupDataset(root=root, group_name=group_name, train=False, transform=None, download=download)

    all_train_indices = list(range(len(raw_train)))
    train_indices, val_indices = split_indices(all_train_indices, val_ratio=val_ratio, seed=seed)

    train_dataset = TransformDataset(raw_train, train_indices, transform=train_transform)
    val_dataset = TransformDataset(raw_train, val_indices, transform=eval_transform)
    test_dataset = TransformDataset(raw_test, list(range(len(raw_test))), transform=eval_transform)

    group_name = canonical_group_name(group_name)
    info = GroupInfo(
        name=group_name,
        fine_indices=GROUP_TO_FINE_INDICES[group_name],
        fine_names=GROUP_TO_FINE_LABELS[group_name],
        num_classes=len(GROUP_TO_FINE_INDICES[group_name]),
    )

    return train_dataset, val_dataset, test_dataset, info
