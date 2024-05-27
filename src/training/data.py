import os
import pandas as pd
import torch
import torchvision

from typing import Callable, List, Optional, Tuple


class JetbotDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        path: str,
        transforms: Optional[Callable] = None,
        shift=0,  # to adjust for latency in model
    ):
        super(JetbotDataset, self).__init__(root=path, transforms=transforms)
        self.labels, self.images = load_files(path)
        self.shift = shift

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.images[idx])
        label = torch.tensor(
            (
                self.labels[min(idx, len(self.labels) - 1)][1:]
                + self.labels[min(idx + self.shift, len(self.labels) - 1)][1:]
                + self.labels[min(idx + 2 * self.shift, len(self.labels) - 1)][1:]
            ),
            dtype=torch.float32,
        )

        if not self.transforms:
            return img, label

        return self.transforms(img, label)


def load_files(path: str) -> Tuple[List[List[float]], List[str]]:
    labels = []
    images = []

    subdirs = [
        subdir for subdir in os.listdir(path) if os.path.isdir(f"{path}/{subdir}")
    ]

    for subdir in subdirs:
        subdir_labels = load_labels(f"{path}/{subdir}.csv")

        for label in subdir_labels:
            labels.append(label)

            img_name = str(int(label[0]))
            img_name = "0" * (4 - len(img_name)) + img_name

            images.append(f"{path}/{subdir}/{img_name}.jpg")

    return labels, images


def load_labels(path: str) -> list[list[float]]:
    return pd.read_csv(path, header=None).values.tolist()
