"""Clip-consistent augmentation for normalized T x C x H x W lip videos."""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class VideoAugmentation:
    def __init__(self):
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.RandomAffine(
                    degrees=5, translate=(0.04, 0.04), scale=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR, fill=0.5,
                ),
            ], p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
        ])

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        frames = self.transforms((video + 1.0) / 2.0)
        if torch.rand(()).item() < 0.35:
            height, width = frames.shape[-2:]
            scale = 0.5 + 0.5 * torch.rand(()).item()
            frames = TF.resize(
                frames, [max(1, round(height * scale)), max(1, round(width * scale))],
                interpolation=InterpolationMode.BILINEAR, antialias=True,
            )
            frames = TF.resize(
                frames, [height, width],
                interpolation=InterpolationMode.BILINEAR, antialias=True,
            )
        return (frames * 2.0 - 1.0).clamp(-1.0, 1.0)
