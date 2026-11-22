"""Image preprocessing shared by training and inference (avoids train/serve skew).

Train-time augmentation is deliberately aimed at the in-cabin domain gap:
* random grayscale + contrast change  -> simulates NIR/IR cabin cameras at night
* strong brightness/gamma jitter      -> tunnels, direct sunlight
* random erasing                      -> sunglasses, hands, masks, hair occlusion
* motion blur                         -> vibration
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class RandomMotionBlur:
    def __init__(self, p: float = 0.2, max_k: int = 9):
        self.p, self.max_k = p, max_k

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return img
        k = int(torch.randint(3, self.max_k + 1, ()).item()) | 1
        kernel = torch.zeros(k, k)
        if torch.rand(()) < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        c = img.shape[0]
        w = kernel.expand(c, 1, k, k).to(img.dtype)
        return torch.nn.functional.conv2d(img.unsqueeze(0), w, padding=k // 2, groups=c).squeeze(0)


def train_transform(size: int = 224, flip: bool = True) -> T.Compose:
    return T.Compose(
        [
            T.ToImage(),
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18), antialias=True),
            T.RandomHorizontalFlip() if flip else T.Identity(),
            T.RandomApply([T.ColorJitter(0.5, 0.5, 0.3, 0.05)], p=0.8),
            T.RandomGrayscale(p=0.3),
            T.RandomApply([T.RandomRotation(12)], p=0.5),
            T.ToDtype(torch.float32, scale=True),
            RandomMotionBlur(0.2),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ]
    )


def eval_transform(size: int = 224) -> T.Compose:
    return T.Compose([T.ToImage(), T.Resize((size, size), antialias=True), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


def bgr_to_rgb3(img: np.ndarray) -> np.ndarray:
    """Any OpenCV image (gray / BGR / BGRA) -> 3-channel RGB uint8."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_batch(imgs_bgr: list[np.ndarray], size: int = 224) -> torch.Tensor:
    tf = eval_transform(size)
    return torch.stack([tf(bgr_to_rgb3(im)) for im in imgs_bgr])
