"""
Image pre-processor built from scratch — no torchvision transforms used
beyond basic tensor conversion.

Pipeline
--------
1. Resize shorter side to `img_size`, preserving aspect ratio
2. Centre-crop to (img_size × img_size)
3. Convert to float32 tensor in [0, 1]
4. Normalise with ImageNet mean/std (same values SigLIP uses)
"""

import math
import torch
import numpy as np
from PIL import Image


# SigLIP normalisation constants (identical to standard ViT pre-processing)
_MEAN = [0.5, 0.5, 0.5]
_STD  = [0.5, 0.5, 0.5]


class ImageProcessor:
    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    # ── Internal steps ────────────────────────────────────────────

    def _resize(self, img: Image.Image) -> Image.Image:
        """Resize so the shorter side == img_size (bicubic)."""
        w, h = img.size
        if w < h:
            new_w = self.img_size
            new_h = int(h * self.img_size / w)
        else:
            new_h = self.img_size
            new_w = int(w * self.img_size / h)
        return img.resize((new_w, new_h), Image.BICUBIC)

    def _center_crop(self, img: Image.Image) -> Image.Image:
        """Centre-crop to (img_size × img_size)."""
        w, h = img.size
        left   = (w - self.img_size) // 2
        top    = (h - self.img_size) // 2
        right  = left + self.img_size
        bottom = top  + self.img_size
        return img.crop((left, top, right, bottom))

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """PIL → float32 tensor (C, H, W) in [0, 1]."""
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, C)
        tensor = torch.from_numpy(arr).permute(2, 0, 1) # (C, H, W)
        return tensor

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1)
        std  = torch.tensor(_STD,  dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    # ── Public API ────────────────────────────────────────────────

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Args:
            img: PIL Image (RGB)
        Returns:
            pixel_values: float32 tensor (3, img_size, img_size)
        """
        img = img.convert("RGB")
        img = self._resize(img)
        img = self._center_crop(img)
        t   = self._to_tensor(img)
        t   = self._normalize(t)
        return t

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.preprocess(img)