import torch
import torch.nn.functional as F
import numpy as np


class VideoTransform:
    def __init__(self, crop_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.crop_size = crop_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def __call__(self, buffer):
        """
        Vectorized transformation on [T, H, W, C] numpy array or tensor -> [C, T, H, W] float32 tensor.
        """
        if isinstance(buffer, np.ndarray):
            t_buf = torch.from_numpy(buffer)
        else:
            t_buf = buffer

        # Convert [T, H, W, C] -> [T, C, H, W] float normalized to [0, 1]
        t_buf = t_buf.permute(0, 3, 1, 2).float().div_(255.0)

        # Vectorized bilinear resize across all frames simultaneously
        t_resized = F.interpolate(t_buf, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)

        # Normalize with ImageNet stats
        t_norm = (t_resized - self.mean) / self.std

        # Permute to [C, T, H, W] expected by video ViT
        return t_norm.permute(1, 0, 2, 3)


def make_transforms(crop_size=256, **kwargs):
    return VideoTransform(crop_size=crop_size)
