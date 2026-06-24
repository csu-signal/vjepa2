import torch
import torchvision.transforms as T

class VideoTransform:
    def __init__(self, crop_size=256):
        self.transform = T.Compose([
            T.ToTensor(), # Converts HWC numpy [0, 255] to CHW tensor [0.0, 1.0]
            T.Resize((crop_size, crop_size), antialias=True), # Fixed resize, no random scaling
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet defaults
        ])

    def __call__(self, buffer):
        # buffer is a list or array of images: [T, H, W, C]
        # We apply the transform frame by frame
        frames = []
        for img in buffer:
            frames.append(self.transform(img))
            
        # Stack back into [T, C, H, W] and permute to [C, T, H, W] which is standard for video models
        return torch.stack(frames).permute(1, 0, 2, 3)

def make_transforms(crop_size=256, **kwargs):
    return VideoTransform(crop_size=crop_size)
