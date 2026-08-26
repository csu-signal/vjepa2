import sys
sys.path.insert(0, "/home/jack/code/vjepa2-probe-guidance/vjepa2")
print(sys.path)

from pathlib import Path
import copy
import os
import math
import glob
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
import torchvision.transforms.v2 as T
from torchvision.io import read_image, ImageReadMode
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from diffusers import AutoencoderKL
import cv2

from app.vjepa_ll_probe_guidance.utils import init_video_model
from app.vjepa_ll_probe_guidance.transforms import make_transforms


STATE_MEAN = torch.tensor([-0.13193196, 0.13471916, 1.4406046, 59.395, -43.585613, -76.49974], dtype=torch.float32, device="cuda:0")
STATE_STD = torch.tensor([1.05345368e-01, 1.75726220e-01, 7.28756189e-02, 7.79652252e+01, 3.07379379e+01, 1.22978004e+02], dtype=torch.float32, device="cuda:0")

ACTION_MEAN = torch.tensor([-2.7906366e-05, 5.4028669e-06, 6.0837847e-05, -1.7707590e-03, 2.4102912e-03, -3.7514704e-04], dtype=torch.float32, device="cuda:0")
ACTION_STD = torch.tensor([0.00498742, 0.00453256, 0.00675807, 0.55955255, 0.43809873, 0.73082167], dtype=torch.float32, device="cuda:0")


class LatentBridge(nn.Module):
    def __init__(self, embed_dim=1024, target_channels=4):
        super().__init__()

        self.temporal_split = nn.Linear(embed_dim, embed_dim * 2)

        self.upconv = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=256,
            kernel_size=2,
            stride=2
        )

        self.act = nn.GELU()
        self.final_proj = nn.Conv2d(256, target_channels, kernel_size=3, padding=1)

    def forward(self, vjepa_tokens, num_frames=16, height=256, width=256):
        T_p = num_frames // 2
        H_p = height // 16
        W_p = width // 16

        x = rearrange(vjepa_tokens, "b (t h w) d -> b t h w d", t=T_p, h=H_p, w=W_p)
        x = self.temporal_split(x)
        x = rearrange(x, "b t h w (repeat d) -> b (t repeat) d h w", repeat=2)
        x = rearrange(x, "b t d h w -> (b t) d h w")

        x = self.upconv(x)
        x = self.act(x)
        vae_latents = self.final_proj(x)

        return vae_latents


def standardize_states(states):
    return (states - STATE_MEAN) / (STATE_STD + 1e-6)


def standardize_actions(actions):
    return (actions - ACTION_MEAN) / (ACTION_STD + 1e-6)


def forward_target(encoder, c, normalize_reps=True):
    c = c[None, :, None, ...].repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(1, 1, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def step_predictor(predictor, rollout_queue, action_queue, state_queue, normalize_reps=True):
    z = torch.cat(list(rollout_queue), dim=1)
    a = torch.stack(list(action_queue), dim=0).unsqueeze(0)
    s = torch.stack(list(state_queue), dim=0).unsqueeze(0)
    standardized_actions = standardize_actions(a)
    standardized_states = standardize_states(s)
    z = predictor(z, standardized_actions, standardized_states)
    if normalize_reps:
        z = F.layer_norm(z, (z.size(-1),))
    return z


def get_action(prev_state, curr_state):
    prev_xyz = prev_state[:3]
    prev_rvec_deg = prev_state[3:]

    curr_xyz = curr_state[:3]
    curr_rvec_deg = curr_state[3:]

    prev_rvec_rad = np.deg2rad(prev_rvec_deg)
    prev_rotation = Rotation.from_rotvec(prev_rvec_rad)
    prev_rot_matrix = prev_rotation.as_matrix()
    prev_rot_matrix_inv = prev_rot_matrix.T

    curr_rvec_rad = np.deg2rad(curr_rvec_deg)
    curr_rotation = Rotation.from_rotvec(curr_rvec_rad)
    curr_rot_matrix = curr_rotation.as_matrix()

    delta_xyz = prev_rot_matrix_inv @ (curr_xyz - prev_xyz)

    delta_rotation_matrix = prev_rot_matrix_inv @ curr_rot_matrix
    delta_rvec_rad = Rotation.from_matrix(delta_rotation_matrix).as_rotvec()
    delta_rvec_deg = np.rad2deg(delta_rvec_rad)

    action = np.concatenate([delta_xyz, delta_rvec_deg], dtype=np.float32)

    return action


def main():
    encoder, predictor = init_video_model(
            device="cuda:0",
            patch_size=16,
            max_num_frames=512,
            tubelet_size=2,
            model_name="vit_large",
            crop_size=256,
            pred_depth=12,
            pred_num_heads=12,
            pred_embed_dim=768,
            action_embed_dim=6,
            predictor_type="ac",
            pred_is_frame_causal=True,
            use_extrinsics=False,
            use_sdpa=True,
            use_rope=True
        )

    encoder.eval()
    predictor.eval()

    def load_state_dict_with_ddp_fix(model, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=True)
        return model

    resume_path = os.path.join("/home/jack/code/vjepa2-probe-guidance/vjepa2/outputs/ll_probe_guidance_vitl_4", "best.pt")
    if os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=torch.device("cpu"))
        encoder = load_state_dict_with_ddp_fix(encoder, checkpoint["encoder"])
        predictor = load_state_dict_with_ddp_fix(predictor, checkpoint["predictor"])
    else:
        print(f"Checkpoint not found at {resume_path}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae_state_dict = torch.load("/home/jack/code/ultrasound_generative_modeling/checkpoints/ultrasound_vae_epoch_2.pt", map_location="cuda:0")
    vae.load_state_dict(vae_state_dict)
    vae = vae.to("cuda:0").eval()

    bridge = LatentBridge(embed_dim=1024, target_channels=4).to("cuda:0")
    bridge_state_dict = torch.load("/home/jack/code/ultrasound_generative_modeling/checkpoints/bridge_epoch_10.pt", map_location="cuda:0")
    bridge.load_state_dict(bridge_state_dict)
    bridge = bridge.to("cuda:0").eval()

    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)

    ultrasound_transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Resize((crop_size, crop_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    display_resize = T.Resize((crop_size, crop_size), antialias=True)

    data_root = "/home/jack/data/probe_guidance_dataset_june/test"
    episode_paths = sorted([
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if d.startswith("episode_") and os.path.isdir(os.path.join(data_root, d))
    ])

    episode_path = episode_paths[0]

    states_path = os.path.join(episode_path, "states_6dof.npy")
    states = np.load(states_path).astype(np.float32)

    rgb_dir = os.path.join(episode_path, "realsense_rgb")
    us_dir = os.path.join(episode_path, "ultrasound")

    rgb_file_count = sum(1 for item in Path(rgb_dir).iterdir() if item.is_file())
    us_file_count = sum(1 for item in Path(us_dir).iterdir() if item.is_file())

    assert rgb_file_count == us_file_count == len(states)

    prev_state = None
    curr_state = None
    rollout_queue = deque(maxlen=2)
    action_queue = deque(maxlen=2)
    state_queue = deque(maxlen=2)
    pred_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in tqdm(range(us_file_count), total=us_file_count):
            us_img_path = os.path.join(us_dir, f"{i:05d}.jpg")
            us_raw = read_image(us_img_path, mode=ImageReadMode.RGB).to("cuda:0", non_blocking=True)
            wm_us_img = ultrasound_transform(us_raw)

            rgb_img_path = os.path.join(rgb_dir, f"{i:05d}.jpg")
            rgb_raw = read_image(rgb_img_path, mode=ImageReadMode.RGB).to("cuda:0", non_blocking=True)

            gt_us_img = display_resize(us_raw).permute(1, 2, 0).cpu().numpy()
            rgb_img = display_resize(rgb_raw).permute(1, 2, 0).cpu().numpy()

            if len(rollout_queue) == 0:
                h = forward_target(encoder, wm_us_img)
                rollout_queue.append(h)

            curr_state = states[i] if not np.any(np.isnan(states[i])) else None
            if curr_state is not None:
                if prev_state is not None:
                    action_queue.append(torch.from_numpy(get_action(prev_state, curr_state)).to("cuda:0"))
                    state_queue.append(torch.from_numpy(curr_state).to("cuda:0"))
                    pred = step_predictor(predictor, rollout_queue, action_queue, state_queue)
                    rollout_queue.append(pred[:, -tokens_per_frame:])
                    pred_vae_latents = bridge(pred[:, -tokens_per_frame:], num_frames=2)
                    decoded = vae.decode(pred_vae_latents[0:1] / 0.18215).sample
                    #decoded = vae.decode(pred_vae_latents[0:1]).sample
                    pred_tensor = ((decoded.squeeze(0) + 1.0) / 2.0).clamp(0.0, 1.0)
                    pred_img = (pred_tensor.permute(1, 2, 0).float().cpu().numpy() * 255.0).astype(np.uint8)
                prev_state = curr_state

            combined_view = np.hstack([rgb_img, gt_us_img, pred_img])
            display_large = cv2.resize(combined_view, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Rollout Visualization: RGB | Ground Truth US | Imagined Rollout", display_large[..., ::-1])
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset
                print(f"Resetting at frame {i}")
                prev_state = None
                curr_state = None
                rollout_queue.clear()
                action_queue.clear()
                state_queue.clear()
                pred_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
