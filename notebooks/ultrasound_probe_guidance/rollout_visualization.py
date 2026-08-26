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
import torchvision.transforms as T
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from diffusers import AutoencoderKL
import cv2

from app.vjepa_ll_probe_guidance.utils import init_video_model
from app.vjepa_ll_probe_guidance.transforms import make_transforms


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
        # TODO learn how the einops rearrange function calls work here
        T_p = num_frames // 2
        H_p = height // 16
        W_p = width // 16

        # Unfold tokens into a grid, each grid point has 1024 dimension
        # vjepa2 tokens original shape: [batch, tokens, embedding dimension]
        # vjepa2 tokens reshaped to: [batch, time, height, width, embedding_dimension]
        x = rearrange(vjepa_tokens, "b (t h w) d -> b t h w d", t=T_p, h=H_p, w=W_p)

        # Use the linear layer to "learn" how to split the tubelet back to 2 individual frames
        # New shape: [batch, time, height, width, embedding_dimension * 2]
        x = self.temporal_split(x)

        # Actually unpack the split frames along the time axis and move embedding dimension to channel spot for 2D convs
        # New shape: [batch, (time * 2), embedding_dimension, height, width]
        # TODO learn how this is working and also verify that its doing what I want it to do
        x = rearrange(x, "b t h w (repeat d) -> b (t repeat) d h w", repeat=2)

        # Merge the batch and the time dimension so that the VAE can decode each frame individually
        # New shape: [batch * (time * 2), embedding_dimension, height, width]
        x = rearrange(x, "b t d h w -> (b t) d h w")

        # Now begin translating the vjepa2 tokens to the VAE latent space for the decoder
        x = self.upconv(x)
        x = self.act(x)
        vae_latents = self.final_proj(x)

        return vae_latents


def standardize_states(states):
    state_mean = torch.tensor([ -0.13193196, 0.13471916, 1.4406046,
        59.395, -43.585613, -76.49974], dtype=torch.float32, device=states.device)
    state_std = torch.tensor([1.05345368e-01, 1.75726220e-01, 7.28756189e-02,
        7.79652252e+01, 3.07379379e+01, 1.22978004e+02], dtype=torch.float32, device=states.device)

    return (states - state_mean) / (state_std + 1e-6)


def standardize_actions(actions):
    action_mean = torch.tensor([-2.7906366e-05,  5.4028669e-06,  6.0837847e-05,
        -1.7707590e-03, 2.4102912e-03, -3.7514704e-04], dtype=torch.float32, device=actions.device)
    action_std = torch.tensor([0.00498742, 0.00453256, 0.00675807,
        0.55955255, 0.43809873, 0.73082167], dtype=torch.float32, device=actions.device)

    return (actions - action_mean) / (action_std + 1e-6)


def forward_target(encoder, c, normalize_reps=True):
    c = c[None, :, None, ...].repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(1, 1, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def step_predictor(predictor, rollout_queue, action_queue, state_queue, normalize_reps=True):
    z = torch.cat(list(rollout_queue), dim=1)
    a = np.stack(list(action_queue), axis=0)
    a = torch.tensor(a, device="cuda:0").unsqueeze(0)
    s = np.stack(list(state_queue), axis=0)
    s = torch.tensor(s, device="cuda:0").unsqueeze(0)
    standardized_actions = standardize_actions(a).to("cuda:0", dtype=torch.float, non_blocking=True)
    standardized_states = standardize_states(s).to("cuda:0", dtype=torch.float, non_blocking=True)
    with torch.no_grad():
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
            # Remove 'module.' prefix if it exists
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
    rgb_transform = T.Compose([
        T.ToTensor(),
        T.Resize((crop_size, crop_size), antialias=True),
    ])
    ultrasound_transform = T.Compose([
        T.ToTensor(),
        T.Resize((crop_size, crop_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
    pred_img = None
    for i in tqdm(range(us_file_count), total=us_file_count):
        # NOTE: frame name format is hardcoded here. it's 5 digits with leading zeros for each frame
        us_img_path = os.path.join(us_dir, f"{i:05d}.jpg")
        us_img = torchvision.io.read_image(us_img_path, mode=torchvision.io.ImageReadMode.RGB)
        us_img = us_img.permute(1, 2, 0).numpy()
        gt_us_img = (rgb_transform(us_img).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
        
        rgb_img_path = os.path.join(rgb_dir, f"{i:05d}.jpg")
        rgb_img = torchvision.io.read_image(rgb_img_path, mode=torchvision.io.ImageReadMode.RGB)
        rgb_img = rgb_img.permute(1, 2, 0).numpy()
        rgb_img = (rgb_transform(rgb_img).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")

        # Show GT ultrasound image
        cv2.imshow("Ground truth ultrasound image", gt_us_img)
        cv2.waitKey(1)
        # Show RGB image
        cv2.imshow("RGB image", rgb_img)
        cv2.waitKey(1)
        # Show decoded prediction image (if available, else placeholder frame)
        if pred_img is not None:
            cv2.imshow("Imagined rollout image", pred_img)
        #else:
        #    cv2.imshow("Imagined rollout image", placeholder_img)

        # init rollout queue
        if len(rollout_queue) == 0:
            wm_us_img = ultrasound_transform(us_img).to("cuda:0", non_blocking=True)
            with torch.no_grad():
                h = forward_target(encoder, wm_us_img)
                rollout_queue.append(h)

        curr_state = states[i] if not np.any(np.isnan(states[i])) else None
        if curr_state is not None:
            if prev_state is not None:
                action_queue.append(get_action(prev_state, curr_state))
                pred = step_predictor(predictor, rollout_queue, action_queue, state_queue)
                rollout_queue.append(pred[:, -tokens_per_frame:])
                pred_vae_latents = bridge(pred[:, -tokens_per_frame:], num_frames=2)
                pred_img = vae.decode(pred_vae_latents[0])
            state_queue.append(curr_state)
        prev_state = curr_state
        #print(f"iter {i}: {len(rollout_queue)=}, {len(action_queue)=}, {len(state_queue)=}") 

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
