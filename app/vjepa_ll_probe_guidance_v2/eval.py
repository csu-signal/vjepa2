import os
import copy
import time
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from app.vjepa_ll_probe_guidance_v2.ll_probe_guidance import LLProbeGuidanceV2Dataset, standardize_states, standardize_actions, load_stats_file
from app.vjepa_ll_probe_guidance_v2.transforms import make_transforms
from app.vjepa_ll_probe_guidance_v2.utils import init_video_model, load_checkpoint, load_pretrained
from src.utils.logging import AverageMeter, get_logger

logger = get_logger(__name__, force=True)


def load_state_dict_with_ddp_fix(model, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)
    return model


def main(args):
    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", "latest.pt")
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    
    which_dtype = cfgs_meta.get("dtype", "float16")
    dtype = torch.bfloat16 if which_dtype.lower() == "bfloat16" else torch.float16
    mixed_precision = True

    # -- MODEL
    cfgs_model = args.get("model")
    model_name = cfgs_model.get("model_name")
    predictor_type = cfgs_model.get("predictor_type", "sfac") 
    is_state_free = (predictor_type == "sfac")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)

    # -- DATA
    cfgs_data = args.get("data")
    data_root = cfgs_data.get("data_root")
    max_num_frames = cfgs_data.get("frames_per_clip", 16)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size", 2)
    fps = cfgs_data.get("fps", 4)
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size", 16)
    num_workers = cfgs_data.get("num_workers", 4)
    pose_source = cfgs_data.get("pose_source", "tip")
    stats_file = cfgs_data.get("stats_file", None)

    if stats_file and os.path.exists(stats_file):
        load_stats_file(stats_file)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp", 1)
    normalize_reps = cfgs_loss.get("normalize_reps", False)
    auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # -- init model
    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=512, 
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=6,
        predictor_type=predictor_type,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
    )
    target_encoder = copy.deepcopy(encoder)

    transform = make_transforms(crop_size=crop_size)

    # -- init dataloader
    dataset = LLProbeGuidanceV2Dataset(
        data_root=data_root,
        frames_per_clip=max_num_frames,
        frame_skip=1,
        frames_per_second=fps,
        transform=transform,
        is_train=False,
        pose_source=pose_source,
        stats_file=stats_file,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=False,
    )

    logger.info(f"Evaluation Dataset Size (Clips): {len(dataset)}")

    # -- load checkpoint
    chk_path = os.path.join(folder, r_file) if not os.path.isabs(r_file) else r_file
    if not os.path.exists(chk_path):
        chk_path = os.path.join(folder, "latest.pt")
    
    logger.info(f"Loading checkpoint from {chk_path}")
    checkpoint = torch.load(chk_path, map_location="cpu")
    
    if "encoder" in checkpoint:
        load_state_dict_with_ddp_fix(encoder, checkpoint["encoder"])
    if "predictor" in checkpoint:
        load_state_dict_with_ddp_fix(predictor, checkpoint["predictor"])
    if "target_encoder" in checkpoint:
        load_state_dict_with_ddp_fix(target_encoder, checkpoint["target_encoder"])

    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    def forward_target(c, b_size):
        with torch.no_grad():
            c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            h = target_encoder(c)
            h = h.view(b_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
            if normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
            return h

    def forward_predictions(z, actions, states, extrinsics, curr_auto_steps):
        def _step_predictor(_z, _a, _s, _e):
            if is_state_free:
                _z = predictor(_z, _a, extrinsics=_e)
            else:
                _z = predictor(_z, _a, _s, extrinsics=_e)
            if normalize_reps:
                _z = F.layer_norm(_z, (_z.size(-1),))
            return _z

        _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
        z_tf = _step_predictor(_z, _a, _s, _e)

        _z = torch.cat([z[:, : tokens_per_frame], z_tf[:, : tokens_per_frame]], dim=1)
       
        rollout_steps = min(curr_auto_steps, actions.size(1)) 
        for n in range(1, rollout_steps):
            _a, _s, _e = actions[:, : n + 1], states[:, : n + 1], extrinsics[:, : n + 1]
            _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
            _z = torch.cat([_z, _z_nxt], dim=1)
        z_ar = _z[:, tokens_per_frame:]
        return z_tf, z_ar

    def loss_fn(z, h):
        _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
        return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

    def load_clips(sample):
        clips = sample[0].to(device, non_blocking=True)
        actions = standardize_actions(sample[1]).to(device, dtype=torch.float, non_blocking=True)
        states = standardize_states(sample[2]).to(device, dtype=torch.float, non_blocking=True)
        extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)
        return (clips, actions, states, extrinsics)

    val_loss_meter = AverageMeter()
    val_jloss_meter = AverageMeter()
    val_sloss_meter = AverageMeter()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
            for sample in tqdm(loader, total=len(loader), desc="Evaluating"):
                clips, actions, states, extrinsics = load_clips(sample)
                h = forward_target(clips, len(clips))
                z_tf, z_ar = forward_predictions(h, actions, states, extrinsics, auto_steps)

                jloss = loss_fn(z_tf, h)
                sloss = loss_fn(z_ar, h)
                loss = jloss + sloss

                val_jloss_meter.update(float(jloss), len(clips))
                val_sloss_meter.update(float(sloss), len(clips))
                val_loss_meter.update(float(loss), len(clips))

    logger.info("=" * 60)
    logger.info(f"Evaluation Complete across {len(dataset)} clips:")
    logger.info(f"Total Loss:  {val_loss_meter.avg:.5f}")
    logger.info(f"Joint Loss:  {val_jloss_meter.avg:.5f}")
    logger.info(f"Seq Loss:    {val_sloss_meter.avg:.5f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True, help="Config YAML path")
    args_cli = parser.parse_args()
    
    with open(args_cli.fname, "r") as f:
        config = yaml.safe_load(f)
    main(config)
