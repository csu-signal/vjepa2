import os
import copy
import time
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_ll_probe_guidance.ll_probe_guidance import LLProbeGuidanceDataset
from app.vjepa_ll_probe_guidance.transforms import make_transforms
from app.vjepa_ll_probe_guidance.utils import init_video_model, load_checkpoint, load_pretrained
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, get_logger, gpu_timer

logger = get_logger(__name__, force=True)


def load_state_dict_with_ddp_fix(model, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if it exists
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
    eval_dataset = LLProbeGuidanceDataset(
        data_root=data_root,
        frames_per_clip=max_num_frames,
        frame_skip=1, 
        frames_per_second=fps,
        transform=transform,
        is_train=False
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Length of eval_loader: {len(eval_loader)}")

    # -- load pretrained weights / checkpoint
    resume_path = os.path.join(folder, r_file)
    if os.path.exists(resume_path):
        logger.info(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=torch.device("cpu"))
        encoder = load_state_dict_with_ddp_fix(encoder, checkpoint["encoder"])
        predictor = load_state_dict_with_ddp_fix(predictor, checkpoint["predictor"])
        target_encoder = load_state_dict_with_ddp_fix(target_encoder, checkpoint["target_encoder"])
    else:
        logger.warning(f"Checkpoint not found at {resume_path}")

    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    loss_meter = AverageMeter()
    jloss_meter = AverageMeter()
    sloss_meter = AverageMeter()

    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for itr, sample in enumerate(eval_loader):
            clips = sample[0].to(device, non_blocking=True)
            actions = sample[1].to(device, dtype=torch.float, non_blocking=True)
            states = sample[2].to(device, dtype=torch.float, non_blocking=True)
            extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                # Target Pass
                c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                h = target_encoder(c)
                h = h.view(clips.shape[0], max_num_frames, -1, h.size(-1)).flatten(1, 2)
                if normalize_reps:
                    h = F.layer_norm(h, (h.size(-1),))

                # Prediction Pass (Teacher Forcing & Auto-Regressive)
                _z, _a, _s, _e = h[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
                
                if is_state_free:
                    z_tf = predictor(_z, _a, extrinsics=_e)
                else:
                    z_tf = predictor(_z, _a, _s, extrinsics=_e)
                    
                if normalize_reps:
                    z_tf = F.layer_norm(z_tf, (z_tf.size(-1),))

                _z_ar_input = torch.cat([h[:, : tokens_per_frame], z_tf[:, : tokens_per_frame]], dim=1)
                for n in range(1, auto_steps):
                    _a, _s, _e = actions[:, : n + 1], states[:, : n + 1], extrinsics[:, : n + 1]
                    if is_state_free:
                        _z_nxt = predictor(_z_ar_input, _a, extrinsics=_e)[:, -tokens_per_frame:]
                    else:
                        _z_nxt = predictor(_z_ar_input, _a, _s, extrinsics=_e)[:, -tokens_per_frame:]
                    
                    if normalize_reps:
                        _z_nxt = F.layer_norm(_z_nxt, (_z_nxt.size(-1),))
                        
                    _z_ar_input = torch.cat([_z_ar_input, _z_nxt], dim=1)
                z_ar = _z_ar_input[:, tokens_per_frame:]

                # Compute Loss
                _h_tf = h[:, tokens_per_frame : z_tf.size(1) + tokens_per_frame]
                _h_ar = h[:, tokens_per_frame : z_ar.size(1) + tokens_per_frame]
                
                jloss = torch.mean(torch.abs(z_tf - _h_tf) ** loss_exp) / loss_exp
                sloss = torch.mean(torch.abs(z_ar - _h_ar) ** loss_exp) / loss_exp
                loss = jloss + sloss

            loss_meter.update(float(loss))
            jloss_meter.update(float(jloss))
            sloss_meter.update(float(sloss))

            if itr % 10 == 0:
                logger.info(f"Eval Batch [{itr}/{len(eval_loader)}] - Loss: {loss_meter.avg:.4f} | JLoss (TF): {jloss_meter.avg:.4f} | SLoss (AR): {sloss_meter.avg:.4f}")

    logger.info("=======================================")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Final Total Energy Loss: {loss_meter.avg:.4f}")
    logger.info(f"Final Teacher Forcing Loss (jloss): {jloss_meter.avg:.4f}")
    logger.info(f"Final Auto-regressive Loss (sloss): {sloss_meter.avg:.4f}")
    logger.info("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True, help="Path to the config yaml file")
    args = parser.parse_args()

    with open(args.fname, "r") as f:
        config = yaml.safe_load(f)
        
    main(config)
