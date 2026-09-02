# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time
import math

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import wandb

from app.vjepa_ll_probe_guidance_v2.ll_probe_guidance import init_data, standardize_states, standardize_actions, load_stats_file
from app.vjepa_ll_probe_guidance_v2.transforms import make_transforms
from app.vjepa_ll_probe_guidance_v2.utils import init_opt, init_video_model, load_checkpoint, load_pretrained
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    eval_freq = cfgs_meta.get("eval_freq", 10)
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", False)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype", "bfloat16")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    predictor_type = cfgs_model.get("predictor_type", "sfac") 
    is_state_free = (predictor_type == "sfac")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)

    # -- DATA
    cfgs_data = args.get("data")
    data_root = cfgs_data.get("data_root")
    val_data_root = cfgs_data.get("val_data_root")
    max_num_frames = cfgs_data.get("frames_per_clip", 16)
    train_batch_size = cfgs_data.get("batch_size")
    val_batch_size = cfgs_data.get("val_batch_size", train_batch_size)
    tubelet_size = cfgs_data.get("tubelet_size", 2)
    fps = cfgs_data.get("fps", 4)
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size", 16)
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 4)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    pose_source = cfgs_data.get("pose_source", "tip")
    stats_file = cfgs_data.get("stats_file", None)

    if stats_file and os.path.exists(stats_file):
        load_stats_file(stats_file)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp", 1.0)
    normalize_reps = cfgs_loss.get("normalize_reps", True)
    auto_steps = cfgs_loss.get("auto_steps", 1)
    use_rollout_curriculum = cfgs_loss.get("use_rollout_curriculum", False)
    # --
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay", 0.04))
    final_wd = float(cfgs_opt.get("final_weight_decay", 0.04))
    num_epochs = cfgs_opt.get("epochs", 100)
    anneal = cfgs_opt.get("anneal", 10)
    warmup = cfgs_opt.get("warmup", 10)
    start_lr = cfgs_opt.get("start_lr", 1e-5)
    lr = cfgs_opt.get("lr", 1e-4)
    final_lr = cfgs_opt.get("final_lr", 0.0)
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if rank == 0:
        wandb.init(
            project="vjepa2-ultrasound-ac-predictor",
            name=args.get("exp_name", f"{model_name}-v2{'-curriculum' if use_rollout_curriculum else ''}"),
            config=args,
            resume="allow" if r_file is not None else False,
        )

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
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
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    video_collator = torch.utils.data.default_collate
    transform = make_transforms(
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data_root=data_root,
        batch_size=train_batch_size,
        frames_per_clip=max_num_frames,
        frame_skip=1,
        fps=fps,
        transform=transform,
        collator=video_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
        pose_source=pose_source,
        stats_file=stats_file,
    )

    val_loader = None
    if rank == 0 and val_data_root and os.path.exists(val_data_root):
        val_loader, _ = init_data(
            data_root=val_data_root,
            batch_size=val_batch_size,
            frames_per_clip=max_num_frames,
            frame_skip=1,
            fps=fps,
            transform=transform,
            collator=video_collator,
            num_workers=num_workers,
            world_size=1,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            rank=0,
            is_train=False,
            pose_source=pose_source,
            stats_file=stats_file,
        )

    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- load pretrained weights
    encoder, predictor, target_encoder = load_pretrained(
        r_path=p_file,
        encoder=encoder,
        predictor=predictor,
        context_encoder_key=context_encoder_key,
        target_encoder_key=target_encoder_key,
        target_encoder=target_encoder,
        load_predictor=load_predictor,
        load_encoder=load_encoder,
    )

    start_epoch = 0
    # -- load training checkpoint
    if resume_path is not None and os.path.exists(resume_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=resume_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": train_batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    def forward_target(c, batch_size):
        with torch.no_grad():
            c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            h = target_encoder(c)
            h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
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
        clips = sample[0].to(device, non_blocking=True)  # [B C T H W]
        actions = standardize_actions(sample[1]).to(device, dtype=torch.float, non_blocking=True)  # [B T-1 6]
        states = standardize_states(sample[2]).to(device, dtype=torch.float, non_blocking=True)  # [B T 6]
        extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)  # [B T 6]
        return (clips, actions, states, extrinsics)

    def validate(encoder, predictor, target_encoder, loader, device, curr_auto_steps):
        encoder.eval()
        predictor.eval()

        val_loss_meter = AverageMeter()
        val_jloss_meter = AverageMeter()
        val_sloss_meter = AverageMeter()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                for sample in tqdm(loader, total=len(loader), desc="Performing validation"):
                    clips, actions, states, extrinsics = load_clips(sample)

                    h = forward_target(clips, len(clips))
                    z_tf, z_ar = forward_predictions(h, actions, states, extrinsics, curr_auto_steps)

                    jloss = loss_fn(z_tf, h)
                    sloss = loss_fn(z_ar, h)
                    loss = jloss + sloss
                    
                    val_jloss_meter.update(float(jloss))
                    val_sloss_meter.update(float(sloss))
                    val_loss_meter.update(float(loss))

        encoder.train()
        predictor.train()
        return val_loss_meter.avg, val_jloss_meter.avg, val_sloss_meter.avg
    
    best_val_loss = math.inf

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        if use_rollout_curriculum:
            if 0 <= epoch < int(num_epochs * 0.25):
                curr_auto_steps = min(2, auto_steps)
            elif int(num_epochs * 0.25) <= epoch < int(num_epochs * 0.50):
                curr_auto_steps = min(4, auto_steps)
            else:
                curr_auto_steps = auto_steps
        else:
            curr_auto_steps = auto_steps

        logger.info(f"Active rollout horizon (auto_steps): {curr_auto_steps}")

        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        # TRAINING
        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            clips, actions, states, extrinsics = load_clips(sample)
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_pass():
                    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                        h = forward_target(clips, train_batch_size)
                        z_tf, z_ar = forward_predictions(h, actions, states, extrinsics, curr_auto_steps)
                        jloss = loss_fn(z_tf, h)
                        sloss = loss_fn(z_ar, h)
                        loss = jloss + sloss
                        return loss, jloss, sloss

                if mixed_precision and scaler is not None:
                    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                        loss, jloss, sloss = forward_pass()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss, jloss, sloss = forward_pass()
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad()
                return loss.item(), jloss.item(), sloss.item(), _new_lr, _new_wd

            (loss, jloss, sloss, _new_lr, _new_wd), gpu_time_ms = gpu_timer(train_step)

            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            iter_time_meter.update((time.time() - itr_start_time) * 1000.0)
            gpu_time_meter.update(gpu_time_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            if itr % log_freq == 0 or itr == ipe - 1:
                logger.info(
                    f"[{epoch + 1:03d}/{num_epochs:03d}][{itr + 1:04d}/{ipe:04d}] "
                    f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                    f"J-Loss: {jloss_meter.val:.4f} ({jloss_meter.avg:.4f}) | "
                    f"S-Loss: {sloss_meter.val:.4f} ({sloss_meter.avg:.4f}) | "
                    f"LR: {_new_lr:.6f} | "
                    f"WD: {_new_wd:.4f} | "
                    f"Data: {data_elapsed_time_meter.avg:.1f}ms | "
                    f"GPU: {gpu_time_meter.avg:.1f}ms"
                )

                if rank == 0:
                    csv_logger.log(
                        epoch + 1,
                        itr + 1,
                        loss_meter.val,
                        int(iter_time_meter.val),
                        int(gpu_time_meter.val),
                        int(data_elapsed_time_meter.val),
                    )
                    wandb.log(
                        {
                            "train/loss": loss_meter.val,
                            "train/jloss": jloss_meter.val,
                            "train/sloss": sloss_meter.val,
                            "train/lr": _new_lr,
                            "train/wd": _new_wd,
                            "train/step": epoch * ipe + itr,
                        }
                    )

        # Periodic Validation
        if val_loader is not None and (epoch + 1) % eval_freq == 0:
            val_loss, val_jloss, val_sloss = validate(
                encoder, predictor, target_encoder, val_loader, device, curr_auto_steps
            )
            logger.info(
                f"[Validation Epoch {epoch + 1}] Loss: {val_loss:.4f} | "
                f"J-Loss: {val_jloss:.4f} | S-Loss: {val_sloss:.4f}"
            )
            if rank == 0:
                wandb.log({
                    "val/loss": val_loss,
                    "val/jloss": val_jloss,
                    "val/sloss": val_sloss,
                    "val/epoch": epoch + 1,
                })
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(epoch + 1, os.path.join(folder, "best.pt"))

        # Save Checkpoints
        if rank == 0:
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_checkpoint(epoch + 1, os.path.join(folder, f"epoch_{epoch + 1}.pt"))

    if rank == 0:
        wandb.finish()
