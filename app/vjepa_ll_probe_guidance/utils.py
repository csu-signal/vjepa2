import torch

from src.models.vision_transformer import vit_base, vit_large, vit_huge, vit_giant

from src.models.ac_predictor import vit_ac_predictor
from app.vjepa_ll_probe_guidance.state_free_ac_predictor import vit_sfac_predictor

# Pass-through imports from the DROID utils
from app.vjepa_droid.utils import init_opt, load_checkpoint, load_pretrained

def init_video_model(
    device,
    patch_size=16,
    model_name="vit_base",
    max_num_frames=512,
    crop_size=256,
    pred_depth=12,
    pred_embed_dim=768,
    pred_num_heads=12,
    action_embed_dim=6,
    use_extrinsics=False,
    predictor_type="sfac", # Can be "sfac" or "ac"
    **kwargs
):
    if model_name == "vit_large":
        encoder = vit_large(patch_size=patch_size, img_size=crop_size, num_frames=max_num_frames, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    encoder.to(device)

    if predictor_type == "sfac":
        predictor = vit_sfac_predictor(
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=pred_embed_dim,
            num_frames=max_num_frames,
            depth=pred_depth,
            num_heads=pred_num_heads,
            patch_size=patch_size,
            img_size=crop_size,
            action_embed_dim=action_embed_dim,
            use_extrinsics=use_extrinsics,
            **kwargs
        )
    elif predictor_type == "ac":
        predictor = vit_ac_predictor(
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=pred_embed_dim,
            num_frames=max_num_frames,
            depth=pred_depth,
            num_heads=pred_num_heads,
            patch_size=patch_size,
            img_size=crop_size,
            action_embed_dim=action_embed_dim,
            use_extrinsics=use_extrinsics,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}. Choose 'sfac' or 'ac'.")

    predictor.to(device)

    return encoder, predictor
