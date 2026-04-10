import argparse
import logging
import os
import json
import math
import random
from typing import Literal, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

logging.basicConfig(level=logging.INFO)

RESOLUTION_MAP = {
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}


# -------------------------
# Utility / Metrics (CPU)
# -------------------------

def flow_farneback_np(a_u8, b_u8, scale=0.25):
    # downsample for speed
    if scale != 1.0:
        a = cv2.resize(a_u8, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b_u8, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        a, b = a_u8, b_u8

    # accept either RGB (H,W,3) or grayscale (H,W)
    if a.ndim == 3:
        ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    else:
        ag = a
    if b.ndim == 3:
        bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    else:
        bg = b

    flow = cv2.calcOpticalFlowFarneback(
        ag, bg, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # if you downsampled, optionally rescale flow back to original resolution
    if scale != 1.0:
        flow = cv2.resize(flow, (a_u8.shape[1], a_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
        flow *= (1.0 / scale)

    return flow.astype(np.float32)  # (H,W,2)


def to_u8_rgb(x_chw: torch.Tensor) -> np.ndarray:
    """
    x_chw: (3,H,W) or (1,H,W) float tensor on any device.
    returns: (H,W,3) uint8
    """
    x = x_chw.detach().float().cpu()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    # map common ranges to [0,255]
    xmin = float(x.min())
    xmax = float(x.max())
    if xmin >= 0.0 and xmax <= 1.5:
        x = x * 255.0
    elif xmin >= -1.5 and xmax <= 1.5:
        x = (x + 1.0) * 0.5 * 255.0
    else:
        x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 255.0

    x = x.clamp(0, 255).byte()
    return x.permute(1, 2, 0).numpy()

def to_u8_gray(x_chw: torch.Tensor) -> np.ndarray:
    """
    x_chw: (3,H,W) or (1,H,W) float tensor.
    returns: (H,W) uint8
    """
    x = x_chw.detach().float().cpu()
    if x.shape[0] == 3:
        g = 0.2989*x[0] + 0.5870*x[1] + 0.1140*x[2]
    else:
        g = x[0]

    # standardize (robust)
    m = float(g.mean())
    s = float(g.std() + 1e-6)
    g = (g - m) / s

    # clamp to a reasonable range then map to uint8
    g = g.clamp(-3, 3)
    g = ((g + 3) / 6.0 * 255.0).clamp(0, 255).byte().numpy()
    return g

def flow_cos_sim(flow_a: np.ndarray, flow_b: np.ndarray, eps=1e-8, mag_thresh=0.2) -> float:
    a = flow_a.reshape(-1, 2)
    b = flow_b.reshape(-1, 2)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    m = (na > mag_thresh) & (nb > mag_thresh)
    if m.sum() == 0:
        return 0.0
    a = a[m]; b = b[m]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps)
    return float(cos.mean())

# -------------------------
# Main generation
# -------------------------
def generate_video(
    prompt: str,
    model_path: str,
    output_path: str,
    image_or_video_path: str = "",
    generate_type: str = "t2v",
    num_inference_steps: int = 50,
    num_frames: int = 81,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
    fps: int = 16,
    width: Optional[int] = None,
    height: Optional[int] = None,
    lora_path: Optional[str] = None,
    lora_rank: int = 128,
    guide: bool = False,
    guide_k: int = 2,
    guide_long_side: int = 256,
    guide_steps: int = 1,
):
    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    N = num_inference_steps

    # Resolution
    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP.get(model_name, None)
    if desired_resolution is not None:
        if width is None or height is None:
            height, width = desired_resolution
            logging.info(f"Using default resolution {desired_resolution} for {model_name}")
        elif (height, width) != desired_resolution and generate_type != "i2v":
            logging.warning(f"{model_name} not supported for custom resolution; using {desired_resolution}")
            height, width = desired_resolution
    else:
        # fallback if unknown
        height = height or 480
        width = width or 720

    # Multi-GPU sharding (balanced) with headroom
    ngpu = torch.cuda.device_count()
    max_memory = None
    device_map = None
    if ngpu >= 2:
        # leave headroom for activations
        max_memory = {i: "60GiB" for i in range(ngpu)}
        device_map = "balanced"

    # Load pipeline
    image = None
    video = None
    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map, max_memory=max_memory
        )
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map, max_memory=max_memory
        )
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map, max_memory=max_memory
        )
        video = load_video(image_or_video_path)

    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="exp")
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    logging.info(f"Scheduler: {pipe.scheduler.__class__.__name__}")

    if device_map is None:
        pipe.to("cuda")

    # VAE memory helpers
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()


    # Callback (attached always; guidance runs only if `guide` and step is in guided_decode_steps)
    def cb_on_step_end(pipe, step_index, timestep, callback_kwargs):
        with torch.no_grad():
            x_full = callback_kwargs["latents"].detach()

            # decode a small packed-time window
            pack_T = x_full.shape[1]
            start = max(0, min(pack_T - 2, pack_T // 2 - 1))
            x_win = x_full[:, start:start + 2]  

            latent_frames = x_win
            frames = pipe.decode_latents(latent_frames) 
            B, C, T, H, W = frames.shape
            assert C == 3

            # downsample
            long_side = guide_long_side
            sc = long_side / max(H, W)
            h2 = max(8, int(H * sc))
            w2 = max(8, int(W * sc))

            frames_s = F.interpolate(
                frames.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W),
                size=(h2, w2),
                mode="bilinear",
                align_corners=False
            ).reshape(B, T, 3, h2, w2)

            B2, T2, C_lat, H_lat, W_lat = latent_frames.shape  # NOTE: T2 should be 2

            # 1) project latents -> 1-channel "image" per latent slice
            lat_proj = latent_frames.float().pow(2).mean(dim=2, keepdim=True).sqrt()  # (B,2,1,H_lat,W_lat)§§   

            # 2) resize to (h2,w2) like you do for RGB
            lat_proj_s = F.interpolate(
                lat_proj.reshape(B2 * T2, 1, H_lat, W_lat),
                size=(h2, w2),
                mode="bilinear",
                align_corners=False
            ).reshape(B2, T2, 1, h2, w2)

            # 3) compute flow on the single adjacent pair (0,1)
            Ia = frames_s[:, 0]  # (B,3,h2,w2)
            Ib = frames_s[:, 1]

            latent_Ia = lat_proj_s[:, 0]  # (B,1,h2,w2)
            latent_Ib = lat_proj_s[:, 1]

            Ia_u8 = to_u8_gray(Ia[0])
            Ib_u8 = to_u8_gray(Ib[0])
            flow_rgb = flow_farneback_np(Ia_u8, Ib_u8, scale=1.0)

            latent_Ia_u8 = to_u8_gray(latent_Ia[0])
            latent_Ib_u8 = to_u8_gray(latent_Ib[0])
            flow_lat = flow_farneback_np(latent_Ia_u8, latent_Ib_u8, scale=1.0)

            sim = flow_cos_sim(flow_rgb, flow_lat)
            logging.info(f"[flow-corr] step={step_index} start={start} sim={sim:.4f}")

        return callback_kwargs

    # Common pipe kwargs
    common_kwargs = dict(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale, 
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )

    if guide:
        common_kwargs.update(
            callback_on_step_end=cb_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )

    if generate_type == "i2v":
        common_kwargs["image"] = image
    elif generate_type == "v2v":
        common_kwargs["video"] = video

    logging.info("About to call pipe(...)")
    video_generate = pipe(**common_kwargs).frames[0]
    logging.info("pipe(...) returned")



    # Save trace + snapshots
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_or_video_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX1.5-5B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--output_path", type=str, default="./output.mp4")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)
    parser.add_argument("--generate_type", type=str, default="t2v")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)

    # Guidance toggles
    parser.add_argument("--guide", action="store_true", help="Enable guidance in callback")
    parser.add_argument("--guide_k", type=int, default=2, help="Packed-time window size K for decode")
    parser.add_argument("--guide_long_side", type=int, default=256, help="Downsample long side for guidance ops")
    parser.add_argument("--guide_steps", type=int, default=1, help="How many last denoise steps to guide")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path or "",
        generate_type=args.generate_type,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
        width=args.width,
        height=args.height,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        guide=args.guide,
        guide_k=args.guide_k,
        guide_long_side=args.guide_long_side,
        guide_steps=args.guide_steps,
    )