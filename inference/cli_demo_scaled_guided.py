"""
CogVideoX video generation + optional sampling-time guidance (training-free).

Usage:
  Baseline:
    python cli_demo_guided.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-2b --output_path OUT/base.mp4 --dtype float16 --seed 42

  Guided:
    python cli_demo_guided.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-2b --output_path OUT/guided.mp4 --dtype float16 --seed 42 --guide

Notes:
- Baseline and guided run the same script; only the --guide flag toggles guidance.
- Metrics are computed on CPU using OpenCV (so they don't consume GPU VRAM).
"""

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
def to_np_uint8(frames):
    """frames can be list[PIL], np array, etc. Return (T,H,W,3) uint8."""
    if isinstance(frames, (list, tuple)):
        return np.stack([np.asarray(f) for f in frames], axis=0).astype(np.uint8)
    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def sobel_edges_np(rgb_u8):
    g = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy + 1e-6).astype(np.float32)  # (H,W)


def flow_farneback_np(a_u8, b_u8, scale=0.25):
    """Compute flow A->B on downsampled images, return flow at that downsampled resolution (no rescale back)."""
    if scale != 1.0:
        a = cv2.resize(a_u8, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b_u8, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        a, b = a_u8, b_u8

    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        ag, bg, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow.astype(np.float32)  # (h,w,2) in pixels of that resolution


def warp_edge_np(edge_b, flow_ab):
    """Warp edge map from B to A using flow A->B (pull B back to A by sampling B at x+flow)."""
    h, w = edge_b.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xx + flow_ab[..., 0]
    map_y = yy + flow_ab[..., 1]
    return cv2.remap(edge_b, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flow_edge_metric(video_frames, flow_scale=0.25, normalize=False):
    """Motion-compensated edge mismatch over consecutive frames. Lower is better."""
    arr = to_np_uint8(video_frames)  # (T,H,W,3)
    T = arr.shape[0]
    errs = []
    for t in range(T - 1):
        a = arr[t]
        b = arr[t + 1]
        if flow_scale != 1.0:
            a_s = cv2.resize(a, (0, 0), fx=flow_scale, fy=flow_scale, interpolation=cv2.INTER_AREA)
            b_s = cv2.resize(b, (0, 0), fx=flow_scale, fy=flow_scale, interpolation=cv2.INTER_AREA)
        else:
            a_s, b_s = a, b

        Ea = sobel_edges_np(a_s)
        Eb = sobel_edges_np(b_s)
        if normalize:
            Ea = Ea / (Ea.mean() + 1e-6)
            Eb = Eb / (Eb.mean() + 1e-6)

        flow = flow_farneback_np(a_s, b_s, scale=1.0)  # already at scale'd resolution
        Eb_w = warp_edge_np(Eb, flow)                  # warp B -> A using +flow
        errs.append(float(np.mean(np.abs(Ea - Eb_w))))
    return float(np.mean(errs))


def temporal_edge_flicker_cpu(video_frames, scale=0.25):
    """Non-motion-compensated edge flicker (penalizes real motion too)."""
    arr = to_np_uint8(video_frames)
    if scale != 1.0:
        arr = np.stack([cv2.resize(f, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in arr], axis=0)
    edges = np.stack([sobel_edges_np(f) for f in arr], axis=0)  # (T,H,W)
    return float(np.mean(np.abs(edges[1:] - edges[:-1])))


# -------------------------
# Torch-side ops for guidance
# -------------------------
def sobel_mag_torch(img):  # img: (B,3,H,W) torch
    g = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)  # (B,1,H,W)


def warp_torch(x, flow_np):
    """x: (B,1,H,W) torch, flow_np: (H,W,2) numpy in pixel units (dx,dy)"""
    B, C, H, W = x.shape
    device = x.device

    flow = torch.from_numpy(flow_np).to(device=device, dtype=torch.float32)  # keep flow float32
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    grid = torch.stack([xx, yy], dim=-1).float()
    grid = grid + flow

    grid_x = 2.0 * (grid[..., 0] / (W - 1)) - 1.0
    grid_y = 2.0 * (grid[..., 1] / (H - 1)) - 1.0
    grid_n = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2) float32
    grid_n = grid_n.to(dtype=x.dtype)  # must match x dtype

    return F.grid_sample(x, grid_n, mode="bilinear", padding_mode="border", align_corners=True)


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

    # Guidance happens only on the last `guide_steps` steps (default 1)
    guided_decode_steps = set(range(max(0, N - guide_steps), N)) if guide else set()

    def rms_per_sample(x, eps=1e-8):
        return (x.float().pow(2).mean() + eps).sqrt().item()

    def rms(x, eps=1e-8):
        if torch.is_tensor(x):
            return (x.float().pow(2).mean() + eps).sqrt()
        return math.sqrt(float(x) * float(x) + eps)

    def rho_schedule(i, N, rho_max=0.0015, p_on=0.90, p_peak=0.98, p_taper=0.99, rho_end=0.0001):
        p = i / (N - 1)
        if p < p_on:
            return 0.0
        if p < p_peak:
            return rho_max * (p - p_on) / (p_peak - p_on)
        if p < p_taper:
            return rho_max
        return rho_end + (rho_max - rho_end) * (1 - (p - p_taper) / (1 - p_taper))

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
    
    print("hf_device_map:", getattr(pipe, "hf_device_map", None))
    print("transformer device:", next(pipe.transformer.parameters()).device)
    print("vae device:", next(pipe.vae.parameters()).device)

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

    trace = {"guide": bool(guide)}
    snapshots = {}
    snapshot_steps = {0, N // 2, N - 1}

    # Scheduler step wrapper (for trust-region)
    orig_step = pipe.scheduler.step
    delta_model_rms_list = []
    trace["delta_model_rms"] = []
    trace["x_t_rms"] = []
    trace["x_prev_rms"] = []

    def step_wrapped(model_output, timestep, sample, *args, **kwargs):
        out = orig_step(model_output, timestep, sample, *args, **kwargs)
        x_t = sample
        x_prev = out[0] if isinstance(out, tuple) else out.prev_sample
        delta = x_prev - x_t
        dn = float(rms(delta).item())
        delta_model_rms_list.append(dn)
        trace["x_t_rms"].append(rms_per_sample(x_t))
        trace["x_prev_rms"].append(rms_per_sample(x_prev))
        trace["delta_model_rms"].append(dn)
        return out

    pipe.scheduler.step = step_wrapped

    # Callback (attached always; guidance runs only if `guide` and step is in guided_decode_steps)
    def cb_on_step_end(pipe, step_index, timestep, callback_kwargs):
        # --- inside cb_on_step_end(pipe, step_index, timestep, callback_kwargs) ---

        rho_t = rho_schedule(step_index, N)
        trace.setdefault("rho", []).append(float(rho_t))

        # gate: last guide_steps only, and rho>0
        if not (guide and (step_index in guided_decode_steps) and (rho_t > 0.0)):
            # snapshots (always)
            if step_index in snapshot_steps:
                snapshots[str(step_index)] = callback_kwargs["latents"].detach().to("cpu", dtype=torch.float16)
            return callback_kwargs

        trace.setdefault("guided_steps", []).append(int(step_index))

        # trust-region reference scale from model step magnitude
        delta_norm = delta_model_rms_list[step_index] if step_index < len(delta_model_rms_list) else delta_model_rms_list[-1]

        with torch.enable_grad():
            lat_old = callback_kwargs["latents"]
            lat_fp32 = lat_old.float()  # master copy for updates/grad accumulation

            B, pack_T, C_lat, H_lat, W_lat = lat_fp32.shape
            if pack_T < 2:
                return callback_kwargs

            decode_dtype = lat_old.dtype

            g_total_full = torch.zeros_like(lat_fp32)  # full gradient accumulator
            E_total = 0.0

            # # loop over ALL adjacent packed pairs: (0,1), (1,2), ..., (pack_T-2, pack_T-1)
            # for i in range(pack_T - 1):
            pair_indices = []  # pair i means slices (i, i+1)
            # trace.setdefault("guided_pair_indices", []).append([int(pair_indices[0]), int(pair_indices[1])])

            for i in pair_indices:
                # fp32 leaf for this pair, but decode in fp16/bf16
                x_pair_fp32 = lat_fp32[:, i:i+2].detach().requires_grad_(True)  # (B,2,C,H,W) fp32 leaf
                x_pair = x_pair_fp32.to(dtype=decode_dtype)                     # cast for VAE decode

                # decode -> (B,3,Tdec,H_img,W_img)
                # NOTE: In CogVideoX, Tdec is NOT guaranteed to equal the packed window length (2).
                # For example, decoding a 2-slice packed window can yield Tdec=8 decoded frames.
                frames = pipe.decode_latents(x_pair)
                B2, C, T, H, W = frames.shape
                # We only need TWO decoded frames to compute a flow/warp constraint.
                # For a fair comparison with the "regular" guidance script, we use the FIRST two
                # decoded frames (t=0 and t=1) regardless of Tdec.
                if not (C == 3 and T >= 2):
                    del x_pair_fp32, x_pair, frames
                    continue

                # downsample decoded frames for cheaper flow/edges
                long_side = guide_long_side
                sc = long_side / max(H, W)
                h2 = max(8, int(H * sc))
                w2 = max(8, int(W * sc))

                # Downsample ALL decoded frames, then select t=0 and t=1.
                frames_s = F.interpolate(
                    frames.permute(0, 2, 1, 3, 4).reshape(B2 * T, 3, H, W),
                    size=(h2, w2),
                    mode="bilinear",
                    align_corners=False
                ).reshape(B2, T, 3, h2, w2)

                # Match "regular" guidance behavior: compute loss over ALL adjacent decoded frames
                # within this decoded chunk.
                pairs = [(j, j + 1) for j in range(T - 1)]
                E_sum = 0.0
                for a, b in pairs:
                    Ia = frames_s[:, a]  # (B,3,h2,w2)
                    Ib = frames_s[:, b]

                    # flow on CPU (stop-grad)
                    Ia_u8 = (Ia[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    Ib_u8 = (Ib[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    flow = flow_farneback_np(Ia_u8, Ib_u8, scale=1.0)

                    Ea = sobel_mag_torch(Ia)  # (B,1,h2,w2)
                    Eb = sobel_mag_torch(Ib)

                    Ea_n = Ea / (Ea.mean(dim=(2, 3), keepdim=True) + 1e-6)
                    Eb_n = Eb / (Eb.mean(dim=(2, 3), keepdim=True) + 1e-6)

                    Eb_w = warp_torch(Eb_n, flow)
                    diff = Ea_n - Eb_w
                    E_sum = E_sum + torch.sqrt(diff * diff + 1e-4).mean()

                # average over adjacent pairs (scale-invariant to T)
                E = E_sum / max(1, len(pairs))
                E_total += float(E.detach().cpu())

                # per-pair gradient (immediate backward; no giant graph)
                g_pair = torch.autograd.grad(E, x_pair_fp32, retain_graph=False, create_graph=False)[0].detach()

                # accumulate into full gradient tensor
                g_total_full[:, i:i+2] += g_pair

                # aggressively free intermediates
                del x_pair_fp32, x_pair, frames, frames_s, E_sum, E, g_pair

                torch.cuda.empty_cache()

        # global trust-region step (single lambda for the whole accumulated gradient)
        g_norm = float(rms(g_total_full))
        lam = float(rho_t * delta_norm / (g_norm + 1e-8))

        lat_new = (lat_fp32 - lam * g_total_full).to(lat_old.dtype)
        callback_kwargs["latents"] = lat_new

        # debug stats
        max_abs_update = float((lat_new.float() - lat_old.float()).abs().max().item())
        num_changed = int((lat_new != lat_old).sum().item())

        trace.setdefault("energy_decode", []).append(float(E_total))
        trace.setdefault("lambda_decode", []).append(float(lam))
        trace.setdefault("grad_rms_decode", []).append(float(g_norm))
        trace.setdefault("max_abs_update", []).append(float(max_abs_update))
        trace.setdefault("num_changed", []).append(int(num_changed))
        trace.setdefault("num_pairs_full", []).append(int(pack_T - 1))

        # snapshots (always)
        if step_index in snapshot_steps:
            snapshots[str(step_index)] = callback_kwargs["latents"].detach().to("cpu", dtype=torch.float16)

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

    # Metrics (CPU)
    flow_raw = flow_edge_metric(video_generate, flow_scale=0.25, normalize=False)
    flow_norm = flow_edge_metric(video_generate, flow_scale=0.25, normalize=True)
    flicker = temporal_edge_flicker_cpu(video_generate, scale=0.25)

    logging.info(f"[metric] flow_edge_raw={flow_raw:.6f} flow_edge_norm={flow_norm:.6f} edge_flicker={flicker:.6f}")

    # Save trace + snapshots
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    trace_path = os.path.splitext(output_path)[0] + "_trace.json"
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)
    snap_path = os.path.splitext(output_path)[0] + "_latentsnap.pt"
    torch.save(snapshots, snap_path)

    logging.info(f"Saved trace to: {trace_path}")
    logging.info(f"Saved snapshots to: {snap_path}")

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