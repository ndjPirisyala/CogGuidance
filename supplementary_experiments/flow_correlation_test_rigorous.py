import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
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


@dataclass
class PairMetrics:
    step_index: int
    timestep: int
    window_start: int
    latent_pair_start: int
    latent_pair_local: int
    latent_window_k: int
    decoded_frames_in_window: int
    rgb_frame_a: int
    rgb_frame_b: int
    cos_sim: float
    cos_sim_w: float
    epe_aligned: float
    mag_corr: float
    support_frac: float
    rgb_mag_mean: float
    lat_mag_mean: float
    projection: str


# -------------------------
# Basic helpers
# -------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def canonicalize_decoded_frames(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 5:
        raise ValueError(f"Expected 5D decoded frames, got shape={tuple(frames.shape)}")

    if frames.shape[1] == 3:      # B, C, T, H, W
        return frames
    if frames.shape[2] == 3:      # B, T, C, H, W
        return frames.permute(0, 2, 1, 3, 4).contiguous()
    if frames.shape[-1] == 3:     # B, T, H, W, C
        return frames.permute(0, 4, 1, 2, 3).contiguous()

    raise ValueError(f"Could not infer RGB channel dimension from decoded frame shape={tuple(frames.shape)}")


def resize_video_frames(frames: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    frames_bcthw = canonicalize_decoded_frames(frames)
    b, c, t, h, w = frames_bcthw.shape
    oh, ow = out_hw
    x = frames_bcthw.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=False)
    return x.reshape(b, t, c, oh, ow)


def long_side_hw(h: int, w: int, long_side: int) -> Tuple[int, int]:
    scale = float(long_side) / float(max(h, w))
    return max(8, int(round(h * scale))), max(8, int(round(w * scale)))


def choose_pair_starts(pack_t: int, num_pairs: int) -> List[int]:
    max_start = max(0, pack_t - 2)
    if max_start == 0:
        return [0]
    if num_pairs <= 1:
        return [max_start // 2]
    xs = np.linspace(0, max_start, num=num_pairs)
    starts = sorted({int(round(x)) for x in xs})
    return starts


def choose_window_start_for_pair(pair_start: int, pack_t: int, window_k: int) -> int:
    # Prefer the pair near the middle of the odd-sized window.
    local_pair = max(0, (window_k - 2) // 2)
    ws = pair_start - local_pair
    ws = max(0, min(pack_t - window_k, ws))
    return int(ws)


def representative_frame_indices(decoded_t: int, latent_k: int) -> List[int]:
    if latent_k < 2:
        raise ValueError("latent_k must be >= 2")
    if decoded_t < 2:
        raise ValueError("decoded_t must be >= 2")
    reps = np.linspace(0, decoded_t - 1, num=latent_k)
    reps = [int(round(x)) for x in reps]
    reps[0] = 0
    reps[-1] = decoded_t - 1
    for i in range(1, len(reps)):
        if reps[i] <= reps[i - 1]:
            reps[i] = min(decoded_t - 1, reps[i - 1] + 1)
    if reps[-1] != decoded_t - 1:
        reps[-1] = decoded_t - 1
    return reps


def flow_farneback_np(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        a_u8,
        b_u8,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


def pair_shared_to_u8(a: np.ndarray, b: np.ndarray, blur_sigma: float = 0.0, q_lo: float = 1.0, q_hi: float = 99.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([a.reshape(-1), b.reshape(-1)], axis=0).astype(np.float32)
    lo = float(np.percentile(x, q_lo))
    hi = float(np.percentile(x, q_hi))
    if hi <= lo:
        hi = lo + 1e-6

    def _map(z: np.ndarray) -> np.ndarray:
        z = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
        if blur_sigma > 0:
            z = cv2.GaussianBlur(z.astype(np.float32), ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        return np.clip(np.round(z * 255.0), 0, 255).astype(np.uint8)

    return _map(a), _map(b)


def rgb_pair_to_gray_u8(rgb_a_chw: torch.Tensor, rgb_b_chw: torch.Tensor, blur_sigma: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    a = rgb_a_chw.detach().float().cpu().numpy()
    b = rgb_b_chw.detach().float().cpu().numpy()
    ga = 0.2989 * a[0] + 0.5870 * a[1] + 0.1140 * a[2]
    gb = 0.2989 * b[0] + 0.5870 * b[1] + 0.1140 * b[2]
    return pair_shared_to_u8(ga, gb, blur_sigma=blur_sigma)


def latent_pair_to_gray_u8(
    lat_a_chw: torch.Tensor,
    lat_b_chw: torch.Tensor,
    out_hw: Tuple[int, int],
    projection: str = "pca",
    blur_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    a = lat_a_chw.detach().float().cpu()
    b = lat_b_chw.detach().float().cpu()
    c, h, w = a.shape

    a2 = F.interpolate(a.unsqueeze(0), size=out_hw, mode="bilinear", align_corners=False)[0]
    b2 = F.interpolate(b.unsqueeze(0), size=out_hw, mode="bilinear", align_corners=False)[0]

    if projection == "rms":
        ga = torch.sqrt(torch.mean(a2 * a2, dim=0) + 1e-8).numpy()
        gb = torch.sqrt(torch.mean(b2 * b2, dim=0) + 1e-8).numpy()
        return pair_shared_to_u8(ga, gb, blur_sigma=blur_sigma)

    xa = a2.reshape(c, -1).t().numpy()
    xb = b2.reshape(c, -1).t().numpy()
    x = np.concatenate([xa, xb], axis=0)
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    xa = (xa - mu) / std
    xb = (xb - mu) / std
    x = np.concatenate([xa, xb], axis=0)

    _, _, vh = np.linalg.svd(x, full_matrices=False)
    pc = vh[0]
    ga = (xa @ pc).reshape(out_hw)
    gb = (xb @ pc).reshape(out_hw)
    return pair_shared_to_u8(ga, gb, blur_sigma=blur_sigma)


def aligned_epe(flow_ref: np.ndarray, flow_test: np.ndarray, mask: np.ndarray) -> float:
    a = flow_ref[mask]
    b = flow_test[mask]
    if len(a) == 0:
        return float("nan")
    denom = float((b * b).sum()) + 1e-8
    scale = float((a * b).sum()) / denom
    err = np.linalg.norm(a - scale * b, axis=-1)
    return float(err.mean())


def magnitude_corr(flow_a: np.ndarray, flow_b: np.ndarray, mask: np.ndarray) -> float:
    ma = np.linalg.norm(flow_a, axis=-1)[mask]
    mb = np.linalg.norm(flow_b, axis=-1)[mask]
    if ma.size < 2:
        return float("nan")
    sa = float(ma.std())
    sb = float(mb.std())
    if sa < 1e-8 or sb < 1e-8:
        return float("nan")
    return float(np.corrcoef(ma, mb)[0, 1])


def flow_metrics(flow_rgb: np.ndarray, flow_lat: np.ndarray, mag_thresh: float = 0.5) -> Tuple[float, float, float, float, float, float, float]:
    mag_rgb = np.linalg.norm(flow_rgb, axis=-1)
    mag_lat = np.linalg.norm(flow_lat, axis=-1)
    mask = (mag_rgb > mag_thresh) & (mag_lat > mag_thresh)
    support = float(mask.mean())
    if not np.any(mask):
        return 0.0, 0.0, float("nan"), float("nan"), support, float(mag_rgb.mean()), float(mag_lat.mean())

    a = flow_rgb[mask]
    b = flow_lat[mask]
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    cos = ((a * b).sum(axis=-1) / denom).astype(np.float32)
    w = np.minimum(np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1))
    cos_w = float((cos * w).sum() / (w.sum() + 1e-8))
    epe = aligned_epe(flow_rgb, flow_lat, mask)
    mc = magnitude_corr(flow_rgb, flow_lat, mask)
    return float(cos.mean()), cos_w, epe, mc, support, float(mag_rgb.mean()), float(mag_lat.mean())


# -------------------------
# Main experiment
# -------------------------
def build_pipeline(model_path: str, dtype: torch.dtype, generate_type: str, image_or_video_path: str, width: Optional[int], height: Optional[int]):
    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP.get(model_name)
    if desired_resolution is not None:
        if width is None or height is None:
            height, width = desired_resolution
            logging.info("Using default resolution %s for %s", desired_resolution, model_name)
        elif (height, width) != desired_resolution and generate_type != "i2v":
            logging.warning("%s not supported for custom resolution; using %s", model_name, desired_resolution)
            height, width = desired_resolution
    else:
        height = height or 480
        width = width or 720

    ngpu = torch.cuda.device_count()
    max_memory = None
    device_map = None
    if ngpu >= 2:
        max_memory = {i: "60GiB" for i in range(ngpu)}
        device_map = "balanced"

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

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    logging.info("Scheduler: %s", pipe.scheduler.__class__.__name__)
    logging.info("Temporal compression ratio: %s", getattr(pipe, "vae_scale_factor_temporal", "unknown"))

    if device_map is None:
        pipe.to("cuda")

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe, image, video, width, height


def run_experiment(
    prompt: str,
    model_path: str,
    output_path: str,
    image_or_video_path: str = "",
    generate_type: str = "t2v",
    num_inference_steps: int = 100,
    num_frames: int = 81,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 16,
    width: Optional[int] = None,
    height: Optional[int] = None,
    corr_steps: Sequence[int] = (40, 45, 48, 49),
    corr_long_side: int = 256,
    corr_num_pairs: int = 3,
    corr_projection: str = "pca",
    corr_blur_sigma: float = 1.0,
    corr_mag_thresh: float = 0.5,
    corr_window_k: int = 3,
    save_video: bool = True,
) -> None:
    seed_everything(seed)

    if corr_window_k < 2:
        raise ValueError("corr_window_k must be >= 2")
    if corr_window_k % 2 == 0:
        logging.warning("corr_window_k=%d is even; odd values like 3 or 5 are more stable for CogVideoX temporal decoding.", corr_window_k)

    pipe, image, video, width, height = build_pipeline(
        model_path=model_path,
        dtype=dtype,
        generate_type=generate_type,
        image_or_video_path=image_or_video_path,
        width=width,
        height=height,
    )

    metrics: List[PairMetrics] = []
    corr_steps = set(int(s) for s in corr_steps)

    def cb_on_step_end(pipe, step_index, timestep, callback_kwargs):
        if step_index not in corr_steps:
            return callback_kwargs

        with torch.no_grad():
            x_full = callback_kwargs["latents"].detach()
            pack_t = x_full.shape[1]
            pair_starts = choose_pair_starts(pack_t=pack_t, num_pairs=corr_num_pairs)

            for pair_start in pair_starts:
                window_start = choose_window_start_for_pair(pair_start, pack_t=pack_t, window_k=corr_window_k)
                window_end = min(pack_t, window_start + corr_window_k)
                x_win = x_full[:, window_start:window_end]
                latent_k = int(x_win.shape[1])
                if latent_k < 2:
                    continue

                frames = pipe.decode_latents(x_win)
                frames_bcthw = canonicalize_decoded_frames(frames)
                _, _, decoded_t, h, w = frames_bcthw.shape
                out_hw = long_side_hw(h, w, corr_long_side)
                frames_s = resize_video_frames(frames_bcthw, out_hw)

                rep_idx = representative_frame_indices(decoded_t=decoded_t, latent_k=latent_k)
                local_pair = int(np.clip(pair_start - window_start, 0, latent_k - 2))
                rgb_a_idx = rep_idx[local_pair]
                rgb_b_idx = rep_idx[local_pair + 1]

                rgb_a = frames_s[0, rgb_a_idx]
                rgb_b = frames_s[0, rgb_b_idx]
                lat_a = x_win[0, local_pair]
                lat_b = x_win[0, local_pair + 1]

                rgb_a_u8, rgb_b_u8 = rgb_pair_to_gray_u8(rgb_a, rgb_b, blur_sigma=corr_blur_sigma)
                lat_a_u8, lat_b_u8 = latent_pair_to_gray_u8(
                    lat_a,
                    lat_b,
                    out_hw=out_hw,
                    projection=corr_projection,
                    blur_sigma=corr_blur_sigma,
                )

                flow_rgb = flow_farneback_np(rgb_a_u8, rgb_b_u8)
                flow_lat = flow_farneback_np(lat_a_u8, lat_b_u8)
                cos_sim, cos_sim_w, epe, mag_corr_val, support, rgb_mag_mean, lat_mag_mean = flow_metrics(
                    flow_rgb,
                    flow_lat,
                    mag_thresh=corr_mag_thresh,
                )
                row = PairMetrics(
                    step_index=int(step_index),
                    timestep=int(timestep),
                    window_start=int(window_start),
                    latent_pair_start=int(pair_start),
                    latent_pair_local=int(local_pair),
                    latent_window_k=int(latent_k),
                    decoded_frames_in_window=int(decoded_t),
                    rgb_frame_a=int(rgb_a_idx),
                    rgb_frame_b=int(rgb_b_idx),
                    cos_sim=cos_sim,
                    cos_sim_w=cos_sim_w,
                    epe_aligned=epe,
                    mag_corr=mag_corr_val,
                    support_frac=support,
                    rgb_mag_mean=rgb_mag_mean,
                    lat_mag_mean=lat_mag_mean,
                    projection=corr_projection,
                )
                metrics.append(row)
                logging.info(
                    "[flow-corr] step=%d t=%s pair_start=%d window=[%d:%d] decoded_t=%d rgb_pair=(%d,%d) cos=%.4f cos_w=%.4f epe=%.4f support=%.4f",
                    step_index,
                    int(timestep),
                    pair_start,
                    window_start,
                    window_end,
                    decoded_t,
                    rgb_a_idx,
                    rgb_b_idx,
                    cos_sim,
                    cos_sim_w,
                    epe if np.isfinite(epe) else float("nan"),
                    support,
                )

        return callback_kwargs

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

    rows = [asdict(m) for m in metrics]
    summary = {
        "n": len(rows),
        "mean_cos": float(np.nanmean([r["cos_sim"] for r in rows])) if rows else float("nan"),
        "mean_cos_w": float(np.nanmean([r["cos_sim_w"] for r in rows])) if rows else float("nan"),
        "mean_epe_aligned": float(np.nanmean([r["epe_aligned"] for r in rows])) if rows else float("nan"),
        "mean_mag_corr": float(np.nanmean([r["mag_corr"] for r in rows])) if rows else float("nan"),
        "mean_support_frac": float(np.nanmean([r["support_frac"] for r in rows])) if rows else float("nan"),
    }
    by_step = {}
    for s in sorted({r["step_index"] for r in rows}):
        rs = [r for r in rows if r["step_index"] == s]
        by_step[str(s)] = {
            "n": len(rs),
            "mean_cos": float(np.nanmean([r["cos_sim"] for r in rs])),
            "mean_cos_w": float(np.nanmean([r["cos_sim_w"] for r in rs])),
            "mean_epe_aligned": float(np.nanmean([r["epe_aligned"] for r in rs])),
            "mean_mag_corr": float(np.nanmean([r["mag_corr"] for r in rs])),
            "mean_support_frac": float(np.nanmean([r["support_frac"] for r in rs])),
        }

    report = {
        "prompt": prompt,
        "model_path": model_path,
        "seed": seed,
        "generate_type": generate_type,
        "num_inference_steps": num_inference_steps,
        "num_frames": num_frames,
        "guidance_scale": guidance_scale,
        "corr_steps": sorted(corr_steps),
        "corr_long_side": corr_long_side,
        "corr_num_pairs": corr_num_pairs,
        "corr_projection": corr_projection,
        "corr_blur_sigma": corr_blur_sigma,
        "corr_mag_thresh": corr_mag_thresh,
        "corr_window_k": corr_window_k,
        "summary": summary,
        "by_step": by_step,
        "rows": rows,
    }

    ensure_dir(output_path)
    stem, _ = os.path.splitext(output_path)
    json_path = stem + "_flowcorr.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Saved flow correlation report to %s", json_path)

    if save_video:
        export_to_video(video_generate, output_path, fps=fps)
        logging.info("Saved video to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--image_or_video_path", type=str, default="")
    parser.add_argument("--generate_type", type=str, default="t2v", choices=["t2v", "i2v", "v2v"])
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--corr_steps", type=str, default="40,45,48,49")
    parser.add_argument("--corr_long_side", type=int, default=256)
    parser.add_argument("--corr_num_pairs", type=int, default=3)
    parser.add_argument("--corr_projection", type=str, default="pca", choices=["pca", "rms"])
    parser.add_argument("--corr_blur_sigma", type=float, default=1.0)
    parser.add_argument("--corr_mag_thresh", type=float, default=0.5)
    parser.add_argument("--corr_window_k", type=int, default=3)
    parser.add_argument("--no_save_video", action="store_true")
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    run_experiment(
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
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
        corr_steps=parse_int_list(args.corr_steps),
        corr_long_side=args.corr_long_side,
        corr_num_pairs=args.corr_num_pairs,
        corr_projection=args.corr_projection,
        corr_blur_sigma=args.corr_blur_sigma,
        corr_mag_thresh=args.corr_mag_thresh,
        corr_window_k=args.corr_window_k,
        save_video=not args.no_save_video,
    )
