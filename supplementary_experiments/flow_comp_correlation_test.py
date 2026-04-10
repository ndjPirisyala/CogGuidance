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
class WindowMetrics:
    step_index: int
    timestep: int
    window_start: int
    latent_window_k: int
    decoded_frames_in_window: int
    projection: str
    cos_sim: float
    cos_sim_w: float
    epe_raw: float
    epe_aligned: float
    mag_corr: float
    support_frac: float
    rgb_mag_mean: float
    lat_mag_mean: float


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
    if frames.shape[1] == 3:
        return frames.contiguous()  # (B,C,T,H,W)
    if frames.shape[2] == 3:
        return frames.permute(0, 2, 1, 3, 4).contiguous()  # (B,T,C,H,W) -> (B,C,T,H,W)
    if frames.shape[-1] == 3:
        return frames.permute(0, 4, 1, 2, 3).contiguous()  # (B,T,H,W,C) -> (B,C,T,H,W)
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


def choose_window_starts(pack_t: int, window_k: int, num_windows: int) -> List[int]:
    max_start = max(0, pack_t - window_k)
    if max_start == 0:
        return [0]
    if num_windows <= 1:
        return [max_start // 2]
    xs = np.linspace(0, max_start, num=num_windows)
    return sorted({int(round(x)) for x in xs})


# -------------------------
# Image / latent preprocessing
# -------------------------
def rgb_to_gray_float(x_chw: torch.Tensor) -> np.ndarray:
    x = x_chw.detach().float().cpu().numpy()
    return (0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]).astype(np.float32)


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


def latent_window_to_gray_float(
    lat_tchw: torch.Tensor,
    out_hw: Tuple[int, int],
    projection: str = "pca",
) -> List[np.ndarray]:
    # lat_tchw: (T,C,H,W)
    t, c, h, w = lat_tchw.shape
    x = F.interpolate(lat_tchw.float(), size=out_hw, mode="bilinear", align_corners=False)  # (T,C,oh,ow)

    if projection == "rms":
        return [torch.sqrt(torch.mean(xi * xi, dim=0) + 1e-8).cpu().numpy().astype(np.float32) for xi in x]

    # Window-shared PCA: one basis for all latent slices in this window.
    feats = x.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()  # (T*oh*ow, C)
    mu = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats_n = (feats - mu) / std
    _, _, vh = np.linalg.svd(feats_n, full_matrices=False)
    pc = vh[0]
    proj = (feats_n @ pc).reshape(t, out_hw[0], out_hw[1])
    return [proj[i].astype(np.float32) for i in range(t)]


# -------------------------
# Flow + composition
# -------------------------
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


def _sample_flow_at_coords(flow: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    fx = cv2.remap(flow[..., 0], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fy = cv2.remap(flow[..., 1], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.stack([fx, fy], axis=-1).astype(np.float32)


def _in_bounds(map_x: np.ndarray, map_y: np.ndarray, h: int, w: int) -> np.ndarray:
    return (map_x >= 0.0) & (map_x <= (w - 1)) & (map_y >= 0.0) & (map_y <= (h - 1))


def compose_forward_flows(flows: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if len(flows) == 0:
        raise ValueError("Need at least one flow to compose")
    h, w, _ = flows[0].shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    total = np.zeros((h, w, 2), dtype=np.float32)
    valid = np.ones((h, w), dtype=bool)

    for flow in flows:
        if flow.shape[:2] != (h, w):
            raise ValueError("All flows must share the same spatial size before composition")
        map_x = xx + total[..., 0]
        map_y = yy + total[..., 1]
        sampled = _sample_flow_at_coords(flow, map_x, map_y)
        valid &= _in_bounds(map_x, map_y, h, w)
        total = total + sampled

    # Final endpoint should also remain in-bounds for the composed flow to be meaningful.
    final_x = xx + total[..., 0]
    final_y = yy + total[..., 1]
    valid &= _in_bounds(final_x, final_y, h, w)
    return total, valid


def compute_adjacent_flows(gray_frames: Sequence[np.ndarray], blur_sigma: float = 0.0) -> List[np.ndarray]:
    if len(gray_frames) < 2:
        raise ValueError("Need at least 2 frames to compute adjacent flows")
    flows = []
    for i in range(len(gray_frames) - 1):
        a_u8, b_u8 = pair_shared_to_u8(gray_frames[i], gray_frames[i + 1], blur_sigma=blur_sigma)
        flows.append(flow_farneback_np(a_u8, b_u8))
    return flows


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


def flow_metrics(
    flow_rgb: np.ndarray,
    flow_lat: np.ndarray,
    valid_rgb: np.ndarray,
    valid_lat: np.ndarray,
    mag_thresh: float = 1.0,
) -> Tuple[float, float, float, float, float, float, float, float]:
    mag_rgb = np.linalg.norm(flow_rgb, axis=-1)
    mag_lat = np.linalg.norm(flow_lat, axis=-1)
    mask = valid_rgb & valid_lat & (mag_rgb > mag_thresh) & (mag_lat > mag_thresh)
    support = float(mask.mean())
    if not np.any(mask):
        return 0.0, 0.0, float("nan"), float("nan"), float("nan"), support, float(mag_rgb.mean()), float(mag_lat.mean())

    a = flow_rgb[mask]
    b = flow_lat[mask]
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    cos = ((a * b).sum(axis=-1) / (na * nb + 1e-8)).astype(np.float32)
    w = np.minimum(na, nb)
    cos_w = float((cos * w).sum() / (w.sum() + 1e-8))
    epe_raw = float(np.linalg.norm(a - b, axis=-1).mean())
    epe_al = aligned_epe(flow_rgb, flow_lat, mask)
    mc = magnitude_corr(flow_rgb, flow_lat, mask)
    return float(cos.mean()), cos_w, epe_raw, epe_al, mc, support, float(mag_rgb.mean()), float(mag_lat.mean())


# -------------------------
# Main experiment
# -------------------------
def build_pipeline(model_path: str, dtype: torch.dtype, generate_type: str, image_or_video_path: str):
    image = None
    video = None
    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe, image, video


def summarize_rows(rows: List[WindowMetrics]) -> dict:
    if not rows:
        return {}
    out = {
        "overall": {
            "n": len(rows),
            "mean_cos": float(np.nanmean([r.cos_sim for r in rows])),
            "mean_cos_w": float(np.nanmean([r.cos_sim_w for r in rows])),
            "mean_epe_raw": float(np.nanmean([r.epe_raw for r in rows])),
            "mean_epe_aligned": float(np.nanmean([r.epe_aligned for r in rows])),
            "mean_mag_corr": float(np.nanmean([r.mag_corr for r in rows])),
            "mean_support_frac": float(np.nanmean([r.support_frac for r in rows])),
        },
        "by_step": {},
    }
    for s in sorted({r.step_index for r in rows}):
        rs = [r for r in rows if r.step_index == s]
        out["by_step"][str(s)] = {
            "n": len(rs),
            "mean_cos": float(np.nanmean([r.cos_sim for r in rs])),
            "mean_cos_w": float(np.nanmean([r.cos_sim_w for r in rs])),
            "mean_epe_raw": float(np.nanmean([r.epe_raw for r in rs])),
            "mean_epe_aligned": float(np.nanmean([r.epe_aligned for r in rs])),
            "mean_mag_corr": float(np.nanmean([r.mag_corr for r in rs])),
            "mean_support_frac": float(np.nanmean([r.support_frac for r in rs])),
        }
    return out


def run_experiment(
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
    corr_steps: Optional[List[int]] = None,
    corr_num_windows: int = 3,
    corr_window_k: int = 3,
    corr_long_side: int = 192,
    corr_blur_sigma: float = 1.0,
    corr_mag_thresh: float = 1.0,
    corr_projection: str = "pca",
):
    seed_everything(seed)
    pipe, image, video = build_pipeline(model_path, dtype, generate_type, image_or_video_path)
    N = num_inference_steps
    corr_steps = corr_steps or [max(0, N - 5), max(0, N - 2), N - 1]
    snapshot_rows: List[WindowMetrics] = []

    def cb_on_step_end(pipe, step_index, timestep, callback_kwargs):
        if step_index not in corr_steps:
            return callback_kwargs
        with torch.no_grad():
            x_full = callback_kwargs["latents"].detach()
            pack_t = x_full.shape[1]
            if pack_t < corr_window_k:
                logging.warning("pack_t=%d < corr_window_k=%d; skipping step %d", pack_t, corr_window_k, step_index)
                return callback_kwargs
            for ws in choose_window_starts(pack_t, corr_window_k, corr_num_windows):
                x_win = x_full[:, ws : ws + corr_window_k]  # (B,K,C,H,W)
                frames = pipe.decode_latents(x_win)
                frames_bcthw = canonicalize_decoded_frames(frames)
                _, _, dec_t, h, w = frames_bcthw.shape
                out_hw = long_side_hw(h, w, corr_long_side)
                frames_s = resize_video_frames(frames_bcthw, out_hw)  # (B,T,3,oh,ow)

                rgb_gray = [rgb_to_gray_float(frames_s[0, i]) for i in range(frames_s.shape[1])]
                lat_gray = latent_window_to_gray_float(x_win[0], out_hw, projection=corr_projection)

                rgb_adj = compute_adjacent_flows(rgb_gray, blur_sigma=corr_blur_sigma)
                lat_adj = compute_adjacent_flows(lat_gray, blur_sigma=corr_blur_sigma)
                flow_rgb, valid_rgb = compose_forward_flows(rgb_adj)
                flow_lat, valid_lat = compose_forward_flows(lat_adj)

                cos, cos_w, epe_raw, epe_al, mag_corr, support, rgb_mag_mean, lat_mag_mean = flow_metrics(
                    flow_rgb, flow_lat, valid_rgb, valid_lat, mag_thresh=corr_mag_thresh
                )
                row = WindowMetrics(
                    step_index=int(step_index),
                    timestep=int(timestep),
                    window_start=int(ws),
                    latent_window_k=int(corr_window_k),
                    decoded_frames_in_window=int(dec_t),
                    projection=str(corr_projection),
                    cos_sim=float(cos),
                    cos_sim_w=float(cos_w),
                    epe_raw=float(epe_raw),
                    epe_aligned=float(epe_al),
                    mag_corr=float(mag_corr),
                    support_frac=float(support),
                    rgb_mag_mean=float(rgb_mag_mean),
                    lat_mag_mean=float(lat_mag_mean),
                )
                snapshot_rows.append(row)
                logging.info(
                    "[flow-compose] step=%d t=%s ws=%d decT=%d cos=%.4f cos_w=%.4f epe_al=%.4f support=%.4f",
                    step_index, str(int(timestep)), ws, dec_t, cos, cos_w, epe_al, support,
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

    ensure_dir(output_path)
    video_generate = pipe(**common_kwargs).frames[0]
    export_to_video(video_generate, output_path, fps=fps)

    report = {
        "config": {
            "prompt": prompt,
            "model_path": model_path,
            "generate_type": generate_type,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "corr_steps": corr_steps,
            "corr_num_windows": corr_num_windows,
            "corr_window_k": corr_window_k,
            "corr_long_side": corr_long_side,
            "corr_blur_sigma": corr_blur_sigma,
            "corr_mag_thresh": corr_mag_thresh,
            "corr_projection": corr_projection,
        },
        "summary": summarize_rows(snapshot_rows),
        "rows": [asdict(r) for r in snapshot_rows],
    }
    json_path = os.path.splitext(output_path)[0] + "_flowcompose.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logging.info("Saved report to %s", json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVideoX composed-flow comparison between decoded RGB and latent projections")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_or_video_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-2b")
    parser.add_argument("--output_path", type=str, default="./output.mp4")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)
    parser.add_argument("--generate_type", type=str, default="t2v", choices=["t2v", "i2v", "v2v"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--corr_steps", type=str, default="40,45,48,49")
    parser.add_argument("--corr_num_windows", type=int, default=3)
    parser.add_argument("--corr_window_k", type=int, default=3)
    parser.add_argument("--corr_long_side", type=int, default=192)
    parser.add_argument("--corr_blur_sigma", type=float, default=1.0)
    parser.add_argument("--corr_mag_thresh", type=float, default=1.0)
    parser.add_argument("--corr_projection", type=str, default="pca", choices=["pca", "rms"])

    args = parser.parse_args()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

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
        dtype=dtype_map[args.dtype],
        seed=args.seed,
        fps=args.fps,
        width=args.width,
        height=args.height,
        corr_steps=parse_int_list(args.corr_steps),
        corr_num_windows=args.corr_num_windows,
        corr_window_k=args.corr_window_k,
        corr_long_side=args.corr_long_side,
        corr_blur_sigma=args.corr_blur_sigma,
        corr_mag_thresh=args.corr_mag_thresh,
        corr_projection=args.corr_projection,
    )
