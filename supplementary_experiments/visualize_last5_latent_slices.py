import argparse
import json
import logging
import math
import os
import random
from typing import Optional, Tuple

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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def chw_float_to_rgb_u8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if x.shape[0] != 3:
        raise ValueError(f"Expected CHW with 1 or 3 channels, got shape={tuple(x.shape)}")

    xmin = float(x.min())
    xmax = float(x.max())
    if xmin >= 0.0 and xmax <= 1.5:
        x = x * 255.0
    elif xmin >= -1.5 and xmax <= 1.5:
        x = (x + 1.0) * 0.5 * 255.0
    else:
        x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 255.0

    x = x.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return x


def resize_long_side_np(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = float(long_side) / float(max(h, w))
    oh = max(8, int(round(h * scale)))
    ow = max(8, int(round(w * scale)))
    return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)


def shared_percentile_normalize_pair(a: np.ndarray, b: np.ndarray, q_lo: float = 1.0, q_hi: float = 99.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([a.reshape(-1), b.reshape(-1)], axis=0).astype(np.float32)
    lo = float(np.percentile(x, q_lo))
    hi = float(np.percentile(x, q_hi))
    if hi <= lo:
        hi = lo + 1e-6

    def _map(z: np.ndarray) -> np.ndarray:
        z = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
        return np.clip(np.round(z * 255.0), 0, 255).astype(np.uint8)

    return _map(a), _map(b)


def latent_pair_projections(lat_a: torch.Tensor, lat_b: torch.Tensor) -> dict:
    a = lat_a.detach().float().cpu().numpy()
    b = lat_b.detach().float().cpu().numpy()
    c, h, w = a.shape

    rms_a = np.sqrt(np.mean(a * a, axis=0) + 1e-8)
    rms_b = np.sqrt(np.mean(b * b, axis=0) + 1e-8)
    rms_a_u8, rms_b_u8 = shared_percentile_normalize_pair(rms_a, rms_b)

    xa = a.reshape(c, -1).T
    xb = b.reshape(c, -1).T
    x = np.concatenate([xa, xb], axis=0)
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    xa = (xa - mu) / std
    xb = (xb - mu) / std
    x = np.concatenate([xa, xb], axis=0)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    pc = vh[0]
    pca_a = (xa @ pc).reshape(h, w)
    pca_b = (xb @ pc).reshape(h, w)
    pca_a_u8, pca_b_u8 = shared_percentile_normalize_pair(pca_a, pca_b)

    return {
        "rms_a": rms_a_u8,
        "rms_b": rms_b_u8,
        "pca_a": pca_a_u8,
        "pca_b": pca_b_u8,
    }


def gray_u8_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def tile_images(images, ncols: int, pad: int = 8, bg: Tuple[int, int, int] = (18, 18, 18)) -> np.ndarray:
    if not images:
        raise ValueError("images must be non-empty")
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    imgs = []
    for img in images:
        if img.ndim == 2:
            img = gray_u8_to_bgr(img)
        canvas = np.full((h, w, 3), bg, dtype=np.uint8)
        y = (h - img.shape[0]) // 2
        x = (w - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        imgs.append(canvas)

    n = len(imgs)
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))
    out = np.full((nrows * h + pad * (nrows + 1), ncols * w + pad * (ncols + 1), 3), bg, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r = idx // ncols
        c = idx % ncols
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        out[y:y + h, x:x + w] = img
    return out


def save_image(path: str, rgb_or_bgr: np.ndarray, assume_rgb: bool = False) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if assume_rgb:
        cv2.imwrite(path, cv2.cvtColor(rgb_or_bgr, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, rgb_or_bgr)


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


def pick_pair_start(pack_t: int, requested_pair_start: int) -> int:
    if pack_t < 2:
        raise ValueError(f"Need at least 2 latent time slices, got pack_t={pack_t}")
    max_start = pack_t - 2
    if requested_pair_start >= 0:
        return max(0, min(requested_pair_start, max_start))
    return max(0, min(max_start, pack_t // 2 - 1))


def run_capture(
    prompt: str,
    model_path: str,
    output_dir: str,
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
    last_n_steps: int = 5,
    pair_start: int = -1,
    latent_viz_long_side: int = 512,
    decoded_frame_long_side: Optional[int] = None,
    save_raw_latents: bool = True,
    save_final_video: bool = True,
) -> None:
    seed_everything(seed)
    ensure_dir(output_dir)

    pipe, image, video, width, height = build_pipeline(
        model_path=model_path,
        dtype=dtype,
        generate_type=generate_type,
        image_or_video_path=image_or_video_path,
        width=width,
        height=height,
    )

    capture_steps = set(range(max(0, num_inference_steps - last_n_steps), num_inference_steps))
    manifest = {
        "prompt": prompt,
        "model_path": model_path,
        "seed": seed,
        "generate_type": generate_type,
        "num_inference_steps": num_inference_steps,
        "num_frames": num_frames,
        "guidance_scale": guidance_scale,
        "capture_steps": sorted(int(x) for x in capture_steps),
        "pair_start_arg": int(pair_start),
        "captures": [],
    }

    def cb_on_step_end(pipe, step_index, timestep, callback_kwargs):
        if step_index not in capture_steps:
            return callback_kwargs

        with torch.no_grad():
            x_full = callback_kwargs["latents"].detach()
            pack_t = int(x_full.shape[1])
            chosen_pair_start = pick_pair_start(pack_t, pair_start)
            x_pair = x_full[:, chosen_pair_start:chosen_pair_start + 2].contiguous()

            frames = pipe.decode_latents(x_pair)
            frames_bcthw = canonicalize_decoded_frames(frames)
            b, c, decoded_t, h, w = frames_bcthw.shape
            if b < 1:
                raise ValueError("Decoded batch is empty")

            step_dir = os.path.join(output_dir, f"step_{step_index:03d}_t_{int(timestep)}")
            frames_dir = os.path.join(step_dir, "decoded_frames")
            ensure_dir(frames_dir)

            lat_a = x_pair[0, 0]
            lat_b = x_pair[0, 1]
            proj = latent_pair_projections(lat_a, lat_b)

            pca_a = resize_long_side_np(gray_u8_to_bgr(proj["pca_a"]), latent_viz_long_side)
            pca_b = resize_long_side_np(gray_u8_to_bgr(proj["pca_b"]), latent_viz_long_side)
            rms_a = resize_long_side_np(gray_u8_to_bgr(proj["rms_a"]), latent_viz_long_side)
            rms_b = resize_long_side_np(gray_u8_to_bgr(proj["rms_b"]), latent_viz_long_side)

            latent_panel = tile_images([
                add_label(pca_a, f"latent slice {chosen_pair_start} | PCA"),
                add_label(pca_b, f"latent slice {chosen_pair_start + 1} | PCA"),
                add_label(rms_a, f"latent slice {chosen_pair_start} | RMS"),
                add_label(rms_b, f"latent slice {chosen_pair_start + 1} | RMS"),
            ], ncols=2)
            save_image(os.path.join(step_dir, "latent_pair_panel.png"), latent_panel, assume_rgb=False)

            decoded_rgb_frames = []
            decoded_tiles = []
            for i in range(decoded_t):
                frame_rgb = chw_float_to_rgb_u8(frames_bcthw[0, :, i])
                if decoded_frame_long_side is not None:
                    frame_rgb = resize_long_side_np(frame_rgb, decoded_frame_long_side)
                decoded_rgb_frames.append(frame_rgb)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                labeled = add_label(frame_bgr, f"decoded frame {i}")
                decoded_tiles.append(labeled)
                save_image(os.path.join(frames_dir, f"frame_{i:03d}.png"), frame_rgb, assume_rgb=True)

            decoded_grid = tile_images(decoded_tiles, ncols=min(4, max(1, decoded_t)), pad=8)
            save_image(os.path.join(step_dir, "decoded_frames_grid.png"), decoded_grid, assume_rgb=False)

            decoded_video_path = os.path.join(step_dir, "decoded_window.mp4")
            export_to_video(decoded_rgb_frames, decoded_video_path, fps=max(1, fps))

            meta = {
                "step_index": int(step_index),
                "timestep": int(timestep),
                "x_full_shape": list(x_full.shape),
                "x_pair_shape": list(x_pair.shape),
                "pair_start": int(chosen_pair_start),
                "decoded_shape_bcthw": list(frames_bcthw.shape),
                "decoded_num_frames": int(decoded_t),
                "latent_slice_a": int(chosen_pair_start),
                "latent_slice_b": int(chosen_pair_start + 1),
                "latent_panel_path": os.path.join(step_dir, "latent_pair_panel.png"),
                "decoded_grid_path": os.path.join(step_dir, "decoded_frames_grid.png"),
                "decoded_video_path": decoded_video_path,
            }
            with open(os.path.join(step_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            if save_raw_latents:
                torch.save(
                    {
                        "step_index": int(step_index),
                        "timestep": int(timestep),
                        "x_pair": x_pair.detach().cpu(),
                        "x_full_shape": tuple(int(v) for v in x_full.shape),
                    },
                    os.path.join(step_dir, "latent_pair.pt"),
                )

            manifest["captures"].append(meta)
            logging.info(
                "[capture] step=%d timestep=%d pair=(%d,%d) x_pair=%s decoded=%s saved_to=%s",
                int(step_index),
                int(timestep),
                int(chosen_pair_start),
                int(chosen_pair_start + 1),
                tuple(x_pair.shape),
                tuple(frames_bcthw.shape),
                step_dir,
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

    if save_final_video:
        final_video_path = os.path.join(output_dir, "final_video.mp4")
        export_to_video(video_generate, final_video_path, fps=fps)
        manifest["final_video_path"] = final_video_path
        logging.info("Saved final video to %s", final_video_path)

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Saved manifest to %s", os.path.join(output_dir, "manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture and visualize 2 latent time slices from the last denoising steps in CogVideoX.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_or_video_path", type=str, default="")
    parser.add_argument("--generate_type", type=str, default="t2v", choices=["t2v", "i2v", "v2v"])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--last_n_steps", type=int, default=5, help="How many final denoising steps to capture.")
    parser.add_argument("--pair_start", type=int, default=-1, help="Packed-time start index for the 2-slice latent pair. -1 means center pair.")
    parser.add_argument("--latent_viz_long_side", type=int, default=512)
    parser.add_argument("--decoded_frame_long_side", type=int, default=None, help="Optional resize for saved decoded RGB frames and grids.")
    parser.add_argument("--no_save_raw_latents", action="store_true")
    parser.add_argument("--no_save_final_video", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    run_capture(
        prompt=args.prompt,
        model_path=args.model_path,
        output_dir=args.output_dir,
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
        last_n_steps=args.last_n_steps,
        pair_start=args.pair_start,
        latent_viz_long_side=args.latent_viz_long_side,
        decoded_frame_long_side=args.decoded_frame_long_side,
        save_raw_latents=not args.no_save_raw_latents,
        save_final_video=not args.no_save_final_video,
    )
