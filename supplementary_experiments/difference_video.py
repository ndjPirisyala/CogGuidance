import os
import numpy as np
from diffusers.utils import load_video, export_to_video

def _to_uint8(frames):
    """
    Accepts:
      - list of PIL images
      - numpy array (T,H,W,3) or (T,3,H,W)
      - torch tensor (T,H,W,3) or (T,3,H,W)
    Returns:
      - np.uint8 array (T,H,W,3)
    """
    if isinstance(frames, (list, tuple)):
        arr = np.stack([np.asarray(f) for f in frames], axis=0)
    else:
        arr = np.asarray(frames)

    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        # (T,3,H,W) -> (T,H,W,3)
        arr = np.transpose(arr, (0, 2, 3, 1))

    if arr.dtype != np.uint8:
        # assume float in [0,1] or [0,255]
        mx = float(arr.max()) if arr.size else 0.0
        if mx <= 1.5:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3) uint8, got {arr.shape} {arr.dtype}")
    return arr

def make_diff_video(base_mp4, guided_mp4, out_mp4, amplify=8.0, fps=8):
    """
    Creates a difference video:
      diff[t] = clip( amplify * abs(base[t] - guided[t]) , 0..255 )

    base_mp4 / guided_mp4: paths to .mp4
    out_mp4: output path
    amplify: 4, 8, 16 etc. Higher makes subtle differences visible.
    fps: output fps
    """
    base = load_video(base_mp4)     # usually list of PIL frames
    guided = load_video(guided_mp4)

    base_u8 = _to_uint8(base)
    guided_u8 = _to_uint8(guided)

    T = min(base_u8.shape[0], guided_u8.shape[0])
    if base_u8.shape != guided_u8.shape:
        # if shapes differ, crop to common T and common H/W via center crop
        H = min(base_u8.shape[1], guided_u8.shape[1])
        W = min(base_u8.shape[2], guided_u8.shape[2])

        def center_crop(x):
            t, h, w, c = x.shape
            y0 = (h - H) // 2
            x0 = (w - W) // 2
            return x[:T, y0:y0+H, x0:x0+W, :]

        base_u8 = center_crop(base_u8)
        guided_u8 = center_crop(guided_u8)
    else:
        base_u8 = base_u8[:T]
        guided_u8 = guided_u8[:T]

    diff = np.abs(base_u8.astype(np.int16) - guided_u8.astype(np.int16)).astype(np.float32)
    diff = np.clip(diff * float(amplify), 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    export_to_video(diff, out_mp4, fps=fps)
    print(f"Saved diff video to: {out_mp4} (amplify={amplify}, fps={fps}, frames={T}, shape={diff.shape[1:]}).")

if __name__ == "__main__":
    # Example usage:
    # make_diff_video("OUT/base.mp4", "OUT/guided.mp4", "OUT/diff_x8.mp4", amplify=8, fps=8)
    make_diff_video(
        base_mp4="/home/nvidia/CogVideoX/CogGuidance/inference/OUTDIR_SCALED/video_15_fixed_0_frames.mp4",
        guided_mp4="/home/nvidia/CogVideoX/CogGuidance/inference/OUTDIR_SCALED/video_19_all_frames.mp4",
        out_mp4="DIF_OUTDIR/diff_15_19.mp4",
        amplify=8.0,
        fps=8
    )