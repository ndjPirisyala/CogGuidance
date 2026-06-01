import cv2
import numpy as np

def read_video_cv2(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from: {path}")
    return np.stack(frames, axis=0).astype(np.uint8)  # (T,H,W,3)

def sobel_mag_gray_u8(frame_u8_rgb):
    g = cv2.cvtColor(frame_u8_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy + 1e-6)

def edge_stats(video_u8):
    T = video_u8.shape[0]
    edges = np.stack([sobel_mag_gray_u8(video_u8[t]) for t in range(T)], axis=0)  # (T,H,W)

    mean_edge_mag = float(edges.mean())
    edge_mag_std_over_time = float(edges.std(axis=0).mean())
    edge_flicker = float(np.abs(edges[1:] - edges[:-1]).mean())

    return {
        "mean_edge_mag": mean_edge_mag,
        "edge_mag_std_over_time": edge_mag_std_over_time,
        "edge_flicker": edge_flicker,
        "T": int(T),
        "H": int(video_u8.shape[1]),
        "W": int(video_u8.shape[2]),
    }

if __name__ == "__main__":
    base_path="/home/nvidia/CogVideoX/CogGuidance/inference/OUTDIR_SCALED/video_15_fixed_0_frames.mp4"
    guided_path="/home/nvidia/CogVideoX/CogGuidance/inference/OUTDIR_SCALED/video_19_all_frames.mp4"

    base_vid = read_video_cv2(base_path)
    guided_vid = read_video_cv2(guided_path)

    T = min(base_vid.shape[0], guided_vid.shape[0])
    base_vid = base_vid[:T]
    guided_vid = guided_vid[:T]

    base_stats = edge_stats(base_vid)
    guided_stats = edge_stats(guided_vid)

    print("BASE  :", base_stats)
    print("GUIDED:", guided_stats)
    print("\nInterpretation:")
    print("- If GUIDED mean_edge_mag << BASE mean_edge_mag: likely smoothing/blur.")
    print("- If GUIDED edge_mag_std_over_time and edge_flicker drop while mean_edge_mag stays similar: true temporal stabilization.")