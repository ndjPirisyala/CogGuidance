import torch
from pathlib import Path

A_path = "OUTDIR/video_23_latentsnap.pt"  # baseline
B_path = "OUTDIR/video_26_latentsnap.pt"  # guided

def stats(t):
    t = t.float()
    return {
        "shape": tuple(t.shape),
        "rms": float(t.pow(2).mean().sqrt().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }

def compare(a, b):
    da = (a.float() - b.float()).abs()
    return {
        "max_abs": float(da.max().item()),
        "mean_abs": float(da.mean().item()),
        "rms_abs": float(da.pow(2).mean().sqrt().item()),
    }

for p in [A_path, B_path]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Not found: {p}")

A = torch.load(A_path, map_location="cpu")
B = torch.load(B_path, map_location="cpu")

Akeys = sorted(A.keys(), key=lambda x: int(x))
Bkeys = sorted(B.keys(), key=lambda x: int(x))
common = [k for k in Akeys if k in B]

print("Baseline keys:", Akeys)
print("Guided   keys:", Bkeys)
print("Common keys  :", common)

for k in common:
    print(f"\n=== snapshot step {k} ===")
    sa = stats(A[k]); sb = stats(B[k])
    print("baseline:", sa)
    print("guided  :", sb)
    print("diff    :", compare(A[k], B[k]))

# quick overall checksum-ish signal
import hashlib
def sha256(path):
    return hashlib.sha256(open(path,'rb').read()).hexdigest()

print("\nSHA256 baseline:", sha256(A_path))
print("SHA256 guided  :", sha256(B_path))