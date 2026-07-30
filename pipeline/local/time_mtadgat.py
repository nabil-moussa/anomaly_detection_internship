import torch
import numpy as np
import time
import sys
from pathlib import Path

# ── point these to your actual files ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from mtad_gat import MTAD_GAT

MODEL_PATH  = r"path\to\your\downloaded\model.pt"
N_FEATURES  = 3
WINDOW_SIZE = 100
N_REPEATS   = 500
# ─────────────────────────────────────────────────────────

model = MTAD_GAT(
    n_features        = N_FEATURES,
    window_size       = WINDOW_SIZE,
    out_dim           = N_FEATURES,
    recon_out_dim     = N_FEATURES,
    gru_hid_dim       = 300,
    forecast_hid_dim  = 300,
    recon_hid_dim     = 300,
    use_gatv2         = False,
    use_vae           = False,
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# warm up
x = torch.randn(1, WINDOW_SIZE, N_FEATURES)
with torch.no_grad():
    _ = model(x)

# time inference
times = []
with torch.no_grad():
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _  = model(x)
        times.append(time.perf_counter() - t0)

mean_ms = np.mean(times) * 1000
std_ms  = np.std(times)  * 1000
print(f"MTAD-GAT CPU per-window inference: "
      f"{mean_ms:.3f} ± {std_ms:.3f} ms")
print(f"Over 11 test cycles × ~1904 windows each: "
      f"{mean_ms * 11 * 1904 / 1000:.1f}s total")

import tracemalloc
tracemalloc.start()

with torch.no_grad():
    for _ in range(100):
        _ = model(x)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"MTAD-GAT peak RAM (CPU inference): "
      f"{peak/1024**2:.1f} MB")