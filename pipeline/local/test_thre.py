import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ── point these to your actual files ──────────────────────
CSV_PATH = r"D:\MaFaulDa\streams\stream_B1.csv"
GT_PATH  = r"D:\MaFaulDa\streams\stream_B1_gt.csv"
SENSOR_COLS = ["underhang_axial", "underhang_radial", 
               "underhang_tangential"]
TRAIN_FRAC = 0.75
LOOKBACK   = 100
# ──────────────────────────────────────────────────────────

df     = pd.read_csv(CSV_PATH)
data   = df[SENSOR_COLS].values.astype("float32")
split  = int(len(data) * TRAIN_FRAC)
x_tr   = data[:split]
x_te   = data[split:]

gt_df  = pd.read_csv(GT_PATH)
labels = np.zeros(len(data), dtype="float32")
for _, row in gt_df.iterrows():
    labels[int(row['start_row']):int(row['end_row'])] = 1.0

y_te = labels[split:][LOOKBACK:]

print("=== VERIFICATION ===")
print(f"Stream length:     {len(data)}")
print(f"Split at 75%:      {split}")
print(f"x_tr shape:        {x_tr.shape}")
print(f"x_te shape:        {x_te.shape}")
print(f"y_te shape:        {y_te.shape}")
print(f"Total anomalous:   {int(labels.sum())}")
print(f"Anomalous in test: {int(y_te.sum())}")
print(f"Anomaly fraction:  {y_te.sum()/len(y_te)*100:.1f}%")
print(f"\nGT file loaded:    YES ({len(gt_df)} fault intervals)")
for _, row in gt_df.iterrows():
    in_test = row['end_row'] > split
    print(f"  {row['fault_type']}: {int(row['start_row'])}-{int(row['end_row'])} "
          f"→ {'TEST' if in_test else 'TRAINING'}")
print(f"\nWill use gt_csv:   {'YES' if Path(GT_PATH).exists() else 'NO — FILE NOT FOUND'}")
print(f"Labels are zero:   {(y_te == 0).all()} "
      f"← should be FALSE")