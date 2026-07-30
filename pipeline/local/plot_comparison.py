import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

TAXONOMY  = r"C:\Users\user\OneDrive\Desktop\MTS\Pipeline\local\results\taxonomy_stream_A1.csv"
DL_PATH   = r"C:\Users\user\OneDrive\Desktop\MTS\Pipeline\local\handoff\dl_scores_stream_A1.json"
STAT_PATH = r"C:\Users\user\OneDrive\Desktop\MTS\Pipeline\local\results\stat_output_stream_A1.json"

df = pd.read_csv(TAXONOMY)

with open(DL_PATH) as f:
    dl_data = json.load(f)
with open(STAT_PATH) as f:
    stat_out = json.load(f)



dl_scores   = np.array(dl_data["test_scores"])
dl_thresh   = dl_data["dl_threshold"]
train_end   = stat_out["train_end"]
test_offset = train_end + 100
ccad_thresh = max(r["threshold"] for r in stat_out["results"])
all_cycles  = stat_out["all_cycles"]
stream_len  = all_cycles[-1][1]
mtad_split = int(0.6 * stream_len)  # 60% boundary
faults = [
    ("S1\nBall fault",   82000,  84500, "#d62728"),
    ("S2\nOuter race",  106000, 111500, "#ff7f0e"),
    ("S3\nBall fault",  111500, 119500, "#2ca02c"),
]

# ── CCAD: only test cycles with real scores ───────────────
test_df = df[df["stat_score_raw"].notna() & 
             (df["stat_score_raw"] != "")].copy()
test_df["stat_score_raw"] = pd.to_numeric(
    test_df["stat_score_raw"], errors="coerce")
test_df = test_df.dropna(subset=["stat_score_raw"])

bar_centers = ((test_df["start"] + test_df["end"]) / 2).values
bar_heights = test_df["stat_score_raw"].values
bar_widths  = (test_df["end"] - test_df["start"]).values * 0.85

# ── Figure ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 4.8),
    sharex=True,
    gridspec_kw={"hspace": 0.28}
)
fig.patch.set_facecolor("white")
xlim = (0, stream_len + 2000)

# ── TOP: CCAD ─────────────────────────────────────────────
# gray background for training region
ax1.axvspan(0, train_end, alpha=0.08, color="gray",
            zorder=0, label="Training (template cycles)")

ax1.bar(bar_centers, bar_heights, width=bar_widths,
        color="#1f77b4", alpha=0.7, zorder=3,
        label="CCAD max z-score per cycle")
ax1.axhline(ccad_thresh, color="red", linewidth=1.5,
            linestyle="--",
            label=f"LOO threshold ({ccad_thresh:.2f})")

ymax1 = test_df["stat_score_raw"].max() * 1.20

for label, gs, ge, col in faults:
    ax1.axvspan(gs, ge, alpha=0.20, color=col, zorder=1)
    ax1.text((gs+ge)/2, ymax1 * 0.97, label,
             ha="center", va="top", fontsize=8,
             color=col, fontweight="bold", linespacing=1.3)

ax1.set_xlim(xlim)
ax1.set_ylim(0, ymax1)
ax1.set_ylabel("Max window z-score", fontsize=11)
ax1.set_title("CCAD — cycle-level anomaly score",
              fontsize=11, pad=3)
ax1.legend(fontsize=8.5, loc="upper left")
ax1.set_facecolor("white")
ax1.grid(True, alpha=0.25)
ax1.tick_params(labelsize=10)

# ── BOTTOM: MTAD-GAT ──────────────────────────────────────
ts_idx = np.arange(len(dl_scores)) + test_offset

mtad_test_start = int(stream_len * 0.75) + 100  # 97225
ax2.axvspan(0, mtad_test_start, alpha=0.08, color="gray",
            zorder=0, label="MTAD-GAT training (no test scores)")
ax2.plot(ts_idx, dl_scores, color="#1f77b4",
         linewidth=0.8, zorder=3,
         label="MTAD-GAT anomaly score")
ax2.axhline(dl_thresh, color="red", linewidth=1.5,
            linestyle="--",
            label=f"POT threshold ({dl_thresh:.3f})")

ymax2 = dl_scores.max() * 1.15

for label, gs, ge, col in faults:
    ax2.axvspan(gs, ge, alpha=0.20, color=col, zorder=1)
    ax2.text((gs+ge)/2, ymax2 * 0.97, label,
             ha="center", va="top", fontsize=8,
             color=col, fontweight="bold", linespacing=1.3)

ax2.set_xlim(xlim)
ax2.set_ylim(0, ymax2)
ax2.set_ylabel("Anomaly score", fontsize=11)
ax2.set_xlabel("Sample index", fontsize=11)
ax2.set_title("MTAD-GAT — timestep-level anomaly score",
              fontsize=11, pad=3)
ax2.legend(fontsize=8.5, loc="upper left")
ax2.set_facecolor("white")
ax2.grid(True, alpha=0.25)
ax2.tick_params(labelsize=10)

plt.subplots_adjust(top=0.93, bottom=0.10, left=0.06, right=0.98)

plt.savefig("results/plots/comparison_A1.pdf",
            bbox_inches="tight", dpi=300)
plt.savefig("results/plots/comparison_A1.png",
            bbox_inches="tight", dpi=300)
print("Saved.")
plt.close()