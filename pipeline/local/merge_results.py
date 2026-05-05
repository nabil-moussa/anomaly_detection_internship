from __future__ import annotations

import numpy as np
from collections import Counter
from pathlib import Path
from typing import Optional

from hybrid_fuser import FusionReport, CycleResult
from utils_io import export_taxonomy_csv



def _explain_stat(c: CycleResult, stat_output: dict) -> str:
    if not c.stat_flagged:
        return "  Statistical: not flagged."
    r = next((r for r in stat_output["results"] if r["cycle_idx"] == c.cycle_idx), None)
    if r is None:
        return "  Statistical: flagged (no window detail available)."

    ws    = np.array(r["window_scores"])
    thr   = r["threshold"]
    above = np.where(ws > thr)[0]
    lines = [f"  Statistical: max_score={r['max_score']:.2f}σ  "
             f"({len(above)}/{len(ws)} windows > threshold={thr:.2f})"]
    if len(above):
        worst = int(np.argmax(ws))
        lines.append(f"    Worst window : W{worst}  score={ws[worst]:.2f}σ")
    if r["mean_score"] > 2 * thr:
        lines.append("    -> Sustained correlation breakdown across the full cycle.")
    elif len(above) == 1:
        lines.append("    -> Localised disruption in one time segment.")
    else:
        lines.append("    -> Partial correlation breakdown across several segments.")
    return "\n".join(lines)


def _explain_dl(c: CycleResult, dl_ts_scores: Optional[np.ndarray]) -> str:
    if not c.dl_flagged:
        return "  DL: not flagged."
    if dl_ts_scores is None or c.dl_score_raw is None:
        return "  DL: flagged (no score detail available)."
    s = c.start
    e = min(c.end + 1, len(dl_ts_scores))
    if s >= len(dl_ts_scores):
        return f"  DL: flagged  raw_score={c.dl_score_raw:.4f}"
    window  = dl_ts_scores[s:e]
    peak    = float(np.nanmax(window))
    peak_t  = int(np.nanargmax(window)) + s
    lines   = [f"  DL: cycle_score={c.dl_score_raw:.4f}  "
               f"peak_timestep={peak_t}  peak_value={peak:.4f}"]
    if peak > 2 * c.dl_score_raw:
        lines.append("    -> Sharp spike: transient amplitude anomaly.")
    else:
        lines.append("    -> Elevated reconstruction error: sustained pattern deviation.")
    return "\n".join(lines)


_TYPE_DESC = {
    "BOTH":
        "HIGH CONFIDENCE — relational breakdown AND amplitude deviation.",
    "STAT_ONLY":
        "EARLY WARNING   — inter-sensor correlation broke down; DL did not fire.\n"
        "                  May indicate subtle or incipient fault.",
    "DL_ONLY":
        "AMPLITUDE/PATTERN — all sensors deviate together or temporal pattern\n"
        "                  changed without altering inter-sensor correlations.",
    "FUSED":
        "MARGINAL        — neither method crossed its own threshold alone;\n"
        "                  weighted combination does. Use with caution.",
    "NORMAL":   "NORMAL",
    "TEMPLATE": "TEMPLATE (training)",
}



def build_taxonomy_report(fusion_report:  FusionReport,
                           stat_output:   Optional[dict],
                           dl_ts_scores:  Optional[np.ndarray],
                           stream_name:   str,
                           results_dir:   str | Path = "results") -> str:
    sep   = "═" * 70
    lines = ["", sep,
             f"  HYBRID ANOMALY TAXONOMY REPORT  —  {stream_name}",
             sep]

    route_label = ("hybrid" if 0 < fusion_report.stat_weight < 1
                   else ("statistical" if fusion_report.stat_weight == 1 else "dl"))
    lines += [
        f"  Route        : {route_label}",
        f"  Weights      : stat={fusion_report.stat_weight:.2f}  "
        f"dl={fusion_report.dl_weight:.2f}",
        f"  Threshold    : {fusion_report.threshold:.2f} (fused normalised score)",
        f"  Total cycles : {fusion_report.n_cycles}",
        f"  Anomalies    : {fusion_report.n_anomalies}  "
        f"({fusion_report.anomaly_rate:.1%})",
        "",
    ]

    # Type distribution
    type_counts = Counter(c.detection_type for c in fusion_report.cycles)
    lines += ["  TYPE DISTRIBUTION", "  " + "─" * 66]
    for t in ["BOTH", "STAT_ONLY", "DL_ONLY", "FUSED", "NORMAL", "TEMPLATE"]:
        n   = type_counts.get(t, 0)
        bar = "█" * min(n, 50)
        lines.append(f"  {t:<12} {bar} {n}")
    lines.append("")

    # Guide
    lines += [
        "  ANOMALY TYPE GUIDE", "  " + "─" * 66,
        "  BOTH       : Both methods flagged — highest confidence.",
        "               Relational breakdown confirmed by reconstruction error.",
        "",
        "  STAT_ONLY  : Only the statistical method flagged.",
        "               Correlation structure changed; amplitude stayed normal.",
        "               Candidate for early-stage fault — worth monitoring.",
        "",
        "  DL_ONLY    : Only MTAD-GAT flagged.",
        "               Sensors may drift together (preserving correlations)",
        "               or temporal pattern changed in a way CCAD missed.",
        "",
        "  FUSED      : Neither method alone crossed its threshold.",
        "               Weighted combination does. Treat as low-confidence.",
        "",
        sep, "  ANOMALY DETAILS", sep,
    ]

    anom = [c for c in fusion_report.cycles if c.is_anomaly]
    if not anom:
        lines.append("  No anomalies detected.")
    else:
        for c in sorted(anom, key=lambda x: x.fused_score, reverse=True):
            sn = f"{c.stat_score_norm:.3f}" if c.stat_score_norm is not None else "n/a"
            dn = f"{c.dl_score_norm:.3f}"   if c.dl_score_norm   is not None else "n/a"
            lines += [
                "",
                f"  {'─'*68}",
                f"  CYCLE {c.cycle_idx+1:<4}  idx {c.start}→{c.end}  ({c.length} pts)",
                f"  Type   : {_TYPE_DESC.get(c.detection_type, c.detection_type)}",
                f"  Scores : stat_norm={sn}  dl_norm={dn}  fused={c.fused_score:.3f}",
                "",
            ]
            if stat_output:
                lines.append(_explain_stat(c, stat_output))
            if dl_ts_scores is not None:
                lines.append(_explain_dl(c, dl_ts_scores))

    lines += ["", sep, ""]
    report_str = "\n".join(lines)

    p = Path(results_dir) / f"taxonomy_report_{stream_name}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report_str, encoding="utf-8")
    print(f"  [REPORT] Saved -> {p}")
    return report_str


# Visualisation

_TYPE_COLORS = {
    "BOTH":      ("#7F1D1D", "#FCA5A5"),
    "STAT_ONLY": ("#1E3A5F", "#BFDBFE"),
    "DL_ONLY":   ("#78350F", "#FDE68A"),
    "FUSED":     ("#4B0082", "#E9D5FF"),
    "NORMAL":    ("#166534", "#D1FAE5"),
    "TEMPLATE":  ("#374151", "#E5E7EB"),
}


def visualise_taxonomy(fusion_report: FusionReport,
                        corr_signal:   Optional[np.ndarray],
                        results_dir:   str | Path,
                        stream_name:   str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [VIZ] matplotlib not available — skipping")
        return None

    fig, ax = plt.subplots(figsize=(20, 5))

    if corr_signal is not None:
        valid = corr_signal[~np.isnan(corr_signal)]
        y0 = float(np.nanmin(valid)) if len(valid) else 0.0
        y1 = float(np.nanmax(valid)) if len(valid) else 1.0
        ax.plot(np.arange(len(corr_signal)), corr_signal,
                color="#2563EB", lw=1.0, alpha=0.6,
                label="Correlation strength", zorder=2)
    else:
        y0, y1 = 0.0, 1.0

    yspan = max(y1 - y0, 1e-3)
    ax.set_ylim(y0 - 0.05*yspan, y1 + 0.20*yspan)

    for c in fusion_report.cycles:
        col_border, col_fill = _TYPE_COLORS.get(c.detection_type, ("#6B7280", "#F3F4F6"))
        rect = mpatches.Rectangle(
            (c.start, y0), c.end - c.start, yspan,
            facecolor=col_fill, edgecolor=col_border,
            linewidth=1.2 if c.is_anomaly else 0.5,
            alpha=0.55 if c.is_anomaly else 0.20, zorder=1,
        )
        ax.add_patch(rect)
        if c.is_anomaly:
            ax.text(c.start + (c.end - c.start) * 0.05,
                    y1 + 0.01*yspan,
                    f"{c.detection_type}\nC{c.cycle_idx+1}\n{c.fused_score:.2f}",
                    fontsize=6, va="bottom", ha="left",
                    color=col_border, clip_on=True)

    handles = [
        mpatches.Patch(facecolor=_TYPE_COLORS[t][1],
                       edgecolor=_TYPE_COLORS[t][0], label=t)
        for t in ["BOTH", "STAT_ONLY", "DL_ONLY", "FUSED", "NORMAL", "TEMPLATE"]
    ]
    ax.legend(handles=handles, loc="upper right", ncol=6,
              fontsize=8, framealpha=0.9)
    ax.set_xlabel("Sample index", fontsize=10)
    ax.set_ylabel("Correlation strength", fontsize=10)
    ax.set_title(
        f"Hybrid anomaly taxonomy — {stream_name}\n"
        f"stat_w={fusion_report.stat_weight:.2f}  "
        f"dl_w={fusion_report.dl_weight:.2f}  "
        f"{fusion_report.n_anomalies} anomalies / {fusion_report.n_cycles} cycles",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    p = Path(results_dir) / "plots" / f"taxonomy_{stream_name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIZ] Taxonomy figure saved -> {p}")
    return str(p)


# ── One-call convenience wrapper ──────────────────────────────────────────────

def merge_and_report(fusion_report:  FusionReport,
                     stat_output:    Optional[dict],
                     dl_ts_scores:   Optional[np.ndarray],
                     corr_signal:    Optional[np.ndarray],
                     stream_name:    str,
                     results_dir:    str | Path = "results") -> dict:
    """Run text report, figure, and CSV in one call."""
    build_taxonomy_report(
        fusion_report, stat_output, dl_ts_scores, stream_name, results_dir)
    fig_path = visualise_taxonomy(
        fusion_report, corr_signal, results_dir, stream_name)
    csv_path = export_taxonomy_csv(fusion_report, results_dir, stream_name)
    return {
        "report": str(Path(results_dir) / f"taxonomy_report_{stream_name}.txt"),
        "figure": fig_path,
        "csv":    str(csv_path),
    }