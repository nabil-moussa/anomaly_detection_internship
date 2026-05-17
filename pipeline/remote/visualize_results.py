import os
import json
import argparse
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#080c10"
SURFACE = "#0e1318"
BORDER  = "#1e2730"
TEXT    = "#c9d5e0"
DIM     = "#5a6a78"

C_BLUE   = "#4a9eff"
C_GREEN  = "#3ddc84"
C_AMBER  = "#ffb340"
C_RED    = "#ff6b6b"
C_PURPLE = "#b78fff"
C_CYAN   = "#40d9f7"
C_GREY   = "#8b949e"

HTML_HEAD = """
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:#080c10; color:#c9d5e0; font-family:'JetBrains Mono',monospace; }
    .chart-wrap { padding: 24px 48px; }
  </style>
</head>"""


def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1),
        margin=dict(l=60, r=40, t=80, b=60),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True)
    return fig


def save_html(fig, path):
    plot_div = fig.to_html(
        full_html=False, include_plotlyjs="cdn",
        config={"responsive": True}
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
{HTML_HEAD}
<body>
<div class="chart-wrap">{plot_div}</div>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_run(dataset, group=None):
    output_dir = f"output/{dataset}/{group}" if dataset == "SMD" else f"output/{dataset}"
    runs = sorted([
        d for d in os.listdir(output_dir)
        if os.path.isdir(f"{output_dir}/{d}") and d != "logs"
    ])
    if not runs:
        raise FileNotFoundError(f"No runs found in {output_dir}")
    return f"{output_dir}/{runs[-1]}"


def load_results(run_path):
    with open(f"{run_path}/summary.txt") as f:
        summary = json.load(f)
    with open(f"{run_path}/config.txt") as f:
        config = json.load(f)
    train_df = pd.read_pickle(f"{run_path}/train_output.pkl")
    test_df  = pd.read_pickle(f"{run_path}/test_output.pkl")
    return summary, config, train_df, test_df


def anomaly_spans(mask):
    mask    = np.asarray(mask, dtype=bool)
    changes = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def best_feature(test_df):
    score_cols = [c for c in test_df.columns if c.startswith("A_Score_")
                  and c != "A_Score_Global"]
    if not score_cols:
        return 0

    true_cols = [f"True_{c.split('_')[-1]}" for c in score_cols
                 if f"True_{c.split('_')[-1]}" in test_df.columns]
    candidates = []
    for col in true_cols:
        vals = test_df[col].values
        avg  = float(np.mean(vals))
        std  = float(np.std(vals))
        if 0.05 < avg < 0.95 and std > 0.01:
            candidates.append((col, std))

    if candidates:
        best_true = max(candidates, key=lambda x: x[1])[0]
        return int(best_true.split("_")[-1])

    variances = {col: test_df[col].var() for col in score_cols}
    best_col  = max(variances, key=variances.get)
    return int(best_col.split("_")[-1])


PAPER_RESULTS = {
    "MSL":  {"precision": 0.8754, "recall": 0.9440, "f1": 0.9084},
    "SMAP": {"precision": 0.8906, "recall": 0.9123, "f1": 0.9013},
}


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Anomaly scores + thresholds
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_scores(test_df, summary, dataset, run_path):
    scores     = test_df["A_Score_Global"].values
    has_labels = "A_True_Global" in test_df.columns
    true_mask  = test_df["A_True_Global"].values.astype(bool) if has_labels \
                 else np.zeros(len(scores), dtype=bool)
    x = np.arange(len(scores))

    bf_thr  = summary["bf_result"].get("threshold")
    pot_thr = summary["pot_result"].get("threshold")
    bf_f1   = summary["bf_result"].get("f1", 0)
    pot_f1  = summary["pot_result"].get("f1", 0)

    from eval_methods import adjust_predicts
    def padj_pred(thr):
        if thr is None:
            return np.zeros(len(scores), dtype=bool)
        if has_labels:
            return adjust_predicts(scores, true_mask.astype(float), thr)
        return scores >= thr

    pred_bf  = padj_pred(bf_thr)
    pred_pot = padj_pred(pot_thr)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.46, 0.18, 0.18, 0.18],
        vertical_spacing=0.03,
        subplot_titles=[
            "Anomaly Score + Thresholds",
            "Ground Truth Anomalies",
            "POT Predictions  (point-adjusted)",
            "Best-F1 Predictions  (point-adjusted)",
        ]
    )

    y_ceil = float(np.percentile(scores, 99.5)) * 1.6

    fig.add_trace(go.Scatter(
        x=x, y=np.clip(scores, 0, y_ceil),
        name="Anomaly Score",
        line=dict(color=C_BLUE, width=0.7),
        fill="tozeroy", fillcolor="rgba(74,158,255,0.05)",
    ), row=1, col=1)

    if has_labels:
        for s, e in anomaly_spans(true_mask):
            fig.add_vrect(x0=s, x1=e, fillcolor=C_RED, opacity=0.10,
                          line_width=0, row=1, col=1)

    if bf_thr is not None:
        fig.add_hline(y=min(bf_thr, y_ceil),
                      line=dict(color=C_GREEN, width=1.5, dash="dash"),
                      annotation_text=f"Best-F1  thr={bf_thr:.4f}  F1={bf_f1:.4f}",
                      annotation_font_color=C_GREEN, row=1, col=1)
    if pot_thr is not None:
        fig.add_hline(y=min(pot_thr, y_ceil),
                      line=dict(color=C_AMBER, width=1.5, dash="dot"),
                      annotation_text=f"POT  thr={pot_thr:.4f}  F1={pot_f1:.4f}",
                      annotation_font_color=C_AMBER, row=1, col=1)


    fig.update_yaxes(range=[0, y_ceil], row=1, col=1)

    if has_labels:
        fig.add_trace(go.Scatter(
            x=x, y=true_mask.astype(float), name="Ground Truth",
            fill="tozeroy", line=dict(color=C_RED, width=0.5),
            fillcolor="rgba(255,107,107,0.55)",
        ), row=2, col=1)
    fig.update_yaxes(range=[-0.05, 1.3], showticklabels=False, row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=pred_pot.astype(float), name="POT Predicted",
        fill="tozeroy", line=dict(color=C_AMBER, width=0.5),
        fillcolor="rgba(255,179,64,0.45)",
    ), row=3, col=1)
    fig.update_yaxes(range=[-0.05, 1.3], showticklabels=False, row=3, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=pred_bf.astype(float), name="Best-F1 Predicted",
        fill="tozeroy", line=dict(color=C_GREEN, width=0.5),
        fillcolor="rgba(61,220,132,0.40)",
    ), row=4, col=1)
    fig.update_yaxes(range=[-0.05, 1.3], showticklabels=False, row=4, col=1)

    fig.update_layout(
        height=750,
        title=dict(text=f"Anomaly Detection — {dataset}", font=dict(size=16, color="#fff")),
        hovermode="x unified",
    )
    apply_theme(fig)
    save_html(fig, f"{run_path}/01_anomaly_scores.html")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Method comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_comparison(summary, dataset, run_path):
    methods = [ "POT", "Best-F1"]
    keys    = ["pot_result", "bf_result"]
    colors  = [C_PURPLE, C_AMBER]
    metrics = ["f1", "precision", "recall"]
    mlabels = ["F1 Score", "Precision", "Recall"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=mlabels,
                        horizontal_spacing=0.08)

    for col_idx, (metric, mlabel) in enumerate(zip(metrics, mlabels), start=1):
        vals  = [summary[k].get(metric, 0) for k in keys]
        names = methods[:]
        cols  = colors[:]
        if dataset in PAPER_RESULTS:
            vals.append(PAPER_RESULTS[dataset][metric])
            names.append("Paper (Table III)")
            cols.append(C_BLUE)

        fig.add_trace(go.Bar(
            x=names, y=vals, marker_color=cols,
            marker_line_color=[BORDER]*len(cols), marker_line_width=1,
            text=[f"{v:.4f}" for v in vals],
            textposition="outside", textfont=dict(color=TEXT, size=10),
            showlegend=False,
        ), row=1, col=col_idx)

        fig.update_yaxes(range=[0, 1.13], gridcolor=BORDER, row=1, col=col_idx)

    fig.update_layout(
        height=480, bargap=0.25,
        title=dict(text=f"Method Comparison — {dataset}", font=dict(size=16, color="#fff")),
    )
    apply_theme(fig)
    save_html(fig, f"{run_path}/02_method_comparison.html")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Forecast vs actual
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast_vs_actual(test_df, summary, dataset, run_path, n_points=3000, feature_override=None):

    feat_idx = feature_override if feature_override is not None else best_feature(test_df)
    print(f"  Forecast plot: using feature {feat_idx}")

    n = min(n_points, len(test_df))

    # find first anomaly and center window there
    has_labels = "A_True_Global" in test_df.columns
    full_mask  = test_df["A_True_Global"].values.astype(bool) if has_labels \
                 else np.zeros(len(test_df), dtype=bool)

    if has_labels and full_mask.any():
        first_anom = np.where(full_mask)[0][0]
        start = max(0, first_anom - 500)
    else:
        start = 0

    end = min(start + n, len(test_df))
    x   = np.arange(end - start)

    true      = test_df[f"True_{feat_idx}"].values[start:end]
    fore      = test_df[f"Forecast_{feat_idx}"].values[start:end]
    recon     = test_df[f"Recon_{feat_idx}"].values[start:end]
    score     = test_df["A_Score_Global"].values[start:end]
    true_mask = full_mask[start:end]

    bf_thr  = summary["bf_result"].get("threshold")
    pot_thr = summary["pot_result"].get("threshold")

    # use global score distribution for ceiling so scale matches html 01
    full_score = test_df["A_Score_Global"].values
    score_ceil = float(np.percentile(full_score[full_score > 0], 99)) * 1.4 \
                 if np.any(full_score > 0) else 1.0

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.27, 0.27, 0.27, 0.19],
        vertical_spacing=0.04,
        subplot_titles=[
            f"Forecasting: predicted next step vs actual  (feature {feat_idx})",
            f"Reconstruction: reconstructed window vs actual  (feature {feat_idx})",
            "Global anomaly score",
            "Ground Truth",
        ]
    )

    def shade_true(row):
        if has_labels:
            for s, e in anomaly_spans(true_mask):
                fig.add_vrect(x0=s, x1=e, fillcolor=C_RED, opacity=0.10,
                              line_width=0, row=row, col=1)

    # row 1: forecast
    fig.add_trace(go.Scatter(x=x, y=true, name="Actual",
                             line=dict(color=C_GREY, width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=fore, name="Forecast",
                             line=dict(color=C_BLUE, width=1.0),
                             fill="tonexty",
                             fillcolor="rgba(74,158,255,0.06)"), row=1, col=1)
    shade_true(1)

    # row 2: reconstruction
    fig.add_trace(go.Scatter(x=x, y=true, name="Actual",
                             line=dict(color=C_GREY, width=1.2),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=recon, name="Reconstruction",
                             line=dict(color=C_AMBER, width=1.0),
                             fill="tonexty",
                             fillcolor="rgba(255,179,64,0.06)"), row=2, col=1)
    shade_true(2)

    # row 3: anomaly score
    fig.add_trace(go.Scatter(
        x=x, y=np.clip(score, 0, score_ceil), name="Anomaly Score",
        line=dict(color=C_RED, width=0.9),
        fill="tozeroy", fillcolor="rgba(255,107,107,0.07)",
    ), row=3, col=1)
    if bf_thr is not None:
        fig.add_hline(y=min(bf_thr, score_ceil),
                      line=dict(color=C_GREEN, width=1.2, dash="dash"),
                      annotation_text="Best-F1", annotation_font_color=C_GREEN,
                      row=3, col=1)
    if pot_thr is not None:
        fig.add_hline(y=min(pot_thr, score_ceil),
                      line=dict(color=C_AMBER, width=1.2, dash="dot"),
                      annotation_text="POT", annotation_font_color=C_AMBER,
                      row=3, col=1)
    shade_true(3)
    fig.update_yaxes(range=[0, score_ceil], row=3, col=1)

    # row 4: ground truth
    if has_labels:
        fig.add_trace(go.Scatter(
            x=x, y=true_mask.astype(float), name="Ground Truth",
            fill="tozeroy", line=dict(color=C_RED, width=0),
            fillcolor="rgba(255,107,107,0.55)",
        ), row=4, col=1)
    fig.update_yaxes(range=[-0.05, 1.3], showticklabels=False, row=4, col=1)

    fig.update_layout(
        height=950,
        title=dict(
            text=f"Forecast & Reconstruction vs Actual — {dataset}  (feature {feat_idx})",
            font=dict(size=16, color="#fff")
        ),
        hovermode="x unified",
    )
    apply_theme(fig)
    save_html(fig, f"{run_path}/03_forecast_vs_actual.html")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Confusion breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_breakdown(summary, dataset, run_path):
    methods = [ "POT", "Best-F1"]
    keys    = [ "pot_result", "bf_result"]
    colors  = [C_PURPLE, C_AMBER]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Absolute counts (TP / FN / FP / TN)",
            "Proportions (% of test set)",
            "Precision vs Recall",
            "F1 / Precision / Recall",
        ],
        horizontal_spacing=0.12, vertical_spacing=0.18,
    )

    conf_keys   = ["TP", "FN", "FP", "TN"]
    conf_colors = [C_GREEN, C_RED, C_AMBER, C_GREY]

    for ci, ck in enumerate(conf_keys):
        vals = [summary[k].get(ck, 0) for k in keys]
        fig.add_trace(go.Bar(
            name=ck, x=methods, y=vals, marker_color=conf_colors[ci],
            text=[f"{int(v):,}" for v in vals],
            textposition="outside", textfont=dict(size=9),
        ), row=1, col=1)

    for ci, ck in enumerate(conf_keys):
        vals = []
        for k in keys:
            total = sum(summary[k].get(x, 0) for x in conf_keys)
            vals.append(summary[k].get(ck, 0) / total * 100 if total > 0 else 0)
        fig.add_trace(go.Bar(
            name=ck, x=methods, y=vals, marker_color=conf_colors[ci],
            text=[f"{v:.1f}%" for v in vals],
            textposition="inside", textfont=dict(size=9, color="#fff"),
            showlegend=False,
        ), row=1, col=2)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="%", row=1, col=2)

    for method, key, color in zip(methods, keys, colors):
        r = summary[key]
        fig.add_trace(go.Scatter(
            x=[r.get("recall", 0)], y=[r.get("precision", 0)],
            mode="markers+text", name=method,
            marker=dict(color=color, size=14, line=dict(color="#fff", width=1)),
            text=[method], textposition="top center",
            textfont=dict(size=10, color=color), showlegend=False,
        ), row=2, col=1)

    if dataset in PAPER_RESULTS:
        p = PAPER_RESULTS[dataset]
        fig.add_trace(go.Scatter(
            x=[p["recall"]], y=[p["precision"]],
            mode="markers+text", name="Paper",
            marker=dict(color=C_BLUE, size=16, symbol="star",
                        line=dict(color="#fff", width=1)),
            text=["Paper"], textposition="top center",
            textfont=dict(size=10, color=C_BLUE), showlegend=False,
        ), row=2, col=1)

    recall_range = np.linspace(0.01, 1, 200)
    for f1_val in [0.5, 0.7, 0.8, 0.9]:
        prec_iso = f1_val * recall_range / (2 * recall_range - f1_val + 1e-9)
        valid    = (prec_iso > 0) & (prec_iso <= 1)
        fig.add_trace(go.Scatter(
            x=recall_range[valid], y=prec_iso[valid],
            mode="lines", line=dict(color=BORDER, width=1, dash="dot"),
            showlegend=False,
        ), row=2, col=1)
        idx = np.argmin(np.abs(recall_range - 0.55))
        if 0 < prec_iso[idx] <= 1:
            fig.add_annotation(x=0.55, y=prec_iso[idx],
                               text=f"F1={f1_val}", showarrow=False,
                               font=dict(size=9, color=DIM), row=2, col=1)

    fig.update_xaxes(title_text="Recall",    range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="Precision", range=[0, 1.05], row=2, col=1)

    for mi, (mk, ml, bc) in enumerate(zip(
        ["f1", "precision", "recall"], ["F1", "Prec", "Rec"],
        [C_BLUE, C_CYAN, C_GREEN]
    )):
        vals = [summary[k].get(mk, 0) for k in keys]
        fig.add_trace(go.Bar(
            name=ml, x=methods, y=vals, marker_color=bc,
            text=[f"{v:.4f}" for v in vals],
            textposition="outside", textfont=dict(size=9),
        ), row=2, col=2)

    if dataset in PAPER_RESULTS:
        for mk in ["f1", "precision", "recall"]:
            fig.add_hline(y=PAPER_RESULTS[dataset][mk],
                          line=dict(color=C_BLUE, width=1, dash="dot"),
                          row=2, col=2)

    fig.update_yaxes(range=[0, 1.15], row=2, col=2)
    fig.update_layout(
        height=900, barmode="group",
        title=dict(text=f"Confusion Breakdown — {dataset}",
                   font=dict(size=16, color="#fff")),
    )
    apply_theme(fig)
    save_html(fig, f"{run_path}/04_confusion_breakdown.html")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=str.upper, default="SMAP")
    parser.add_argument("--group",    type=str,       default="1-1")
    parser.add_argument("--run_path", type=str,       default=None)
    parser.add_argument("--feature",  type=int,       default=None)
    args = parser.parse_args()

    dataset  = args.dataset
    group    = args.group if dataset == "SMD" else None
    run_path = args.run_path or get_latest_run(dataset, group)

    print(f"\nLoading results from: {run_path}")
    summary, config, train_df, test_df = load_results(run_path)

    print("\nGenerating visualizations...")
    plot_anomaly_scores(test_df, summary, dataset, run_path)
    plot_method_comparison(summary, dataset, run_path)
    plot_forecast_vs_actual(test_df, summary, dataset, run_path, feature_override=args.feature)
    plot_confusion_breakdown(summary, dataset, run_path)

    print(f"\nAll HTML files saved to: {run_path}/")
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  RESULTS — {dataset}")
    print(sep)
    for method, key in [
                         ("POT",     "pot_result"),
                         ("Best-F1", "bf_result")]:
        r = summary[key]
        print(f"  {method:<12}  "
              f"F1={r.get('f1',0):.4f}  "
              f"P={r.get('precision',0):.4f}  "
              f"R={r.get('recall',0):.4f}")
    if dataset in PAPER_RESULTS:
        p   = PAPER_RESULTS[dataset]
        gap = p["f1"] - summary["bf_result"].get("f1", 0)
        print(sep)
        print(f"  {'Paper':<12}  "
              f"F1={p['f1']:.4f}  P={p['precision']:.4f}  R={p['recall']:.4f}  (Table III)")
        status = "within variance ✓" if abs(gap) < 0.02 else "check config"
        print(f"\n  Gap Best-F1 vs Paper: {gap:+.4f}  [{status}]")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()