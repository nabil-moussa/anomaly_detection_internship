from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): 
        return [_clean(v) for v in obj]
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj) 


# Adequacy handoff

def save_adequacy_for_remote(adequacy_result: dict,
                              handoff_dir: str | Path,
                              stream_name: str) -> Path:

    p = Path(handoff_dir) / f"adequacy_{stream_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(adequacy_result), indent=2), encoding="utf-8")
    print(f"  [IO] Adequacy saved -> {p}")
    return p


def load_adequacy(handoff_dir: str | Path, stream_name: str) -> Optional[dict]:
    p = Path(handoff_dir) / f"adequacy_{stream_name}.json"
    if not p.exists():
        print(f"  [IO] Adequacy file not found: {p}")
        return None
    return json.loads(p.read_text())


# DL score handoff

def save_dl_scores(train_scores: np.ndarray,
                   test_scores:  np.ndarray,
                   handoff_dir:  str | Path,
                   stream_name:  str,
                   dl_threshold: Optional[float] = None,
                   extra: Optional[dict] = None) -> Path:

    p = Path(handoff_dir) / f"dl_scores_{stream_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stream":        stream_name,
        "train_scores":  train_scores.tolist(),
        "test_scores":   test_scores.tolist(),
        "n_train":       int(len(train_scores)),
        "n_test":        int(len(test_scores)),
        "dl_threshold":  float(dl_threshold) if dl_threshold is not None else None,
    }
    if extra:
        payload["meta"] = _clean(extra)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  [IO] DL scores saved -> {p}")
    return p


def load_dl_scores(handoff_dir: str | Path,
                   stream_name: str) -> Optional[dict]:

    p = Path(handoff_dir) / f"dl_scores_{stream_name}.json"
    if not p.exists():
        print(f"  [IO] DL scores file not found: {p}")
        return None
    d = json.loads(p.read_text())
    d["train_scores"] = np.array(d["train_scores"])
    d["test_scores"]  = np.array(d["test_scores"])
    return d


# CCAD output persistence

def save_stat_output(ccad_out: dict,
                     results_dir: str | Path,
                     stream_name: str) -> Path:
    
    p = Path(results_dir) / f"stat_output_{stream_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)

    slim = {
        "group_id":             ccad_out["group_id"],
        "group_cols":           ccad_out["group_cols"],
        "group_period":         ccad_out["group_period"],
        "seg_method":           ccad_out["seg_method"],
        "train_end":            ccad_out["train_end"],
        "all_cycles":           ccad_out["all_cycles"],
        "all_template_indices": sorted(ccad_out["all_template_indices"]),
        "corr_anomalies":       sorted(ccad_out["corr_anomalies"]),
        "amp_anomalies":        sorted(ccad_out["amp_anomalies"]),
        "final_anomalies":      sorted(ccad_out["final_anomalies"]),
        "results": [
            {
                "cycle_idx":            r["cycle_idx"],
                "cluster_id":           r["cluster_id"],
                "is_anomaly":           r["is_anomaly"],
                "max_score":            float(r["max_score"]),
                "mean_score":           float(r["mean_score"]),
                "windows_above_thresh": r["windows_above_thresh"],
                "threshold":            float(r["threshold"]),
                "shift_applied":        int(r["shift_applied"]),
                "window_scores":        r["window_scores"].tolist(),
            }
            for r in ccad_out["results"]
        ],
        "cycle_amp_scores": [
            float(s) if (s is not None and not np.isnan(float(s))) else None
            for s in ccad_out["cycle_amp_scores"]
        ],
        "amp_threshold": (
            float(ccad_out["amp_threshold"])
            if ccad_out["amp_threshold"] is not None else None
        ),
    }

    p.write_text(json.dumps(_clean(slim), indent=2), encoding="utf-8")
    print(f"  [IO] CCAD output saved -> {p}")
    return p


def load_stat_output(results_dir: str | Path, stream_name: str) -> Optional[dict]:
    p = Path(results_dir) / f"stat_output_{stream_name}.json"
    if not p.exists():
        print(f"  [IO] Stat output file not found: {p}")
        return None
    d = json.loads(p.read_text())
    # Restore sets and arrays
    d["all_template_indices"] = set(d["all_template_indices"])
    d["corr_anomalies"]       = set(int(x) for x in d["corr_anomalies"])
    d["amp_anomalies"]        = set(int(x) for x in d["amp_anomalies"])
    d["final_anomalies"]      = set(int(x) for x in d["final_anomalies"])
    for r in d["results"]:
        r["window_scores"] = np.array(r["window_scores"])
    return d


# Taxonomy CSV export

def export_taxonomy_csv(fusion_report,
                         results_dir: str | Path,
                         stream_name: str) -> Path:
    """Write one row per cycle with all scores and the taxonomy label."""
    p = Path(results_dir) / f"taxonomy_{stream_name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in fusion_report.cycles:
        rows.append({
            "cycle_idx":       c.cycle_idx + 1,
            "start":           c.start,
            "end":             c.end,
            "length":          c.length,
            "stat_score_raw":  c.stat_score_raw,
            "dl_score_raw":    c.dl_score_raw,
            "stat_score_norm": c.stat_score_norm,
            "dl_score_norm":   c.dl_score_norm,
            "fused_score":     c.fused_score,
            "is_anomaly":      c.is_anomaly,
            "detection_type":  c.detection_type,
            "stat_flagged":    c.stat_flagged,
            "dl_flagged":      c.dl_flagged,
        })
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"  [IO] Taxonomy CSV saved -> {p}")
    return p