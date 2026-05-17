from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def export_after_prediction(trainer,predictor,x_train,x_test,summary: dict,stream_name: str,handoff_dir: str | Path) -> Path:

    print(f"\n[export_scores] Extracting DL scores for handoff...")


    train_df = predictor.get_score(x_train)
    test_df  = predictor.get_score(x_test)

    train_scores = train_df["A_Score_Global"].values
    test_scores  = test_df["A_Score_Global"].values

    dl_threshold = summary.get("pot_result", {}).get("threshold")

    # Build payload
    handoff_dir = Path(handoff_dir)
    handoff_dir.mkdir(parents=True, exist_ok=True)
    p = handoff_dir / f"dl_scores_{stream_name}.json"

    payload = {
        "stream":        stream_name,
        "train_scores":  train_scores.tolist(),
        "test_scores":   test_scores.tolist(),
        "n_train":       int(len(train_scores)),
        "n_test":        int(len(test_scores)),
        "dl_threshold":  dl_threshold,
        "meta": {
            "pot_f1":       summary.get("pot_result", {}).get("f1"),
        },
    }

    p.write_text(json.dumps(payload, indent=2))
    print(f"[export_scores] DL scores saved -> {p}")
    print(f"  train_scores : n={len(train_scores)}  "
          f"range=[{train_scores.min():.5f}, {train_scores.max():.5f}]")
    print(f"  test_scores  : n={len(test_scores)}  "
          f"range=[{test_scores.min():.5f}, {test_scores.max():.5f}]")
    print(f"  dl_threshold : {dl_threshold}")
    return p


def export_from_saved(save_path: str | Path,
                      stream_name: str,
                      handoff_dir: str | Path) -> Path:
    """
    Alternative: load already-saved train_output.pkl / test_output.pkl from
    a previous MTAD-GAT run and export without re-running inference.
    """
    save_path = Path(save_path)
    print(f"\n[export_scores] Loading saved outputs from {save_path} ...")

    train_df = pd.read_pickle(save_path / "train_output.pkl")
    test_df  = pd.read_pickle(save_path / "test_output.pkl")

    train_scores = train_df["A_Score_Global"].values
    test_scores  = test_df["A_Score_Global"].values

    dl_threshold = None
    summary_path = save_path / "summary.txt"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            dl_threshold = summary.get("pot_result", {}).get("threshold")
        except Exception:
            pass

    handoff_dir = Path(handoff_dir)
    handoff_dir.mkdir(parents=True, exist_ok=True)
    p = handoff_dir / f"dl_scores_{stream_name}.json"

    payload = {
        "stream":        stream_name,
        "train_scores":  train_scores.tolist(),
        "test_scores":   test_scores.tolist(),
        "n_train":       int(len(train_scores)),
        "n_test":        int(len(test_scores)),
        "dl_threshold":  dl_threshold,
    }
    p.write_text(json.dumps(payload, indent=2))
    print(f"[export_scores] DL scores saved -> {p}")
    return p


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export DL anomaly scores to JSON handoff file.")
    parser.add_argument("--save_path",    required=True,
                        help="Path to MTAD-GAT output dir (contains *.pkl)")
    parser.add_argument("--stream_name",  required=True,
                        help="Stream identifier, e.g. stream_A")
    parser.add_argument("--handoff_dir",  required=True,
                        help="Directory to write dl_scores_{stream}.json")
    args = parser.parse_args()

    export_from_saved(args.save_path, args.stream_name, args.handoff_dir)