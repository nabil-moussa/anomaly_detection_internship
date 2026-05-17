from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from router        import route, print_decision, STAT_THR, HYBRID_THR
from hybrid_fuser  import HybridFuser, build_stat_scores_array, dl_flagged_from_scores
from merge_results import merge_and_report
from utils_io      import (save_adequacy_for_remote, save_stat_output,
                            load_stat_output)

# Config

CONFIG_FILE = Path(__file__).parent / "mesocentre_config.json"

DEFAULT_CONFIG = {
    "host":          "helios1.univ-fcomte.fr",
    "user":          "nmoussa",
    "remote_dir":    "/Work/Users/nmoussa/mtad-gat-pytorch",
    "handoff_dir":   "/Work/Users/nmoussa/handoff",
    "conda_env":     "mtad",
    "poll_interval": 30,
    "job_timeout":   14400,
}

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=mtad_{stream}
#SBATCH --output={remote_dir}/logs/mtad_{stream}_%j.out
#SBATCH --error={remote_dir}/logs/mtad_{stream}_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu

module load anaconda3@2022.10/gcc-12.1.0
source activate {conda_env}
export MPLCONFIGDIR=$WORK/matplotlib_cache
export LD_PRELOAD=/Home/Users/{user}/.conda/envs/{conda_env}/lib/libstdc++.so.6
mkdir -p $MPLCONFIGDIR
mkdir -p {handoff_dir}
mkdir -p {remote_dir}/logs

cd {remote_dir}

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python train.py \\
  --dataset CUSTOM \\
  --epochs 50 \\
  --lookback 100 \\
  --normalize False \\
  --use_vae False \\
  --use_gatv2 False \\
  --gru_hid_dim 300 \\
  --fc_hid_dim 300 \\
  --recon_hid_dim 300 \\
  --gamma 0.8 \\
  --init_lr 0.001 \\
  --bs 256 \\
  --use_cuda True \\
  --use_sr_cleaning False \\
  --n_seeds 1 \\
  --seeds 42

"""


def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        stored = json.loads(CONFIG_FILE.read_text())
        cfg.update(stored)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    print(f"  Config saved -> {CONFIG_FILE}")


def setup_wizard() -> None:
    print("\n" + "="*60)
    print("  MESOCENTRE SSH SETUP")
    print("="*60)
    print("  Press Enter to keep the default shown in brackets.\n")
    cfg = load_config()
    for key, label in [
        ("host",         "Hostname"),
        ("user",         "Username"),
        ("remote_dir",   "Remote working directory (mtad-gat-pytorch path)"),
        ("handoff_dir",  "Remote handoff directory"),
        ("conda_env",    "Conda environment name"),
    ]:
        val = input(f"  {label} [{cfg[key]}]: ").strip()
        if val:
            cfg[key] = val
    save_config(cfg)

    print("\n  Testing SSH connection...")
    ok, out = _ssh(cfg, "echo SSH_OK", capture=True)
    if ok and "SSH_OK" in out:
        print("  Connection successful.\n")
    else:
        print(f"  WARNING: could not connect. Output: {out}")
        print("  Set up passwordless SSH keys:")
        print("    ssh-keygen -t ed25519")
        print(f"    ssh-copy-id {cfg['user']}@{cfg['host']}\n")


# SSH / SCP helpers 

def _ssh_opts(cfg: dict) -> list:
    import platform
    base = [
        "-o", "ConnectTimeout=20",
        "-o", "StrictHostKeyChecking=no",
    ]
    if platform.system() != "Windows":
        socket = str(Path(__file__).parent / f".ssh_ctl_{cfg['user']}")
        base += [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={socket}",
            "-o", "ControlPersist=600",
        ]
    return base


def _ssh(cfg: dict, command: str, capture: bool = False):
    cmd = ["ssh"] + _ssh_opts(cfg) + [f"{cfg['user']}@{cfg['host']}", command]
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=120)
        return r.returncode == 0, (r.stdout + r.stderr).strip() if capture else ""
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, ("ssh not found.")


def _scp_up(cfg: dict, local: str, remote: str) -> bool:
    cmd = ["scp"] + _ssh_opts(cfg) + [local, f"{cfg['user']}@{cfg['host']}:{remote}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return r.returncode == 0
    except Exception as e:
        print(f"    scp upload error: {e}")
        return False


def _scp_down(cfg: dict, remote: str, local: str) -> bool:
    cmd = ["scp"] + _ssh_opts(cfg) + [f"{cfg['user']}@{cfg['host']}:{remote}", local]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return r.returncode == 0
    except Exception as e:
        print(f"    scp download error: {e}")
        return False


# Data preparation & upload

def prepare_and_upload_data(df: pd.DataFrame, sensor_cols: list,
                             cfg: dict, local_tmp: str,
                             train_frac: float = 0.75) -> bool:
    import pickle
    print("\n  [DATA] Preparing pkl files for MTAD-GAT...")
    Path(local_tmp).mkdir(parents=True, exist_ok=True)

    data  = df[sensor_cols].values.astype("float32")
    split = int(len(data) * train_frac)
    x_tr  = data[:split]
    lookback = 100
    x_te  = data[split:]
    y_te  = np.zeros(max(0, len(x_te) - lookback), dtype="float32")

    for obj, name in [(x_tr, "CUSTOM_train.pkl"),
                       (x_te, "CUSTOM_test.pkl"),
                       (y_te, "CUSTOM_test_label.pkl")]:
        with open(Path(local_tmp) / name, "wb") as f:
            pickle.dump(obj, f)

    print(f"    train={x_tr.shape}  test={x_te.shape}")

    remote_custom = f"{cfg['remote_dir']}/datasets/custom"
    _ssh(cfg, f"mkdir -p {remote_custom}")

    for name in ["CUSTOM_train.pkl", "CUSTOM_test.pkl", "CUSTOM_test_label.pkl"]:
        local_path = str(Path(local_tmp) / name)
        print(f"    Uploading {name} ...", end=" ", flush=True)
        if _scp_up(cfg, local_path, f"{remote_custom}/{name}"):
            print("OK")
        else:
            print("FAILED")
            return False
    return True


# SLURM submission & polling 

def submit_job(cfg: dict, stream: str, local_tmp: str) -> str | None:
    script = SLURM_TEMPLATE.format(
        stream      = stream,
        remote_dir  = cfg["remote_dir"],
        handoff_dir = cfg["handoff_dir"],
        conda_env   = cfg["conda_env"],
        user        = cfg["user"],
    )
    local_sh  = str(Path(local_tmp) / f"run_{stream}.sh")
    remote_sh = f"{cfg['remote_dir']}/run_{stream}.sh"
    Path(local_sh).write_bytes(script.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))

    print("\n  [SLURM] Uploading job script...", end=" ", flush=True)
    if not _scp_up(cfg, local_sh, remote_sh):
        print("FAILED")
        return None
    print("OK")

    _ssh(cfg, f"chmod +x {remote_sh}")
    ok, out = _ssh(cfg, f"cd {cfg['remote_dir']} && sbatch {remote_sh}", capture=True)
    if not ok:
        print(f"  [SLURM] sbatch failed: {out}")
        return None

    job_id = next((w for w in out.split() if w.isdigit()), None)
    if job_id:
        print(f"  [SLURM] Submitted — job ID: {job_id}")
    else:
        print(f"  [SLURM] Unexpected output: {out}")
    return job_id


def poll_job(cfg: dict, job_id: str) -> bool:
    interval = cfg["poll_interval"]
    timeout  = cfg["job_timeout"]
    print(f"\n  [POLL] Waiting for job {job_id} (checking every {interval}s)...")
    start = time.time()

    while time.time() - start < timeout:
        ok, out = _ssh(cfg, f"squeue -j {job_id} -h", capture=True)
        if ok and job_id in out:
            elapsed = int(time.time() - start)
            tokens  = out.split()
            state   = next((t for t in tokens if t in ("R","PD","CG","F","CA")), "?")
            label   = {"R":"RUNNING","PD":"PENDING","CG":"COMPLETING",
                       "F":"FAILED","CA":"CANCELLED"}.get(state, state)
            print(f"    [{elapsed:>5}s] {label}", end="\r", flush=True)
            time.sleep(interval)
        else:
            print()
            ok2, sacct = _ssh(cfg,
                f"sacct -j {job_id} --format=State --noheader | head -1",
                capture=True)
            state = sacct.strip().split()[0] if sacct.strip() else "UNKNOWN"
            print(f"  [POLL] Final state: {state}")
            return state in ("COMPLETED", "COMPLETING", "UNKNOWN")

    print(f"\n  [POLL] Timed out after {timeout//3600}h.")
    return False


def download_scores(cfg: dict, stream: str, local_handoff: str) -> str | None:
    remote = f"{cfg['handoff_dir']}/dl_scores_{stream}.json"
    local  = str(Path(local_handoff) / f"dl_scores_{stream}.json")
    print("\n  [DOWNLOAD] Fetching scores...", end=" ", flush=True)
    if _scp_down(cfg, remote, local):
        print("OK")
        return local
    print("FAILED")
    return None


def _import_adequacy():
    try:
        return __import__("CCAD_check")
    except ImportError:
        print("adequacy checker not found.")
        sys.exit(1)


def _import_pipeline():
    try:
        return __import__("CCAD")
    except ImportError:
        print("ERROR: CCAD pipeline not found.")
        sys.exit(1)


#    Adequacy & statistical wrappers

def run_adequacy(df, sensor_cols, train_frac=0.75, min_corr=0.15):
    adeq = _import_adequacy()
    train_end    = int(len(df) * train_frac)
    rough_window = max(10, train_end // 500)

    print("\n[1/2] Checking sensor quality...")
    all_sq     = adeq.check_sensor_quality(df.iloc[:train_end], sensor_cols)
    valid_cols = [c for c, v in all_sq.items() if v["passed"]]
    print(f"      {len(valid_cols)}/{len(sensor_cols)} sensors passed")
    if len(valid_cols) < 2:
        return 0.0, None, []

    print("[2/2] Estimating period and grouping...")
    corr_all = adeq.multivariate_corr_signal(df.iloc[:train_end], valid_cols, rough_window)
    global_period, _, _, _ = adeq.estimate_period(corr_all)
    period_mat, conf_mat   = adeq.pairwise_periods(df, valid_cols, train_end, rough_window)

    from itertools import combinations
    df_tr         = df.iloc[:train_end]
    n             = len(valid_cols)
    mean_corr_mat = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        a  = df_tr[valid_cols[i]].dropna().values
        b  = df_tr[valid_cols[j]].dropna().values
        nb = min(len(a), len(b))
        if nb >= 10:
            mc = abs(np.corrcoef(a[:nb], b[:nb])[0, 1])
            if not np.isnan(mc):
                mean_corr_mat[i, j] = mean_corr_mat[j, i] = mc

    group_indices, _ = adeq.group_sensors_by_period(
        valid_cols, period_mat, conf_mat, mean_corr_mat, min_corr=min_corr)

    group_results = [
        adeq.assess_group(df, [valid_cols[i] for i in idx],
                           train_end, rough_window, period_hint=global_period)
        for idx in group_indices
    ]
    if not group_results:
        return 0.0, None, []

    group_results.sort(key=lambda x: x["overall"], reverse=True)
    best = group_results[0]
    return float(best["overall"]), best, group_results


def run_statistical(df, sensor_cols, period=None):
    pipe         = _import_pipeline()
    normal_end   = int(len(df) * 0.75)
    auto_win_est = max(10, normal_end // 500)
    corr_est     = pipe.correlation_strength(df.iloc[:normal_end], sensor_cols, auto_win_est)

    if period is None:
        period, _ = pipe.estimate_period_from_corr(corr_est)
        if period is None:
            period = 50
    print(f"  Period = {period} samples")
    return pipe.run_pipeline_for_group(df, sensor_cols, period, group_id=period)


# Main 
def _resolve_dl_threshold(dl_data: dict) -> float | None:
    raw_thr = dl_data.get("dl_threshold")
    if raw_thr is None:
        return None
    floor = float(np.percentile(dl_data["test_scores"], 95))
    resolved = float(max(raw_thr, floor))
    print(f"  dl_threshold: raw={raw_thr:.6f}  p95_floor={floor:.6f}  using={resolved:.6f}")
    return resolved



def main():
    parser = argparse.ArgumentParser(description="Hybrid anomaly detection pipeline")
    parser.add_argument("--setup",         action="store_true",
                        help="Interactive mesocentre SSH setup")
    parser.add_argument("--csv",           default=None)
    parser.add_argument("--stream",        default=None)
    parser.add_argument("--drop",          nargs="*", default=[])
    parser.add_argument("--timestamp",     default="timestamp")
    parser.add_argument("--dl_scores",     default=None,
                        help="Path to existing DL scores JSON — skips SSH entirely")
    parser.add_argument("--force_route",   default=None,
                        choices=["statistical", "hybrid", "dl"])
    parser.add_argument("--stat_thr",      type=float, default=STAT_THR)
    parser.add_argument("--hybrid_thr",    type=float, default=HYBRID_THR)
    parser.add_argument("--fuse_thr",      type=float, default=0.5)
    parser.add_argument("--train_frac",    type=float, default=0.75)
    parser.add_argument("--results_dir",   default="results")
    parser.add_argument("--handoff_dir",   default="handoff")
    parser.add_argument("--skip_adequacy", action="store_true")
    parser.add_argument("--no_ssh",        action="store_true",
                        help="Print manual instructions instead of using SSH")
    args = parser.parse_args()

    if args.setup:
        setup_wizard()
        return

    if args.csv is None or args.stream is None:
        parser.error("--csv and --stream are required")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.handoff_dir).mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    # Load data
    print(f"\n{'='*65}")
    print(f"  HYBRID PIPELINE  —  {args.stream}")
    print(f"{'='*65}")
    df = pd.read_csv(args.csv)
    drop_cols = [c for c in list(args.drop) + [args.timestamp] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  Dropped: {drop_cols}")
    sensor_cols = list(df.columns)
    print(f"  {df.shape[0]} rows × {len(sensor_cols)} sensors")

    # Adequacy
    adequacy_score = None
    adequacy_result = None
    if not args.skip_adequacy:
        print(f"\n{'─'*65}")
        print("  ADEQUACY CHECK")
        print(f"{'─'*65}")
        adequacy_score, adequacy_result, _ = run_adequacy(
            df, sensor_cols, train_frac=args.train_frac)
        print(f"\n  Best group adequacy score: {adequacy_score:.3f}")
        if adequacy_result:
            save_adequacy_for_remote(adequacy_result, args.handoff_dir, args.stream)
    else:
        if not args.force_route:
            parser.error("--skip_adequacy requires --force_route")
        adequacy_score = {"statistical":1.0,"hybrid":0.62,"dl":0.3}[args.force_route]

    # Routing
    print(f"\n{'─'*65}")
    print("  ROUTING")
    print(f"{'─'*65}")
    decision = route(args.stream, adequacy_score,
                     stat_thr=args.stat_thr, hybrid_thr=args.hybrid_thr)
    if args.force_route:
        decision.route = args.force_route
        decision.stat_weight = {"statistical":1.0,"hybrid":decision.stat_weight,"dl":0.0}[args.force_route]
        decision.dl_weight   = 1.0 - decision.stat_weight
    print_decision(decision)
    decision.save(Path(args.results_dir) / f"routing_{args.stream}.json")

    # Statistical
    stat_out = None
    period   = adequacy_result["period"] if adequacy_result else None
    if decision.route in ("statistical", "hybrid"):
        print(f"\n{'─'*65}")
        print("  STATISTICAL PIPELINE  (CCAD)")
        print(f"{'─'*65}")
        stat_out = run_statistical(df, sensor_cols, period=period)
        if stat_out is None:
            print("  WARNING: statistical pipeline returned None.")
            if decision.route == "statistical":
                sys.exit(0)
            print("  Falling back to DL-only.")
            decision.route = "dl"; decision.stat_weight = 0.0; decision.dl_weight = 1.0
        else:
            save_stat_output(stat_out, args.results_dir, args.stream)
            pipe = _import_pipeline()
            pipe.print_group_report(stat_out)
            pipe.visualise_group(stat_out,
                save_dir=str(Path(args.results_dir) / "plots" / "ccad"))

    # DL scores
    dl_data = None
    if decision.route in ("hybrid", "dl"):

        # --dl_scores flag
        if args.dl_scores and Path(args.dl_scores).exists():
            print(f"\n  Loading DL scores from {args.dl_scores}")
            d = json.loads(Path(args.dl_scores).read_text())
            d["train_scores"] = np.array(d["train_scores"])
            d["test_scores"]  = np.array(d["test_scores"])
            dl_data = d

        # already downloaded in handoff/
        elif (p := Path(args.handoff_dir) / f"dl_scores_{args.stream}.json").exists():
            print(f"\n  Found existing DL scores at {p}")
            d = json.loads(p.read_text())
            d["train_scores"] = np.array(d["train_scores"])
            d["test_scores"]  = np.array(d["test_scores"])
            dl_data = d

        # SSH → submit → poll → download
        elif not args.no_ssh:
            print(f"\n{'─'*65}")
            print("  MESOCENTRE  —  submitting MTAD-GAT job")
            print(f"{'─'*65}")

            if not prepare_and_upload_data(df, sensor_cols, cfg,
                                            args.handoff_dir, args.train_frac):
                print("  Data upload failed. Check SSH config (--setup).")
                sys.exit(1)

            job_id = submit_job(cfg, args.stream, args.handoff_dir)
            if not job_id:
                print("  Job submission failed.")
                sys.exit(1)

            if not poll_job(cfg, job_id):
                sys.exit(0)

            local_scores = download_scores(cfg, args.stream, args.handoff_dir)
            if local_scores and Path(local_scores).exists():
                d = json.loads(Path(local_scores).read_text())
                d["train_scores"] = np.array(d["train_scores"])
                d["test_scores"]  = np.array(d["test_scores"])
                dl_data = d
            else:
                print("  Download failed. Run manually then re-run with --dl_scores.")
                sys.exit(1)

        # --no_ssh
        else:
            script = SLURM_TEMPLATE.format(
                stream=args.stream, remote_dir=cfg["remote_dir"],
                handoff_dir=cfg["handoff_dir"], conda_env=cfg["conda_env"],
                user=cfg["user"])
            Path(args.handoff_dir, f"run_{args.stream}.sh").write_bytes(script.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
            prepare_and_upload_data.__doc__
            # prepare pkl files locally for manual upload
            import pickle
            data  = df[sensor_cols].values.astype("float32")
            split = int(len(data) * args.train_frac)
            for obj, name in [(data[:split], "CUSTOM_train.pkl"),
                               (data[split:], "CUSTOM_test.pkl"),
                               (np.zeros(len(data)-split,"float32"), "CUSTOM_test_label.pkl")]:
                with open(Path(args.handoff_dir)/name, "wb") as f:
                    pickle.dump(obj, f)
            print(f"  pkl files written to {args.handoff_dir}/ for manual upload.")
            if decision.route == "dl":
                sys.exit(0)
            print("  Falling back to statistical-only.")
            decision.route = "statistical"; decision.stat_weight = 1.0; decision.dl_weight = 0.0

    # Fusion
    print(f"\n{'─'*65}")
    print(f"  FUSION  (route={decision.route.upper()})")
    print(f"{'─'*65}")

    all_cycles  = stat_out["all_cycles"] if stat_out else []
    stat_scores = build_stat_scores_array(stat_out) if stat_out else np.full(len(all_cycles), np.nan)

    n_total = len(df)
    if dl_data is not None:
        dl_full = np.full(n_total, np.nan)
        ts = dl_data["test_scores"]
        dl_full[n_total - len(ts):] = ts
        dl_threshold = _resolve_dl_threshold(dl_data)
    else:
        dl_full = np.zeros(n_total)
        dl_threshold = None

    stat_flagged    = stat_out["final_anomalies"]      if stat_out else set()
    template_cycles = stat_out["all_template_indices"] if stat_out else set()
    dl_flagged      = (dl_flagged_from_scores(dl_full, all_cycles, dl_threshold)
                       if dl_threshold is not None and all_cycles else set())

    fuser = HybridFuser(stat_weight=decision.stat_weight, threshold=args.fuse_thr)
    if stat_out:
        train_end    = stat_out["train_end"]
        tr_idx       = [i for i, (s, e, _) in enumerate(all_cycles) if e <= train_end]
        stat_train   = stat_scores[tr_idx] if tr_idx else stat_scores
        dl_for_fit = dl_full[~np.isnan(dl_full)]  # use full stream, not just train
        dl_for_fit = dl_for_fit if len(dl_for_fit) else np.zeros(1)
        fuser.fit(stat_train, dl_for_fit)

    fusion_report = fuser.fuse(
        stat_scores=stat_scores, dl_ts_scores=dl_full, cycles=all_cycles,
        stat_flagged=stat_flagged, dl_flagged=dl_flagged,
        template_cycles=template_cycles, stream=args.stream)

    from hybrid_fuser import print_fusion_summary
    print_fusion_summary(fusion_report)

    # Reports
    print(f"\n{'─'*65}")
    print("  GENERATING REPORTS")
    print(f"{'─'*65}")
    outputs = merge_and_report(
        fusion_report  = fusion_report,
        stat_output    = load_stat_output(args.results_dir, args.stream) if stat_out else None,
        dl_ts_scores   = dl_full if dl_data is not None else None,
        corr_signal    = stat_out["corr"].values if stat_out else None,
        stream_name    = args.stream,
        results_dir    = args.results_dir,
    )

    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE  —  {args.stream}")
    print(f"{'='*65}")
    print(f"  Route      : {decision.route.upper()}")
    if adequacy_score is not None:
        print(f"  Adequacy   : {adequacy_score:.3f}")
    print(f"  Anomalies  : {fusion_report.n_anomalies} / {fusion_report.n_cycles} cycles")
    print(f"\n  Output files:")
    for k, v in outputs.items():
        if v:
            print(f"    {k:<10} {v}")
    print()


if __name__ == "__main__":
    main()