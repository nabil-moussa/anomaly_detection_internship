import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import os


#df = pd.read_csv("D:\\MaFaulDa\\streams\\stream_C.csv")
#df = df.drop(columns=['tachometer','microphone','overhang_axial','overhang_tangential','overhang_radial'])

df = pd.read_csv(r"C:\Users\user\Downloads\ccad_drift_test (1).csv")
#df = df[['P-3','T-4']]
####
# import numpy as np
# import pandas as pd
# import json

# SC        = 'SMAP'
# DATA_ROOT = r"C:\Users\user\OneDrive\Desktop\Fall 2025\Deep Learning\Demo_NLP\data"

# with open(f'{DATA_ROOT}/{SC}/split_info_{SC}.json', 'r') as f:
#     split_info = json.load(f)

# sensor_cols = list(split_info.keys())

# series_dict = {}
# for col in sensor_cols:
#     tr = np.load(f'{DATA_ROOT}/train/{col}.npy')
#     te = np.load(f'{DATA_ROOT}/test/{col}.npy')
#     tr = tr[:, 0] if tr.ndim > 1 else tr
#     te = te[:, 0] if te.ndim > 1 else te
#     series_dict[col] = np.concatenate([tr, te])

# max_len = max(len(v) for v in series_dict.values())
# for col in series_dict:
#     arr = series_dict[col]
#     if len(arr) < max_len:
#         series_dict[col] = np.concatenate([arr, np.full(max_len - len(arr), np.nan)])

# df = pd.DataFrame(series_dict)
# df=df[['E-10', 'E-11', 'E-12','E-1','E-2']]
# print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} sensors")
# print(f"Sensors: {sensor_cols[:10]}{'...' if len(sensor_cols)>10 else ''}")
####
sensor_cols = list(df.columns)
sensor_cols = [c for c in sensor_cols if c != 'timestamp']

print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} sensors")
print(f"Sensors: {sensor_cols[:10]}{'...' if len(sensor_cols)>10 else ''}")

def run_cusum_full_stream(all_results_sorted, loo_scores, loo_mean, threshold,
                          train_end_cycle_idx):
    
    raw_scores = np.array([r['max_score'] for r in all_results_sorted], dtype=float)
    
    loo_std = float(np.std(loo_scores)) if len(loo_scores) >= 3 else 0.0
    score_range = max(float(np.max(raw_scores)) - float(np.min(raw_scores)), 1e-4)
    loo_std = max(loo_std, score_range * 0.05, loo_mean * 0.10, 1e-4)
    
    target    = 0.0
    slack     = 0.5
    cusum_thr = 3.0
    
    print(f"      CUSUM: loo_mean={loo_mean:.4f}  loo_std={loo_std:.4f}  "
          f"score_range={score_range:.4f}  norm_thr={cusum_thr:.1f}")

    norm_scores = (raw_scores - loo_mean) / loo_std
    norm_scores = np.clip(norm_scores, 0, None)

    warmup = 3
    S = 0.0
    cusum_vals, alarms = [], []
    
    for i, sc in enumerate(norm_scores):
        dev = sc - target - slack
        S   = max(0.0, S + dev)
        
        alarm = (S > cusum_thr and i >= warmup)
        
        cusum_vals.append(S) 
        alarms.append(alarm)
        
        if alarm:
            S = 0.0

    for r, sv, alarm, ns in zip(all_results_sorted, cusum_vals, alarms, norm_scores):
        frac = sv / (cusum_thr + 1e-9)
        r['cusum_value']      = sv
        r['cusum_alarm']      = alarm
        r['cusum_thr']        = cusum_thr
        r['cusum_norm_score'] = float(ns)
        r['cusum_state'] = ('alarm'        if alarm      else
                            'accumulating' if frac > 0.5 else
                            'rising'       if frac > 0.1 else
                            'normal')

    return cusum_thr, cusum_vals, alarms

def correlation_strength(df, cols, window):
    out = []
    for i in range(len(df)):
        if i < window:
            out.append(np.nan); continue
        sub = df[cols].iloc[i-window:i]
        nc  = [c for c in cols if sub[c].std() > 1e-9]
        if len(nc) < 2:
            out.append(np.nan); continue
        cm = sub[nc].corr().values
        ut = cm[np.triu_indices_from(cm, k=1)]
        vp = ut[~np.isnan(ut)]
        out.append(np.mean(np.abs(vp)) if len(vp) else np.nan)
    return pd.Series(out, index=df.index)


def normalize_cycle(signal, start, end, target_length):
    cycle = signal.iloc[start:end+1].values
    cycle = cycle[~np.isnan(cycle)]
    if len(cycle) < 5:
        return None
    return np.interp(np.linspace(0,1,target_length), np.linspace(0,1,len(cycle)), cycle)


def _detect_nan_boundaries(signal, min_points):
    is_nan = signal.isna()
    starts = np.where((~is_nan) & is_nan.shift(1, fill_value=True))[0]
    ends   = np.where(is_nan & (~is_nan.shift(1, fill_value=True)))[0]
    if not len(starts) or not len(ends):
        return []
    if not is_nan.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if not is_nan.iloc[-1]:
        ends = np.append(ends, len(signal)-1)
    n = min(len(starts), len(ends))
    return [(s, e, e-s) for s, e in zip(starts[:n], ends[:n]) if e-s > min_points]


def detect_cycles_consistent_nan_gaps(signal, min_points=5, tolerance=0.4):
    is_nan = signal.isna()
    in_gap, gap_start, gaps = False, None, []
    for i, v in enumerate(is_nan):
        if v and not in_gap:
            in_gap, gap_start = True, i
        elif not v and in_gap:
            in_gap = False
            gaps.append((gap_start, i-1, i-gap_start))
    if in_gap:
        gaps.append((gap_start, len(signal)-1, len(signal)-gap_start))
    if not gaps:
        return []
    gl = np.array([g[2] for g in gaps])
    counts, edges = np.histogram(gl, bins=max(5, len(gl)//2))
    mid = (edges[np.argmax(counts)] + edges[np.argmax(counts)+1]) / 2
    lo, hi = mid*(1-tolerance), mid*(1+tolerance)
    cs = signal.copy()
    for gs, ge, g_ in gaps:
        if not (lo <= g_ <= hi):
            cs[signal.index[gs:ge+1]] = np.nan
            cs = cs.interpolate(method='linear')
    return _detect_nan_boundaries(cs, min_points)


def detect_cycles_energy_troughs(raw_df, cols, signal, min_points=5,
                                  smoothing=10, trough_pct=20):
    energy = raw_df[cols].abs().sum(axis=1).rolling(smoothing, center=True).mean()
    thr    = np.percentile(energy.dropna(), trough_pct)
    syn    = energy.copy().astype(float)
    syn[energy < thr] = np.nan
    return _detect_nan_boundaries(syn.reindex(signal.index), min_points)


def detect_cycles_autocorrelation(signal, min_points=5):
    clean = signal.dropna().values
    ac    = np.correlate(clean - clean.mean(), clean - clean.mean(), mode='full')
    ac    = ac[len(ac)//2:]
    ac   /= (ac[0] + 1e-9)
    peaks, _ = find_peaks(ac, height=0.3, distance=min_points)
    if not len(peaks):
        return []
    period = int(peaks[0])
    return [(i, i+period, period) for i in range(0, len(signal.values)-period, period)]


def segmentation_quality(cycles, signal):
    if len(cycles) < 3:
        return float('inf')
    lengths = [l for _,_,l in cycles]
    cv = np.std(lengths) / (np.mean(lengths) + 1e-9)
    if cv > 0.8:
        return float('inf')
    tl  = min(100, int(np.median(lengths)))
    smp = [normalize_cycle(signal, s, e, tl) for s,e,_ in cycles[:10]]
    smp = [x for x in smp if x is not None]
    #if len(smp) < 3:
    #    return float('inf')
    arr = np.array(smp)
    med = np.median(arr, axis=0)
    spread = np.mean([np.linalg.norm(c-med) for c in arr]) / (np.std(signal.dropna()) + 1e-9)
    return spread + cv*2


def detect_cycles_with_fallback(signal, raw_df, cols, min_points=5,
                                 forced_period=None):
    candidates = []
    for label, cycles in [
        ("NaN boundary",        _detect_nan_boundaries(signal, min_points)),
        ("Consistent NaN gaps", detect_cycles_consistent_nan_gaps(signal, min_points)),
        ("Energy troughs",      detect_cycles_energy_troughs(raw_df, cols, signal, min_points)),
        ("Autocorrelation",     detect_cycles_autocorrelation(signal, min_points)),
    ]:
        q = segmentation_quality(cycles, signal)
        print(f"    [{label}] {len(cycles)} cycles | quality={q:.3f}")
        candidates.append((q, cycles, label))

    if forced_period is not None and forced_period > min_points:
        stride_cycles = [(s, s + forced_period - 1, forced_period)
                         for s in range(0, len(signal) - forced_period, forced_period)]
        q_stride = segmentation_quality(stride_cycles, signal)
        print(f"    [Fixed stride p={forced_period}] {len(stride_cycles)} cycles | quality={q_stride:.3f}")
        candidates.append((q_stride, stride_cycles, f"Fixed stride p={forced_period}"))

    candidates.sort(key=lambda x: x[0])

    # Pick best method that has at least 3 cycles
    for bq, bc, bn in candidates:
        if len(bc) >= 3:
            print(f"    ✓ Best: {bn} | quality={bq:.3f} | {len(bc)} cycles")
            return bc, bn

    raise RuntimeError("All segmentation methods failed and no period available for fallback.")


def cluster_cycles_into_templates(all_normalised, all_cycles, signal, max_k=5, min_sil=0.15, min_size=1):
    valid = [(i, nc) for i, nc in enumerate(all_normalised) if nc is not None]
    if len(valid) < 3:
        return {0: [p[0] for p in valid]}, set()
    indices, arrays = zip(*valid)
    X = np.array(arrays)
    if len(X) < 4:
        return {0: list(indices)}, set()
    best_k, best_s = 1, -1
    for k in range(2, min(max_k+1, len(X))):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        s   = silhouette_score(X, lbl)
        print(f"      k={k}: silhouette={s:.3f}")
        if s > best_s:
            best_s, best_k = s, k
    if best_s < min_sil:
        best_k = 1
    if best_k == 1:
        return {0: list(indices)}, set()
    km  = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    lbl = km.fit_predict(X)
    clusters = {}
    for l, i in zip(lbl, indices):
        clusters.setdefault(int(l), []).append(i)
    return clusters, set()


def select_template_cycles(all_cycles, signal, target_length, candidate_indices,
                            keep_frac=0.30):
    norm, raw, vidx = [], [], []
    for i in candidate_indices:
        s, e, _ = all_cycles[i]
        nc = normalize_cycle(signal, s, e, target_length)
        raw_vals = signal.iloc[s:e+1].dropna().values
        if nc is not None and len(raw_vals) >= 5:
            norm.append(nc)
            vidx.append(i)
            raw.append(np.interp(np.linspace(0,1,target_length),
                                 np.linspace(0,1,len(raw_vals)), raw_vals))
    if len(norm) < 3:
        raise ValueError("Not enough valid cycles for template.")

    arr_raw = np.array(raw)  

    from sklearn.metrics import pairwise_distances
    D        = pairwise_distances(arr_raw, metric='euclidean')
    mean_dist = D.mean(axis=1)

    n_keep   = max(3, int(len(norm) * keep_frac))
    keep_idx = np.argsort(mean_dist)[:n_keep]

    return np.array(norm)[keep_idx], [vidx[i] for i in keep_idx]    

def build_template(template_cycles, window_size):
    tl  = template_cycles.shape[1]
    nw  = tl // window_size
    tmpl = {'mean_of_means': [], 'std_of_means': [],
            'mean_of_stds':  [], 'std_of_stds':  [],
            'n_windows': nw, 'window_size': window_size, 'target_length': tl}
    for w in range(nw):
        s, e = w*window_size, (w+1)*window_size
        wm = [np.mean(c[s:e]) for c in template_cycles]
        ws = [np.std(c[s:e])  for c in template_cycles]
        tmpl['mean_of_means'].append(np.mean(wm))
        tmpl['std_of_means'].append(np.std(wm))
        tmpl['mean_of_stds'].append(np.mean(ws))
        tmpl['std_of_stds'].append(np.std(ws))
    for k in ['mean_of_means','std_of_means','mean_of_stds','std_of_stds']:
        tmpl[k] = np.array(tmpl[k])
    return tmpl


def align_cycle_to_template(cycle, template_median, max_shift_frac=0.10):
    ms   = max(1, int(len(cycle)*max_shift_frac))
    corr = np.correlate(cycle-cycle.mean(), template_median-template_median.mean(), mode='full')
    ctr  = len(cycle)-1
    srch = corr[ctr-ms:ctr+ms+1]
    bs   = np.argmax(srch) - ms
    if bs > 0:
        aligned = np.concatenate([cycle[bs:], cycle[-bs:]])
    elif bs < 0:
        aligned = np.concatenate([cycle[:bs], cycle[-bs:]])
    else:
        aligned = cycle.copy()
    return aligned, bs


def score_single_cycle(nc, tmpl, template_median=None, max_shift_frac=0.10):
    eps, shift = 1e-9, 0
    if template_median is not None:
        nc, shift = align_cycle_to_template(nc, template_median, max_shift_frac)
    scores, details = [], []
    for w in range(tmpl['n_windows']):
        s, e = w*tmpl['window_size'], (w+1)*tmpl['window_size']
        wd   = nc[s:e]
        om, os_ = np.mean(wd), np.std(wd)

        mean_range = tmpl['mean_of_means'].max() - tmpl['mean_of_means'].min()
        std_floor  = max(tmpl['std_of_means'][w],
                         mean_range * 0.02,   # 2% of mean range
                         eps)
        std_floor2 = max(tmpl['std_of_stds'][w],
                         tmpl['mean_of_stds'][w] * 0.10,  # 10% of expected std
                         eps)

        zm   = abs(om  - tmpl['mean_of_means'][w]) / std_floor
        zs   = abs(os_ - tmpl['mean_of_stds'][w])  / std_floor2
        comb = (zm + zs) / 2.0
        scores.append(comb)
        details.append({
            'window_idx': w, 'obs_mean': om, 'obs_std': os_,
            'exp_mean': tmpl['mean_of_means'][w],
            'exp_std':  tmpl['mean_of_stds'][w],
            'z_mean': zm, 'z_std': zs, 'combined': comb
        })
    return np.array(scores), details, shift


def calibrate_threshold(template_cycles, tmpl, template_median,
                        percentile=99, iqr_k=2, min_n_for_iqr=8):
    max_scores = []
    for i in range(len(template_cycles)):
        loo = np.delete(template_cycles, i, axis=0)
        if len(loo) < 2:
            continue
        lt  = build_template(loo, tmpl['window_size'])
        lm  = np.median(loo, axis=0)
        s, _, _ = score_single_cycle(template_cycles[i], lt, lm)
        max_scores.append(np.max(s))

    if not max_scores:
        return 3.0, [], float(np.mean(max_scores) if max_scores else 1.0)

    if len(max_scores) >= min_n_for_iqr:
        q1, q3 = np.percentile(max_scores, [25, 75])
        iqr    = q3 - q1
        thr    = q3 + iqr_k * iqr
        print(f"      Threshold (IQR k={iqr_k}): {thr:.3f} "
              f"| Q1={q1:.3f} Q3={q3:.3f} IQR={iqr:.3f} n={len(max_scores)}")
    else:
        thr = np.percentile(max_scores, percentile)
        print(f"      Threshold ({percentile}th pct, n={len(max_scores)}): {thr:.3f}")

    return thr, max_scores, float(np.mean(max_scores))


def pca_amplitude_detection(raw_df, cols, all_cycles, template_global_idx_set,
                              normal_cycle_indices, target_length=100,
                              ev_target=0.75, thr_pct=95):
    def extract_norm(ci):
        s, e, _ = all_cycles[ci]
        seg = raw_df[cols].iloc[s:e]
        out = {}
        for col in cols:
            v = seg[col].dropna().values
            if len(v) < 5:
                out[col] = None
            else:
                out[col] = np.interp(np.linspace(0,1,target_length),
                                     np.linspace(0,1,len(v)), v)
        return out

    sensor_pcas = {}
    tmpl_data   = {c: [] for c in cols}
    for i in sorted(template_global_idx_set):
        nc = extract_norm(i)
        if all(nc[c] is not None for c in cols):
            for c in cols:
                tmpl_data[c].append(nc[c])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in cols:
            X = np.array(tmpl_data[c])
            if X.shape[0] < 3:
                continue
            mask = X.std(axis=0) >= 1e-9
            X    = X[:, mask]
            if X.shape[1] < 2:
                continue
            sc  = StandardScaler()
            Xs  = sc.fit_transform(X)
            n_c = min(int(ev_target * min(Xs.shape)), Xs.shape[0]-1, Xs.shape[1])
            if n_c < 1:
                continue
            pca = PCA(n_components=n_c)
            pca.fit(Xs)
            sensor_pcas[c] = (pca, sc, mask)

    if not sensor_pcas:
        print("    PCA: no valid sensors — skipping")
        return set(), None, [None]*len(all_cycles)

    mean_var = np.mean([sensor_pcas[c][0].explained_variance_ratio_.sum() for c in sensor_pcas])
    print(f"    PCA mean explained variance: {mean_var*100:.1f}%")

    def score_cycle(ci):
        nc = extract_norm(ci)
        errs = []
        for c in sensor_pcas:
            if nc[c] is None:
                continue
            pca, sc, mask = sensor_pcas[c]
            x   = nc[c][mask]
            if len(x) < 2:
                continue
            xs  = sc.transform(x.reshape(1,-1))
            proj = pca.inverse_transform(pca.transform(xs))
            errs.append(float(np.mean((xs-proj)**2)))
        return np.mean(errs) if errs else np.nan

    holdout = [i for i in normal_cycle_indices if i not in template_global_idx_set]
    h_errs  = [e for e in (score_cycle(i) for i in holdout) if not np.isnan(e)]
    if not h_errs:
        print("    PCA: no valid holdout cycles — skipping")
        return set(), None, [None]*len(all_cycles)

    print(f"    Holdout: n={len(h_errs)} | min={min(h_errs):.5f} mean={np.mean(h_errs):.5f} max={max(h_errs):.5f}")
    amp_thr = np.percentile(h_errs, thr_pct)
    print(f"    Amplitude threshold ({thr_pct}th pct): {amp_thr:.5f}")

    amp_scores, amp_anom = [], set()
    for i in range(len(all_cycles)):
        if i in template_global_idx_set:
            amp_scores.append(None); continue
        err = score_cycle(i)
        amp_scores.append(err)
        if not np.isnan(err) and err > amp_thr:
            amp_anom.add(i)
    print(f"    Amplitude anomalies: {len(amp_anom)}")
    return amp_anom, amp_thr, amp_scores


# Per-Group Pipeline

def run_pipeline_for_group(df, group_cols, group_period, group_id, min_cycles=3):
    print(f"\n{'='*65}")
    print(f"GROUP period={group_period}pts | {len(group_cols)} sensors")
    print(f"Sensors: {group_cols}")
    print('='*65)

    if len(group_cols) < 2:
        print("  ✗ Need ≥2 sensors"); return None

    train_end  = int(len(df) * 0.75)
    df_train   = df.iloc[:train_end]
    df_full    = df

    auto_window = max(10, group_period // 8)
    print(f"\n  STEP 2 – Correlation signal (window={auto_window})")
    corr_train = correlation_strength(df_train, group_cols, auto_window)
    print(f"    {len(corr_train)} pts | {corr_train.isna().sum()} NaN")

    if corr_train.dropna().std() < 1e-9:
        print("  ✗ Constant correlation signal"); return None

    print(f"\n  STEP 3 – Segmentation")
    try:
        train_cycles, seg_method = detect_cycles_with_fallback(
            corr_train, df_train, group_cols,
            forced_period=group_period)
    except RuntimeError as e:
        print(f"  ✗ {e}"); return None

    if len(train_cycles) < min_cycles:
        print(f"  ✗ Only {len(train_cycles)} cycles"); return None

    lengths          = [l for _,_,l in train_cycles]
    median_cycle_len = int(np.median(lengths))
    target_length    = min(100, median_cycle_len)
    window_size      = max(3, target_length // 10)
    print(f"    cycles={len(train_cycles)} | median_len={median_cycle_len} | "
          f"target={target_length} | win_size={window_size}")

    print(f"\n  STEP 4 – Normalising {len(train_cycles)} training cycles")
    train_normalised = [normalize_cycle(corr_train, s, e, target_length)
                        for s,e,_ in train_cycles]
    valid_norm = sum(1 for n in train_normalised if n is not None)
    print(f"    {valid_norm}/{len(train_cycles)} valid")

    print(f"\n  STEP 5 – Clustering")
    clusters, structural_outliers = cluster_cycles_into_templates(
        train_normalised, train_cycles, corr_train)
    print(f"    {len(clusters)} cluster(s) | {len(structural_outliers)} structural outliers")

    print(f"\n  STEP 6 – Templates (window_size={window_size})")
    cluster_loo_means = {}
    cluster_templates, cluster_medians = {}, {}
    cluster_thresholds, cluster_loo_scores = {}, {}
    template_global_idx_sets = {}

    for cid, cycle_indices in clusters.items():
        print(f"    Cluster {cid} ({len(cycle_indices)} cycles):")
        try:
            tn, tgi = select_template_cycles(train_cycles, corr_train, target_length,cycle_indices, keep_frac=0.50)
        except ValueError as e:
            print(f"      ✗ {e}"); continue
        tmpl  = build_template(tn, window_size)
        med_c = np.median(tn, axis=0)
        thr, loo, loo_mean = calibrate_threshold(tn, tmpl, med_c)
        cluster_templates[cid]        = tmpl
        cluster_medians[cid]          = med_c
        cluster_thresholds[cid]       = thr
        cluster_loo_scores[cid]       = loo
        cluster_loo_means[cid]        = loo_mean

        template_global_idx_sets[cid] = set(tgi)

    if not cluster_templates:
        print("  ✗ No valid clusters"); return None

    train_template_idx_sets = {
        cid: set(idx_set)
        for cid, idx_set in template_global_idx_sets.items()
    }
    all_template_indices = set()
    for s in train_template_idx_sets.values():
        all_template_indices.update(s)
    print(f"\n  STEP 7 – Scoring test cycles (full stream)")
    corr_full = correlation_strength(df_full, group_cols, auto_window)

    test_period = int(np.median([l for _, _, l in train_cycles]))
    full_cycles = [(s, s + test_period - 1, test_period)
                for s in range(0, len(corr_full) - test_period, test_period)]

    all_template_indices = set()
    for idx_set in train_template_idx_sets.values():
        all_template_indices.update(idx_set)

    template_global_idx_sets = {
        cid: set(idx_set)
        for cid, idx_set in train_template_idx_sets.items()
    }

    all_normalised = [normalize_cycle(corr_full, s, e, target_length) for s, e, _ in full_cycles]

    valid_test = [i for i, (s, e, _) in enumerate(full_cycles)
                if i not in all_template_indices
                and all_normalised[i] is not None]

    valid_score = [i for i, (s, e, _) in enumerate(full_cycles)
                if i not in all_template_indices
                and all_normalised[i] is not None]

    valid_test  = [i for i in valid_score if full_cycles[i][0] >= train_end]
    valid_train_nontmpl = [i for i in valid_score if full_cycles[i][0] < train_end]
    print(f"    Template: {len(all_template_indices)} | "
        f"Train non-tmpl: {len(valid_train_nontmpl)} | "
        f"Test: {len(valid_test)}")

    results = []
    for vi in valid_score:
        nc       = all_normalised[vi]
        best_cid = min(cluster_medians,
                    key=lambda c: np.linalg.norm(nc - cluster_medians[c]))
        tmpl     = cluster_templates[best_cid]
        med_c    = cluster_medians[best_cid]
        thr      = cluster_thresholds[best_cid]
        ws, det, shift = score_single_cycle(nc, tmpl, med_c)
        min_anom = max(2, int(tmpl['n_windows'] * 0.25))
        above    = np.sum(ws > thr)
        max_s    = np.max(ws)
        is_anom  = (above >= min_anom) or (max_s > 2 * thr)
        results.append({
            'cycle_idx':            vi,
            'cluster_id':           best_cid,
            'is_anomaly':           is_anom,
            'max_score':            max_s,
            'mean_score':           np.mean(ws),
            'windows_above_thresh': int(above),
            'window_scores':        ws,
            'window_details':       det,
            'shift_applied':        shift,
            'threshold':            thr,
            'is_train':             full_cycles[vi][0] < train_end,
        })

    all_cycles     = full_cycles
    corr           = corr_full
    corr_anomalies = {r['cycle_idx'] for r in results if r['is_anomaly']}

#CUSUM over full stream
    print(f"\n  STEP 7b – CUSUM drift detection (full stream)")
    cusum_meta = {}
    # Sort ALL results by cycle index (train non-template + test)

    all_scored_dict = {r['cycle_idx']: r for r in results}

    all_scored = []
    for i, (s, e, l) in enumerate(all_cycles):
        if i in all_scored_dict:
            all_scored.append(all_scored_dict[i])
        elif i in all_template_indices:
            best_cid = min(cluster_medians,
                        key=lambda c: np.linalg.norm(
                            (all_normalised[i] if all_normalised[i] is not None 
                                else cluster_medians[c]) - cluster_medians[c]))
            synthetic = {
                'cycle_idx':   i,
                'cluster_id':  best_cid,
                'is_anomaly':  False,
                'max_score':   cluster_loo_means[best_cid],
                'mean_score':  cluster_loo_means[best_cid],
                'windows_above_thresh': 0,
                'window_scores': np.array([]),
                'window_details': [],
                'shift_applied': 0,
                'threshold': cluster_thresholds[best_cid],
                'is_template': True,
            }
            all_scored.append(synthetic)

    all_scored = sorted(all_scored, key=lambda r: r['cycle_idx'])

    train_end_cycle_idx = sum(
        1 for r in all_scored
        if all_cycles[r['cycle_idx']][0] < train_end
    )

    cusum_anomalies = set()
    for cid in cluster_templates:
        cluster_results = sorted(
            [r for r in all_scored if r['cluster_id'] == cid],
            key=lambda r: r['cycle_idx']
        )
        if not cluster_results:
            continue

        cid_train_end = sum(
            1 for r in cluster_results
            if all_cycles[r['cycle_idx']][0] < train_end
        )

        cusum_thr, cvals, calarms = run_cusum_full_stream(
            cluster_results,
            loo_scores           = cluster_loo_scores[cid],
            loo_mean             = cluster_loo_means[cid],
            threshold            = cluster_thresholds[cid],
            train_end_cycle_idx  = cid_train_end,
        )
        for r in cluster_results:
            cusum_meta[r['cycle_idx']] = {
                'cusum_value': r.get('cusum_value', 0.0),
                'cusum_alarm': r.get('cusum_alarm', False),
                'cusum_thr':   cusum_thr,
            }
            if r.get('cusum_alarm', False):
                cusum_anomalies.add(r['cycle_idx'])
        print(f"    Cluster {cid}: CUSUM thr={cusum_thr:.4f}  "
              f"alarms={sum(calarms)}  over {len(cluster_results)} cycles")

    # Spike detection: raw threshold crossed but CUSUM never alarmed and the cycle is in the training region
    above_thr = {r['cycle_idx'] for r in results 
                if r['max_score'] > cluster_thresholds[r['cluster_id']]}

    true_spikes = set()
    for ci in above_thr:
        neighbors_also_above = (
            (ci-1) in above_thr or (ci+1) in above_thr
        )
        is_in_train = all_cycles[ci][0] < train_end
        r_ci = next((r for r in results if r['cycle_idx'] == ci), None)

        cusum_alarm = r_ci.get('cusum_alarm', False) if r_ci else False

        if is_in_train and not neighbors_also_above and not cusum_alarm:
            true_spikes.add(ci)

    final_cusum = cusum_anomalies - true_spikes
    both_cusum  = corr_anomalies & cusum_anomalies

    print(f"\n  ── CUSUM Summary ────────────────────────────────────")
    print(f"    CUSUM alarms        : {len(cusum_anomalies)}")
    print(f"    Spikes ignored      : {len(true_spikes)}")
    print(f"    FINAL (CUSUM)       : {len(final_cusum)}")

####
    print(f"\n  STEP 8 – PCA amplitude detection")
    train_start_to_full = {}
    for ti, (ts, te, tl) in enumerate(train_cycles):
        best_fi = min(range(len(full_cycles)), key=lambda fi: abs(full_cycles[fi][0] - ts))
        train_start_to_full[ti] = best_fi

    normal_indices = list({train_start_to_full[i]
                        for cid in clusters
                        for i in clusters[cid]
                        if i in train_start_to_full})
    try:
        amp_anom, amp_thr, amp_scores = pca_amplitude_detection(
            df_full, group_cols, all_cycles, all_template_indices,
            normal_indices, target_length)
        amp_anom.update(structural_outliers)
    except Exception as e:
        print(f"    PCA failed: {e}")
        amp_anom, amp_thr, amp_scores = set(structural_outliers), None, [None]*len(all_cycles)

    final = corr_anomalies | amp_anom
    both  = corr_anomalies & amp_anom

    print(f"\n  ── Summary ──────────────────────────────────────────")
    print(f"    Correlation anomalies : {len(corr_anomalies)}")
    print(f"    Amplitude anomalies   : {len(amp_anom)}")
    print(f"    Both (high confidence): {len(both)}")
    print(f"    FINAL (union)         : {len(final)}")

    return {'train_end': train_end,'group_id': group_id, 'group_cols': group_cols, 'group_period': group_period,
            'seg_method': seg_method, 'corr': corr, 'all_cycles': all_cycles,
            'all_normalised': all_normalised, 'target_length': target_length,
            'window_size': window_size, 'clusters': clusters,
            'cluster_templates': cluster_templates, 'cluster_medians': cluster_medians,
            'cluster_thresholds': cluster_thresholds, 'cluster_loo_scores': cluster_loo_scores,
            'template_global_idx_sets': template_global_idx_sets,
            'all_template_indices': all_template_indices, 'results': results,
            'corr_anomalies': corr_anomalies, 'amp_anomalies': amp_anom,
            'amp_threshold': amp_thr, 'cycle_amp_scores': amp_scores,
            'final_anomalies': final, 'both_methods': both, 'cusum_anomalies':  final_cusum,
            'cusum_spikes':     true_spikes,
            'both_cusum':       both_cusum,
            'cusum_meta':       cusum_meta,'cluster_loo_means':  cluster_loo_means,
            'cluster_loo_scores': cluster_loo_scores,}


# VISUALISATION
CMAP_CLUSTERS = ['#2563EB', '#059669', '#7C3AED', '#D97706', '#DC2626']

def visualise_group(out, save_dir='group_plots'):
    os.makedirs(save_dir, exist_ok=True)
    gid    = out['group_id']
    corr   = out['corr']
    cycles = out['all_cycles']
    res    = out['results']
    clusters   = out['clusters']
    cl_tmpl    = out['cluster_templates']
    cl_med     = out['cluster_medians']
    cl_thr     = out['cluster_thresholds']
    cl_loo     = out['cluster_loo_scores']
    tmpl_idx   = out['template_global_idx_sets']
    all_tmpl   = out['all_template_indices']
    corr_anom  = out['corr_anomalies']
    amp_anom   = out['amp_anomalies']
    amp_scores = out['cycle_amp_scores']
    amp_thr    = out['amp_threshold']
    all_norm   = out['all_normalised']
    tl         = out['target_length']
    ws         = out['window_size']
    final      = out['final_anomalies']
    seg        = out['seg_method']
    n_cl       = len(clusters)
    figs_saved = []

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(corr.index, corr.values, color='#2563EB', lw=1.2, label='Correlation strength')
    y0, y1 = float(corr.dropna().min()), float(corr.dropna().max())
    yspan  = y1 - y0 or 1.0
    ax.set_ylim(y0 - 0.05*yspan, y1 + 0.18*yspan)

    #last_tmpl_end = max((cycles[j][1] for j in all_tmpl if j < len(cycles)), default=0) if all_tmpl else 0

    for i, (start, end, length) in enumerate(cycles):
        cycle_cluster = next((cid for cid, idxs in clusters.items() if i in idxs), None)
        in_corr = i in corr_anom
        in_amp  = i in amp_anom
        #in_tmpl = (i in all_tmpl) or (end <= last_tmpl_end and not in_corr and not in_amp)
        in_tmpl = i in all_tmpl

        in_c    = in_corr
        in_a    = in_amp
        xe      = min(end, len(corr)-1)

        if in_tmpl:
            col_fill   = '#BFDBFE'
            col_border = CMAP_CLUSTERS[cycle_cluster % len(CMAP_CLUSTERS)] if cycle_cluster is not None else '#6B7280'
            lw, alpha  = 1.0, 0.35
            tag = f"T{i+1}"
        elif in_c and in_a:
            col_fill, col_border, lw, alpha = '#FCA5A5', '#7F1D1D', 2.0, 0.55
            r   = next((r for r in res if r['cycle_idx']==i), None)
            tag = f"A{i+1}\nBOTH\n{r['max_score']:.1f}σ" if r else f"A{i+1}\nBOTH"
        elif in_c:
            col_fill, col_border, lw, alpha = '#FECACA', '#DC2626', 1.8, 0.45
            r   = next((r for r in res if r['cycle_idx']==i), None)
            tag = f"A{i+1}\nCORR\n{r['max_score']:.1f}σ" if r else f"A{i+1}\nCORR"
        elif in_a:
            col_fill, col_border, lw, alpha = '#FDE68A', '#D97706', 1.8, 0.45
            amp = amp_scores[i]
            tag = f"A{i+1}\nAMP\n{amp:.3f}" if amp is not None else f"A{i+1}\nAMP"
        else:
            col_fill, col_border, lw, alpha = '#D1FAE5', '#059669', 0.8, 0.25
            tag = f"N{i+1}"

        rect = mpatches.Rectangle((corr.index[start], y0), xe - start, yspan,
                                   facecolor=col_fill, edgecolor=col_border,
                                   linewidth=lw, alpha=alpha)
        ax.add_patch(rect)
        ax.text(corr.index[start], y1 + 0.01*yspan, tag,
                fontsize=6, va='bottom', ha='left', color=col_border, clip_on=True)

    if all_tmpl:
        # use train_end directly, not the last template cycle index
        train_end = out['train_end']
        ax.axvline(x=train_end, color='black', lw=1.5,
                ls='--', label='Template / Test split')

    handles = [
        mpatches.Patch(facecolor='#BFDBFE', edgecolor='#2563EB', label='Template cycle'),
        mpatches.Patch(facecolor='#D1FAE5', edgecolor='#059669', label='Normal test'),
        mpatches.Patch(facecolor='#FECACA', edgecolor='#DC2626', label='Corr. anomaly'),
        mpatches.Patch(facecolor='#FDE68A', edgecolor='#D97706', label='Amp. anomaly'),
        mpatches.Patch(facecolor='#FCA5A5', edgecolor='#7F1D1D', label='Both methods'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=5, framealpha=0.9)
    ax.set_xlabel('Sample index', fontsize=10)
    ax.set_ylabel('Correlation strength', fontsize=10)
    ax.set_title(
        f"Group period={gid}pts  |  Sensors: {', '.join(out['group_cols'][:8])}"
        f"{'...' if len(out['group_cols'])>8 else ''}\n"
        f"Seg: {seg}  |  {len(cycles)} cycles  |  "
        f"{len(final)} anomalies  (corr={len(corr_anom)}, amp={len(amp_anom)})",
        fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/group_{gid:04d}_fig1_signal.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    figs_saved.append(p)
    print(f"  Saved: {p}")

    nc_cids = list(cl_tmpl.keys())
    ncols   = len(nc_cids)
    fig, axes = plt.subplots(1, ncols, figsize=(9*ncols, 6), squeeze=False)

    for ax, cid in zip(axes[0], nc_cids):
        tmpl   = cl_tmpl[cid]
        med_c  = cl_med[cid]
        thr    = cl_thr[cid]
        t_norm = np.array([all_norm[i] for i in tmpl_idx[cid] if i < len(all_norm) and all_norm[i] is not None])
        x      = np.arange(tl)

        for nc in t_norm:
            ax.plot(x, nc, color='#93C5FD', lw=0.8, alpha=0.4)
        ax.plot(x, med_c, color='#1D4ED8', lw=2.5, label='Template median', zorder=5)

        xw = np.arange(tmpl['n_windows']) * ws + ws/2
        ax.plot(xw, tmpl['mean_of_means'], 'o--', color='#B91C1C', lw=1.5,
                ms=6, label='Window mean ± 2σ', zorder=4)
        ax.fill_between(xw,
                        tmpl['mean_of_means'] - 2*tmpl['std_of_means'],
                        tmpl['mean_of_means'] + 2*tmpl['std_of_means'],
                        color='#FCA5A5', alpha=0.25)

        shown_anom, shown_norm = False, False
        for r in res:
            if r['cluster_id'] != cid:
                continue
            vi = r['cycle_idx']
            nc = all_norm[vi]
            if nc is None:
                continue
            if r['is_anomaly']:
                label  = 'Anomalous' if not shown_anom else None
                ax.plot(x, nc, color='#DC2626', lw=1.8, alpha=0.85, label=label, zorder=6)
                shown_anom = True
            else:
                label  = 'Normal test' if not shown_norm else None
                ax.plot(x, nc, color='#34D399', lw=1.0, alpha=0.35, label=label, zorder=3)
                shown_norm = True

        ax.axhline(y=thr, color='orange', lw=1, ls=':', alpha=0.5)
        ax.set_title(f"Cluster {cid} — {len(clusters.get(cid,[]))} cycles\n"
                     f"threshold={thr:.3f}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Normalised position')
        ax.set_ylabel('Corr. strength')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.2)

    fig.suptitle(f"Group period={gid}pts — Normalised cycle overlays",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = f"{save_dir}/group_{gid:04d}_fig2_cycles.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    figs_saved.append(p)
    print(f"  Saved: {p}")

    ncols = len(cl_loo)
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 4), squeeze=False)
    for ax, cid in zip(axes[0], cl_loo.keys()):
        loo = cl_loo[cid]; thr = cl_thr[cid]
        ax.hist(loo, bins=20, color='#3B82F6', edgecolor='white', alpha=0.8)
        ax.axvline(thr, color='#DC2626', lw=2, ls='--', label=f'Threshold = {thr:.3f}')
        for r in res:
            if r['cluster_id'] != cid:
                continue
            color = '#DC2626' if r['is_anomaly'] else '#059669'
            ax.axvline(r['max_score'], color=color, lw=1.0, alpha=0.6, ls='-')
        ax.set_title(f"Cluster {cid} — LOO calibration", fontweight='bold')
        ax.set_xlabel('Max window score')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Group period={gid}pts — Threshold calibration",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = f"{save_dir}/group_{gid:04d}_fig3_loo.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    figs_saved.append(p)
    print(f"  Saved: {p}")

    valid_amp = [(i, s) for i,s in enumerate(amp_scores)
                 if s is not None and not np.isnan(s)]
    if valid_amp and amp_thr is not None:
        idxs, scores_ = zip(*valid_amp)
        colors = ['#DC2626' if i in amp_anom else '#60A5FA' for i in idxs]
        fig, ax = plt.subplots(figsize=(max(10, len(idxs)//3), 4))
        ax.bar(range(len(idxs)), scores_, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(amp_thr, color='#DC2626', lw=2, ls='--', label=f'Threshold = {amp_thr:.4f}')
        ax.set_xticks(range(len(idxs)))
        ax.set_xticklabels([f"C{i+1}" for i in idxs], rotation=60, fontsize=7)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('PCA reconstruction error')
        ax.set_title(f"Group period={gid}pts — PCA amplitude scores per cycle", fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for xi, (ci, sc) in enumerate(zip(idxs, scores_)):
            if ci in amp_anom:
                ax.text(xi, sc, f'  {sc:.3f}', va='bottom', ha='left',
                        color='#7F1D1D', fontsize=7, rotation=30)
        plt.tight_layout()
        p = f"{save_dir}/group_{gid:04d}_fig4_amplitude.png"
        fig.savefig(p, dpi=130, bbox_inches='tight')
        plt.close(fig)
        figs_saved.append(p)
        print(f"  Saved: {p}")

    anom_res = [r for r in res if r['is_anomaly']]
    if anom_res:
        nw_max  = max(r['window_scores'].shape[0] for r in anom_res)
        labels  = [f"C{r['cycle_idx']+1}" for r in anom_res]
        matrix  = np.array([np.pad(r['window_scores'], (0, nw_max - len(r['window_scores'])))
                             for r in anom_res])

        fig, ax = plt.subplots(figsize=(max(8, nw_max), max(3, len(anom_res)*0.5 + 1.5)))
        cmap    = LinearSegmentedColormap.from_list('anomaly',
                    ['#DBEAFE','#93C5FD','#EF4444','#7F1D1D'])
        im      = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0)
        plt.colorbar(im, ax=ax, label='Z-score')
        ax.set_xticks(range(nw_max))
        ax.set_xticklabels([f"Win {w}" for w in range(nw_max)], fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        for yi, r in enumerate(anom_res):
            thr = r['threshold']
            for xi, sc in enumerate(r['window_scores']):
                if sc > thr:
                    ax.add_patch(mpatches.Rectangle((xi-0.5, yi-0.5), 1, 1,
                                 fill=False, edgecolor='black', lw=2))
                ax.text(xi, yi, f"{sc:.1f}", ha='center', va='center',
                        fontsize=7, color='white' if sc > thr else 'black')
        ax.set_title(f"Group period={gid}pts — Window scores for anomalous cycles\n"
                     f"(black border = exceeds threshold)", fontweight='bold')
        plt.tight_layout()
        p = f"{save_dir}/group_{gid:04d}_fig5_heatmap.png"
        fig.savefig(p, dpi=130, bbox_inches='tight')
        plt.close(fig)
        figs_saved.append(p)
        print(f"  Saved: {p}")

    for r in anom_res:
        vi  = r['cycle_idx']
        cid = r['cluster_id']
        nc  = all_norm[vi]
        if nc is None:
            continue
        tmpl  = cl_tmpl[cid]
        med_c = cl_med[cid]
        thr   = cl_thr[cid]
        nw    = tmpl['n_windows']
        x     = np.arange(tl)
        xw    = np.arange(nw) * ws + ws/2

        fig, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        ax  = axes2[0]
        ax.fill_between(x,
                        np.interp(x, xw, tmpl['mean_of_means'] - 2*tmpl['std_of_means']),
                        np.interp(x, xw, tmpl['mean_of_means'] + 2*tmpl['std_of_means']),
                        color='#BFDBFE', alpha=0.5, label='Template ±2σ band')
        ax.plot(x, med_c, color='#1D4ED8', lw=2, label='Template median', zorder=4)
        ax.plot(x, nc, color='#DC2626', lw=2, ls='--',
                label=f'Cycle {vi+1} (anomalous)', zorder=5)
        s_c, e_c, _ = cycles[vi]
        det_type = 'BOTH' if (vi in corr_anom and vi in amp_anom) else \
                   ('CORR' if vi in corr_anom else 'AMP')
        ax.set_title(f"Cycle {vi+1}  [{det_type}]  idx={s_c}→{e_c}\n"
                     f"max={r['max_score']:.2f}σ  shift={r['shift_applied']}", fontweight='bold')
        ax.set_xlabel('Normalised position')
        ax.set_ylabel('Corr. strength')
        ax.legend(fontsize=9); ax.grid(alpha=0.2)

        ax2 = axes2[1]
        dets = r['window_details']
        x_w  = np.arange(len(dets))
        wz   = [d['z_mean'] for d in dets]
        sz   = [d['z_std']  for d in dets]
        comb = [d['combined'] for d in dets]
        bar_colors = ['#DC2626' if c > thr else '#60A5FA' for c in comb]
        ax2.bar(x_w - 0.25, wz, 0.25, color='#3B82F6', alpha=0.8, label='z_mean')
        ax2.bar(x_w,        sz, 0.25, color='#8B5CF6', alpha=0.8, label='z_std')
        ax2.bar(x_w + 0.25, comb, 0.25, color=bar_colors, alpha=0.9, label='combined')
        ax2.axhline(thr, color='#DC2626', lw=1.5, ls='--', label=f'Threshold={thr:.2f}')
        ax2.axhline(2*thr, color='#7F1D1D', lw=1, ls=':', alpha=0.6, label=f'2×Thr={2*thr:.2f}')
        ax2.set_xticks(x_w)
        ax2.set_xticklabels([f"W{d['window_idx']}\n{d['window_idx']*ws}-{(d['window_idx']+1)*ws}"
                             for d in dets], fontsize=8)
        ax2.set_ylabel('Z-score')
        ax2.set_title(f"Window-level z-scores | cluster={cid}", fontweight='bold')
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(axis='y', alpha=0.2)
        fig.suptitle(f"Group period={gid}pts — Anomaly detail: Cycle {vi+1}",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        p = f"{save_dir}/group_{gid:04d}_fig6_anomaly_detail_cycle{vi+1}.png"
        fig.savefig(p, dpi=130, bbox_inches='tight')
        plt.close(fig)
        figs_saved.append(p)
        print(f"  Saved: {p}")

    lengths = [l for _,_,l in cycles]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=min(30, len(lengths)), color='#60A5FA', edgecolor='white', alpha=0.85)
    ax.axvline(np.median(lengths), color='#DC2626', lw=2, ls='--',
               label=f'Median = {int(np.median(lengths))}pts')
    ax.axvline(group_period := out['group_period'], color='#059669', lw=2, ls='-.',
               label=f'Group period = {group_period}pts')
    ax.set_xlabel('Cycle length (pts)')
    ax.set_ylabel('Count')
    ax.set_title(f"Group period={gid}pts — Cycle length distribution\n"
                 f"{len(cycles)} cycles | seg: {seg}", fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = f"{save_dir}/group_{gid:04d}_fig7_lengths.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    figs_saved.append(p)
    print(f"  Saved: {p}")

    return figs_saved


# ANOMALY REPORT TEXT
def visualise_cusum_html(out, save_dir='group_plots'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    gid        = out['group_id']
    all_cycles = out['all_cycles']
    results    = out['results']
    cusum_meta = out['cusum_meta']
    corr       = out['corr']
    cusum_anom = out['cusum_anomalies']
    raw_anom   = out['corr_anomalies']
    spikes     = out['cusum_spikes']
    train_end  = out['train_end']
    cl_loo_means  = out['cluster_loo_means']
    cl_loo_scores = out['cluster_loo_scores']

    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=[
            'Correlation signal — cycle regions colored by CUSUM state',
            'CUSUM statistic per test cycle'
        ]
    )

    # Correlation signal trace
    fig.add_trace(go.Scatter(
        x=corr.index, y=corr.values,
        mode='lines',
        line=dict(color='#1e3a8a', width=1.2),
        name='Correlation strength',
        showlegend=True,
    ), row=1, col=1)

    valid_corr = corr.dropna()
    y_min = float(valid_corr.min())
    y_max = float(valid_corr.max())

    # Color each cycle by CUSUM state
    for i, (start, end, length) in enumerate(all_cycles):
        xe    = min(end, len(corr) - 1)
        r_obj = next((r for r in results if r['cycle_idx'] == i), None)

        if start < train_end:
            r_obj = next((r for r in results if r['cycle_idx'] == i), None)
            is_template = i in out['all_template_indices']
            if r_obj is None or is_template:
                cfill   = 'rgba(191,219,254,0.25)'
                cborder = 'rgba(96,165,250,0.5)'
                label   = f'TR{i+1}'
            else:
                state = r_obj.get('cusum_state', 'normal')
                if state == 'alarm':
                    cfill, cborder = 'rgba(220,40,40,0.20)', 'rgba(200,40,40,0.6)'
                    label = f'TR-ALARM\nC{i+1}'
                elif state == 'accumulating':
                    cfill, cborder = 'rgba(255,140,0,0.18)', 'rgba(200,120,0,0.5)'
                    label = f'TR↑C{i+1}'
                elif state == 'rising':
                    cfill, cborder = 'rgba(255,220,0,0.15)', 'rgba(180,160,0,0.4)'
                    label = f'TR~C{i+1}'
                elif i in spikes:
                    cfill, cborder = 'rgba(120,120,120,0.20)', 'rgba(100,100,100,0.6)'
                    label = f'SPIKE\nC{i+1}'
                else:
                    cfill   = 'rgba(191,219,254,0.20)' 
                    cborder = 'rgba(96,165,250,0.4)'
                    label   = f'TR{i+1}'
        elif r_obj is None:
            cfill   = 'rgba(200,200,200,0.18)'
            cborder = 'rgba(150,150,150,0.5)'
            label   = f'N{i+1}'
        else:
            cv    = r_obj.get('cusum_value', 0.0)
            cthr  = r_obj.get('cusum_thr', 1.0) or 1.0
            alarm = r_obj.get('cusum_alarm', False)
            is_spike = i in spikes

            if alarm:
                cfill   = 'rgba(220,40,40,0.30)'
                cborder = 'red'
                label   = f'ALARM\nC{i+1}'
            elif is_spike:
                cfill   = 'rgba(120,120,120,0.20)'
                cborder = 'rgba(100,100,100,0.6)'
                label   = f'SPIKE\nC{i+1}'
            elif cv > cthr * 0.5:
                frac  = min(cv / cthr, 1.0)

                if frac > 0.75:
                    cfill   = 'rgba(239,68,68,0.30)' 
                    cborder = 'rgba(220,38,38,0.80)'
                else:
                    cfill   = 'rgba(251,146,60,0.28)' 
                    cborder = 'rgba(234,88,12,0.80)'
                label = f'↑C{i+1}'
            else:
                cfill   = 'rgba(34,197,94,0.15)'
                cborder = 'rgba(21,128,61,0.6)'
                label   = f'N{i+1}'

        fig.add_shape(
            type='rect',
            x0=corr.index[start], x1=corr.index[xe],
            y0=y_min, y1=y_max,
            fillcolor=cfill,
            line=dict(color=cborder, width=1.5),
            layer='below', row=1, col=1
        )
        fig.add_annotation(
            x=corr.index[start], y=y_max,
            text=label, showarrow=False,
            xanchor='left', yanchor='top',
            bgcolor='white', bordercolor=cborder, borderwidth=1,
            font=dict(size=8), row=1, col=1
        )

    # Train/test split line
    fig.add_vline(x=train_end, line_color='black',
                  line_dash='dash', line_width=1.5, row=1, col=1)
    fig.add_annotation(x=train_end, y=y_max, text='Train / Test',
                       showarrow=False, yshift=10,
                       bgcolor='white', bordercolor='black',
                       row=1, col=1)

    # CUSUM statistic per test cycle 
    all_cusum_results = sorted(
        [r for r in results if 'cusum_value' in r],
        key=lambda r: r['cycle_idx'])

    if all_cusum_results:
        cycle_nums  = [r['cycle_idx'] + 1 for r in all_cusum_results]
        cusum_vals  = [r.get('cusum_value', 0.0) for r in all_cusum_results]
        cusum_thrs  = [r.get('cusum_thr', None) for r in all_cusum_results]
        alarm_flags = [r.get('cusum_alarm', False) for r in all_cusum_results]
        raw_scores = [r.get('cusum_norm_score', 0.0) for r in all_cusum_results]
        cycle_starts = [all_cycles[r['cycle_idx']][0] for r in all_cusum_results]
        # CUSUM line
        fig.add_trace(go.Scatter(
            x=cycle_starts, y=cusum_vals,
            mode='lines+markers',
            line=dict(color='#7c3aed', width=2),
            marker=dict(
                size=8,
                color=['#dc2626' if a else '#7c3aed' for a in alarm_flags],
                symbol=['star' if a else 'circle' for a in alarm_flags],
            ),
            name='CUSUM S(t)',
        ), row=2, col=1)

        # CUSUM threshold line
        if cusum_thrs and any(v is not None for v in cusum_thrs):
            thr_val = next(v for v in cusum_thrs if v is not None)
            fig.add_hline(
                y=thr_val, line_color='red',
                line_dash='dash', line_width=2,
                annotation_text=f'CUSUM threshold = {thr_val:.3f}',
                annotation_position='top right',
                row=2, col=1
            )

        fig.add_trace(go.Bar(
            x=cycle_starts,  y=raw_scores,
            name='Raw max_score',
            marker_color='rgba(59,130,246,0.35)',
            showlegend=True,
        ), row=2, col=1)

        for cs, alarm, cv in zip(cycle_starts, alarm_flags, cusum_vals):
            if alarm:
                fig.add_vline(
                    x=cs, line_color='red',
                    line_dash='dot', line_width=2,
                    row=2, col=1
                )
                fig.add_annotation(
                    x=cs, y=max(cusum_vals) * 1.05,
                    text='ALARM', showarrow=False,
                    font=dict(color='red', size=10),
                    row=2, col=1
                )

        # Mark spikes
        for r in all_cusum_results:
            if r['cycle_idx'] in spikes:
                cs = all_cycles[r['cycle_idx']][0]
                fig.add_annotation(
                    x=cs,
                    y=r.get('cusum_value', 0),
                    text='spike\n(ignored)',
                    showarrow=True, arrowhead=2,
                    font=dict(color='gray', size=9),
                    row=2, col=1
                )

    fig.update_layout(
        title=dict(
            text=(f'Group {gid}pts — CUSUM drift detection<br>'
                  f'<span style="font-size:12px">'
                  f'RED=alarm  ORANGE=accumulating  GREEN=normal  GRAY=spike(ignored)'
                  f'</span>'),
            x=0.5, font=dict(size=14)
        ),
        template='plotly_white',
        height=750,
        showlegend=True,
        legend=dict(orientation='h', y=-0.08),
    )
    fig.update_xaxes(title_text='Cycle number', row=2, col=1)
    fig.update_yaxes(title_text='Correlation strength', row=1, col=1)
    fig.update_yaxes(title_text='Score / CUSUM S(t)', row=2, col=1)

    path = f'{save_dir}/group_{gid:04d}_cusum.html'
    fig.write_html(path, include_plotlyjs='cdn', full_html=True)
    print(f'  [CUSUM HTML] saved → {path}')
    return path

def print_group_report(out):
    gid  = out['group_id']
    print(f"\n{'='*70}")
    print(f"ANOMALY REPORT  —  Group period={gid}pts | {len(out['group_cols'])} sensors")
    print(f"Sensors: {', '.join(out['group_cols'])}")
    print(f"{'='*70}")
    print(f"  Segmentation  : {out['seg_method']}")
    print(f"  Total cycles  : {len(out['all_cycles'])}")
    print(f"  Corr anomalies: {len(out['corr_anomalies'])}")
    print(f"  Amp anomalies : {len(out['amp_anomalies'])}")
    print(f"  Both          : {len(out['both_methods'])}")
    print(f"  FINAL (union) : {len(out['final_anomalies'])}")

    if not out['final_anomalies']:
        print("\n  No anomalies detected.")
        return

    for gi in sorted(out['final_anomalies']):
        start, end, length = out['all_cycles'][gi]
        in_c = gi in out['corr_anomalies']
        in_a = gi in out['amp_anomalies']
        print(f"\n  {'─'*68}")
        print(f"  CYCLE {gi+1}  |  idx {start} → {end}  |  {length} pts")
        if in_c and in_a:
            print(f"  Detection : BOTH methods (high confidence)")
        elif in_c:
            print(f"  Detection : Correlation pattern change only")
        else:
            print(f"  Detection : Amplitude change only")
        if in_a and out['cycle_amp_scores'][gi] is not None:
            print(f"  Amplitude : recon.error={out['cycle_amp_scores'][gi]:.5f}"
                  f"  (thr={out['amp_threshold']:.5f})")
        if in_c:
            r = next((r for r in out['results'] if r['cycle_idx']==gi), None)
            if r:
                tmpl = out['cluster_templates'][r['cluster_id']]
                thr  = r['threshold']
                print(f"  Cluster   : {r['cluster_id']}")
                print(f"  Corr      : max={r['max_score']:.2f}σ | mean={r['mean_score']:.2f}σ"
                      f" | shift={r['shift_applied']}"
                      f" | windows>{thr:.2f}: {r['windows_above_thresh']}/{tmpl['n_windows']}")
                ws_ = out['window_size']
                print(f"\n  {'Win':<5} {'Pos':<10} {'Obs μ':<9} {'Exp μ':<9} "
                      f"{'Obs σ':<9} {'Exp σ':<9} {'z_μ':<8} {'z_σ':<8} {'Score':<8} Status")
                print(f"  {'─'*82}")
                for d in r['window_details']:
                    w   = d['window_idx']
                    pos = f"{w*ws_}-{(w+1)*ws_}"
                    flag = " ← ANOMALY" if d['combined'] > thr else ""
                    print(f"  {w:<5} {pos:<10} "
                          f"{d['obs_mean']:<9.3f} {d['exp_mean']:<9.3f} "
                          f"{d['obs_std']:<9.3f} {d['exp_std']:<9.3f} "
                          f"{d['z_mean']:<8.2f} {d['z_std']:<8.2f} "
                          f"{d['combined']:<8.2f}{flag}")


# MAIN EXECUTION
def estimate_period_from_corr(signal, min_period=10):
    clean = signal.dropna().values
    clean = clean - clean.mean()
    results = {}
    try:
        ac = np.correlate(clean, clean, mode='full')
        ac = ac[len(ac)//2:]
        ac /= (ac[0] + 1e-9)
        peaks, props = find_peaks(ac, height=0.2, distance=min_period)
        if len(peaks):
            best = peaks[np.argmax(props['peak_heights'])]
            results['autocorr'] = (int(best), float(props['peak_heights'].max()))
    except Exception:
        pass
    try:
        fft_v = np.abs(np.fft.rfft(clean))
        freqs = np.fft.rfftfreq(len(clean))
        fft_v[0] = 0
        mask = (freqs >= 1.0/len(clean)) & (freqs <= 1.0/min_period)
        if mask.any():
            mf = fft_v.copy(); mf[~mask] = 0
            di = np.argmax(mf)
            if freqs[di] > 0:
                results['fft'] = (int(round(1.0/freqs[di])),
                                   float(mf[di]/(mf.sum()+1e-9)))
    except Exception:
        pass
    if not results:
        return None, 0.0
    if len(results) == 1:
        m = list(results)[0]
        return results[m]
    p_ac, c_ac   = results['autocorr']
    p_fft, c_fft = results['fft']
    if abs(p_ac - p_fft) / (max(p_ac, p_fft) + 1e-9) < 0.15:
        combined = int(round((p_ac*c_ac + p_fft*c_fft) / (c_ac+c_fft+1e-9)))
        return combined, (c_ac+c_fft)/2
    return (p_ac, c_ac) if c_ac >= c_fft else (p_fft, c_fft)

normal_end = int(len(df) * 0.75)
df_normal  = df.iloc[:normal_end]

print("\nEstimating period from all-sensor correlation signal (normal region)...")
auto_win_est = max(10, len(df_normal) // 500)   # rough window for estimation
corr_est = correlation_strength(df_normal, sensor_cols, auto_win_est)
period, conf = estimate_period_from_corr(corr_est)

if period is None:
    print("  Could not estimate period — using fixed fallback of 50 samples")
    period = 50
print(f"  Period = {period} samples  (conf={conf:.4f})")

group_outputs = []
out = run_pipeline_for_group(df, sensor_cols, period, group_id=period)
if out is not None:
    group_outputs.append(out)

# Print reports and generate visualizations
print("\n\n" + "#"*70)
print("# ANOMALY REPORTS — ALL GROUPS")
print("#"*70)

all_figs = {}
for out in group_outputs:
    print_group_report(out)
    print(f"\n  Generating visualizations for group period={out['group_id']}pts...")
    figs = visualise_group(out)
    visualise_cusum_html(out)
    all_figs[out['group_id']] = figs

# Cross-group merge
print(f"\n\n{'='*70}")
print("CROSS-GROUP ANOMALY MERGE")
print('='*70)

all_flagged = []
for out in group_outputs:
    for ci in out['final_anomalies']:
        start, end, _ = out['all_cycles'][ci]
        in_c  = ci in out['corr_anomalies']
        in_a  = ci in out['amp_anomalies']
        det   = 'BOTH' if (in_c and in_a) else ('CORR' if in_c else 'AMP')
        r     = next((r for r in out['results'] if r['cycle_idx']==ci), None)
        all_flagged.append({
            'group_period': out['group_period'],
            'cycle_idx':    ci,
            'idx_start':    start,
            'idx_end':      end,
            'det_type':     det,
            'score':        r['max_score'] if r else None,
            'sensors':      out['group_cols'],
        })

print(f"\nTotal anomaly events: {len(all_flagged)}")
print(f"\nBy group:")
for out in group_outputs:
    n = len(out['final_anomalies'])
    print(f"  period={out['group_period']:>4}pts | {n} anomalies | {len(out['group_cols'])} sensors")

print(f"\nAll anomalous events (sorted by time index):")
print(f"{'─'*70}")
for f in sorted(all_flagged, key=lambda x: x['idx_start']):
    sc = f"{f['score']:.2f}σ" if f['score'] is not None else "—"
    print(f"  [{f['det_type']:<4}]  idx {f['idx_start']:>4}→{f['idx_end']:>4}  "
          f"| group={f['group_period']}pts  | score={sc}")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PLOTLY VISUALIZATIONS
PLOTLY_DIR = 'group_plots'
def save_plotly_html(fig, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.write_html(path, include_plotlyjs='cdn', full_html=True)
    print(f"  [HTML] saved → {path}")

CLUSTER_COLORS = [
    ('rgba(0,100,255,0.12)',  'rgba(0,100,255,0.6)'),
    ('rgba(0,180,130,0.12)', 'rgba(0,180,130,0.6)'),
    ('rgba(180,0,180,0.12)', 'rgba(180,0,180,0.6)'),
    ('rgba(255,140,0,0.12)', 'rgba(255,140,0,0.6)'),
    ('rgba(100,100,0,0.12)', 'rgba(100,100,0,0.6)'),
]

for out in group_outputs:
    gid         = out['group_id']
    corr_full   = out['corr']
    all_cycles  = out['all_cycles']
    clusters    = out['clusters']
    all_template_indices = out['all_template_indices']
    template_global_idx_sets = out['template_global_idx_sets']
    corr_anomalies    = out['corr_anomalies']
    amplitude_anomalies = out['amp_anomalies']
    cycle_amp_scores  = out['cycle_amp_scores']
    results           = out['results']
    all_normalised    = out['all_normalised']
    cluster_templates = out['cluster_templates']
    cluster_medians   = out['cluster_medians']
    cluster_thresholds = out['cluster_thresholds']
    final_anomalies   = out['final_anomalies']
    seg_method        = out['seg_method']
    target_length     = out['target_length']
    window_size       = out['window_size']
    n_clusters        = len(clusters)

    print(f"\n{'='*60}")
    print(f"PLOTLY PLOTS — Group period={gid}pts")
    print(f"{'='*60}")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=corr_full.index, y=corr_full.values,
        name="Correlation Strength",
        line=dict(width=1.5, color='steelblue'), connectgaps=False
    ))

    valid_corr = corr_full.dropna()
    y_min = float(valid_corr.min()) if len(valid_corr) else 0
    y_max = float(valid_corr.max()) if len(valid_corr) else 1

    #last_tmpl_end = max((all_cycles[j][1] for j in all_template_indices if j < len(all_cycles)), default=0) if all_template_indices else 0

    for i, (start, end, length) in enumerate(all_cycles):
        cycle_cluster = next((cid for cid, idxs in clusters.items() if i in idxs), None)
        in_corr     = i in corr_anomalies
        in_amp      = i in amplitude_anomalies
        #in_template = (i in all_template_indices) or (end <= last_tmpl_end and not in_corr and not in_amp)
        in_template = i in all_template_indices

        xe = min(end, len(corr_full) - 1)

        if in_template:
            col = cycle_cluster % len(CLUSTER_COLORS) if cycle_cluster is not None else 0
            cfill, cborder = CLUSTER_COLORS[col]
            label = f"T{i+1}"
        elif in_corr and in_amp:
            cfill, cborder = 'rgba(139,0,0,0.28)', 'darkred'
            r = next((r for r in results if r['cycle_idx'] == i), None)
            label = f"A{i+1} BOTH {r['max_score']:.1f}σ" if r else f"A{i+1} BOTH"
        elif in_corr:
            cfill, cborder = 'rgba(220,50,50,0.20)', 'red'
            r = next((r for r in results if r['cycle_idx'] == i), None)
            label = f"A{i+1} CORR {r['max_score']:.1f}σ" if r else f"A{i+1} CORR"
        elif in_amp:
            cfill, cborder = 'rgba(255,140,0,0.22)', 'darkorange'
            amp = cycle_amp_scores[i]
            label = f"A{i+1} AMP {amp:.3f}" if amp is not None else f"A{i+1} AMP"
        else:
            cfill, cborder = 'rgba(50,180,50,0.15)', 'rgba(50,180,50,0.7)'
            label = f"N{i+1}"

        fig1.add_shape(
            type="rect",
            x0=corr_full.index[start], x1=corr_full.index[xe],
            y0=y_min, y1=y_max,
            fillcolor=cfill, line=dict(width=1.5, color=cborder), layer="below"
        )
        fig1.add_annotation(
            x=corr_full.index[start], y=y_max,
            text=label, showarrow=False,
            xanchor="left", yanchor="top",
            bgcolor="white", bordercolor=cborder, borderwidth=1,
            font=dict(size=8)
        )

    if all_template_indices:
        split_x = out['train_end']
        fig1.add_shape(type="line",
            x0=split_x, x1=split_x, y0=y_min, y1=y_max,
            line=dict(color="black", width=2, dash="dash"))
        fig1.add_annotation(x=split_x, y=y_max,
            text="Template / Test Split", showarrow=False, yshift=10,
            bgcolor="white", bordercolor="black", borderwidth=1)

    fig1.update_layout(
        title=(f"Group {gid}pts | seg={seg_method} | {n_clusters} cluster(s) | "
               f"{len(final_anomalies)} anomalies "
f"(corr={len(corr_anomalies)}, amp={len(amplitude_anomalies)})"),
        xaxis_title="Sample index", yaxis_title="Correlation Strength",
        template="plotly_white", height=500
    )
    save_plotly_html(fig1, f"{PLOTLY_DIR}/group_{gid:04d}_plotly_fig1_signal.html")

    for cid in cluster_templates:
        tmpl      = cluster_templates[cid]
        median_c  = cluster_medians[cid]
        threshold = cluster_thresholds[cid]
        tmpl_norm = [all_normalised[i] for i in template_global_idx_sets[cid]
             if i < len(all_normalised) and all_normalised[i] is not None]

        fig2 = go.Figure()
        x_pos = np.arange(target_length)

        for k, nc in enumerate(tmpl_norm):
            fig2.add_trace(go.Scatter(
                x=x_pos, y=nc, mode='lines',
                line=dict(color='lightblue', width=1), opacity=0.4,
                showlegend=(k == 0), name="Template cycles", legendgroup="tmpl"
            ))
        fig2.add_trace(go.Scatter(
            x=x_pos, y=median_c, mode='lines',
            line=dict(color='royalblue', width=2.5), name="Template median"
        ))

        shown_anom, shown_norm = False, False
        for r in results:
            if r['cluster_id'] != cid:
                continue
            vi = r['cycle_idx']
            nc = all_normalised[vi]
            if nc is None:
                continue
            s, e, _ = all_cycles[vi]
            if r['is_anomaly']:
                fig2.add_trace(go.Scatter(
                    x=x_pos, y=nc, mode='lines',
                    line=dict(color='red', width=2), opacity=0.8,
                    showlegend=not shown_anom, name="Anomalous", legendgroup="anom",
                    hovertemplate=f"<b>Cycle {vi+1}</b> max={r['max_score']:.2f}σ<br>idx {s}→{e}<extra></extra>"
                ))
                shown_anom = True
            else:
                fig2.add_trace(go.Scatter(
                    x=x_pos, y=nc, mode='lines',
                    line=dict(color='mediumseagreen', width=1.2), opacity=0.35,
                    showlegend=not shown_norm, name="Normal test", legendgroup="norm",
                    hovertemplate=f"<b>Cycle {vi+1}</b> normal<br>idx {s}→{e}<extra></extra>"
                ))
                shown_norm = True

        x_win = np.arange(tmpl['n_windows']) * window_size + window_size / 2
        fig2.add_trace(go.Scatter(
            x=x_win, y=tmpl['mean_of_means'],
            mode='lines+markers', line=dict(color='darkred', width=2),
            marker=dict(size=7), name="Window mean"
        ))
        fig2.add_trace(go.Scatter(
            x=x_win, y=tmpl['mean_of_means'] + 2*tmpl['std_of_means'],
            mode='lines', line=dict(color='darkred', width=1, dash='dash'), name="±2σ band"
        ))
        fig2.add_trace(go.Scatter(
            x=x_win, y=tmpl['mean_of_means'] - 2*tmpl['std_of_means'],
            mode='lines', line=dict(color='darkred', width=1, dash='dash'),
            showlegend=False, fill='tonexty', fillcolor='rgba(180,0,0,0.07)'
        ))
        fig2.update_layout(
            title=f"Group {gid}pts — Cluster {cid} normalised cycles | thr={threshold:.3f}",
            xaxis_title="Normalised position", yaxis_title="Correlation strength",
            template="plotly_white", height=500
        )
        save_plotly_html(fig2, f"{PLOTLY_DIR}/group_{gid:04d}_plotly_fig2_cycles_cluster{cid}.html")

    amp_thr = out['amp_threshold']
    valid_amp = [(i, s) for i, s in enumerate(cycle_amp_scores)
                 if s is not None and not np.isnan(s)]

    if valid_amp and amp_thr is not None:
        idxs_amp, scores_amp = zip(*valid_amp)
        colors_amp = ['red' if i in amplitude_anomalies else 'steelblue' for i in idxs_amp]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=[f"C{i+1}" for i in idxs_amp], y=scores_amp,
            marker_color=colors_amp, name="PCA recon error",
            hovertemplate="Cycle %{x}<br>Error: %{y:.4f}<extra></extra>"
        ))
        fig3.add_hline(y=amp_thr, line_color="red", line_dash="dash",
                       annotation_text=f"Threshold={amp_thr:.4f}")
        fig3.update_layout(
            title=f"Group {gid}pts — PCA amplitude scores per cycle",
            xaxis_title="Cycle", yaxis_title="Reconstruction error",
            template="plotly_white", height=400
        )
        save_plotly_html(fig3, f"{PLOTLY_DIR}/group_{gid:04d}_plotly_fig3_amplitude.html")

    # anom_res = [r for r in results if r['is_anomaly']]
    # if anom_res:
    #     nw_max  = max(r['window_scores'].shape[0] for r in anom_res)
    #     labels  = [f"C{r['cycle_idx']+1}" for r in anom_res]
    #     matrix  = np.array([
    #         np.pad(r['window_scores'], (0, nw_max - len(r['window_scores'])))
    #         for r in anom_res
    #     ])
    #     fig4 = go.Figure(go.Heatmap(
    #         z=matrix, x=[f"Win {w}" for w in range(nw_max)], y=labels,
    #         colorscale='RdBu_r', text=np.round(matrix, 2), texttemplate="%{text}",
    #         hovertemplate="Cycle %{y} | %{x}<br>Score: %{z:.2f}<extra></extra>"
    #     ))
    #     for yi, (r, thr_v) in enumerate(zip(anom_res, [r['threshold'] for r in anom_res])):
    #         for xi, sc in enumerate(r['window_scores']):
    #             if sc > thr_v:
    #                 fig4.add_shape(type="rect",
    #                     x0=xi-0.5, x1=xi+0.5, y0=yi-0.5, y1=yi+0.5,
    #                     line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)")
    #     fig4.update_layout(
    #         title=f"Group {gid}pts — Window z-scores for anomalous cycles (black=above threshold)",
    #         template="plotly_white", height=max(300, len(anom_res)*60 + 150)
    #     )
    #     save_plotly_html(fig4, f"{PLOTLY_DIR}/group_{gid:04d}_plotly_fig4_heatmap.html")

print(f"\n\n{'='*70}")
print("PIPELINE COMPLETE")
print('='*70)
print(f"  Groups processed    : {len(group_outputs)}")
print(f"  Total anomaly events: {len(all_flagged)}")
print(f"  Plot files saved to : group_plots/")
for gid, figs in all_figs.items():
    print(f"\n  Group period={gid}pts:")
    for f in figs:
        print(f"    {os.path.basename(f)}")

# EXPORT ANOMALY SEQUENCES TO CSV
import csv

if all_flagged:
    anomaly_rows = []
    for out in group_outputs:
        gid = out['group_id']
        for ci in sorted(out['final_anomalies']):
            start, end, length = out['all_cycles'][ci]

            in_c = ci in out['corr_anomalies']
            in_a = ci in out['amp_anomalies']
            det  = 'BOTH' if (in_c and in_a) else ('CORR' if in_c else 'AMP')
            r    = next((r for r in out['results'] if r['cycle_idx'] == ci), None)

            # Pull the raw sensor values for this cycle's index range
            cycle_df = df.iloc[start:end+1].copy()
            cycle_df.insert(0, 'sample_index', range(start, end+1))
            cycle_df.insert(1, 'cycle_idx',    ci + 1)
            cycle_df.insert(2, 'group_period', gid)
            cycle_df.insert(3, 'cycle_start',  start)
            cycle_df.insert(4, 'cycle_end',    end)
            cycle_df.insert(5, 'cycle_length', length)
            cycle_df.insert(6, 'detection_type', det)
            cycle_df.insert(7, 'max_corr_score',
                            round(r['max_score'], 4) if r else None)
            cycle_df.insert(8, 'amp_recon_error',
                            round(out['cycle_amp_scores'][ci], 5)
                            if out['cycle_amp_scores'][ci] is not None else None)
            anomaly_rows.append(cycle_df)

    anomaly_csv = pd.concat(anomaly_rows, ignore_index=True)
    out_path    = 'group_plots/anomaly_sequences.csv'
    anomaly_csv.to_csv(out_path, index=False)
    print(f"\n  Anomaly sequences CSV saved → {out_path}")
    print(f"  Rows: {len(anomaly_csv)} | Anomalous cycles: {len(anomaly_rows)}")
    print(f"  Columns: {list(anomaly_csv.columns)}")
else:
    print("\n  No anomalies detected — CSV not generated.")