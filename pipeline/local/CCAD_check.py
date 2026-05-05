
print("CCAD adequacy script starting...")

import numpy as np
import pandas as pd
import os
import sys
import warnings
from itertools import combinations
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')

# CONFIG 
class CFG:
    csv         = r"D:\\MaFaulDa\\streams\\stream_B.csv"
    drop        = ['tachometer', 'microphone']
    timestamp   = 'timestamp'
    train_frac  = 0.75
    min_corr    = 0.15    # min mean |corr| for a sensor pair to be grouped
    out         = 'ccad_adequacy_report.txt'
    plot        = True    # set False to skip diagnostic plots


# UTILITIES
def rolling_mean_abs_corr(series_a, series_b, window):

    n = len(series_a)
    out = np.full(n, np.nan)
    a, b = series_a.values, series_b.values
    for i in range(window, n):
        wa, wb = a[i-window:i], b[i-window:i]
        if wa.std() < 1e-9 or wb.std() < 1e-9:
            continue
        out[i] = abs(np.corrcoef(wa, wb)[0, 1])
    return out


def multivariate_corr_signal(df, cols, window):

    n = len(df)
    out = np.full(n, np.nan)
    arr = df[cols].values
    for i in range(window, n):
        block = arr[i-window:i]
        active = [j for j in range(len(cols)) if block[:, j].std() > 1e-9]
        if len(active) < 2:
            continue
        sub = block[:, active]
        cm  = np.corrcoef(sub.T)
        ut  = cm[np.triu_indices_from(cm, k=1)]
        vp  = ut[~np.isnan(ut)]
        if len(vp):
            out[i] = np.mean(np.abs(vp))
    return pd.Series(out)


def estimate_period(signal, min_period=10):

    clean = signal.dropna().values
    if len(clean) < min_period * 3:
        return None, 0.0, 'insufficient data', None

    clean_c = clean - clean.mean()
    results = {}

    try:
        ac = np.correlate(clean_c, clean_c, mode='full')
        ac = ac[len(ac)//2:]
        ac /= (ac[0] + 1e-9)
        peaks, props = find_peaks(ac, height=0.2, distance=min_period)
        if len(peaks):
            best = peaks[np.argmax(props['peak_heights'])]
            results['autocorr'] = (int(best), float(props['peak_heights'].max()))
    except Exception:
        ac = None

    try:
        fft_v = np.abs(np.fft.rfft(clean_c))
        freqs = np.fft.rfftfreq(len(clean_c))
        fft_v[0] = 0
        mask = (freqs >= 1.0/len(clean_c)) & (freqs <= 1.0/min_period)
        if mask.any():
            mf = fft_v.copy(); mf[~mask] = 0
            di = np.argmax(mf)
            if freqs[di] > 0:
                results['fft'] = (int(round(1.0/freqs[di])),
                                  float(mf[di]/(mf.sum()+1e-9)))
    except Exception:
        pass

    if not results:
        return None, 0.0, 'no peaks found', ac

    if len(results) == 1:
        m = list(results)[0]
        p, c = results[m]
        return p, c, m, ac

    p_ac, c_ac   = results['autocorr']
    p_fft, c_fft = results['fft']
    agree = abs(p_ac - p_fft) / (max(p_ac, p_fft) + 1e-9) < 0.15
    if agree:
        combined = int(round((p_ac*c_ac + p_fft*c_fft) / (c_ac+c_fft+1e-9)))
        return combined, (c_ac+c_fft)/2, 'autocorr+fft (agree)', ac
    if c_ac >= c_fft:
        return p_ac, c_ac, 'autocorr (higher conf)', ac
    return p_fft, c_fft, 'fft (higher conf)', ac


def period_drift_continuous(signal, period, n_windows=10):

    clean = signal.dropna()
    n     = len(clean)
    step  = n // n_windows
    if step < period * 2:
        return [], np.nan, 'segment too short for drift analysis'

    local_periods = []
    centres       = []
    for w in range(n_windows):
        s = w * step
        e = min(s + step * 2, n)   # overlapping windows for smoother estimate
        seg = clean.iloc[s:e]
        p, c, _, _ = estimate_period(seg, min_period=max(5, period//4))
        if p is not None and c > 0.1:
            local_periods.append(p)
            centres.append(s + (e-s)//2)

    if len(local_periods) < 3:
        return list(zip(centres, local_periods)), np.nan, 'too few valid segments'

    lp  = np.array(local_periods)
    cv  = lp.std() / (lp.mean() + 1e-9)
    return list(zip(centres, local_periods)), float(cv), 'ok'


def training_purity(signal, window=50):

    clean = signal.dropna().values
    if len(clean) < window * 3:
        return np.nan
    mu, sigma = clean.mean(), clean.std()
    threshold = mu - 2 * sigma
    n_windows = len(clean) // window
    anomalous = 0
    for w in range(n_windows):
        seg = clean[w*window:(w+1)*window]
        if seg.mean() < threshold:
            anomalous += 1
    return anomalous / n_windows


def segmentation_quality_estimate(signal, period):

    clean = signal.dropna()
    n     = len(clean)
    if period is None or period < 5 or n < period * 3:
        return np.inf

    cycles = [(s, s+period-1, period) for s in range(0, n-period, period)]
    if len(cycles) < 3:
        return np.inf

    tl  = min(100, period)
    smp = []
    for s, e, _ in cycles[:15]:
        seg = clean.iloc[s:e+1].values
        seg = seg[~np.isnan(seg)]
        if len(seg) < 5:
            continue
        resampled = np.interp(np.linspace(0,1,tl), np.linspace(0,1,len(seg)), seg)
        smp.append(resampled)

    if len(smp) < 3:
        return np.inf

    arr    = np.array(smp)
    med    = np.median(arr, axis=0)
    spread = np.mean([np.linalg.norm(c-med) for c in arr]) / (clean.std() + 1e-9)
    return float(spread)


# SENSOR-LEVEL CHECKS
def check_sensor_quality(df, cols):
    """
    Per-sensor quality: missing fraction, std, and whether it passes.
    Returns dict keyed by sensor name.
    """
    results = {}
    for c in cols:
        missing = df[c].isna().mean()
        std     = df[c].std()
        passed  = (missing <= 0.30) and (std >= 1e-6)
        results[c] = {
            'missing_frac': float(missing),
            'std':          float(std),
            'passed':       passed,
            'reason':       ('ok' if passed else
                             ('near-constant' if std < 1e-6 else 'too many NaN'))
        }
    return results


# PAIRWISE PERIOD ESTIMATION FOR GROUPING
def pairwise_periods(df, cols, train_end, window):

    n   = len(cols)
    P   = np.full((n, n), np.nan)   # period matrix
    C   = np.full((n, n), 0.0)      # confidence matrix
    df_tr = df.iloc[:train_end]

    for i, j in combinations(range(n), 2):
        sig = rolling_mean_abs_corr(df_tr[cols[i]], df_tr[cols[j]], window)
        sig = pd.Series(sig).dropna()
        if len(sig) < window * 3:
            continue
        p, c, _, _ = estimate_period(sig)
        if p is not None:
            P[i, j] = P[j, i] = p
            C[i, j] = C[j, i] = c
        P[i, i] = 0
        P[j, j] = 0

    return P, C


def group_sensors_by_period(cols, period_matrix, conf_matrix,
                             mean_corr_matrix, min_corr=0.15,
                             period_tol=0.20):

    n = len(cols)

    D = np.ones((n, n))
    np.fill_diagonal(D, 0)

    for i, j in combinations(range(n), 2):
        pi, pj = period_matrix[i, j], period_matrix[j, i]
        mc     = mean_corr_matrix[i, j]
        if np.isnan(pi) or np.isnan(pj) or mc < min_corr:
            D[i, j] = D[j, i] = 1.0
        else:
            rel_diff = abs(pi - pj) / (max(pi, pj) + 1e-9)
            D[i, j]  = D[j, i] = min(rel_diff / period_tol, 1.0)

    # Hierarchical clustering — cut at distance 0.5
    try:
        condensed = squareform(D)
        Z         = linkage(condensed, method='average')
        labels    = fcluster(Z, t=0.5, criterion='distance')
    except Exception:
        labels = np.arange(n) + 1

    groups = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)

    valid_groups   = []
    invalid_sensors = []

    for label, indices in groups.items():
        if len(indices) < 2:
            invalid_sensors.extend(indices)
            continue
        # Check that at least one pair in the group has adequate correlation
        has_corr = any(
            mean_corr_matrix[i, j] >= min_corr
            for i, j in combinations(indices, 2)
            if not np.isnan(mean_corr_matrix[i, j])
        )
        if has_corr:
            valid_groups.append(indices)
        else:
            invalid_sensors.extend(indices)

    return valid_groups, invalid_sensors

# SCORING
def score_correlation_strength(mean_corr):
    if np.isnan(mean_corr):
        return 0.0
    return float(np.clip((mean_corr - 0.05) / 0.45, 0, 1))


def score_periodicity(confidence):
    if np.isnan(confidence) or confidence is None:
        return 0.0
    return float(np.clip(confidence / 0.8, 0, 1))


def score_period_stability(drift_cv):
    if np.isnan(drift_cv) or drift_cv is None:
        return 0.5   # unknown — neutral
    return float(np.clip(1.0 - drift_cv / 0.35, 0, 1))


def score_training_purity(anomalous_frac):
    if np.isnan(anomalous_frac) or anomalous_frac is None:
        return 0.5
    return float(np.clip(1.0 - anomalous_frac / 0.25, 0, 1))


def score_shape_consistency(seg_quality):
    if np.isinf(seg_quality) or np.isnan(seg_quality):
        return 0.0
    return float(np.clip(1.0 - seg_quality / 8.0, 0, 1))


def score_sensor_completeness(n_passed, n_total):
    if n_total == 0:
        return 0.0
    return float(n_passed / n_total)


def overall_score(scores_dict):
    weights = {
        'correlation_strength': 0.25,
        'periodicity':          0.30,
        'period_stability':     0.20,
        'training_purity':      0.10,
        'shape_consistency':    0.10,
        'sensor_completeness':  0.05,
    }
    total = 0.0
    for k, w in weights.items():
        v = scores_dict.get(k, 0.0)
        total += w * v
    return float(total)


def adequacy_label(score):
    if score >= 0.75:
        return '✓ SUITABLE'
    elif score >= 0.50:
        return '~ MARGINAL'
    else:
        return '✗ NOT SUITABLE'


# GROUP ASSESSMENT
def assess_group(df, cols, train_end, window, period_hint=None):
    df_tr = df.iloc[:train_end]

    # Correlation signal
    corr_sig = multivariate_corr_signal(df_tr, cols, window)
    mean_corr = float(corr_sig.dropna().mean()) if corr_sig.dropna().any() else np.nan

    # Period estimation
    period, conf, method, _ = estimate_period(corr_sig)
    if period is None and period_hint is not None:
        period, conf, method = period_hint, 0.0, 'fallback hint'

    # Period drift 
    drift_pairs, drift_cv, drift_status = period_drift_continuous(
        corr_sig, period or 50, n_windows=10)

    # Training purity
    purity_frac = training_purity(corr_sig)

    # Shape consistency
    seg_q = segmentation_quality_estimate(corr_sig, period)

    # Sensor completeness
    sq = check_sensor_quality(df_tr, cols)
    n_pass = sum(1 for v in sq.values() if v['passed'])

    # Scores
    s_corr   = score_correlation_strength(mean_corr)
    s_period = score_periodicity(conf)
    s_drift  = score_period_stability(drift_cv)
    s_purity = score_training_purity(purity_frac)
    s_shape  = score_shape_consistency(seg_q)
    s_sens   = score_sensor_completeness(n_pass, len(cols))

    scores = {
        'correlation_strength': s_corr,
        'periodicity':          s_period,
        'period_stability':     s_drift,
        'training_purity':      s_purity,
        'shape_consistency':    s_shape,
        'sensor_completeness':  s_sens,
    }
    ov = overall_score(scores)

    return {
        'cols':            cols,
        'mean_corr':       mean_corr,
        'period':          period,
        'period_conf':     conf,
        'period_method':   method,
        'drift_cv':        drift_cv,
        'drift_pairs':     drift_pairs,
        'drift_status':    drift_status,
        'purity_frac':     purity_frac,
        'seg_quality':     seg_q,
        'n_sensors_pass':  n_pass,
        'n_sensors_total': len(cols),
        'sensor_quality':  sq,
        'scores':          scores,
        'overall':         ov,
        'label':           adequacy_label(ov),
    }


# REPORT 
BAR_WIDTH = 20

def score_bar(v):
    filled = int(round(v * BAR_WIDTH))
    return '[' + '█' * filled + '░' * (BAR_WIDTH - filled) + f'] {v:.2f}'


def format_drift_series(drift_pairs, period):
    if not drift_pairs:
        return '      (no data)'
    lines = []
    for centre, lp in drift_pairs:
        pct = (lp - period) / (period + 1e-9) * 100
        bar = '▲' if pct > 5 else ('▼' if pct < -5 else '─')
        lines.append(f'      sample ~{centre:>6}  :  period={lp:>5}  ({pct:+.1f}%)  {bar}')
    return '\n'.join(lines)


def format_group_report(g, idx):
    s = g['scores']
    lines = []
    lines.append(f'\n  GROUP {idx} — [{", ".join(g["cols"])}]')
    lines.append(f'  {"─"*66}')

    # Sensor quality
    lines.append(f'  Sensors       : {g["n_sensors_pass"]}/{g["n_sensors_total"]} pass quality filter')
    for cname, cinfo in g['sensor_quality'].items():
        flag = '✓' if cinfo['passed'] else '✗'
        lines.append(f'    {flag} {cname:<35} std={cinfo["std"]:.4f}  missing={cinfo["missing_frac"]:.1%}  ({cinfo["reason"]})')

    lines.append('')

    # Metrics + scores
    mc = g['mean_corr']
    lines.append(f'  Correlation strength    {score_bar(s["correlation_strength"])}')
    lines.append(f'    Mean |corr| in training = {mc:.3f}' if not np.isnan(mc) else '    Mean |corr| = n/a')

    lines.append(f'  Periodicity             {score_bar(s["periodicity"])}')
    if g['period'] is not None:
        lines.append(f'    Period = {g["period"]} samples  |  conf = {g["period_conf"]:.3f}  |  method = {g["period_method"]}')
    else:
        lines.append('    No dominant period detected')

    lines.append(f'  Period stability        {score_bar(s["period_stability"])}')
    if not np.isnan(g['drift_cv']):
        lines.append(f'    Drift CV = {g["drift_cv"]:.3f}  ({g["drift_status"]})')
        lines.append(f'    Continuous period estimates:')
        lines.append(format_drift_series(g['drift_pairs'], g['period'] or 1))
    else:
        lines.append(f'    Drift analysis: {g["drift_status"]}')

    lines.append(f'  Training purity         {score_bar(s["training_purity"])}')
    if not np.isnan(g['purity_frac']):
        lines.append(f'    {g["purity_frac"]:.1%} of training windows below warning threshold')

    lines.append(f'  Shape consistency       {score_bar(s["shape_consistency"])}')
    if not np.isinf(g['seg_quality']):
        lines.append(f'    Segmentation spread score = {g["seg_quality"]:.3f}')
    else:
        lines.append('    Segmentation spread score = n/a')

    lines.append(f'  Sensor completeness     {score_bar(s["sensor_completeness"])}')

    lines.append('')
    lines.append(f'  OVERALL ADEQUACY        {score_bar(g["overall"])}  {g["label"]}')

    # Failure reasons
    reasons = []
    if s['correlation_strength'] < 0.30:
        reasons.append('weak inter-sensor correlation')
    if s['periodicity'] < 0.30:
        reasons.append('no clear periodic structure in correlation signal')
    if s['period_stability'] < 0.30:
        reasons.append('significant period drift — template may not generalise')
    if s['training_purity'] < 0.40:
        reasons.append('training region appears contaminated with anomalies')
    if s['shape_consistency'] < 0.30:
        reasons.append('cycle shapes are highly inconsistent')
    if reasons:
        lines.append(f'  Issues: {"; ".join(reasons)}')

    return '\n'.join(lines)


def format_full_report(df, group_results, ungroupable, all_sensor_quality, train_end):
    lines = []
    lines.append('')
    lines.append('═' * 70)
    lines.append('  CCAD ADEQUACY & GROUPING REPORT')
    lines.append('═' * 70)
    lines.append(f'  Input       : {os.path.basename(CFG.csv)}')
    lines.append(f'  Shape       : {df.shape[0]} samples × {df.shape[1]} columns')
    lines.append(f'  Train end   : sample {train_end} ({CFG.train_frac:.0%} of stream)')
    lines.append(f'  Sensors     : {len(all_sensor_quality)} candidate')
    lines.append('')

    # Global sensor quality
    lines.append('  SENSOR QUALITY FILTER')
    lines.append('  ' + '─' * 66)
    passed  = [c for c, v in all_sensor_quality.items() if v['passed']]
    dropped = [c for c, v in all_sensor_quality.items() if not v['passed']]
    lines.append(f'  Passed  : {len(passed)}/{len(all_sensor_quality)}')
    for c in passed:
        v = all_sensor_quality[c]
        lines.append(f'    ✓ {c:<40} std={v["std"]:.4f}  missing={v["missing_frac"]:.1%}')
    for c in dropped:
        v = all_sensor_quality[c]
        lines.append(f'    ✗ {c:<40} std={v["std"]:.4f}  missing={v["missing_frac"]:.1%}  → {v["reason"]}')

    # Group reports
    lines.append('')
    lines.append('  CANDIDATE GROUPS')
    lines.append('  ' + '─' * 66)
    if not group_results:
        lines.append('  No valid sensor groups found.')
    else:
        for idx, g in enumerate(group_results, 1):
            lines.append(format_group_report(g, idx))

    # Ungroupable sensors
    if ungroupable:
        lines.append('')
        lines.append('  UNGROUPABLE SENSORS')
        lines.append('  ' + '─' * 66)
        lines.append(f'  The following sensors could not be grouped with any other sensor')
        lines.append(f'  (insufficient correlation or no shared dominant period):')
        for c in ungroupable:
            lines.append(f'    • {c}')

    # Recommendation
    lines.append('')
    lines.append('  RECOMMENDATION')
    lines.append('  ' + '─' * 66)
    suitable  = [g for g in group_results if g['overall'] >= 0.75]
    marginal  = [g for g in group_results if 0.50 <= g['overall'] < 0.75]
    unsuitable = [g for g in group_results if g['overall'] < 0.50]

    if suitable:
        lines.append(f'  {len(suitable)} group(s) SUITABLE for CCAD:')
        for g in suitable:
            lines.append(f'    → [{", ".join(g["cols"])}]  (score={g["overall"]:.2f}, period={g["period"]})')
    if marginal:
        lines.append(f'  {len(marginal)} group(s) MARGINAL — results may be unreliable:')
        for g in marginal:
            lines.append(f'    ~ [{", ".join(g["cols"])}]  (score={g["overall"]:.2f})')
            lines.append(f'      Consider tuning thresholds or collecting more data.')
    if unsuitable:
        lines.append(f'  {len(unsuitable)} group(s) NOT SUITABLE:')
        for g in unsuitable:
            lines.append(f'    ✗ [{", ".join(g["cols"])}]  (score={g["overall"]:.2f})')
    if not suitable and not marginal:
        lines.append('  No groups meet the minimum adequacy threshold.')
        lines.append('  CCAD is not recommended for this dataset.')
        lines.append('  Consider: non-cyclic anomaly detectors (Isolation Forest,')
        lines.append('            LSTM autoencoder, or statistical process control).')

    lines.append('')
    lines.append('═' * 70)
    return '\n'.join(lines)


# PLOTS
def save_plots(group_results, save_dir='ccad_adequacy_plots'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [WARN] matplotlib not available — skipping plots')
        return

    os.makedirs(save_dir, exist_ok=True)

    for gi, g in enumerate(group_results, 1):
        # Score radar-style bar chart
        scores = g['scores']
        labels = list(scores.keys())
        values = [scores[k] for k in labels]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        ax = axes[0]
        colors = ['#059669' if v >= 0.75 else '#D97706' if v >= 0.50 else '#DC2626'
                  for v in values]
        bars = ax.barh(labels, values, color=colors, edgecolor='white')
        ax.axvline(0.75, color='#059669', lw=1.5, ls='--', alpha=0.7, label='Suitable')
        ax.axvline(0.50, color='#D97706', lw=1.5, ls='--', alpha=0.7, label='Marginal')
        ax.set_xlim(0, 1)
        ax.set_title(f'Group {gi} — Component scores\nOverall: {g["overall"]:.2f}  {g["label"]}',
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:.2f}', va='center', fontsize=9)

        # Period drift plot
        ax2 = axes[1]
        dp = g['drift_pairs']
        if dp and not np.isnan(g['drift_cv']):
            centres, periods = zip(*dp)
            ax2.plot(centres, periods, 'o-', color='#2563EB', lw=2, ms=6)
            ax2.axhline(g['period'], color='#DC2626', lw=1.5, ls='--',
                        label=f'Estimated period={g["period"]}')
            ax2.fill_between(centres,
                             g['period'] * 0.85, g['period'] * 1.15,
                             alpha=0.1, color='#059669', label='±15% band')
            ax2.set_xlabel('Sample index')
            ax2.set_ylabel('Local period estimate')
            ax2.set_title(f'Period drift  (CV={g["drift_cv"]:.3f})', fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Drift analysis unavailable',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Period drift', fontweight='bold')

        fig.suptitle(f'Group {gi}: [{", ".join(g["cols"][:5])}{"..." if len(g["cols"])>5 else ""}]',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = f'{save_dir}/group_{gi:02d}_adequacy.png'
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'  [PLOT] saved → {path}')


# MAIN
def main():
    print(f'\nLoading {CFG.csv} ...')
    try:
        df = pd.read_csv(CFG.csv)
    except Exception as e:
        print(f'ERROR: could not read CSV — {e}')
        sys.exit(1)

    drop_cols = list(CFG.drop) + [CFG.timestamp]
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f'  Dropped: {drop_cols}')

    sensor_cols = list(df.columns)
    print(f'  {df.shape[0]} samples × {len(sensor_cols)} sensors')
    print(f'  Sensors: {sensor_cols}')

    train_end = int(len(df) * CFG.train_frac)

    print('\nChecking sensor quality ...')
    all_sq = check_sensor_quality(df.iloc[:train_end], sensor_cols)
    valid_cols = [c for c, v in all_sq.items() if v['passed']]
    print(f'  {len(valid_cols)}/{len(sensor_cols)} sensors passed')

    if len(valid_cols) < 2:
        print('ERROR: fewer than 2 valid sensors — CCAD cannot run on this dataset.')
        sys.exit(1)

    rough_window = max(10, train_end // 500)
    print(f'\nCorrelation window = {rough_window} samples')

    print('\nEstimating global period ...')
    corr_all = multivariate_corr_signal(df.iloc[:train_end], valid_cols, rough_window)
    global_period, global_conf, global_method, _ = estimate_period(corr_all)
    if global_period is not None:
        print(f'  Global period = {global_period} samples  '
              f'(conf={global_conf:.3f}, method={global_method})')
    else:
        print('  Could not estimate global period')

    n_pairs = len(valid_cols) * (len(valid_cols)-1) // 2
    print(f'\nComputing pairwise periods ({len(valid_cols)} sensors, {n_pairs} pairs) ...')
    period_mat, conf_mat = pairwise_periods(df, valid_cols, train_end, rough_window)

    df_tr = df.iloc[:train_end]
    mean_corr_mat = np.zeros((len(valid_cols), len(valid_cols)))
    for i, j in combinations(range(len(valid_cols)), 2):
        a = df_tr[valid_cols[i]].dropna().values
        b = df_tr[valid_cols[j]].dropna().values
        n = min(len(a), len(b))
        if n < 10:
            continue
        mc = abs(np.corrcoef(a[:n], b[:n])[0, 1])
        mean_corr_mat[i, j] = mean_corr_mat[j, i] = mc if not np.isnan(mc) else 0.0

    print('\nGrouping sensors by shared correlation period ...')
    group_indices, ungroupable_indices = group_sensors_by_period(
        valid_cols, period_mat, conf_mat, mean_corr_mat,
        min_corr=CFG.min_corr)

    ungroupable = [valid_cols[i] for i in ungroupable_indices]
    print(f'  Found {len(group_indices)} group(s), {len(ungroupable)} ungroupable')

    print('\nAssessing groups ...')
    group_results = []
    for gi, indices in enumerate(group_indices, 1):
        cols = [valid_cols[i] for i in indices]
        print(f'  Group {gi}: {cols}')
        result = assess_group(df, cols, train_end, rough_window,
                               period_hint=global_period)
        group_results.append(result)
        print(f'    Score: {result["overall"]:.2f}  {result["label"]}')

    group_results.sort(key=lambda x: x['overall'], reverse=True)

    report = format_full_report(df, group_results, ungroupable, all_sq, train_end)
    print(report)

    try:
        with open(CFG.out, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'Report saved → {CFG.out}')
    except Exception as e:
        print(f'Could not save report file: {e}')

    # Plots 
    if CFG.plot:
        print('\nGenerating plots ...')
        save_plots(group_results)

    print('\nDone.')
    return group_results


if __name__ == '__main__':
    main()