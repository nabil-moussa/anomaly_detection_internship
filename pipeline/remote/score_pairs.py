import pickle
import numpy as np
from itertools import combinations

with open('datasets/data/processed/SMAP_train.pkl', 'rb') as f:
    train = pickle.load(f).reshape(-1, 25)
with open('datasets/data/processed/SMAP_test.pkl', 'rb') as f:
    test = pickle.load(f).reshape(-1, 25)
with open('datasets/data/processed/SMAP_test_label.pkl', 'rb') as f:
    labels = pickle.load(f).reshape(-1)

WINDOW  = 128
T_train = len(train)
all_data = np.concatenate([train, test], axis=0).astype(np.float64)
T_total  = len(all_data)

def fast_rolling_corr(x, y, window):
    T = len(x)
    out = np.zeros(T)
    
    cs_x  = np.cumsum(x)
    cs_y  = np.cumsum(y)
    cs_x2 = np.cumsum(x**2)
    cs_y2 = np.cumsum(y**2)
    cs_xy = np.cumsum(x * y)

    # rolling sums from index window..T
    sum_x  = cs_x[window-1:]  - np.concatenate([[0], cs_x[:-(window)]])
    sum_y  = cs_y[window-1:]  - np.concatenate([[0], cs_y[:-(window)]])
    sum_x2 = cs_x2[window-1:] - np.concatenate([[0], cs_x2[:-(window)]])
    sum_y2 = cs_y2[window-1:] - np.concatenate([[0], cs_y2[:-(window)]])
    sum_xy = cs_xy[window-1:] - np.concatenate([[0], cs_xy[:-(window)]])

    mean_x = sum_x / window
    mean_y = sum_y / window
    cov    = sum_xy / window - mean_x * mean_y
    var_x  = np.maximum(sum_x2 / window - mean_x**2, 0)
    var_y  = np.maximum(sum_y2 / window - mean_y**2, 0)
    denom  = np.sqrt(var_x * var_y)

    valid = denom > 1e-9
    corr  = np.where(valid, np.abs(cov / np.where(valid, denom, 1)), 0.0)

    # corr has length T - window + 1, place starting at index window-1
    out[window-1:] = corr
    return out
pair_stats = []
pairs = list(combinations(range(25), 2))
print(f"Scoring {len(pairs)} pairs...", flush=True)

for idx, (i, j) in enumerate(pairs):
    if idx % 50 == 0:
        print(f"  {idx}/{len(pairs)}...", flush=True)

    rolling = fast_rolling_corr(all_data[:, i], all_data[:, j], WINDOW)

    train_corr = rolling[:T_train]
    test_corr  = rolling[T_train:]

    if train_corr.std() < 1e-3:
        continue

    test_labels = labels[:len(test_corr)]
    normal_corr = test_corr[test_labels == 0]
    anom_corr   = test_corr[test_labels == 1]

    if len(anom_corr) < 10 or len(normal_corr) < 10:
        continue

    sep = abs(anom_corr.mean() - normal_corr.mean()) / (test_corr.std() + 1e-9)

    pair_stats.append({
        'pair':        (i, j),
        'sep_score':   sep,
        'train_std':   float(train_corr.std()),
        'anom_mean':   float(anom_corr.mean()),
        'normal_mean': float(normal_corr.mean()),
        'train_mean':  float(train_corr.mean()),
    })

pair_stats.sort(key=lambda x: x['sep_score'], reverse=True)

print(f"\nTop 15 most informative pairs:")
print(f"{'Pair':<12} {'Sep score':>10} {'Train std':>10} "
      f"{'Normal corr':>12} {'Anom corr':>12}")
print('-' * 60)
for p in pair_stats[:15]:
    print(f"{str(p['pair']):<12} {p['sep_score']:>10.4f} "
          f"{p['train_std']:>10.4f} {p['normal_mean']:>12.4f} "
          f"{p['anom_mean']:>12.4f}")

print(f"\nAll pairs with sep_score > 0.1:")
for p in pair_stats:
    if p['sep_score'] > 0.1:
        print(f"  {p['pair']}  sep={p['sep_score']:.4f}  "
              f"normal={p['normal_mean']:.4f}  anom={p['anom_mean']:.4f}")