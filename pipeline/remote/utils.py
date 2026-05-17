import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import random
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


#  Reproducible DataLoader workers

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

_g = torch.Generator()
_g.manual_seed(42)


#  Data helpers 

def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    return data, scaler


def get_data_dim(dataset):
    dims = {
        "SMAP":   25,
        "MSL":    55,
        "CUSTOM":  3,
    }
    if dataset in dims:
        return dims[dataset]
    if str(dataset).startswith("machine"):
        return 38
    raise ValueError(f"Unknown dataset: {dataset}")


def get_target_dims(dataset):
    if dataset in ("SMAP", "MSL"):
        return [0]
    if dataset in ("SMD", "CUSTOM"):
        return None
    raise ValueError(f"Unknown dataset: {dataset}")


def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, train_start=0, test_start=0):

    prefix = "datasets"
    if str(dataset).startswith("machine"):
        prefix += "/ServerMachineDataset/processed"
    elif dataset in ("MSL", "SMAP"):
        prefix += "/data/processed"
    elif dataset == "CUSTOM":
        prefix = "datasets/custom"

    train_end = None if max_train_size is None else train_start + max_train_size
    test_end  = None if max_test_size  is None else test_start  + max_test_size

    print(f"Loading dataset: {dataset}")
    x_dim = get_data_dim(dataset)

    with open(os.path.join(prefix, f"{dataset}_train.pkl"), "rb") as f:
        train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end]

    try:
        with open(os.path.join(prefix, f"{dataset}_test.pkl"), "rb") as f:
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end]
    except (KeyError, FileNotFoundError):
        test_data = None

    try:
        with open(os.path.join(prefix, f"{dataset}_test_label.pkl"), "rb") as f:
            test_label = pickle.load(f).reshape(-1)[test_start:test_end]
    except (KeyError, FileNotFoundError):
        test_label = None

    if normalize:
        train_data, scaler = normalize_data(train_data)
        test_data, _       = normalize_data(test_data, scaler=scaler)

    print(f"  train shape: {train_data.shape}")
    print(f"  test shape:  {test_data.shape}")
    print(f"  label shape: {None if test_label is None else test_label.shape}")
    return (train_data, None), (test_data, test_label)


def prepare_custom_dataset(csv_path, label_path, train_ratio=0.6,
                            normalize=True, save_dir="datasets/custom"):
    os.makedirs(save_dir, exist_ok=True)

    df   = pd.read_csv(csv_path, sep=None, engine="python")
    data = df.values.astype(np.float32)
    n    = len(data)

    labels = np.zeros(n, dtype=np.float32)
    ldf    = pd.read_csv(label_path, sep=None, engine="python")
    for _, row in ldf.iterrows():
        labels[int(row["start_row"]):int(row["end_row"])] = 1.0

    split   = int(n * train_ratio)
    x_train = data[:split]
    x_test  = data[split:]
    y_test  = labels[split:]

    if labels[:split].sum() > 0:
        print(f"WARNING: {int(labels[:split].sum())} anomaly timesteps in train split.")

    if normalize:
        scaler  = MinMaxScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test  = scaler.transform(x_test).astype(np.float32)

    with open(f"{save_dir}/CUSTOM_train.pkl",      "wb") as f: pickle.dump(x_train, f)
    with open(f"{save_dir}/CUSTOM_test.pkl",       "wb") as f: pickle.dump(x_test,  f)
    with open(f"{save_dir}/CUSTOM_test_label.pkl", "wb") as f: pickle.dump(y_test,  f)

    print(f"Saved to {save_dir}/")
    print(f"  train: {x_train.shape}  anomalies: {int(labels[:split].sum())}")
    print(f"  test:  {x_test.shape}   anomalies: {int(y_test.sum())}")
    return x_train, x_test, y_test


#  Dataset / DataLoader

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data       = data
        self.window     = window
        self.target_dim = target_dim
        self.horizon    = horizon

    def __getitem__(self, index):
        x = self.data[index: index + self.window]
        y = self.data[index + self.window: index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


def create_data_loaders(train_dataset, batch_size, val_split=0.1,
                        shuffle=True, test_dataset=None):
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = val_loader = test_loader = None

    if val_split == 0.0:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            worker_init_fn=seed_worker, generator=g,
        )
        print(f"train_size: {len(train_dataset)}")
    else:
        dataset_size = len(train_dataset)
        indices      = list(range(dataset_size))
        split        = int(np.floor(val_split * dataset_size))

        rng = np.random.default_rng(42)
        if shuffle:
            rng.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, train_indices),
            batch_size=batch_size, shuffle=True,
            worker_init_fn=seed_worker, generator=g,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, val_indices),
            batch_size=batch_size, shuffle=False,
            worker_init_fn=seed_worker, generator=g,
        )
        print(f"train_size: {len(train_indices)}  val_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            worker_init_fn=seed_worker, generator=g,
        )
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


#  Plotting 

def plot_losses(losses, save_path="", plot=True):
    for split, title in [("train", "Training"), ("val", "Validation")]:
        plt.plot(losses[f"{split}_forecast"], label="Forecast loss")
        plt.plot(losses[f"{split}_recon"],    label="Recon loss")
        plt.plot(losses[f"{split}_total"],    label="Total loss")
        plt.title(f"{title} losses")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.savefig(f"{save_path}/{split}_losses.png", bbox_inches="tight")
        if plot:
            plt.show()
        plt.close()



def load(model, PATH, device="cpu"):
    model.load_state_dict(torch.load(PATH, map_location=device))


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    # per channel normalisation for SMAP 
    if dataset.upper() != "SMAP":
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f"./datasets/data/{dataset.lower()}_train_md.csv")
    else:
        md = pd.read_csv("./datasets/data/labeled_anomalies.csv")
        md = md[md["spacecraft"] == dataset.upper()]

    md = md[md["chan_id"] != "P-2"].sort_values(by=["chan_id"])

    sep_cuma = np.cumsum(md["num_values"].values) - lookback
    s = [0] + sep_cuma.tolist()

    expected_len = int(sep_cuma[-1])
    if abs(expected_len - len(scores)) > 50:
        print(f"WARNING: CSV expects ~{expected_len} scores, got {len(scores)}. "
              f"Channel boundaries may be misaligned.")

    for c_start, c_end in [(s[i], s[i + 1]) for i in range(len(s) - 1)]:
        e_s = adjusted_scores[c_start: c_end + 1].copy()
        e_min, e_max = np.min(e_s), np.max(e_s)
        if e_max - e_min > 1e-8:
            p99 = np.percentile(e_s, 99)
            p01 = np.percentile(e_s,  1)
            if p99 > p01 + 1e-8:
                e_s = np.clip((e_s - p01) / (p99 - p01), 0.0, 10.0)
            else:
                e_s = np.zeros_like(e_s)
        else:
            e_s = np.zeros_like(e_s)
        adjusted_scores[c_start: c_end + 1] = e_s

    return np.clip(adjusted_scores, 0.0, None)


#  SR-based training data cleaning

def _spectral_residual(series):
    eps = 1e-8
    n   = len(series)
    fft           = np.fft.fft(series)
    log_amplitude = np.log(np.abs(fft) + eps)

    avg_log = np.zeros_like(log_amplitude)
    w = 3
    for i in range(n):
        avg_log[i] = np.mean(log_amplitude[max(0, i - w // 2): min(n, i + w // 2 + 1)])

    new_fft  = np.exp(log_amplitude - avg_log) * np.exp(1j * np.angle(fft))
    saliency = np.abs(np.fft.ifft(new_fft))
    return saliency


def _sr_anomaly_scores(series):
    saliency = _spectral_residual(series)
    scores   = np.zeros_like(saliency)
    w = 100
    for i in range(len(saliency)):
        local = saliency[max(0, i - w // 2): min(len(saliency), i + w // 2 + 1)]
        scores[i] = (saliency[i] - np.mean(local)) / (np.std(local) + 1e-8)
    return scores


def clean_training_data(x_train, threshold=3.0):
    T, k           = x_train.shape
    x_cleaned      = x_train.copy()
    total_replaced = 0

    print(f"Applying SR-based data cleaning (threshold={threshold})...")

    for feat in range(k):
        series       = x_train[:, feat].astype(np.float64)
        anomaly_mask = _sr_anomaly_scores(series) > threshold
        n_anom       = anomaly_mask.sum()
        total_replaced += n_anom
        if n_anom == 0:
            continue

        clean_series = series.copy()
        for idx in np.where(anomaly_mask)[0]:
            left = idx - 1
            while left >= 0 and anomaly_mask[left]:
                left -= 1
            right = idx + 1
            while right < T and anomaly_mask[right]:
                right += 1

            if left < 0 and right >= T:
                pass  # entire series anomalous leave as is
            elif left < 0:
                clean_series[idx] = series[right]
            elif right >= T:
                clean_series[idx] = series[left]
            else:
                alpha             = (idx - left) / (right - left)
                clean_series[idx] = (1 - alpha) * series[left] + alpha * series[right]

        x_cleaned[:, feat] = clean_series

    pct = 100 * total_replaced / (T * k)
    print(f"SR cleaning done: replaced {total_replaced} points ({pct:.2f}%) across {k} features.")
    return x_cleaned
