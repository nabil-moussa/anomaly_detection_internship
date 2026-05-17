import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from eval_methods import adjust_predicts, calc_point2point, bf_search
from utils import SlidingWindowDataset, adjust_anomaly_scores
import torch


class Predictor:

    def __init__(self, model, window_size, n_features, pred_args,
                 summary_file_name="summary.txt"):
        self.model             = model
        self.window_size       = window_size
        self.n_features        = n_features
        self.dataset           = pred_args["dataset"]
        self.target_dims       = pred_args["target_dims"]
        self.scale_scores      = pred_args["scale_scores"]
        self.q                 = pred_args["q"]
        self.level             = pred_args["level"]
        self.dynamic_pot       = pred_args["dynamic_pot"]
        self.use_mov_av        = pred_args["use_mov_av"]
        self.gamma             = pred_args["gamma"]
        self.save_path         = pred_args["save_path"]
        self.use_vae           = pred_args.get("use_vae", False)
        self.batch_size        = 256
        self.use_cuda          = True
        self.summary_file_name = summary_file_name

    #  Score computation 

    def get_score(self, values):
        print("Predicting and calculating anomaly scores..")
        data   = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds, recons = [], []

        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                if self.use_vae:
                    y_hat, _, __, ___ = self.model(x)
                else:
                    y_hat, _ = self.model(x)

                recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                if self.use_vae:
                    _, window_recon, __, ___ = self.model(recon_x)
                else:
                    _, window_recon = self.model(recon_x)

                preds.append(y_hat.detach().cpu().numpy())
                recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds  = np.concatenate(preds,  axis=0)
        recons = np.concatenate(recons, axis=0)

        actual_full = values.detach().cpu().numpy()[self.window_size: self.window_size + len(preds)]

        if self.target_dims is not None:
            actual_forecast = actual_full[:, self.target_dims]
        else:
            actual_forecast = actual_full

        df_dict = {}

        forecast_scores = np.zeros(len(actual_forecast))
        for i in range(preds.shape[1]):
            df_dict[f"Forecast_{i}"] = preds[:, i]
            df_dict[f"True_{i}"]     = actual_forecast[:, i]
            forecast_scores         += (preds[:, i] - actual_forecast[:, i]) ** 2
        forecast_scores /= preds.shape[1]

        recon_scores = np.zeros(len(actual_full))
        for i in range(recons.shape[1]):
            df_dict[f"Recon_{i}"] = recons[:, i]
            recon_scores         += (recons[:, i] - actual_full[:, i]) ** 2
        recon_scores /= recons.shape[1]

        a_score = (forecast_scores + self.gamma * recon_scores) / (1 + self.gamma)

        if self.scale_scores:
            q75, q25 = np.percentile(a_score, [75, 25])
            iqr      = q75 - q25
            median   = np.median(a_score)
            a_score  = (a_score - median) / (1 + iqr)

        for i in range(recons.shape[1]):
            df_dict[f"A_Score_{i}"] = (recons[:, i] - actual_full[:, i]) ** 2

        df = pd.DataFrame(df_dict)
        df["A_Score_Global"] = a_score
        return df

    #  POT thresholding

    def _run_pot(self, train_scores, test_scores, true_anomalies):
        from spot import SPOT

        print("\n  [POT] Running POT thresholding...")

        median   = float(np.median(train_scores))
        q75, q25 = np.percentile(train_scores, [75, 25])
        iqr      = float(q75 - q25)
        if iqr < 1e-8:
            iqr = float(np.std(train_scores)) + 1e-8

        norm_train = (train_scores - median) / iqr
        norm_test  = (test_scores  - median) / iqr

        # Target: p97 of train in normalised space
        target_norm = float(np.percentile(norm_train, 97.0))

        n_val     = max(500, int(len(norm_train) * 0.15))
        norm_init = norm_train[:-n_val]
        norm_val  = norm_train[-n_val:]

        level_grid = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        q_grid     = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]

        best_pair     = (self.level, self.q)
        best_dist     = float("inf")
        best_norm_thr = target_norm

        for level in level_grid:
            for q in q_grid:
                try:
                    s = SPOT(q)
                    s.fit(norm_init, norm_val)
                    s.initialize(level=level, min_extrema=False, verbose=False)
                    ret      = s.run(dynamic=self.dynamic_pot, with_alarm=False)
                    norm_thr = float(np.mean(ret["thresholds"]))
                    dist     = abs(norm_thr - target_norm)
                    if dist < best_dist:
                        best_dist     = dist
                        best_pair     = (level, q)
                        best_norm_thr = norm_thr
                except Exception:
                    continue

        best_level, best_q = best_pair
        raw_threshold = best_norm_thr * iqr + median

        lo = float(np.percentile(norm_train, 95.0)  * iqr + median)
        hi = float(np.percentile(norm_train, 99.9)  * iqr + median)
        final_threshold = float(np.clip(raw_threshold, lo, hi))

        print(f"  [POT] level={best_level}  q={best_q}  threshold={final_threshold:.6f}")

        pred, p_latency = adjust_predicts(
            test_scores, true_anomalies, final_threshold, calc_latency=True
        )

        if true_anomalies is not None:
            p_t = calc_point2point(pred, true_anomalies)
            return {
                "f1":        float(p_t[0]),
                "precision": float(p_t[1]),
                "recall":    float(p_t[2]),
                "TP":        float(p_t[3]),
                "TN":        float(p_t[4]),
                "FP":        float(p_t[5]),
                "FN":        float(p_t[6]),
                "threshold": float(final_threshold),
                "latency":   float(p_latency),
                "pot_level": float(best_level),
                "pot_q":     float(best_q),
            }
        return {
            "threshold": float(final_threshold),
            "pot_level": float(best_level),
            "pot_q":     float(best_q),
        }

    #  Main entry point

    def predict_anomalies(self, train, test, true_anomalies,
                          load_scores=False, save_output=True):

        # 1. Compute or load scores
        if load_scores:
            print("Loading anomaly scores from disk")
            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df  = pd.read_pickle(f"{self.save_path}/test_output.pkl")
        else:
            train_pred_df = self.get_score(train)
            test_pred_df  = self.get_score(test)

        train_anomaly_scores = train_pred_df["A_Score_Global"].values
        test_anomaly_scores  = test_pred_df["A_Score_Global"].values

        # 2. Per-dataset score adjustment (SMAP channel normalisation)
        if not load_scores:
            train_anomaly_scores = adjust_anomaly_scores(
                train_anomaly_scores, self.dataset, True, self.window_size
            )
            test_anomaly_scores = adjust_anomaly_scores(
                test_anomaly_scores, self.dataset, False, self.window_size
            )
            train_pred_df["A_Score_Global"] = train_anomaly_scores
            test_pred_df["A_Score_Global"]  = test_anomaly_scores

        # 3. POT thresholding
        p_eval = self._run_pot(train_anomaly_scores, test_anomaly_scores, true_anomalies)

        # 4. Optional exponential smoothing
        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = (
                pd.DataFrame(train_anomaly_scores)
                .ewm(span=smoothing_window).mean().values.flatten()
            )
            test_anomaly_scores = (
                pd.DataFrame(test_anomaly_scores)
                .ewm(span=smoothing_window).mean().values.flatten()
            )


        print("Running Best-F1 search (oracle)...")
        if true_anomalies is not None:
            bf_eval = bf_search(
                test_anomaly_scores, true_anomalies,
                start=0.01,
                end=float(np.percentile(test_anomaly_scores, 99.5)),
                step_num=200,
                verbose=False,
            )
        else:
            bf_eval = {}

        # 7. Print results
        PAPER = {
            "SMAP": (0.8906, 0.9123, 0.9013),
            "MSL":  (0.8754, 0.9440, 0.9084),
        }
        sep = "-" * 66
        print(f"\n{sep}")
        print(f"  RESULTS  {self.dataset}")
        print(sep)
        print(f"  {'Method':<22} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print(sep)
        for name, res in [("POT", p_eval), ("Best-F1 (oracle)", bf_eval)]:
            p = res.get("precision", 0)
            r = res.get("recall",    0)
            f = res.get("f1",        0)
            print(f"  {name:<22} {p:>8.4f} {r:>8.4f} {f:>8.4f}")
        if self.dataset in PAPER:
            pp = PAPER[self.dataset]
            print(sep)
            print(f"  {'Paper (MTAD-GAT)':<22} {pp[0]:>8.4f} {pp[1]:>8.4f} {pp[2]:>8.4f}  (Table III)")
            gap = pp[2] - p_eval.get("f1", 0)
            print(f"\n  Gap paper vs POT: {gap:+.4f}  "
                  f"({'within variance' if abs(gap) < 0.03 else 'check config'})")
        print(f"{sep}\n")

        # 8. Serialise summary
        def _floatify(d):
            return {k: float(v) for k, v in d.items()
                    if isinstance(v, (int, float, np.floating, np.integer))}

        summary = {
            "pot_result":     _floatify(p_eval),
            "bf_result":      _floatify(bf_eval),
        }
        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(summary, f, indent=2)

        # 9. Save DataFrames
        if save_output:
            global_epsilon = p_eval["threshold"]

            test_pred_df["A_True_Global"]  = true_anomalies
            train_pred_df["Thresh_Global"] = global_epsilon
            test_pred_df["Thresh_Global"]  = global_epsilon

            train_pred_df["A_Pred_Global"] = (
                train_anomaly_scores >= global_epsilon
            ).astype(int)

            test_preds_global = (test_anomaly_scores >= global_epsilon).astype(int)
            if true_anomalies is not None:
                test_preds_global = adjust_predicts(
                    None, true_anomalies, global_epsilon, pred=test_preds_global
                )
            test_pred_df["A_Pred_Global"] = test_preds_global

            print(f"Saving output -> {self.save_path}/<train/test>_output.pkl")
            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        print("Done.")
        return summary
