from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional


# Data containers

@dataclass
class CycleResult:
    cycle_idx:       int
    start:           int
    end:             int
    length:          int
    stat_score_raw:  Optional[float]
    dl_score_raw:    Optional[float] 
    stat_score_norm: Optional[float]
    dl_score_norm:   Optional[float] 
    fused_score:     float 
    is_anomaly:      bool
    detection_type:  str 
    stat_flagged:    bool
    dl_flagged:      bool


@dataclass
class FusionReport:
    stream:       str
    stat_weight:  float
    dl_weight:    float
    threshold:    float
    n_cycles:     int
    n_anomalies:  int
    anomaly_rate: float
    cycles:       List[CycleResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    def anomalous_cycles(self) -> List[CycleResult]:
        return [c for c in self.cycles if c.is_anomaly]


# Normaliser

class RobustNormaliser:
    def __init__(self, lo_pct: float = 5.0, hi_pct: float = 99.0):
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct
        self.lo_: float = 0.0
        self.hi_: float = 1.0

    def fit(self, scores: np.ndarray) -> "RobustNormaliser":
        valid = scores[~np.isnan(scores)]
        if len(valid) == 0:
            self.lo_, self.hi_ = 0.0, 1.0
            return self
        self.lo_ = float(np.percentile(valid, self.lo_pct))
        self.hi_ = float(np.percentile(valid, self.hi_pct))
        if self.hi_ <= self.lo_:
            self.hi_ = self.lo_ + 1e-9
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        out = (scores - self.lo_) / (self.hi_ - self.lo_)
        return np.clip(out, 0.0, 1.0)

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        return self.fit(scores).transform(scores)


# Main fuser

class HybridFuser:

    def __init__(self, stat_weight: float = 0.5, threshold: float = 0.5):
        self.stat_weight = float(np.clip(stat_weight, 0.0, 1.0))
        self.dl_weight   = 1.0 - self.stat_weight
        self.threshold   = threshold
        self._stat_norm  = RobustNormaliser()
        self._dl_norm    = RobustNormaliser()
        self._fitted     = False

    def fit(self,stat_train_scores: np.ndarray, dl_train_scores:   np.ndarray) -> "HybridFuser":
        self._stat_norm.fit(stat_train_scores)
        self._dl_norm.fit(dl_train_scores)
        self._fitted = True
        return self

    @staticmethod
    def _dl_to_cycle_scores(dl_ts_scores: np.ndarray,
                             cycles: List[Tuple[int, int, int]]) -> np.ndarray:
        out = np.full(len(cycles), np.nan)
        n   = len(dl_ts_scores)
        for i, (start, end, _) in enumerate(cycles):
            s = max(0, start)
            e = min(end + 1, n)
            if s >= e:
                continue
            window = dl_ts_scores[s:e]
            valid  = window[~np.isnan(window)]
            if len(valid):
                out[i] = float(np.max(valid))
        return out

    # ------------------------------------------------------------------
    def fuse(self,
             stat_scores:     np.ndarray,
             dl_ts_scores:    np.ndarray,
             cycles:          List[Tuple[int, int, int]],
             stat_flagged:    Optional[set]  = None,
             dl_flagged:      Optional[set]  = None,
             template_cycles: Optional[set]  = None,
             stream:          str            = "stream") -> FusionReport:

        if not self._fitted:
            valid_stat = stat_scores[~np.isnan(stat_scores)]
            self._stat_norm.fit(valid_stat if len(valid_stat) else np.array([0.0, 1.0]))
            dl_cycle_raw = self._dl_to_cycle_scores(dl_ts_scores, cycles)
            valid_dl = dl_cycle_raw[~np.isnan(dl_cycle_raw)]
            self._dl_norm.fit(valid_dl if len(valid_dl) else np.array([0.0, 1.0]))

        if stat_flagged    is None: stat_flagged    = set()
        if dl_flagged      is None: dl_flagged      = set()
        if template_cycles is None: template_cycles = set()

        dl_cycle_raw  = self._dl_to_cycle_scores(dl_ts_scores, cycles)
        stat_norm_all = self._stat_norm.transform(
            np.where(np.isnan(stat_scores), np.nan, stat_scores))
        dl_norm_all   = self._dl_norm.transform(
            np.where(np.isnan(dl_cycle_raw), np.nan, dl_cycle_raw))

        cycle_results = []
        for i, (start, end, length) in enumerate(cycles):
            sn = float(stat_norm_all[i]) if not np.isnan(stat_norm_all[i]) else None
            dn = float(dl_norm_all[i])   if not np.isnan(dl_norm_all[i])   else None

            # Weighted fused score
            if   sn is not None and dn is not None:
                fused = self.stat_weight * sn + self.dl_weight * dn
            elif sn is not None:
                fused = float(sn)
            elif dn is not None:
                fused = float(dn)
            else:
                fused = 0.0

            is_anom = fused >= self.threshold
            sf = i in stat_flagged
            df = i in dl_flagged

            # Anomaly type taxonomy
            if i in template_cycles:
                det_type = "TEMPLATE"
            elif not is_anom and not sf and not df:
                det_type = "NORMAL"
            elif sf and df:
                det_type = "BOTH"
            elif sf and not df:
                det_type = "STAT_ONLY"
            elif df and not sf:
                det_type = "DL_ONLY"
            else:
                det_type = "FUSED"

            cycle_results.append(CycleResult(
                cycle_idx       = i,
                start           = start,
                end             = end,
                length          = length,
                stat_score_raw  = float(stat_scores[i]) if not np.isnan(stat_scores[i]) else None,
                dl_score_raw    = float(dl_cycle_raw[i]) if not np.isnan(dl_cycle_raw[i]) else None,
                stat_score_norm = sn,
                dl_score_norm   = dn,
                fused_score     = float(fused),
                is_anomaly      = bool(is_anom),
                detection_type  = det_type,
                stat_flagged    = sf,
                dl_flagged      = df,
            ))

        n_anom = sum(1 for c in cycle_results if c.is_anomaly)
        n_cyc  = len(cycle_results)

        return FusionReport(
            stream       = stream,
            stat_weight  = self.stat_weight,
            dl_weight    = self.dl_weight,
            threshold    = self.threshold,
            n_cycles     = n_cyc,
            n_anomalies  = n_anom,
            anomaly_rate = float(n_anom / n_cyc) if n_cyc else 0.0,
            cycles       = cycle_results,
        )



def build_stat_scores_array(ccad_out: dict) -> np.ndarray:
    n      = len(ccad_out["all_cycles"])
    scores = np.full(n, np.nan)
    for r in ccad_out["results"]:
        scores[r["cycle_idx"]] = r["max_score"]
    return scores

def dl_flagged_from_scores(dl_ts_scores: np.ndarray,
                            cycles: List[Tuple[int, int, int]],
                            dl_threshold: float) -> set:

    cycle_scores = HybridFuser._dl_to_cycle_scores(dl_ts_scores, cycles)
    return {i for i, s in enumerate(cycle_scores)
            if not np.isnan(s) and s >= dl_threshold}


def print_fusion_summary(report: FusionReport) -> None:
    types = {}
    for c in report.cycles:
        types[c.detection_type] = types.get(c.detection_type, 0) + 1

    print(f"\n{'═'*60}")
    print(f"  FUSION REPORT  —  {report.stream}")
    print(f"{'═'*60}")
    print(f"  Weights       : stat={report.stat_weight:.2f}  dl={report.dl_weight:.2f}")
    print(f"  Threshold     : {report.threshold:.2f}")
    print(f"  Total cycles  : {report.n_cycles}")
    print(f"  Anomalies     : {report.n_anomalies}  ({report.anomaly_rate:.1%})")
    print(f"\n  Type breakdown:")
    for t, n in sorted(types.items()):
        bar = "█" * min(n, 40)
        print(f"    {t:<12} {bar} {n}")

    if report.anomalous_cycles():
        print(f"\n  Anomalous cycles (sorted by fused score):")
        print(f"  {'Idx':<6} {'Start':>6} {'End':>6} {'Stat':>7} {'DL':>7} {'Fused':>7}  Type")
        print(f"  {'─'*60}")
        for c in sorted(report.anomalous_cycles(),
                         key=lambda x: x.fused_score, reverse=True):
            sn = f"{c.stat_score_norm:.3f}" if c.stat_score_norm is not None else "  n/a"
            dn = f"{c.dl_score_norm:.3f}"   if c.dl_score_norm   is not None else "  n/a"
            print(f"  {c.cycle_idx+1:<6} {c.start:>6} {c.end:>6} {sn:>7} {dn:>7} "
                  f"{c.fused_score:>7.3f}  {c.detection_type}")
    print(f"{'═'*60}\n")