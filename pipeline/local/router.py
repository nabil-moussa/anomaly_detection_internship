from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path


STAT_THR   = 0.7
HYBRID_THR = 0.50


@dataclass
class RoutingDecision:
    stream:        str
    adequacy:      float
    route:         str          # "statistical"  "hybrid"  "dl"
    stat_weight:   float
    dl_weight:     float
    stat_thr:      float = STAT_THR
    hybrid_thr:    float = HYBRID_THR

    rationale: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RoutingDecision":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)


def route(stream: str, adequacy: float,
          stat_thr: float = STAT_THR,
          hybrid_thr: float = HYBRID_THR) -> RoutingDecision:

    adequacy = float(np.clip(adequacy, 0.0, 1.0))

    if adequacy >= stat_thr:
        decision = "statistical"
        stat_w   = 1.0
        rationale = (
            f"Adequacy {adequacy:.3f} ≥ {stat_thr} → statistical path only. "
            "Periodic structure is strong enough; DL overhead not justified."
        )

    elif adequacy >= hybrid_thr:
        decision = "hybrid"
        stat_w = (adequacy - hybrid_thr) / (stat_thr - hybrid_thr)
        stat_w = float(np.clip(stat_w, 0.0, 1.0))
        rationale = (
            f"Adequacy {adequacy:.3f} in [{hybrid_thr}, {stat_thr}) → hybrid path. "
            f"stat_weight={stat_w:.3f}, dl_weight={1-stat_w:.3f}."
        )

    else:
        decision = "dl"
        stat_w   = 0.0
        rationale = (
            f"Adequacy {adequacy:.3f} < {hybrid_thr} → DL only. "
            "Signal lacks periodic structure; statistical method unreliable."
        )

    return RoutingDecision(
        stream      = stream,
        adequacy    = adequacy,
        route       = decision,
        stat_weight = stat_w,
        dl_weight   = 1.0 - stat_w,
        stat_thr    = stat_thr,
        hybrid_thr  = hybrid_thr,
        rationale   = rationale,
    )


def print_decision(dec: RoutingDecision) -> None:
    bar_len = 30
    filled  = int(dec.adequacy * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"\n{'─'*60}")
    print(f"  ROUTING DECISION  —  {dec.stream}")
    print(f"{'─'*60}")
    print(f"  Adequacy score : [{bar}] {dec.adequacy:.3f}")
    print(f"  Route          : {dec.route.upper()}")
    if dec.route == "hybrid":
        sw_bar = "█" * int(dec.stat_weight * 20) + "░" * int(dec.dl_weight * 20)
        print(f"  Weights        : stat [{sw_bar}] dl")
        print(f"                   {dec.stat_weight:.3f}                    {dec.dl_weight:.3f}")
    print(f"  Rationale      : {dec.rationale}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    for score in [0.82, 0.63, 0.41]:
        d = route("test_stream", score)
        print_decision(d)