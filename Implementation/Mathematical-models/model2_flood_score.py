"""
model2_flood_score.py
=====================

Model 2 — FloodScore (Intelligent Flood Suppression)
----------------------------------------------------

    FloodScore = w_seen * P_seen
               + w_tbl  * (1 - table_full_ratio)
               + w_trf  * (1 - traffic_load)

Higher score => allow flood.  Lower score => suppress.

Defaults (0.5 / 0.3 / 0.2) and a 0.5 decision threshold match the deck.
P_seen is a recency-weighted "have we seen this MAC lately?" probability
in [0, 1] — easiest produced by an EWMA over learning events.
"""

from __future__ import annotations

import math
from typing import Tuple


class FloodScoreModel:
    def __init__(self,
                 w_seen: float = 0.5,
                 w_table: float = 0.3,
                 w_traffic: float = 0.2,
                 threshold: float = 0.5):
        total = w_seen + w_table + w_traffic
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(f"Weights must sum to 1.0 (got {total})")
        self.w_seen = w_seen
        self.w_table = w_table
        self.w_traffic = w_traffic
        self.threshold = threshold

    def score(self, p_seen: float,
              table_full_ratio: float,
              traffic_load: float) -> float:
        p_seen = max(0.0, min(1.0, p_seen))
        table_full_ratio = max(0.0, min(1.0, table_full_ratio))
        traffic_load = max(0.0, min(1.0, traffic_load))
        return (self.w_seen * p_seen
                + self.w_table * (1 - table_full_ratio)
                + self.w_traffic * (1 - traffic_load))

    def decide(self, p_seen: float,
               table_full_ratio: float,
               traffic_load: float) -> Tuple[str, float]:
        s = self.score(p_seen, table_full_ratio, traffic_load)
        return ("FLOOD" if s >= self.threshold else "SUPPRESS"), s


# ---------------------------------------------------------------------------
# Demo — reproduces Slide 6 worked examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fs = FloodScoreModel()

    print("Model 2 — FloodScore")
    print("-" * 50)

    d1, s1 = fs.decide(p_seen=0.0, table_full_ratio=0.8, traffic_load=0.7)
    print(f"  new MAC, 80% full, 70% traffic  -> "
          f"score={s1:.3f}  decision={d1}   (slide 0.12, SUPPRESS)")

    d2, s2 = fs.decide(p_seen=0.8, table_full_ratio=0.3, traffic_load=0.1)
    print(f"  known MAC, 30% full, 10% traffic -> "
          f"score={s2:.3f}  decision={d2}   (slide 0.79, FLOOD)")
