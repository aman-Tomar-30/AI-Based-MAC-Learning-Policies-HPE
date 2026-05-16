"""
model4_flap_detection.py
========================

Model 4 — Flap Detection via Z-Score
------------------------------------

    Z(mac) = (flap_rate_mac - mu) / sigma

flap_rate_mac is the number of port changes per minute for one MAC.
mu and sigma are computed across all MACs currently being observed.

Z > z_threshold (default 3.0) flags the MAC as anomalously flapping.
A small-sample guard returns Z = 0 when fewer than `min_samples` MACs
are available or when sigma == 0.
"""

from __future__ import annotations

import statistics
from typing import Dict, List


class FlapDetectionModel:
    def __init__(self, z_threshold: float = 3.0, min_samples: int = 5):
        self.z_threshold = z_threshold
        self.min_samples = min_samples

    @staticmethod
    def flap_rate(flap_count: int, window_seconds: float) -> float:
        """Convert a raw flap count over a window to flaps per minute."""
        if window_seconds <= 0:
            return 0.0
        return flap_count / (window_seconds / 60.0)

    def z_scores(self, flap_rates: Dict[str, float]) -> Dict[str, float]:
        rates = list(flap_rates.values())
        if len(rates) < self.min_samples:
            return {m: 0.0 for m in flap_rates}
        mu = statistics.fmean(rates)
        sigma = statistics.pstdev(rates)
        if sigma == 0:
            return {m: 0.0 for m in flap_rates}
        return {m: (r - mu) / sigma for m, r in flap_rates.items()}

    def anomalies(self, flap_rates: Dict[str, float]) -> List[str]:
        return [m for m, z in self.z_scores(flap_rates).items()
                if z > self.z_threshold]


# ---------------------------------------------------------------------------
# Demo — one obvious offender in a population of stable MACs
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    flap_rates = {
        "AA:AA:01": 0.2,
        "AA:AA:02": 0.3,
        "AA:AA:03": 0.1,
        "AA:AA:04": 0.4,
        "AA:AA:05": 0.2,
        "AA:AA:06": 15.0,    # the loop offender
    }

    fd = FlapDetectionModel(z_threshold=2.0)
    zs = fd.z_scores(flap_rates)

    print("Model 4 — Flap Detection via Z-Score")
    print("-" * 50)
    for mac, z in zs.items():
        tag = "  <-- ANOMALY" if z > fd.z_threshold else ""
        print(f"  {mac}  rate={flap_rates[mac]:>5.2f} flaps/min  "
              f"Z={z:+.3f}{tag}")
    print(f"  Flagged MACs: {fd.anomalies(flap_rates)}")
