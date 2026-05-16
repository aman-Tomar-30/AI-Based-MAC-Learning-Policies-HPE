"""
model1_dynamic_ttl.py
=====================

Model 1 — Dynamic TTL Adjustment
--------------------------------

    TTL(mac) = TTL_base * alpha * F_activity * F_stability * F_pressure

    F_activity  = 1 + log(1 + tx_count)         # natural log
    F_stability = 1 / (1 + flap_count)
    F_pressure  = 1 - (occupied / capacity)

Regimes (from Slides 2 and 3):
    ACTIVE     : tx_count >= 16
    IDLE       : 1 <= tx_count <= 15
    INACTIVE   : tx_count == 0
    STABLE     : flap_count == 0
    FLAPPING   : 1 <= flap_count <= 9
    UNSTABLE   : flap_count >= 10
    NOT FULL   : occupancy < 0.65
    NEARLY FULL: 0.65 <= occupancy < 0.85
    FULL       : occupancy >= 0.85
"""

from __future__ import annotations

import math


class DynamicTTLModel:
    def __init__(self, ttl_base: float = 300.0, alpha: float = 0.5):
        self.ttl_base = ttl_base
        self.alpha = alpha

    # --- individual factors (kept separate so callers can introspect) ----
    @staticmethod
    def f_activity(tx_count: int) -> float:
        return 1.0 + math.log(1 + tx_count)

    @staticmethod
    def f_stability(flap_count: int) -> float:
        return 1.0 / (1 + flap_count)

    @staticmethod
    def f_pressure(occupied: int, capacity: int) -> float:
        if capacity <= 0:
            return 0.0
        return 1.0 - (occupied / capacity)

    # --- main API --------------------------------------------------------
    def compute_ttl(self, tx_count: int, flap_count: int,
                    occupied: int, capacity: int) -> float:
        return (self.ttl_base
                * self.alpha
                * self.f_activity(tx_count)
                * self.f_stability(flap_count)
                * self.f_pressure(occupied, capacity))

    # --- regime labels (handy for logging / dashboards) ------------------
    @staticmethod
    def activity_state(tx_count: int) -> str:
        if tx_count == 0:
            return "INACTIVE"
        if tx_count <= 15:
            return "IDLE"
        return "ACTIVE"

    @staticmethod
    def stability_state(flap_count: int) -> str:
        if flap_count == 0:
            return "STABLE"
        if flap_count <= 9:
            return "FLAPPING"
        return "UNSTABLE"

    @staticmethod
    def pressure_state(occupied: int, capacity: int) -> str:
        occ = occupied / capacity if capacity else 1.0
        if occ < 0.65:
            return "NOT_FULL"
        if occ < 0.85:
            return "NEARLY_FULL"
        return "FULL"


# ---------------------------------------------------------------------------
# Demo — reproduces Slide 4 worked examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ttl = DynamicTTLModel(ttl_base=300.0, alpha=0.5)

    print("Model 1 — Dynamic TTL Adjustment")
    print("-" * 50)

    ttl1 = ttl.compute_ttl(tx_count=50, flap_count=0, occupied=40, capacity=100)
    print(f"  tx=50, flap=0, occ=40%  -> TTL = {ttl1:7.2f} s  "
          f"(EXTEND, slide ~443)")

    ttl2 = ttl.compute_ttl(tx_count=2, flap_count=4, occupied=90, capacity=100)
    print(f"  tx=2,  flap=4, occ=90%  -> TTL = {ttl2:7.2f} s  "
          f"(SHRINK FAST, slide ~4.4)")

    print()
    print(f"  States for tx=50, flap=0, occ=40%:")
    print(f"    activity  = {ttl.activity_state(50)}")
    print(f"    stability = {ttl.stability_state(0)}")
    print(f"    pressure  = {ttl.pressure_state(40, 100)}")
