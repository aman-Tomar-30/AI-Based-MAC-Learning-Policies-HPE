"""
mac_table_manager.py
====================

Optional glue layer that wires the four standalone models against a live
MAC table. Drop this into a Mininet/Ryu/SDN controller to get periodic
TTL refresh, intelligent flood decisions, priority-based eviction, and
statistical flap detection in one object.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from mac_entry import MACEntry
from model1_dynamic_ttl import DynamicTTLModel
from model2_flood_score import FloodScoreModel
from model3_priority_score import PriorityScoreModel
from model4_flap_detection import FlapDetectionModel


class MACTableManager:
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.table: Dict[str, MACEntry] = {}
        self.ttl_model = DynamicTTLModel()
        self.flood_model = FloodScoreModel()
        self.priority_model = PriorityScoreModel()
        self.flap_model = FlapDetectionModel()

    # ---- TTL refresh pass (call on every aging tick) --------------------
    def refresh_ttls(self) -> None:
        occ = len(self.table)
        for entry in self.table.values():
            entry.ttl = self.ttl_model.compute_ttl(
                entry.tx_count, entry.flap_count, occ, self.capacity)

    # ---- Flood decision for an unknown destination ----------------------
    def should_flood(self, dst_mac: str,
                     traffic_load: float,
                     recency_seconds: float = 60.0) -> Tuple[str, float]:
        # crude P_seen: 1.0 if seen within window, decaying linearly to 0
        p_seen = 0.0
        e = self.table.get(dst_mac)
        if e is not None:
            age = e.age()
            p_seen = max(0.0, 1.0 - age / recency_seconds)
        full_ratio = len(self.table) / self.capacity
        return self.flood_model.decide(p_seen, full_ratio, traffic_load)

    # ---- Eviction when table is full ------------------------------------
    def evict_if_needed(self, k: int = 1) -> List[MACEntry]:
        if len(self.table) < self.capacity:
            return []
        victims = self.priority_model.pick_victims(self.table.values(), k)
        for v in victims:
            self.table.pop(v.mac, None)
        return victims

    # ---- Statistical flap detection -------------------------------------
    def detect_flapping(self, window_seconds: float = 60.0) -> List[str]:
        rates = {m: self.flap_model.flap_rate(e.flap_count, window_seconds)
                 for m, e in self.table.items()}
        return self.flap_model.anomalies(rates)


# ---------------------------------------------------------------------------
# Quick end-to-end demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    mgr = MACTableManager(capacity=10)   # 4 entries -> ~40% occupancy
    now = time.time()

    mgr.table = {
        "MAC-A": MACEntry("MAC-A", port=1, tx_count=1,  flap_count=3,
                          learned_at=now - 280),
        "MAC-B": MACEntry("MAC-B", port=2, tx_count=20, flap_count=0,
                          learned_at=now - 150),
        "MAC-C": MACEntry("MAC-C", port=3, tx_count=5,  flap_count=1,
                          learned_at=now - 290),
        "MAC-D": MACEntry("MAC-D", port=4, tx_count=50, flap_count=0,
                          learned_at=now - 30),
    }

    mgr.refresh_ttls()
    print("After TTL refresh:")
    for e in mgr.table.values():
        print(f"  {e.mac}: TTL = {e.ttl:7.2f} s   "
              f"(tx={e.tx_count}, flaps={e.flap_count})")

    print()
    decision, score = mgr.should_flood("MAC-Z", traffic_load=0.4)
    print(f"Flood unknown MAC-Z (table 40% full, load 40%): "
          f"score={score:.3f} -> {decision}")

    print()
    # Force eviction by filling the table to capacity
    for i in range(6):
        mac = f"FILL-{i:02x}"
        mgr.table[mac] = MACEntry(mac, port=10 + i, tx_count=0, flap_count=0,
                                  learned_at=now - 200)
    victims = mgr.evict_if_needed(k=1)
    print(f"Evicted: {[v.mac for v in victims]}")
