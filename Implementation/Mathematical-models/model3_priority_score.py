"""
model3_priority_score.py
========================

Model 3 — Entry Priority Score (Eviction Ranking)
-------------------------------------------------

    Priority(mac) = w1 * (age / TTL_max)
                  + w2 * (1 / tx_count)
                  + w3 * flap_count

Higher Priority => evict first.

Defaults (w1=1, w2=1, w3=0.1, TTL_max=300) reproduce the slide's
worked numbers exactly:
    MAC-A: age=280, tx=1,  flaps=3 -> 0.93 + 1.00 + 0.30 = 2.23
    MAC-B: age=150, tx=20, flaps=0 -> 0.50 + 0.05 + 0.00 = 0.55
    MAC-C: age=290, tx=5,  flaps=1 -> 0.97 + 0.20 + 0.10 = 1.27
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from mac_entry import MACEntry


class PriorityScoreModel:
    def __init__(self, w1: float = 1.0, w2: float = 1.0, w3: float = 0.1,
                 ttl_max: float = 300.0):
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.ttl_max = ttl_max

    def score(self, age: float, tx_count: int, flap_count: int) -> float:
        age_term = age / self.ttl_max if self.ttl_max > 0 else 0.0
        tx_term = 1.0 / tx_count if tx_count > 0 else 1.0     # never-used = worst
        return self.w1 * age_term + self.w2 * tx_term + self.w3 * flap_count

    def rank(self, entries: Iterable[MACEntry],
             now: Optional[float] = None) -> List[Tuple[MACEntry, float]]:
        """Return entries sorted highest-priority (evict first) -> lowest."""
        scored = [(e, self.score(e.age(now), e.tx_count, e.flap_count))
                  for e in entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def pick_victims(self, entries: Iterable[MACEntry], k: int = 1,
                     now: Optional[float] = None) -> List[MACEntry]:
        return [e for e, _ in self.rank(entries, now)[:k]]


# ---------------------------------------------------------------------------
# Demo — reproduces Slide 7 worked example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    now = time.time()
    entries = [
        MACEntry("MAC-A", port=1, tx_count=1,  flap_count=3,
                 learned_at=now - 280),
        MACEntry("MAC-B", port=2, tx_count=20, flap_count=0,
                 learned_at=now - 150),
        MACEntry("MAC-C", port=3, tx_count=5,  flap_count=1,
                 learned_at=now - 290),
    ]

    pri = PriorityScoreModel(w1=1.0, w2=1.0, w3=0.1, ttl_max=300.0)

    print("Model 3 — Entry Priority Score")
    print("-" * 50)
    for entry, score in pri.rank(entries, now=now):
        print(f"  {entry.mac}: age={entry.age(now):5.1f}s  "
              f"tx={entry.tx_count:>2}  flaps={entry.flap_count}  "
              f"->  priority = {score:.3f}")
    print("  Eviction order (slide): MAC-A first, MAC-C second, MAC-B last.")
