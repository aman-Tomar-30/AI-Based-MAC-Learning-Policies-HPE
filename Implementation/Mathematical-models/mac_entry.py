"""
mac_entry.py
============

Shared data structure for a single MAC table row.
Imported by Model 3 (eviction priority) and the manager.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MACEntry:
    """A single MAC table entry."""
    mac: str
    port: int
    tx_count: int = 0                                # frames seen from this MAC
    flap_count: int = 0                              # number of port changes observed
    learned_at: float = field(default_factory=time.time)
    ttl: float = 0.0                                 # computed TTL (seconds)
    last_port_change: float = field(default_factory=time.time)

    def age(self, now: Optional[float] = None) -> float:
        """Seconds since this entry was first learned."""
        return (now if now is not None else time.time()) - self.learned_at
