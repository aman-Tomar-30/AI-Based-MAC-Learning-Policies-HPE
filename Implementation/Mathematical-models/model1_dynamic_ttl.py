"""
Model 1: Dynamic TTL Adjustment (Shrink or Extend Expiry)
Instead of a fixed aging timer, computing TTL per entry based on real-time factors.
"""

import math


def calculate_dynamic_ttl(
    tx_count, flap_count, occupied, capacity=1000, ttl_base=300, alpha=0.5
):
    """Computes dynamic TTL for an entry.

    Formula: TTL(mac) = TTL_base * alpha * F_activity * F_stability * F_pressure
    """
    # 1. Activity Factor (If Higher tx_count -> entry stays longer)
    f_activity = 1 + math.log(1 + tx_count)

    # 2. Stability Factor (Flapping MAC -> shrinks fast)
    f_stability = 1 / (1 + flap_count)

    # 3. Table Pressure Factor (Full table -> shrinks to free space)
    occupancy_ratio = occupied / capacity
    f_pressure = 1 - occupancy_ratio

    
    ttl = ttl_base * alpha * f_activity * f_stability * f_pressure

    # State classifications 
    activity_state = (
        "ACTIVE"
        if tx_count >= 16
        else ("IDLE" if 1 <= tx_count <= 15 else "INACTIVE")
    )
    stability_state = (
        "STABLE"
        if flap_count == 0
        else ("UNSTABLE" if flap_count >= 10 else "FLAPPING")
    )

    if occupancy_ratio >= 0.85:
        pressure_state = "FULL (Eviction Active)"
    elif occupancy_ratio >= 0.65:
        pressure_state = "NEARLY FULL (Compression Begins)"
    else:
        pressure_state = "NOT FULL (Runs Freely)"

    return {
        "ttl_seconds": round(ttl, 2),
        "states": {
            "activity": activity_state,
            "stability": stability_state,
            "pressure": pressure_state,
        },
    }


if __name__ == "__main__":
    print("--- Testing Model 1: Dynamic TTL ---")

    # Example 1: Active, stable(no flapping), low occupancy (Should EXTEND)
    test1 = calculate_dynamic_ttl(tx_count=50, flap_count=0, occupied=400)
    print(f"Test 1 (Active/Stable): {test1['ttl_seconds']}s -> {test1['states']}")

    # Example 2: Inactive, flapping, high occupancy (Should SHRINK FAST)
    test2 = calculate_dynamic_ttl(tx_count=2, flap_count=4, occupied=900)
    print(f"Test 2 (Unstable/High Load): {test2['ttl_seconds']}s -> {test2['states']}")