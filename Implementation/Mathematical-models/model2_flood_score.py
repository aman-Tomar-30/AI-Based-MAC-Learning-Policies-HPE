"""Model 2: Flood Suppression Score (Avoid Flooding)

Weighs MAC confidence, table pressure, and traffic load to suppress unnecessary
broadcasts.
"""


def evaluate_flood_suppression(
    p_seen, occupied, capacity=1000, current_pps=0, max_pps=100, weights=[0.5, 0.3, 0.2]
):
    """Calculates FloodScore to mitigate broadcast storms.

    Formula: FloodScore(mac) = w1*P_seen + w2*(1 - P_table_full) + w3*(1 -
    P_high_traffic)
    """
    w1, w2, w3 = weights
    p_table_full = occupied / capacity
    p_high_traffic = current_pps / max_pps

    
    flood_score = (
        (w1 * p_seen) + (w2 * (1 - p_table_full)) + (w3 * (1 - p_high_traffic))
    )

    
    action = "FLOOD ALLOWED" if flood_score > 0.6 else "SUPPRESS FLOOD"

    return {"flood_score": round(flood_score, 4), "action": action}


if __name__ == "__main__":
    print("\n--- Testing Model 2: Flood Suppression Score ---")

    # Example 1: New MAC entry, table 80% full, traffic 70% full
    test1 = evaluate_flood_suppression(
        p_seen=0, occupied=800, current_pps=70, max_pps=100
    )
    print(f"Test 1 (High Load, New MAC): Score {test1['flood_score']} -> Action: {test1['action']}")

    # Example 2: Known MAC recently, table 30% full, low traffic
    test2 = evaluate_flood_suppression(
        p_seen=0.8, occupied=300, current_pps=10, max_pps=100
    )
    print(f"Test 2 (Stable, Low Load): Score {test2['flood_score']} -> Action: {test2['action']}")