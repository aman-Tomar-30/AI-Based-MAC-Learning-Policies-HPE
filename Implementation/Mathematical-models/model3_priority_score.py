"""Model 3: Entry Priority Score (Who Gets Evicted First?)

Determines deterministic, line-rate eviction metrics rather than relying on standard
oldest-first or random selection policies.
"""


def calculate_eviction_priority(age, ttl_max, tx_count, flap_count, weights=[0.4, 0.3, 0.3]):
    """Computes eviction priority score. Higher Priority = Evict this entry first.

    Formula: Priority(mac) = w1*(age/TTL_max) + w2*(1/tx_count) + w3*flap_count
    """
    w1, w2, w3 = weights

    term_age = age / ttl_max if ttl_max > 0 else 0
    term_tx = 1 / tx_count if tx_count > 0 else float("inf")

    priority_score = (w1 * term_age) + (w2 * term_tx) + (w3 * flap_count)
    return round(priority_score, 4)


if __name__ == "__main__":
    print("\n--- Testing Model 3: Cache Eviction Priority Engine ---")

    
    custom_weights = [1.0, 1.0, 0.1]

    mac_a = calculate_eviction_priority(
        age=280, ttl_max=300, tx_count=1, flap_count=3, weights=custom_weights
    )
    mac_b = calculate_eviction_priority(
        age=150, ttl_max=300, tx_count=20, flap_count=0, weights=custom_weights
    )
    mac_c = calculate_eviction_priority(
        age=290, ttl_max=300, tx_count=5, flap_count=1, weights=custom_weights
    )

    results = [("MAC-A", mac_a), ("MAC-B", mac_b), ("MAC-C", mac_c)]
    # Sort by priority score descending
    results.sort(key=lambda x: x[1], reverse=True)

    for name, score in results:
        print(f"Entry ID: {name} | Eviction Priority Score: {score}")