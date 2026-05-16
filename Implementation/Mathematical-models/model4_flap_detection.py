"""Model 4: Flap Detection using Z-Score.

Evaluates how anomalous a specific MAC address's port updates are compared to the
rest of the active network nodes.
"""

import numpy as np

def evaluate_zscore_flap(flap_rate_mac, global_flap_rates, threshold=2.0):
    """Checks port behavior against structural statistics.

    Formula: Z_score(mac) = (flap_rate_mac - mu_flap) / sigma_flap
    """
    mu_flap = np.mean(global_flap_rates)
    sigma_flap = np.std(global_flap_rates)

    if sigma_flap == 0:
        z_score = 0.0
    else:
        z_score = (flap_rate_mac - mu_flap) / sigma_flap

    # Final threshold condition
    decision = "BLOCK (Abnormal Activity)" if z_score > threshold else "ALLOW"

    return {
        "z_score": round(z_score, 4),
        "mean_rate": round(mu_flap, 4),
        "std_dev": round(sigma_flap, 4),
        "decision": decision,
    }


if __name__ == "__main__":
    print("\n--- Testing Model 4: Z-Score Flap Detection Engine ---")

    # Sample global environment port-changes per minute context
    global_rates = [1.2, 2.0, 1.5, 0.8, 2.3, 1.1, 14.5]

    # Node 1: Stable system target
    stable_node = evaluate_zscore_flap(flap_rate_mac=1.5, global_flap_rates=global_rates)
    print(f"Stable Host Evaluation -> Z-Score: {stable_node['z_score']} | Action: {stable_node['decision']}")

    # Node 2: Anomalous flapping loop target
    flapping_node = evaluate_zscore_flap(flap_rate_mac=14.5, global_flap_rates=global_rates)
    print(f"Flapping Loop Evaluation -> Z-Score: {flapping_node['z_score']} | Action: {flapping_node['decision']}")