import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ─────────────────────────────────────────
# LOAD TRAINING LOG
# ─────────────────────────────────────────
def load_log(path=None):
    if path is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(BASE_DIR, 'results', 'logs', 'training_csv.csv')    
    episodes = []
    rewards  = []
    losses   = []
    epsilons = []

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))
            losses.append(float(row['avg_loss']))
            epsilons.append(float(row['epsilon']))

    return (
        np.array(episodes),
        np.array(rewards),
        np.array(losses),
        np.array(epsilons)
    )


# ─────────────────────────────────────────
# ROLLING AVERAGE
# smooths noisy reward curve
# ─────────────────────────────────────────
def rolling_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(np.mean(data[start:i+1]))
    return np.array(result)


# ─────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────
def plot(episodes, rewards, losses, epsilons):

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        'DQN Training Results — Switch Flood Reduction',
        fontsize=14,
        fontweight='bold'
    )

    # ── Plot 1 — Reward Curve ────────────
    ax1 = axes[0]
    ax1.plot(
        episodes, rewards,
        color='lightblue',
        alpha=0.4,
        linewidth=0.8,
        label='Raw Reward'
    )
    ax1.plot(
        episodes, rolling_avg(rewards, window=20),
        color='blue',
        linewidth=2,
        label='Rolling Avg (20 ep)'
    )
    ax1.axhline(
        y=np.max(rewards),
        color='green',
        linestyle='--',
        linewidth=1,
        label=f'Best: {np.max(rewards):.1f}'
    )
    ax1.axhline(
        y=np.mean(rewards),
        color='orange',
        linestyle='--',
        linewidth=1,
        label=f'Mean: {np.mean(rewards):.1f}'
    )
    ax1.set_title('Episode Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(episodes))

    # ── Plot 2 — Loss Curve ──────────────
    ax2 = axes[1]
    ax2.plot(
        episodes, losses,
        color='salmon',
        alpha=0.4,
        linewidth=0.8,
        label='Raw Loss'
    )
    ax2.plot(
        episodes, rolling_avg(losses, window=20),
        color='red',
        linewidth=2,
        label='Rolling Avg (20 ep)'
    )
    ax2.set_title('Training Loss (MSE)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(episodes))

    # ── Plot 3 — Epsilon Decay ───────────
    ax3 = axes[2]
    ax3.plot(
        episodes, epsilons,
        color='green',
        linewidth=2,
        label='Epsilon'
    )
    ax3.axhline(
        y=0.05,
        color='red',
        linestyle='--',
        linewidth=1,
        label='Min Epsilon (0.05)'
    )
    ax3.set_title('Epsilon Decay (Exploration vs Exploitation)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, len(episodes))
    ax3.set_ylim(0, 1.1)

    # ── save and show ────────────────────
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(
        'results/plots/training_results.png',
        dpi=150,
        bbox_inches='tight'
    )
    print("Plot saved to results/plots/training_results.png")
    plt.show()


# ─────────────────────────────────────────
# PRINT SUMMARY STATS
# ─────────────────────────────────────────
def print_summary(rewards, losses, epsilons):
    print("\n════════════════════════════════════════")
    print("         TRAINING SUMMARY               ")
    print("════════════════════════════════════════")
    print(f"  Total Episodes  : {len(rewards)}")
    print(f"  Best Reward     : {np.max(rewards):+.3f}")
    print(f"  Worst Reward    : {np.min(rewards):+.3f}")
    print(f"  Avg Reward      : {np.mean(rewards):+.3f}")
    print(f"  Final Avg (last 100): "
          f"{np.mean(rewards[-100:]):+.3f}")
    print(f"  Initial Loss    : {losses[0]:.4f}")
    print(f"  Final Loss      : {losses[-1]:.4f}")
    print(f"  Loss Reduction  : "
          f"{((losses[0]-losses[-1])/losses[0]*100):.1f}%")
    print(f"  Final Epsilon   : {epsilons[-1]:.4f}")
    print("════════════════════════════════════════\n")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    episodes, rewards, losses, epsilons = load_log()
    print_summary(rewards, losses, epsilons)
    plot(episodes, rewards, losses, epsilons)