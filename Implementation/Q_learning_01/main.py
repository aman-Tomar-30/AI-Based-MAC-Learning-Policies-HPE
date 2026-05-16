from training.train import run_training
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_performance(rewards_history):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, color='blue', alpha=0.3, label="Raw Reward")

    rolling_avg = pd.Series(rewards_history).rolling(window=50).mean()
    plt.plot(rolling_avg, color='red', linewidth=2, label="Learning Trend")
    plt.title("Agent Learning Progress")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)

def plot_q_heatmap(q_table, encoder):
    plt.figure(figsize=(12, 6))
    actions = ['LEARN', 'EVICT', 'FLOOD', 'ASSIGN', 'BLOCK', 'UPDATE']
    scenarios = [f"State_{i}" for i in range(len(q_table))]
    
    sns.heatmap(q_table, annot=True, fmt=".1f", cmap="YlGnBu", 
                xticklabels=actions, yticklabels=scenarios)
    plt.title("Q-Table")
    plt.show()

if __name__ == "__main__":
    DATA_FILE = "data/raw/traffic_scenarios.csv"
    
    agent, encoder, rewards_history = run_training(DATA_FILE)
    
    plot_performance(rewards_history)
    
    plot_q_heatmap(agent.q_table, encoder)
    