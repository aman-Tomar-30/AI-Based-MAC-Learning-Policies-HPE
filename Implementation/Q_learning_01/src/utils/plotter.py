import matplotlib.pyplot as plt

def plot_learning_curve(rewards, filename="results/plots/training_reward.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Reward Evolution (Q-Learning)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()