# main.py

import torch
import numpy as np
import random
import os
from src.env.switch_env       import SwitchEnvironment
from src.agent.networks        import DQNNetworks
from src.training.train        import Trainer


# ─────────────────────────────────────────
# REPRODUCIBILITY
# set seeds so results are repeatable
# ─────────────────────────────────────────
def set_seeds(seed=42):
    """
    Sets random seeds for reproducibility.
    Same seed = same random numbers = same results.
    Important for debugging and comparison.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    print(f"Seeds set to {seed}")


# ─────────────────────────────────────────
# SYSTEM INFO
# ─────────────────────────────────────────
def print_system_info():
    """
    Prints system and configuration info
    before training starts.
    """
    print("\n════════════════════════════════════════")
    print("         SWITCH RL — DQN PROJECT        ")
    print("════════════════════════════════════════")
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  CUDA Available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Device          : CPU")

    print("════════════════════════════════════════\n")


# ─────────────────────────────────────────
# TRAIN MODE
# ─────────────────────────────────────────
def train():
    """
    Initializes all components and starts training.
    """
    # ── setup ────────────────────────────
    set_seeds(42)
    print_system_info()

    # ── create environment ───────────────
    print("Initializing Switch Environment...")
    env = SwitchEnvironment()
    print("Switch Environment ready.\n")

    # ── create networks ──────────────────
    print("Initializing DQN Networks...")
    networks = DQNNetworks()
    networks.print_architecture()
    print("Networks ready.\n")

    # ── create trainer ───────────────────
    print("Initializing Trainer...")
    trainer = Trainer(env, networks)
    print("Trainer ready.\n")

    # ── start training ───────────────────
    trainer.train()


# ─────────────────────────────────────────
# TEST MODE
# run trained model without training
# ─────────────────────────────────────────
def test(model_path='results/models/best_model.pth'):
    """
    Loads saved model and runs test episodes.
    No training, no epsilon exploration.
    Agent acts purely greedily.

    Parameters:
        model_path : path to saved .pth file
    """
    print("\n════════════════════════════════════════")
    print("         SWITCH RL — TEST MODE          ")
    print("════════════════════════════════════════")

    # ── check model exists ───────────────
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        print("Train first using: python main.py train")
        return

    # ── setup ────────────────────────────
    set_seeds(42)

    # ── create components ────────────────
    env      = SwitchEnvironment()
    networks = DQNNetworks()
    networks.load(model_path)
    print(f"Model loaded from {model_path}\n")

    # ── run 5 test episodes ──────────────
    TEST_EPISODES = 5
    test_rewards  = []

    for episode in range(1, TEST_EPISODES + 1):
        state          = env.reset()
        episode_reward = 0.0
        done           = False
        step           = 0

        print(f"\n── Test Episode {episode} ───────────────")

        while not done:
            step += 1

            # purely greedy — no exploration
            q_values = networks.get_q_values(state)
            action   = int(np.argmax(q_values))
            print(f"    Q-Values → "
                f"LEARN:{q_values[0]:+.3f} | "
                f"EVICT:{q_values[1]:+.3f} | "
                f"FLOOD:{q_values[2]:+.3f} | "
                f"BLOCK:{q_values[3]:+.3f} | "
                f"UNBLOCK:{q_values[4]:+.3f} | "
                f"INC_AGE:{q_values[5]:+.3f} | "
                f"DEC_AGE:{q_values[6]:+.3f}")

            next_state, reward, done, info = env.step(action)

            print(f"  Step {step:3d} | "
                  f"Action: {info['action_name']:<16} | "
                  f"Outcome: {info['outcome']:<24} | "
                  f"Reward: {reward:+.3f} | "
                  f"Situation: {info['situation']}")

            episode_reward += reward
            state           = next_state

        test_rewards.append(episode_reward)
        print(f"\n  Episode {episode} Total Reward: "
              f"{episode_reward:+.3f}")

    # ── test summary ─────────────────────
    print("\n════════════════════════════════════════")
    print("            TEST SUMMARY                ")
    print("════════════════════════════════════════")
    print(f"  Episodes Tested : {TEST_EPISODES}")
    print(f"  Avg Reward      : {np.mean(test_rewards):+.3f}")
    print(f"  Best Reward     : {np.max(test_rewards):+.3f}")
    print(f"  Worst Reward    : {np.min(test_rewards):+.3f}")
    print("════════════════════════════════════════\n")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == '__main__':
    import sys

    # default → train
    # python main.py       → train
    # python main.py train → train
    # python main.py test  → test

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        train()