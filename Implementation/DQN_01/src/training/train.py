import os
import sys
import torch
import numpy as np

# ABSOLUTE PATHS
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')
LOGS_DIR   = os.path.join(BASE_DIR, 'results', 'logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

from src.config import (
    GAMMA,
    BATCH_SIZE,
    MAX_EPISODES,
    MAX_STEPS,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    TARGET_UPDATE_N
)
from src.actions             import get_action_name, NUM_ACTIONS
from src.agent.replay_buffer import ReplayBuffer
from src.training.target_update import TargetUpdater
from src.rewards             import episode_summary


# ─────────────────────────────────────────
# LOGGER
# terminal → only episode summary lines
# log file → everything
# ─────────────────────────────────────────
class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        # everything goes to log file
        self.log.write(message)
        self.log.flush()

        # only episode summary line goes to terminal
        if 'Episode' in message and '/' in message:
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class Trainer:
    """
    Main training loop for DQN.
    Connects environment, agent, replay buffer,
    networks, and target updater together.
    """

    def __init__(self, env, networks):
        self.env            = env
        self.networks       = networks
        self.replay_buffer  = ReplayBuffer()
        self.target_updater = TargetUpdater(networks)

        # ── epsilon ──────────────────────────
        self.epsilon        = EPSILON_START

        # ── tracking ─────────────────────────
        self.total_steps    = 0
        self.training_steps = 0
        self.episode_rewards= []
        self.episode_losses = []
        self.best_reward    = float('-inf')

    # ─────────────────────────────────────────
    # SELECT ACTION — epsilon greedy
    # ─────────────────────────────────────────
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, NUM_ACTIONS)
        else:
            q_values = self.networks.get_q_values(state)
            return int(np.argmax(q_values))

    # ─────────────────────────────────────────
    # TRAIN STEP — one mini batch update
    # ─────────────────────────────────────────
    def train_step(self):
        # ── sample mini batch ────────────────
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(BATCH_SIZE)

        # ── convert to tensors ───────────────
        device        = self.networks.device
        states_t      = torch.FloatTensor(states).to(device)
        actions_t     = torch.LongTensor(actions).to(device)
        rewards_t     = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t       = torch.FloatTensor(dones).to(device)

        # ── predicted Q — prediction network ─
        q_predicted = self.networks.get_predicted_q(
            states_t, actions_t
        )  # (batch, 1)

        # ── target Q — target network ────────
        max_q_next = self.networks.get_target_q(
            next_states_t
        )  # (batch, 1)

        # ── bellman equation ─────────────────
        y_i = rewards_t.unsqueeze(1) + \
              GAMMA * max_q_next * (1 - dones_t.unsqueeze(1))

        # ── loss ─────────────────────────────
        loss = self.networks.loss_fn(q_predicted, y_i.detach())

        # ── backpropagation ──────────────────
        self.networks.optimizer.zero_grad()
        loss.backward()

        # ── gradient clipping ────────────────
        torch.nn.utils.clip_grad_norm_(
            self.networks.prediction_network.parameters(),
            max_norm=1.0
        )

        # ── weight update ────────────────────
        self.networks.optimizer.step()

        # ── target network check ─────────────
        self.training_steps += 1
        updated = self.target_updater.step()

        if updated:
            print(f"    Target network updated at "
                  f"training step {self.training_steps}")

        return loss.item()

    # ─────────────────────────────────────────
    # RUN ONE EPISODE
    # ─────────────────────────────────────────
    def run_episode(self, episode_num):
        state          = self.env.reset()
        episode_reward = 0.0
        episode_losses = []
        done           = False
        step           = 0

        while not done and step < MAX_STEPS:
            step             += 1
            self.total_steps += 1

            # ── select action ────────────────
            action = self.select_action(state)

            # ── environment step ─────────────
            next_state, reward, done, info = self.env.step(action)

            # ── store transition ─────────────
            self.replay_buffer.store(
                state, action, reward, next_state, done
            )

            # ── train if buffer ready ────────
            if self.replay_buffer.is_ready():
                loss = self.train_step()
                episode_losses.append(loss)

            # ── log every 50 steps ───────────
            if step % 50 == 0:
                q_values = self.networks.get_q_values(state)
                print(f"  Ep {episode_num:4d} | "
                    f"Step {step:4d} | "
                    f"Action: {info['action_name']:<16} | "
                    f"Outcome: {info['outcome']:<24} | "
                    f"Reward: {reward:+.3f} | "
                    f"Situation: {info['situation']:<15} | "
                    f"ε: {self.epsilon:.3f}")
                print(f"    Q-Values → "
                    f"LEARN:{q_values[0]:+.3f} | "
                    f"EVICT:{q_values[1]:+.3f} | "
                    f"FLOOD:{q_values[2]:+.3f} | "
                    f"BLOCK:{q_values[3]:+.3f} | "
                    f"UNBLOCK:{q_values[4]:+.3f} | "
                    f"INC_AGE:{q_values[5]:+.3f} | "
                    f"DEC_AGE:{q_values[6]:+.3f}")

            episode_reward += reward
            state           = next_state

        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        return episode_reward, avg_loss, step

    # ─────────────────────────────────────────
    # DECAY EPSILON
    # ─────────────────────────────────────────
    def decay_epsilon(self):
        self.epsilon = max(
            EPSILON_END,
            self.epsilon * EPSILON_DECAY
        )

    # ─────────────────────────────────────────
    # SAVE BEST MODEL
    # ─────────────────────────────────────────
    def save_best(self, episode_reward, episode_num):
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.networks.save(
                os.path.join(MODELS_DIR, 'best_model.pth')
            )
            print(f"  ★ New best reward: {self.best_reward:.3f} "
                  f"at episode {episode_num}")

    # ─────────────────────────────────────────
    # TRAIN — main loop
    # ─────────────────────────────────────────
    def train(self):

        # ── setup logger ─────────────────────
        log_path   = os.path.join(LOGS_DIR, 'training_log.txt')
        logger     = Logger(log_path)
        sys.stdout = logger

        print("\n════════════════════════════════════════")
        print("          DQN TRAINING STARTED          ")
        print("════════════════════════════════════════")
        self.networks.print_architecture()

        # ── CSV log ──────────────────────────
        csv_path = os.path.join(LOGS_DIR, 'training_csv.csv')
        csv_file = open(csv_path, 'w')
        csv_file.write("episode,reward,avg_loss,steps,epsilon\n")
        csv_file.flush()

        for episode in range(1, MAX_EPISODES + 1):

            # ── run episode ──────────────────
            episode_reward, avg_loss, steps = \
                self.run_episode(episode)

            # ── decay epsilon ────────────────
            self.decay_epsilon()

            # ── track ────────────────────────
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(avg_loss)

            # ── save best ────────────────────
            self.save_best(episode_reward, episode)

            # ── episode summary ──────────────
            stats = self.env.get_episode_stats()

            print(f"\nEpisode {episode:4d}/{MAX_EPISODES} | "
                  f"Reward: {episode_reward:+8.3f} | "
                  f"Avg Loss: {avg_loss:.5f} | "
                  f"Steps: {steps:3d} | "
                  f"ε: {self.epsilon:.3f} | "
                  f"Buffer: {len(self.replay_buffer):,}")

            episode_summary(
                episode_reward,
                steps,
                stats['outcome_log']
            )

            # ── write CSV ────────────────────
            csv_file.write(
                f"{episode},"
                f"{episode_reward:.4f},"
                f"{avg_loss:.6f},"
                f"{steps},"
                f"{self.epsilon:.4f}\n"
            )
            csv_file.flush()

            # ── checkpoint every 100 ─────────
            if episode % 100 == 0:
                path = os.path.join(
                    MODELS_DIR,
                    f'checkpoint_ep{episode}.pth'
                )
                self.networks.save(path)
                self._print_progress(episode)

        # ── training complete ─────────────────
        csv_file.close()
        self.networks.save(
            os.path.join(MODELS_DIR, 'final_model.pth')
        )

        print("\n════════════════════════════════════════")
        print("         DQN TRAINING COMPLETE          ")
        print("════════════════════════════════════════")
        print(f"  Total Episodes  : {MAX_EPISODES}")
        print(f"  Total Steps     : {self.total_steps:,}")
        print(f"  Training Steps  : {self.training_steps:,}")
        print(f"  Best Reward     : {self.best_reward:.3f}")
        print(f"  Final Epsilon   : {self.epsilon:.3f}")
        print(f"  Models saved to : {MODELS_DIR}")
        print(f"  Logs saved to   : {LOGS_DIR}")
        print("════════════════════════════════════════\n")

        # ── restore terminal ──────────────────
        sys.stdout = logger.terminal
        logger.close()

        print("\nTraining complete.")
        print(f"Logs → {log_path}")
        print(f"CSV  → {csv_path}")
        print(f"Models → {MODELS_DIR}")

    # ─────────────────────────────────────────
    # PRINT PROGRESS
    # ─────────────────────────────────────────
    def _print_progress(self, episode):
        last_100 = self.episode_rewards[-100:]
        avg      = np.mean(last_100)
        best     = np.max(last_100)
        worst    = np.min(last_100)

        print(f"\n── Progress at Episode {episode} ────────")
        print(f"  Last 100 Avg Reward : {avg:+.3f}")
        print(f"  Last 100 Best       : {best:+.3f}")
        print(f"  Last 100 Worst      : {worst:+.3f}")
        print(f"  Total Steps         : {self.total_steps:,}")
        print(f"  Training Steps      : {self.training_steps:,}")
        print(f"  Buffer Size         : {len(self.replay_buffer):,}")
        print(f"  Epsilon             : {self.epsilon:.4f}")
        print("────────────────────────────────────────\n")