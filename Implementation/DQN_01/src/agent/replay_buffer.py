import numpy as np
import random
from src.config import BUFFER_CAPACITY, BATCH_SIZE, MIN_BUFFER_SIZE


class ReplayBuffer:
    """
    Circular queue that stores past transitions.
    Stores raw state values (not normalized).
    Normalization happens at training time.

    Each transition:
    (state, action, reward, next_state, done)
    """

    def __init__(self, capacity=BUFFER_CAPACITY):
        self.capacity    = capacity
        self.buffer      = [None] * capacity  # fixed size array
        self.pointer     = 0                  # where next transition goes
        self.size        = 0                  # how many transitions stored

    # ─────────────────────────────────────────
    # STORE — push one transition into buffer
    # ─────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        """
        Stores one transition into buffer.
        Overwrites oldest transition when buffer is full.

        Parameters:
            state      : numpy array (3,) normalized
            action     : int   (0-6)
            reward     : float
            next_state : numpy array (3,) normalized
            done       : bool
        """
        # pack transition as tuple
        transition = (
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        )

        # write at current pointer position
        # overwrites oldest if buffer full
        self.buffer[self.pointer] = transition

        # move pointer forward using modulo
        # wraps around to 0 when reaches capacity
        self.pointer = (self.pointer + 1) % self.capacity

        # track actual size until buffer is full
        self.size = min(self.size + 1, self.capacity)

    # ─────────────────────────────────────────
    # SAMPLE — random mini batch
    # ─────────────────────────────────────────
    def sample(self, batch_size=BATCH_SIZE):
        """
        Randomly samples batch_size transitions from buffer.
        Returns separate numpy arrays for each component.

        Returns:
            states      : numpy array (batch_size, 3)
            actions     : numpy array (batch_size,)
            rewards     : numpy array (batch_size,)
            next_states : numpy array (batch_size, 3)
            dones       : numpy array (batch_size,)
        """
        if not self.is_ready():
            raise Exception(
                f"Buffer not ready. "
                f"Has {self.size} transitions, "
                f"needs {MIN_BUFFER_SIZE} minimum."
            )

        # randomly sample batch_size indices
        indices = random.sample(range(self.size), batch_size)

        # unpack sampled transitions into separate arrays
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )

        return (
            np.array(states,      dtype=np.float32),   # (64, 3)
            np.array(actions,     dtype=np.int64),      # (64,)
            np.array(rewards,     dtype=np.float32),    # (64,)
            np.array(next_states, dtype=np.float32),    # (64, 3)
            np.array(dones,       dtype=np.float32)     # (64,) float for math
        )

    # ─────────────────────────────────────────
    # IS READY — enough transitions to train
    # ─────────────────────────────────────────
    def is_ready(self):
        """
        Returns True if buffer has enough transitions
        to start training.
        """
        return self.size >= MIN_BUFFER_SIZE

    # ─────────────────────────────────────────
    # IS FULL — buffer at capacity
    # ─────────────────────────────────────────
    def is_full(self):
        """
        Returns True if buffer has reached max capacity.
        Old transitions are now being overwritten.
        """
        return self.size == self.capacity

    # ─────────────────────────────────────────
    # STATS — for logging and debugging
    # ─────────────────────────────────────────
    def stats(self):
        """
        Prints current buffer status.
        """
        fill_pct = (self.size / self.capacity) * 100
        bar      = '█' * int(fill_pct / 5)
        print("\n── Replay Buffer ───────────────────")
        print(f"  Capacity  : {self.capacity:,}")
        print(f"  Stored    : {self.size:,}")
        print(f"  Pointer   : {self.pointer:,}")
        print(f"  Fill      : {fill_pct:.1f}%  |{bar}")
        print(f"  Ready     : {self.is_ready()}")
        print(f"  Full      : {self.is_full()}")
        print("────────────────────────────────────\n")

    # ─────────────────────────────────────────
    # CLEAR — empty the buffer
    # ─────────────────────────────────────────
    def clear(self):
        """
        Resets buffer to empty state.
        Use when starting fresh training run.
        """
        self.buffer  = [None] * self.capacity
        self.pointer = 0
        self.size    = 0
        print("Replay buffer cleared.")

    # ─────────────────────────────────────────
    # __len__ — pythonic size check
    # ─────────────────────────────────────────
    def __len__(self):
        return self.size