
import numpy as np
from src.config import (
    MAC_TABLE_CAPACITY,
    MAX_FLOOD_RATE,
    MAX_AGING_TIMER
)


STATE_NAMES = [
    'mac_table_entries',
    'flood_pressure',
    'entry_age'
]


def normalize_state(raw_mac_entries, raw_flood_rate, raw_avg_age):
    """
    Takes raw values directly from switch environment.
    Returns normalized numpy array of shape (3,) with values in [0, 1].

    Parameters:
        raw_mac_entries : int   → current number of MAC entries in table
        raw_flood_rate  : float → current packets/sec being flooded
        raw_avg_age     : float → average age of entries in seconds

    Returns:
        numpy array [mac_norm, flood_norm, age_norm]
    """

    # clamp values to valid range before normalizing
    # prevents values going above 1.0 due to simulation spikes
    raw_mac_entries = np.clip(raw_mac_entries, 0, MAC_TABLE_CAPACITY)
    raw_flood_rate  = np.clip(raw_flood_rate,  0, MAX_FLOOD_RATE)
    raw_avg_age     = np.clip(raw_avg_age,     0, MAX_AGING_TIMER)

    # normalize each state to 0-1
    mac_norm   = raw_mac_entries / MAC_TABLE_CAPACITY
    flood_norm = raw_flood_rate  / MAX_FLOOD_RATE
    age_norm   = raw_avg_age     / MAX_AGING_TIMER

    return np.array([mac_norm, flood_norm, age_norm], dtype=np.float32)


# ─────────────────────────────────────────
# EXTRACT RAW STATE FROM SWITCH STATE DICT
# ─────────────────────────────────────────
def extract_state(switch_state):
    """
    Pulls raw values from switch_state dictionary.
    Calls normalize_state and returns normalized vector.

    switch_state dict expected keys:
        'mac_entries' : int
        'flood_rate'  : float
        'avg_entry_age' : float
    """

    raw_mac   = switch_state['mac_entries']
    raw_flood = switch_state['flood_rate']
    raw_age   = switch_state['avg_entry_age']

    return normalize_state(raw_mac, raw_flood, raw_age)


# ─────────────────────────────────────────
# PRINT STATE (for debugging)
# ─────────────────────────────────────────
def print_state(state_vector):
    """
    Prints normalized state vector in readable format.
    state_vector: numpy array of shape (3,)
    """
    print("\n── Current State ──────────────────")
    for name, value in zip(STATE_NAMES, state_vector):
        bar = '█' * int(value * 20)   # visual bar 0-20 chars
        print(f"  {name:<22} {value:.3f}  |{bar}")
    print("───────────────────────────────────\n")


# ─────────────────────────────────────────
# VALIDATE STATE VECTOR
# ─────────────────────────────────────────
def validate_state(state_vector):
    """
    Checks that all state values are in valid range [0, 1].
    Raises ValueError if any value is out of range.
    Used during debugging to catch normalization errors early.
    """
    for i, (name, value) in enumerate(zip(STATE_NAMES, state_vector)):
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"State '{name}' at index {i} has value {value:.4f} "
                f"which is outside valid range [0, 1]. "
                f"Check normalization constants in config.py."
            )
    return True


# ─────────────────────────────────────────
# BATCH NORMALIZE (for replay buffer samples)
# ─────────────────────────────────────────
def normalize_batch(raw_states):
    """
    Normalizes a batch of raw states from replay buffer.
    Used during training when mini batch of 64 is sampled.

    Parameters:
        raw_states: numpy array of shape (batch_size, 3)
                    each row is [raw_mac, raw_flood, raw_age]

    Returns:
        normalized numpy array of shape (batch_size, 3)
    """
    divisors = np.array(
        [MAC_TABLE_CAPACITY, MAX_FLOOD_RATE, MAX_AGING_TIMER],
        dtype=np.float32
    )

    # clip entire batch
    raw_states = np.clip(raw_states, 0, divisors)

    # divide entire batch in one operation
    return (raw_states / divisors).astype(np.float32)