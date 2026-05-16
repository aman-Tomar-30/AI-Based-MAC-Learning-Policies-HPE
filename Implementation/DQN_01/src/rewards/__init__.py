from src.config import (
    REWARD_CORRECT_DELIVERY,
    REWARD_FLOODING_AVOIDED,
    REWARD_PREVENTIVE_EVICTION,
    REWARD_NECESSARY_FLOOD,
    REWARD_WRONG_PORT_STALE,
    REWARD_EARLY_ACTIVE_EVICTION,
    REWARD_TABLE_OVERFLOW,
    REWARD_UNNECESSARY_FLOOD,
    TABLE_PREVENTIVE_THRESHOLD,
    TABLE_OVERFLOW_THRESHOLD,
    ENTRY_ACTIVE_AGE_THRESHOLD
)

# ─────────────────────────────────────────
# OUTCOME TO REWARD MAPPING
# ─────────────────────────────────────────
OUTCOME_REWARD_MAP = {
    'correct_delivery'      : REWARD_CORRECT_DELIVERY,
    'flooding_avoided'      : REWARD_FLOODING_AVOIDED,
    'preventive_eviction'   : REWARD_PREVENTIVE_EVICTION,
    'necessary_flood'       : REWARD_NECESSARY_FLOOD,
    'wrong_port_stale'      : REWARD_WRONG_PORT_STALE,
    'early_active_eviction' : REWARD_EARLY_ACTIVE_EVICTION,
    'table_overflow'        : REWARD_TABLE_OVERFLOW,
    'unnecessary_flood'     : REWARD_UNNECESSARY_FLOOD
}

# ─────────────────────────────────────────
# MAIN REWARD FUNCTION
# ─────────────────────────────────────────
def calculate_reward(outcome, next_state_vector, action):
    """
    Calculates reward based on:
        1. outcome  — what happened after action (primary signal)
        2. next_state_vector — normalized [mac, flood, age]
                               used for shaping bonus/penalty
        3. action   — action taken (for context)

    Returns:
        reward : float
        reason : string (for logging)
    """

    # ─────────────────────────────────────
    # BASE REWARD FROM OUTCOME
    # ─────────────────────────────────────
    if outcome not in OUTCOME_REWARD_MAP:
        raise ValueError(f"Unknown outcome: '{outcome}'. "
                         f"Valid outcomes: {list(OUTCOME_REWARD_MAP.keys())}")

    base_reward = OUTCOME_REWARD_MAP[outcome]
    reason      = outcome

    # ─────────────────────────────────────
    # REWARD SHAPING — STATE BASED BONUS
    # adds small adjustments on top of base reward
    # helps agent learn faster by giving denser signal
    # ─────────────────────────────────────
    mac_norm   = next_state_vector[0]
    flood_norm = next_state_vector[1]
    age_norm   = next_state_vector[2]

    shaping_bonus = 0.0

    # bonus if mac table improved (lower is better)
    if mac_norm < TABLE_PREVENTIVE_THRESHOLD:
        shaping_bonus += 0.05   # small bonus for keeping table healthy

    # bonus if flood pressure is low
    if flood_norm < 0.30:
        shaping_bonus += 0.05   # small bonus for low flooding

    # penalty if table dangerously full regardless of action
    if mac_norm >= TABLE_OVERFLOW_THRESHOLD:
        shaping_bonus -= 0.10   # extra nudge to act when critical

    # penalty if agent flooded unnecessarily and table was fine
    if outcome == 'unnecessary_flood' and mac_norm < 0.50:
        shaping_bonus -= 0.10   # stronger penalty for flooding on healthy table

    # ─────────────────────────────────────
    # FINAL REWARD
    # ─────────────────────────────────────
    total_reward = round(base_reward + shaping_bonus, 4)

    return total_reward, reason


# ─────────────────────────────────────────
# SITUATION CLASSIFIER
# used for logging and monitoring only
# not used in reward calculation
# ─────────────────────────────────────────
def classify_situation(state_vector):
    """
    Classifies current switch situation into one of 4 categories.
    Used for logging and episode summary only.
    Does NOT affect reward calculation.

    Parameters:
        state_vector: normalized [mac_norm, flood_norm, age_norm]

    Returns:
        situation: string
    """
    mac_norm   = state_vector[0]
    flood_norm = state_vector[1]

    if mac_norm >= TABLE_OVERFLOW_THRESHOLD and flood_norm >= 0.80:
        return 'CRITICAL'
    elif mac_norm >= TABLE_PREVENTIVE_THRESHOLD and flood_norm >= 0.60:
        return 'PRESSURE'
    elif mac_norm >= 0.70 and flood_norm >= 0.40:
        return 'MODERATE_STRESS'
    else:
        return 'HEALTHY'


# ─────────────────────────────────────────
# EPISODE SUMMARY
# called at end of each episode
# ─────────────────────────────────────────
def episode_summary(total_reward, step_count, outcome_log):
    """
    Prints summary of episode performance.

    Parameters:
        total_reward : float → sum of all rewards in episode
        step_count   : int   → how many steps episode ran
        outcome_log  : list  → list of outcome strings from each step
    """
    print("\n══ Episode Summary ════════════════════")
    print(f"  Total Reward  : {total_reward:.3f}")
    print(f"  Steps         : {step_count}")
    print(f"  Avg Reward    : {total_reward/max(step_count,1):.3f}")

    print("\n  Outcome Breakdown:")
    for outcome in OUTCOME_REWARD_MAP:
        count = outcome_log.count(outcome)
        if count > 0:
            reward_val = OUTCOME_REWARD_MAP[outcome]
            sign = '+' if reward_val > 0 else ''
            print(f"    {outcome:<28} × {count:>3}  "
                  f"({sign}{reward_val})")

    print("═══════════════════════════════════════\n")


# ─────────────────────────────────────────
# DONE CONDITION
# checks if episode should end
# ─────────────────────────────────────────
def is_done(state_vector, step, max_steps):
    """
    Episode ends if:
        1. Table completely overflows (mac_norm >= 1.0)
        2. Max steps reached

    Parameters:
        state_vector : normalized state
        step         : current step count
        max_steps    : max allowed steps per episode

    Returns:
        done   : bool
        reason : string
    """
    mac_norm = state_vector[0]

    if mac_norm >= 1.0:
        return True, 'table_full'

    if step >= max_steps:
        return True, 'max_steps_reached'

    return False, None