import numpy as np
import random
from src.config import (
    MAC_TABLE_CAPACITY,
    MAX_FLOOD_RATE,
    MAX_AGING_TIMER,
    MAX_STEPS,
    TABLE_OVERFLOW_THRESHOLD,
    TABLE_PREVENTIVE_THRESHOLD
)
from src.states  import extract_state, validate_state, print_state
from src.rewards import calculate_reward, classify_situation, is_done
from src.actions import execute_action, get_action_name, NUM_ACTIONS


class SwitchEnvironment:
    """
    Simulates a single L2 switch in a Dragonfly topology.
    Manages MAC table, flood pressure, entry aging.
    Agent interacts via step() and reset().
    """

    def __init__(self):
        self.total_capacity = MAC_TABLE_CAPACITY
        self.max_steps      = MAX_STEPS
        self.step_count     = 0
        self.outcome_log    = []
        self.reward_log     = []
        self.switch_state   = None
        self.reset()

    # ─────────────────────────────────────────
    # RESET — start new episode
    # ─────────────────────────────────────────
    def reset(self):
        """
        Resets switch to initial state.
        Called at start of every episode.
        Returns initial normalized state vector.
        """
        self.step_count  = 0
        self.outcome_log = []
        self.reward_log  = []

        # initialize switch state dictionary
        self.switch_state = {
            'mac_entries'    : random.randint(1000, 4000),  # start with some entries
            'flood_rate'     : random.uniform(100, 600),    # some baseline flooding
            'avg_entry_age'  : random.uniform(100, 800),    # some aged entries
            'aging_timer'    : 0.5,                         # normalized, start at middle
            'blocked_ports'  : [],                          # no blocked ports initially
            'mac_table'      : self._generate_mac_table(),  # populate initial MAC table
            'total_capacity' : self.total_capacity
        }

        # extract and return normalized state vector
        state = extract_state(self.switch_state)
        validate_state(state)
        return state

    # ─────────────────────────────────────────
    # STEP — agent takes one action
    # ─────────────────────────────────────────
    def step(self, action):
        """
        Executes action on switch environment.
        Returns (next_state, reward, done, info).

        Parameters:
            action : int → action index 0-6

        Returns:
            next_state : numpy array shape (3,)
            reward     : float
            done       : bool
            info       : dict (for logging/debugging)
        """
        self.step_count += 1

        # ── execute action ──────────────────
        self.switch_state, outcome = execute_action(
            action,
            self.switch_state
        )

        # ── simulate environment dynamics ───
        # environment changes naturally each step
        # independent of agent action
        self._simulate_network_dynamics()

        # ── extract next state ──────────────
        next_state = extract_state(self.switch_state)
        validate_state(next_state)

        # ── calculate reward ────────────────
        reward, reason = calculate_reward(
            outcome,
            next_state,
            action
        )

        # ── check done ──────────────────────
        done, done_reason = is_done(
            next_state,
            self.step_count,
            self.max_steps
        )

        # ── log everything ──────────────────
        self.outcome_log.append(outcome)
        self.reward_log.append(reward)

        # ── build info dict ─────────────────
        info = {
            'step'          : self.step_count,
            'action_name'   : get_action_name(action),
            'outcome'       : outcome,
            'reason'        : reason,
            'done_reason'   : done_reason,
            'situation'     : classify_situation(next_state),
            'mac_fill'      : self.switch_state['mac_entries'] / self.total_capacity,
            'flood_rate'    : self.switch_state['flood_rate'],
            'aging_timer'   : self.switch_state['aging_timer'],
            'blocked_ports' : len(self.switch_state['blocked_ports'])
        }

        return next_state, reward, done, info

    # ─────────────────────────────────────────
    # SIMULATE NETWORK DYNAMICS
    # natural changes in switch state each step
    # independent of agent action
    # ─────────────────────────────────────────
    def _simulate_network_dynamics(self):
        """
        Simulates natural network traffic behavior each step.
        New devices connecting, traffic fluctuations,
        natural MAC aging, background flood events.
        """

        # ── new MACs arriving naturally ─────
        # random devices sending frames → new entries
        new_macs = random.randint(0, 80)
        self.switch_state['mac_entries'] = min(
            self.total_capacity,
            self.switch_state['mac_entries'] + new_macs
        )

        # ── natural aging ───────────────────
        # entries get older each step
        aging_rate = self.switch_state['aging_timer'] * 30
        self.switch_state['avg_entry_age'] = min(
            MAX_AGING_TIMER,
            self.switch_state['avg_entry_age'] + aging_rate
        )

        # ── entries naturally expiring ──────
        # some entries age out based on aging timer
        expiry_count = int(
            self.switch_state['aging_timer'] * 20
        )
        self.switch_state['mac_entries'] = max(
            0,
            self.switch_state['mac_entries'] - expiry_count
        )

        # ── background flood fluctuation ────
        # natural ebb and flow of flooding
        flood_change = random.uniform(-50, 100)
        self.switch_state['flood_rate'] = max(
            0,
            min(MAX_FLOOD_RATE, self.switch_state['flood_rate'] + flood_change)
        )

        # ── update mac table entries ages ───
        for mac in self.switch_state['mac_table']:
            self.switch_state['mac_table'][mac]['age'] += 1

        # ── occasionally add stale entry ────
        # simulates real network where old devices reconnect
        if random.random() < 0.15:
            self._add_random_mac_entry()

        # ── table pressure increases flood ──
        # if table nearly full, flooding increases
        mac_fill = self.switch_state['mac_entries'] / self.total_capacity
        if mac_fill >= TABLE_OVERFLOW_THRESHOLD:
            self.switch_state['flood_rate'] = min(
                MAX_FLOOD_RATE,
                self.switch_state['flood_rate'] + 100
            )

    # ─────────────────────────────────────────
    # GENERATE INITIAL MAC TABLE
    # ─────────────────────────────────────────
    def _generate_mac_table(self):
        """
        Creates initial MAC table with random entries.
        Simulates a switch that has been running for some time.
        Returns dict: {mac_address: {port, age, hits}}
        """
        mac_table = {}
        num_initial = random.randint(100, 400)

        for i in range(num_initial):
            mac = f"AA:BB:CC:DD:{i//256:02X}:{i%256:02X}"
            mac_table[mac] = {
                'port' : random.randint(1, 24),
                'age'  : random.randint(0, 3600),
                'hits' : random.randint(1, 100)
            }

        return mac_table

    # ─────────────────────────────────────────
    # ADD RANDOM MAC ENTRY
    # ─────────────────────────────────────────
    def _add_random_mac_entry(self):
        """
        Adds a new random MAC entry to simulate
        a new device connecting to switch.
        """
        if self.switch_state['mac_entries'] < self.total_capacity:
            new_id = random.randint(0, 65535)
            mac = f"CC:DD:EE:FF:{new_id//256:02X}:{new_id%256:02X}"
            self.switch_state['mac_table'][mac] = {
                'port' : random.randint(1, 24),
                'age'  : 0,
                'hits' : 1
            }
            self.switch_state['mac_entries'] += 1

    # ─────────────────────────────────────────
    # RENDER — print current state
    # ─────────────────────────────────────────
    def render(self):
        """
        Prints current switch state.
        Call during debugging to see what environment looks like.
        """
        state = extract_state(self.switch_state)
        print_state(state)
        print(f"  Step          : {self.step_count}")
        print(f"  MAC Entries   : {self.switch_state['mac_entries']}/{self.total_capacity}")
        print(f"  Flood Rate    : {self.switch_state['flood_rate']:.1f} pkt/s")
        print(f"  Avg Age       : {self.switch_state['avg_entry_age']:.1f} sec")
        print(f"  Aging Timer   : {self.switch_state['aging_timer']:.2f}")
        print(f"  Blocked Ports : {self.switch_state['blocked_ports']}")
        print(f"  Situation     : {classify_situation(state)}")

    # ─────────────────────────────────────────
    # EPISODE STATS
    # ─────────────────────────────────────────
    def get_episode_stats(self):
        """
        Returns episode statistics after episode ends.
        Used by trainer for logging.
        """
        return {
            'total_reward'  : sum(self.reward_log),
            'avg_reward'    : sum(self.reward_log) / max(len(self.reward_log), 1),
            'total_steps'   : self.step_count,
            'outcome_log'   : self.outcome_log,
            'reward_log'    : self.reward_log
        }