LEARN_MAC        = 0
EVICT_ENTRY      = 1
FLOOD            = 2
BLOCK_PORT       = 3
UNBLOCK_PORT     = 4
INCREASE_AGING   = 5
DECREASE_AGING   = 6

NUM_ACTIONS = 7

ACTION_NAMES = {
    LEARN_MAC      : "LEARN_MAC",
    EVICT_ENTRY    : "EVICT_ENTRY",
    FLOOD          : "FLOOD",
    BLOCK_PORT     : "BLOCK_PORT",
    UNBLOCK_PORT   : "UNBLOCK_PORT",
    INCREASE_AGING : "INCREASE_AGING",
    DECREASE_AGING : "DECREASE_AGING"
}

# ─────────────────────────────────────────
# ACTION EXECUTOR
# ─────────────────────────────────────────
from src.config import (
    TABLE_PREVENTIVE_THRESHOLD,
    TABLE_OVERFLOW_THRESHOLD,
    ENTRY_ACTIVE_AGE_THRESHOLD,
    AGING_INCREASE_STEP,
    AGING_DECREASE_STEP,
    MIN_AGING_TIMER_NORM,
    MAX_AGING_TIMER_NORM
)

def execute_action(action, switch_state):
    """
    Takes action index and current switch state.
    Modifies switch state based on action.
    Returns updated switch state and action outcome.

    switch_state is a dictionary:
    {
        'mac_entries'    : float,   # raw count
        'flood_rate'     : float,   # raw packets/sec
        'avg_entry_age'  : float,   # raw seconds
        'aging_timer'    : float,   # normalized 0-1
        'blocked_ports'  : list,    # list of blocked port ids
        'mac_table'      : dict,    # mac_address: {port, age, hits}
        'total_capacity' : int      # max MAC entries
    }

    outcome tells reward function what actually happened:
    'correct_delivery', 'flooding_avoided', 'preventive_eviction',
    'necessary_flood', 'unnecessary_flood', 'wrong_port_stale',
    'early_active_eviction', 'table_overflow'
    """

    outcome = None
    mac_fill = switch_state['mac_entries'] / switch_state['total_capacity']

    # ─────────────────────────────────────────
    # ACTION 0 — LEARN_MAC
    # switch learns incoming MAC naturally
    # outcome depends on whether table is full or not
    # ─────────────────────────────────────────
    if action == LEARN_MAC:
        if mac_fill >= TABLE_OVERFLOW_THRESHOLD:
            # table too full to learn → forced flood
            switch_state['flood_rate'] += 50
            outcome = 'necessary_flood'
        else:
            # table has space → learn MAC → correct delivery possible
            outcome = 'correct_delivery'

    # ─────────────────────────────────────────
    # ACTION 1 — EVICT_ENTRY
    # remove one entry from MAC table
    # outcome depends on whether evicted entry was active or stale
    # ─────────────────────────────────────────
    elif action == EVICT_ENTRY:
        if len(switch_state['mac_table']) == 0:
            # nothing to evict → unnecessary action → flood
            switch_state['flood_rate'] += 30
            outcome = 'unnecessary_flood'
        else:
            # find oldest entry
            oldest_mac = max(
                switch_state['mac_table'],
                key=lambda m: switch_state['mac_table'][m]['age']
            )
            entry = switch_state['mac_table'][oldest_mac]
            entry_age_norm = entry['age'] / MAX_AGING_TIMER_NORM

            if entry_age_norm < ENTRY_ACTIVE_AGE_THRESHOLD:
                # entry is fresh → active → evicting it is wrong
                del switch_state['mac_table'][oldest_mac]
                switch_state['mac_entries'] = max(0, switch_state['mac_entries'] - 1)
                switch_state['flood_rate'] += 20   # evicted active entry causes re-flood
                outcome = 'early_active_eviction'
            elif mac_fill >= TABLE_PREVENTIVE_THRESHOLD:
                # entry is stale + table filling → preventive eviction
                del switch_state['mac_table'][oldest_mac]
                switch_state['mac_entries'] = max(0, switch_state['mac_entries'] - 1)
                switch_state['flood_rate'] = max(0, switch_state['flood_rate'] - 30)
                outcome = 'preventive_eviction'
            else:
                # entry stale but table not yet critical → still okay
                del switch_state['mac_table'][oldest_mac]
                switch_state['mac_entries'] = max(0, switch_state['mac_entries'] - 1)
                outcome = 'preventive_eviction'

    # ─────────────────────────────────────────
    # ACTION 2 — FLOOD
    # agent explicitly floods packet to all ports
    # ─────────────────────────────────────────
    elif action == FLOOD:
        if mac_fill >= TABLE_OVERFLOW_THRESHOLD:
            # table full → flooding was necessary
            switch_state['flood_rate'] += 100
            outcome = 'necessary_flood'
        else:
            # table not full → flooding was unnecessary
            switch_state['flood_rate'] += 150
            outcome = 'unnecessary_flood'

    # ─────────────────────────────────────────
    # ACTION 3 — BLOCK_PORT
    # stop a port from generating new MAC entries
    # ─────────────────────────────────────────
    elif action == BLOCK_PORT:
        if mac_fill >= TABLE_PREVENTIVE_THRESHOLD:
            # table filling → blocking port is preventive
            if len(switch_state['blocked_ports']) == 0:
                switch_state['blocked_ports'].append('port_1')
            switch_state['flood_rate'] = max(0, switch_state['flood_rate'] - 50)
            switch_state['mac_entries'] = max(0, switch_state['mac_entries'] - 20)
            outcome = 'flooding_avoided'
        else:
            # table fine → blocking port unnecessarily
            switch_state['flood_rate'] += 40
            outcome = 'unnecessary_flood'

    # ─────────────────────────────────────────
    # ACTION 4 — UNBLOCK_PORT
    # re-enable a previously blocked port
    # ─────────────────────────────────────────
    elif action == UNBLOCK_PORT:
        if len(switch_state['blocked_ports']) > 0:
            # there is a blocked port → unblock it
            switch_state['blocked_ports'].pop()
            outcome = 'correct_delivery'
        else:
            # no blocked ports → unnecessary action → minor flood
            switch_state['flood_rate'] += 20
            outcome = 'unnecessary_flood'

    # ─────────────────────────────────────────
    # ACTION 5 — INCREASE_AGING
    # reduce TTL → entries expire faster
    # useful when table full of stale entries
    # ─────────────────────────────────────────
    elif action == INCREASE_AGING:
        if switch_state['aging_timer'] > MIN_AGING_TIMER_NORM:
            # reduce aging timer → entries expire sooner
            switch_state['aging_timer'] = max(
                MIN_AGING_TIMER_NORM,
                switch_state['aging_timer'] - AGING_INCREASE_STEP
            )
            # simulate entries expiring → mac count drops
            switch_state['mac_entries'] = max(
                0,
                switch_state['mac_entries'] - 50
            )
            switch_state['flood_rate'] = max(
                0,
                switch_state['flood_rate'] - 20
            )
            outcome = 'preventive_eviction'
        else:
            # aging timer already at minimum → no effect
            switch_state['flood_rate'] += 10
            outcome = 'unnecessary_flood'

    # ─────────────────────────────────────────
    # ACTION 6 — DECREASE_AGING
    # increase TTL → entries live longer
    # useful when same MACs keep re-flooding
    # ─────────────────────────────────────────
    elif action == DECREASE_AGING:
        if switch_state['aging_timer'] < MAX_AGING_TIMER_NORM:
            # increase aging timer → entries persist longer
            switch_state['aging_timer'] = min(
                MAX_AGING_TIMER_NORM,
                switch_state['aging_timer'] + AGING_DECREASE_STEP
            )
            # entries persist → less re-flooding of same known devices
            switch_state['flood_rate'] = max(
                0,
                switch_state['flood_rate'] - 15
            )
            outcome = 'flooding_avoided'
        else:
            # aging timer already at maximum → no effect
            switch_state['flood_rate'] += 10
            outcome = 'unnecessary_flood'

    # ─────────────────────────────────────────
    # TABLE OVERFLOW CHECK (after every action)
    # ─────────────────────────────────────────
    if switch_state['mac_entries'] >= switch_state['total_capacity']:
        switch_state['flood_rate'] += 200
        outcome = 'table_overflow'

    return switch_state, outcome


# ─────────────────────────────────────────
# HELPER — get action name for logging
# ─────────────────────────────────────────
def get_action_name(action):
    return ACTION_NAMES.get(action, "UNKNOWN")