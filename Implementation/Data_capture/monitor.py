import subprocess
import time
import re
from collections import defaultdict
from generate_csv import get_output_csv

"""
This script collects:

1. MAC Table Entries
2. Flood Pressure
3. Entry Age

"""

# =========================
# CONFIGURATION
# =========================

SWITCHES = subprocess.check_output(["ovs-vsctl", "list-br"], text=True).split()   #gives list of switches
SWITCH = SWITCHES[0]
print(SWITCHES[0])

INTERVAL = 3              # seconds

# Estimated limits (customize according to topology)
MAX_MAC_CAPACITY = 20
MAX_FLOOD_RATE = 10000       
MAX_ENTRY_AGE = 300          # seconds


# =========================
# HELPER FUNCTIONS - which run commands on terminal
# =========================

def run_cmd(cmd):
    """Run shell command"""
    result = subprocess.check_output(cmd, shell=True, text=True)
    return result.strip()


# =========================================================
# 1. MAC TABLE ENTRIES
# =========================================================

def get_mac_table_entries(sw):

    try:
        output = run_cmd(f"ovs-appctl fdb/show {sw}")

        mac_entries = []

        for line in output.splitlines():

            # Example:
            # port VLAN MAC                 Age
            # 1    1     00:00:00:00:00:01 12

            match = re.search(
                r"(\d+)\s+(\d+)\s+([0-9a-f:]{17})\s+(\d+)",
                line,
                re.IGNORECASE
            )

            if match:

                port = match.group(1)
                vlan = match.group(2)
                mac = match.group(3)
                age = int(match.group(4))

                mac_entries.append({
                    "vlan": vlan,
                    "mac": mac,
                    "port": port,
                    "age": age
                })

        return mac_entries

    except Exception as e:
        print(f"[ERROR] MAC table fetch failed: {e}")
        return []


# =========================================================
# 2. FLOOD PRESSURE
# =========================================================

def get_flood_pressure(sw):
    """
    Estimate flooding using OpenFlow flow stats

    Flood packets are usually NORMAL/FLOOD actions.
    """

    try:
        output = run_cmd(f"ovs-ofctl dump-flows {sw}")

        flood_packets = 0

        for line in output.splitlines():

            if "FLOOD" in line or "NORMAL" in line:

                match = re.search(r"n_packets=(\d+)", line)

                if match:
                    flood_packets += int(match.group(1))

        return flood_packets

    except Exception as e:
        print(f"[ERROR] Flood stats failed: {e}")
        return 0


# =========================================================
# 3. ENTRY AGE
# =========================================================

def calculate_entry_age(mac_entries):

    if not mac_entries:
        return 0

    ages = [entry["age"] for entry in mac_entries]

    return sum(ages) / len(ages)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(value, max_value):

    if max_value == 0:
        return 0

    return round(value / max_value, 4)


# =========================================================
# MAIN MONITOR LOOP
# =========================================================

def monitor(sw):

    print("\n========== SDN MONITOR STARTED ==========\n")
    data = []

    for n in range(100): #no.of lines of data you want


        #print(f"\n========== SWITCH {sw} ==========")
        
        # -----------------------------------
        # MAC TABLE
        # -----------------------------------
        mac_entries = get_mac_table_entries(sw)
        current_entries = len(mac_entries)
        mac_fill = normalize(
            current_entries,
            MAX_MAC_CAPACITY
        )

        # -----------------------------------
        # FLOOD PRESSURE
        # -----------------------------------
        flood_rate = get_flood_pressure(sw)
        flood_pressure = normalize(
            flood_rate,
            MAX_FLOOD_RATE
        )

        # -----------------------------------
        # ENTRY AGE
        # -----------------------------------
        avg_age = calculate_entry_age(mac_entries)
        age_score = normalize(
            avg_age,
            MAX_ENTRY_AGE
        )

        # ===================================
        # PRINT RESULTS
        # ===================================
        """
        print(f"\n[1] MAC Table Entries")
        print(f"Current Entries : {current_entries}")
        print(f"Fill Percentage : {mac_fill}")

        print(f"\n[2] Flood Pressure")
        print(f"Flood Packets   : {flood_rate}")
        print(f"Flood Score     : {flood_pressure}")

        print(f"\n[3] Entry Age")
        print(f"Average Age     : {avg_age:.2f} sec")
        print(f"Age Score       : {age_score}")
        
        """

        #STATES - MAC_FILL FLOOD_PRESSURE AVG_SCORE 
        state = [mac_fill, flood_pressure, age_score]
        print(f"States: {state}")
        data.append(state)

        time.sleep(INTERVAL)
    
    print(data)
    get_output_csv(data)

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    monitor(SWITCH)