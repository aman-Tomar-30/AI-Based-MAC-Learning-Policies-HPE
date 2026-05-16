import pandas as pd

class StateEncoder: # to convert raw data to state index
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        self.unique_scenarios = df.drop_duplicates().reset_index(drop=True) # removes duplicate rows
        
    def get_state_index(self, row): 
        match = self.unique_scenarios[
            (self.unique_scenarios['mac_fill'] == row['mac_fill']) &
            (self.unique_scenarios['flood_pressure'] == row['flood_pressure']) &
            (self.unique_scenarios['port_traffic'] == row['port_traffic']) &
            (self.unique_scenarios['vlan_info'] == row['vlan_info']) &
            (self.unique_scenarios['entry_age'] == row['entry_age']) &
            (self.unique_scenarios['new_mac_rate'] == row['new_mac_rate'])
        ]
        return int(match.index[0]) if not match.empty else 0

    def total_states(self):
        return len(self.unique_scenarios)   
    
