import pandas as pd
from src.states.states import StateEncoder
from src.rewards.reward_function import get_reward

class SwitchEnv:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.encoder = StateEncoder(data_path)
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        row = self.data.iloc[self.current_idx]
        return self.encoder.get_state_index(row)

    def step(self, action):
        row = self.data.iloc[self.current_idx]
        
        try:
            fill = pd.to_numeric(row['mac_fill']) 
        except:
            # Fallback for string-based labels
            fill_map = {"empty": 0, "almost_full": 85, "full": 100}
            fill = fill_map.get(row['mac_fill'], 0)

        reward = get_reward(action, fill, row['flood_pressure'], 
                            row['entry_age'], row['new_mac_rate'])

        self.current_idx = (self.current_idx + 1) % len(self.data)
        next_row = self.data.iloc[self.current_idx]
        next_s = self.encoder.get_state_index(next_row)
        
        return next_s, reward, False
    