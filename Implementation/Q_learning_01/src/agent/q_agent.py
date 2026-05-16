import numpy as np
import random

class QAgent:
    def __init__(self, states, actions=6, alpha=0.1, gamma=0.9):
        self.q_table = np.zeros((states, actions)) # initially, all = 0
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = 1.0 # purely random in beginning

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:     # if random number generate < epsilon
            return random.randint(0, 5)  # choose random action
        return np.argmax(self.q_table[state])  # else choose action with max q value

    def update(self, s, a, r, s_next):
        old_val = self.q_table[s, a] 
        next_max = np.max(self.q_table[s_next]) 

        # Bellman equation
        # Q(s,a) = Q(s,a) + α [ r + γ maxQ(s',a') − Q(s,a) ]
        self.q_table[s, a] = old_val + self.alpha * (r + self.gamma * next_max - old_val)

    def decay_epsilon(self):
        # 0.5% reduced after every ep : 1.0 − 0.005 = 0.995
        # but epsilon should not be < 0.01 bcz if reached to 0, agent won't explore
        self.epsilon = max(0.01, self.epsilon * 0.995) 