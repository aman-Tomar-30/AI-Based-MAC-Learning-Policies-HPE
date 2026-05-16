import torch
import torch.nn as nn
import numpy as np
from src.config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    LEARNING_RATE
)


# ─────────────────────────────────────────
# NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────
class DQNNetwork(nn.Module):
    """
    Single neural network for DQN.
    Used for both prediction and target network.
    Architecture: [3] → [64] → [64] → [7]

    Input  : normalized state vector (3 values)
    Output : Q-value for each action (7 values)
    """

    def __init__(self):
        super(DQNNetwork, self).__init__()

        # ── layer definitions ───────────────
        self.fc1 = nn.Linear(INPUT_SIZE,  HIDDEN_SIZE)  # 3  → 64
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)  # 64 → 64
        self.fc3 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)  # 64 → 7

        # ── activation function ─────────────
        self.relu = nn.ReLU()

        # ── weight initialization ───────────
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Xavier initialization for all layers.
        Sets weights to small random values
        scaled by layer size.
        Prevents vanishing/exploding gradients.
        """
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)   # biases start at 0

    def forward(self, x):
        """
        Forward pass through network.
        Called automatically by PyTorch.

        Parameters:
            x : tensor of shape (batch_size, 3)
                or (3,) for single state

        Returns:
            tensor of shape (batch_size, 7)
            or (7,) for single state
            → Q-value for each of 7 actions
        """
        # ── hidden layer 1 ──────────────────
        x = self.fc1(x)       # weighted sum + bias: 3 → 64
        x = self.relu(x)      # ReLU activation

        # ── hidden layer 2 ──────────────────
        x = self.fc2(x)       # weighted sum + bias: 64 → 64
        x = self.relu(x)      # ReLU activation

        # ── output layer ────────────────────
        x = self.fc3(x)       # weighted sum + bias: 64 → 7
                               # NO activation → raw Q-values
                               # Q-values can be negative
        return x


# ─────────────────────────────────────────
# DQN AGENT NETWORKS
# manages both prediction and target network
# ─────────────────────────────────────────
class DQNNetworks:
    """
    Manages prediction network and target network together.
    Handles:
        - forward pass for action selection
        - forward pass for Q-value prediction
        - forward pass for target Q-value
        - hard copy from prediction to target
    """

    def __init__(self):
        # ── device setup ────────────────────
        # use GPU if available, else CPU
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        # ── create both networks ─────────────
        self.prediction_network = DQNNetwork().to(self.device)
        self.target_network     = DQNNetwork().to(self.device)

        # ── copy prediction → target ─────────
        # both start with identical weights
        self.hard_update()

        # ── freeze target network ────────────
        # target network never receives gradients
        for param in self.target_network.parameters():
            param.requires_grad = False

        # ── optimizer for prediction network ─
        self.optimizer = torch.optim.Adam(
            self.prediction_network.parameters(),
            lr=LEARNING_RATE
        )

        # ── loss function ────────────────────
        self.loss_fn = nn.MSELoss()

        print(f"Prediction Network parameters: "
              f"{sum(p.numel() for p in self.prediction_network.parameters())}")
        print(f"Target Network parameters    : "
              f"{sum(p.numel() for p in self.target_network.parameters())}")

    # ─────────────────────────────────────────
    # GET Q-VALUES — prediction network
    # used for action selection
    # ─────────────────────────────────────────
    def get_q_values(self, state):
        """
        Forward pass through prediction network.
        Used for action selection during environment interaction.

        Parameters:
            state : numpy array (3,)

        Returns:
            q_values : numpy array (7,)
        """
        # convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)

        # no gradient needed for action selection
        with torch.no_grad():
            q_values = self.prediction_network(state_tensor)

        return q_values.cpu().numpy()

    # ─────────────────────────────────────────
    # GET PREDICTED Q — prediction network
    # used during training for loss calculation
    # ─────────────────────────────────────────
    def get_predicted_q(self, states, actions):
        """
        Forward pass through prediction network.
        Gets Q-value ONLY for the action that was taken.
        Used during training.

        Parameters:
            states  : tensor (batch_size, 3)
            actions : tensor (batch_size,)

        Returns:
            q_predicted : tensor (batch_size, 1)
        """
        # forward pass → all Q-values
        all_q_values = self.prediction_network(states)  # (64, 7)

        # select only Q-value of action that was taken
        # actions.unsqueeze(1) → (64, 1)
        # gather picks one Q-value per row
        q_predicted = all_q_values.gather(1, actions.unsqueeze(1))

        return q_predicted  # (64, 1)

    # ─────────────────────────────────────────
    # GET TARGET Q — target network
    # used during training for y_i calculation
    # ─────────────────────────────────────────
    def get_target_q(self, next_states):
        """
        Forward pass through TARGET network.
        Gets max Q-value across all actions for next state.
        Used to calculate y_i = r + gamma * max Q(s', a'; theta-)

        Parameters:
            next_states : tensor (batch_size, 3)

        Returns:
            max_q : tensor (batch_size, 1)
        """
        # no gradient — target network never trains
        with torch.no_grad():
            all_q_values = self.target_network(next_states)  # (64, 7)
            max_q = all_q_values.max(dim=1, keepdim=True)[0]  # (64, 1)

        return max_q

    # ─────────────────────────────────────────
    # HARD UPDATE — copy prediction → target
    # called every TARGET_UPDATE_N steps
    # ─────────────────────────────────────────
    def hard_update(self):
        """
        Copies all weights and biases from
        prediction network to target network.
        Target network becomes snapshot of
        prediction network at this moment.
        """
        self.target_network.load_state_dict(
            self.prediction_network.state_dict()
        )

    # ─────────────────────────────────────────
    # SAVE — save prediction network weights
    # ─────────────────────────────────────────
    def save(self, path):
        """
        Saves prediction network weights to disk.
        Call after training completes.

        Parameters:
            path : string → file path e.g. 'results/models/dqn.pth'
        """
        torch.save(
            self.prediction_network.state_dict(),
            path
        )
        print(f"Model saved to {path}")

    # ─────────────────────────────────────────
    # LOAD — load saved weights
    # ─────────────────────────────────────────
    def load(self, path):
        """
        Loads saved weights into prediction network.
        Also copies to target network.

        Parameters:
            path : string → file path to saved .pth file
        """
        self.prediction_network.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.hard_update()
        print(f"Model loaded from {path}")

    # ─────────────────────────────────────────
    # PRINT ARCHITECTURE
    # ─────────────────────────────────────────
    def print_architecture(self):
        """
        Prints network architecture summary.
        """
        print("\n── Network Architecture ────────────")
        print(f"  Input  : {INPUT_SIZE}  (mac, flood, age)")
        print(f"  Hidden : {HIDDEN_SIZE} → {HIDDEN_SIZE}")
        print(f"  Output : {OUTPUT_SIZE}  (one per action)")
        print(f"  Activation : ReLU (hidden), None (output)")
        print(f"  Optimizer  : Adam  lr={LEARNING_RATE}")
        print(f"  Loss       : MSELoss")
        print("────────────────────────────────────\n")
        print(self.prediction_network)