import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Input shape: (N, channels, board_size, board_size)
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        # Shape after conv1: (N, channels, board_size, board_size)
        out = self.bn2(self.conv2(out))
        # Shape after conv2: (N, channels, board_size, board_size)
        out += residual
        return F.relu(out)
        # Output shape: (N, channels, board_size, board_size)

class GomokuNet(nn.Module):
    def __init__(self, board_size=15, num_residual_blocks=10):
        super(GomokuNet, self).__init__()
        self.board_size = board_size

        # Input layer
        self.conv_input = nn.Sequential(
            nn.Conv2d(5, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Input shape: (N, 5, board_size, board_size)
        # Output shape: (N, 256, board_size, board_size)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )
        # Input shape: (N, 256, board_size, board_size)
        # Output shape: (N, 256, board_size, board_size)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )
        # Input shape: (N, 256, board_size, board_size)
        # Output shape: (N, board_size * board_size)

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        # Input shape: (N, 256, board_size, board_size)
        # Output shape: (N, 1)

    def forward(self, x):
        # Input shape: (N, 5, board_size, board_size)
        x = self.conv_input(x)
        # Shape after conv_input: (N, 256, board_size, board_size)
        x = self.residual_tower(x)
        # Shape after residual_tower: (N, 256, board_size, board_size)
        policy = self.policy_head(x)
        # Shape of policy: (N, board_size * board_size)
        value = self.value_head(x)
        # Shape of value: (N, 1)
        return policy, value

    def save_model(self, filepath):
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.load_state_dict(torch.load(filepath))
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")

    def predict_policy(self, game):
        """
        Predicts the policy (move probabilities) for the given game state.

        :param game: A Gomoku game instance
        :return: A list of (move, probability) tuples
        """
        board_tensor = self.prepare_input(game)
        with torch.no_grad():
            policy, _ = self(board_tensor)

        # Convert policy to probabilities
        policy = F.softmax(policy.squeeze(), dim=0).cpu().numpy()

        # Create a list of (move, probability) tuples for all legal moves
        moves = game.get_legal_moves()
        move_probs = [(move, policy[move[0] * self.board_size + move[1]]) for move in moves]

        # Normalize probabilities to sum to 1
        total_prob = sum(prob for _, prob in move_probs)
        move_probs = [(move, prob / total_prob) for move, prob in move_probs]

        return move_probs

    def predict_value(self, game):
        """
        Predicts the value of the given game state.

        :param game: A Gomoku game instance
        :return: A float value between -1 and 1
        """
        board_tensor = self.prepare_input(game)
        with torch.no_grad():
            _, value = self(board_tensor)
        return value.item()

    def prepare_input(self, game):
        """
        Prepares the input tensor for the neural network from the game state.

        :param game: A Gomoku game instance
        :return: A tensor of shape (1, 5, board_size, board_size)
        """
        # Create a tensor with 5 channels
        input_tensor = np.zeros((5, self.board_size, self.board_size), dtype=np.float32)

        # First channel: player 1's pieces
        input_tensor[0] = (game.board == 1).astype(np.float32)

        # Second channel: player 2's pieces
        input_tensor[1] = (game.board == 2).astype(np.float32)

        # Third channel: current player
        input_tensor[2] = np.full((self.board_size, self.board_size),
                                  game.current_player == 1, dtype=np.float32)

        # Fourth channel: move number (normalized)
        input_tensor[3] = np.full((self.board_size, self.board_size),
                                  game.move_count / (self.board_size ** 2), dtype=np.float32)

        # Fifth channel: last move
        if game.last_move:
            input_tensor[4, game.last_move[0], game.last_move[1]] = 1.0

        # Convert to PyTorch tensor and add batch dimension
        return torch.from_numpy(input_tensor).unsqueeze(0)

    def loss(self, policy_output, value_output, policy_targets, value_targets):
        """
        Compute the combined loss for policy and value outputs.

        :param policy_output: Predicted policy (move probabilities)
        :param value_output: Predicted value of the position
        :param policy_targets: True policy (from MCTS)
        :param value_targets: True game outcome
        :return: Combined loss (scalar)
        """
        policy_loss = F.cross_entropy(policy_output, policy_targets)
        value_loss = F.mse_loss(value_output, value_targets)
        return policy_loss + value_loss

class GomokuDataset(Dataset):
    def __init__(self, board_states, policies, values):
        self.board_states = board_states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.board_states)

    def __getitem__(self, idx):
        return self.board_states[idx], self.policies[idx], self.values[idx]

def create_data_loader(board_states, policies, values, batch_size=32):
    dataset = GomokuDataset(board_states, policies, values)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, optimizer, board_states, policies, values, epochs=10, batch_size=32, device='cpu'):
    """
    Trains the model on the provided data.

    :param model: GomokuNet model to train
    :param optimizer: Optimizer for updating model weights
    :param board_states: List of board states (input to the model)
    :param policies: List of policy targets
    :param values: List of value targets
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training
    :param device: Device for training ('cpu' or 'cuda')
    """
    model.to(device)
    model.train()

    data_loader = create_data_loader(board_states, policies, values, batch_size)

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            board_states_batch, policy_targets, value_targets = batch
            board_states_batch = board_states_batch.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            policy_output, value_output = model(board_states_batch)

            policy_loss = F.cross_entropy(policy_output, policy_targets)
            value_loss = F.mse_loss(value_output, value_targets)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

    print("Training completed")
