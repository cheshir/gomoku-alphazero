import torch
import torch.nn as nn
import torch.nn.functional as F
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

def prepare_input(board, current_player, move_number, last_move):
    board_size = board.shape[0]
    # Create a tensor with 5 channels and the size of the board
    input_tensor = np.zeros((5, board_size, board_size), dtype=np.float32)

    # First channel: player 1's pieces
    # Values: 1.0 if cell contains player 1's piece, otherwise 0.0
    input_tensor[0][board == 1] = 1.0

    # Second channel: player 2's pieces
    # Values: 1.0 if cell contains player 2's piece, otherwise 0.0
    input_tensor[1][board == 2] = 1.0

    # Third channel: current player
    # Values: 1.0 for all cells if current player is 1, otherwise 0.0
    input_tensor[2][:] = 1.0 if current_player == 1 else 0.0

    # Fourth channel: move number (normalized)
    # Values: move_number / (board_size * board_size)
    input_tensor[3][:] = move_number / (board_size * board_size)

    # Fifth channel: last move
    # Values: 1.0 for the cell of the last move, otherwise 0.0
    if last_move:
        input_tensor[4][last_move[0], last_move[1]] = 1.0

    # Add batch dimension and convert to PyTorch tensor
    # Final shape: (1, 5, board_size, board_size)
    return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)
