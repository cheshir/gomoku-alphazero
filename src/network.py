import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuNet(nn.Module):
    def __init__(self, board_size=15):
        super(GomokuNet, self).__init__()
        self.board_size = board_size

        # First convolutional layer: extracts simple local features.
        # (N, 4, 15, 15) -> (N, 64, 15, 15)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)

        # Second convolutional layer: increases the number of features.
        # (N, 64, 15, 15) -> (N, 128, 15, 15)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Third convolutional layer: extracts more complex features.
        # (N, 128, 15, 15) -> (N, 128, 15, 15)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Fully connected layer: transforms 2D features to 1D.
        # (N, 128 * 15 * 15) -> (N, 256)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)

        # Fully connected layer for policy: returns action probabilities.
        # (N, 256) -> (N, 15 * 15)
        self.fc2 = nn.Linear(256, board_size * board_size)

        # Fully connected layer for value: returns position evaluation.
        # (N, 256) -> (N, 1)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = F.relu(self.fc1(x))

        policy = self.fc2(x)
        value = torch.tanh(self.fc3(x))

        return policy, value

def prepare_input(board, current_player, move_number):
    board_size = board.shape[0]
    # Create a tensor with 4 channels and the size of the board.
    # Shape: (4, board_size, board_size)
    input_tensor = np.zeros((4, board_size, board_size), dtype=np.float32)

    # First channel for player 1's pieces.
    # Values: 1.0 if cell contains player 1's piece, otherwise 0.0
    input_tensor[0][board == 1] = 1.0

    # Second channel for player 2's pieces.
    # Values: 1.0 if cell contains player 2's piece, otherwise 0.0
    input_tensor[1][board == 2] = 1.0

    # Third channel for the current player.
    # Values: 1.0 for all cells if current player is 1, otherwise 2.0
    input_tensor[2][:] = current_player

    # Fourth channel for the normalized move number.
    # Values: move_number / (board_size * board_size)
    input_tensor[3][:] = move_number / (board_size * board_size)

    # Add batch size dimension and convert to PyTorch tensor.
    # Shape: (1, 4, board_size, board_size)
    return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)
