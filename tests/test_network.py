import pytest
import numpy as np
import torch
from network import GomokuNet, prepare_input

@pytest.fixture(scope="module")
def net():
    return GomokuNet(board_size=15)

def test_network_initialization(net):
    # Check the shape of weights after network initialization
    assert net.conv1.weight.shape == (64, 4, 3, 3)
    assert net.conv2.weight.shape == (128, 64, 3, 3)
    assert net.conv3.weight.shape == (128, 128, 3, 3)
    assert net.fc1.weight.shape == (256, 128 * 15 * 15)
    assert net.fc2.weight.shape == (15 * 15, 256)
    assert net.fc3.weight.shape == (1, 256)

def test_forward_pass(net):
    # Create an input tensor with shape (1, 4, 15, 15)
    input_tensor = torch.zeros((1, 4, 15, 15))
    policy, value = net(input_tensor)
    # Check the shape of output tensors
    assert policy.shape == (1, 15 * 15)
    assert value.shape == (1, 1)

def test_prepare_input_shape():
    board = np.array([
        [0, 1, 0, 0, 2],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 2, 0],
        [1, 0, 0, 0, 0],
        [0, 2, 0, 0, 1]
    ])
    current_player = 1
    move_number = 10
    input_tensor = prepare_input(board, current_player, move_number)

    # Check that the shape is (1, 4, board_size, board_size)
    assert input_tensor.shape == (1, 4, board.shape[0], board.shape[1])
