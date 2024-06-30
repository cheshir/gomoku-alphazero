import pytest
import numpy as np
import torch
from network import GomokuNet, prepare_input, ResidualBlock

@pytest.fixture(scope="module")
def net():
    return GomokuNet(board_size=15, num_residual_blocks=10)

def test_network_initialization(net):
    # Check the shape of weights after network initialization
    assert net.conv_input[0].weight.shape == (256, 5, 3, 3)
    assert isinstance(net.residual_tower[0], ResidualBlock)
    assert len(net.residual_tower) == 10
    assert net.policy_head[0].weight.shape == (2, 256, 1, 1)
    assert net.policy_head[4].weight.shape == (15 * 15, 2 * 15 * 15)
    assert net.value_head[0].weight.shape == (1, 256, 1, 1)
    assert net.value_head[4].weight.shape == (256, 15 * 15)
    assert net.value_head[6].weight.shape == (1, 256)

def test_forward_pass(net):
    # Create an input tensor with shape (1, 5, 15, 15)
    input_tensor = torch.zeros((1, 5, 15, 15))
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
    last_move = (2, 3)  # Example last move
    input_tensor = prepare_input(board, current_player, move_number, last_move)

    # Check that the shape is (1, 5, board_size, board_size)
    assert input_tensor.shape == (1, 5, board.shape[0], board.shape[1])

def test_prepare_input_channels():
    board = np.array([
        [0, 1, 0],
        [2, 0, 1],
        [0, 2, 0]
    ])
    current_player = 2
    move_number = 5
    last_move = (1, 0)
    input_tensor = prepare_input(board, current_player, move_number, last_move)

    # Check first channel (player 1 positions)
    assert torch.equal(input_tensor[0, 0], torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32))

    # Check second channel (player 2 positions)
    assert torch.equal(input_tensor[0, 1], torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32))

    # Check third channel (current player)
    assert torch.all(input_tensor[0, 2] == 0)  # Current player is 2, so all values should be 0

    # Check fourth channel (normalized move number)
    assert torch.all(input_tensor[0, 3] == 5 / 9)  # 5 / (3 * 3)

    # Check fifth channel (last move)
    expected_last_move = torch.zeros((3, 3))
    expected_last_move[1, 0] = 1
    assert torch.equal(input_tensor[0, 4], expected_last_move)

def test_residual_block():
    block = ResidualBlock(64)
    input_tensor = torch.randn(1, 64, 15, 15)
    output = block(input_tensor)
    assert output.shape == (1, 64, 15, 15)
