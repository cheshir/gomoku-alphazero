import pytest
import torch
from gomoku import Gomoku
from network import GomokuNet, ResidualBlock

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
    game = Gomoku(board_size=15)
    model = GomokuNet(board_size=15)
    input_tensor = model.prepare_input(game)
    assert input_tensor.shape == (1, 5, 15, 15)

def test_prepare_input_channels():
    game = Gomoku(board_size=15)
    model = GomokuNet(board_size=15)
    game.make_move(7, 7)
    input_tensor = model.prepare_input(game)
    assert input_tensor[0, 0, 7, 7] == 1  # Player 1's piece
    assert input_tensor[0, 2, 0, 0] == 0  # Current player is 2
    assert input_tensor[0, 3, 0, 0] == 1 / (15 * 15)  # Move count
    assert input_tensor[0, 4, 7, 7] == 1  # Last move

def test_residual_block():
    block = ResidualBlock(64)
    input_tensor = torch.randn(1, 64, 15, 15)
    output = block(input_tensor)
    assert output.shape == (1, 64, 15, 15)
