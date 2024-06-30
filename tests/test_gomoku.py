import pytest
import numpy as np
from gomoku import Gomoku


def test_init():
    game = Gomoku()
    assert game.board_size == 15
    assert np.all(game.board == 0)
    assert game.current_player == 1


def test_make_move():
    game = Gomoku()
    assert game.make_move(7, 7) == True
    assert game.board[7, 7] == 1
    assert game.current_player == 2
    assert game.make_move(7, 7) == False  # Can't move to an occupied space


def test_check_winner_placeholder():
    game = Gomoku()
    assert game.check_winner() == 0


def test_is_full():
    game = Gomoku(board_size=3)
    assert game.is_full() == False
    game.board = np.ones((3, 3), dtype=int)
    assert game.is_full() == True


def test_reset():
    game = Gomoku()
    game.make_move(7, 7)
    game.reset()
    assert np.all(game.board == 0)
    assert game.current_player == 1


def test_custom_board_size():
    game = Gomoku(board_size=9)
    assert game.board_size == 9
    assert game.board.shape == (9, 9)