import numpy as np


class Gomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.current_player = 3 - self.current_player  # Switch player (1 -> 2, 2 -> 1)
            return True
        return False

    def check_winner(self):
        # This is a placeholder. We'll implement the full winning check later.
        return 0  # 0 means no winner yet

    def is_full(self):
        return np.all(self.board != 0)

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1