import numpy as np
from network import GomokuNet

class Gomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.network = GomokuNet(board_size)

    def make_move(self, row, col, switch_player=True):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            if switch_player:
                self.current_player = 3 - self.current_player  # Switch player (1 -> 2, 2 -> 1)
            return True
        return False

    def check_winner(self):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # vertical, horizontal, diagonal (both directions)
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 0:
                    continue
                for d in directions:
                    if self._check_direction(row, col, d[0], d[1]):
                        return self.board[row, col]
        return 0  # No winner yet

    def _check_direction(self, row, col, dr, dc):
        player = self.board[row, col]
        count = 0
        for i in range(5):
            r = row + dr * i
            c = col + dc * i
            if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
            else:
                break
        return count == 5

    def is_full(self):
        return np.all(self.board != 0)

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
