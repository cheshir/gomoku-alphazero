import numpy as np
from network import GomokuNet

NUMBER_OF_CONSECUTIVE_PIECES_TO_WIN = 5

class Gomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.move_count = 0

    def make_move(self, row, col, switch_player=True):
        if self.board[row, col] != 0:
            return False
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.move_count += 1
        if switch_player:
            self.current_player = 3 - self.current_player
        return True

    def get_legal_moves(self):
        return [(row, col) for row in range(self.board_size)
                for col in range(self.board_size)
                if self.board[row, col] == 0]

    def is_game_over(self):
        return self.check_winner() != 0 or self.is_full()

    def check_winner(self):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # vertical, horizontal, diagonal (both directions)
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 0:
                    continue
                for dr, dc in directions:
                    if self._check_direction(row, col, dr, dc):
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
        return count == NUMBER_OF_CONSECUTIVE_PIECES_TO_WIN

    def is_full(self):
        return np.all(self.board != 0)

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1

    def clone(self):
        new_game = Gomoku(self.board_size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        new_game.move_count = self.move_count
        return new_game

