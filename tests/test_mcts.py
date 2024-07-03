import unittest
import numpy as np
from mcts import MCTS, MCTSNode
from gomoku import Gomoku
from network import GomokuNet

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.game = Gomoku(board_size=5)  # Use small board for testing
        model = GomokuNet(board_size=5)
        self.mcts = MCTS(model=model, num_simulations=100)

    def test_search_returns_move(self):
        move = self.mcts.search(self.game)
        self.assertIsNotNone(move)
        self.assertTrue(isinstance(move, tuple))
        self.assertEqual(len(move), 2)

    def test_search_on_finished_game(self):
        # Create winning position.
        self.game.board = np.array([
            [1, 1, 1, 1, 1],
            [0, 2, 0, 2, 2],
            [2, 0, 2, 2, 2],
            [2, 0, 1, 1, 2],
            [2, 0, 1, 1, 2],
        ])
        move = self.mcts.search(self.game)
        self.assertIsNone(move)

    def test_expand_node(self):
        root = MCTSNode(self.game)
        self.mcts.expand_node(root)
        self.assertEqual(len(root.children), 25)  # There should be 25 legal moves on the empty board

    def test_backpropagate(self):
        game = self.game.clone()
        root = MCTSNode(game)
        child = MCTSNode(game, parent=root)
        search_path = [root, child]
        self.mcts.backpropagate(search_path, 1)
        self.assertEqual(root.visit_count, 1)
        self.assertEqual(root.value_sum, 0)
        self.assertEqual(child.visit_count, 1)
        self.assertEqual(child.value_sum, 1)


if __name__ == '__main__':
    unittest.main()
