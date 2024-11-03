import unittest
import numpy as np
import torch
from mcts import MCTS, MCTSNode
from gomoku import Gomoku
from network import GomokuNet

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.game = Gomoku(board_size=5)  # Use small board for testing
        model = GomokuNet(board_size=5)
        self.mcts = MCTS(model=model, num_simulations=10, num_parallel=2, batch_size=2)

    def test_search_returns_move(self):
    # Mock only the model's forward method to return uniform probabilities
        def mock_forward(x):
            batch_size = x.shape[0]
            policy = torch.ones((batch_size, 25)) / 25  # Uniform policy
            value = torch.zeros(batch_size)  # Neutral value
            return policy, value
        self.mcts.model.__call__ = mock_forward

        move = self.mcts.search(self.game)
        self.assertIsNotNone(move)
        self.assertTrue(isinstance(move, tuple))
        self.assertEqual(len(move), 2)

    def test_search_on_finished_game(self):
        # Mock the model's predict_policy method to return uniform probabilities
        def mock_predict_policy(game):
            moves = game.get_legal_moves()
            probs = np.ones(len(moves)) / len(moves) if moves else []
            return list(zip(moves, probs))
        self.mcts.model.predict_policy = mock_predict_policy
        
        # Mock the model's prepare_input and forward methods
        def mock_prepare_input(game):
            return torch.zeros((1, 3, 5, 5))
        self.mcts.model.prepare_input = mock_prepare_input
        
        def mock_forward(x):
            batch_size = x.shape[0]
            policy = torch.ones((batch_size, 25)) / 25
            value = torch.zeros(batch_size)
            return policy, value
        self.mcts.model.__call__ = mock_forward

        # Create winning position
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
        # Set policy directly since we're not using the network in this test
        root.policy = np.ones(self.game.board_size * self.game.board_size) / (self.game.board_size * self.game.board_size)
        self.mcts.expand_node(root)
        self.assertEqual(len(root.children), 25)  # There should be 25 legal moves on the empty board
        # Verify each child has the correct prior probability
        expected_prior = 1.0 / (self.game.board_size * self.game.board_size)
        for child in root.children.values():
            self.assertAlmostEqual(child.prior, expected_prior)

    def test_backpropagate(self):
        game = self.game.clone()
        root = MCTSNode(game)
        child = MCTSNode(game, parent=root)
        search_path = [root, child]
        self.mcts.backpropagate(search_path, 1)
        # Account for virtual loss being removed during backpropagation
        self.assertEqual(root.visit_count, 1 - self.mcts.virtual_loss)
        self.assertEqual(root.value_sum, self.mcts.virtual_loss)
        self.assertEqual(child.visit_count, 1 - self.mcts.virtual_loss)
        self.assertEqual(child.value_sum, 1 + self.mcts.virtual_loss)


if __name__ == '__main__':
    unittest.main()
