import numpy as np
from mcts import MCTS
from gomoku import Gomoku

def self_play(model, num_games=100, mcts_simulations=800, temperature=1.0):
    mcts = MCTS(model, num_simulations=mcts_simulations)
    training_data = []

    for _ in range(num_games):
        game = Gomoku()
        game_history = []

        while not game.is_game_over():
            canonical_board = game.board * game.current_player
            mcts_probs = mcts.get_action_prob(game, temperature=temperature)
            game_history.append((canonical_board, game.current_player, mcts_probs))

            action = max(mcts_probs, key=lambda x: x[1])[0]
            game.make_move(*action)

        winner = game.check_winner()
        training_data.extend(
            (board, mcts_prob, winner * player)
            for board, player, mcts_prob in game_history
        )

    return training_data