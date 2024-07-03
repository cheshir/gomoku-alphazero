from flask import Flask, render_template, jsonify, request
import numpy as np
import torch
from network import GomokuNet
from gomoku import Gomoku

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize the model
board_size = 15
model = GomokuNet(board_size)
#model.load_state_dict(torch.load("gomoku_model_final.pth"))
model.eval()

# Initialize a new game
game = Gomoku(board_size)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    row = data['row']
    col = data['col']
    current_player = data['current_player']

    if game.check_winner() != 0:
        return jsonify({'error': 'Game has ended'}), 400

    if game.make_move(row, col):
        board = game.board
        move_number = np.sum(board != 0)
        input_tensor = model.prepare_input(board, current_player, move_number)

        with torch.no_grad():
            policy, _ = model(input_tensor)

        np.set_printoptions(precision=3, suppress=True) # Ensure NumPy is initialized
        policy = policy.squeeze().detach().cpu().numpy()  # Ensure tensor is detached and on CPU
        # Remove invalid moves (positions already occupied) and normalize probabilities
        valid_moves = (board == 0).flatten()
        policy = np.where(valid_moves, policy, 0)  # Set probability to 0 for invalid moves

        # Normalize policy
        total_prob = np.sum(policy)
        if total_prob > 0:
            policy /= total_prob
        else:
            # Default to uniform distribution if no valid moves
            policy = np.ones(board_size * board_size) * valid_moves / np.sum(valid_moves)

        policy = np.maximum(policy, 0)  # Ensure no negative probabilities
        policy /= np.sum(policy)  # Ensure the probabilities sum to 1

        move = np.random.choice(board_size * board_size, p=policy)
        row, col = divmod(move, board_size)
        game.make_move(row, col)

        response = {
            'board': game.board.astype(int).tolist(),
            'current_player': int(game.current_player),
            'winner': int(game.check_winner())
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid move'}), 400

@app.route('/reset', methods=['POST'])
def reset():
    global game
    game = Gomoku(board_size)
    return jsonify({'board': game.board.tolist(), 'current_player': game.current_player})


if __name__ == '__main__':
    app.run(debug=True)