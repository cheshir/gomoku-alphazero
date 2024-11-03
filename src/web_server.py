from flask import Flask, render_template, jsonify, request
from mcts import MCTS
from network import GomokuNet
from gomoku import Gomoku
import torch

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize the model
board_size = 15
model = GomokuNet(board_size)
model.load_state_dict(torch.load("gomoku_model_final.pth"))
model.eval()

# Initialize a new game
game = Gomoku(board_size)

mcts = MCTS(model, num_simulations=800)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    row = data['row']
    col = data['col']

    if game.check_winner() != 0:
        return jsonify({'error': 'Game has ended'}), 400

    if game.make_move(row, col):
        if game.check_winner() == 0:
            # Bot's turn
            bot_move = mcts.get_best_move(game)
            game.make_move(*bot_move)

        response = {
            'board': game.board.astype(int).tolist(),
            'current_player': int(game.current_player),
            'winner': int(game.check_winner())
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid move'}), 400

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    mcts.num_simulations = int(data['mcts_simulations'])
    return jsonify({'success': True})

@app.route('/reset', methods=['POST'])
def reset():
    global game
    game = Gomoku(board_size)
    return jsonify({'board': game.board.tolist(), 'current_player': game.current_player})


if __name__ == '__main__':
    app.run(debug=True)
