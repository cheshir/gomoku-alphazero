## Files structure
.
├── Makefile
├── requirements.txt
├── src
│  ├── __init__.py
│  ├── gomoku.py
│  ├── mcts.py
│  ├── network.py
│  ├── self_play.py
│  ├── train.py
│  └── web_server.py
├── templates
│  └── index.html
├── tests
│  ├── test_gomoku.py
│  ├── test_mcts.py
│  └── test_network.py

## Files content

### Makefile
```
.PHONY: test
test:
	pytest -v tests

.PHONY: serve
serve:
	python3.9 src/web_server.py

```

### requirements.txt
```
torch
torchvision
torchaudio
numpy==1.26.4
matplotlib
jupyter
pytest
flask
Flask-Session

```

### src/__init__.py
```

```

### src/gomoku.py
```
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


```

### src/mcts.py
```
import math
import numpy as np
from gomoku import Gomoku

class MCTSNode:
    def __init__(self, game, parent=None, action=None):
        self.game = game
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0

class MCTS:
    """
    Monte Carlo Tree Search implementation for the game of Gomoku.

    This class implements the MCTS algorithm, which is used to determine the best move
    in a given game state. It uses a neural network model to evaluate positions and
    guide the search.

    Attributes:
        model: A neural network model used for position evaluation and move prediction.
        num_simulations: The number of simulations to run for each search.
        c_puct: The exploration constant used in the UCB formula.
    """

    def __init__(self, model, num_simulations=800, c_puct=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        """
        Initializes the MCTS object.

        Args:
            model: A neural network model for position evaluation and move prediction.
            num_simulations: The number of simulations to run for each search.
            c_puct: The exploration constant used in the UCB formula.
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, game):
        if game.is_game_over() or not game.get_legal_moves():
            return None

        root = MCTSNode(game)

        legal_moves = game.get_legal_moves()
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))

        for move, prob in self.get_policy(game):
            if move in root.children:
                root.children[move].prior = (1 - self.dirichlet_epsilon) * prob + self.dirichlet_epsilon * noise[legal_moves.index(move)]

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.children and not node.game.is_game_over():
                node = self.select_child(node)
                search_path.append(node)

            if not node.game.is_game_over():
                self.expand_node(node)
                node = self.select_child(node)
                search_path.append(node)

            value = self.evaluate(node.game)
            self.backpropagate(search_path, value)

        return self.select_action(root)

    def select_child(self, node):
        """
        Selects the child node with the highest UCB score.
        """
        return max(node.children.values(), key=lambda n: self.ucb_score(n))

    def ucb_score(self, node):
        """
        Calculates the Upper Confidence Bound (UCB) score for a node.
        """
        q_value = node.value_sum / node.visit_count if node.visit_count > 0 else 0
        u_value = (self.c_puct * node.prior *
                   math.sqrt(node.parent.visit_count) / (1 + node.visit_count))
        return q_value + u_value

    def expand_node(self, node):
        """
        Expands the given node by adding all possible child nodes.
        """
        policy = self.get_policy(node.game)
        for move, prob in policy:
            if move not in node.children:
                new_game = node.game.clone()
                new_game.make_move(*move)
                new_node = MCTSNode(new_game, parent=node, action=move)
                new_node.prior = prob
                node.children[move] = new_node

    def evaluate(self, game):
        """
        Evaluates the given game state using the value network.
        """
        return self.model.predict_value(game)

    def backpropagate(self, search_path, value):
        """
        Updates the statistics of all nodes in the search path.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = 1 - value  # Flip the value for the opponent

    def select_action(self, root):
        """
        Selects the best action based on the visit counts of the root's children.
        """
        return max(root.children, key=lambda move: root.children[move].visit_count)

    def get_policy(self, game):
        """
        Gets the move probabilities for the current game state from the policy network.
        """
        return self.model.predict_policy(game)

    def get_action_prob(self, game, temperature=1):
        """
        Returns the action probabilities based on visit counts and temperature.
        """
        root = MCTSNode(game)
        self.search(game)

        visits = [child.visit_count for child in root.children.values()]
        actions = list(root.children.keys())

        if temperature == 0:
            action = actions[np.argmax(visits)]
            probs = [0] * len(actions)
            probs[actions.index(action)] = 1
            return [(a, p) for a, p in zip(actions, probs)]

        visits = [v ** (1. / temperature) for v in visits]
        total = sum(visits)
        probs = [v / total for v in visits]

        return [(a, p) for a, p in zip(actions, probs)]

    def get_best_move(self, game):
        root = MCTSNode(game)
        return self.search(game)

    def select_action_with_temperature(self, root, temperature):
        visits = [child.visit_count for child in root.children.values()]
        actions = list(root.children.keys())

        if temperature == 0:
            return actions[np.argmax(visits)]

        visits = [v ** (1. / temperature) for v in visits]
        total = sum(visits)
        probs = [v / total for v in visits]

        return np.random.choice(actions, p=probs)

```

### src/network.py
```
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Input shape: (N, channels, board_size, board_size)
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        # Shape after conv1: (N, channels, board_size, board_size)
        out = self.bn2(self.conv2(out))
        # Shape after conv2: (N, channels, board_size, board_size)
        out += residual
        return F.relu(out)
        # Output shape: (N, channels, board_size, board_size)

class GomokuNet(nn.Module):
    def __init__(self, board_size=15, num_residual_blocks=10):
        super(GomokuNet, self).__init__()
        self.board_size = board_size

        # Input layer
        self.conv_input = nn.Sequential(
            nn.Conv2d(5, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Input shape: (N, 5, board_size, board_size)
        # Output shape: (N, 256, board_size, board_size)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )
        # Input shape: (N, 256, board_size, board_size)
        # Output shape: (N, 256, board_size, board_size)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )
        # Input shape: (N, 256, board_size, board_size)
        # Output shape: (N, board_size * board_size)

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        # Input shape: (N, 256, board_size, board_size)
        # Output shape: (N, 1)

    def forward(self, x):
        # Input shape: (N, 5, board_size, board_size)
        x = self.conv_input(x)
        # Shape after conv_input: (N, 256, board_size, board_size)
        x = self.residual_tower(x)
        # Shape after residual_tower: (N, 256, board_size, board_size)
        policy = self.policy_head(x)
        # Shape of policy: (N, board_size * board_size)
        value = self.value_head(x)
        # Shape of value: (N, 1)
        return policy, value

    def save_model(self, filepath):
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.load_state_dict(torch.load(filepath))
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")

    def predict_policy(self, game):
        """
        Predicts the policy (move probabilities) for the given game state.

        :param game: A Gomoku game instance
        :return: A list of (move, probability) tuples
        """
        board_tensor = self.prepare_input(game)
        with torch.no_grad():
            policy, _ = self(board_tensor)

        # Convert policy to probabilities
        policy = F.softmax(policy.squeeze(), dim=0).cpu().numpy()

        # Create a list of (move, probability) tuples for all legal moves
        moves = game.get_legal_moves()
        move_probs = [(move, policy[move[0] * self.board_size + move[1]]) for move in moves]

        # Normalize probabilities to sum to 1
        total_prob = sum(prob for _, prob in move_probs)
        move_probs = [(move, prob / total_prob) for move, prob in move_probs]

        return move_probs

    def predict_value(self, game):
        """
        Predicts the value of the given game state.

        :param game: A Gomoku game instance
        :return: A float value between -1 and 1
        """
        board_tensor = self.prepare_input(game)
        with torch.no_grad():
            _, value = self(board_tensor)
        return value.item()

    def prepare_input(self, game):
        """
        Prepares the input tensor for the neural network from the game state.

        :param game: A Gomoku game instance
        :return: A tensor of shape (1, 5, board_size, board_size)
        """
        # Create a tensor with 5 channels
        input_tensor = np.zeros((5, self.board_size, self.board_size), dtype=np.float32)

        # First channel: player 1's pieces
        input_tensor[0] = (game.board == 1).astype(np.float32)

        # Second channel: player 2's pieces
        input_tensor[1] = (game.board == 2).astype(np.float32)

        # Third channel: current player
        input_tensor[2] = np.full((self.board_size, self.board_size),
                                  game.current_player == 1, dtype=np.float32)

        # Fourth channel: move number (normalized)
        input_tensor[3] = np.full((self.board_size, self.board_size),
                                  game.move_count / (self.board_size ** 2), dtype=np.float32)

        # Fifth channel: last move
        if game.last_move:
            input_tensor[4, game.last_move[0], game.last_move[1]] = 1.0

        # Convert to PyTorch tensor and add batch dimension
        return torch.from_numpy(input_tensor).unsqueeze(0)

    def loss(self, policy_output, value_output, policy_targets, value_targets):
        """
        Compute the combined loss for policy and value outputs.

        :param policy_output: Predicted policy (move probabilities)
        :param value_output: Predicted value of the position
        :param policy_targets: True policy (from MCTS)
        :param value_targets: True game outcome
        :return: Combined loss (scalar)
        """
        policy_loss = F.cross_entropy(policy_output, policy_targets)
        value_loss = F.mse_loss(value_output, value_targets)
        return policy_loss + value_loss

class GomokuDataset(Dataset):
    def __init__(self, board_states, policies, values):
        self.board_states = board_states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.board_states)

    def __getitem__(self, idx):
        return self.board_states[idx], self.policies[idx], self.values[idx]

def create_data_loader(board_states, policies, values, batch_size=32):
    dataset = GomokuDataset(board_states, policies, values)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, optimizer, board_states, policies, values, epochs=10, batch_size=32, device='cpu'):
    """
    Trains the model on the provided data.

    :param model: GomokuNet model to train
    :param optimizer: Optimizer for updating model weights
    :param board_states: List of board states (input to the model)
    :param policies: List of policy targets
    :param values: List of value targets
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training
    :param device: Device for training ('cpu' or 'cuda')
    """
    model.to(device)
    model.train()

    data_loader = create_data_loader(board_states, policies, values, batch_size)

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            board_states_batch, policy_targets, value_targets = batch
            board_states_batch = board_states_batch.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            policy_output, value_output = model(board_states_batch)

            policy_loss = F.cross_entropy(policy_output, policy_targets)
            value_loss = F.mse_loss(value_output, value_targets)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

    print("Training completed")

```

### src/self_play.py
```
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
```

### src/train.py
```
import torch
import torch.optim as optim
from network import GomokuNet, create_data_loader
from self_play import self_play
import os
import glob

def get_latest_model(model_dir="models"):
    """Find the latest model in the models directory."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_files = glob.glob(os.path.join(model_dir, "model_iteration_*.pth"))
    if not model_files:
        return None
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

def train_iteration(model, num_games=100, mcts_simulations=800, epochs=10, batch_size=32, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Generating self-play data...")
    training_data = self_play(model, num_games=num_games, mcts_simulations=mcts_simulations)

    print("Training model...")
    model.to(device)
    model.train()

    board_states, policies, values = zip(*training_data)
    data_loader = create_data_loader(board_states, policies, values, batch_size)

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            board_states_batch, policy_targets, value_targets = [item.to(device) for item in batch]

            optimizer.zero_grad()
            policy_output, value_output = model(board_states_batch)

            loss = model.loss(policy_output, value_output, policy_targets, value_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

    print("Training completed")
    return model

def train(
    model,
    start_iteration=1,
    total_iterations=10,
    games_per_iteration=100,
    mcts_simulations=800,
    epochs=5,
    batch_size=32,
    device='cpu'
):
    for i in range(start_iteration, start_iteration + total_iterations):
        print(f"Training iteration {i}")
        model = train_iteration(
            model,
            num_games=games_per_iteration,
            mcts_simulations=mcts_simulations,
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )

        # Save model after each iteration
        model.save_model(f"models/model_iteration_{i}.pth")

    print("Training complete!")
    return model

def main():
    board_size = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    latest_model = get_latest_model()
    if latest_model:
        print(f"Continuing training from {latest_model}")
        model = GomokuNet(board_size=board_size)
        model.load_model(latest_model)
        start_iteration = int(latest_model.split('_')[-1].split('.')[0]) + 1
    else:
        print("Starting new training")
        model = GomokuNet(board_size=board_size)
        start_iteration = 1

    total_iterations = 10  # Number of training iterations
    model = train(
        model,
        start_iteration=start_iteration,
        total_iterations=total_iterations,
        games_per_iteration=100,
        mcts_simulations=800,
        epochs=5,
        batch_size=32,
        device=device
    )


if __name__ == "__main__":
    main()
```

### src/web_server.py
```
from flask import Flask, render_template, jsonify, request
from mcts import MCTS
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
```

### templates/index.html
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gomoku</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
</head>
<body>
    <div class="container">
        <h1>Gomoku</h1>
        <div id="loading" class="hidden">Bot is thinking...</div>
        <div id="status">Player 1's turn</div>
        <div id="board"></div>
        <button onclick="resetGame()">Reset Game</button>

        <div class="settings">
            <label for="mcts-simulations">MCTS Simulations:</label>
            <input type="number" id="mcts-simulations" value="800" min="100" max="10000">
            <button onclick="updateSettings()">Update</button>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/axios/0.21.1/axios.min.js"></script>
    <script src="{{ url_for('static', filename='js/script.js') }}"></script>
</body>
</html>

```

### tests/test_gomoku.py
```
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

def test_check_winner_horizontal():
    game = Gomoku()
    for i in range(5):
        game.make_move(7, i, switch_player=False)
    assert game.check_winner() == 1

def test_check_winner_vertical():
    game = Gomoku()
    for i in range(5):
        game.make_move(i, 7, switch_player=False)
    assert game.check_winner() == 1

def test_check_winner_diagonal():
    game = Gomoku()
    for i in range(5):
        game.make_move(i, i, switch_player=False)
    assert game.check_winner() == 1

def test_check_winner_anti_diagonal():
    game = Gomoku()
    for i in range(5):
        game.make_move(i, 4 - i, switch_player=False)
    assert game.check_winner() == 1

```

### tests/test_mcts.py
```
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

```

### tests/test_network.py
```
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

```

