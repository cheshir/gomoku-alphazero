import math
import numpy as np
from gomoku import Gomoku
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from collections import defaultdict

class MCTSNode:
    def __init__(self, game, parent=None, action=None):
        self.game = game
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0
        self.lock = threading.Lock()  # Lock for thread safety
        self.policy = None
        self.value = None

class MCTS:
    """
    Monte Carlo Tree Search implementation for the game of Gomoku.

    This class implements the MCTS algorithm, which is used to determine the best move
    in a given game state. It uses a neural network model to evaluate positions and
    guide the search. Supports parallel search using multiple threads and batch prediction.

    Attributes:
        model: A neural network model used for position evaluation and move prediction.
        num_simulations: The number of simulations to run for each search.
        c_puct: The exploration constant used in the UCB formula.
        num_parallel: Number of parallel threads to use for search.
    """

    def __init__(
            self, 
            model, 
            num_simulations=800, 
            c_puct=1.0, 
            dirichlet_alpha=0.3, 
            dirichlet_epsilon=0.25,
            virtual_loss=3.0, 
            num_parallel=8,
            batch_size=8
        ):
        """
        Initializes the MCTS object.

        Args:
            model: A neural network model for position evaluation and move prediction.
            num_simulations: The number of simulations to run for each search.
            c_puct: The exploration constant used in the UCB formula.
            num_parallel: Number of parallel threads to use.
            batch_size: Size of batches for network prediction.
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.virtual_loss = virtual_loss
        self.num_parallel = num_parallel
        self.batch_size = batch_size
        self.prediction_queue = []
        self.prediction_queue_lock = threading.Lock()
        self.board_size = model.board_size
        
        # Pattern values for move ordering
        self.pattern_values = {
            'FIVE': 100000,    # Five in a row
            'OPEN_FOUR': 10000,  # Four with both ends open
            'FOUR': 1000,      # Four with one end blocked
            'OPEN_THREE': 100,  # Three with both ends open
            'THREE': 10,       # Three with one end blocked
            'OPEN_TWO': 8,     # Two with both ends open
            'TWO': 2          # Two with one end blocked
        }
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonals

    def run_single_simulation(self, root):
        """Runs a single MCTS simulation with batched predictions"""
        node = root
        search_path = [node]

        # Selection
        while node.children and not node.game.is_game_over():
            node = self.select_child(node)
            search_path.append(node)

        # Expansion and evaluation
        if not node.game.is_game_over():
            # Get policy prediction if needed
            if node.policy is None:
                input_tensor = self.model.prepare_input(node.game)
                with torch.no_grad():
                    policy, value = self.model(input_tensor.unsqueeze(0))
                    policy = torch.softmax(policy, dim=1)
                    node.policy = policy[0].cpu().numpy()
                    node.value = value[0].item()

            with node.lock:
                if not node.children:  # Check again under lock
                    self.expand_node(node)
            
            node = self.select_child(node)
            search_path.append(node)

        # Get value if not already set
        if node.value is None:
            input_tensor = self.model.prepare_input(node.game)
            with torch.no_grad():
                _, value = self.model(input_tensor.unsqueeze(0))
                node.value = value[0].item()

        # Backpropagation
        self.backpropagate(search_path, node.value)

    def search(self, game):
        if game.is_game_over() or not game.get_legal_moves():
            return None

        root = MCTSNode(game)
        
        # Get initial policy for root
        input_tensor = self.model.prepare_input(game)
        with torch.no_grad():
            policy, _ = self.model(input_tensor.unsqueeze(0))
            policy = torch.softmax(policy, dim=1)
            root.policy = policy[0].cpu().numpy()

        # Initial expansion of root node
        self.expand_node(root)
        
        # Apply Dirichlet noise to root's children
        legal_moves = game.get_legal_moves()
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        
        for i, move in enumerate(legal_moves):
            if move in root.children:
                child = root.children[move]
                move_idx = move[0] * self.board_size + move[1]
                child.prior = (1 - self.dirichlet_epsilon) * root.policy[move_idx] + \
                            self.dirichlet_epsilon * noise[i]

        # Run parallel simulations
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = [
                executor.submit(self.run_single_simulation, root)
                for _ in range(self.num_simulations)
            ]
            
            # Wait for all simulations to complete
            for future in futures:
                future.result()

        return self.select_action(root)

    def process_prediction_queue(self):
        """Process queued nodes in batches for network prediction"""
        with self.prediction_queue_lock:
            if not self.prediction_queue:
                return
            
            nodes_to_process = self.prediction_queue[:self.batch_size]
            self.prediction_queue = self.prediction_queue[self.batch_size:]

        if nodes_to_process:
            # Prepare batch input
            batch_states = torch.stack([
                self.model.prepare_input(node.game)
                for node in nodes_to_process
            ])

            # Get batch predictions
            with torch.no_grad():
                policies, values = self.model(batch_states)
                policies = torch.softmax(policies, dim=1)

            # Update nodes with predictions
            for node, policy, value in zip(nodes_to_process, policies, values):
                node.policy = policy.cpu().numpy()
                node.value = value.item()

    def select_child(self, node):
        """
        Selects the child node with the highest UCB score.
        Thread-safe selection using virtual loss.
        """
        with node.lock:
            best_score = float('-inf')
            best_child = None
            for child in node.children.values():
                score = self.ucb_score(child)
                if score > best_score:
                    best_score = score
                    best_child = child
            
            # Apply virtual loss
            if best_child:
                best_child.visit_count += self.virtual_loss
                best_child.value_sum -= self.virtual_loss
            
            return best_child

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
        Uses cached policy if available, otherwise waits for prediction.
        """
        if node.policy is None:
            # Wait for prediction to be processed
            while node.policy is None:
                pass
        
        legal_moves = node.game.get_legal_moves()
        move_scores = defaultdict(float)
        
        for move in legal_moves:
            # Combine pattern-based evaluation with policy network probability
            pattern_score = self.evaluate_move(
                node.game.board.copy(), 
                move, 
                node.game.current_player
            )
            policy_score = node.policy[move[0] * self.board_size + move[1]]
            
            # Combine scores
            move_scores[move] = 0.7 * pattern_score + 0.3 * policy_score
        
        # Sort moves by score and create child nodes in order
        sorted_moves = sorted(legal_moves, key=lambda m: move_scores[m], reverse=True)
        
        for move in sorted_moves:
            if move not in node.children:
                new_game = node.game.clone()
                new_game.make_move(*move)
                new_node = MCTSNode(new_game, parent=node, action=move)
                new_node.prior = node.policy[move[0] * self.board_size + move[1]]
                node.children[move] = new_node

    def backpropagate(self, search_path, value):
        """
        Updates the statistics of all nodes in the search path.
        Thread-safe backpropagation.
        """
        for node in reversed(search_path):
            with node.lock:
                node.visit_count += 1
                node.value_sum += value
                # Remove virtual loss if it was applied
                node.visit_count -= self.virtual_loss
                node.value_sum += self.virtual_loss
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

    def evaluate_move(self, board, move, player):
        """
        Evaluate a potential move's value based on patterns it creates or blocks.
        """
        row, col = move
        if board[row, col] != 0:  # If cell is not empty
            return -float('inf')
            
        # Make temporary move
        board[row, col] = player
        score = 0
        
        # Check patterns for both attack and defense
        attack_score = self._check_patterns(board, move, player)
        opponent = 3 - player
        defense_score = self._check_patterns(board, move, opponent) * 0.9  # Slightly lower weight for defensive moves
        
        # Undo temporary move
        board[row, col] = 0
        
        # Combine scores
        score = max(attack_score, defense_score)
        
        # Add distance-based heuristic
        score += self._distance_heuristic(board, move)
        
        return score
    
    def _check_patterns(self, board, move, player):
        """Check for various patterns around a move."""
        row, col = move
        score = 0
        
        for dr, dc in self.directions:
            sequence = self._get_sequence(board, row, col, dr, dc, player)
            
            if self._check_five(sequence):
                score += self.pattern_values['FIVE']
            elif self._check_open_four(sequence):
                score += self.pattern_values['OPEN_FOUR']
            elif self._check_four(sequence):
                score += self.pattern_values['FOUR']
            elif self._check_open_three(sequence):
                score += self.pattern_values['OPEN_THREE']
            elif self._check_three(sequence):
                score += self.pattern_values['THREE']
            elif self._check_open_two(sequence):
                score += self.pattern_values['OPEN_TWO']
            elif self._check_two(sequence):
                score += self.pattern_values['TWO']
                
        return score
    
    def _get_sequence(self, board, row, col, dr, dc, player):
        """Get a sequence of cells in a given direction."""
        sequence = []
        for i in range(-4, 5):  # Look 4 cells in each direction
            r, c = row + dr * i, col + dc * i
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                sequence.append(board[r, c])
            else:
                sequence.append(-1)  # Out of bounds
        return sequence
    
    def _check_five(self, seq):
        """Check for five in a row."""
        return any(all(x == seq[i] for x in seq[i:i+5]) for i in range(5))
    
    def _check_open_four(self, seq):
        """Check for open four."""
        return '0' + '1' * 4 + '0' in ''.join(map(str, seq))
    
    def _check_four(self, seq):
        """Check for four."""
        return '1' * 4 in ''.join(map(str, seq))
    
    def _check_open_three(self, seq):
        """Check for open three."""
        pattern = ''.join(map(str, seq))
        return '0' + '1' * 3 + '0' in pattern or '01110' in pattern
    
    def _check_three(self, seq):
        """Check for three."""
        return '1' * 3 in ''.join(map(str, seq))
    
    def _check_open_two(self, seq):
        """Check for open two."""
        pattern = ''.join(map(str, seq))
        return '0' + '1' * 2 + '0' in pattern
    
    def _check_two(self, seq):
        """Check for two."""
        return '1' * 2 in ''.join(map(str, seq))
    
    def _distance_heuristic(self, board, move):
        """
        Calculate distance-based heuristic.
        Moves closer to existing pieces are generally more valuable.
        """
        row, col = move
        min_distance = float('inf')
        
        # Find closest existing piece
        for r in range(max(0, row-3), min(self.board_size, row+4)):
            for c in range(max(0, col-3), min(self.board_size, col+4)):
                if board[r, c] != 0:
                    distance = abs(r - row) + abs(c - col)
                    min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0
        
        return 5.0 / (min_distance + 1)  # Convert distance to score
