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
