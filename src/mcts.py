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
    def __init__(self, model, num_simulations=800, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, game):
        """
        Performs Monte Carlo Tree Search from the given game state.

        This method builds a search tree by repeatedly selecting, expanding,
        and evaluating nodes. It balances exploration and exploitation to find
        the most promising moves.
        """
        if game.is_game_over():
            return None

        root = MCTSNode(game)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse the tree to a leaf node
            while node.children and not node.game.is_game_over():
                node = self.select_child(node)
                search_path.append(node)

            # Expansion and evaluation
            if not node.game.is_game_over():
                self.expand_node(node)
                node = self.select_child(node)
                search_path.append(node)

            value = self.evaluate(node.game)

            # Backpropagation: update statistics for all nodes in the search path
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
            return actions, probs

        visits = [v ** (1. / temperature) for v in visits]
        total = sum(visits)
        probs = [v / total for v in visits]

        return actions, probs
