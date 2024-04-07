from typing import Callable, List


class Minimax():
    def __init__(self, game_functionals: object) -> None:
        self.game_functionals = game_functionals
        self.first_len = 0
        self.first_node = None
        self.values = dict()
        self.n = -1
        self.first_player = True
        self.prev_heuristic = 0

    def alphabeta(
            self, node: list, heuristic: Callable, depth: int, alpha: int, beta: int, max_player: bool, i=0
    ):
        if i == 0:      # only true during first call
            self.first_len = len(node)
            self.first_node = node
            self.values = dict()
            self.n = -1
            self.first_player = max_player
            self.prev_heuristic = 0
            i += 1
        if depth == 0 or len(node) == 1:   # either max depth or terminal state
            final_value = heuristic(node)
            node = self.first_node
            return final_value

        if max_player:
            value = float('-inf')
            for move in node:
                if len(node) == self.first_len:     # true only during the very first moves
                    self.n += 1
                    self.values[self.n] = 0
                node.remove(move)
                child = node
                heuristic_value = heuristic(node)
                max_player = self.next_player(heuristic_value, self.prev_heuristic, max_player)
                self.prev_heuristic = heuristic_value
                value = max(value, self.alphabeta(child, heuristic, depth-1, alpha, beta, max_player, i))
                alpha = max(alpha, value)
                if value >= beta:
                    break
                if self.first_player:
                    self.values[self.n] = max(self.values[self.n], value)
            return value
        else:
            value = float('inf')
            for move in node:
                if len(node) == self.first_len:     # true only during the very first moves
                    self.n += 1
                    self.values[self.n] = 0
                node.remove(move)
                child = node
                max_player = self.next_player(heuristic(node), self.prev_heuristic, max_player)
                self.prev_heuristic = heuristic(node)
                value = min(value, self.alphabeta(child, heuristic, depth-1, alpha, beta, max_player, i))
                beta = min(beta, value)
                if value <= alpha:
                    break
                if not self.first_player:
                    self.values[self.n] = min(self.values[self.n], value)
            return value

    def next_player(self, heuristic: int, prev_heuristic: int, player: bool) -> bool:
        return self.game_functionals.next_player(heuristic, prev_heuristic, player)