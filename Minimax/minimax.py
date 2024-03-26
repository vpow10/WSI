from dots_and_boxes import DotsAndBoxesState, DotsAndBoxesMove
from typing import Callable
from heuristic_DnB import heuristic_DnB


class Minimax():
    def __init__(self, size) -> None:
        self.size = size

    def get_heuristic(self, heuristic: Callable, node: Callable, max: bool, scores: Callable):
        if len(node()) == 1:
            return scores['max'] if max else scores['min']
        return heuristic(self.size)

    def alphabeta(
            self, node: Callable, depth: int, alpha: int, beta: int, max: bool, scores: Callable
    ):
        if depth == 0 or len(node()) == 1:
            return self.get_heuristic(node, max, scores)
        if max:
            value = float('-inf')
            for move in node():
                value = max(value, self.alphabeta(move, depth-1, alpha, beta, max, scores))
                alpha = max(alpha, value)
                if value >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in node():
                value = min(value, self.alphabeta(move, depth-1, alpha, beta, max, scores))
                beta = min(beta, value)
                if value <= alpha:
                    break
            return value
