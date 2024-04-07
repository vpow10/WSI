from typing import Callable, List


class Minimax():
    def __init__(self, size) -> None:
        self.size = size

    def get_heuristic(self, heuristic: Callable, moves: List, max: bool, scores: Callable):
        if len(moves) == 1:
            return scores()['max'] if max else scores()['min']
        return heuristic(self.size)

    def alphabeta(
            self, node: Callable, heuristic: Callable, make_move: Callable, depth: int, alpha: int, beta: int, max_player: bool, scores: Callable
    ):
        best_moves = dict()
        moves = node()
        if depth == 0 or len(moves) == 1:
            return self.get_heuristic(heuristic, moves, max_player, scores)
        if max_player:
            value = float('-inf')
            for move in moves:
                make_move(move)
                if value in best_moves.keys():
                    best_moves[value].append(move)
                else:
                    best_moves[value] = [move]
                value = max(value, self.alphabeta(node, heuristic, make_move, depth-1, alpha, beta, max_player, scores))
                alpha = max(alpha, value)
                if value >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in moves:
                make_move(move)
                value = min(value, self.alphabeta(move, depth-1, alpha, beta, max_player, scores))
                beta = min(beta, value)
                if value <= alpha:
                    break
            return value
