from typing import Callable
from heuristic_DnB import heuristic_DnB
from random import choice
from copy import deepcopy


class Minimax():
    def __init__(self) -> None:
        self.possible_moves = dict()
        self.last_scores_max = 0
        self.last_scores_min = 0
        self.first_move = None

    # def get_heuristic(self, heuristic: Callable, node: Callable, max: bool, scores: Callable):
    #     if len(node()) == 1:
    #         return scores['max'] if max else scores['min']
    #     return heuristic(self.size)

    def alphabeta(
            self, node: Callable, game: object, state0: object, make_move: Callable, depth: int, alpha: int, beta: int, max_player: bool, scores: Callable, i=-1
    ):
        if i == -1:
            self.possible_moves = dict()
            i += 1
        moves = node()
        if depth == 0:   # max depth
            # make_move(move)
            return scores()['max'] - scores()['min']
        if len(moves) == 1:
            value = scores()['max'] - scores()['min']
            self.possible_moves[value] = [moves[0]]
            return value
        #     # make_move(moves[0])
        #     return scores()['max'] - scores()['min']
        if max_player:
            value = float('-inf')
            for move in moves:
                if i == 0:
                    self.first_move = deepcopy(move)
                    self.last_scores_max = 0
                    self.last_scores_min = 0
                make_move(move)
                scores_max = scores()['max']
                max_player = True if scores_max - self.last_scores_max == 1 else False
                i += 1
                self.last_scores_max = scores_max
                value = max(value, self.alphabeta(node, game, state0, make_move, depth-1, alpha, beta, max_player, scores, i))
                if value in self.possible_moves.keys():
                    self.possible_moves[value].append(self.first_move)
                else:
                    self.possible_moves[value] = [self.first_move]
                alpha = max(alpha, value)
                if value >= beta:
                    break
            i = 0
            return value
        else:
            value = float('inf')
            for move in moves:
                if i == 0:
                    self.first_move = deepcopy(move)
                    self.last_scores_max = 0
                    self.last_scores_min = 0
                make_move(move)
                scores_min = scores()['min']
                max_player = False if scores_min - self.last_scores_min == 1 else True
                i += 1
                value = min(value, self.alphabeta(node, game, state0, make_move, depth-1, alpha, beta, max_player, scores, i))
                if value in self.possible_moves.keys():
                    self.possible_moves[value].append(self.first_move)
                else:
                    self.possible_moves[value] = [self.first_move]
                beta = min(beta, value)
                if value <= alpha:
                    break
            i = 0
            game.state = state0
            return value

    def choose_move(self, value: int):
        return choice(self.possible_moves[value])