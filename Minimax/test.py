from typing import Dict, Callable, List
import sys
sys.path.insert(1, '/home/jasiek/WSI/two-player-games/two_player_games/games')
sys.path.insert(2, '/home/jasiek/WSI/two-player-games')
from dots_and_boxes import DotsAndBoxes
from random import choice
from minimax import Minimax
from copy import deepcopy


class Minimax:
    def __init__(self, size: int):
        self.first_player = True
        self.values = {}
        self.all_moves = DotsAndBoxes(size).get_moves()
        self.prev_score_max = 0
        self.prev_score_min = 0
        self.n = 0
        self.starting_len = 0
        self.after_break = False

    def alphabeta(self, state, depth, alpha, beta, maximizing_player, i=0):
        if i == 0:      # only true during first call
            self.values = dict()
            self.first_player = maximizing_player
            self.first_player_name = 'max' if maximizing_player else 'min'
            self.second_player_name = 'min' if maximizing_player else 'max'
            self.prev_score_max = 0
            self.prev_score_min = 0
            i += 1
            self.starting_len = len(state.get_moves())
        if depth == 0 or len(state.get_moves()) == 0:
            return state.get_scores()['max'] - state.get_scores()['min']
        if maximizing_player:
            value = float('-inf')
            for move in state.get_moves():
                child = deepcopy(state)
                # child._current_player = self.first_player_name
                # child._other_player = self.second_player_name
                child = child.make_move(move)
                score = child.get_scores()['max']
                next_player = self.next_player(score, self.prev_score_max, maximizing_player)
                self.prev_score_max = score
                new_value = self.alphabeta(child, depth-1, alpha, beta, next_player, i)
                value = max(value, new_value)
                alpha = max(alpha, value)
                if len(child.get_moves()) == self.starting_len - 1:
                    move_index = self.all_moves.index(move)
                    if new_value not in self.values:
                        self.values[new_value] = [move_index]
                    else:
                        self.values[new_value].append(move_index)
                if value >= beta:
                    # TODO: TO PSUJE
                    self.after_break = True
                    break
            return value
        else:
            value = float('inf')
            for move in state.get_moves():
                child = deepcopy(state)
                # child._current_player = self.second_player_name
                # child._other_player = self.first_player_name
                child = child.make_move(move)
                score = child.get_scores()['min']
                next_player = self.next_player(score, self.prev_score_min, maximizing_player)
                self.prev_score_min = score
                new_value = self.alphabeta(child, depth-1, alpha, beta, next_player, i)
                value = min(value, new_value)
                beta = min(beta, value)
                if len(child.get_moves()) == self.starting_len - 1:
                    move_index = self.all_moves.index(move)
                    if new_value not in self.values:
                        self.values[new_value] = [move_index]
                    else:
                        self.values[new_value].append(move_index)
                if value <= alpha:
                    # TODO: TO PSUJE
                    break
            return value

    def next_player(self, score: int, prev_score: int, player: bool) -> bool:
        if player:
            return True if score - prev_score == 1 else False
        else:
            return False if score - prev_score == 1 else True