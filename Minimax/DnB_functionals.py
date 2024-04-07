from typing import Dict, Callable, List
import sys
sys.path.insert(1, '/home/jasiek/WSI/two-player-games/two_player_games/games')
sys.path.insert(2, '/home/jasiek/WSI/two-player-games')
from dots_and_boxes import DotsAndBoxes
from random import choice
from minimax import Minimax
from copy import deepcopy


class DnB_functionals:
    def __init__(self, size: int, depth: int) -> None:
        self.game = DotsAndBoxes(size, first_player='max', second_player='min')
        self.simulation = DotsAndBoxes(size, first_player='max', second_player='min')
        self.all_moves = DotsAndBoxes(size).get_moves()
        self.state = self.game.state
        self.depth = depth

    # def check_scores(self, scores: Dict, prev_scores: Dict, max_player: bool) -> bool:
    #     """
    #     Checks whether any player made a box
    #     """
    #     if max_player:
    #         return True if scores['max'] - prev_scores['max'] == 1 else False
    #     else:
    #         return False if scores['min'] - prev_scores['min'] == 1 else True

    def undo_simulation(self) -> None:
        self.simulation.state = self.state

    def heuristic(self, node) -> int:
        for move in self.all_moves:
            if move not in node:
                self.simulation.make_move(move)
        scores = self.simulation.state.get_scores()
        self.undo_simulation()
        return scores['max'] - scores['min']

    def next_player(self, heuristic: int, prev_heuristic: int, player: bool) -> bool:
        if player:
            return True if heuristic - prev_heuristic == 1 else False
        else:
            return False if prev_heuristic - heuristic == -1 else True

    def simulate(self, depth, alpha, beta, max_player):
        minimax = Minimax(self)
        print('--------------')
        print(self.game.state)
        print('Game starts! First player: Max')
        while not self.game.is_finished():
            moves = self.game.get_moves()
            best_value = minimax.alphabeta(moves, self.heuristic, depth, alpha, beta, max_player)
            print('Value:', best_value)
            possible_moves = [index for index in minimax.values.keys() if minimax.values[index] == best_value]
            move = self.all_moves[choice(possible_moves)]
            self.game.make_move(move)
            print(self.game.state)
        print('--------------')
        winner = self.game.get_winner()
        if winner is None:
            print('Draw!')
        else:
            print('Winner: Player ' + winner)