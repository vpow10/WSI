import sys
sys.path.insert(1, '/home/jasiek/WSI_files/two-player-games/two_player_games/games')
sys.path.insert(2, '/home/jasiek/WSI_files/two-player-games')
from dots_and_boxes import DotsAndBoxes
import random
from minimax import Minimax
from copy import deepcopy
from DnB_functionals import check_scores


size = 3
depth = 3
minimax = Minimax()
game = DotsAndBoxes(size, first_player='max', second_player='min')
last_value = 0
last_scores = 0
i = 0
print('--------------')
print('Game starts! First player: Max')
while not game.is_finished():
    simulation = deepcopy(game)
    scores = game.state.get_scores()
    if i == 0:
        max_player = True
    else:
        max_player = check_scores(scores, last_scores, max_player)
    last_scores = scores
    best_value = minimax.alphabeta(simulation.get_moves, simulation, game.state, simulation.make_move, depth, float('-inf'), float('inf'), max_player, simulation.state.get_scores)
    print('Value:', best_value)
    move = minimax.choose_move(best_value)
    player = 'Max' if max_player else 'Min'
    print(f'Move by {player}:', move.connection, move.loc)
    game.make_move(move)
    print(game.state)
    i += 1

print('--------------')
winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner)