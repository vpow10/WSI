import sys
sys.path.insert(1, '/home/jansz/Documents/WSI/two-player-games/two_player_games/games')
sys.path.insert(1, '/home/jansz/Documents/WSI/two-player-games')
from dots_and_boxes import DotsAndBoxes, DotsAndBoxesMove, DotsAndBoxesState
import random
from minimax import Minimax

minimax = Minimax(3)
game = DotsAndBoxes(2)
i = 1
while not game.is_finished():
    moves = game.get_moves()
    for x in moves:
        print(x.connection, x.loc)
    move = minimax.alphabeta(game.get_moves, 3, float('-inf'), float('inf'), True, game.state.get_scores())
    game.make_move(move)
    print("Move number: ", i)
    i += 1

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)