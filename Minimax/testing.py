import sys
sys.path.insert(1, '/home/jasiek/WSI/two-player-games/two_player_games/games')
sys.path.insert(2, '/home/jasiek/WSI/two-player-games')
from dots_and_boxes import DotsAndBoxes
import random
from test import Minimax
from copy import deepcopy
from DnB_functionals import DnB_functionals


def random_game(size, depth):
    game = DotsAndBoxes(size, first_player='max', second_player='min')
    all_moves = game.get_moves()
    minimax = Minimax(size)
    i = 0
    prev_score_max = 0
    prev_score_min = 0
    current_player = True    # start with max player
    while not game.is_finished():
        state = game.state
        minimax.alphabeta(state, depth, float('-inf'), float('inf'), current_player)
        try:
            possible_moves = minimax.values[max(minimax.values.keys())] if current_player else minimax.values[min(minimax.values.keys())]
            print(minimax.values)
        except ValueError:      # depth equals 0 so moves are random
            possible_moves = [i for i in range(len(all_moves)) if all_moves[i] in game.get_moves()]
        move = random.choice(possible_moves)
        print(move)
        game.make_move(all_moves[move])
        print(game.state)
        score = game.state.get_scores()['max'] if current_player else game.state.get_scores()['min']
        i += 1
        next_player = minimax.next_player(score, prev_score_max if current_player else prev_score_min, current_player)
        if current_player:
            prev_score_max = score
        else:
            prev_score_min = score
        current_player = next_player
    winner = game.get_winner()
    if winner is None:
        print('Draw!')
    else:
        print('Winner: Player ' + winner)

def test_different_depths(size, depth1, depth2):
    game = DotsAndBoxes(size, first_player='max', second_player='min')
    all_moves = game.get_moves()
    minimax1 = Minimax(size)
    minimax2 = Minimax(size)
    prev_score_max = 0
    prev_score_min = 0
    current_player = True    # start with minimax1 player
    while not game.is_finished():
        state = game.state
        if current_player:
            minimax1.alphabeta(state, depth1, float('-inf'), float('inf'), current_player)
            try:
                possible_moves1 = minimax1.values[max(minimax1.values.keys())]
                print(minimax1.values)
            except ValueError:      # depth equals 0 so moves are random
                possible_moves1 = [i for i in range(len(all_moves)) if all_moves[i] in game.get_moves()]
            move = random.choice(possible_moves1)
            print(move)
            game.make_move(all_moves[move])
            print(game.state)
            score = game.state.get_scores()['max']
            next_player = minimax1.next_player(score, prev_score_max, current_player)
            prev_score_max = score
            current_player = next_player
        else:
            minimax2.alphabeta(state, depth2, float('-inf'), float('inf'), current_player)
            try:
                possible_moves2 = minimax2.values[min(minimax2.values.keys())]
                print(minimax2.values)
            except ValueError:      # depth equals 0 so moves are random
                possible_moves2 = [i for i in range(len(all_moves)) if all_moves[i] in game.get_moves()]
            print(possible_moves2)
            move = random.choice(possible_moves2)
            print(move)
            game.make_move(all_moves[move])
            print(game.state)
            score = game.state.get_scores()['min']
            next_player = minimax2.next_player(score, prev_score_min, current_player)
            prev_score_min = score
            current_player = next_player
    winner = game.get_winner()
    if winner is None:
        print('Draw!')
    else:
        print('Winner: Player ' + winner)

test_different_depths(3, 3, 1)
# random_game(3, 4)