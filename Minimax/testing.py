from two_player_games.games.dots_and_boxes import DotsAndBoxes
import random
from minimax import Minimax
import numpy as np
from typing import Dict, Tuple
import gc
import csv


def random_game(size: int, depth: int, track: bool = False) -> str:
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
        except ValueError:      # depth equals 0 so moves are random
            possible_moves = [i for i in range(len(all_moves)) if all_moves[i] in game.get_moves()]
        move = random.choice(possible_moves)
        game.make_move(all_moves[move])
        score = game.state.get_scores()['max'] if current_player else game.state.get_scores()['min']
        i += 1
        next_player = minimax.next_player(score, prev_score_max if current_player else prev_score_min, current_player)
        if current_player:
            player = 'max'
            prev_score_max = score
        else:
            player = 'min'
            prev_score_min = score
        current_player = next_player
        if track:
            print('o-' * 10 + 'o')
            print('Move number: ' + str(i))
            print('Scores: ' + str(game.state.get_scores()))
            print('Current player: ' + player)
            print('Minimax values: ' + str(minimax.values))
            print('Move: ' + str(move))
            print('Current state: ' + str(game.state))
    winner = game.get_winner()
    print('Winner: ' + str(winner))
    return winner

def test_different_depths(size: int, depth1: int, depth2: int, track: bool = False) -> str:
    game = DotsAndBoxes(size, first_player='max', second_player='min')
    all_moves = game.get_moves()
    minimax1 = Minimax(size)
    minimax2 = Minimax(size)
    prev_score_max = 0
    prev_score_min = 0
    i = 0
    current_player = True    # start with minimax1 player
    while not game.is_finished():
        state = game.state
        if current_player:
            minimax = minimax1
            player = 'max'
            minimax1.alphabeta(state, depth1, float('-inf'), float('inf'), current_player)
            try:
                possible_moves1 = minimax1.values[max(minimax1.values.keys())]
            except ValueError:      # depth equals 0 so moves are random
                possible_moves1 = [i for i in range(len(all_moves)) if all_moves[i] in game.get_moves()]
            move = random.choice(possible_moves1)
            game.make_move(all_moves[move])
            score = game.state.get_scores()['max']
            next_player = minimax1.next_player(score, prev_score_max, current_player)
            prev_score_max = score
            current_player = next_player
        else:
            minimax = minimax2
            player = 'min'
            minimax2.alphabeta(state, depth2, float('-inf'), float('inf'), current_player)
            try:
                possible_moves2 = minimax2.values[min(minimax2.values.keys())]
            except ValueError:      # depth equals 0 so moves are random
                possible_moves2 = [i for i in range(len(all_moves)) if all_moves[i] in game.get_moves()]
            move = random.choice(possible_moves2)
            game.make_move(all_moves[move])
            score = game.state.get_scores()['min']
            next_player = minimax2.next_player(score, prev_score_min, current_player)
            prev_score_min = score
            current_player = next_player
        i += 1
        if track:
            print('o-' * 10 + 'o')
            print('Move number: ' + str(i))
            print('Scores: ' + str(game.state.get_scores()))
            print('Current player: ' + player)
            print('Minimax values: ' + str(minimax.values))
            print('Move: ' + str(move))
            print('Current state: ' + str(game.state))
    winner = game.get_winner()
    scores = game.state.get_scores()
    # print('Winner: ' + str(winner))
    return winner, scores

def test_depths(size: int) -> Dict[Tuple[int, int], Dict[str, int]]:
    data = {}
    for i in range(0, 6):
        for j in range(0, 6):
            max_score = []
            min_score = []
            results = {'max': 0, 'min': 0, 'draw': 0,
                        'max_point_avg': 0, 'min_point_avg': 0}
            print('Depth1: ' + str(i) + ' Depth2: ' + str(j))
            for n in range(50):
                print('Testing number ' + str(n) + ' out of 50')
                winner, scores = test_different_depths(size, i, j)
                if winner is None:
                    results['draw'] += 1
                else:
                    results[winner] += 1
                max_score.append(scores['max'])
                min_score.append(scores['min'])
            results['max_point_avg'] = np.mean(max_score)
            results['min_point_avg'] = np.mean(min_score)
            data[(i, j)] = results
    return data

def save_to_csv(data: Dict[Tuple[int, int], Dict[str, int]], filename: str) -> None:
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Depth1', 'Depth2', 'Max', 'Min', 'Draw',
                          'Max_point_avg', 'Min_point_avg'])
        for key, value in data.items():
            writer.writerow([key[0], key[1], value['max'], value['min'], value['draw'],
                            value['max_point_avg'], value['min_point_avg']])
    print('Data saved to ' + filename)

random_game(3, 3, track=True)
test_different_depths(3, 3, 3, track=True)
save_to_csv(test_depths(3), 'results3.csv')
save_to_csv(test_depths(4), 'results4.csv')