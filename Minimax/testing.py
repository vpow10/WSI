import sys
sys.path.insert(1, '/home/jasiek/WSI_files/two-player-games/two_player_games/games')
sys.path.insert(2, '/home/jasiek/WSI_files/two-player-games')
from dots_and_boxes import DotsAndBoxes
import random
from minimax import Minimax
from copy import deepcopy
from DnB_functionals import DnB_functionals


size = 3
depth = 3
DnB_functional = DnB_functionals(size, depth)
DnB_functional.simulate(depth, float('inf'), float('inf'), True)