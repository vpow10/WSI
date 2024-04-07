from typing import Dict


def heuristic_DnB(size: int, move: Dict):
    if size <= 1:
        return "Wrong board!"
    dots = [[0 for _ in range(size)] for _ in range(size)]

    # corners

    dots[0][0], dots[0][size-1], dots[size-1][0], dots[size-1][size-1] = 2, 2, 2, 2

    # edges

    for i in range(size-2):
        dots[0][i+1], dots[i+1][0], dots[size-1][i+1], dots[i+1][size-1] = 3, 3, 3, 3

    # inside

    for i in range(size-2):
        for n in range(size-2):
            dots [i+1][n+1] = 4


# for row in heuristic_DnB(6):
#     print(row)