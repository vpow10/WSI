from typing import Dict


def check_scores(scores: Dict, prev_scores: Dict, max_player: bool) -> bool:
    """
    Checks whether any player made a box
    """
    if max_player:
        return True if scores['max'] - prev_scores['max'] == 1 else False
    else:
        return False if scores['min'] - prev_scores['min'] == 1 else True
