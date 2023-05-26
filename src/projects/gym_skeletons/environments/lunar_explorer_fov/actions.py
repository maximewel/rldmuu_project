from enum import Enum

class Actions(Enum):
    """Discretize and name the action values"""
    NOTHING = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    DRILL = 5