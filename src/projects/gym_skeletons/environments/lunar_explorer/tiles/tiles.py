from typing import Tuple
import numpy as np

from tiles.tile_abstract import AbstractTile
from tiles.tiletype import TileType
from actions import Actions

class EndTile(AbstractTile):
    tileType= TileType.END

    END_VALUE = 200

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        """Regardless of the action, reaching this tile means the end."""
        return 0, 0, True, self.END_VALUE

class FrailTile(AbstractTile):
    tileType= TileType.FRAIL

    FALL_CHANCE = 0.20

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        x, y, done = None, None, None
        has_moved: bool = True

        match action:
            case Actions.NOTHING | Actions.DRILL:
                x, y, done = 0, 0, False
                has_moved = False
            case Actions.LEFT:
                x, y = -1, 0
            case Actions.RIGHT:
                x, y = 1, 0
            case Actions.UP:
                x, y = 0, -1
            case Actions.DOWN:
                x, y = 0, 1

        #If player is speeding, there is a chance to die on the frail tail
        if has_moved and np.abs(sum(speed)) > 0 and np.random.rand() < self.FALL_CHANCE:
            x, y, done = 0, 0, True

        return x, y, done, 0

class FastTile(AbstractTile):
    tileType= TileType.FAST

    SPEED_BOOST_CHANCE = 0.5

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        abs_speed = np.abs(sum(speed))
        x,y = 0,0
        has_moved: bool = True
        match action:
            case Actions.LEFT:
                x, y = -1, 0
            case Actions.RIGHT:
                x, y = 1, 0
            case Actions.UP:
                x, y = 0, -1
            case Actions.DOWN:
                x, y = 0, 1
            case _:
                has_moved = False

        if has_moved and abs_speed > 0 and np.random.rand() < self.SPEED_BOOST_CHANCE:
            x, y = 2*x, 2*y
        
        return x, y, False, 0

class RandomTile(AbstractTile):
    tileType= TileType.RANDOM
    moves: list = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        match action:
            case Actions.LEFT | Actions.RIGHT | Actions.UP | Actions.DOWN:
                rand_id = np.random.choice(len(self.moves))
                x, y = self.moves[rand_id]       
            case _:
                x, y = 0, 0

        return x, y, False, 0

class StandardTile(AbstractTile):
    tileType= TileType.STANDARD

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        match action:
            case Actions.LEFT:
                x, y = -1, 0
            case Actions.RIGHT:
                x, y = 1, 0
            case Actions.UP:
                x, y = 0, -1
            case Actions.DOWN:
                x, y = 0, 1

        return x, y, False, 0
    
class MineralTile(AbstractTile):
    tileType= TileType.MINERAL

    value: float

    MIN_VALUE = 10
    MAX_VALUE = 25

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        reward = None

        if action == Actions.DRILL:
            #Upon drilling, this mineral value is empty
            reward = self.value
            self.value = 0
        else:
            reward = 0
        
        return 0, 0, False, reward
