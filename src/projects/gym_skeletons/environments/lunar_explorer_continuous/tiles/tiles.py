from typing import Tuple
import numpy as np

from tiles.tile_abstract import AbstractTile
from tiles.tiletype import TileType
from actions import Actions

class EndTile(AbstractTile):
    tileType= TileType.END

    END_VALUE = 100

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        """Regardless of the action, reaching this tile means the end."""
        return 0, 0, True, self.END_VALUE

class FrailTile(AbstractTile):
    tileType= TileType.FRAIL

    FALL_CHANCE = 1.0
    FALL_SPEED_THRESHOLD = 0.75
    FALL_REWARD = -100

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        x, y, done = None, None, None
        has_moved: bool = True
        reward = 0

        #Frail tiles half the movement speed of the player while it is moving on them
        halved_x_speed, halved_y_speed = speed[0] / 2.0, speed[1] / 2.0

        match action:
            case Actions.NOTHING | Actions.DRILL:
                x, y, done = 0, 0, False
                has_moved = False
            case _:
                x, y = halved_x_speed, halved_y_speed

        #If player is speeding, there is a chance to die on the frail tail
        if has_moved and np.abs(sum(speed)) > self.FALL_SPEED_THRESHOLD and np.random.rand() < self.FALL_CHANCE:
            x, y, done, reward = 0, 0, True, self.FALL_REWARD

        return x, y, done, reward

class FastTile(AbstractTile):
    tileType= TileType.FAST

    def __init__(self) -> None:
        super().__init__()
    
    def bound_value(self, value: float, high: any, low: float) -> float:
        return min(high, max(low, value))

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        speed_x, speed_y = speed

        double_speed_x, double_speed_y = self.bound_value(2*speed_x, 1, -1), self.bound_value(2*speed_y, 1, -1)
        
        return double_speed_x, double_speed_y, False, 0

class RandomTile(AbstractTile):
    tileType= TileType.RANDOM
    moves: list = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        speed_amplitude = sum(speed)
        match action:
            case Actions.LEFT | Actions.RIGHT | Actions.UP | Actions.DOWN:
                moves = [(speed_amplitude, 0), (-speed_amplitude, 0), (0, speed_amplitude), (0, -speed_amplitude)]
                x, y = moves[np.random.choice(len(moves))]
            case _:
                x, y = 0, 0

        return x, y, False, 0

class StandardTile(AbstractTile):
    tileType= TileType.STANDARD

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        return speed[0], speed[1], False, 0
    
class MineralTile(StandardTile):
    tileType= TileType.MINERAL

    value: float

    MIN_VALUE = 20
    MAX_VALUE = 50

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def execute(self, action: Actions, speed: Tuple[int, int]) -> Tuple[int, int, bool, float]:
        #If players wants to drill, deplete this mineral's valaue
        if action == Actions.DRILL:
            #Upon drilling, this mineral value is empty
            reward = self.value
            self.value = 0
            return 0, 0, False, reward
        
        #Simply act as a standard tile for movement
        return super().execute(action, speed)

    def has_mineral(self) -> bool:
        return self.value != 0