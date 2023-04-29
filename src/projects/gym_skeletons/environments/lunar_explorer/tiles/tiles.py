from typing import Tuple

from tiles.tile_abstract import AbstractTile
from tiles.tiletype import TileType

from actions import Actions

class EndTile(AbstractTile):
    tileType= TileType.END

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: int) -> Tuple[int, int, bool, float]:
        return super().execute(action)

class FrailTile(AbstractTile):
    tileType= TileType.FRAIL

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: int) -> Tuple[int, int, bool, float]:
        return super().execute(action)


class FastTile(AbstractTile):
    tileType= TileType.FAST

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: int) -> Tuple[int, int, bool, float]:
        return super().execute(action)


class RandomTile(AbstractTile):
    tileType= TileType.RANDOM

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: int) -> Tuple[int, int, bool, float]:
        return super().execute(action)


class StandardTile(AbstractTile):
    tileType= TileType.STANDARD

    def __init__(self) -> None:
        super().__init__()

    def execute(self, action: Actions) -> Tuple[int, int, bool, float]:
        match action:
            case Actions.LEFT:
                pass
            case Actions.RIGHT:
                pass
            case Actions.UP:
                pass
            case Actions.DOWN:
                pass

class MineralTile(AbstractTile):
    tileType= TileType.MINERAL

    value: float

    MIN_VALUE = 10
    MAX_VALUE = 25

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def execute(self, action: Actions) -> Tuple[int, int, bool, float]:
        reward = None

        if action == Actions.DRILL:
            #Upon drilling, this mineral value is empty
            reward = self.value
            self.value = 0
        else:
            reward = 0
        
        return 0, 0, False, reward
