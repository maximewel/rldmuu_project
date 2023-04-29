from abc import ABC, abstractmethod

from numpy import ndarray
import numpy as np

from tiles.tiletype import TileType

from tiles.tiles import *

class AbstractGenerator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate(self, seed: int) -> ndarray:
        raise NotImplementedError("Generate method not implemented")
    
    @classmethod
    def tiletype_to_tile(cls, tiletype: TileType) -> AbstractTile:
        """Transform a tiletype to a tile"""
        tile = None
        match tiletype:
            case TileType.END:
                tile = EndTile()
            case TileType.FRAIL:
                tile = FrailTile()
            case TileType.FAST:
                tile = FastTile()
            case TileType.RANDOM:
                tile = RandomTile()
            case TileType.STANDARD:
                tile = StandardTile()
            case TileType.MINERAL:
                value = np.random.rand(MineralTile.MIN_VALUE, MineralTile.MAX_VALUE+1)
                tile = MineralTile(value)
        return tile