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
        ### Every match/case on enum worked untill now. However, working with the enum yielded incoherenet behaviors.
        ### This might of course be due to an understanding from me of the match/case pattern. But this is very easily resolved
        ### Using the value directly, as we know they are unique ints.
        match tiletype.value:
            case TileType.END.value:
                tile = EndTile()
            case TileType.FRAIL.value:
                tile = FrailTile()
            case TileType.FAST.value:
                tile = FastTile()
            case TileType.RANDOM.value:
                tile = RandomTile()
            case TileType.STANDARD.value:
                tile = StandardTile()
            case TileType.MINERAL.value:
                value = np.random.randint(MineralTile.MIN_VALUE, MineralTile.MAX_VALUE+1)
                tile = MineralTile(value)
            case _:
                raise Exception(f"Can not create tile from tiletype {tiletype}")
        return tile