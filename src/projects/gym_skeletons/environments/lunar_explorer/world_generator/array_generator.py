from abc import ABC, abstractmethod

from numpy import ndarray
import numpy as np

from tiles.tiletype import TileType
from tiles.tiles import *

from world_generator.abstract_generator import AbstractGenerator

class ArrayGenerator(AbstractGenerator):
    tiletype_array: ndarray[TileType]

    def __init__(self, array: ndarray[TileType]) -> None:
        super().__init__()
        self.tiletype_array = array

    def generate(self, seed: int) -> ndarray:
        np.random.seed(seed)

        w, h = self.tiletype_array.shape

        tile_map = []

        for x in range(w):
            tile_map[x] = []
            for y in range(h):
                tiletype = self.tiletype_array[x, y]
                tile = self.tiletype_to_tile(tiletype)
                tile_map[x][y] = tile

        tile_array = np.array(tile_map)
        return tile_array