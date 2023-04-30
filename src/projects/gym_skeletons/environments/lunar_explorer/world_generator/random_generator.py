from abc import ABC, abstractmethod

from tiles.tiletype import TileType
from world_generator.abstract_generator import AbstractGenerator

import numpy as np

class RandomGenerator(AbstractGenerator):
    world_size: int

    def __init__(self, size: int) -> None:
        super().__init__()

        self.world_size = size

    def generate(self, seed: int):
        """Random a uniformely distributed choice of the possible tiles"""
        #TODO: Change probability distribution at will. Sparse world ? Rich world ? Lot of terrains ? Etc...
        # p = ...

        #END tile is not part of the random generation
        grid = np.random.choice(a=[t for t in TileType if t is not TileType.END], size=(self.world_size, self.world_size))

        end_x, end_y = np.random.choice(range(1, grid.shape[0]+1)), np.random.choice(range(1, grid.shape[1]+1))

        print(f"End coordinates at {(end_x, end_y)}")
        grid[end_x, end_y] = TileType.END

        #Force player position at 0,0 to be standard tile
        grid[0,0] = TileType.STANDARD

        return np.vectorize(self.tiletype_to_tile)(grid)