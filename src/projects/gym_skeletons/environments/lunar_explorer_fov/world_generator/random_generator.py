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
        tiletypes_ratios = [10, 2, 2, 1, 3]
        tiletypes = [TileType.STANDARD, TileType.FAST, TileType.FRAIL, TileType.RANDOM, TileType.MINERAL]

        #Compute probabilities from ratio
        p = [tile_ratio / sum(tiletypes_ratios) for tile_ratio in tiletypes_ratios]

        grid = np.random.choice(a=tiletypes, p=p, size=(self.world_size, self.world_size))

        end_x, end_y = np.random.choice(range(1, grid.shape[0])), np.random.choice(range(1, grid.shape[1]))

        grid[end_x, end_y] = TileType.END

        #Force player position at 0,0 to be standard tile
        grid[0,0] = TileType.STANDARD

        return np.vectorize(self.tiletype_to_tile)(grid)