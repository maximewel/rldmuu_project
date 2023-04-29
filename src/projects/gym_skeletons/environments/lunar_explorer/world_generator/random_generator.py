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
        grid = np.random.choice(a=[t for t in TileType], size=(self.world_size, self.world_size))
        return np.vectorize(self.tiletype_to_tile)(grid)

