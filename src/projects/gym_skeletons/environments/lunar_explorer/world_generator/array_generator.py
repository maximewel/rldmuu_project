from abc import ABC, abstractmethod

from numpy import ndarray
import numpy as np

from tiles.tiletype import TileType
from tiles.tiles import *

from world_generator.abstract_generator import AbstractGenerator

class ArrayGenerator(AbstractGenerator):
    tiletype_array: ndarray[TileType]

    def __init__(self, array: ndarray[TileType] | list) -> None:
        super().__init__()
        if isinstance(array, list):
            array = np.array(array)
        
        #NP array are y,x, need to take transpose
        self.tiletype_array = array.T

    def generate(self, seed: int) -> ndarray:
        return np.vectorize(self.tiletype_to_tile)(self.tiletype_array)