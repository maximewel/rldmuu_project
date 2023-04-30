import numpy as np
from typing import Tuple

from tiles.tile_abstract import AbstractTile
from tiles.tiletype import TileType

import tkinter as tk
from PIL import Image, ImageTk

from abc import ABC, abstractmethod

class LunarRenderer(ABC):
    
    @abstractmethod
    def render(self, grid: np.ndarray[AbstractTile], player_position: Tuple[int, int]) -> None:
        pass

class LunarTextRenderer(LunarRenderer):

    tile_to_ascii = {
        TileType.STANDARD.value: "→",
        TileType.FRAIL.value: "↝",
        TileType.RANDOM.value: "↬",
        TileType.FAST.value: "↠",

        TileType.END.value: "⛢",

        TileType.MINERAL.value: "★",
    }

    def render(self, grid: np.ndarray[AbstractTile], player_position: Tuple[int, int]) -> None:
        player_x, player_y = player_position
        w,h = grid.shape
        for y in range(h):
            tiles = grid[:, y]
            line = [f"|{self.tile_to_ascii[tile.tileType.value]}|" for tile in tiles]
            if y == player_y:
                line[player_x] = "|X|"
            print(''.join(line))
        print()

class Lunar2DRenderer(LunarRenderer):
    frame: tk.Frame
    labels: np.ndarray[tk.Label] = None
    window: tk.Tk

    def __init__(self) -> None:
        super().__init__()
        self.init_frame()
        
    def init_frame(self):
        self.window = tk.Tk()
        self.window.title("Lunar explorer")

        self.frame = tk.Frame(self.window)
        self.frame.pack(expand=True)
    
    def create_label(self, grid: np.ndarray) -> None:
        self.labels = np.zeros_like(grid)
        w, h = grid.shape

        for x in range(w):
            for y in range(h):
                self.labels[x,y] = tk.Label(self.frame, text=f"{x, y}")
                self.labels[x,y].grid(row=x, column=y)

    def render(self, grid: np.ndarray[AbstractTile], player_position: Tuple[int, int]) -> None:
        if self.labels is None:
            self.labels = self.create_label(grid)
        
        self.window.update()