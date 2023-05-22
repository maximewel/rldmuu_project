import numpy as np
from typing import Tuple

from tiles.tile_abstract import AbstractTile
from tiles.tiletype import TileType

import tkinter as tk
from PIL import Image, ImageTk

from enum import Enum

import numpy as np

import os

from abc import ABC, abstractmethod

from tkinter import NW

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


class ImageNames(Enum):
    ROVER = "./sprites/rover.png"
    END = "./sprites/end.png"

    STANDARD = "./sprites/terrain/standard.png"
    FAST = "./sprites/terrain/fast.png"
    RANDOM = "./sprites/terrain/crater.png"
    FRAIL = "./sprites/terrain/frail.png"

    MINERAL_FOLDER = "./sprites/minerals/PNG"

class Lunar2DRenderer(LunarRenderer):
    window: tk.Tk
    canvas: tk.Canvas
    frame: tk.Frame

    labels: np.ndarray[tk.Label] = None

    LABEL_SIZE = (64, 64)

    rover_image: Image
    rover_canevas_image: any

    images = {}

    def get_abs_path(self, rel_path: str) -> str:
        return os.path.join(os.path.dirname(__file__), rel_path)

    def load_image(self, image_path: str):
        image_abs_path = self.get_abs_path(image_path)

        img= Image.open(image_abs_path)

        resized_image= img.resize(self.LABEL_SIZE)

        return ImageTk.PhotoImage(resized_image, master=self.window)

    def load_images(self):
        print(f"Rendered loading all images...")
        #Single player sprite
        self.rover_image = self.load_image(ImageNames.ROVER.value)

        #All single terrain sprites
        for (tiletype, image_name) in [(TileType.END, ImageNames.END.value), (TileType.STANDARD, ImageNames.STANDARD.value), 
                                       (TileType.FAST, ImageNames.FAST.value), (TileType.FRAIL, ImageNames.FRAIL.value), (TileType.RANDOM, ImageNames.RANDOM.value)]:
            self.images[tiletype] = self.load_image(image_name)

        #Mineral sprites, taken at random later
        self.images[TileType.MINERAL] = []
        for filename in os.scandir(self.get_abs_path(ImageNames.MINERAL_FOLDER.value)):
            self.images[TileType.MINERAL].append(self.load_image(filename))

        print("Loading finished")

    def __init__(self) -> None:
        super().__init__()

        self.init_frame()
        self.load_images()
        
    def init_frame(self):
        self.window = tk.Tk()
        self.window.title("Lunar explorer")
        self.window.withdraw()

        self.canvas = tk.Canvas(self.window)
        self.canvas.pack()
    
    def get_image(self, tile: AbstractTile, has_player: bool = False) -> Image:
        if has_player:
            return self.rover_image
        
        if tile.tileType == TileType.MINERAL:
            mineral_len = len(self.images[TileType.MINERAL])
            return self.images[TileType.MINERAL][tile.value % mineral_len]

        return self.images[tile.tileType]
    
    def update_labels(self, grid: np.ndarray[TileType], player_position: Tuple[int, int]) -> None:
        w, h = grid.shape

        self.canvas.delete("all")

        for x in range(w):
            for y in range(h):
                pix_x, pix_y = self.pos_to_pix((x, y))
                tile: AbstractTile = grid[x, y]
                tile_image = self.get_image(tile, has_player=False)
                self.canvas.create_image(pix_x, pix_y, image=tile_image, anchor=NW)

        pix_x, pix_y = self.pos_to_pix(player_position)
        self.canvas.create_image(pix_x, pix_y, image=self.rover_image, anchor=tk.CENTER)

    def create_labels(self, grid: np.ndarray[TileType]) -> None:
        self.labels = np.zeros_like(grid)
        w, h = grid.shape
        cell_w, cell_h = self.LABEL_SIZE

        self.canvas.config(width=w*cell_w, height=h*cell_h)
                
        self.window.deiconify()
        self.window.lift()
    
    def pos_to_pix(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Compute the player's pixel position from its x,y position, unit being a single cell."""
        x,y = position
        w_cell, h_cell = self.LABEL_SIZE

        return int(np.round(x*w_cell)), int(np.round(y*h_cell))

    def render(self, grid: np.ndarray[AbstractTile], player_position: Tuple[float, float]) -> None:
        if self.labels is None:
            self.create_labels(grid)
        
        self.update_labels(grid, player_position)
        
        self.window.update()