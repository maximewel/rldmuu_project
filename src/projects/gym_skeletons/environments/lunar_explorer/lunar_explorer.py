from gymnasium.core import ObsType
from numpy import ndarray
from gymnasium import spaces

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from base_env import BaseEnv
from tiles.tile_abstract import AbstractTile
from world_generator.abstract_generator import AbstractGenerator
from world_generator.random_generator import RandomGenerator
from view.render import LunarTextRenderer, Lunar2DRenderer, LunarRenderer
from actions import Actions

import numpy as np

class LunarExplorer(BaseEnv):

    grid: ndarray[AbstractTile]

    renderer: LunarRenderer

    player_x: int
    player_y: int

    player_speed_x: int
    player_speed_y: int

    w: int
    h: int

    SPEED_INC = 1
    MAX_SPEED = 1

    MAX_DRILL = 3

    def __init__(self, seed: int, render, size: int, world_generator: AbstractGenerator = None, renderer: LunarRenderer = None):
        np.random.seed(seed)

        super().__init__(render)

        if not world_generator:
            world_generator = RandomGenerator(size)

        self.grid = world_generator.generate(seed)
        self.w, self.h = self.grid.shape

        self.renderer = renderer or LunarTextRenderer()

        # The observation contains (x, y, Vx, Vy)
        self.observation_space = spaces.Discrete(size * size * 6)
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(len(Actions))

        self.player_x = 0
        self.player_y = 0

        self.player_speed_x = 0
        self.player_speed_y = 0

        print(f"Lunar explorer started")

    def reset(self) -> ndarray:
        pass

    def get_observation(self) -> ndarray:
        pass

    def compute_reward(self, action) -> float:
        pass

    def update(self, action) -> bool:
        super().update()
        
    def render(self) -> None:
        self.renderer.render(self.grid, (self.player_x, self.player_y))

    def step(self, action) -> tuple[ObsType, float, bool]:
        """
        Act in the environment. Done in 3 steps:
        * Execute tile's behavior wrt the action
        * Compute new player position
        * Compute speed. Render final player position
        """
        action = Actions(action)
        print(f"Using step {action}")

        #Compute tile exec
        tile: AbstractTile = self.grid[self.player_x, self.player_y]
        print(f"Stepping on tile {tile.tileType.name}")
        offset_x, offset_y, done, reward = tile.execute(action, (self.player_speed_x, self.player_speed_y))

        print(f"New player offset: {offset_x, offset_y}")

        #Each action is penalized by 1 so that agents find the most optimized routes
        if not done:
            reward -= 1

        #Compute new position
        if offset_x != 0:
            self.player_speed_y = 0
            if offset_x > 0:
                self.player_x = min(self.player_x + offset_x, self.grid.shape[0])
                self.player_speed_x = min(self.player_speed_x + self.SPEED_INC, self.MAX_SPEED)
            else :
                self.player_x = max(self.player_x + offset_x, 0)
                self.player_speed_x = max(self.player_speed_x - self.SPEED_INC, -self.MAX_SPEED)
        
        if offset_y != 0:
            self.player_speed_x = 0
            if offset_y > 0:
                self.player_y = min(self.player_y + offset_y, self.grid.shape[1])
                self.player_speed_y = min(self.player_speed_y + self.SPEED_INC, self.MAX_SPEED)
            else :
                self.player_y = max(self.player_y + offset_y, 0)
                self.player_speed_y = max(self.player_speed_y - self.SPEED_INC, -self.MAX_SPEED)
        
        print(f"New player position: {self.player_x, self.player_y}")
        
        observation = (self.player_x, self.player_y, self.player_speed_x, self.player_speed_y)

        return observation, done, reward

class Tile:
    def __init__(self) -> None:
        pass

"""
to reuse our environment in experiments, we convert our environment to a gym adapted environment with:

>>>> GymEnvClassName = MyEnv.to_gym_env

and add the following to the __init__.py of the environments package:

register(
    id="MyEnv", # Name of our environment
    entry_point="environments.base_env:MyEnvGym", # Essentially: gym_skeletons.environments.file_name:GymEnvClassName
    max_episode_steps=300, # Forces the environment episodes to end once the agent played for max_episode_steps steps
)
"""

LunarExplorerEnvGym = LunarExplorer.to_gym_env





