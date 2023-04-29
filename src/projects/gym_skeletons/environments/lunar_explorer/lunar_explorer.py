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

    def __init__(self, seed: int, render, size: int, world_generator: AbstractGenerator = None, renderer: LunarRenderer = None):
        np.random.seed(seed)

        super().__init__(render)

        if not world_generator:
            world_generator = RandomGenerator(size)

        self.grid = world_generator.generate(seed)

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
        pass

    def render(self) -> None:
        self.renderer.render(self.grid, (self.player_x, self.player_y))

    def step(self, action) -> tuple[ObsType, float, bool]:
        return None, 0, False

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





