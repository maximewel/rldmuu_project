from gymnasium.core import ObsType
from numpy import ndarray
from gymnasium import spaces

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from base_env import BaseEnv
from tiles.tile_abstract import AbstractTile
from tiles.tiletype import TileType
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

    drill_count: int
    has_drilled: bool

    SPEED_INC = 1
    MAX_SPEED = 1

    MAX_DRILL = 3
    EMPTY_DRILL_PENALTY = 20

    world_generator: AbstractGenerator
    seed: int

    verboe: bool

    def __init__(self, render, size: int, seed: int = None, world_generator: AbstractGenerator = None, renderer: LunarRenderer = None, verbose: bool = False):
        self.verbose = verbose
        self.seed = seed
        np.random.seed(seed)

        super().__init__(render)

        self.world_generator = world_generator or RandomGenerator(size) 

        self.renderer = renderer or Lunar2DRenderer()

        # The observation contains (x, y, Vx, Vy, has_mineral, has_drill)
        self.observation_space = spaces.Box(low=np.array([0, 0, -1, -1, 0, 0]), high=np.array([size-1, size-1, 1, 1, 1, 1]), shape=(6,), dtype=np.int32)
        # We have multiple actions, corresponding to the ones found in the enum
        self.action_space = spaces.Discrete(len(Actions))

        print(f"Lunar explorer started")

    def init_variables(self) -> None:
        """Initiate internal variables"""
        self.player_x = 0
        self.player_y = 0

        self.player_speed_x = 0
        self.player_speed_y = 0

        self.drill_count = self.MAX_DRILL

        self.grid = self.world_generator.generate(self.seed)

    def reset(self) -> ndarray:
        #Reset internal variables
        np.random.seed(self.seed)

        self.init_variables()

        np.random.seed()

        return self.get_observation()

    def get_observation(self) -> ndarray:
        tile: AbstractTile = self.grid[self.player_x, self.player_y]
        mineral_observation = 1 if tile.has_mineral() else 0
        drill_observation = 1 if self.drill_count >= 0 else 0
        return np.array([self.player_x, self.player_y, self.player_speed_x, self.player_speed_y, mineral_observation, drill_observation], dtype=np.int32)

    def compute_reward(self, action) -> float:
        pass

    def update(self, action) -> bool:
        super().update()
        
    def render(self) -> None:
        self.renderer.render(self.grid, (self.player_x, self.player_y))
    
    def in_bound(self, low, high, value) -> bool:
        return value >= low and value <= high

    def step(self, action) -> tuple[ObsType, float, bool]:
        """
        Act in the environment. Done in 3 steps:
        * Execute tile's behavior wrt the action
        * Compute new player position
        * Compute speed. Render final player position
        """
        action = Actions(action)
        if self.verbose:
            print(f"Using step {action}")

        has_drilled = False
        if action == Actions.DRILL:
            has_drilled = True
            if self.drill_count >= 0:
                self.drill_count -= 1
            else:
                action = Actions.NOTHING

        #Compute tile exec
        tile: AbstractTile = self.grid[self.player_x, self.player_y]
        offset_x, offset_y, done, reward = tile.execute(action, (self.player_speed_x, self.player_speed_y))
        if self.verbose:
            print(f"Stepping on tile {tile.tileType.name}")
            print(f"New player offset: {offset_x, offset_y}")
        
        #Punish 'empty drills'
        if has_drilled and reward <= 0:
            reward -= self.EMPTY_DRILL_PENALTY

        #Each action is penalized by 1 so that agents find the most optimized routes
        if not done:
            reward -= 1

        #Compute new position
        if offset_x == 0 and offset_y == 0:
            self.player_speed_x, self.player_speed_y = 0, 0
        
        if offset_x != 0:
            self.player_speed_y = 0
            new_x = self.player_x + offset_x
            
            #If going out of bound: Cancel move := don't update position + cancel speed
            if not self.in_bound(0, self.grid.shape[0]-1, new_x):
                self.player_speed_x = 0
            else:
                self.player_x = new_x
                if offset_x > 0:
                    self.player_speed_x = min(self.player_speed_x + self.SPEED_INC, self.MAX_SPEED)
                else :
                    self.player_speed_x = max(self.player_speed_x - self.SPEED_INC, -self.MAX_SPEED)
        
        if offset_y != 0:
            self.player_speed_x = 0

            new_y = self.player_y + offset_y

            if not self.in_bound(0, self.grid.shape[1]-1, new_y):
                self.player_speed_y = 0
            else:
                self.player_y = new_y
                if offset_y > 0:
                    self.player_speed_y = min(self.player_speed_y + self.SPEED_INC, self.MAX_SPEED)
                else :
                    self.player_speed_y = max(self.player_speed_y - self.SPEED_INC, -self.MAX_SPEED)
        
        if self.verbose:
            print(f"New player position: {self.player_x, self.player_y}, in env reward: {reward}")
        
        observation = self.get_observation()

        return observation, reward, done

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
