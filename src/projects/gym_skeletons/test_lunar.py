import numpy as np
import gymnasium
from time import sleep

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), "./environments/lunar_explorer/"))

from environments.lunar_explorer.world_generator.array_generator import ArrayGenerator
from environments.lunar_explorer.view.render import Lunar2DRenderer
from environments.lunar_explorer.tiles.tiletype import TileType
from environments.lunar_explorer.actions import Actions

# Run the environments/__init__.py to let python know about our custom environments
import environments

import time

def test_render():
    n_episodes = 3000

    env = gymnasium.make("Lunar-explorer", render=True, size=10, max_episode_steps=500, seed=43)

    env.reset()

    for i in range(5):
        env.step(2)
        time.sleep(1)
    for i in range(5):
        env.step(4)
        time.sleep(1)

def test_moves():
    """Test a simple loop through the grid"""
    world_generator = ArrayGenerator([
        [TileType.STANDARD, TileType.STANDARD],
        [TileType.STANDARD, TileType.STANDARD]
    ])

    env = gymnasium.make("Lunar-explorer", render=True, size=2, max_episode_steps=500, world_generator=world_generator)

    env.reset()

    env.step(Actions.RIGHT.value)
    env.step(Actions.DOWN.value)
    env.step(Actions.LEFT.value)
    env.step(Actions.UP.value)

def test_random():
    """Test going through a random tile multiple times"""
    world_generator = ArrayGenerator([
        [TileType.STANDARD, TileType.STANDARD, TileType.STANDARD],
        [TileType.STANDARD, TileType.RANDOM, TileType.STANDARD],
        [TileType.STANDARD, TileType.STANDARD, TileType.STANDARD]
    ])

    env = gymnasium.make("Lunar-explorer", render=True, size=2, max_episode_steps=500, world_generator=world_generator)

    for i in range(3):
        print(f"Test {i}")
        env.reset()

        env.step(Actions.RIGHT.value)
        env.step(Actions.DOWN.value)
        env.step(Actions.DOWN.value)
    
def test_mineral():
    """Test a simple loop through the grid"""
    world_generator = ArrayGenerator([
        [TileType.STANDARD, TileType.MINERAL, TileType.STANDARD],
        [TileType.STANDARD, TileType.MINERAL, TileType.STANDARD],
        [TileType.STANDARD, TileType.MINERAL, TileType.STANDARD]
    ])

    env = gymnasium.make("Lunar-explorer", render=True, size=2, max_episode_steps=500, world_generator=world_generator)

    for i in range(3):
        print(f"Test {i}")
        env.reset()

        obs, reward, terminated, truncated, info = env.step(Actions.RIGHT.value)

        obs, reward, terminated, truncated, info = env.step(Actions.DRILL.value)
        print(f"First drill: reward = {reward}")
        obs, reward, terminated, truncated, info = env.step(Actions.DRILL.value)
        print(f"Second drill: reward = {reward}")

        env.step(Actions.DOWN.value)
        obs, reward, terminated, truncated, info = env.step(Actions.DRILL.value)
        print(f"First drill: reward = {reward}")
        obs, reward, terminated, truncated, info = env.step(Actions.DRILL.value)
        print(f"Second drill: reward = {reward}")

        env.step(Actions.DOWN.value)
        obs, reward, terminated, truncated, info = env.step(Actions.DRILL.value)
        print(f"First drill: reward = {reward}")
        obs, reward, terminated, truncated, info = env.step(Actions.DRILL.value)
        print(f"Second drill: reward = {reward}")

def test_fast():
    world_generator = ArrayGenerator([
        [TileType.STANDARD, TileType.STANDARD, TileType.STANDARD, TileType.STANDARD],
        [TileType.STANDARD, TileType.FAST, TileType.STANDARD, TileType.STANDARD],
        [TileType.STANDARD, TileType.STANDARD, TileType.STANDARD, TileType.FAST],
        [TileType.STANDARD, TileType.FAST, TileType.STANDARD, TileType.STANDARD]
    ])

    env = gymnasium.make("Lunar-explorer", render=True, size=2, max_episode_steps=500, world_generator=world_generator)

    print(f"Test right fast")
    env.reset()

    env.step(Actions.DOWN.value)
    env.step(Actions.DOWN.value)
    env.step(Actions.DOWN.value)
    env.step(Actions.RIGHT.value)
    env.step(Actions.RIGHT.value)

    print(f"Test bottom fast")
    env.reset()

    env.step(Actions.RIGHT.value)
    env.step(Actions.DOWN.value)
    env.step(Actions.DOWN.value)

def test_2d():
    env = gymnasium.make("Lunar-explorer", render=True, size=9, max_episode_steps=500, renderer=Lunar2DRenderer(), seed=42)
    env.reset()

    import time
    for i in range(100):
        obs, reward, terminated, truncated, info = env.step(np.random.choice(Actions).value)
        print(reward)
        if terminated:
            env.reset()
        time.sleep(0.5)

if __name__ == '__main__':
    # test_render()

    # test_moves()

    # test_random()

    # test_mineral()

    #test_fast()

    test_2d()