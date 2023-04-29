import numpy as np
import gymnasium
from time import sleep

from agents import examples

# Run the environments/__init__.py to let python know about our custom environments
import environments

import time


if __name__ == '__main__':
    n_episodes = 3000
    render = False

    env = gymnasium.make("Lunar-explorer", render=True, size=10, max_episode_steps=500, seed=42)

    env.reset()

    for i in range(2):
        env.step(1)
        time.sleep(1)