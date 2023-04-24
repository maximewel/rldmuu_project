import gymnasium as gym
from algorithms.algo import Rlalgorithm

from enums.environments import Environments, Bounds
import numpy as np

def start(algorithm: Rlalgorithm, environment: Environments = Environments.LUNAR_LANDER, max_episodes: int | None = None, show: bool = True, 
          show_episode=False, init: bool = True, seed: int | None = None):
    """Start env with algorithm"""

    env = gym.make(environment.value, render_mode="human" if show else None)

    if init:
        try:
            algorithm.set_env(env.observation_space, env.action_space, Bounds[environment.name])
        except Exception:
            algorithm.set_env(env.observation_space, env.action_space, None)

    observation, info = env.reset(seed=seed)
    algorithm.set_state(observation)

    metrics = []

    try:
        episode = 0

        iteration = 0
        epsilons = []
        rewards = []

        while (max_episodes is None) or (episode < max_episodes):
            iteration += 1
            action = algorithm.next_action()

            observation, reward, terminated, truncated, info = env.step(action)

            epsilon, reward = algorithm.update(action, observation, reward)
            epsilons.append(epsilon)
            rewards.append(reward)

            if terminated or truncated or iteration > 1000: #Avoid getting stuck
                observation, info = env.reset(seed=seed)
                algorithm.set_state(observation)

                sum_reward = np.sum(rewards)
                if show:
                    print(f"Episode {episode} stopped at iteration {iteration} : {info}")
                    print(f"Reward: {sum_reward}")

                metrics.append((iteration, sum_reward))
                episode += 1

                if show_episode:
                    print(f"\rEpisode {episode}/{max_episodes}", end="")

                iteration = 0
                rewards = []

    except KeyboardInterrupt:
        print(f"Closing")
    finally:
        env.close()
    
    return metrics, epsilons