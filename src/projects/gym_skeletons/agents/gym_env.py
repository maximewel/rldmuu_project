import gymnasium as gym
from algorithms.algo import Rlalgorithm

from enums.environments import Environments, Bounds
import numpy as np
from time import sleep

EP_SHOW = 100

def start(algorithm: Rlalgorithm, environment: Environments = Environments.LUNAR_LANDER, max_episodes: int | None = None, max_iteration: int | None = 1000, render: bool = True, 
          init: bool = True, seed: int | None = None):
    """Start env with algorithm"""

    try:
        env = gym.make(environment.value, render=render, size=10, seed=seed)
    except TypeError as e:
        env = gym.make(environment.value, render_mode="human" if render else None)

    if init:
        try:
            algorithm.set_env(env.observation_space, env.action_space, Bounds[environment.name])
        except Exception as e:
            algorithm.set_env(env.observation_space, env.action_space, None)

    observation, info = env.reset()
    algorithm.set_state(observation)

    total_rewards = []
    iterations = []
    epsilons = []

    try:
        episode = 1
        iteration = 0

        rewards = []
        while (max_episodes is None) or (episode <= max_episodes):
            iteration += 1
            action = algorithm.next_action()

            observation, reward, terminated, truncated, info = env.step(action)
            if render:
                print(action)
                print(observation)
                sleep(0.2)
            
            if terminated:
                observation = None
                
            epsilon, reward = algorithm.update(action, observation, reward)
            epsilons.append(epsilon)
            rewards.append(reward)

            if terminated or truncated or iteration > max_iteration: #Avoid getting stuck
                print(f"\rEpisode {episode}/{max_episodes}", end="")
                observation, info = env.reset(seed=seed)
                algorithm.set_state(observation)

                sum_reward = np.sum(rewards)
                if render:
                    print(f"Episode {episode} stopped at iteration {iteration} : {info}, Reward: {sum_reward}")

                total_rewards.append(sum_reward)
                iterations.append(iteration)
                

                if episode % EP_SHOW == 0:
                    print()
                    print(f"Average rewards over last {EP_SHOW} ep: {np.mean(total_rewards[-EP_SHOW:])}")
                
                episode += 1
                iteration = 0
                rewards = []

    except KeyboardInterrupt:
        print(f"Closing")
    finally:
        env.close()
    
    return iterations, total_rewards, epsilons