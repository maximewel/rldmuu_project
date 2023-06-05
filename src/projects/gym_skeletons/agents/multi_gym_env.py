import gymnasium as gym
from algorithms.algo import Rlalgorithm

from enums.environments import Environments, Bounds
import numpy as np
from time import sleep
from gymnasium import Env

EP_SHOW = 100
SIMULTANEOUS_environment_store = 5

class MultiEnv():
    env_count: int
    env_configuration: dict
    environment: Environments

    #Each env has a current state stored alongside the environment
    environment_store: list[tuple[any, Env]]

    current_env_index: int

    def __init__(self, algorithm: Rlalgorithm, environment: Environments, env_count: int, env_configuration: dict) -> None:
        self.env_count = env_count
        self.env_configuration = env_configuration
        self.environment = environment

        self.current_env_index = 0

        #Init all environments
        self.environment_store = [None] * self.env_count
        [self.init_env(i) for i in range(self.env_count)] 

        #Init algorithm
        self.init_algorithm(algorithm)
    
    def init_env(self, index: int):
        """Start a new env, place it at the given location [index]"""
        seed = np.random.randint(0, 1e6)

        env = gym.make(**self.env_configuration, seed=seed)
        obs, info = env.reset()
        self.environment_store[index] = (obs, env)
    
    def init_algorithm(self, algorithm: Rlalgorithm):
        """Init the algorithm's knowledge of the environment by using the first created one"""
        obs, env = self.environment_store[0]

        try:
            algorithm.set_env(env.observation_space, env.action_space, Bounds[self.environment.name])
        except Exception:
            algorithm.set_env(env.observation_space, env.action_space, None)
    
    def change_env_get_state(self) -> tuple[any, int]:
        """Select a new environment as the current played one, randomly. 
        Return the associated current state and index"""
        self.current_env_index = np.random.choice(self.env_count)
        obs, env = self.environment_store[self.current_env_index]
        return obs, self.current_env_index
        
    def step(self, action):
        """Execute the action from the environment. Return the whole bundle. If the environment is finished, start a new one."""
        #Execute action
        obs, env = self.environment_store[self.current_env_index]
        observation, reward, terminated, truncated, info = env.step(action)

        #Update current state in store
        self.environment_store[self.current_env_index] = (observation, env)

        if terminated:
            #Setting next env to none (equivalent to say done=true for the DQN later down the line)
            observation = None
            self.init_env(self.current_env_index)

        return observation, reward, terminated, truncated, info

    def stop(self):
        """Stop all envorinments"""
        for obs, env in self.environment_store:
            env.close()

def train(algorithm: Rlalgorithm, environment: Environments = Environments.LUNAR_LANDER, max_episodes: int | None = None, max_iteration: int | None = 300,
          fov_on_player: bool = True):
    """Start env with algorithm"""

    if environment.value in [Environments.LUNAR_EXPLORER.value, Environments.LUNAR_EXPLORER_CONTINUOUS.value, Environments.LUNAR_EXPLORER_FOV.value]:
        env_config = {"id": environment.value, "render": False, "size": 10, "fov_on_player": fov_on_player}
    else:
        env_config = {"id":environment.value, "render_mode": None}
    
    multi_env_manager = MultiEnv(algorithm, environment, SIMULTANEOUS_environment_store, env_config)

    total_rewards = []
    end_rewards = []

    iterations = []
    epsilons = []

    try:
        episode = 1

        iteration = [0 for _ in range(SIMULTANEOUS_environment_store)]
        episode_rewards = [[] for _ in range(SIMULTANEOUS_environment_store)]

        while (max_episodes is None) or (episode <= max_episodes):
            #Get current played env, set it to the algorithm
            current_state, current_env_index = multi_env_manager.change_env_get_state()
            algorithm.set_state(current_state)

            iteration[current_env_index] += 1

            action = algorithm.next_action()

            observation, reward, terminated, truncated, info = multi_env_manager.step(action)

            epsilon, reward = algorithm.update(action, observation, reward)
            epsilons.append(epsilon)
            episode_rewards[current_env_index].append(reward)

            if terminated or truncated: #Avoid getting stuck
                print(f"\rEpisode {episode}/{max_episodes}", end="")

                end_rewards.append(reward)
                sum_reward = np.sum(episode_rewards[current_env_index])
                total_rewards.append(sum_reward)
                iterations.append(iteration[current_env_index])

                if episode % EP_SHOW == 0:
                    print()
                    print(f"Average rewards over last {EP_SHOW} episodes\n\
                          \tMean: {np.mean(total_rewards[-EP_SHOW:])}\n\
                          \tEnd: {np.mean(end_rewards[-EP_SHOW:])}")

                episode += 1
                iteration[current_env_index] = 0
                episode_rewards[current_env_index] = []

    except KeyboardInterrupt:
        print("Closing")
    finally:
        multi_env_manager.stop()
    
    return iterations, total_rewards, epsilons