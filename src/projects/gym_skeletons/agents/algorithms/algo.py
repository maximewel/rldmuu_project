from abc import ABC
from gymnasium.spaces import Box, Discrete
import numpy as np

from enums.environments import Bounds

class Rlalgorithm(ABC):
    #Current state
    state: any

    state_space: Box
    actions_spaces: Discrete
    amplitudes: list[int | float]   #One amplitude value per metric
    steps: list[float]              #One step value per metric. 
    starts: list[float]             #Start of state spaces
    discretization_per_metric: list #Number of discrete quantization per metric

    #Q-table
    Q: np.ndarray

    def set_env(self, state_space: Box | Discrete, actions_spaces: Discrete, env_bounds: Bounds):
        self.state_space = state_space 
        self.actions_spaces = actions_spaces

        #If there are no env bounds, then we have a fully discrete problem. Q is simply a [state, action] array
        if env_bounds is None:
            self.Q = np.zeros((state_space.n, actions_spaces.n), dtype=np.float64)

        #Calculate Q from the bounds and their discretization number
        else:
            bounds_list = env_bounds.value

            self.Q = np.zeros([discretization_number for (min, max , discretization_number) in bounds_list] + [actions_spaces.n])

            self.starts = [min for (min, max, discretization_number) in bounds_list]
            self.discretization_per_metric = [discretization_number for (min, max , discretization_number) in bounds_list]

            self.steps = []
            self.amplitudes = []
            for (min, max , discretization_number) in bounds_list:
                amplitude = max - min
                self.amplitudes.append(amplitude)
                self.steps.append(amplitude/discretization_number)
            
            print(f"Initialized algorithm with {len(bounds_list)} metrics\nAmplitudes: {self.amplitudes} \nSteps: {self.steps}\n")
        
        print(f"Q shape:{self.Q.shape}")

    def discrete_observation(self, observation: np.ndarray) -> np.ndarray:
        #Discretisation of the observation is made in the context of bounded metrics.
        #The value of each metric is localized in its amplitude (max - min). Then it is placed in the floor
        #nearest discrete 'bucket'. This allow us to quickly retrieve the discrete value without having to perform distances.
        if isinstance(self.state_space, Discrete) or observation is None:
            return observation

        discrete_array = []
        for metric_index in range(len(observation)):
            metric_value = observation[metric_index] - self.starts[metric_index]

            metric_steps_float = metric_value / self.steps[metric_index]

            #If the value is at the very maximum, chances are we are trying to put it ont he "next bucket". Treat this edge case.
            discrete_index = int(np.floor(metric_steps_float))

            discretization_quantum = self.discretization_per_metric[metric_index]
            if discrete_index >= discretization_quantum:
                discrete_index = discretization_quantum - 1

            discrete_array.append(discrete_index)

        return tuple(discrete_array)

    def next_action(self):
        """Apply policy to determine next action from Q-table"""
        pass

    def update(self, action, state, reward):
        """Update the state of the RL algo with the observed parameters"""
        pass

    def set_state(self, observation: any):
        self.state = self.discrete_observation(observation)