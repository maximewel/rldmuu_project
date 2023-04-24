from .algo import Rlalgorithm
import numpy as np
import random

from gymnasium.spaces import Discrete

class RlModel():
    N_sas: np.ndarray
    N_sa: np.ndarray
    P_table: np.ndarray
    reward_table: np.ndarray

    n_states: int
    n_actions: int

    def __init__(self, n_states, n_actions) -> None:
        self.n_states = n_states
        self.n_actions = n_actions

        #Init reward table at 1, meaning we will be optimistic and try to explore a bit more
        self.reward_table = np.zeros((n_states, n_actions))
        self.N_sa = np.zeros((n_states, n_actions))

        self.N_sas = np.zeros((n_states, n_actions, n_states))
        self.P_table = np.ones((n_states, n_actions, n_states)) #Init with value 1 / |s| to avoid undefined state values
        self.P_table = self.P_table / n_states
    
    def update_model(self, state, action, next_state, reward):
        #Inc conters on state and next state (sa, sas)
        self.N_sa[state, action] += 1
        self.N_sas[state, action, next_state] += 1

        #Update P_table
        self.P_table[state, action, :] = (self.N_sas[state, action, :] + 1) / (self.N_sa[state, action] + self.n_states)

        #Log reward in reward table
        self.reward_table[state, action] += (reward - self.reward_table[state, action]) / self.N_sa[state, action]

class GreedyQIteration(Rlalgorithm):
    epsilon: float
    model: RlModel
    V: np.ndarray

    gamma: float
    alpha: float
    epsilon_decay: float

    state: any = None

    n_states: int
    n_actions: int
    update_per_iteration: int

    def __init__(self, epsilon: float, alpha: float, gamma: float, epsilon_decay: float, update_per_iteration: int = 10) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.update_per_iteration = update_per_iteration
    
    def set_env(self, state_space: Discrete, actions_spaces: Discrete, env_bounds):
        super().set_env(state_space, actions_spaces, env_bounds)

        self.V = np.zeros(state_space.n)
        self.model = RlModel(state_space.n, actions_spaces.n)

        self.n_states = state_space.n
        self.n_actions = actions_spaces.n
        
    # boiler plate
    def next_action(self):
        if self.state is None:
            raise Exception("Action called before setting state")
        
        if (random.uniform(0, 1) < self.epsilon):
            return self.actions_spaces.sample()
        return np.argmax(self.Q[self.state])

    def update(self, action: int, observation: any, reward: int | float):
        next_state = self.discrete_observation(observation)
        self.model.update_model(self.state, action, next_state, reward)
        ## One or the other: Value iteration or Qfunction
        #self.update_q_function_value_iter()
        self.update_q_function_qlearning(action)

        self.state = next_state

        self.epsilon *= self.epsilon_decay
        return self.epsilon, reward
    
    def update_q_function_qlearning(self, action):
        #Update current vlaue
        self.update_q_value(self.state, action)

        #Sample multiple values
        for _ in range(self.update_per_iteration):
            random_state = random.randrange(0, self.n_states)
            random_action = random.randrange(0, self.n_actions)
            self.update_q_value(random_state, random_action)
        
    def update_q_value(self, state, action):
        """Calculate the Q-value for a given state and acton"""
        old_Q = self.Q[state, action]
        rho_t = self.model.reward_table[state, action]

        #Iterate over possible next states values given by the P function
        sum_value = 0
        #print(np.sum(self.model.P_table[state, action, :]))
        for model_possible_next_state in range(self.n_states):
            sum_value += self.model.P_table[state, action, model_possible_next_state] * np.max(self.Q[model_possible_next_state, :])
        self.Q[state, action] = old_Q + self.alpha * (rho_t + (self.gamma*sum_value) - old_Q)