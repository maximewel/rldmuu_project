from gymnasium.spaces import Box, Discrete
from enums.environments import Bounds
from .algo import Rlalgorithm
import gymnasium
import numpy as np
import random

# IN PROGRESS
# QLearning with Eligibility traces
class QLearningEliTra(Rlalgorithm):
    def __init__(self, 
                 discount=0.99,
                 learning_rate=0.3,
                 lam = 0.0,
                 eps=1.,
                 eps_decay=0.9999,
                 eps_final=1e-2):
        super().__init__()
        #self.set_env()
        #self.n_actions = 1
        #self.n_states = 2
        #self.Q = np.zeros([self.n_states, self.n_actions])
        #print(self.Q)
        self.discount = discount # gamma
        self.learning_rate = learning_rate # alpha
        self.lam = lam
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_start = eps
        self.eps_final = eps_final


    # def set_env(self, state_space: Box | Discrete, actions_spaces: Discrete, env_bounds: Bounds):
    #     super().set_env(state_space, actions_spaces, env_bounds)
    #     # print("SET_ENV")
    #     # print(np.argmax(self.eligibility))

    def set_state(self, observation: any):
        super().set_state(observation)
        self.eligibility = np.zeros_like(self.Q)
        self.prev_state = self.discrete_observation(observation)

    def next_action(self):
        """Apply policy to determine next action from Q-table"""
        #Epsilon: 'epsilon' chance that action is random. Otherwise, take best possible (eps-greedy)
        if (random.uniform(0, 1) < self.eps):
            return self.actions_spaces.sample()
        return np.argmax(self.Q[self.state])

    def update(self, action: int, observation: any, reward: int | float):
        """Update the state of the RL algo with the observed parameters"""
        #Get index out of obs
        next_state = self.discrete_observation(observation)
        currentQ = self.Q[self.state][action]
        next_best_Q = np.max(self.Q[next_state])
        # print("next_state", next_state)
        # print("self.Q.shape", self.Q.shape)
        # print("reward", reward)
        # print("self.discount", self.discount)
        # print("self.Q[next_state]", self.Q[next_state])
        # print("np.max(self.Q[next_state])", np.max(self.Q[next_state]))
        # print("self.Q[self.state][action]", self.Q[self.state][action])
        td_error = reward + self.discount * next_best_Q - currentQ
        # print("self.eligibility[self.state][action]", self.eligibility[self.state][action])

        self.eligibility[self.state][action] += 1 # accumulating traces
        # self.eligibility[self.state][action] = (1 - self.learning_rate) * self.eligibility[self.state][action] + 1  # dutch traces
        # self.eligibility[self.state][action] = 1 # replacing traces

        # print("self.learning_rate", self.learning_rate)
        # #print("self.eligibility", self.eligibility)
        # print("np.max(self.eligibility)", np.max(self.eligibility))
        # print("np.min(self.eligibility)", np.min(self.eligibility))
        # print("self.eligibility.shape", self.eligibility.shape)
        # print("td_error", td_error)
        # print("before self.Q.shape", self.Q.shape)
        # print("(self.learning_rate * td_error * self.eligibility).shape", (self.learning_rate * td_error * self.eligibility).shape)
        # print("np.max((self.learning_rate * td_error * self.eligibility))", np.max((self.learning_rate * td_error * self.eligibility)))
        self.Q = self.Q + self.learning_rate * td_error * self.eligibility
        # print("after self.Q.shape", self.Q.shape)
        # print("self.lam", self.lam)
        # print("self.discount", self.discount)
        self.eligibility = self.discount * self.lam * self.eligibility # You forgot this

        self.state = next_state

        self.eps *= self.eps_decay

        return (self.eps, reward)
