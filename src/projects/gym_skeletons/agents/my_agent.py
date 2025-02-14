from .algo import Rlalgorithm
import gymnasium
import numpy as np
import random

from .base_agent import BaseAgent

# IN PROGRESS
# QLearning with Eligibility traces
class QLearningEliTra(Rlalgorithm):
    def __init__(self, 
                 discount=0.99,
                 learning_rate=0.01,
                 lam = 0.7,
                 eps=1.,
                 eps_decay=0.9999,
                 eps_final=1e-2):
        #self.n_actions = 1
        #self.n_states = 2
        #self.Q = np.zeros([self.n_states, self.n_actions])
        self.eligibility = self.Q
        self.discount = discount # gamma
        self.learning_rate = learning_rate # alpha
        self.lam = lam
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_start = eps
        self.eps_final = eps_final

        self.prev_state = -1
        self.prev_action = -1

    def next_action(self):
        """Apply policy to determine next action from Q-table"""
        #Epsilon: 'epsilon' chance that action is random. Otherwise, take best possible (eps-greedy)
        if (random.uniform(0, 1) < self.epsilon  or np.argmax(self.Q[self.state]) == 0):
            return self.actions_spaces.sample()
        return np.argmax(self.Q[self.state])

    def update(self, action: int, observation: any, reward: int | float):
        """Update the state of the RL algo with the observed parameters"""
        #Get index out of obs
        next_state = self.discrete_observation(observation)

        td_error = self.reward + self.discount * np.argmax(self.Q[self.state]) - self.Q[self.prev_state, action]

        self.eligibility[self.prev_state, self.action] += 1

        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.Q[s, a] += self.alpha * self.eligibility[s, a] * td_error

            # You could do this with:
            # self.Q[:, :] = self.alpha * self.eligibility * td_error

        self.eligibility *= self.lam * self.discount # You forgot this
        
        #print(self.eligibility)
        self.prev_state = self.state
        #self.prev_action = action
        #self.prev_reward = reward
        self.state = next_state
        self.eps *= self.eps_decay

        #Update Qtable based on observed value
        # currentQ = self.Q[self.state][action]

        # next_best_Q = np.max(self.Q[next_state])

        # self.Q[self.state][action] = currentQ + self.alpha * (reward + self.gamma * next_best_Q - currentQ)
        
        # self.state = next_state

        # self.epsilon = self.epsilon * self.epsilon_decay

        # return (self.epsilon, reward)
