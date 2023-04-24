from .algo import Rlalgorithm
import numpy as np
import random

class QLearning(Rlalgorithm):
    # E-greedy is often used for the policy of Q-learning
    epsilon: float

    # Parameters inherent to Q-function
    alpha: float            # Learning rate
    gamma: float            # Future Reward discount
    epsilon_decay: float    # Decay rate of eps, 0 < decay < 1

    def __init__(self, epsilon: float, alpha: float, gamma: float, epsilon_decay: float) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay

    def next_action(self):
        """Apply policy to determine next action from Q-table"""
        #Epsilon: 'epsilon' chance that action is random. Otherwise, take best possible (eps-greedy)
        if (random.uniform(0, 1) < self.epsilon):
            return self.actions_spaces.sample()
        return np.argmax(self.Q[self.state])

    def update(self, action: int, observation: any, reward: int | float):
        """Update the state of the RL algo with the observed parameters"""
        #Get index out of obs
        next_state = self.discrete_observation(observation)

        #Update Qtable based on observed value
        currentQ = self.Q[self.state][action]

        next_best_Q = np.max(self.Q[next_state])

        self.Q[self.state][action] = currentQ + self.alpha * (reward + self.gamma * next_best_Q - currentQ)
        
        self.state = next_state

        self.epsilon = self.epsilon * self.epsilon_decay

        return (self.epsilon, reward)