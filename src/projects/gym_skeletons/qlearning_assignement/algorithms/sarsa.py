from .algo import Rlalgorithm
import numpy as np
import random

class Sarsa(Rlalgorithm):
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

        #In the Q-learning, the next state observed is always the best one (np.max, optimitic approach).
        #This is out-of-policy as this does not respect the action choice that will be chosen - as the policy might be
        #Different than strictly the next best move. Sarsa stays in policy. To simulate that, we will simply change state and
        #Look at our next action performed to update the Q-table.

        old_q = self.Q[self.state][action]
        old_state = self.state

        self.state = next_state
        next_action = self.next_action()
        next_q = self.Q[next_state][next_action]

        self.Q[old_state][action] = old_q + self.alpha * (reward + self.gamma * next_q - old_q)

        self.epsilon = self.epsilon * self.epsilon_decay
        return (self.epsilon, reward)