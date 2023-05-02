import gymnasium
import numpy as np

from .base_agent import BaseAgent

# IN PROGRESS
# QLearning with Eligibility traces
class QLearningEliTra(BaseAgent):
    def __init__(self, 
                 env,
                 discount=0.99,
                 learning_rate=0.01,
                 eps=1.,
                 eps_decay=0.9999,
                 eps_final=1e-2):
        super().__init__(env)
        self.env = env
        print(self.env.observation_space.shape)
        print(self.env.action_space.shape)
        self.n_actions = env.action_space.shape
        self.n_states = 2
        self.Q = []
        self.discount = discount
        self.learning_rate = learning_rate
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_final = eps_final

    def make_decision(self, observation, explore=False) -> int:
        print("observation:", observation)
        if explore and np.random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            # QLearning: argmax action according to Q
            pass
            

    def learn(self, state, action, reward, next_state) -> dict:
        pass

