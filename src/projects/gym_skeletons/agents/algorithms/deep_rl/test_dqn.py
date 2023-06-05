# Principle of DQN: https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-3-q-learning-with-neural-networks-algorithm-dqn-1e22ee928ecd
# DQN applied with Pytorch (Framework I know): https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Other example: https://colab.research.google.com/drive/1w5xFX2wJvtuVbcrDHny7YPcTdGqMOqMu#sandboxMode=true&scrollTo=HdBCQEGNcVnV

# Interesting page: https://xusophia.github.io/DataSciFinalProj/

from ..algo import Rlalgorithm

from collections import namedtuple, deque
import random

from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class DqnNetwork(nn.Module):

    def __init__(self, n_observations: int, n_actions: int, hidden_layer_neurons: int, hidden_layers_count: int):
        super().__init__()
        
        #Init model

        #Init layer
        self.model = nn.Sequential(
            nn.Linear(n_observations, hidden_layer_neurons),
            nn.ReLU()
        )

        #Number of hidden layers
        for _ in range(hidden_layers_count):
            self.model.append(nn.Linear(hidden_layer_neurons, hidden_layer_neurons))
            self.model.append(nn.ReLU())

        #Output layer
        self.model.append(nn.Linear(hidden_layer_neurons, n_actions))

    def forward(self, x):
        return self.model(x)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def last_value(self):
        if self.__len__() <= 0:
            return None
        return self.memory.pop()

    def __len__(self):
        return len(self.memory)

class DqnAlgorithm(Rlalgorithm):
    """The DQN algorithm implements the "algo" interface, but does not want its default behavior as is can accept input values into the network.
    
    Will only work for environments with (Box, Discrete) combinations"""

    #Hyper-parameters
    t: int
    k: int  #Epsilon arithmetic decay
    epsilon: float
    gamma: float

    batch_size: int
    hidden_layer_count: int
    hidden_layer_neurons: int
    TAU: float
    lr: float
    target_update_episodes: int #Updates after which the target's weight are copied into the online

    #DNN
    policy_net: nn.Module
    target_net: nn.Module

    optimizer: torch.optim.Adam
    loss_fn: nn.SmoothL1Loss
    
    #Used for replay
    memory: ReplayMemory

    terrain_size: int

    def __init__(self, k: int = 50000, epsilon: float = 1.0, gamma: float = 0.9, lr: float = 1e-4, hidden_layer_neurons: int = 32, hidden_layers_count: int = 1, batch_size: int = 128, 
                 tau: float = 0.005, target_update_episodes: int = 20, terrain_size: int = 10) -> None:
        super().__init__()
        self.t = 0
        self.k = k
        self.epsilon = epsilon

        self.gamma = gamma

        self.lr = lr
        self.batch_size = batch_size
        self.hidden_layer_neurons = hidden_layer_neurons
        self.hidden_layer_count = hidden_layers_count
        self.TAU = tau
        self.target_update_episodes = target_update_episodes

        self.terrain_size = terrain_size
    
    def set_env(self, state_space: Box , actions_spaces: Discrete, env_bounds: any):
        if not (isinstance(state_space, Box) and isinstance(actions_spaces, Discrete)):
            raise Exception("DQN only works for (box, Discrete) obseravtion, states spaces. Please choose a fitting environment.")

        self.state_space = state_space 
        self.actions_spaces = actions_spaces

        #Retrieve state, action properties from the spaces
        n_observations = state_space.shape[0]
        n_actions = actions_spaces.n

        #Create the two NN models with this information. They start the same but are not updated at the same time
        self.policy_net = DqnNetwork(n_observations, n_actions, self.hidden_layer_neurons, self.hidden_layer_count)
        self.target_net = DqnNetwork(n_observations, n_actions, self.hidden_layer_neurons, self.hidden_layer_count)

        #Synchronize both states to have replicas
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayMemory(10000)
    
    def normalize_observation(self, observation: np.ndarray):
        if observation is None:
            return None
        
        x,y = observation[0], observation[1]

        x /= self.terrain_size
        y /= self.terrain_size

        normalized_obs = observation.astype(np.float32)

        normalized_obs[0] = x
        normalized_obs[1] = y

        return normalized_obs

    def next_action(self):
        """Apply policy to determine next action using current self policy model's estimations, on the same basis as Q-learning"""
        if (random.uniform(0, 1) < self.epsilon):
            return self.actions_spaces.sample()

        with torch.no_grad():
            action_as_tensor = self.policy_net(self.state).max(1)[1].view(1, 1)
            return action_as_tensor.item()

    def update(self, action, observation, reward):
        """Update the state of the RL algo with the observed parameters"""
        
        next_state = None if observation is None else torch.tensor(self.normalize_observation(observation), dtype=torch.float32).unsqueeze(0)

        reward_as_tensor = torch.tensor([reward])
        action_as_tensor = torch.tensor([[action]])

        self.memory.push(self.state, action_as_tensor, next_state, reward_as_tensor)
        self.state = next_state

        self.optimize_model()
        self.update_models()
        
        self.t += 1
        self.epsilon = self.k / (self.k + self.t)

        return self.epsilon, reward
    
    def update_models(self):
        """Update the secondary model according to the policy model"""

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        
        if self.t % self.target_update_episodes == 0:
            #When we are at a full update episode, load the online state dict into the target
            self.target_net.load_state_dict(policy_net_state_dict)
        else:
            #Do a very simple tradeoff on each parameter of the target network: target <- (tau*target) + ((1-tau) * policy)
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            
            self.target_net.load_state_dict(target_net_state_dict)
    
    def optimize_model(self):
        """Train the main model with internal memory sampling (replay)"""
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def set_state(self, observation: any):
        observation = self.normalize_observation(observation)

        #Don't call discretization method as we want the continuous input in the NN
        self.state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)