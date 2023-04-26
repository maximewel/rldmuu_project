import tensorflow as tf

import numpy as np
from models import QPolicy
from gymnasium.spaces import Box, Discrete

import os

# disable gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DQL(tf.keras.Model):
    def __init__(self, observation_space, action_space, lr=1e-3, discount=0.99, target_update_period=15,
                 epsilon_decay=0.9995, epsilon_min=0.01):

        if isinstance(observation_space, Box):
            state_dim = observation_space.shape[0]
        else: # we are discrete, might need to be converted to one_hots
            state_dim = 1
        num_actions = action_space.n
        super(DQL, self).__init__(name='DQL')
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.eps_decay = epsilon_decay
        self.eps_min = epsilon_min
        self.epsilon = 1.

        self.update_index = 0

        # Instantiate the DQN, optimizer, etc.
        # We need to use a replay buffer and a target network for stability.

        self.qnn = QPolicy(state_dim, num_actions)
        self.target_qnn = QPolicy(state_dim, num_actions)
        self.target_qnn.set_weights(self.qnn.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.replay_buffer = ReplayBuffer(state_dim, int(1e5))

    def act(self, state):
        # The behaviour of our currently learning policy
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return self.qnn.act(state)

    def update(self, state, action, reward, new_state, done, log_name=""):
        # Save data in the replay buffer, sample a batch and learn from it
        self.replay_buffer.record(state, action, reward, new_state, done)

        batch = self.replay_buffer.sample(32)
        if batch is None:
            return
        
        states, new_states, actions, rewards, dones = batch

        loss = self._train(states, actions, rewards, new_states, dones, gpu=-1)

        #Eps decay
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

        tf.summary.scalar(name=f"{log_name}_epsilon", data=self.epsilon)
        tf.summary.scalar(name=f"{log_name}_loss", data=loss)

        if (self.update_index % self.target_update_period) == 0:
            self.target_qnn.set_weights(self.qnn.get_weights())
        
        self.update_index += 1

    @tf.function
    def _train(self, states, actions, rewards, new_states, dones, gpu):
        '''
        inner training function, arguments must be in datatypes recognized by tensorflow.

        tf.function decorator in Tensorflow2 speeds up execution. Compiles the code with C++
        Main training function
        Deals with array differently that python, make sure to use tensorflow methods to handle arrays.
        '''

        batch_size = len(states)

        # Possible to execute on a gpu, but for simplicity we do it on cpu with gpu =  -1
        with tf.device("/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"):
            
            with tf.GradientTape() as tape:

                current_q = self.qnn(states)

                indexed_actions = tf.concat(
                    [tf.expand_dims(tf.range(0, batch_size), axis=1), actions],
                    axis=1
                )
            
                current_q_sa = tf.gather_nd(current_q, indexed_actions) #Q[s,a]

                next_q = self.target_qnn(new_states)

                next_q_max = tf.reduce_max(next_q, axis=1)

                # Q = Q + alpha(next_q_max * discount + reward - Q)
                # Minimize (..)
                td_error = (1-dones) * next_q_max * self.discount + rewards - current_q_sa

                loss = tf.reduce_mean(tf.square(td_error))
            
            gradient = tape.gradient(loss, self.qnn.trainable_variables)

            #Update weights
            self.optimizer.apply_gradients(zip(gradient, self.qnn.trainable_variables))

        return loss

    def reset(self, state):
        # Nothing to do, we reset by calling DQL(args)
        pass


class ReplayBuffer:

    def __init__(self, observation_dim, size=5000):
        self.idx = 0
        self.observation_dim = observation_dim
        self.size = size

        self.states = np.zeros((size, observation_dim), dtype=np.float32)
        self.new_states = np.zeros((size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((size, 1), dtype=np.int32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

    def record(self, state, action, reward, new_state, done):
        current_id = self.idx & self.size

        # save the data
        self.states[current_id] = state
        self.new_states[current_id] = new_state
        self.actions[current_id] = action
        self.rewards[current_id] = reward
        self.dones[current_id] = done

        self.idx += 1

    def sample(self, sample_size=256):
        # Wait until enough data in the buffer, then sample uniformly from it
        if self.idx <= 2*sample_size:
            return None
        
        sample_ids = np.random.choice(min(self.idx, self.size), sample_size, replace=True)

        return (self.states[sample_ids], self.new_states[sample_ids], self.actions[sample_ids], self.rewards[sample_ids], self.dones[sample_ids])