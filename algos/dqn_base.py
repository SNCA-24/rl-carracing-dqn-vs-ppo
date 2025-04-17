# algos/dqn_base.py

import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    """
    Basic DQN agent. All hyperparameters are passed in via the `config` dict,
    which should correspond to the `dqn:` section of config.yaml.
    """
    def __init__(self, state_shape, action_size, config):
        self.state_shape = state_shape      # e.g. (84,84,4)
        self.action_size = action_size      # number of discrete actions
        
        # Unpack hyperparameters from config
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_start"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.train_start = config["train_start"]
        self.update_target_frequency = config["update_target_frequency"]
        
        # Experience replay memory
        self.memory = deque(maxlen=config["memory_size"])
        
        # Build primary and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Counter to track when to sync target network
        self.target_update_counter = 0

    def _build_model(self):
        """Constructs the Q-network with a simple 3‑conv + dense architecture."""
        model = Sequential([
            Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                   input_shape=self.state_shape),
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from primary model to the target network."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """
        Epsilon‑greedy action selection.
        `state` is expected as a (1, H, W, C) numpy array.
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        """Sample a minibatch and do a single gradient update on the Q‑network."""
        if len(self.memory) < self.train_start:
            return
        
        # Sample a batch
        minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        # Prepare arrays
        states = np.zeros((len(minibatch),) + self.state_shape, dtype=np.float32)
        next_states = np.zeros_like(states)
        actions, rewards, dones = [], [], []
        
        for i, (s, a, r, s_next, done) in enumerate(minibatch):
            states[i] = s[0]           # remove batch dim
            next_states[i] = s_next[0]
            actions.append(a)
            rewards.append(r)
            dones.append(done)
        
        # Predict Q(s,·) and Qₜₐᵣ₉ₑₜ(s,·)
        q_current = self.model.predict(states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)
        
        # Build targets
        for i in range(len(minibatch)):
            if dones[i]:
                q_current[i][actions[i]] = rewards[i]
            else:
                q_current[i][actions[i]] = (
                    rewards[i] + self.gamma * np.max(q_next_target[i])
                )
        
        # Gradient step
        self.model.fit(states, q_current,
                       batch_size=self.batch_size,
                       verbose=0, shuffle=False)
        
        # Decay ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Sync target network periodically
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_frequency:
            self.update_target_model()
            self.target_update_counter = 0

    def load_model(self, filepath):
        """Load network weights and sync target network."""
        self.model.load_weights(filepath)
        self.update_target_model()

    def save_model(self, filepath):
        """Save primary network weights to `filepath`."""
        # Ensure correct file extension for Keras save_weights
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.model.save_weights(filepath)