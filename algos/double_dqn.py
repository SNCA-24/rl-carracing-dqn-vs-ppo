import numpy as np
import random
from algos.dqn_base import DQNAgent

class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent: extends DQNAgent by using the online network to select the best next action
    and the target network to evaluate its Q-value, reducing overestimation bias.
    """
    def replay(self):
        """Train the network with batches from replay memory"""
        # Only start training once enough samples are in memory
        if len(self.memory) < self.train_start:
            return
        
        # Sample a random minibatch
        minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        # Prepare state and next_state arrays
        states = np.zeros((len(minibatch),) + self.state_shape, dtype=np.float32)
        next_states = np.zeros_like(states)
        actions, rewards, dones = [], [], []
        
        for i, (s, a, r, s_next, done) in enumerate(minibatch):
            states[i] = s[0]
            next_states[i] = s_next[0]
            actions.append(a)
            rewards.append(r)
            dones.append(done)
        
        # Q-values for current states from primary network
        q_current = self.model.predict(states, verbose=0)
        
        # Double DQN: use primary network to choose actions for next_states
        q_next_primary = self.model.predict(next_states, verbose=0)
        # Use target network to evaluate those actions
        q_next_target = self.target_model.predict(next_states, verbose=0)
        
        # Vectorized build of training targets
        batch_size = len(minibatch)
        indices = np.arange(batch_size)
        best_actions = np.argmax(q_next_primary, axis=1)
        # compute TD targets: r + gamma * Q_target[next_state, best_action]
        td_targets = np.array(rewards, dtype=np.float32) + self.gamma * q_next_target[indices, best_actions]
        # for terminal transitions, override with reward only
        td_targets = np.where(dones, rewards, td_targets)
        # update Q-values for the taken actions
        q_current[indices, actions] = td_targets
        
        # Perform gradient descent step
        self.model.fit(states, q_current, batch_size=self.batch_size, verbose=0, shuffle=False)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network at fixed intervals
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_frequency:
            self.update_target_model()
            self.target_update_counter = 0
