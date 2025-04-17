import numpy as np
import random
from algos.dqn_base import DQNAgent

class SumTree:
    """SumTree for Prioritized Experience Replay"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority, data):
        # insert data with given priority
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx, priority):
        # change priority and propagate
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        # retrieve leaf index, priority, and data for value v
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left]:
                parent_idx = left
            else:
                v -= self.tree[left]
                parent_idx = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]

class PERDQNAgent(DQNAgent):
    """
    DQN Agent with Prioritized Experience Replay.
    Inherits base DQNAgent, overrides memory and replay.
    """
    def __init__(self, state_shape, action_size, dqn_config, per_config):
        # initialize DQN parameters via base class
        super().__init__(state_shape, action_size, dqn_config)
        # PER parameters
        self.alpha = per_config["alpha"]
        self.beta = per_config["beta"]
        self.beta_increment = per_config["beta_increment"]
        self.epsilon_per = per_config["epsilon_per"]
        # replace replay memory with SumTree
        mem_capacity = dqn_config["memory_size"]
        self.memory = SumTree(mem_capacity)

    def remember(self, state, action, reward, next_state, done):
        """Store experience with max priority"""
        # highest existing priority or 1.0 if empty
        max_prio = np.max(self.memory.tree[-self.memory.capacity:]) if self.memory.size > 0 else 1.0
        self.memory.add(max_prio, (state, action, reward, next_state, done))

    def replay(self):
        """Train using prioritized replay with importance sampling"""
        if self.memory.size < self.train_start:
            return
        batch_size = self.batch_size
        # calculate segment
        total = self.memory.total_priority()
        segment = total / batch_size
        batch = []
        idxs = []
        ISWeights = np.zeros((batch_size, 1), dtype=np.float32)
        # sample batch
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, prio, data = self.memory.get_leaf(s)
            idxs.append(idx)
            batch.append(data)
            sampling_prob = prio / total
            ISWeights[i, 0] = (self.memory.size * sampling_prob) ** (-self.beta)
        ISWeights /= ISWeights.max()
        # prepare arrays
        states = np.zeros((batch_size,) + self.state_shape, dtype=np.float32)
        next_states = np.zeros_like(states)
        actions, rewards, dones = [], [], []
        for i, (s, a, r, s_next, done) in enumerate(batch):
            states[i] = s[0]
            next_states[i] = s_next[0]
            actions.append(a)
            rewards.append(r)
            dones.append(done)
        # compute targets (Double DQN style)
        q_current = self.model.predict(states, verbose=0)
        q_next_primary = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)
        # Vectorized TD target computation
        indices = np.arange(batch_size)
        # Select best next actions from primary network
        best_next_actions = np.argmax(q_next_primary, axis=1)
        # Compute raw TD targets
        td_targets = np.array(rewards, dtype=np.float32) + self.gamma * q_next_target[indices, best_next_actions]
        # Override for terminal transitions
        td_targets = np.where(dones, rewards, td_targets)
        # Compute TD errors
        td_errors = np.abs(td_targets - q_current[indices, actions])
        # Update Q-current for actions taken
        q_current[indices, actions] = td_targets

        # Update priorities in the SumTree
        new_priorities = (td_errors + self.epsilon_per) ** self.alpha
        for idx, p in zip(idxs, new_priorities):
            self.memory.update(idx, p)
        # train with importance sampling weights
        self.model.fit(states, q_current,
                       sample_weight=ISWeights.flatten(),
                       batch_size=batch_size, verbose=0, shuffle=False)
        # anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # update target network
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_frequency:
            self.update_target_model()
            self.target_update_counter = 0
