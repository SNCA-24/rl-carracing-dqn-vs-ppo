import gymnasium as gym
import numpy as np
import yaml
from gymnasium.spaces import Discrete, Box

class CarRacingEnv(gym.Wrapper):
    """
    Wrapper for CarRacing-v2 environment with a discrete action space and raw RGB observations.

    Observation:
      - Type: Box(0,255, shape=(96,96,3), dtype=uint8)
    Actions (Discrete 5):
      - 0: No action ([0,0,0])
      - 1: Left      ([-1,0,0])
      - 2: Right     ([1,0,0])
      - 3: Accelerate([0,1,0])
      - 4: Brake     ([0,0,0.8])
    """

    def __init__(self,render_mode = None):
        # Create the base environment
        mode = None if render_mode in (None, 'none', '', 'null') else render_mode
        env = gym.make('CarRacing-v2', render_mode=mode)
        super().__init__(env)

        # Load discrete action mappings from config file
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        raw_actions = cfg['env']['action_space']
        self.actions = {i: np.array(a, dtype=np.float32) for i, a in enumerate(raw_actions)}

        self.action_space = Discrete(len(self.actions))

        # Use the same observation space as the base env (RGB uint8 frames)
        self.observation_space = Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        """Reset the environment and return initial observation and info dict"""
        obs, info = super().reset(**kwargs)
        return obs, info

    def step(self, action_idx):
        """Execute a discrete action and return (obs, reward, done, truncated, info)"""
        cont_action = self.actions[int(action_idx)]
        obs, reward, done, truncated, info = self.env.step(cont_action)
        return obs, reward, done, truncated, info

    def close(self):
        """Close the environment cleanly"""
        super().close()
