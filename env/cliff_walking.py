from copy import deepcopy

import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv

class WindyCliffWalking(CliffWalkingEnv):
    def __init__(self, wind: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wind = wind

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        if np.random.rand() <= self.wind and not terminated:
            obs, rew, terminated, truncated, info = super().step(2) # move down
        
        return obs, rew, terminated, truncated, info