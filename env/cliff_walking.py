import numpy as np
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
    
class EasyCliffWalking(CliffWalkingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, a):
        s, r, t, tr, i = super().step(a)
        if t:
            return (s, +100, t, tr, i)
        return (s, r, t, tr, i)