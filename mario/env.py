import gymnasium as gym
from gymnasium import spaces
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import cv2

class MarioGym(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        
        # --- THE FIX ---
        # Instead of gym_super_mario_bros.make(), which adds the broken TimeLimit wrapper,
        # we instantiate the raw environment class directly.
        self.env = SuperMarioBrosEnv()
        
        # Apply the Joypad wrapper (reduces buttons from 256 -> 7)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        # Define the shapes for the Neural Network (Gymnasium API)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(7)
        
        # Render setup
        if render:
            self.env.reset()

    def step(self, action):
        # 1. Step the Old Env (Returns 4 values)
        obs, reward, done, info = self.env.step(action)
        
        # 2. Convert to New Gymnasium API (5 values)
        terminated = done
        truncated = False 
        
        # 3. Preprocess Image (Color -> Gray, Resize)
        obs = self.preprocess(obs)
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the Old Env
        obs = self.env.reset()
        
        # Preprocess
        obs = self.preprocess(obs)
        
        # Return (obs, info) required by Gymnasium
        return obs, {}

    def preprocess(self, observation):
        # Convert to Grayscale
        if len(observation.shape) == 3 and observation.shape[2] == 3:
            gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        else:
            gray = observation
        
        # Resize to 84x84 (Standard for RL)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Add the channel dimension: (84, 84) -> (84, 84, 1)
        resized = np.expand_dims(resized, axis=-1)
        
        return resized.astype(np.uint8)
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
