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
        # For rendering, we need to pass render_mode or set it up properly
        self.env = SuperMarioBrosEnv()
        
        # Apply the Joypad wrapper (reduces buttons from 256 -> 7)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        # Define the shapes for the Neural Network (Gymnasium API)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(7)
        
        # Reward shaping tracking variables
        self.prev_x_pos = 0
        self.prev_y_pos = 0
        self.prev_score = 0
        self.prev_time = 400  # Mario starts with 400 time
        self.steps_without_progress = 0
        self.last_action = None
        self.in_air = False
        self.episode_start_time = 400
        
        # Enable rendering if requested
        self.render_mode = render
        if render:
            # Make sure the window is visible for rendering
            try:
                # Some versions need this to enable rendering
                self.env.render()
            except:
                pass

    def step(self, action):
        # 1. Step the Old Env (Returns 4 values)
        obs, reward, done, info = self.env.step(action)
        
        # 2. Reward Shaping - Modify reward to encourage progress, jumping, and completion
        shaped_reward = self._shape_reward(reward, info, done, action)
        
        # 3. Convert to New Gymnasium API (5 values)
        terminated = done
        
        # Time limit enforcement: truncate if time runs too low (simulates reduced time limit)
        # This encourages faster play - effectively creates a 250 second time limit
        current_time = info.get('time', self.prev_time)
        truncated = False
        if current_time < 150:  # If time is very low, truncate (simulating 250 second limit)
            truncated = True
            done = True  # Also mark as done 
        
        # 4. Preprocess Image (Color -> Gray, Resize)
        obs = self.preprocess(obs)
        
        # 5. Track last action for jump detection
        self.last_action = action
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _shape_reward(self, base_reward, info, done, action):
        """
        Shape the reward to encourage progress, jumping, and completion.
        
        Strategy:
        1. Harsh penalty for standing still (encourages constant movement)
        2. Reward for jumping actions (encourages obstacle navigation)
        3. Large completion bonus (encourages finishing level)
        4. Speed bonus (encourages fast completion)
        5. Strong incomplete penalty (discourages giving up)
        6. Scale down survival rewards (prevents standing still strategy)
        """
        # Extract game state from info dict
        # Common keys in gym_super_mario_bros: x_pos, y_pos, score, time, coins, flag_get
        current_x = info.get('x_pos', info.get('x', self.prev_x_pos))
        current_y = info.get('y_pos', info.get('y', self.prev_y_pos))
        current_score = info.get('score', self.prev_score)
        current_time = info.get('time', self.prev_time)
        flag_get = info.get('flag_get', False)
        
        # Initialize on first step if we don't have previous values
        if self.prev_x_pos == 0 and current_x > 0:
            self.prev_x_pos = current_x
            self.episode_start_time = current_time
        if self.prev_score == 0 and current_score > 0:
            self.prev_score = current_score
        if self.prev_y_pos == 0 and current_y > 0:
            self.prev_y_pos = current_y
        
        # 1. Distance-based reward (most important!)
        # Reward for moving right, penalize moving backward
        x_delta = current_x - self.prev_x_pos
        if x_delta > 0:
            # Moving right - reward proportional to distance
            distance_reward = x_delta * 0.1  # 0.1 per pixel moved right
            self.steps_without_progress = 0
        elif x_delta < 0:
            # Moving backward - penalty
            distance_reward = x_delta * 0.05  # Penalty for going backward
            self.steps_without_progress = 0
        else:
            # Standing still - HARSH penalty that increases over time
            self.steps_without_progress += 1
            # Start at -0.05, scale up to -0.1 after 10 steps
            standing_penalty = -0.05 - (0.05 * min(self.steps_without_progress / 10, 1.0))
            distance_reward = standing_penalty
        
        # 2. Jumping reward (encourage exploration and obstacle navigation)
        # SIMPLE_MOVEMENT actions: 0=NOOP, 1=right, 2=right+A, 3=right+B, 4=right+A+B, 5=A, 6=left
        # Actions 2, 3, 4, 5 include jump (A button)
        jump_actions = [2, 3, 4, 5]
        jump_reward = 0.0
        if action in jump_actions:
            jump_reward = 0.02  # Reward for attempting to jump
        
        # Also reward for being in air (y-position changes)
        y_delta = current_y - self.prev_y_pos
        if y_delta != 0:
            # In air or falling - small reward for vertical movement
            jump_reward += 0.01
            self.in_air = True
        else:
            self.in_air = False
        
        # 3. Completion reward (HUGE bonus for finishing level)
        completion_reward = 0.0
        if flag_get:
            completion_reward = 200.0  # Large reward for completing level
            # Additional speed bonus based on remaining time
            time_remaining = current_time
            speed_bonus = time_remaining * 0.05  # 0.05 per second remaining
            completion_reward += speed_bonus
        
        # 4. Score-based reward (coins, enemies killed)
        score_delta = current_score - self.prev_score
        score_reward = score_delta * 0.01  # Scale down score rewards
        
        # 5. Time-based reward (survival) - heavily scaled down
        # This prevents the agent from just standing still to accumulate time rewards
        time_reward = base_reward * 0.05  # Even more reduced (was 0.1)
        
        # 5b. Time pressure penalty (encourages faster completion)
        # Penalty increases as time runs out (simulates reduced time limit)
        time_pressure = 0.0
        if current_time < 250:  # Below 250 seconds, start applying pressure
            time_under_limit = 250 - current_time
            time_pressure = -0.001 * time_under_limit  # Small penalty that grows
        
        # 6. Incomplete level penalty (if episode ends without completion)
        incomplete_penalty = 0.0
        if done and not flag_get:
            # Strong penalty for not completing the level
            incomplete_penalty = -30.0
            # Additional penalty if didn't make much progress
            if current_x < 200:  # Didn't get far at all
                incomplete_penalty -= 20.0  # Extra -20 for very poor performance
        
        # 7. Progress bonus (small bonus for any forward movement)
        progress_bonus = 0.01 if x_delta > 0 else 0.0
        
        # Total shaped reward
        total_reward = (
            time_reward +           # Scaled survival reward (minimal)
            distance_reward +       # Progress/standing still penalty (most important)
            jump_reward +           # Jumping rewards
            score_reward +          # Achievement rewards
            progress_bonus +        # Small progress bonus
            completion_reward +     # Completion and speed bonus
            incomplete_penalty +    # Incomplete penalty
            time_pressure            # Time pressure (encourages speed)
        )
        
        # Update tracking variables
        self.prev_x_pos = current_x
        self.prev_y_pos = current_y
        self.prev_score = current_score
        self.prev_time = current_time
        
        return total_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the Old Env (returns just obs in old gym API)
        obs = self.env.reset()
        
        # Reset reward shaping tracking variables
        # We'll initialize these properly after first step when we have info
        self.prev_x_pos = 0
        self.prev_y_pos = 0
        self.prev_score = 0
        self.prev_time = 400  # Mario typically starts with 400 time
        self.steps_without_progress = 0
        self.last_action = None
        self.in_air = False
        self.episode_start_time = 400
        
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
