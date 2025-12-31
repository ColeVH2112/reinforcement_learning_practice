import gymnasium as gym
from gymnasium import spaces 
from vizdoom import DoomGame
import cv2
import numpy as np
import os
import vizdoom

class VizDoomGym(gym.Env):
    def __init__(self, render=False, config=None):
        super().__init__()
        #Set up game engine
        self.game = DoomGame()
        if config is None:
            base_path = os.path.dirname(vizdoom.__file__)
            config = os.path.join(base_path, 'scenarios', 'deadly_corridor.cfg')
            
        self.game.load_config(config)
        self.game.set_window_visible(render) #training? False : True
        self.game.init()

        # set up observation space
        #Shrink "screen" size in order to speed training
        self.observation_space = spaces.Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)

        #Define Action: 7 buttons- left, right, shoot,...
        self.action_space = spaces.Discrete(7)

        #Tracking var. for reward
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52

    #Create step method in order to run every frame of game for learning
    def step(self, action):
        #convert action into button pressed
        actions = np.identity(7, dtype=np.uint8)

        #make move-- can speed up by increasing second int argument(repeat action " " times)
        move_reward = self.game.make_action(actions[action],4)
        reward = 0

        #screen
        state = self.game.get_state()
        info = {}

        if state:
            #simplify screen to grayscale
            img = state.screen_buffer
            state = self.grayscale(img)

            #Reward Shaping
            game_variables = self.game.get_state().game_variables
            
            # Handle different numbers of game variables
            # deadly_corridor.cfg typically only has health (1 variable)
            # Extract what's available, use defaults for missing ones
            num_vars = len(game_variables)
            health = game_variables[0] if num_vars > 0 else 100
            damage_taken = game_variables[1] if num_vars > 1 else 0
            hitcount = game_variables[2] if num_vars > 2 else 0
            ammo = game_variables[3] if num_vars > 3 else 52

            #calc changes 
            damage_taken_delta = -damage_taken + self.damage_taken
            hitcount_delta = hitcount - self.hitcount
            ammo_delta = ammo - self.ammo

            #update hist
            self.damage_taken = damage_taken
            self.hitcount = hitcount
            self.ammo = ammo

            #incentive struct
            reward = move_reward + damage_taken_delta*10 + hitcount_delta*200 + ammo_delta*5

            info["ammo"] = ammo
            info["health"] = health

        else:
            #Game over -> black
            state = np.zeros(self.observation_space.shape)
            info = {}

        done = self.game.is_episode_finished()

        terminated = done
        truncated = False
        
        return state, reward, terminated, truncated, info

    
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game.new_episode()
        game_state = self.game.get_state()
        if game_state:
            state = game_state.screen_buffer
            state = self.grayscale(state)
        else:
            state = np.zeros(self.observation_space.shape)
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52
        return state, {}

    def close(self):
        self.game.close()
        
        
