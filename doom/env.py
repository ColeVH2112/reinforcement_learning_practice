from gym import Env, spaces
from vizdoom import DoomGame
import cv2
import numpy as np

class VizDoomGym(Env):
    def __init__(self, render=False, config=None):
        super().__init__()
        #Set up game engine
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(render) #training? False : True
        self.game.init()

        # set up observation space
        #Shrink "screen" size in order to speed training
        self.observation_space = space.Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)

        #Define Action: 7 buttons- left, right, shoot,...
        self.action_space = spaces.Discrete(7)

        #Tracking var. for reward
        self.damage_take = 0
        self.hitcount = 0
        self.ammo = 52

    #Create step method in order to run every frame of game for learning
    def step(self, action):
        #convert action into button pressed
        actions = np.identity(7, dtype=np.uint8)

        #make move-- can speed up by increasing second int argument(repeat action " " times)
        move_reward = self.game.make_action(actions[action],2)
        reward = 0

        #screen
        state = self.game.get_state()

        if state:
            #simplify screen to grayscale
            img = state.screen_buffer
            state = self.grayscale(img)

            #Reward Shaping
            game_variables = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo = game_variables

            #calc changes 
            damage_taken_delta = -damage_taken + self.damge_taken
            hitcount_delta = hitcount - self.hitcount
            ammo_delta = ammo - self.ammo

            #update hist
            self.damage_taken = damage_taken
            self.hitcount = hitcount
            self.ammo = ammo

            #incentive struct
            reward = move_reward + damage_taken_delta*10 + hicount_delta*200 + ammo_delta*5

            info = {"health": health, "ammo":ammo}

        else:
            #Game over -> black
            state = np.zeros(self.observation_space.shape)
            info = {}

        done = self.game.is_episode_finished()
        return state,reward,done,info

    
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52
        return self.grayscale(state)

    def close(self):
        self.game.close()
        
        
