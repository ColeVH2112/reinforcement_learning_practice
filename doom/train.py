from stable_baselines3 import PPO
from env import VizDoomGym
from callback import TrainAndLoggingCallback

#config
CHECKPOINT_DIR = './train_doom'
LOG_DIR = './logs_doom'

def train():
    #initialize env
    env = VizDoomGym(render=False)

    #init callback
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    #init model
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)

    #start training
    print("training started")
    model.learn(total_timesteps=100000, callback=callback)
    print("training complete")

if __name__ == "__main__":
    train()
    
