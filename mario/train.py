from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from env import MarioGym
import os

# Config
CHECKPOINT_DIR = './train_mario'
LOG_DIR = './logs_mario'

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
        return True

def make_env():
    # Factory function that returns the env
    return MarioGym(render=False)

def train():
    # 1. Setup Vectorized Environment
    # We wrap the factory function in DummyVecEnv
    env = DummyVecEnv([make_env])
    
    # 2. Frame Stacking
    # This stacks 4 frames so the AI can see movement (velocity)
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    
    model = PPO(
        'CnnPolicy',
        env,
        verbose = 1,
        tensorboard_log = LOG_DIR,
        learning_rate = 0.0001,
        n_steps = 2048,
        batch_size = 64,
        n_epochs = 10,
        gamma = 0.99,
        gae_lambda = 0.95,
        ent_coef = 0.01
    )
    
    print("Training Started...")
    model.learn(total_timesteps=1000000, callback=callback)
    
    final_path = os.path.join(CHECKPOINT_DIR, "final_model")
    model.save(final_path)
    print("Saved Final Model.")

if __name__ == "__main__":
    train()
