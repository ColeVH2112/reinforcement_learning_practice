from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env import MarioGym
import time

def main():
    # 1. Recreate the environment exactly as it was in training
    # We must wrap it in DummyVecEnv and VecFrameStack because the model expects 4 stacked frames
    env = DummyVecEnv([lambda: MarioGym(render=True)])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    
    # 2. Load the model
    # If you just started, you might not have a saved zip yet. 
    # If so, comment out the load line and use this to test a random agent:
    # model = PPO("CnnPolicy", env) 
    
    # If you have a checkpoint, use this:
    try:
        model = PPO.load("./train_mario/best_model_10000.zip", env=env)
        print("Loaded saved model.")
    except:
        print("No model found, using random agent to test display.")
        model = PPO("CnnPolicy", env)

    obs = env.reset()
    
    # 3. Play Loop
    print("Starting visual check...")
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        # Mario environment renders automatically due to render=True in init, 
        # but we add a sleep so you can see it with human eyes.
        time.sleep(0.04)

if __name__ == "__main__":
    main()
