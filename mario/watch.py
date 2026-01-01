from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from env import MarioGym
import os
import glob
import time

# Config
CHECKPOINT_DIR = './train_mario'

def find_latest_model(checkpoint_dir):
    """Find the latest model checkpoint"""
    model_files = glob.glob(os.path.join(checkpoint_dir, 'best_model_*.zip'))
    if not model_files:
        # Try final model
        final_model = os.path.join(checkpoint_dir, 'final_model.zip')
        if os.path.exists(final_model):
            return final_model
        return None
    
    # Sort by the number in the filename (e.g., best_model_100000.zip -> 100000)
    model_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    return model_files[-1]

def watch_model(model_path=None, num_episodes=5):
    """
    Watch the trained model play Super Mario Bros
    
    Args:
        model_path: Path to the model file. If None, uses the latest model.
        num_episodes: Number of episodes to run
    """
    # Find model if not specified
    if model_path is None:
        model_path = find_latest_model(CHECKPOINT_DIR)
        if model_path is None:
            print(f"No models found in {CHECKPOINT_DIR}")
            print("Make sure training has created at least one checkpoint.")
            return
        print(f"Using latest model: {os.path.basename(model_path)}")
    else:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return
    
    # Create environment (factory function for DummyVecEnv)
    def make_env():
        return MarioGym(render=True)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Stack frames for temporal information (4 frames) - must match training!
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"\n{'='*60}")
    print(f"Starting {num_episodes} episodes - Watch the AI play Mario!")
    print(f"{'='*60}\n")
    
    total_reward = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"Episode {episode + 1}/{num_episodes} - Starting...")
        print("(A game window should appear showing Mario playing)")
        
        while not done:
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            
            # Render the game (this shows the visual window)
            env.render()
            
            # Small delay so you can see the action (adjust speed: lower = faster)
            time.sleep(0.02)  # ~50 FPS
            
            # Handle vectorized environment (reward and done are arrays)
            episode_reward += sum(reward) if hasattr(reward, '__iter__') else reward
            done = done[0] if hasattr(done, '__iter__') else done
            steps += 1
            
            # Print episode stats periodically
            if steps % 100 == 0:
                print(f"  Step {steps}: Reward={episode_reward:.1f}")
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
        
        print(f"Episode {episode + 1} finished!")
        print(f"  Total Steps: {steps}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print()
    
    # Print summary
    print(f"{'='*60}")
    print("Summary:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Average Reward: {total_reward / num_episodes:.2f}")
    if episode_rewards:
        print(f"  Best Episode: {max(episode_rewards):.2f}")
        print(f"  Worst Episode: {min(episode_rewards):.2f}")
    print(f"{'='*60}")
    
    env.close()
    print("\nDone! Hope you enjoyed watching the AI play Mario!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch trained Mario RL agent play')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (default: latest model)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    
    args = parser.parse_args()
    
    watch_model(model_path=args.model, num_episodes=args.episodes)
