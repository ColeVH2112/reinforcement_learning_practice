from stable_baselines3 import PPO
from env import VizDoomGym
import os
import glob

# Config
CHECKPOINT_DIR = './train_doom'

def find_latest_model(checkpoint_dir):
    """Find the latest model checkpoint"""
    model_files = glob.glob(os.path.join(checkpoint_dir, 'best_model_*.zip'))
    if not model_files:
        return None
    
    # Sort by the number in the filename (e.g., best_model_100000.zip -> 100000)
    model_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    return model_files[-1]

def watch_model(model_path=None, num_episodes=5):
    """
    Watch the trained model play Doom
    
    Args:
        model_path: Path to the model file. If None, uses the latest model.
        num_episodes: Number of episodes to run
    """
    # Find model if not specified
    if model_path is None:
        model_path = find_latest_model(CHECKPOINT_DIR)
        if model_path is None:
            print(f"No models found in {CHECKPOINT_DIR}")
            return
        print(f"Using latest model: {os.path.basename(model_path)}")
    else:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return
    
    # Initialize environment with rendering enabled
    print("Initializing environment with rendering...")
    env = VizDoomGym(render=True)
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env)
    
    print(f"\n{'='*60}")
    print(f"Starting {num_episodes} episodes - Watch the AI play!")
    print(f"{'='*60}\n")
    
    total_reward = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"Episode {episode + 1}/{num_episodes} - Starting...")
        
        while not done:
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Print episode stats periodically
            if steps % 100 == 0:
                health = info.get('health', 'N/A')
                ammo = info.get('ammo', 'N/A')
                print(f"  Step {steps}: Reward={episode_reward:.1f}, Health={health}, Ammo={ammo}")
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
        
        print(f"Episode {episode + 1} finished!")
        print(f"  Total Steps: {steps}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Health: {info.get('health', 'N/A')}")
        print(f"  Ammo: {info.get('ammo', 'N/A')}")
        print()
    
    # Print summary
    print(f"{'='*60}")
    print("Summary:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Average Reward: {total_reward / num_episodes:.2f}")
    print(f"  Best Episode: {max(episode_rewards):.2f}")
    print(f"  Worst Episode: {min(episode_rewards):.2f}")
    print(f"{'='*60}")
    
    env.close()
    print("\nDone! Hope you enjoyed watching the AI play!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch trained Doom RL agent play')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (default: latest model)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    
    args = parser.parse_args()
    
    watch_model(model_path=args.model, num_episodes=args.episodes)

