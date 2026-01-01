# Reinforcement Learning Practice

This repository contains reinforcement learning implementations using Stable-Baselines3, focusing on game environments like VizDoom and Super Mario Bros.

## Projects

### 1. Doom Agent (`doom/`)

A PPO (Proximal Policy Optimization) agent trained on VizDoom's "deadly_corridor" scenario.

#### Files:
- `env.py` - Custom Gymnasium environment wrapper for VizDoom
- `train.py` - Training script using PPO
- `watch.py` - Script to watch trained agent play
- `callback.py` - Callback for saving checkpoints during training

#### Features:
- Custom reward shaping based on damage taken, hit count, and ammo
- Grayscale image preprocessing (100x160)
- 7 discrete actions (movement + shooting)
- Automatic checkpoint saving every 10,000 steps

#### Usage:

**Training:**
```bash
cd doom
python train.py
```

**Watching trained agent:**
```bash
cd doom
python watch.py --episodes 5
python watch.py --model ./train_doom/best_model_100000.zip --episodes 10
```

### 2. Mario Agent (`mario/`)

A PPO agent trained on Super Mario Bros using the `gym_super_mario_bros` environment.

#### Files:
- `env.py` - Custom Gymnasium environment wrapper for Super Mario Bros
- `train.py` - Training script with PPO
- `watch.py` - Script to watch trained agent play

#### Features:
- Custom environment wrapper (similar to Doom structure)
- Simplified action space (7 actions: SIMPLE_MOVEMENT)
- Grayscale observation preprocessing (84x84)
- Frame stacking (4 frames) for temporal information
- CNN policy for visual input processing

#### Usage:

**Training:**
```bash
cd mario
python train.py
```

**Watching trained agent:**
```bash
cd mario
python watch.py --episodes 5
python watch.py --model ./train_mario/best_model_100000.zip --episodes 10
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For VizDoom, you may need to install additional system dependencies. See [VizDoom documentation](https://github.com/mwydmuch/ViZDoom).

3. **Important for Mario**: The `nes_py` library (used by `gym_super_mario_bros`) is incompatible with NumPy 2.0. You must use NumPy < 2.0:
   ```bash
   pip install "numpy<2.0"
   ```
   Or reinstall from requirements:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

## Requirements

- `gymnasium` - Modern Gym API
- `stable-baselines3[extra]` - RL algorithms
- `vizdoom` - Doom game environment
- `opencv-python` - Image processing
- `gym_super_mario_bros` - Super Mario Bros environment
- `nes_py` - NES emulator backend
- `numpy` - Numerical operations

## Training Tips

### Doom:
- Training typically requires 100,000+ timesteps for decent performance
- Monitor TensorBoard logs: `tensorboard --logdir ./logs_doom`
- Adjust reward shaping in `env.py` to encourage desired behaviors

### Mario:
- Frame stacking helps the agent understand motion
- Training may require 500,000+ timesteps for good performance
- Monitor TensorBoard logs: `tensorboard --logdir ./logs_mario`

## Model Checkpoints

Both agents save checkpoints during training:
- Checkpoints saved every 10,000 steps in `train_doom/` or `train_mario/`
- Final model saved at the end of training
- Use `watch.py` to load and test any checkpoint

## Hyperparameters

### Doom PPO:
- Learning rate: 0.0001
- n_steps: 2048
- Policy: CnnPolicy

### Mario PPO:
- Learning rate: 0.000001
- n_steps: 512
- Policy: CnnPolicy

## Future Improvements

- [ ] Add DQN, A2C, and other algorithms
- [ ] Implement evaluation metrics and benchmarking
- [ ] Add hyperparameter tuning utilities
- [ ] Create visualization tools for training progress
- [ ] Add support for more game environments
- [ ] Implement curriculum learning

## License

This is a personal learning project. Use at your own discretion.

