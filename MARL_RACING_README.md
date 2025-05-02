# Multi-Agent Proximal Policy Optimization (MAPPO) for MARL Racing Environment

This implementation trains agents using Multi-Agent Proximal Policy Optimization (MAPPO) in the MetaDrive Multi-Agent Racing environment. The agents learn to race against each other in a competitive setting.

## Overview

MAPPO is an extension of PPO for multi-agent reinforcement learning that uses a centralized value function and decentralized policies. This implementation trains two agents to race against each other in the `MultiAgentRacingSafeEnv` environment from MetaDrive.

## Requirements

- Python 3.7+
- Ray (with RLlib)
- MetaDrive
- PyTorch
- Gymnasium

You can install the required packages with:

```bash
pip install ray[rllib] metadrive-simulator torch gymnasium
```

## Files

- `marl_racing_env_wrapper.py`: Wrapper for the MetaDrive MARL racing environment to make it compatible with RLlib
- `train_mappo_marl_racing.py`: Main training script for MAPPO in the racing environment
- `run_trained_marl_racing.py`: Script to run and visualize trained policies

## Training

The `train_mappo_marl_racing.py` script trains MAPPO agents in the racing environment:

```bash
# Train from scratch
python train_mappo_marl_racing.py --num-gpus 1 --num-workers 4 --iterations 1000

# Train with traffic
python train_mappo_marl_racing.py --traffic-density 0.15 --seed 42

# Resume training from a checkpoint
python train_mappo_marl_racing.py --resume /path/to/checkpoint --num-gpus 1

# Specify checkpoint directory
python train_mappo_marl_racing.py --checkpoint-dir ./checkpoints --checkpoint-freq 20
```

### Training Parameters

- `--env`: Environment name for registration (default: "marl_race_env")
- `--num-gpus`: Number of GPUs to use for training (default: 0)
- `--num-workers`: Number of environment runner processes (default: 4)
- `--iterations`: Number of training iterations (default: 1000)
- `--checkpoint-dir`: Directory to save checkpoints (default: ~/ray_results/mappo_racing)
- `--checkpoint-freq`: Frequency to save checkpoints (default: 10)
- `--resume`: Path to checkpoint to resume training from
- `--eval`: Run evaluation after training
- `--traffic-density`: Traffic density for the environment (0.0 to 1.0, default: 0.0)
- `--seed`: Random seed for environment (default: 0)

## Running Trained Agents

The `run_trained_marl_racing.py` script runs trained policies in the environment:

```bash
python run_trained_marl_racing.py --checkpoint /path/to/checkpoint --num-episodes 10 --traffic-density 0.15
```

### Running Parameters

- `--checkpoint`: Path to the checkpoint directory (required)
- `--num-episodes`: Number of episodes to run (default: 5)
- `--seed`: Random seed (default: 0)
- `--traffic-density`: Traffic density (0.0 to 1.0, default: 0.15)
- `--delay`: Delay between steps in seconds for visualization (default: 0.0)

## Environment Configuration

The MARL racing environment is configured with:

- Two agents racing against each other
- A finish line at the end of the track
- Continuous simulation (no termination on crashes or boundary violations)
- Optional traffic with configurable density
- Racing-specific rewards including bonuses for leading and winning

## MAPPO Configuration

The MAPPO implementation uses:

- Two-hidden-layer neural networks with 256 units and tanh activations
- Separate value function networks
- PPO with GAE (Generalized Advantage Estimation)
- Appropriate hyperparameters for racing (gamma=0.99, lambda=0.95, etc.)

## Checkpoints

Checkpoints are saved to the specified directory (default: `~/ray_results/mappo_racing/`). The script saves:
- Periodic checkpoints based on the specified frequency
- Checkpoints when performance improves
- A final checkpoint at the end of training

## Customization

You can customize the training by modifying:
- Network architecture in the `model_config` dictionary
- PPO hyperparameters in the `training` section
- Environment parameters in the `env_config` dictionary
- Multi-agent settings in the `multi_agent` section
