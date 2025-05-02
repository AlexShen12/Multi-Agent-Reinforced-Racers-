# Multi-Agent Proximal Policy Optimization (MAPPO) for Racing Environment

This implementation of Multi-Agent Proximal Policy Optimization (MAPPO) trains agents in the `RLLibMappoEnv` racing environment using RLlib.

## Overview

MAPPO is an extension of PPO for multi-agent reinforcement learning that uses a centralized value function and decentralized policies. This implementation trains two agents to race against each other in the `RLLibMappoEnv` environment from MetaDrive.

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

## Scripts

### Training

The `train_mappo_racing.py` script trains MAPPO agents in the racing environment:

```bash
# Train from scratch
python train_mappo_racing.py --num-gpus 1 --num-workers 4 --iterations 1000

# Resume training from a checkpoint
python train_mappo_racing.py --resume /path/to/checkpoint --num-gpus 1
```

Key parameters:
- `--num-gpus`: Number of GPUs to use for training (default: 0)
- `--num-workers`: Number of rollout worker processes (default: 4)
- `--iterations`: Number of training iterations (default: 1000)
- `--checkpoint-freq`: Frequency to save checkpoints (default: 10)
- `--resume`: Path to checkpoint to resume training from
- `--eval`: Run evaluation after training

### Evaluation

The `evaluate_mappo_racing.py` script evaluates trained policies:

```bash
python evaluate_mappo_racing.py --checkpoint /path/to/checkpoint --num-episodes 10
```

Key parameters:
- `--checkpoint`: Path to the checkpoint directory (required)
- `--num-episodes`: Number of episodes to evaluate (default: 10)
- `--render`: Render the environment (default: True)
- `--seed`: Random seed (default: 0)

## MAPPO Configuration

The MAPPO implementation uses the following key components:

1. **Centralized Value Function**: The value function has access to global information during training, while policies remain decentralized.

2. **Policy Architecture**:
   - Two hidden layers with 256 units each
   - Tanh activation functions
   - Separate networks for policy and value function

3. **PPO Hyperparameters**:
   - Learning rate: 3e-4 (policy), 1e-3 (value function)
   - Discount factor (gamma): 0.99
   - GAE lambda: 0.95
   - Clip parameter: 0.2
   - Value function clip parameter: 10.0
   - Entropy coefficient: 0.01
   - Value function loss coefficient: 0.5

4. **Training Configuration**:
   - Batch size: 4000
   - Mini-batch size: 128
   - SGD iterations: 10
   - Separate value function optimizer

## Checkpoints

Checkpoints are saved to `~/ray_results/mappo_racing/` by default. The script saves:
- Periodic checkpoints based on the specified frequency
- Checkpoints when performance improves
- A final checkpoint at the end of training

## Integration with MetaDrive

This implementation is designed to work with the `RLLibMappoEnv` environment from MetaDrive, which provides:
- Two-agent racing scenario
- Continuous action space (2D: steering and acceleration)
- Observation space with 91 dimensions per agent
- Racing-specific rewards including bonuses for leading and winning

## Customization

You can customize the training by modifying:
- Network architecture in the `model_config` dictionary
- PPO hyperparameters in the `training` section
- Environment parameters in the `environment` section
- Multi-agent settings in the `multi_agent` section
