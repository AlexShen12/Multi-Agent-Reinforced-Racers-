# Stable-Baselines3 PPO Policy for MetaDrive Racing

This implementation replaces the default expert policy in the MetaDrive multi-agent racing environment with a custom PPO policy using Stable-Baselines3. It includes:

1. Custom policy network architecture matching the expert policy
2. Multi-agent training with separate models for each agent
3. Persistence mechanisms for saving and loading model weights
4. Integration with the existing environment

## Files

- `sb3_ppo_policy.py`: Contains the PPO policy implementation and integration code
- `train_sb3_ppo_racing.py`: Training script for the PPO policy
- `drive_in_marl_racing_env.py`: Modified to support the new PPO policy

## Requirements

```
pip install stable-baselines3
```

## Usage

### Running with the SB3 PPO Policy

```bash
python -m metadrive.examples.drive_in_marl_racing_env --use_sb3_ppo --models_dir trained_models
```

### Training the SB3 PPO Policy

```bash
python -m metadrive.examples.train_sb3_ppo_racing --render --episodes 1000 --models_dir trained_models
```

## Key Features

### Network Architecture

The policy network architecture matches the expert policy:
- Input layer: 275 dimensions (matching expert observation space)
- Two hidden layers with 256 units each and tanh activation
- Output layer providing the mean and log_std for actions

### Multi-Agent Training

Each agent has its own PPO model with separate weights. The models are trained independently but share the same environment.

### Persistence

Models are saved after each episode and loaded when the environment is reset. This ensures that training progress is not lost if the program is terminated.

### Manual Reset

Press the BACKSPACE key during simulation to manually reset the episode and reinitialize the weights for that episode only. This is useful for testing different initialization points without losing the overall training progress.

### Integration with Environment

The SB3 PPO policy is integrated with the existing environment through the `replace_expert` function, which replaces the default expert policy with our custom PPO policy.

## Implementation Details

### Policy Network

```python
class MetaDrivePPOPolicy(ActorCriticPolicy):
    """
    Custom PPO policy for MetaDrive with architecture matching the expert policy.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        # Use custom feature extractor
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = MetaDrivePPOFeaturesExtractor
            
        # Set default network architecture if not provided
        if net_arch is None:
            # Two hidden layers with 256 units each
            net_arch = [256, 256]
            
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
```

### Multi-Agent Trainer

```python
class MultiAgentPPOTrainer:
    """
    Manages multiple PPO models for multi-agent training in MetaDrive.
    Handles model creation, training, saving, and loading for each agent.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_agents: int = 2,
        models_dir: str = "trained_models",
        verbose: int = 1
    ):
        # Create directory for saving models
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Dictionary to store PPO models for each agent
        self.models = {}
        
        # Dictionary to track if an agent's model has been reset this episode
        self.reset_status = {}
        
        # Initialize models for each agent
        self._initialize_models()
```

### Expert Replacement

```python
def replace_expert():
    """Replace the original expert with our SB3 PPO expert."""
    import metadrive.examples.ppo_expert as ppo_expert_module
    # Store the original expert for fallback
    ppo_expert_module.original_expert = ppo_expert_module.expert
    # Replace with our SB3 PPO expert
    ppo_expert_module.expert = sb3_ppo_expert
```

## Customization

You can customize the PPO policy by modifying the following parameters in `sb3_ppo_policy.py`:

- Network architecture: Change the `net_arch` parameter in `MetaDrivePPOPolicy`
- Learning rate: Change the `learning_rate` parameter in `MultiAgentPPOTrainer._create_new_model`
- Batch size: Change the `batch_size` parameter in `MultiAgentPPOTrainer._create_new_model`
- Number of epochs: Change the `n_epochs` parameter in `MultiAgentPPOTrainer._create_new_model`
- Gamma: Change the `gamma` parameter in `MultiAgentPPOTrainer._create_new_model`
- GAE lambda: Change the `gae_lambda` parameter in `MultiAgentPPOTrainer._create_new_model`
- Clip range: Change the `clip_range` parameter in `MultiAgentPPOTrainer._create_new_model`
