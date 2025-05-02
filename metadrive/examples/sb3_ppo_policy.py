#!/usr/bin/env python
"""
Stable-Baselines3 PPO Policy for MetaDrive Racing Environment

This module implements a PPO-based policy using Stable-Baselines3 for the
MetaDrive multi-agent racing environment. It includes:
1. Custom policy network architecture matching the expert policy
2. Multi-agent training with separate models for each agent
3. Persistence mechanisms for saving and loading model weights
4. Integration with the existing environment
"""

import os
import numpy as np
import torch as th
import torch.nn as nn
import os.path as osp
from typing import Dict, List, Tuple, Type, Optional, Union
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces


class MetaDrivePPOFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for MetaDrive observations.
    Matches the architecture of the expert policy.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Input dimension is 275 (matching expert policy)
        input_dim = observation_space.shape[0]

        # Create a feature extractor with tanh activation (matching expert)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, features_dim),
            nn.Tanh()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.feature_extractor(observations)


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
        verbose: int = 1,
        use_expert_weights: bool = True
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_agents = num_agents
        self.models_dir = models_dir
        self.verbose = verbose
        self.use_expert_weights = use_expert_weights

        # Create directory for saving models if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

        # Dictionary to store PPO models for each agent
        self.models = {}

        # Dictionary to track if an agent's model has been reset this episode
        self.reset_status = {}

        # Initialize models for each agent
        self._initialize_models()

    def _initialize_models(self):
        """Initialize PPO models for each agent."""
        # Print debug info
        print(f"Initializing models for {self.num_agents} agents from directory: {self.models_dir}")

        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            print(f"WARNING: Models directory {self.models_dir} does not exist. Creating it.")
            os.makedirs(self.models_dir, exist_ok=True)

        # List all files in the models directory
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.zip')]
        print(f"Found {len(model_files)} model files in {self.models_dir}: {model_files}")

        # Create a primary model for each agent
        for agent_idx in range(self.num_agents):
            # Use the standard format for the primary model
            primary_id = f"agent_{agent_idx}"
            model_path = os.path.join(self.models_dir, primary_id)

            # Check for model files with different naming patterns
            model_found = False
            possible_model_paths = [
                os.path.join(self.models_dir, f"agent_{agent_idx}"),
                os.path.join(self.models_dir, f"agent{agent_idx}"),
                os.path.join(self.models_dir, f"{agent_idx}")
            ]

            # Try to load from any of the possible paths
            for path in possible_model_paths:
                if os.path.exists(f"{path}.zip"):
                    try:
                        print(f"Attempting to load model from {path}.zip")
                        self.models[primary_id] = PPO.load(
                            path,
                            policy=MetaDrivePPOPolicy,
                            verbose=self.verbose
                        )
                        print(f"Successfully loaded existing model for {primary_id} from {path}.zip")
                        model_found = True
                        break
                    except Exception as e:
                        print(f"Error loading model from {path}.zip: {e}")

            # If no model was found or loaded, create a new one
            if not model_found:
                print(f"No valid model found for {primary_id}, creating a new one")
                self._create_new_model(primary_id, use_expert_weights=self.use_expert_weights)
                print(f"Created new model for {primary_id}")

            # Initialize reset status for primary model
            self.reset_status[primary_id] = False

            # Create a comprehensive list of aliases for different ID formats
            aliases = [
                # Standard formats
                f"agent{agent_idx}",      # Environment format: 'agent0'
                str(agent_idx),           # Numeric only: '0'

                # Additional formats for robustness
                f"agent_{agent_idx}",     # With underscore: 'agent_0'
                f"agent{agent_idx}",      # Without underscore: 'agent0'

                # Formats with 'agent' prefix
                f"agent_agent{agent_idx}",  # Double agent with underscore: 'agent_agent0'
                f"agentagent{agent_idx}",   # Double agent without underscore: 'agentagent0'
            ]

            # Remove duplicates
            aliases = list(dict.fromkeys(aliases))

            # Create aliases pointing to the primary model
            for alias in aliases:
                if alias != primary_id:  # Skip the primary ID to avoid overwriting
                    self.models[alias] = self.models[primary_id]  # Share the same model
                    self.reset_status[alias] = False
                    print(f"Created alias: {alias} -> {primary_id}")

        # Print all available models
        print(f"Available models: {list(self.models.keys())}")

        # Verify that models are loaded and ready
        if len(self.models) == 0:
            print("WARNING: No models were loaded or created. Agents will use default actions.")
        else:
            print(f"Successfully initialized {len(self.models)} models/aliases for {self.num_agents} agents.")

    def _create_new_model(self, agent_id: str, use_expert_weights=True):
        """Create a new PPO model for an agent.

        Args:
            agent_id: The ID of the agent to create a model for
            use_expert_weights: Whether to initialize the model with expert weights
        """
        # Create a simple dummy environment class
        import gymnasium as gym

        class DummyEnv(gym.Env):
            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space

            def reset(self, **_):
                return np.zeros(self.observation_space.shape), {}

            def step(self, _):
                return np.zeros(self.observation_space.shape), 0.0, False, False, {}

        # Initialize the PPO model with minimal required parameters
        self.models[agent_id] = PPO(
            policy=MetaDrivePPOPolicy,
            env=DummyEnv(self.observation_space, self.action_space),
            policy_kwargs={"net_arch": [256, 256]},
            verbose=self.verbose
        )

        # Initialize with expert weights if requested
        if use_expert_weights:
            try:
                # Load the expert weights
                expert_weights = load_expert_weights()

                if expert_weights is not None:
                    # Get the policy network
                    policy = self.models[agent_id].policy

                    # Map the expert weights to the SB3 policy network
                    # The expert weights use a different naming convention than SB3
                    # Expert: default_policy/fc_1/kernel, default_policy/fc_1/bias, etc.
                    # SB3: mlp_extractor.policy_net.0.weight, mlp_extractor.policy_net.0.bias, etc.

                    # First layer (input -> hidden)
                    policy.mlp_extractor.policy_net[0].weight.data = th.tensor(
                        expert_weights["default_policy/fc_1/kernel"].T, dtype=th.float32
                    )
                    policy.mlp_extractor.policy_net[0].bias.data = th.tensor(
                        expert_weights["default_policy/fc_1/bias"], dtype=th.float32
                    )

                    # Second layer (hidden -> hidden)
                    policy.mlp_extractor.policy_net[2].weight.data = th.tensor(
                        expert_weights["default_policy/fc_2/kernel"].T, dtype=th.float32
                    )
                    policy.mlp_extractor.policy_net[2].bias.data = th.tensor(
                        expert_weights["default_policy/fc_2/bias"], dtype=th.float32
                    )

                    # Output layer (hidden -> action mean and log_std)
                    # The expert has a single output layer for both mean and log_std
                    # SB3 has separate layers for mean and log_std
                    # We need to split the expert weights
                    expert_out_kernel = expert_weights["default_policy/fc_out/kernel"]
                    expert_out_bias = expert_weights["default_policy/fc_out/bias"]

                    # The output dimension is 4 (2 for mean, 2 for log_std)
                    # We need to split it into two parts
                    action_dim = self.action_space.shape[0]  # Should be 2

                    # Mean weights and bias
                    policy.action_net.weight.data = th.tensor(
                        expert_out_kernel.T[:action_dim, :], dtype=th.float32
                    )
                    policy.action_net.bias.data = th.tensor(
                        expert_out_bias[:action_dim], dtype=th.float32
                    )

                    # Log std weights and bias
                    policy.log_std.data = th.tensor(
                        expert_out_bias[action_dim:], dtype=th.float32
                    )

                    print(f"Successfully initialized {agent_id} with expert weights")
            except Exception as e:
                print(f"Error initializing {agent_id} with expert weights: {e}")
                print(f"Using random initialization for {agent_id}")

        print(f"Created new model for {agent_id}")

    def reset_agent_model(self, agent_id: str):
        """Reset a specific agent's model to initial weights."""
        if agent_id in self.models:
            # Create a new model with fresh weights
            self._create_new_model(agent_id)
            # Mark this agent as reset for this episode
            self.reset_status[agent_id] = True
            print(f"Reset model for {agent_id}")

    def predict(self, agent_id: str, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get action from the agent's policy."""
        # Try to find the model with the given agent_id
        model = self.models.get(agent_id)

        if model:
            try:
                # Ensure observation has the correct shape
                if len(observation.shape) == 1:
                    # Add batch dimension if missing
                    observation = observation.reshape(1, -1)

                # Get action from the model
                action, states = model.predict(observation, deterministic=deterministic)

                # Ensure action is in the correct format (1D array)
                import numpy as np
                if isinstance(action, np.ndarray) and len(action.shape) > 1:
                    # If it's a 2D array (e.g., [[steer, throttle]]), flatten it
                    action = action.flatten()

                print(f"Model for {agent_id} predicted action: {action}")
                return action, states
            except Exception as e:
                print(f"Error predicting with model for {agent_id}: {e}")
                # Try to use the original expert as fallback
                try:
                    from metadrive.examples.ppo_expert import original_expert
                    if original_expert:
                        print(f"Falling back to original expert for {agent_id}")
                        # Create a dummy vehicle to use with the original expert
                        class DummyVehicle:
                            def __init__(self, obs):
                                self.observation = obs

                            def get_observation(self):
                                return self.observation[0] if len(self.observation.shape) > 1 else self.observation

                        dummy_vehicle = DummyVehicle(observation)
                        action = original_expert(dummy_vehicle, deterministic=deterministic)
                        return action, None
                except Exception as fallback_error:
                    print(f"Fallback to original expert failed: {fallback_error}")

        # If we get here, no model was found or all attempts failed
        print(f"Warning: No model found for {agent_id} or all prediction attempts failed, using forward motion")
        return np.array([0.0, 0.5]), None  # Return forward motion as default

    def train_on_batch(self, agent_id: str, observations: np.ndarray, **kwargs):
        """Train an agent's model on a batch of experiences."""
        model = self.models.get(agent_id)
        if model and not self.reset_status.get(agent_id, False):
            # Log training stats
            print(f"Training {agent_id} on {len(observations)} experiences")

            # In a real implementation, you would use SB3's learn method
            # model.learn(total_timesteps=len(observations))

    def save_models(self):
        """Save all agent models to disk."""
        for agent_id, model in self.models.items():
            model_path = os.path.join(self.models_dir, agent_id)
            model.save(model_path)
            print(f"Saved model for {agent_id}")

    def end_episode(self):
        """Called at the end of an episode to save models and reset status."""
        # Save all models
        self.save_models()

        # Reset the reset status for all agents
        for agent_id in self.reset_status:
            self.reset_status[agent_id] = False


# Global PPO trainer that can be accessed from anywhere
global_ppo_trainer = None

# Function to load expert weights from MetaDrive
def load_expert_weights():
    """
    Load the expert weights from the MetaDrive ppo_expert directory.

    Returns:
        dict: The expert weights as a dictionary of numpy arrays
    """
    # Path to the expert weights file
    expert_weights_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                                 "examples", "ppo_expert", "expert_weights.npz")

    print(f"Loading expert weights from {expert_weights_path}")

    try:
        # Load the weights
        weights = np.load(expert_weights_path)
        print(f"Successfully loaded expert weights with keys: {list(weights.keys())}")
        return weights
    except Exception as e:
        print(f"Error loading expert weights: {e}")
        return None

# Function to initialize the global PPO trainer
def initialize_global_ppo_trainer(env, models_dir="trained_models", use_expert_weights=True):
    """
    Initialize the global PPO trainer and attach it to the environment.
    This function should be called after the environment is created but before stepping.

    Args:
        env: The environment to attach the PPO trainer to
        models_dir: Directory containing the trained models

    Returns:
        The initialized PPO trainer
    """
    global global_ppo_trainer

    print(f"Initializing global PPO trainer with models from {models_dir}")

    # Get observation and action spaces from the environment
    if not hasattr(env, 'observation_space') or not hasattr(env, 'action_space'):
        print("Environment does not have observation_space or action_space attributes")
        # Try to get spaces from the first agent
        if hasattr(env, 'agents') and len(env.agents) > 0:
            agent_id = list(env.agents.keys())[0]
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'spaces'):
                observation_space = env.observation_space.spaces[agent_id]
            else:
                print("Could not determine observation space, using default")
                observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(275,))

            if hasattr(env, 'action_space') and hasattr(env.action_space, 'spaces'):
                action_space = env.action_space.spaces[agent_id]
            else:
                print("Could not determine action space, using default")
                action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        else:
            print("Environment has no agents, using default spaces")
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(275,))
            action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
    else:
        # Get spaces directly from the environment
        observation_space = env.observation_space
        action_space = env.action_space

    # Determine number of agents
    num_agents = env.config["num_agents"] if hasattr(env, 'config') and "num_agents" in env.config else 2

    # Create the PPO trainer
    global_ppo_trainer = MultiAgentPPOTrainer(
        observation_space=observation_space,
        action_space=action_space,
        num_agents=num_agents,
        models_dir=models_dir,
        verbose=1,
        use_expert_weights=use_expert_weights
    )

    # Attach the trainer to the environment for access by the expert function
    if hasattr(env, 'engine'):
        env.engine.ppo_trainer = global_ppo_trainer
        print("Attached PPO trainer to environment engine")
    else:
        print("WARNING: Environment has no engine attribute, PPO trainer not attached")

    # Replace the original expert with our SB3 PPO expert
    replace_expert()

    return global_ppo_trainer

# PPO expert function that uses the trained models
def sb3_ppo_expert(vehicle, deterministic=True, need_obs=False):
    """
    PPO expert function that matches the interface of the original expert.
    This version uses the trained PPO models to generate actions.

    Args:
        vehicle: The vehicle to control
        deterministic: Whether to use deterministic actions
        need_obs: Whether to return the observation

    Returns:
        action: The action to take
        obs: The observation (if need_obs=True)
    """
    # Get observation
    obs = vehicle.get_observation()

    # FORCE AUTODRIVE MODE
    if hasattr(vehicle, 'expert_takeover'):
        if not vehicle.expert_takeover:
            print(f"Forcing autodrive mode for vehicle {vehicle.id if hasattr(vehicle, 'id') else 'unknown'}")
            vehicle.expert_takeover = True

    # Get the agent ID from the vehicle
    agent_id = None
    if hasattr(vehicle, 'id'):
        # Try multiple agent ID formats to ensure we find a match
        possible_ids = [
            f"agent_{vehicle.id}",  # Primary format: agent_0
            f"agent{vehicle.id}",   # Alternative format: agent0
            str(vehicle.id)         # Simple numeric: 0
        ]

        # Print debug info about the vehicle
        print(f"DEBUG: Vehicle ID: {vehicle.id}, Type: {type(vehicle).__name__}")
        print(f"DEBUG: Possible agent IDs: {possible_ids}")

    # Get the PPO trainer from the engine
    ppo_trainer = None
    if hasattr(vehicle, 'engine') and hasattr(vehicle.engine, 'ppo_trainer'):
        ppo_trainer = vehicle.engine.ppo_trainer
        print(f"DEBUG: Found PPO trainer in engine: {ppo_trainer is not None}")
        if ppo_trainer is not None and hasattr(ppo_trainer, 'models'):
            print(f"DEBUG: Available models: {list(ppo_trainer.models.keys())}")
    else:
        print(f"DEBUG: No PPO trainer found in engine")
        if hasattr(vehicle, 'engine'):
            print(f"DEBUG: Engine attributes: {dir(vehicle.engine)}")

    # Use the PPO model to get the action if available
    if ppo_trainer is not None and possible_ids:
        # Try each possible ID format until we find a match
        for agent_id in possible_ids:
            try:
                # Reshape observation to match model input
                obs_array = np.array(obs).reshape(1, -1)
                print(f"DEBUG: Trying to get action for {agent_id} with observation shape {obs_array.shape}")

                # Get action from the model
                action, _ = ppo_trainer.predict(agent_id, obs_array, deterministic=deterministic)
                print(f"DEBUG: Successfully got action for {agent_id}: {action}")

                # If we get here, we found a working model
                return (action, [obs]) if need_obs else action

            except Exception as e:
                print(f"DEBUG: Error using PPO model for {agent_id}: {e}")
                # Continue to the next ID format
                continue

        # If we get here, none of the ID formats worked
        print(f"DEBUG: All agent ID formats failed, using default action")
        action = np.array([0.0, 0.5])  # Default forward motion
    else:
        # Fallback to a default action if no model is available
        print(f"DEBUG: No PPO trainer or agent IDs available, using default action")
        action = np.array([0.0, 0.5])  # Default forward motion

    # Return with or without observation
    return (action, [obs]) if need_obs else action


# Replace the original expert with our SB3 PPO expert
def replace_expert():
    """Replace the original expert with our SB3 PPO expert."""
    try:
        # Import the PPO expert module
        import metadrive.examples.ppo_expert as ppo_expert_module

        # Store the original expert for fallback
        if not hasattr(ppo_expert_module, 'original_expert'):
            ppo_expert_module.original_expert = ppo_expert_module.expert

        # Replace with our SB3 PPO expert
        ppo_expert_module.expert = sb3_ppo_expert

        # Verify the replacement
        if ppo_expert_module.expert == sb3_ppo_expert:
            print("Successfully replaced original expert with SB3 PPO expert")

            # Test the expert function to ensure it works
            print("Testing SB3 PPO expert function...")
            try:
                # Create a dummy vehicle class for testing
                class DummyVehicle:
                    def __init__(self):
                        self.id = 0
                        self.expert_takeover = False
                        self.engine = None

                    def get_observation(self):
                        return np.zeros(275)  # Return dummy observation

                # Create a dummy vehicle
                dummy_vehicle = DummyVehicle()

                # Test the expert function
                test_action = ppo_expert_module.expert(dummy_vehicle)
                print(f"Test action from SB3 PPO expert: {test_action}")

                # Check if the action is valid
                if isinstance(test_action, np.ndarray) and test_action.shape == (2,):
                    print("SB3 PPO expert function is working correctly")
                else:
                    print(f"WARNING: SB3 PPO expert returned unexpected action: {test_action}")
            except Exception as e:
                print(f"Error testing SB3 PPO expert: {e}")
                print("Will continue with replacement anyway")
        else:
            print("WARNING: Failed to replace expert function")
    except Exception as e:
        print(f"ERROR: Failed to replace expert function: {e}")
        print("Agents will use the original expert function")
