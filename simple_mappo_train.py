#!/usr/bin/env python
"""
Simplified MAPPO Training Script for Racing Environment
"""

import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from metadrive.envs.marl_envs.rllib_mappo_env import RLLibMappoEnv

def env_creator(env_config):
    """Create the RLLibMappoEnv environment."""
    return RLLibMappoEnv(env_config)

def main():
    # Initialize Ray
    ray.init()
    
    # Register the environment
    register_env("RLLibMappoEnv", env_creator)
    
    # Create a sample environment to get observation and action spaces
    sample_env = RLLibMappoEnv({})
    obs_space = sample_env.observation_spaces["agent0"]
    act_space = sample_env.action_spaces["agent0"]
    
    # Define the policy network architecture
    model_config = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
    }
    
    # Create the MAPPO configuration
    config = (
        PPOConfig()
        .environment("RLLibMappoEnv", env_config={
            "num_agents": 2,
            "enable_finish_line": True,
            "terminate_on_finish": False,
        })
        .framework("torch")
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=1,
            num_cpus_per_env_runner=1,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=1000,
            minibatch_size=100,
            num_sgd_iter=5,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            use_gae=True,
            model=model_config,
        )
        .multi_agent(
            policies={
                "agent0": PolicySpec(
                    policy_class=None,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={}
                ),
                "agent1": PolicySpec(
                    policy_class=None,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={}
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
    )
    
    # Create the trainer
    trainer = config.build()
    
    # Train for one iteration
    print("Starting training...")
    result = trainer.train()
    
    # Print results
    print(f"Training completed. Episode reward mean: {result['episode_reward_mean']:.2f}")
    
    # Save checkpoint
    checkpoint_dir = os.path.expanduser("~/ray_results/mappo_racing_simple")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = trainer.save(checkpoint_dir)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()
