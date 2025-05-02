#!/usr/bin/env python
"""
Multi-Agent Proximal Policy Optimization (MAPPO) Training Script for Racing Environment

This script implements MAPPO training for the RLLibMappoEnv racing environment.
It handles:
1. Environment registration and configuration
2. Policy setup for multiple agents
3. MAPPO trainer configuration
4. Training loop with checkpointing
5. Evaluation of trained policies

Usage:
    python train_mappo_racing.py [--resume CHECKPOINT_PATH]

"""

import argparse
import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from rllib_mappo_env_wrapper import RLLibMappoEnvWrapper

# Define constants
CHECKPOINT_BASE_DIR = os.path.expanduser("~/ray_results/mappo_racing")
MAX_ITERATIONS = 1000

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MAPPO agents in racing environment")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs to use for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of environment runner processes to use")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS,
                        help="Number of training iterations to run")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Frequency at which to save checkpoints (iterations)")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation after training")
    return parser.parse_args()

def env_creator(env_config):
    """Create the RLLibMappoEnv environment."""
    return RLLibMappoEnvWrapper(env_config)

def get_mappo_config(args):
    """
    Configure the MAPPO trainer.

    Args:
        args: Command line arguments

    Returns:
        PPOConfig object with MAPPO settings
    """
    # Create a sample environment to get observation and action spaces
    sample_env = RLLibMappoEnvWrapper({})
    obs_space = sample_env.observation_spaces["agent0"]
    act_space = sample_env.action_spaces["agent0"]

    # Define the policy network architecture
    model_config = {
        "fcnet_hiddens": [256, 256],  # Two hidden layers with 256 units each
        "fcnet_activation": "tanh",   # Tanh activation function
        "vf_share_layers": False,     # Separate value function network
    }

    # Create the MAPPO configuration
    config = (
        PPOConfig()
        # Disable the new API stack to avoid compatibility issues
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(RLLibMappoEnvWrapper, env_config={
            "num_agents": 2,
            "enable_finish_line": True,
            "terminate_on_finish": False,
        })
        .framework("torch")
        .resources(
            num_gpus=args.num_gpus,
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_cpus_per_env_runner=1,
            rollout_fragment_length=200,
        )

        .training(
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            use_gae=True,
            model=model_config,
            # MAPPO-specific settings
            # Note: In newer RLlib versions, centralized critic is configured in the model
        )
        .debugging(
            log_level="INFO",
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=5,
            evaluation_config={
                "render_env": True,
            },
        )
        .multi_agent(
            policies={
                "agent0": PolicySpec(
                    policy_class=None,  # use default policy
                    observation_space=obs_space,
                    action_space=act_space,
                    config={}
                ),
                "agent1": PolicySpec(
                    policy_class=None,  # use default policy
                    observation_space=obs_space,
                    action_space=act_space,
                    config={}
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            # Optional: set policies to train
            # policies_to_train=["agent0", "agent1"],
        )
        # Configure checkpointing in the training loop instead
    )

    return config

def train(args):
    """
    Train MAPPO agents in the racing environment.

    Args:
        args: Command line arguments
    """
    # Initialize Ray
    ray.init()

    # Register the environment
    register_env("RLLibMappoEnv", env_creator)

    # Get the MAPPO configuration
    config = get_mappo_config(args)

    # Create the trainer
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer = config.build(checkpoint_path=args.resume)
    else:
        trainer = config.build()

    # Training loop
    checkpoint_dir = CHECKPOINT_BASE_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting training for {args.iterations} iterations...")
    best_reward = float("-inf")

    for i in range(args.iterations):
        # Train for one iteration
        result = trainer.train()

        # Log metrics
        episode_reward_mean = result.get("episode_reward_mean", 0.0)
        training_time = result.get("time_total_s", 0.0)

        # Print available metrics
        print(f"Iteration {i+1}/{args.iterations}")
        print(f"Available metrics: {list(result.keys())}")
        print(f"Episode Reward Mean: {episode_reward_mean:.2f}")
        print(f"Training Time: {training_time:.2f}s")

        # Save checkpoint if improved
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            checkpoint_path = trainer.save(checkpoint_dir)
            print(f"New best model saved to {checkpoint_path}")

        # Save periodic checkpoint based on frequency
        if (i + 1) % args.checkpoint_freq == 0:
            checkpoint_path = trainer.save(checkpoint_dir)
            print(f"Periodic checkpoint saved to {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = trainer.save(checkpoint_dir)
    print(f"Final checkpoint saved to {final_checkpoint_path}")

    # Run evaluation if requested
    if args.eval:
        evaluate(trainer)

    # Shutdown Ray
    ray.shutdown()

    return final_checkpoint_path

def evaluate(trainer, num_episodes=5):
    """
    Evaluate the trained policies.

    Args:
        trainer: The trained PPO trainer
        num_episodes: Number of episodes to evaluate
    """
    print(f"\nRunning evaluation for {num_episodes} episodes...")

    # Create environment for evaluation
    env = RLLibMappoEnvWrapper({"use_render": True})

    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")

        # Reset environment
        obs, _ = env.reset()
        done = {"__all__": False}
        episode_rewards = {"agent0": 0, "agent1": 0}

        # Run episode
        step = 0
        while not done["__all__"]:
            # Get actions from policies
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = trainer.compute_single_action(
                    agent_obs, policy_id=agent_id
                )

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Update done status
            done = {"__all__": all(terminated.values()) or all(truncated.values())}

            # Update rewards
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward

            # Render
            env.render(
                text={
                    "Race Progress": {agent_id: f"{progress:.2f}" for agent_id, progress in env.agent_progress.items()},
                    "Leading": next((agent_id for agent_id in env.agents.keys() if env.agent_progress.get(agent_id, 0) == max(env.agent_progress.values())), None),
                    "Race Winner": env.race_winner if env.race_finished else "None",
                    "Rewards": {agent_id: f"{r:.2f}" for agent_id, r in episode_rewards.items()},
                }
            )

            step += 1

        print(f"Episode {episode+1} completed with rewards: {episode_rewards}")

    env.close()

if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = train(args)
    print(f"Training completed. Final checkpoint: {checkpoint_path}")
