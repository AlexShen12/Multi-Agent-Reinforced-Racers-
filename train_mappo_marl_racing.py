#!/usr/bin/env python
"""
Multi-Agent Proximal Policy Optimization (MAPPO) Training Script for MARL Racing Environment

This script implements MAPPO training for the MultiAgentRacingSafeEnv environment.
It handles:
1. Environment registration and configuration
2. Policy setup for multiple agents
3. MAPPO trainer configuration
4. Training loop with checkpointing
5. Evaluation of trained policies

Usage:
    python train_mappo_marl_racing.py [--resume CHECKPOINT_PATH]

    Options:
        --env ENV_NAME          Environment name (default: marl_race_env)
        --num-workers N         Number of environment runner processes (default: 4)
        --num-gpus N            Number of GPUs to use (default: 0)
        --iterations N          Number of training iterations (default: 1000)
        --checkpoint-dir DIR    Directory to save checkpoints (default: ~/ray_results/mappo_racing)
        --checkpoint-freq N     Frequency to save checkpoints (default: 10)
        --resume PATH           Path to checkpoint to resume training from
        --eval                  Run evaluation after training

"""

import argparse
import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from marl_racing_env_wrapper import MARLRacingEnvWrapper

# Define constants
CHECKPOINT_BASE_DIR = os.path.expanduser("~/ray_results/mappo_racing")
MAX_ITERATIONS = 1000

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MAPPO agents in MARL racing environment")
    parser.add_argument("--env", type=str, default="marl_race_env",
                        help="Environment name for registration")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs to use for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of environment runner processes to use")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS,
                        help="Number of training iterations to run")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_BASE_DIR,
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Frequency at which to save checkpoints (iterations)")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation after training")
    parser.add_argument("--traffic-density", type=float, default=0.0,
                        help="Traffic density for the environment (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for environment")
    return parser.parse_args()

def env_creator(env_config):
    """Create the MARL Racing environment."""
    # Add a unique seed to each environment instance to avoid conflicts
    import os
    import time
    import numpy as np

    # Generate a unique seed for each environment instance
    pid = os.getpid()
    timestamp = int(time.time() * 1000) % 10000
    random_num = np.random.randint(0, 10000)
    unique_seed = (pid + timestamp + random_num) % 2**31

    # Update the environment config with the unique seed
    if env_config is None:
        env_config = {}
    env_config = env_config.copy()  # Make a copy to avoid modifying the original

    if "start_seed" in env_config:
        env_config["start_seed"] = env_config["start_seed"] + unique_seed
    else:
        env_config["start_seed"] = unique_seed

    # Create the environment with the updated config
    return MARLRacingEnvWrapper(env_config)

def get_mappo_config(args):
    """
    Configure the MAPPO trainer.

    Args:
        args: Command line arguments

    Returns:
        PPOConfig object with MAPPO settings
    """
    # Environment configuration
    env_config = {
        "num_agents": 2,
        "enable_finish_line": True,
        "terminate_on_finish": False,
        "horizon": 2000,
        "traffic_density": args.traffic_density,
        "start_seed": args.seed,

        # Disable termination conditions for continuous training
        "crash_done": False,
        "out_of_road_done": False,
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "out_of_route_done": False,
        "on_continuous_line_done": False,
        "on_broken_line_done": False,
        "truncate_as_terminate": False,

        # Racing-specific settings
        "leading_reward_factor": 1.0,  # Reward for being in the lead
        "winning_reward": 10.0,        # Reward for winning the race
    }

    # Create a sample environment to get observation and action spaces
    sample_env = MARLRacingEnvWrapper(env_config)
    obs_space = sample_env.observation_spaces["agent0"]
    act_space = sample_env.action_spaces["agent0"]
    sample_env.close()

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
        # Environment configuration
        .environment(args.env, env_config=env_config)
        .framework("torch")
        .resources(
            num_gpus=args.num_gpus,
        )
        # Environment runner configuration
        .env_runners(
            num_env_runners=args.num_workers,
            num_cpus_per_env_runner=1,
            rollout_fragment_length=200,
            remote_worker_envs=True,  # Create environments in remote workers
            remote_env_batch_wait_ms=0,  # Don't wait for environments to be ready
            recreate_failed_workers=True,  # Recreate workers if they fail
        )
        # Training configuration
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
        )
        # Debugging configuration
        .debugging(
            log_level="INFO",
        )
        # Evaluation configuration
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=5,
            evaluation_config={
                "render_env": True,
            },
        )
        # Multi-agent configuration
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
    )

    return config

def train(args):
    """
    Train MAPPO agents in the MARL racing environment.

    Args:
        args: Command line arguments
    """
    # Initialize Ray with appropriate resources
    ray.init(
        num_cpus=args.num_workers + 2,  # Add extra CPUs for the driver and trainer
        ignore_reinit_error=True,
        include_dashboard=False,  # Disable dashboard for better performance
        _system_config={
            "automatic_object_spilling_enabled": False,  # Disable object spilling for better performance
        }
    )

    # Register the environment
    register_env(args.env, env_creator)

    # Get the MAPPO configuration
    config = get_mappo_config(args)

    # Create the trainer
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer = config.build(checkpoint_path=args.resume)
    else:
        trainer = config.build()

    # Training loop
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting training for {args.iterations} iterations...")
    print(f"Environment: {args.env}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Traffic density: {args.traffic_density}")
    print(f"Seed: {args.seed}")

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

        # Print agent-specific rewards if available
        if "policy_reward_mean" in result:
            print("Policy Rewards:")
            for policy_id, reward in result["policy_reward_mean"].items():
                print(f"  {policy_id}: {reward:.2f}")

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
        evaluate(trainer, args)

    # Shutdown Ray
    ray.shutdown()

    return final_checkpoint_path

def evaluate(trainer, args, num_episodes=5):
    """
    Evaluate the trained policies.

    Args:
        trainer: The trained PPO trainer
        args: Command line arguments
        num_episodes: Number of episodes to evaluate
    """
    print(f"\nRunning evaluation for {num_episodes} episodes...")

    # Create environment for evaluation with rendering enabled
    env_config = {
        "num_agents": 2,
        "enable_finish_line": True,
        "terminate_on_finish": False,
        "use_render": True,
        "traffic_density": args.traffic_density,
        "start_seed": args.seed,
    }

    env = MARLRacingEnvWrapper(env_config)

    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")

        # Reset environment
        obs, _ = env.reset()
        done = {"__all__": False}
        episode_rewards = {agent_id: 0 for agent_id in env.possible_agents}

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

            # Render with race information
            try:
                env.render(
                    text={
                        "Race Progress": {agent_id: f"{progress:.2f}" for agent_id, progress in env.agent_progress.items()},
                        "Leading": next((agent_id for agent_id in env.agents if agent_id in env.agent_progress and
                                        env.agent_progress.get(agent_id, 0) == max(env.agent_progress.values())), None),
                        "Race Winner": env.race_winner if env.race_finished else "None",
                        "Rewards": {agent_id: f"{r:.2f}" for agent_id, r in episode_rewards.items()},
                        "Episode": f"{episode+1}/{num_episodes}",
                        "Step": step,
                    }
                )
            except Exception as e:
                print(f"Error during rendering: {e}")

            step += 1

        print(f"Episode {episode+1} completed with rewards: {episode_rewards}")

    env.close()

if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = train(args)
    print(f"Training completed. Final checkpoint: {checkpoint_path}")
