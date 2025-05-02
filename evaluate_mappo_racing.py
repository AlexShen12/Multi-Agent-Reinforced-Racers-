#!/usr/bin/env python
"""
Evaluation Script for MAPPO Racing Agents

This script loads trained MAPPO policies and evaluates them in the racing environment.
It visualizes the agents' performance and displays metrics.

Usage:
    python evaluate_mappo_racing.py --checkpoint PATH_TO_CHECKPOINT

"""

import argparse
import os
import ray
import numpy as np
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from metadrive.envs.marl_envs.rllib_mappo_env import RLLibMappoEnv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MAPPO agents in racing environment")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Render the environment")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    return parser.parse_args()

def env_creator(env_config):
    """Create the RLLibMappoEnv environment."""
    return RLLibMappoEnv(env_config)

def evaluate(args):
    """
    Evaluate trained MAPPO agents.

    Args:
        args: Command line arguments
    """
    # Initialize Ray
    ray.init()

    # Register the environment
    register_env("RLLibMappoEnv", env_creator)

    # Load the trained policies
    print(f"Loading checkpoint from {args.checkpoint}")
    trainer = PPO.from_checkpoint(args.checkpoint)

    # Create environment for evaluation
    env_config = {
        "use_render": args.render,
        "num_agents": 2,
        "enable_finish_line": True,
        "terminate_on_finish": False,
    }
    env = RLLibMappoEnv(env_config)

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    win_counts = {"agent0": 0, "agent1": 0, "tie": 0}

    print(f"\nRunning evaluation for {args.num_episodes} episodes...")

    for episode in range(args.num_episodes):
        print(f"Episode {episode+1}/{args.num_episodes}")

        # Reset environment
        obs, _ = env.reset(seed=args.seed + episode)
        done = {"__all__": False}
        episode_reward = {"agent0": 0, "agent1": 0}
        step = 0

        # Run episode
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
                episode_reward[agent_id] += reward

            # Render
            if args.render:
                env.render(
                    text={
                        "Race Progress": {agent_id: f"{progress:.2f}" for agent_id, progress in env.agent_progress.items()},
                        "Leading": next((agent_id for agent_id in env.agents.keys() if env.agent_progress.get(agent_id, 0) == max(env.agent_progress.values())), None),
                        "Race Winner": env.race_winner if env.race_finished else "None",
                        "Episode": f"{episode+1}/{args.num_episodes}",
                        "Step": step,
                        "Rewards": {agent_id: f"{r:.2f}" for agent_id, r in episode_reward.items()},
                    }
                )

            step += 1

        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        # Determine winner
        if env.race_finished and env.race_winner:
            win_counts[env.race_winner] += 1
        elif env.race_finished:
            # If race is finished but no winner (tie)
            win_counts["tie"] += 1

        print(f"Episode {episode+1} completed:")
        print(f"  Length: {step} steps")
        print(f"  Rewards: {episode_reward}")
        print(f"  Winner: {env.race_winner if env.race_finished else 'None'}")

    env.close()

    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} steps")
    print("Average Rewards:")
    for agent_id in ["agent0", "agent1"]:
        avg_reward = np.mean([ep[agent_id] for ep in episode_rewards])
        print(f"  {agent_id}: {avg_reward:.2f}")

    print("Win Counts:")
    for agent_id, count in win_counts.items():
        print(f"  {agent_id}: {count} ({count/args.num_episodes*100:.1f}%)")

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
