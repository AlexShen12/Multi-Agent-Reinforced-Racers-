#!/usr/bin/env python
"""
Run Trained MAPPO Racing Agents

This script loads trained MAPPO policies and runs them in the racing environment
without evaluation metrics, just for visualization.

Usage:
    python run_trained_agents.py --checkpoint PATH_TO_CHECKPOINT

"""

import argparse
import os
import ray
import time
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from metadrive.envs.marl_envs.rllib_mappo_env import RLLibMappoEnv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run trained MAPPO agents in racing environment")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between steps in seconds (for visualization)")
    return parser.parse_args()

def env_creator(env_config):
    """Create the RLLibMappoEnv environment."""
    return RLLibMappoEnv(env_config)

def run_agents(args):
    """
    Run trained MAPPO agents in the racing environment.

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

    # Create environment for running
    env_config = {
        "use_render": True,
        "num_agents": 2,
        "enable_finish_line": True,
        "terminate_on_finish": False,
        # Optional: customize map
        # "map": "CSRCR",  # Use a circular map for racing
    }
    env = RLLibMappoEnv(env_config)

    print(f"\nRunning agents for {args.num_episodes} episodes...")

    for episode in range(args.num_episodes):
        print(f"Episode {episode+1}/{args.num_episodes}")

        # Reset environment
        obs, _ = env.reset(seed=args.seed + episode)
        done = {"__all__": False}
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

            # Render
            env.render(
                text={
                    "Race Progress": {agent_id: f"{progress:.2f}" for agent_id, progress in env.agent_progress.items()},
                    "Leading": next((agent_id for agent_id in env.agents.keys() if env.agent_progress.get(agent_id, 0) == max(env.agent_progress.values())), None),
                    "Race Winner": env.race_winner if env.race_finished else "None",
                    "Episode": f"{episode+1}/{args.num_episodes}",
                    "Step": step,
                }
            )

            # Optional delay for visualization
            if args.delay > 0:
                time.sleep(args.delay)

            step += 1

        print(f"Episode {episode+1} completed in {step} steps")
        print(f"Winner: {env.race_winner if env.race_finished else 'None'}")

        # Short pause between episodes
        time.sleep(1.0)

    env.close()

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    args = parse_args()
    run_agents(args)
