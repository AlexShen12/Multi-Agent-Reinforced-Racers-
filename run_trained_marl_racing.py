#!/usr/bin/env python
"""
Run Trained MAPPO Racing Agents

This script loads trained MAPPO policies and runs them in the MARL racing environment.

Usage:
    python run_trained_marl_racing.py --checkpoint PATH_TO_CHECKPOINT

"""

import argparse
import os
import ray
import time
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from marl_racing_env_wrapper import MARLRacingEnvWrapper

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run trained MAPPO agents in MARL racing environment")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--traffic-density", type=float, default=0.15,
                        help="Traffic density (0.0 to 1.0)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between steps in seconds (for visualization)")
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

def run_agents(args):
    """
    Run trained MAPPO agents in the MARL racing environment.

    Args:
        args: Command line arguments
    """
    # Initialize Ray
    ray.init()

    # Register the environment
    register_env("marl_race_env", env_creator)

    # Load the trained policies
    print(f"Loading checkpoint from {args.checkpoint}")
    trainer = PPO.from_checkpoint(args.checkpoint)

    # Create environment for running
    env_config = {
        "num_agents": 2,
        "enable_finish_line": True,
        "terminate_on_finish": False,
        "use_render": True,
        "traffic_density": args.traffic_density,
        "start_seed": args.seed,
        "horizon": 2000,

        # Disable termination conditions for continuous simulation
        "crash_done": False,
        "out_of_road_done": False,
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "out_of_route_done": False,
        "on_continuous_line_done": False,
        "on_broken_line_done": False,
        "truncate_as_terminate": False,
    }

    env = MARLRacingEnvWrapper(env_config)

    print(f"\nRunning agents for {args.num_episodes} episodes...")

    for episode in range(args.num_episodes):
        print(f"Episode {episode+1}/{args.num_episodes}")

        # Reset environment
        obs, _ = env.reset(seed=args.seed + episode)
        done = {"__all__": False}
        episode_rewards = {agent_id: 0 for agent_id in env.possible_agents}
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
                        "Episode": f"{episode+1}/{args.num_episodes}",
                        "Step": step,
                    }
                )
            except Exception as e:
                print(f"Error during rendering: {e}")

            # Optional delay for visualization
            if args.delay > 0:
                time.sleep(args.delay)

            step += 1

            # Check if race is finished
            if hasattr(env, 'race_finished') and env.race_finished:
                print(f"Race finished! Winner: {env.race_winner}")
                if args.delay < 1.0:  # Add a short pause to see the winner
                    time.sleep(1.0)

        print(f"Episode {episode+1} completed in {step} steps")
        print(f"Rewards: {episode_rewards}")
        print(f"Winner: {env.race_winner if hasattr(env, 'race_finished') and env.race_finished else 'None'}")

        # Short pause between episodes
        time.sleep(1.0)

    env.close()

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    args = parse_args()
    run_agents(args)
