#!/usr/bin/env python

"""
IPPO (Independent PPO) Training Script for MetaDrive MARL Racing Environment

This script implements training for the MetaDrive multi-agent racing environment
using Ray RLlib's Independent PPO algorithm. It supports:
- Loading expert checkpoints for bootstrapping
- Customizable training parameters
- Regular checkpointing
- Resuming from checkpoints
- Shared or individual policies per agent
- TensorBoard logging
"""

import argparse
import os
import random
import time
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.algorithms.ippo import IPPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from torch.utils.tensorboard import SummaryWriter

from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import ray
        assert ray.__version__ >= "2.8.0", f"Ray version {ray.__version__} is too old. Please install Ray ≥ 2.8.0"
    except (ImportError, AssertionError) as e:
        raise ImportError(f"Ray ≥ 2.8.0 is required, error: {e}. "
                         f"Install with: pip install 'ray[rllib]>=2.8.0' torch gymnasium metadrive")
    
    try:
        import gymnasium
    except ImportError:
        raise ImportError("Gymnasium is required. Install with: pip install gymnasium")
    
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    try:
        import metadrive
    except ImportError:
        raise ImportError("MetaDrive is required. Install with: pip install -e .")


def register_env(env_name="marl_racing"):
    """Register the MetaDrive MARL racing environment."""
    from ray.tune.registry import register_env
    
    def env_creator(_):
        return MultiAgentRacingEnv({
            "use_render": False,
            "num_agents": 12,
            "horizon": 3000,
            "out_of_road_done": True,
            "idle_done": True,
            "crash_done": False,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "distance": 50,
                    "num_others": 0,
                    "gaussian_noise": 0.0,
                    "dropout_prob": 0.0,
                    "add_others_navi": False
                },
                "enable_reverse": False,
                "random_navi_mark_color": True,
            },
        })
    
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))
    return env_name


def seed_everything(seed=0):
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ray.init(configure_logging=False, include_dashboard=False, _system_config={"worker_cap": 1}, object_store_memory=10**9)
    return seed


def setup_and_train(args):
    """Setup training configuration and start the training process."""
    env_name = register_env()
    
    # Create temporary environment to get observation and action spaces
    env = MultiAgentRacingEnv({
        "use_render": args.render,
        "num_agents": 12,
    })
    obs_space = env.observation_space["agent0"]
    act_space = env.action_space["agent0"]
    env.close()
    
    # Setup policy mapping function
    if args.individual:
        def policy_mapping_fn(agent_id, *_):
            return f"policy_{agent_id}"
    else:
        def policy_mapping_fn(agent_id, *_):
            return "shared"
    
    # Configure IPPO
    config = (
        IPPOConfig()
        .environment(env_name)
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length=args.rollout_fragment,
        )
        .training(
            train_batch_size=args.train_batch,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=args.clip_param,
            lr=args.lr,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            model={"fcnet_hiddens": [256, 256]},
        )
        .resources(
            num_gpus=args.num_gpus,
        )
        .multi_agent(
            policies={
                "shared": (None, obs_space, act_space, {}),
            } if not args.individual else None,
            policy_mapping_fn=policy_mapping_fn,
        )
        .framework("torch")
        .debugging(seed=args.seed)
    )
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    tb_path = os.path.join(args.outdir, "tb")
    os.makedirs(tb_path, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(tb_path)
    
    # Initialize IPPO algorithm
    algo = config.build()
    
    # Load checkpoint if requested
    if args.resume:
        # Find most recent checkpoint
        checkpoints = [d for d in os.listdir(args.outdir) if d.startswith("checkpoint_")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("_")[1]))
            latest_checkpoint = os.path.join(args.outdir, checkpoints[-1])
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            algo.restore(latest_checkpoint)
            # Extract initial timestep from checkpoint name
            initial_timestep = int(checkpoints[-1].split("_")[1]) * algo.config.train_batch_size
        else:
            print("No checkpoints found for resuming. Starting fresh training.")
            initial_timestep = 0
    elif args.expert_checkpoint:
        if os.path.exists(args.expert_checkpoint):
            print(f"Loading expert checkpoint: {args.expert_checkpoint}")
            algo.restore(args.expert_checkpoint)
            initial_timestep = 0
        else:
            print(f"Expert checkpoint not found: {args.expert_checkpoint}")
            print("Starting with random initialization.")
            initial_timestep = 0
    else:
        initial_timestep = 0
    
    # Training loop
    timesteps = initial_timestep
    start_time = time.time()
    iteration = 0
    
    while timesteps < args.total_timesteps:
        # Train for one iteration
        result = algo.train()
        timesteps += result["timesteps_total"] - (result.get("timesteps_total_prev", 0) or initial_timestep)
        iteration += 1
        
        # Log metrics
        print(f"Iteration {iteration}, total steps: {timesteps}")
        print(f"Mean episode reward: {result['episode_reward_mean']}")
        
        # Log agent-specific metrics to TensorBoard
        policies = algo.get_policy_map()
        for policy_id, policy in policies.items():
            # Extract agent ID from policy ID if using individual policies
            agent_id = policy_id.replace("policy_", "") if args.individual else "shared"
            
            # Log policy metrics
            if f"policy_{policy_id}_reward" in result:
                reward = result[f"policy_{policy_id}_reward"]
                print(f"  {agent_id} reward: {reward}")
                writer.add_scalar(f"agent/{agent_id}/reward", reward, timesteps)
            
            # For shared policy, log overall metrics
            if policy_id == "shared" or args.individual:
                writer.add_scalar("train/episode_reward_mean", result["episode_reward_mean"], timesteps)
                writer.add_scalar("train/episode_length_mean", result["episode_len_mean"], timesteps)
        
        # Save checkpoint periodically
        if iteration % args.save_every_iters == 0:
            checkpoint_path = os.path.join(args.outdir, f"checkpoint_{iteration:06d}")
            print(f"Saving checkpoint to {checkpoint_path}")
            algo.save(checkpoint_path)
            
            # Save individual policy state dictionaries
            for policy_id, policy in policies.items():
                state_dict = policy.get_weights()
                torch.save(state_dict, os.path.join(args.outdir, f"{policy_id}.pth"))
    
    # Final save
    checkpoint_path = os.path.join(args.outdir, f"checkpoint_{iteration:06d}")
    print(f"Saving final checkpoint to {checkpoint_path}")
    algo.save(checkpoint_path)
    
    # Save final policy state dictionaries
    for policy_id, policy in policies.items():
        state_dict = policy.get_weights()
        torch.save(state_dict, os.path.join(args.outdir, f"{policy_id}.pth"))
    
    writer.close()
    algo.stop()
    
    # Print training summary
    total_time = time.time() - start_time
    print(f"Training complete: {iteration} iterations, {timesteps} timesteps")
    print(f"Total time: {total_time:.2f} seconds")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train IPPO agents in MARL Racing")
    
    # Environment options
    parser.add_argument("--render", action="store_true", help="Enable environment rendering")
    
    # Training options
    parser.add_argument("--total-timesteps", type=int, default=10000000, 
                        help="Total number of timesteps to train (default: 10M)")
    parser.add_argument("--expert-checkpoint", type=str, default=None,
                        help="Path to expert checkpoint for bootstrapping")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume training from latest checkpoint in outdir")
    parser.add_argument("--save-every-iters", type=int, default=5,
                        help="Save checkpoint every N iterations (default: 5)")
    parser.add_argument("--outdir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints (default: ./checkpoints)")
    parser.add_argument("--individual", action="store_true",
                        help="Use individual policies for each agent instead of shared policy")
    
    # Algorithm hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95, 
                        help="GAE lambda parameter (default: 0.95)")
    parser.add_argument("--clip-param", type=float, default=0.2, 
                        help="PPO clip parameter (default: 0.2)")
    parser.add_argument("--rollout-fragment", type=int, default=200,
                        help="Rollout fragment length (default: 200)")
    parser.add_argument("--train-batch", type=int, default=4000,
                        help="Train batch size (default: 4000)")
    
    # Resource allocation
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of rollout worker processes (default: 4)")
    parser.add_argument("--num-gpus", type=float, default=0,
                        help="Number of GPUs to use (default: 0)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_dependencies()
    seed_everything(args.seed)
    setup_and_train(args)
