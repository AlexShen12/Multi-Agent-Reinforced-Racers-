#!/usr/bin/env python
"""
Visualize MAPPO Training Progress

This script loads training progress data from a checkpoint directory
and visualizes key metrics over time.

Usage:
    python visualize_training.py --logdir PATH_TO_CHECKPOINT_DIR

Author: Augment Agent
"""

import argparse
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize MAPPO training progress")
    parser.add_argument("--logdir", type=str, required=True,
                        help="Path to the training log directory")
    parser.add_argument("--save", action="store_true",
                        help="Save the plots instead of displaying them")
    parser.add_argument("--output-dir", type=str, default="./plots",
                        help="Directory to save plots (if --save is used)")
    return parser.parse_args()

def load_progress_data(log_dir):
    """
    Load training progress data from log files.
    
    Args:
        log_dir: Path to the log directory
        
    Returns:
        Dictionary of training metrics
    """
    # Find all progress.json files
    progress_files = glob.glob(os.path.join(log_dir, "**", "progress.json"), recursive=True)
    
    if not progress_files:
        raise ValueError(f"No progress.json files found in {log_dir}")
    
    # Use the most recent progress file
    progress_file = sorted(progress_files, key=os.path.getmtime)[-1]
    print(f"Loading training data from {progress_file}")
    
    # Load the data
    data = []
    with open(progress_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    # Extract metrics
    metrics = {
        "iterations": [],
        "episode_reward_mean": [],
        "episode_reward_min": [],
        "episode_reward_max": [],
        "episode_len_mean": [],
        "policy_reward_mean": {},
        "policy_loss": {},
        "vf_loss": {},
        "entropy": {},
        "kl": [],
        "time_total_s": [],
    }
    
    # Check if we have per-policy metrics
    if data and "policy_reward_mean" in data[0]:
        for policy_id in data[0]["policy_reward_mean"].keys():
            metrics["policy_reward_mean"][policy_id] = []
            metrics["policy_loss"][policy_id] = []
            metrics["vf_loss"][policy_id] = []
            metrics["entropy"][policy_id] = []
    
    # Extract data
    for i, entry in enumerate(data):
        metrics["iterations"].append(i)
        metrics["episode_reward_mean"].append(entry.get("episode_reward_mean", np.nan))
        metrics["episode_reward_min"].append(entry.get("episode_reward_min", np.nan))
        metrics["episode_reward_max"].append(entry.get("episode_reward_max", np.nan))
        metrics["episode_len_mean"].append(entry.get("episode_len_mean", np.nan))
        metrics["kl"].append(entry.get("info", {}).get("learner", {}).get("default_policy", {}).get("kl", np.nan))
        metrics["time_total_s"].append(entry.get("time_total_s", np.nan))
        
        # Per-policy metrics
        if "policy_reward_mean" in entry:
            for policy_id, value in entry["policy_reward_mean"].items():
                metrics["policy_reward_mean"].setdefault(policy_id, []).append(value)
        
        # Extract policy metrics from info
        info = entry.get("info", {}).get("learner", {})
        for policy_id in metrics["policy_loss"].keys():
            policy_info = info.get(policy_id, {})
            metrics["policy_loss"][policy_id].append(policy_info.get("policy_loss", np.nan))
            metrics["vf_loss"][policy_id].append(policy_info.get("vf_loss", np.nan))
            metrics["entropy"][policy_id].append(policy_info.get("entropy", np.nan))
    
    return metrics

def plot_metrics(metrics, args):
    """
    Plot training metrics.
    
    Args:
        metrics: Dictionary of training metrics
        args: Command line arguments
    """
    # Create output directory if saving plots
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up the figure style
    plt.style.use('ggplot')
    
    # Plot 1: Episode rewards
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics["iterations"], metrics["episode_reward_mean"], label="Mean", linewidth=2)
    ax.fill_between(
        metrics["iterations"],
        metrics["episode_reward_min"],
        metrics["episode_reward_max"],
        alpha=0.3,
        label="Min-Max Range"
    )
    ax.set_title("Episode Rewards During Training", fontsize=14)
    ax.set_xlabel("Training Iterations", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if args.save:
        plt.savefig(os.path.join(args.output_dir, "episode_rewards.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    # Plot 2: Per-policy rewards
    if metrics["policy_reward_mean"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for policy_id, rewards in metrics["policy_reward_mean"].items():
            ax.plot(metrics["iterations"], rewards, label=f"{policy_id}", linewidth=2)
        ax.set_title("Per-Policy Rewards During Training", fontsize=14)
        ax.set_xlabel("Training Iterations", fontsize=12)
        ax.set_ylabel("Policy Reward", fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if args.save:
            plt.savefig(os.path.join(args.output_dir, "policy_rewards.png"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    # Plot 3: Episode length
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics["iterations"], metrics["episode_len_mean"], linewidth=2)
    ax.set_title("Episode Length During Training", fontsize=14)
    ax.set_xlabel("Training Iterations", fontsize=12)
    ax.set_ylabel("Episode Length (steps)", fontsize=12)
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if args.save:
        plt.savefig(os.path.join(args.output_dir, "episode_length.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    # Plot 4: Policy losses
    if metrics["policy_loss"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for policy_id, losses in metrics["policy_loss"].items():
            ax.plot(metrics["iterations"], losses, label=f"{policy_id}", linewidth=2)
        ax.set_title("Policy Loss During Training", fontsize=14)
        ax.set_xlabel("Training Iterations", fontsize=12)
        ax.set_ylabel("Policy Loss", fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if args.save:
            plt.savefig(os.path.join(args.output_dir, "policy_loss.png"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    # Plot 5: Value function losses
    if metrics["vf_loss"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for policy_id, losses in metrics["vf_loss"].items():
            ax.plot(metrics["iterations"], losses, label=f"{policy_id}", linewidth=2)
        ax.set_title("Value Function Loss During Training", fontsize=14)
        ax.set_xlabel("Training Iterations", fontsize=12)
        ax.set_ylabel("VF Loss", fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if args.save:
            plt.savefig(os.path.join(args.output_dir, "vf_loss.png"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    # Plot 6: Entropy
    if metrics["entropy"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for policy_id, entropy_values in metrics["entropy"].items():
            ax.plot(metrics["iterations"], entropy_values, label=f"{policy_id}", linewidth=2)
        ax.set_title("Policy Entropy During Training", fontsize=14)
        ax.set_xlabel("Training Iterations", fontsize=12)
        ax.set_ylabel("Entropy", fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if args.save:
            plt.savefig(os.path.join(args.output_dir, "entropy.png"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    # Plot 7: KL Divergence
    if any(not np.isnan(kl) for kl in metrics["kl"]):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics["iterations"], metrics["kl"], linewidth=2)
        ax.set_title("KL Divergence During Training", fontsize=14)
        ax.set_xlabel("Training Iterations", fontsize=12)
        ax.set_ylabel("KL Divergence", fontsize=12)
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if args.save:
            plt.savefig(os.path.join(args.output_dir, "kl_divergence.png"), dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    # Plot 8: Training time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics["iterations"], metrics["time_total_s"], linewidth=2)
    ax.set_title("Cumulative Training Time", fontsize=14)
    ax.set_xlabel("Training Iterations", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if args.save:
        plt.savefig(os.path.join(args.output_dir, "training_time.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    if args.save:
        print(f"Plots saved to {args.output_dir}")

def main(args):
    """
    Main function to visualize training progress.
    
    Args:
        args: Command line arguments
    """
    # Load training data
    metrics = load_progress_data(args.logdir)
    
    # Plot metrics
    plot_metrics(metrics, args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
