#!/usr/bin/env python
"""
Test script for SB3 PPO integration in MetaDrive.
This script creates a simple environment and tests the SB3 PPO policy.
"""

import os
import numpy as np
from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv
from metadrive.examples.sb3_ppo_policy import initialize_global_ppo_trainer

# Create the environment
env = MultiAgentRacingSafeEnv(
    {
        "use_render": True,
        "manual_control": False,
        "num_agents": 2,
        "start_seed": 42,
        "horizon": 2000,
        "map": 5,
        "traffic_density": 0.15,
        "use_AI_protector": True,
        "enable_finish_line": True,
        "finish_line_at_end": True,
        "terminate_on_finish": False,
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "out_of_road_done": False,
        "vehicle_config": {
            "enable_reverse": True,
            "show_lidar": False,
            "show_side_detector": False,
            "show_lane_line_detector": False,
        }
    }
)

# Reset the environment
obs, _ = env.reset()

# Create models directory if it doesn't exist
models_dir = "trained_models"
os.makedirs(models_dir, exist_ok=True)

# Initialize the global PPO trainer
ppo_trainer = initialize_global_ppo_trainer(env, models_dir)

# Print debug information
print("\n==== DEBUG: AGENT INFORMATION ====")
print(f"Agent IDs in env.agents: {list(env.agents.keys())}")
print(f"Agent IDs in observation: {list(obs.keys())}")
for agent_id, agent in env.agents.items():
    print(f"Agent ID: {agent_id}, Type: {type(agent).__name__}, Vehicle ID: {agent.id if hasattr(agent, 'id') else 'N/A'}")
print("==== END DEBUG ====\n")

# Set expert_takeover to True for all agents
for agent_id, agent in env.agents.items():
    agent.expert_takeover = True
    print(f"Set agent {agent_id} to expert takeover mode")

# Main simulation loop
for i in range(1, 100):
    # Default action is to do nothing
    actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}

    # Step the environment
    obs, rewards, terminated, truncated, info = env.step(actions)

    # Print agent positions and actions
    for agent_id, agent in env.agents.items():
        print(f"Agent {agent_id} position: {agent.position}, velocity: {agent.velocity if hasattr(agent, 'velocity') else 'N/A'}")

    # Render
    env.render()

    # Check if all agents are done
    if all(terminated.values()):
        print("All agents terminated. Resetting environment.")
        obs, _ = env.reset()

# Close the environment
env.close()
