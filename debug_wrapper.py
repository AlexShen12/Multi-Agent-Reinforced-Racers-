#!/usr/bin/env python
"""
Debug script for the MARL Racing Environment Wrapper
"""

from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv

print("Importing environment successful")

# Create the environment
env = MultiAgentRacingSafeEnv({})
print("Environment created successfully")

# Initialize the environment
obs, info = env.reset()
print("Environment reset successfully")
print(f"Observation keys: {list(obs.keys())}")
print(f"Observation shape: {obs[list(obs.keys())[0]].shape}")
print(f"Agent IDs: {list(env.agents.keys())}")

# Close the environment
env.close()
print("Environment closed successfully")
