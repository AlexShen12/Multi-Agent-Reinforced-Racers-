#!/usr/bin/env python
"""
Script to force all agents in MetaDrive Racing environment into autodrive mode.
This script modifies the sb3_ppo_expert function to always return a fixed action
and ensures all agents have expert_takeover=True.
"""

import os
import argparse
import numpy as np
from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv
from metadrive.examples.sb3_ppo_policy import replace_expert

# Override the expert function to always return a fixed action
def force_autodrive_expert(vehicle, deterministic=True, need_obs=False):
    """
    A simple expert function that always returns a fixed action.
    This ensures the vehicle moves forward at a constant speed.
    """
    # Set expert_takeover to True to ensure autodrive
    if hasattr(vehicle, 'expert_takeover'):
        vehicle.expert_takeover = True
    
    # Return a fixed action: [steering, acceleration]
    # 0.0 steering = go straight, 0.5 acceleration = moderate speed
    action = np.array([0.0, 0.5])
    
    # Get observation if needed
    obs = vehicle.get_observation() if need_obs else None
    
    # Return with or without observation
    return (action, [obs]) if need_obs else action

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--models_dir", type=str, default="trained_models", help="Directory for models")
    args = parser.parse_args()
    
    # Create directory for models if it doesn't exist
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Replace the expert function with our forced autodrive expert
    from metadrive.examples.sb3_ppo_policy import sb3_ppo_expert
    import metadrive.examples.sb3_ppo_policy as policy_module
    policy_module.sb3_ppo_expert = force_autodrive_expert
    replace_expert()
    
    # Create the environment
    env = MultiAgentRacingSafeEnv({
        "use_render": True,
        "manual_control": True,
        "num_agents": 2,
        "start_seed": args.seed,
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
    })
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Print agent information
    print("\n==== AGENT INFORMATION ====")
    print(f"Agent IDs in env.agents: {list(env.agents.keys())}")
    print(f"Agent IDs in observation: {list(obs.keys())}")
    
    # Force all agents into autodrive mode
    for agent_id, agent in env.agents.items():
        print(f"\nAgent ID: {agent_id}")
        print(f"  Type: {type(agent).__name__}")
        
        # Force autodrive mode
        agent.expert_takeover = True
        print(f"  Set expert_takeover to True")
    
    print("\n==== AUTODRIVE ENFORCED ====")
    print("All agents are now in autodrive mode.")
    print("The expert function has been replaced with a simple forward-driving expert.")
    
    # Main loop
    try:
        while True:
            # Step the environment with empty actions (expert will override)
            actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}
            _, _, terminated, truncated, _ = env.step(actions)
            
            # Re-enforce autodrive mode every step
            for agent_id, agent in env.agents.items():
                if not agent.expert_takeover:
                    print(f"Re-enabling autodrive for agent {agent_id}")
                    agent.expert_takeover = True
            
            # Check if all agents are done
            if all(terminated.values()) or all(truncated.values()):
                print("All agents terminated. Resetting environment...")
                obs, _ = env.reset()
                
                # Force autodrive mode again after reset
                for agent_id, agent in env.agents.items():
                    agent.expert_takeover = True
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
