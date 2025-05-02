#!/usr/bin/env python
"""
Simple script to run the MetaDrive Racing environment with forced autodrive.
This script directly modifies the vehicle control logic to ensure all agents
stay in autodrive mode.
"""

import time
from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv

def main():
    # Create the environment
    env = MultiAgentRacingSafeEnv({
        "use_render": True,
        "manual_control": True,  # Allow manual control for testing
        "num_agents": 2,
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
    
    # Force all agents into autodrive mode
    for agent_id, agent in env.agents.items():
        print(f"Setting agent {agent_id} to expert takeover mode")
        agent.expert_takeover = True
    
    # Main loop
    step_counter = 0
    try:
        while True:
            step_counter += 1
            
            # Step the environment with empty actions (expert will override)
            actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}
            _, _, terminated, truncated, _ = env.step(actions)
            
            # Every 10 steps, check and re-enforce autodrive mode
            if step_counter % 10 == 0:
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
                    print(f"Reset agent {agent_id} to expert takeover mode")
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
