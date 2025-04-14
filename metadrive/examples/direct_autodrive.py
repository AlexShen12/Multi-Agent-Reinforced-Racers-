#!/usr/bin/env python
"""
Script to directly modify the vehicle class to force autodrive mode.
"""

import time
from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv
from metadrive.component.vehicle.base_vehicle import BaseVehicle

# Store the original update_state method
original_update_state = BaseVehicle.update_state

# Create a patched update_state method that forces expert_takeover
def patched_update_state(self, *args, **kwargs):
    # Force expert_takeover to True
    self.expert_takeover = True
    
    # Call the original method
    return original_update_state(self, *args, **kwargs)

def main():
    # Patch the BaseVehicle class
    print("Patching BaseVehicle.update_state to force expert_takeover=True")
    BaseVehicle.update_state = patched_update_state
    
    # Create the environment
    env = MultiAgentRacingSafeEnv({
        "use_render": True,
        "manual_control": True,
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
    
    # Verify all agents are in autodrive mode
    for agent_id, agent in env.agents.items():
        print(f"Agent {agent_id} expert_takeover: {agent.expert_takeover}")
    
    print("\n==== AUTODRIVE ENFORCED ====")
    print("All agents will stay in autodrive mode due to the patched update_state method.")
    
    # Main loop
    try:
        while True:
            # Step the environment with empty actions (expert will override)
            actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}
            _, _, terminated, truncated, _ = env.step(actions)
            
            # Check if all agents are done
            if all(terminated.values()) or all(truncated.values()):
                print("All agents terminated. Resetting environment...")
                obs, _ = env.reset()
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    # Restore the original method
    BaseVehicle.update_state = original_update_state
    print("Restored original BaseVehicle.update_state method")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
