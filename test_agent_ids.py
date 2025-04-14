#!/usr/bin/env python
"""
Simple script to test agent IDs in MetaDrive
"""

from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv

def main():
    # Create the environment
    env = MultiAgentRacingSafeEnv({
        "use_render": True,
        "manual_control": False,
        "num_agents": 2,
        "horizon": 2000,
    })
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Print agent information
    print("\n==== AGENT INFORMATION ====")
    print(f"Agent IDs in env.agents: {list(env.agents.keys())}")
    print(f"Agent IDs in observation: {list(obs.keys())}")
    
    for agent_id, agent in env.agents.items():
        print(f"\nAgent ID: {agent_id}")
        print(f"  Type: {type(agent).__name__}")
        print(f"  ID: {agent.id if hasattr(agent, 'id') else 'N/A'}")
        print(f"  Name: {agent.name if hasattr(agent, 'name') else 'N/A'}")
        
        # Print some attributes
        print("  Some attributes:")
        for attr in ['agent_id', 'vehicle_id', 'id', 'name']:
            if hasattr(agent, attr):
                print(f"    {attr}: {getattr(agent, attr)}")
    
    # Close the environment
    env.close()
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
