#!/usr/bin/env python
"""
Script to force all agents in MetaDrive into autodrive mode.
"""

from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv

def force_autodrive():
    """Force all agents in the environment into autodrive mode."""
    # Create the environment
    env = MultiAgentRacingSafeEnv({
        "use_render": True,
        "manual_control": True,
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
        
        # Force autodrive mode
        print("  Setting expert_takeover to True...")
        agent.expert_takeover = True
        print(f"  expert_takeover is now: {agent.expert_takeover}")
    
    print("\n==== AUTODRIVE ENFORCED ====")
    print("All agents are now in autodrive mode.")
    print("You can now run the environment with agents in autodrive mode.")
    
    # Run the environment for a while
    print("\nRunning environment for 1000 steps...")
    for _ in range(1000):
        # Default action is to do nothing
        actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}
        
        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Check if all agents are done
        if all(terminated.values()) or all(truncated.values()):
            print("All agents terminated. Resetting environment...")
            obs, _ = env.reset()
            
            # Force autodrive mode again after reset
            for agent_id, agent in env.agents.items():
                agent.expert_takeover = True
                print(f"Reset agent {agent_id} to expert takeover mode")
    
    # Close the environment
    env.close()
    
    print("\nTest complete!")

if __name__ == "__main__":
    force_autodrive()
