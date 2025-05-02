#!/usr/bin/env python
"""
Wrapper for RLLibMappoEnv to make it compatible with RLlib's MultiAgentEnv interface.
"""

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from metadrive.envs.marl_envs.rllib_mappo_env import RLLibMappoEnv

class RLLibMappoEnvWrapper(MultiAgentEnv):
    """
    Wrapper for RLLibMappoEnv to make it compatible with RLlib's MultiAgentEnv interface.
    """
    
    def __init__(self, config=None):
        """Initialize the environment wrapper."""
        self.env = RLLibMappoEnv(config)
        
        # Set required attributes for MultiAgentEnv
        self.possible_agents = ["agent0", "agent1"]
        self.agents = self.possible_agents.copy()
        
        # Set spaces
        self.observation_space = self.env.observation_spaces
        self.action_space = self.env.action_spaces
        
        # Set observation and action spaces for each agent
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed)
        self.agents = self.possible_agents.copy()
        return obs, info
    
    def step(self, action_dict):
        """Step the environment."""
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        
        # Check if all agents are done
        if all(terminated.values()):
            self.agents = []
        
        return obs, rewards, terminated, truncated, info
    
    def render(self, mode="human", **kwargs):
        """Render the environment."""
        return self.env.render(mode=mode, **kwargs)
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    @property
    def unwrapped(self):
        """Return the base environment."""
        return self.env.unwrapped
