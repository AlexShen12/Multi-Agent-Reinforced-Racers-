from ray.rllib.env.multi_agent_env import MultiAgentEnv
from metadrive.envs.marl_envs.rllib_mappo_env import RLLibMappoEnv

class RLLibDelegatorEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.env = RLLibMappoEnv(config)
        self.agents = self.possible_agents = ["agent0", "agent1"]
        
        # Set action_spaces and observation_spaces
        self.action_spaces = self.env.action_space
        self.observation_spaces = self.env.observation_space

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action_dict):
        return self.env.step(action_dict)