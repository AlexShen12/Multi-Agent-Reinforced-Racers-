# THIS IS FOR DELEGATING TO ROUNDABOUT FOR RLLib
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from metadrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv
from metadrive.constants import DEFAULT_AGENT
from metadrive.utils.config import Config

class RoundaboutRLLibDelegatorEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config.update({
            "truncate_as_terminate": False,  # Don't treat truncation as termination
            "action_check": False, # If action_check is left at its Safe-MetaDrive default True, every action is clipped to ±1 and any NaNs are replaced by zeros.
            "allow_respawn": False,
            "num_agents": 2,
            "crash_done": False,
            "crash_vehicle_done":False,
            "crash_object_done":False,
            "out_of_road_done": False,

            # ===== Cost settings =====
            # "crash_vehicle_cost": 13,
            # "crash_object_cost": 7,
            # "out_of_road_cost": 19,  # Increased to provide stronger incentive for staying on road
            # "on_yellow_line_cost": 3.0,  # Cost for driving on yellow line (wrong side of road)
            # "on_white_line_cost": 1.0,  # Cost for driving on white line

            # ===== Reward Settings Used By MultiAgentMeta =====
            # out_of_road_penalty=10,
            # crash_vehicle_penalty=10,
            # crash_object_penalty=10,
            # crash_vehicle_cost=13,
            # crash_object_cost=7,
            # out_of_road_cost=0,  # Do not count out of road into cost!

            # ===== Environmental Setting =====
            # "traffic_density": 0.10,
            # "traffic_mode": 'respawn',

            # ===== Vehicle settings =====
            "vehicle_config": dict(
                # enable_reverse=True,  # Enable reverse driving
                vehicle_model="static_default",  # Use static model for consistent behavior
            ),
            
        })
        self.env = MultiAgentRoundaboutEnv(config)
        self.env.BIG_REWARD = 0
        self.agents = self.possible_agents = ["agent0", "agent1"]
        
        # Set action_spaces and observation_spaces
        self.action_spaces = self.env.action_space
        self.observation_spaces = self.env.observation_space

    def reset(self, *, seed=None, options=None):
        print("Overall reward", self.env.BIG_REWARD)
        self.env.BIG_REWARD = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action_dict):
        return self.env.step(action_dict)
    
    def print_config(self):
        # ===== PRINT ALL CONFIG VALUES =====
        for key in self.env.config.keys():
            if isinstance(self.env.config[key], dict):
                print(f"DICTIONARY {key}")
                for sub_key in self.env.config[key].keys():
                    if isinstance(self.env.config[key][sub_key], Config):
                        print(f"     SUB CONFIG {sub_key}")
                        for sub_sub_key in self.env.config[key][sub_key].get_dict().keys():
                            print(f"          {sub_sub_key}: {self.env.config[key][sub_key].get_dict()[sub_sub_key]}")

                    elif isinstance(self.env.config[key][sub_key], dict):
                        print(f"     SUB DICTIONARY {sub_key}")
                        for sub_sub_key in self.env.config[key][sub_key].keys():
                            print(f"          {sub_sub_key}: {self.env.config[key][sub_key][sub_sub_key]}")
                    else:
                        print(f"     {sub_key}: {self.env.config[key][sub_key]}")
            else:
                print(f"{key}: {self.env.config[key]}")