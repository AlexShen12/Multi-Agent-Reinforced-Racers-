#!/usr/bin/env python
"""
Wrapper for MultiAgentRacingSafeEnv to make it compatible with RLlib's MultiAgentEnv interface.
"""

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv

class MARLRacingEnvWrapper(MultiAgentEnv):
    """
    Wrapper for MultiAgentRacingSafeEnv to make it compatible with RLlib's MultiAgentEnv interface.
    This wrapper ensures the environment properly exposes the required attributes and methods
    for RLlib's multi-agent training.
    """

    def __init__(self, config=None):
        """Initialize the environment wrapper."""
        # Default configuration if none provided
        if config is None:
            config = {}

        # Ensure we have the minimum required configuration
        default_config = {
            "num_agents": 2,
            "enable_finish_line": True,
            "terminate_on_finish": False,
            "horizon": 2000,
            "traffic_density": 0.0,  # No traffic by default for training
            "use_render": False,
            "manual_control": False,
            "crash_done": False,
            "out_of_road_done": False,
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "out_of_route_done": False,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,
            "truncate_as_terminate": False,
        }

        # Update default config with provided config
        for key, value in config.items():
            default_config[key] = value

        # Reset the MetaDrive engine initialization flag if needed
        try:
            import metadrive.engine.engine_utils as engine_utils
            if engine_utils.engine_initialized():
                engine_utils._ENGINE_INITIALIZED = False
                print("Reset MetaDrive engine initialization flag")
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not reset engine initialization flag: {e}")

        # Create the environment
        self.env = MultiAgentRacingSafeEnv(default_config)

        # Set required attributes for MultiAgentEnv
        # Convert agent IDs to match the expected format in the training script
        self.possible_agents = [f"agent{i}" for i in range(default_config["num_agents"])]
        self.agents = self.possible_agents.copy()

        # Initialize the environment to get observation and action spaces
        temp_obs, _ = self.env.reset()

        # Create observation and action spaces dictionaries
        self.observation_spaces = {}
        self.action_spaces = {}

        # Map the environment's spaces to our wrapper's spaces
        for i, agent_id in enumerate(self.possible_agents):
            # Get the original agent ID from the environment
            orig_agent_id = list(self.env.observation_space.spaces.keys())[i]

            # Set the observation and action spaces
            self.observation_spaces[agent_id] = self.env.observation_space.spaces[orig_agent_id]
            self.action_spaces[agent_id] = self.env.action_space.spaces[orig_agent_id]

        # Set spaces for compatibility
        self.observation_space = self.observation_spaces
        self.action_space = self.action_spaces

        # Store the mapping between our agent IDs and the environment's agent IDs
        self.agent_id_map = {}
        for i, agent_id in enumerate(self.possible_agents):
            if i < len(self.env.agents):
                orig_agent_id = list(self.env.agents.keys())[i]
                self.agent_id_map[agent_id] = orig_agent_id
            else:
                # If we have more possible agents than actual agents, use the last one
                self.agent_id_map[agent_id] = list(self.env.agents.keys())[-1]

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed)
        self.agents = self.possible_agents.copy()

        # Update agent ID mapping after reset
        self.agent_id_map = {}
        for i, agent_id in enumerate(self.possible_agents):
            if i < len(self.env.agents):
                orig_agent_id = list(self.env.agents.keys())[i]
                self.agent_id_map[agent_id] = orig_agent_id
            else:
                # If we have more possible agents than actual agents, use the last one
                self.agent_id_map[agent_id] = list(self.env.agents.keys())[-1]

        # Convert observation keys to match our agent IDs
        converted_obs = {}
        for i, agent_id in enumerate(self.possible_agents):
            if i < len(obs):
                orig_agent_id = list(obs.keys())[i]
                converted_obs[agent_id] = obs[orig_agent_id]

        # Convert info keys to match our agent IDs
        converted_info = {}
        if info:
            for i, agent_id in enumerate(self.possible_agents):
                if i < len(info):
                    orig_agent_id = list(info.keys())[i] if list(info.keys()) else None
                    if orig_agent_id:
                        converted_info[agent_id] = info[orig_agent_id]

        return converted_obs, converted_info

    def step(self, action_dict):
        """Step the environment."""
        # Convert action keys to match the environment's agent IDs
        converted_actions = {}
        for agent_id, action in action_dict.items():
            if agent_id in self.agent_id_map:
                converted_actions[self.agent_id_map[agent_id]] = action

        # Step the environment
        obs, rewards, terminated, truncated, info = self.env.step(converted_actions)

        # Convert observation, rewards, terminated, truncated, and info keys to match our agent IDs
        converted_obs = {}
        converted_rewards = {}
        converted_terminated = {}
        converted_truncated = {}
        converted_info = {}

        for agent_id in self.agents:
            orig_agent_id = self.agent_id_map.get(agent_id)
            if orig_agent_id in obs:
                converted_obs[agent_id] = obs[orig_agent_id]
                converted_rewards[agent_id] = rewards[orig_agent_id]
                converted_terminated[agent_id] = terminated[orig_agent_id]
                converted_truncated[agent_id] = truncated[orig_agent_id]
                if orig_agent_id in info:
                    converted_info[agent_id] = info[orig_agent_id]

        # Check if all agents are done
        if all(converted_terminated.values()):
            self.agents = []

        return converted_obs, converted_rewards, converted_terminated, converted_truncated, converted_info

    def render(self, mode="human", **kwargs):
        """Render the environment."""
        return self.env.render(mode=mode, **kwargs)

    def close(self):
        """Close the environment."""
        if hasattr(self, 'env') and self.env is not None:
            result = self.env.close()
            # Reset the MetaDrive engine initialization flag
            try:
                import metadrive.engine.engine_utils as engine_utils
                if engine_utils.engine_initialized():
                    engine_utils._ENGINE_INITIALIZED = False
                    print("Reset MetaDrive engine initialization flag on close")
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not reset engine initialization flag on close: {e}")
            return result
        return None

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self.env.unwrapped

    @property
    def agent_progress(self):
        """Return the agent progress for rendering."""
        if hasattr(self.env, 'agent_progress'):
            # Convert keys to match our agent IDs
            converted_progress = {}
            for agent_id in self.agents:
                orig_agent_id = self.agent_id_map.get(agent_id)
                if orig_agent_id in self.env.agent_progress:
                    converted_progress[agent_id] = self.env.agent_progress[orig_agent_id]
            return converted_progress
        return {}

    @property
    def race_finished(self):
        """Return whether the race is finished."""
        return hasattr(self.env, 'race_finished') and self.env.race_finished

    @property
    def race_winner(self):
        """Return the race winner."""
        if hasattr(self.env, 'race_winner') and self.env.race_winner:
            # Convert the winner to our agent ID format
            for agent_id, orig_agent_id in self.agent_id_map.items():
                if orig_agent_id == self.env.race_winner:
                    return agent_id
        return None
