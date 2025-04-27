"""
Multi-Agent Racing Safe MetaDrive Environment

This environment extends the SafeMetaDriveEnv to support Multi-Agent Reinforcement Learning (MARL)
between two cars in a racing scenario. It incorporates:
1. Concurrent control of two vehicles
2. Race-based reward structure (leading car gets bonus)
3. Support for reverse driving
4. Continuous simulation despite collisions or boundary violations
"""

import math
import numpy as np
from typing import Dict, Any, Tuple

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.utils import Config, clip
import gymnasium as gym

# Define constants for the racing environment
RACING_SAFE_METADRIVE_DEFAULT_CONFIG = dict(
    # ===== Stuff ChatGPT said to add =====
    action_check=False, # If action_check is left at its Safe-MetaDrive default True, every action is clipped to ±1 and any NaNs are replaced by zeros.

    # ===== Multi-agent settings =====
    manual_control=False, # Default
    is_multi_agent=True,
    num_agents=2,  # Fixed at 2 for racing scenario
    allow_respawn=False,  # Fixes the agent number in the environment

    # Ensure we're using the correct agent IDs

    # ===== Racing settings =====
    # Reward for being in the lead
    leading_reward_factor=0.1,  # Reward multiplier for being in the lead
    winning_reward=100.0,  # Additional reward for winning the race
    # winning_reward=10.0,  # Additional reward for winning the race
    checkpoint_reward=20.0,  # Reward for reaching a checkpoint
    # checkpoint_reward=2.0,  # Reward for reaching a checkpoint
    # first_checkpoint_bonus=1.0,  # Additional bonus for being first to reach a checkpoint
    first_checkpoint_bonus=5.0,  # Additional bonus for being first to reach a checkpoint
    num_checkpoints=5,  # Number of checkpoints to place around the track

    # ===== Finish line settings =====
    enable_finish_line=True,  # Whether to add a finish line
    finish_line_at_end=True,  # Place finish line at the end of the last track segment
    terminate_on_finish=True,  # End episode when finish line is crossed
    finish_line_width=10.0,  # Width of the finish line for detection

    # ===== Safe driving settings =====
    # Override termination conditions to allow continuous simulation
    crash_vehicle_done=False,
    crash_object_done=False,
    out_of_road_done=False,

    # ===== Cost settings =====
    crash_vehicle_cost=10.0,
    crash_object_cost=5.0,
    out_of_road_cost=20.0,  # Increased to provide stronger incentive for staying on road
    on_yellow_line_cost=3.0,  # Cost for driving on yellow line (wrong side of road)
    on_white_line_cost=1.0,  # Cost for driving on white line

    # ===== Vehicle settings =====
    vehicle_config=dict(
        enable_reverse=True,  # Enable reverse driving
        vehicle_model="static_default",  # Use static model for consistent behavior
    ),

    # Agent configuration is handled in _post_process_config
)


class RLLibMappoEnv(MultiAgentMetaDrive):
    """
    A Multi-Agent Racing environment based on SafeMetaDriveEnv.

    This environment features:
    1. Two cars racing against each other
    2. Modified reward structure to incentivize racing
    3. Continuous simulation despite collisions or boundary violations
    4. Support for reverse driving
    """

    def default_config(self) -> Config:
        # Start with SafeMetaDriveEnv config
        config = super(RLLibMappoEnv, self).default_config()
        # Update with racing-specific settings
        config.update(RACING_SAFE_METADRIVE_DEFAULT_CONFIG, allow_add_new_key=True)
        return config

    def _post_process_config(self, config):
        # Process config using parent method first
        config = super(RLLibMappoEnv, self)._post_process_config(config)

        # Ensure we have agent_configs for agent0 and agent1
        if "agent_configs" not in config:
            config["agent_configs"] = {}

        # Create agent configs if not present
        for i in range(2):  # Fixed at 2 agents for racing
            agent_id = f"agent{i}"
            if agent_id not in config["agent_configs"]:
                config["agent_configs"][agent_id] = {}

        # Set spawn lanes if not specified
        if "spawn_lane_index" not in config["agent_configs"]["agent0"]:
            config["agent_configs"]["agent0"]["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)

        if "spawn_lane_index" not in config["agent_configs"]["agent1"]:
            config["agent_configs"]["agent1"]["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 1)

        # Set special colors for better visibility
        config["agent_configs"]["agent0"]["use_special_color"] = True
        config["agent_configs"]["agent1"]["use_special_color"] = True

        return config

    def __init__(self, config=None):
        if config==None:
            config = {}
        config.update({
            "num_agents": 2,
            # "start_seed": args.seed,  # Use the provided seed for reproducible track generation
            "horizon": 2000,  # Increase max steps to ensure agents have enough time to reach the finish line

            # ===== Random Track Generation =====
            "num_scenarios": 100,  # Generate a large number of different scenarios
            # Instead of using a fixed map like "CSRCR", we'll use a random map
            # Setting map to an integer value will generate a random track with that many blocks
            "map": 5,  # Generate a random track with 5 blocks
            "random_agent_model": False,  # Use consistent vehicle models
            "random_lane_width": True,  # Randomize lane width
            "random_lane_num": False,  # Keep lane number consistent

            # ===== Traffic Settings =====
            # Add traffic to the environment with a density of 0.15
            "traffic_density": 0.15,  # Controls the number of traffic vehicles (0.0 to 1.0)
            "traffic_mode": "trigger",  # Traffic appears when agent approaches
            "random_traffic": True,  # Randomize traffic positions

            # ===== Agent Settings =====
            # "use_AI_protector": True,  # Ensure both agents use autodrive
            "use_AI_protector": False,  # THIS USES AI TO STEER YOU BACK INTO THE LANE. DO NOT ENABLE

            # ===== Finish Line Settings =====
            "enable_finish_line": True,  # Add a definitive finish line
            "finish_line_at_end": True,  # Place finish line at the end of the last track segment
            "terminate_on_finish": False,  # IMPORTANT: Don't end episode until both agents cross the finish line

            # ===== Termination Settings =====
            # Disable all termination conditions that might end the episode prematurely
            "crash_vehicle_done": False,  # Don't end when vehicles crash
            "crash_object_done": False,  # Don't end when hitting objects
            "out_of_road_done": False,  # Don't end when going off-road
            "on_continuous_line_done": False,  # Don't end when crossing continuous lines
            "on_broken_line_done": False,  # Don't end when crossing broken lines
            "out_of_route_done": False,  # Don't end when going off route
            "crash_done": False,  # Don't end on any crash
            "truncate_as_terminate": False,  # Don't treat truncation as termination

            # ===== Vehicle Configuration =====
            "vehicle_config": {
                "enable_reverse": True,
                "show_lidar": False,
                "show_side_detector": False,
                "show_lane_line_detector": False,
            }
        })
        super(RLLibMappoEnv, self).__init__(config)
        self.action_spaces = {
            "agent0": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "agent1": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        }
        self.observation_spaces = {
            "agent0": gym.spaces.Box(low=0.0, high=1.0, shape=(91,), dtype=np.float32),
            "agent1": gym.spaces.Box(low=0.0, high=1.0, shape=(91,), dtype=np.float32)
        }

        # Initialize episode cost tracking for each agent
        self.episode_cost = {}

        # Track agent progress for racing rewards
        self.agent_progress = {}
        self.agent_positions = {}
        self.race_finished = False
        self.race_winner = None
        self.previous_winner = None  # Track the winner from the previous race

        # Initialize checkpoint tracking
        self.checkpoints = []
        self.agent_checkpoints = {}
        self.checkpoint_first_agent = {}

        # Initialize finish line tracking
        self.finish_line = None
        self.finish_line_crossed = {}

        # Initialize agent-specific attributes
        self._init_agent_attributes()

    def _init_agent_attributes(self):
        """Initialize agent-specific attributes."""
        # Get agent IDs from config
        agent_ids = list(self.config["agent_configs"].keys())
        if not agent_ids:
            agent_ids = ["agent0", "agent1"]  # Default if not specified

        # Initialize tracking dictionaries for each agent
        for agent_id in agent_ids:
            self.episode_cost[agent_id] = 0.0
            self.agent_progress[agent_id] = 0.0
            self.agent_positions[agent_id] = (0, 0)

    def reset(self, *args, **kwargs):
        """Reset the environment and initialize agent tracking."""
        # Store the previous winner before resetting race state
        if self.race_finished and self.race_winner:
            self.previous_winner = self.race_winner

        # Ensure all agents are properly cleaned up before reset
        # This is important to prevent issues with agents not respawning
        if hasattr(self, 'engine') and hasattr(self.engine, 'agent_manager'):
            # Force clear any agents in the dying queue
            if hasattr(self.engine.agent_manager, '_dying_objects'):
                for v_name, (v, _) in list(self.engine.agent_manager._dying_objects.items()):
                    self.engine.agent_manager._remove_vehicle(v)
                self.engine.agent_manager._dying_objects.clear()

        # Reset episode costs and race status
        self._init_agent_attributes()
        self.race_finished = False
        self.race_winner = None

        # Initialize off-road and wrong-side tracking
        self.off_road_counter = {}
        self.off_road_status = {}
        self.wrong_side_counter = {}
        self.wrong_side_status = {}

        # Initialize checkpoint tracking
        self.agent_checkpoints = {}
        self.checkpoint_first_agent = {}

        # Initialize finish line tracking
        self.finish_line = None
        self.finish_line_crossed = {}

        # Reset the environment
        obs, info = super(RLLibMappoEnv, self).reset(*args, **kwargs)
        self.previous_positions = {
            agent_id: agent.position
            for agent_id, agent in self.agents.items()
        }

        # Initialize agent positions and tracking variables after reset
        for agent_id, agent in self.agents.items():
            # COMMENTED OUT
            # # Ensure all agents are in autodrive mode (expert takeover)
            # agent.expert_takeover = True

            self.agent_positions[agent_id] = agent.position
            self.off_road_counter[agent_id] = 0
            self.off_road_status[agent_id] = False
            self.wrong_side_counter[agent_id] = 0
            self.wrong_side_status[agent_id] = False
            self.agent_checkpoints[agent_id] = set()  # Track which checkpoints this agent has passed
            self.finish_line_crossed[agent_id] = False  # Reset finish line crossing status

            # COMMENTED OUT
            # print(f"Set agent {agent_id} to expert takeover mode (autodrive)")

        # Create checkpoints along the track
        self._create_checkpoints()

        # Create finish line if enabled
        if self.config["enable_finish_line"]:
            self._create_finish_line()

        return obs, info

    def step(self, actions):
        """
        Step the environment with actions from all agents.

        Args:
            actions: Dict mapping agent_ids to their respective actions

        Returns:
            obs: Dict of observations for each agent
            rewards: Dict of rewards for each agent
            terminated: Dict indicating if each agent is terminated
            truncated: Dict indicating if each agent is truncated
            info: Dict of additional info for each agent
        """
        # Initialize checkpoint info for this step
        self.checkpoint_info = {agent_id: [] for agent_id in self.agents.keys()}

        # Step the environment
        obs, rewards, terminated, truncated, info = super(RLLibMappoEnv, self).step(actions)

        # Update agent progress for racing rewards
        self._update_agent_progress()

        # Check for finish line crossing
        if self.config["enable_finish_line"] and self.finish_line:
            self._check_finish_line_crossing(info)

            # Check if we need to end the episode after the 3-second buffer
            if self.config["terminate_on_finish"] and self.race_finished:
                # Check if 3 seconds have passed since the finish line was crossed
                if self.finish_line["finish_time"] is not None:
                    import time
                    current_time = time.time()
                    elapsed_time = current_time - self.finish_line["finish_time"]

                    # COMMENTED OUT
                    # If 3 seconds have passed, terminate all agents
                    # if elapsed_time >= 3.0:
                    if elapsed_time >= 1.0:
                        # Set all agents to terminated
                        for agent_id in self.agents.keys():
                            terminated[agent_id] = True
                        print("1 second elapsed after finish line crossing. Ending episode.")

        # Apply racing rewards
        rewards = self._apply_racing_rewards(rewards, info)

        # Apply checkpoint rewards
        rewards = self._apply_checkpoint_rewards(rewards, info)

        # Check for race completion
        self._check_race_completion(terminated, info)

        # Add race information to info dict
        for agent_id in self.agents.keys():
            if agent_id in info:
                info[agent_id].update({
                    "progress": self.agent_progress[agent_id],
                    "is_leading": self._is_agent_leading(agent_id),
                    "race_finished": self.race_finished,
                    "race_winner": self.race_winner,
                    "previous_winner": self.previous_winner,
                    "finish_line_crossed": self.finish_line_crossed.get(agent_id, False),
                    "checkpoints_passed": len(self.agent_checkpoints.get(agent_id, set())),
                    "total_checkpoints": len(self.checkpoints)
                })

                # Add checkpoint info if available
                if agent_id in self.checkpoint_info and self.checkpoint_info[agent_id]:
                    info[agent_id]["checkpoint_reached"] = True
                    info[agent_id]["checkpoint_info"] = self.checkpoint_info[agent_id]

        return obs, rewards, terminated, truncated, info

    def _create_checkpoints(self):
        """Create checkpoints along the track."""
        self.checkpoints = []
        self.checkpoint_first_agent = {}

        # Get the map and road network
        if not hasattr(self.engine, 'current_map') or not self.engine.current_map:
            return

        # Get the road network from the map
        road_network = self.engine.current_map.road_network
        if not road_network or not hasattr(road_network, 'graph'):
            return

        # Get all lanes from the road network
        all_lanes = []
        for from_node, to_nodes in road_network.graph.items():
            for to_node, lanes in to_nodes.items():
                all_lanes.extend(lanes)

        if not all_lanes:
            return

        # Create evenly spaced checkpoints along the track
        num_checkpoints = self.config["num_checkpoints"]
        checkpoint_lanes = []

        # Try to select lanes that are evenly distributed around the track
        if len(all_lanes) >= num_checkpoints:
            step = len(all_lanes) // num_checkpoints
            for i in range(0, len(all_lanes), step):
                if len(checkpoint_lanes) < num_checkpoints:
                    checkpoint_lanes.append(all_lanes[i])
        else:
            # If not enough lanes, use all available lanes
            checkpoint_lanes = all_lanes

        # Create checkpoints at the middle of each selected lane
        for i, lane in enumerate(checkpoint_lanes):
            checkpoint_id = i
            checkpoint_pos = lane.position(lane.length / 2, 0)
            self.checkpoints.append({
                "id": checkpoint_id,
                "position": checkpoint_pos,
                "lane": lane,
                "radius": 5.0  # Detection radius
            })
            self.checkpoint_first_agent[checkpoint_id] = None

        print(f"Created {len(self.checkpoints)} checkpoints along the track")

    def _update_agent_progress(self):
        """Update the progress of each agent along the track and check for checkpoints."""
        for agent_id, agent in self.agents.items():
            current_position = agent.position
            if agent_id in self.agent_positions:
                prev_position = self.agent_positions[agent_id]
                # Calculate distance traveled along the road
                if hasattr(agent, 'navigation') and agent.navigation and agent.navigation.current_ref_lanes:
                    # Use road direction to measure progress
                    lane = agent.navigation.current_ref_lanes[-1]
                    road_direction = lane.heading_theta_at(lane.local_coordinates(current_position)[0])
                    direction_vector = (math.cos(road_direction), math.sin(road_direction))
                    # Project movement onto road direction
                    movement = (current_position[0] - prev_position[0], current_position[1] - prev_position[1])
                    progress = movement[0] * direction_vector[0] + movement[1] * direction_vector[1]
                    self.agent_progress[agent_id] += progress

                    # Check if agent has reached any checkpoints
                    self._check_checkpoints(agent_id, current_position)

            # Store current position for next step
            self.agent_positions[agent_id] = current_position

    def _check_checkpoints(self, agent_id, position):
        """Check if an agent has reached any checkpoints."""
        for checkpoint in self.checkpoints:
            checkpoint_id = checkpoint["id"]
            checkpoint_pos = checkpoint["position"]
            checkpoint_radius = checkpoint["radius"]

            # Skip if agent has already passed this checkpoint
            if checkpoint_id in self.agent_checkpoints[agent_id]:
                continue

            # Calculate distance to checkpoint
            dx = position[0] - checkpoint_pos[0]
            dy = position[1] - checkpoint_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)

            # Check if agent has reached the checkpoint
            if distance <= checkpoint_radius:
                # Agent has reached this checkpoint
                self.agent_checkpoints[agent_id].add(checkpoint_id)

                # Check if this is the first agent to reach this checkpoint
                is_first = False
                if self.checkpoint_first_agent[checkpoint_id] is None:
                    self.checkpoint_first_agent[checkpoint_id] = agent_id
                    is_first = True

                # Add checkpoint info to agent's info dict in the next step
                if not hasattr(self, 'checkpoint_info'):
                    self.checkpoint_info = {}
                if agent_id not in self.checkpoint_info:
                    self.checkpoint_info[agent_id] = []

                self.checkpoint_info[agent_id].append({
                    "checkpoint_id": checkpoint_id,
                    "is_first": is_first
                })

    def _is_agent_leading(self, agent_id):
        """Check if the agent is currently leading the race."""
        if len(self.agent_progress) <= 1:
            return True  # If only one agent, it's leading by default

        # Find the agent with the most progress
        leading_agent = max(self.agent_progress.items(), key=lambda x: x[1])[0]
        return agent_id == leading_agent

    def _get_lead_margin(self, agent_id):
        """Get the margin by which an agent is leading (if it is leading)."""
        if not self._is_agent_leading(agent_id) or len(self.agent_progress) <= 1:
            return 0.0

        # Calculate lead margin (distance ahead of second place)
        progress_values = sorted(self.agent_progress.values(), reverse=True)
        if len(progress_values) > 1:
            lead_margin = progress_values[0] - progress_values[1]
            return lead_margin
        return 0.0

    def _apply_racing_rewards(self, rewards, info):
        """Apply racing-specific rewards to the base rewards."""
        # Copy rewards to avoid modifying the original
        modified_rewards = rewards.copy()

        for agent_id in self.agents.keys():
            # Add leading bonus if agent is in the lead
            if self._is_agent_leading(agent_id):
                # Base leading reward
                leading_bonus = self.config["leading_reward_factor"]

                # Additional bonus based on lead margin
                lead_margin = self._get_lead_margin(agent_id)
                margin_bonus = min(lead_margin * 0.01, 0.1)  # Cap at 0.1

                # Apply the bonuses
                modified_rewards[agent_id] += leading_bonus + margin_bonus

                # Add info about the bonuses
                if agent_id in info:
                    info[agent_id]["leading_bonus"] = leading_bonus
                    info[agent_id]["margin_bonus"] = margin_bonus

            # Add winning reward if this agent has won the race
            if self.race_winner == agent_id:
                modified_rewards[agent_id] += self.config["winning_reward"]
                if agent_id in info:
                    info[agent_id]["winning_reward"] = self.config["winning_reward"]
        return modified_rewards

    def _apply_checkpoint_rewards(self, rewards, info):
        """Apply checkpoint-specific rewards to the base rewards."""
        # Copy rewards to avoid modifying the original
        modified_rewards = rewards.copy()

        # Process checkpoint rewards for each agent
        for agent_id in self.agents.keys():
            if agent_id in self.checkpoint_info and self.checkpoint_info[agent_id]:
                # Agent has reached one or more checkpoints this step
                for checkpoint_data in self.checkpoint_info[agent_id]:
                    checkpoint_id = checkpoint_data["checkpoint_id"]
                    is_first = checkpoint_data["is_first"]

                    # Base reward for reaching a checkpoint
                    checkpoint_reward = self.config["checkpoint_reward"]

                    # Additional bonus for being first to reach this checkpoint
                    first_bonus = self.config["first_checkpoint_bonus"] if is_first else 0.0

                    # Apply the rewards
                    total_checkpoint_reward = checkpoint_reward + first_bonus
                    modified_rewards[agent_id] += total_checkpoint_reward

                    # Add info about the checkpoint rewards
                    if agent_id in info:
                        if "checkpoint_rewards" not in info[agent_id]:
                            info[agent_id]["checkpoint_rewards"] = []

                        info[agent_id]["checkpoint_rewards"].append({
                            "checkpoint_id": checkpoint_id,
                            "is_first": is_first,
                            "reward": total_checkpoint_reward
                        })

        return modified_rewards

    def _check_race_completion(self, terminated, info):
        """Check if any agent has completed the race."""
        for agent_id, agent in self.agents.items():
            # Check if agent has reached destination
            if agent_id in info and info[agent_id].get("arrive_dest", False):
                # This agent has finished the race
                if not self.race_finished:
                    self.race_finished = True
                    self.race_winner = agent_id

    def cost_function(self, vehicle_id: str):
        """
        Compute cost for safety violations.
        Implements a cost function similar to SafeMetaDriveEnv but for multi-agent racing.
        Includes incremental penalties for driving off-road.
        """
        # Calculate cost based on safety violations
        cost = 0.0
        step_info = {}

        # Get the vehicle
        vehicle = self.vehicles[vehicle_id]

        # Crash with vehicle
        if hasattr(vehicle, 'crash_vehicle') and vehicle.crash_vehicle:
            cost += self.config["crash_vehicle_cost"]
            step_info["crash_vehicle"] = True

        # Crash with object
        if hasattr(vehicle, 'crash_object') and vehicle.crash_object:
            cost += self.config["crash_object_cost"]
            step_info["crash_object"] = True

        # Out of road - with incremental penalty
        if hasattr(vehicle, 'out_of_road') and vehicle.out_of_road:
            # Initialize off-road tracking if not already done
            if not hasattr(self, 'off_road_counter'):
                self.off_road_counter = {}
            if not hasattr(self, 'off_road_status'):
                self.off_road_status = {}

            # Initialize for this vehicle if needed
            if vehicle_id not in self.off_road_counter:
                self.off_road_counter[vehicle_id] = 0
            if vehicle_id not in self.off_road_status:
                self.off_road_status[vehicle_id] = False

            # Check if this is a new off-road event or continuing
            if not self.off_road_status[vehicle_id]:
                # First time off-road
                self.off_road_status[vehicle_id] = True
                self.off_road_counter[vehicle_id] = 1
            else:
                # Continuing off-road, increment counter
                self.off_road_counter[vehicle_id] += 1

            # Apply incremental cost - increases exponentially the longer the vehicle stays off-road
            # Base cost + exponential growth based on steps off-road
            # This creates a rapidly escalating penalty to strongly incentivize returning to the road
            incremental_cost = self.config["out_of_road_cost"] * (1.0 + 0.2 * (self.off_road_counter[vehicle_id]))
            cost += incremental_cost

            # Add info about off-road status
            step_info["out_of_road"] = True
            step_info["off_road_steps"] = self.off_road_counter[vehicle_id]
            step_info["off_road_cost"] = incremental_cost
        else:
            # Reset off-road status when back on road
            if hasattr(self, 'off_road_status') and vehicle_id in self.off_road_status and self.off_road_status[vehicle_id]:
                self.off_road_status[vehicle_id] = False

        # Check for driving on yellow line (wrong side of road) with escalating penalty
        # Initialize wrong-side tracking if not already done
        if not hasattr(self, 'wrong_side_counter'):
            self.wrong_side_counter = {}
        if not hasattr(self, 'wrong_side_status'):
            self.wrong_side_status = {}

        # Initialize for this vehicle if needed
        if vehicle_id not in self.wrong_side_counter:
            self.wrong_side_counter[vehicle_id] = 0
        if vehicle_id not in self.wrong_side_status:
            self.wrong_side_status[vehicle_id] = False

        # Check if on yellow line (wrong side of road)
        on_wrong_side = False
        yellow_line_cost = 0.0

        if hasattr(vehicle, 'on_yellow_continuous_line') and vehicle.on_yellow_continuous_line:
            yellow_line_cost += self.config["on_yellow_line_cost"]
            step_info["on_yellow_continuous_line"] = True
            on_wrong_side = True
        if hasattr(vehicle, 'on_yellow_broken_line') and vehicle.on_yellow_broken_line:
            yellow_line_cost += self.config["on_yellow_line_cost"] * 0.5  # Less penalty for broken line
            step_info["on_yellow_broken_line"] = True
            on_wrong_side = True

        if on_wrong_side:
            # Check if this is a new wrong-side event or continuing
            if not self.wrong_side_status[vehicle_id]:
                # First time on wrong side
                self.wrong_side_status[vehicle_id] = True
                self.wrong_side_counter[vehicle_id] = 1
            else:
                # Continuing on wrong side, increment counter
                self.wrong_side_counter[vehicle_id] += 1

            # Apply exponential cost increase for staying on wrong side
            escalation_factor = 1.0 + 0.3 * (self.wrong_side_counter[vehicle_id] ** 1.5)
            yellow_line_cost *= escalation_factor

            cost += yellow_line_cost
            step_info["yellow_line_cost"] = yellow_line_cost
            step_info["wrong_side"] = True
            step_info["wrong_side_steps"] = self.wrong_side_counter[vehicle_id]
        else:
            # Reset wrong-side status when back on correct side
            if self.wrong_side_status[vehicle_id]:
                self.wrong_side_status[vehicle_id] = False

        # Check for driving on white line
        white_line_cost = 0.0
        if hasattr(vehicle, 'on_white_continuous_line') and vehicle.on_white_continuous_line:
            white_line_cost += self.config["on_white_line_cost"]
            step_info["on_white_continuous_line"] = True
        if hasattr(vehicle, 'on_white_broken_line') and vehicle.on_white_broken_line:
            white_line_cost += self.config["on_white_line_cost"] * 0.5  # Less penalty for broken line
            step_info["on_white_broken_line"] = True
        if white_line_cost > 0:
            cost += white_line_cost
            step_info["white_line_cost"] = white_line_cost

        # Track cost for this specific agent
        if vehicle_id in self.episode_cost:
            self.episode_cost[vehicle_id] += cost
            step_info["agent_cost"] = self.episode_cost[vehicle_id]
            step_info["total_cost"] = self.episode_cost[vehicle_id]

        return cost, step_info

    def _create_finish_line(self):
        """Create a finish line that stretches across the entire ending track segment."""
        self.finish_line = None

        # Get the map and road network
        if not hasattr(self.engine, 'current_map') or not self.engine.current_map:
            return

        # Get the blocks from the map (track segments)
        if not hasattr(self.engine.current_map, 'blocks') or not self.engine.current_map.blocks:
            return

        # Get the road network from the map
        road_network = self.engine.current_map.road_network
        if not road_network or not hasattr(road_network, 'graph'):
            return

        # Look for a specific ending block type if possible
        ending_block = None

        # Try to find a block that represents the end of the track
        # In some maps, there might be specific ending blocks
        for block in self.engine.current_map.blocks:
            # Check if this is a designated end block by its type or properties
            if hasattr(block, 'SOCKET_OUT') and not block.SOCKET_OUT:
                # This block has no output socket, so it's likely an ending block
                ending_block = block
                break
            elif hasattr(block, 'type') and 'end' in str(block.type).lower():
                # Block type contains 'end', suggesting it's an ending block
                ending_block = block
                break
            elif hasattr(block, 'id') and 'end' in str(block.id).lower():
                # Block ID contains 'end', suggesting it's an ending block
                ending_block = block
                break

        # If we couldn't find a specific ending block, use the last block by ID
        if not ending_block:
            max_block_id = -1
            for block in self.engine.current_map.blocks:
                if hasattr(block, 'id'):
                    # Try to convert ID to int for comparison if it's a string
                    try:
                        block_id = int(block.id) if isinstance(block.id, str) else block.id
                        if block_id > max_block_id:
                            max_block_id = block_id
                            ending_block = block
                    except (ValueError, TypeError):
                        # If conversion fails, continue to the next block
                        pass

        # If we still don't have an ending block, use the last block in the list
        if not ending_block and len(self.engine.current_map.blocks) > 0:
            ending_block = self.engine.current_map.blocks[-1]

        if not ending_block:
            print("Could not find an ending block for the finish line")
            return

        print(f"Found ending block: {ending_block.id if hasattr(ending_block, 'id') else 'Unknown ID'}")

        # Get all lanes in the ending block
        block_lanes = []

        # Try different methods to get the lanes
        if hasattr(ending_block, 'get_lanes'):
            block_lanes = ending_block.get_lanes()
        elif hasattr(ending_block, 'get_out_lanes'):
            block_lanes = ending_block.get_out_lanes()

        # If we couldn't get lanes directly from the block, try the road network
        if not block_lanes:
            # Try to find lanes connected to the ending block
            if hasattr(ending_block, 'end_node'):
                for from_node, to_nodes in road_network.graph.items():
                    if from_node == ending_block.end_node:
                        for _, lanes in to_nodes.items():
                            block_lanes.extend(lanes)

            # If still no lanes, try the start node
            if not block_lanes and hasattr(ending_block, 'start_node'):
                for from_node, to_nodes in road_network.graph.items():
                    if from_node == ending_block.start_node:
                        for _, lanes in to_nodes.items():
                            block_lanes.extend(lanes)

        # If we still don't have lanes, try to get all lanes in the road network
        if not block_lanes:
            for _, to_nodes in road_network.graph.items():
                for _, lanes in to_nodes.items():
                    block_lanes.extend(lanes)

        if not block_lanes:
            print("Could not find any lanes for the finish line")
            return

        # Get the width of the entire track at the finish line position
        # by finding the leftmost and rightmost points of all lanes
        lane_position = 0.99  # 99% along the lane length - very end of the lane

        # Calculate the average position and heading across all lanes
        avg_pos_x = 0
        avg_pos_y = 0
        avg_heading = 0
        valid_lanes = 0

        # Store lane positions for later use
        lane_positions = []

        for lane in block_lanes:
            try:
                # Get position at the very end of the lane
                pos = lane.position(lane.length * lane_position, 0)
                heading = lane.heading_theta_at(lane.length * lane_position)

                # Store this position
                lane_positions.append((pos, heading))

                avg_pos_x += pos[0]
                avg_pos_y += pos[1]
                avg_heading += heading
                valid_lanes += 1
            except Exception as e:
                print(f"Error getting lane position: {e}")

        if valid_lanes == 0:
            print("No valid lanes found for finish line")
            return

        # Calculate average position and heading
        avg_pos_x /= valid_lanes
        avg_pos_y /= valid_lanes
        avg_heading /= valid_lanes

        # Calculate the width of the track by finding the maximum distance
        # between any two lanes at the finish line position
        max_width = 0
        for i in range(len(lane_positions)):
            for j in range(i+1, len(lane_positions)):
                try:
                    pos_i = lane_positions[i][0]  # Get the stored position
                    pos_j = lane_positions[j][0]  # Get the stored position
                    dist = math.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                    max_width = max(max_width, dist)
                except Exception:
                    pass

        # Ensure minimum width for the finish line
        # Make the finish line wider in the track width direction
        # But not so wide that it crosses to the wrong side of the road
        track_width = max(max_width + 15.0, 40.0)  # Add more margin and ensure wider minimum width

        # Create the finish line at the end of the track
        self.finish_line = {
            "position": (avg_pos_x, avg_pos_y),
            "width": track_width,
            "heading": avg_heading,  # Original road direction
            "lanes": block_lanes,
            "lane_position": lane_position,
            "finish_time": None,  # Will store the time when first agent crosses the line
            "visible": False  # Don't show the finish line visually
        }

        # Visualize the finish line if rendering is enabled
        if self.config["use_render"]:
            self._visualize_finish_line()

        print(f"Created finish line across the entire track at the ending segment")

    def _visualize_finish_line(self):
        """Add a visual representation of the finish line that stretches across the entire track."""
        if not self.finish_line or not hasattr(self.engine, 'render_pipeline'):
            return

        # Check if the finish line should be visible
        if not self.finish_line.get("visible", True):
            # Don't visualize the finish line
            return

        # The finish line is invisible, so we don't need to do anything
        # This method is kept for compatibility with the existing code

    def _check_finish_line_crossing(self, info):
        """Check if any agent has crossed the finish line."""
        if not self.finish_line:
            return

        finish_pos = self.finish_line["position"]
        finish_heading = self.finish_line["heading"]
        finish_width = self.finish_line["width"]

        # Calculate perpendicular direction to the finish line
        perp_x = math.sin(finish_heading)
        perp_y = -math.cos(finish_heading)

        # Calculate the finish line endpoints
        half_width = finish_width / 2
        start_x = finish_pos[0] - perp_x * half_width
        start_y = finish_pos[1] - perp_y * half_width
        end_x = finish_pos[0] + perp_x * half_width
        end_y = finish_pos[1] + perp_y * half_width

        # Calculate the direction along the finish line
        line_dir_x = end_x - start_x
        line_dir_y = end_y - start_y
        line_length = math.sqrt(line_dir_x**2 + line_dir_y**2)

        # Normalize the direction vector
        if line_length > 0:
            line_dir_x /= line_length
            line_dir_y /= line_length

        # Calculate the normal to the finish line (direction to check for crossing)
        normal_x = -line_dir_y
        normal_y = line_dir_x

        # Store previous positions to detect crossing
        if not hasattr(self, 'previous_positions'):
            self.previous_positions = {agent_id: agent.position for agent_id, agent in self.agents.items()}

        for agent_id, agent in self.agents.items():
            # Skip if already crossed
            if self.finish_line_crossed.get(agent_id, False):
                continue

            # Get current and previous agent positions
            current_pos = agent.position
            previous_pos = self.previous_positions.get(agent_id, current_pos)

            # Update previous position for next check
            self.previous_positions[agent_id] = current_pos

            # Check if the agent is close enough to the finish line
            # First, calculate the projection of the agent position onto the finish line
            agent_to_start_x = current_pos[0] - start_x
            agent_to_start_y = current_pos[1] - start_y

            # Project onto the line direction
            proj = agent_to_start_x * line_dir_x + agent_to_start_y * line_dir_y

            # Check if the projection is within the line segment
            if 0 <= proj <= line_length:
                # Calculate the perpendicular distance to the line
                perp_dist = abs(agent_to_start_x * normal_x + agent_to_start_y * normal_y)

                # Check if the agent is close enough to the line
                # Widen the detection threshold but ensure it doesn't cross to the wrong side
                # The typical lane width is around 3.5-4.0 units, so we'll use 5.0 as threshold
                # This is wide enough to detect the vehicle but not so wide it crosses lanes
                if perp_dist <= 5.0:  # Wider detection threshold
                    # Check if the agent crossed the line by checking the sign of the dot product
                    # of the movement vector and the line normal
                    movement_x = current_pos[0] - previous_pos[0]
                    movement_y = current_pos[1] - previous_pos[1]

                    # Skip if the agent hasn't moved
                    if movement_x == 0 and movement_y == 0:
                        continue

                    # Calculate dot product with the normal
                    dot_product = movement_x * normal_x + movement_y * normal_y

                    # If dot product is positive, the agent crossed the line in the correct direction
                    if dot_product > 0:
                        self.finish_line_crossed[agent_id] = True

                        # Mark race as finished and set winner
                        if not self.race_finished:
                            self.race_finished = True
                            self.race_winner = agent_id

                            # Record the finish time
                            import time
                            self.finish_line["finish_time"] = time.time()

                            # Add winning reward
                            if agent_id in info:
                                info[agent_id]["finish_line_crossed"] = True

                            print(f"Agent {agent_id} has crossed the finish line and won the race!")
                            print(f"Episode will end in 1 seconds...")

    def done_function(self, vehicle_id):
        """
        Override the done function to allow continuous simulation.
        Only terminate when reaching the destination, crossing the finish line, or max steps.
        """
        # Get the vehicle
        vehicle = self.vehicles[vehicle_id]

        # Initialize done info
        done_info = {
            "crash_vehicle": False,
            "crash_object": False,
            "out_of_road": False,
            "arrive_dest": False,
            "finish_line": False,
            "max_step": False
        }

        # Check for crash with vehicle
        if hasattr(vehicle, 'crash_vehicle') and vehicle.crash_vehicle:
            done_info["crash_vehicle"] = True

        # Check for crash with object
        if hasattr(vehicle, 'crash_object') and vehicle.crash_object:
            done_info["crash_object"] = True

        # Check for out of road
        if hasattr(vehicle, 'out_of_road') and vehicle.out_of_road:
            done_info["out_of_road"] = True

        # Check for arrival at destination
        if hasattr(vehicle, 'arrive_destination') and vehicle.arrive_destination:
            done_info["arrive_dest"] = True

        # Check if this agent has crossed the finish line
        if self.config["enable_finish_line"] and vehicle_id in self.finish_line_crossed and self.finish_line_crossed[vehicle_id]:
            done_info["finish_line"] = True

        # Check for max steps
        if self.episode_step >= self.config["horizon"]:
            done_info["max_step"] = True

        # Terminate for max steps, arrival at destination, or finish line crossing
        done = done_info["max_step"] or done_info["arrive_dest"]

        # Add finish line termination if enabled
        if self.config["enable_finish_line"] and self.config["terminate_on_finish"]:
            done = done or done_info["finish_line"]

        return done, done_info


if __name__ == "__main__":
    # Example usage
    env = RLLibMappoEnv(
        {
            "use_render": True,
            # "manual_control": True,
            "num_agents": 2,
            "map": "CSRCR",  # Use a circular map for racing
            "vehicle_config": {
                "enable_reverse": True,
                "show_lidar": False,
                "show_side_detector": False,
                "show_lane_line_detector": False,
            }
        }
    )

    # Reset the environment
    obs, _ = env.reset()

    # Main simulation loop
    for i in range(1, 100000):
        # Random actions for demonstration
        actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}

        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Render with race information
        env.render(
            text={
                "Race Progress": {agent_id: f"{progress:.2f}" for agent_id, progress in env.agent_progress.items()},
                "Leading": next((agent_id for agent_id in env.agents.keys() if env.agent_progress.get(agent_id, 0) == max(env.agent_progress.values())), None),
                "Race Winner": env.race_winner if env.race_finished else "None",
            }
        )

        # Check if all agents are done
        if all(terminated.values()):
            print("All agents terminated. Resetting environment.")
            obs, _ = env.reset()

    env.close()
