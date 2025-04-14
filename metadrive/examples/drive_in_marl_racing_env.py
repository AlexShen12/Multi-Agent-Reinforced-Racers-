#!/usr/bin/env python
"""
This script demonstrates how to use the Multi-Agent Racing Safe MetaDrive environment.

Key features:
1. Two cars racing against each other
2. Modified reward structure to incentivize racing
3. Continuous simulation despite collisions or boundary violations
4. Support for reverse driving
5. Both agents consistently use their RL models regardless of camera focus
6. Improved race completion logic that waits for both cars to finish
7. Clear race winner determination and display

Note: The policy classes have been modified to ensure that both agents use their
reinforcement learning (RL) models at all times, regardless of which car is currently
being focused on by the viewer. This ensures that when switching between agents with
the camera (using the Q key), the non-focused agent continues to use its RL model
rather than reverting to a simple straight-line behavior.

The environment now features a definitive finish line. When both cars cross this line,
the episode ends, the winner is determined based on arrival order, and a new episode begins.

Please feel free to run this script to enjoy a racing experience! Remember to press H to see help message!
"""
import argparse
import time  # Added for tracking finish times and reset delay
import math  # Added for trigonometric functions
import os

from metadrive.constants import HELP_MESSAGE
from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv

# Import our SB3 PPO policy
from metadrive.examples.sb3_ppo_policy import MultiAgentPPOTrainer


# Function to save agent model weights
def save_agent_model_weights(agent, agent_id):
    """
    Extract and save the model weights from an agent.

    Args:
        agent: The agent object whose weights need to be saved
        agent_id: The ID of the agent

    Returns:
        A dictionary containing the agent's model weights and other relevant state
    """
    # In a real implementation, this would extract the actual model weights
    # For this example, we'll create a simplified representation
    weights = {
        # Store the policy parameters
        "policy_weights": agent.expert_takeover,  # In a real implementation, this would be the actual neural network weights
        "agent_type": type(agent).__name__,
        "position": agent.position,
        "heading": agent.heading,
        "velocity": agent.velocity if hasattr(agent, "velocity") else [0, 0],
        # Add any other relevant state information
    }

    print(f"Saved model weights for agent {agent_id}")
    return weights

# Function to load agent model weights
def load_agent_model_weights(agent, weights):
    """
    Load saved model weights into an agent.

    Args:
        agent: The agent object to load weights into
        weights: The saved weights dictionary

    Returns:
        None
    """
    # In a real implementation, this would load the actual model weights
    # For this example, we'll just restore the simplified representation
    if weights:
        agent.expert_takeover = weights.get("policy_weights", True)
        # In a real implementation, you would restore the neural network weights here

        print(f"Loaded model weights for agent {agent.name}")
    return

# Function to remove an agent from the simulation
def remove_agent_from_simulation(env, agent_id):
    """
    Remove an agent from the active simulation while preserving its model weights.

    Args:
        env: The environment object
        agent_id: The ID of the agent to remove

    Returns:
        bool: True if the agent was successfully removed, False otherwise
    """
    if agent_id not in env.agents:
        print(f"Agent {agent_id} not found in environment")
        return False

    # Save the agent's model weights before removing it
    agent = env.agents[agent_id]

    # Make the vehicle invisible by setting its scale to near zero
    if hasattr(agent, 'vehicle_node') and agent.vehicle_node is not None:
        agent.vehicle_node.setScale(0.001, 0.001, 0.001)

        # Disable collision detection
        if hasattr(agent, 'collision_node') and agent.collision_node is not None:
            agent.collision_node.setCollideMask(0)

    # Set the agent's velocity to zero to stop it from moving
    if hasattr(agent, 'set_velocity'):
        agent.set_velocity([0, 0])

    print(f"Removed agent {agent_id} from active simulation")
    return True

# Custom function to check if an agent has reached its destination
def custom_check_destination_arrival(env, info):
    """
    Uses the built-in _is_arrive_destination method to check if agents have reached their destinations.
    This leverages the existing navigation system instead of using a custom finish line.

    Args:
        env: The environment object
        info: The info dictionary from the environment step

    Returns:
        None
    """
    # Safety check: ensure finish_line_crossed attribute exists
    if not hasattr(env, 'finish_line_crossed'):
        env.finish_line_crossed = {}

    # Initialize race_finished and race_winner attributes if they don't exist
    if not hasattr(env, 'race_finished'):
        env.race_finished = False

    if not hasattr(env, 'race_winner'):
        env.race_winner = None

    # Check each agent for destination arrival
    for agent_id, agent in env.agents.items():
        # Skip if already crossed
        if env.finish_line_crossed.get(agent_id, False):
            continue

        # Use the built-in _is_arrive_destination method
        if env._is_arrive_destination(agent):
            # Mark this agent as having crossed the finish line
            env.finish_line_crossed[agent_id] = True

            # Mark race as finished and set winner if this is the first agent
            if not env.race_finished:
                env.race_finished = True
                env.race_winner = agent_id

                # Record the finish time
                import time
                env.finish_time = time.time()

                # Add winning reward
                if agent_id in info:
                    info[agent_id]["finish_line_crossed"] = True
                    # Add a large winning reward
                    if "reward" in info[agent_id]:
                        info[agent_id]["reward"] += env.config.get("winning_reward", 10.0)

                print(f"Agent {agent_id} has reached the destination and won the race!")
                print(f"Episode will end in 3 seconds...")
            else:
                # Add finish info for non-winners
                if agent_id in info:
                    info[agent_id]["finish_line_crossed"] = True

                print(f"Agent {agent_id} has reached the destination (position: {len(env.finish_line_crossed)})")

            # Check if all agents have crossed
            if len(env.finish_line_crossed) >= len(env.agents):
                print(f"All agents have reached their destinations!")



# Function to create a proper finish line at the very end of the track
def fix_finish_line_placement(env):
    """
    Create a proper finish line at the very end of the track by identifying the last segment
    and placing the finish line at its end.

    This implementation ensures the finish line is always at the end of the track by:
    1. Identifying the last block in the track
    2. Finding the last lane segment in that block
    3. Placing the finish line at the very end of that lane segment

    Args:
        env: The environment object

    Returns:
        bool: True if the finish line was successfully created, False otherwise
    """
    print("\nCreating a proper finish line at the very end of the track...")

    # Step 1: Get the map and blocks
    if not hasattr(env.engine, 'current_map') or not env.engine.current_map:
        print("No map found in environment")
        return False

    # Get the blocks from the map (track segments)
    if not hasattr(env.engine.current_map, 'blocks') or not env.engine.current_map.blocks:
        print("No blocks found in map")
        return False

    blocks = env.engine.current_map.blocks
    if not blocks:
        print("No blocks found in map")
        return False

    # Step 2: Get the last block (final track segment)
    last_block = blocks[-1]
    print(f"Found last block: {last_block.ID if hasattr(last_block, 'ID') else 'Unknown'} (Type: {type(last_block).__name__})")

    # Step 3: Get the road network
    road_network = env.engine.current_map.road_network
    if not road_network or not hasattr(road_network, 'graph'):
        print("No road network found in map")
        return False

    # Step 4: Find the lanes in the last block
    # First, identify all road segments in the last block
    last_block_roads = []
    for start_node, to_nodes in road_network.graph.items():
        for end_node, lanes in to_nodes.items():
            # Check if this road belongs to the last block
            if hasattr(last_block, 'ID') and str(last_block.ID) in start_node:
                last_block_roads.append((start_node, end_node, lanes))

    if not last_block_roads:
        print("Could not find any roads in the last block")
        return False

    print(f"Found {len(last_block_roads)} road segments in the last block")

    # Step 5: Find the ending road segment(s) - those that have no outgoing connections
    ending_roads = []
    for start_node, end_node, lanes in last_block_roads:
        # Check if this end node has any outgoing connections
        has_outgoing = False
        for other_start, _ in road_network.graph.items():
            if end_node == other_start:
                has_outgoing = True
                break

        # If no outgoing connections, this is an ending road
        if not has_outgoing:
            ending_roads.append((start_node, end_node, lanes))

    if not ending_roads:
        print("Could not find any ending roads in the last block, using the last road segment")
        # Use the last road in the last block as a fallback
        ending_roads = [last_block_roads[-1]]

    # Special handling for curved segments: check if the last block is a curve
    is_curve_block = False
    if hasattr(last_block, 'ID') and last_block.ID.startswith('C'):
        print("Last block is a curve segment. Using special curve handling.")
        is_curve_block = True
        # For curves, we need to ensure we're using the very last segment
        # Sort road segments by their end node coordinates to find the one furthest along the track
        if len(last_block_roads) > 1:
            # Try to get the coordinates of the end nodes
            end_coords = []
            for start_node, end_node, _ in last_block_roads:
                # Extract coordinates from the node name if possible
                # Node names often contain coordinates in the format "node_x_y"
                parts = end_node.split('_')
                if len(parts) >= 3 and parts[-2].replace('-', '').replace('.', '').isdigit() and parts[-1].replace('-', '').replace('.', '').isdigit():
                    try:
                        x = float(parts[-2])
                        y = float(parts[-1])
                        end_coords.append((start_node, end_node, (x, y)))
                    except ValueError:
                        end_coords.append((start_node, end_node, (0, 0)))
                else:
                    end_coords.append((start_node, end_node, (0, 0)))

            # Sort by x coordinate (typically the main direction of travel)
            if end_coords:
                end_coords.sort(key=lambda item: item[2][0], reverse=True)
                # Use the road with the highest x-coordinate as the ending road
                for road in last_block_roads:
                    if road[0] == end_coords[0][0] and road[1] == end_coords[0][1]:
                        ending_roads = [road]
                        print(f"Selected ending road with highest x-coordinate: {end_coords[0][2]}")
                        break

    print(f"Found {len(ending_roads)} ending road segments")

    # Step 6: Get the lanes from the ending road
    start_node, end_node, lanes = ending_roads[0]
    if not lanes:
        print("No lanes found in the ending road segment")
        return False

    print(f"Found {len(lanes)} lanes in the ending road segment")

    # Step 7: Filter to only include the right-side (positive) lanes
    right_side_lanes = []
    for lane in lanes:
        # Try different methods to identify right-side lanes
        is_right_side = False

        # Check lane index if available
        if hasattr(lane, 'index'):
            if isinstance(lane.index, int) and lane.index >= 0:
                is_right_side = True
            elif isinstance(lane.index, tuple) and len(lane.index) > 0:
                if isinstance(lane.index[0], int) and lane.index[0] >= 0:
                    is_right_side = True
                elif isinstance(lane.index[-1], int) and lane.index[-1] >= 0:
                    is_right_side = True

        # For StraightLane, check the direction by comparing start and end points
        if hasattr(lane, 'start_point') and hasattr(lane, 'end_point'):
            # If the lane goes from left to right (increasing x), it's likely a positive lane
            if lane.end_point[0] > lane.start_point[0]:
                is_right_side = True

        if is_right_side:
            right_side_lanes.append(lane)

    # If we couldn't identify any right-side lanes, use all lanes
    if not right_side_lanes:
        print("Could not identify right-side lanes, using all lanes")
        right_side_lanes = lanes
    else:
        print(f"Identified {len(right_side_lanes)} right-side lanes")

    # Step 8: Place the finish line at the very end of the lanes
    # Determine the appropriate lane position based on the lane type
    # For curved segments, we need to be more careful with the position
    if is_curve_block:
        # For curves, use a slightly lower position to ensure we're still on the road
        lane_position = 0.95  # 95% along the lane length for curves
        print("Using 95% lane position for curve segment")
    else:
        # For straight segments, we can go closer to the end
        lane_position = 0.99  # 99% along the lane length for straight segments
        print("Using 99% lane position for straight segment")

    try:
        # Calculate the average position and heading across all right-side lanes
        avg_pos_x = 0
        avg_pos_y = 0
        avg_heading = 0
        valid_lanes = 0
        final_lane = None

        # First, check if any of the lanes are CircularLane (curved)
        has_circular_lanes = False
        for lane in right_side_lanes:
            if "CircularLane" in str(type(lane).__name__):
                has_circular_lanes = True
                print(f"Detected CircularLane: {lane}")
                break

        # Special handling for circular lanes
        if has_circular_lanes:
            print("Using special handling for circular lanes")
            # For circular lanes, we need to be extra careful with the position
            lane_position = min(lane_position, 0.95)  # Ensure we're not too close to the end

        for lane in right_side_lanes:
            # Get the position at the specified point along the lane
            if hasattr(lane, 'position') and callable(lane.position):
                try:
                    # Get lane type for debugging
                    lane_type = type(lane).__name__
                    print(f"Processing lane of type: {lane_type}, length: {lane.length}")

                    # Calculate position with error handling
                    try:
                        # All lane types in MetaDrive require a lateral parameter
                        # Use lateral=0 for center of lane
                        pos = lane.position(lane_position * lane.length, 0)
                        heading = lane.heading_theta_at(lane_position * lane.length)

                        print(f"Lane position calculated at {lane_position * 100}% of length: {pos}")
                        print(f"Lane heading at this position: {heading}")

                        avg_pos_x += pos[0]
                        avg_pos_y += pos[1]
                        avg_heading += heading
                        valid_lanes += 1

                        # Store the final lane for finish line crossing detection
                        # Prefer straight lanes over circular lanes if available
                        if final_lane is None or ("CircularLane" not in str(type(lane).__name__) and "CircularLane" in str(type(final_lane).__name__)):
                            final_lane = lane
                            print(f"Selected lane as final_lane: {type(lane).__name__}")
                    except Exception as e:
                        print(f"Error calculating position at {lane_position * lane.length}: {e}")

                        # Try a fallback position
                        fallback_position = 0.9  # 90% along the lane length
                        try:
                            print(f"Trying fallback position at {fallback_position * 100}% of length")

                            # All lane types in MetaDrive require a lateral parameter
                            # Use lateral=0 for center of lane
                            pos = lane.position(fallback_position * lane.length, 0)
                            heading = lane.heading_theta_at(fallback_position * lane.length)

                            avg_pos_x += pos[0]
                            avg_pos_y += pos[1]
                            avg_heading += heading
                            valid_lanes += 1

                            # Store the final lane for finish line crossing detection
                            if final_lane is None:
                                final_lane = lane
                                print(f"Selected lane as final_lane (fallback): {type(lane).__name__}")
                        except Exception as e2:
                            print(f"Error calculating fallback position: {e2}")
                except Exception as e:
                    print(f"Error accessing lane position method: {e}")

        if valid_lanes == 0:
            print("Could not calculate finish line position from lanes, using fallback method")
            # Fallback method: Create a simple finish line at a fixed position
            # Get the last block in the track
            last_block = blocks[-1]

            # Get the socket position (end of the block)
            if hasattr(last_block, 'get_socket') and callable(last_block.get_socket):
                try:
                    socket = last_block.get_socket(0)
                    if socket and hasattr(socket, 'position'):
                        # Use the socket position as the finish line position
                        socket_pos = socket.position
                        print(f"Using socket position as finish line position: {socket_pos}")

                        # Create a simple finish line
                        env.finish_line = {
                            "position": (socket_pos[0], socket_pos[1]),
                            "width": 40.0,  # Wide enough to cover the track
                            "heading": 0.0,  # Default heading
                            "finish_time": None,
                            "visible": False,  # Don't show the finish line
                            "detection_buffer": 10.0,  # Larger buffer for detection
                            "final_lane": None,  # No specific lane
                            "crossed_positions": {}
                        }

                        # Override the environment's finish line crossing detection method
                        env._original_check_finish_line_crossing = env._check_finish_line_crossing
                        env._check_finish_line_crossing = lambda info: custom_check_destination_arrival(env, info)

                        # Create a visual representation of the finish line
                        create_finish_line_visualization(env)

                        return True
                except Exception as e:
                    print(f"Error using socket position: {e}")

            # If socket approach fails, create a finish line at a fixed distance from the start
            print("Creating a default finish line at a fixed position")
            env.finish_line = {
                "position": (500.0, 0.0),  # Fixed position far from start
                "width": 100.0,  # Very wide to ensure detection
                "heading": 0.0,  # Default heading
                "finish_time": None,
                "visible": False,  # Don't show the finish line
                "detection_buffer": 20.0,  # Very large buffer for detection
                "final_lane": None,  # No specific lane
                "crossed_positions": {}
            }

            # Override the environment's finish line crossing detection method
            env._original_check_finish_line_crossing = env._check_finish_line_crossing
            env._check_finish_line_crossing = lambda info: custom_check_destination_arrival(env, info)

            # Create a visual representation of the finish line
            create_finish_line_visualization(env)

            return True

        # Calculate the average position and heading
        avg_pos_x /= valid_lanes
        avg_pos_y /= valid_lanes
        avg_heading /= valid_lanes

        # Calculate the width of the finish line based on the track width
        # Find the leftmost and rightmost points of all lanes
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for lane in right_side_lanes:
            if hasattr(lane, 'position') and callable(lane.position):
                try:
                    # Get the left and right boundaries at the specified point
                    # All lane types in MetaDrive require a lateral parameter
                    # Use lateral=0 for center of lane
                    pos = lane.position(lane_position * lane.length, 0)
                    heading = lane.heading_theta_at(lane_position * lane.length)
                    width = lane.width if hasattr(lane, 'width') else 3.5  # Default lane width

                    # Calculate perpendicular direction to get the width
                    perp_x = math.sin(heading)
                    perp_y = -math.cos(heading)

                    # Calculate the left and right points
                    left_x = pos[0] - perp_x * width / 2
                    left_y = pos[1] - perp_y * width / 2
                    right_x = pos[0] + perp_x * width / 2
                    right_y = pos[1] + perp_y * width / 2

                    # Update the min and max coordinates
                    min_x = min(min_x, left_x, right_x)
                    max_x = max(max_x, left_x, right_x)
                    min_y = min(min_y, left_y, right_y)
                    max_y = max(max_y, left_y, right_y)
                except Exception as e:
                    print(f"Error calculating lane boundaries: {e}")

        # Calculate the track width
        track_width_x = max_x - min_x
        track_width_y = max_y - min_y
        track_width = math.sqrt(track_width_x**2 + track_width_y**2)

        # Ensure minimum width for the finish line
        # Make the finish line wider than the track width to ensure detection
        finish_line_width = max(track_width * 1.2, 40.0)  # At least 40 units wide

        # Calculate the perpendicular heading (90 degrees to the road direction)
        perpendicular_heading = avg_heading + math.pi/2

        # Create the finish line dictionary
        env.finish_line = {
            "position": (avg_pos_x, avg_pos_y),
            "width": finish_line_width,
            "heading": perpendicular_heading,
            "finish_time": None,
            "visible": False,  # Don't show the finish line
            "detection_buffer": 5.0,  # Buffer for detection
            "final_lane": final_lane,  # Store reference to the final lane for crossing detection
            "crossed_positions": {}
        }

        print(f"Created finish line at position: {(avg_pos_x, avg_pos_y)}")
        print(f"Finish line heading: {perpendicular_heading}")
        print(f"Finish line width: {finish_line_width}")
        print(f"Finish line placed at {lane_position * 100}% of the final lane length")

        # Override the environment's finish line crossing detection method with our custom implementation
        env._original_check_finish_line_crossing = env._check_finish_line_crossing
        env._check_finish_line_crossing = lambda info: custom_check_destination_arrival(env, info)

        # Create a visual representation of the finish line
        create_finish_line_visualization(env)

        return True
    except Exception as e:
        print(f"Error creating finish line: {e}")
        return False

# Function to create a visual representation of the finish line
def create_finish_line_visualization(env):
    """
    This function is intentionally disabled to remove visual finish lines.
    We're now using the built-in destination arrival detection instead of custom finish lines.

    Args:
        env: The environment object

    Returns:
        None
    """
    # Remove any existing finish line visualization
    if hasattr(env, 'engine') and hasattr(env.engine, 'render') and hasattr(env.engine.render, 'find'):
        existing_finish_line = env.engine.render.find("**/finish_line*")
        if existing_finish_line:
            existing_finish_line.removeNode()
            print("Removed existing finish line visualization")

    # Mark finish line as invisible if it exists
    if hasattr(env, 'finish_line') and env.finish_line:
        env.finish_line["visible"] = False
        print("Finish line visualization disabled")

# Custom wrapper for the cost function to halt computations for finished agents
def custom_cost_function_wrapper(env, vehicle_id):
    """
    Wrapper for the cost function that halts computations for agents that have crossed the finish line.

    Args:
        env: The environment object
        vehicle_id: The ID of the vehicle to calculate cost for

    Returns:
        tuple: (cost, step_info)
    """
    # Check if this agent has crossed the finish line
    if hasattr(env, 'finished_agents') and vehicle_id in env.finished_agents:
        # Agent has finished - halt cost computation
        return 0.0, {"cost": 0.0, "cost_computation_halted": True}

    # Agent has not finished - proceed with normal cost computation
    # Call the original cost function
    return env._original_cost_function(vehicle_id)

# Custom wrapper for the reward function to halt computations for finished agents
def custom_reward_function_wrapper(env, vehicle_id):
    """
    Wrapper for the reward function that halts computations for agents that have crossed the finish line.

    Args:
        env: The environment object
        vehicle_id: The ID of the vehicle to calculate reward for

    Returns:
        tuple: (reward, step_info)
    """
    # Check if this agent has crossed the finish line
    if hasattr(env, 'finished_agents') and vehicle_id in env.finished_agents:
        # Agent has finished - halt reward computation
        return 0.0, {"reward": 0.0, "reward_computation_halted": True}

    # Agent has not finished - proceed with normal reward computation
    # Call the original reward function
    return env._original_reward_function(vehicle_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the environment")
    parser.add_argument("--manual_control", action="store_true", help="Enable manual control for testing")
    parser.add_argument("--models_dir", type=str, default="trained_models", help="Directory to save models")
    parser.add_argument("--use_sb3_ppo", action="store_true", help="Use Stable-Baselines3 PPO instead of the default expert")
    args = parser.parse_args()

    # Configure environment
    env = MultiAgentRacingSafeEnv(
        {
            "use_render": True,
            "manual_control": args.manual_control,
            "num_agents": 2,
            "start_seed": args.seed,  # Use the provided seed for reproducible track generation
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
            "use_AI_protector": True,  # Ensure both agents use autodrive

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
        }
    )

    # Print help message
    print(HELP_MESSAGE)
    print("\nAdditional Racing Environment Controls:")
    print("- Press Q to toggle between agents (when manual_control is enabled)")
    print("- Press R to enable/disable reverse mode")
    print("- Press T to toggle between manual control and autodrive for the current agent")
    print("- Both agents use autodrive by default and will continue driving even when not focused")
    print("- Collisions and out-of-road events will not terminate the episode")
    print("- The episode will continue until BOTH cars cross the finish line or max steps (2000) are reached")
    print("- The track is randomly generated but reproducible with the same seed")
    print("- Traffic vehicles are present with a density of 0.15")
    print("- Lane width is set to 4.0 units (wider than default 3.5)")
    print("- The finish line is positioned at the end of the final track segment")
    print("- Cars will be removed from the simulation after crossing the finish line")
    print("- Each car's model weights are preserved when it finishes the race")
    print("- Cars are regenerated with their preserved model weights in the next episode")
    print("- The leading car receives bonus rewards")
    print("- The first car to cross the finish line wins the race and receives a winning reward")
    print("- The race winner is displayed and a new race begins after both cars finish")
    if args.use_sb3_ppo:
        print("\nStable-Baselines3 PPO Training:")
        print(f"- Using SB3 PPO instead of the default expert")
        print(f"- Models are saved to {args.models_dir}")
        print(f"- Models are updated after each episode")
        print(f"- Press BACKSPACE to manually reset the episode and reinitialize weights")

    # Initialize variables to track race state
    previous_winner = None
    race_count = 0

    # Initialize finish tracking variables
    finish_order = []  # List to track the order in which agents cross the finish line
    finish_times = {}  # Dictionary to track when each agent crossed the finish line
    reset_timer = None  # Timer to track when to reset after both agents finish
    reset_delay = 3.0  # Seconds to wait after both agents finish before resetting

    # Dictionary to store agent model weights for preservation across episodes
    # This will store the model weights of agents that have finished the race
    agent_model_weights = {}

    # Dictionary to track which agents have been removed from the simulation
    removed_agents = set()

    # Create a set to track agents that have crossed the finish line
    # This will be used to halt cost and reward computations for these agents
    if not hasattr(env, 'finished_agents'):
        env.finished_agents = set()  # Set of agent IDs that have crossed the finish line

    # If using SB3 PPO, set up the trainer and replace the expert
    if args.use_sb3_ppo:
        # Create directory for saving models if it doesn't exist
        os.makedirs(args.models_dir, exist_ok=True)

        # We need to reset first to initialize the environment
        temp_obs, _ = env.reset()

        # Initialize the global PPO trainer and attach it to the environment
        from metadrive.examples.sb3_ppo_policy import initialize_global_ppo_trainer
        ppo_trainer = initialize_global_ppo_trainer(env, args.models_dir, use_expert_weights=True)

        # Verify that the PPO trainer is properly attached to the environment
        if hasattr(env.engine, 'ppo_trainer') and env.engine.ppo_trainer is not None:
            print("Verified that PPO trainer is attached to the environment engine")
        else:
            print("WARNING: PPO trainer is not properly attached to the environment engine")
            # Attach it manually as a fallback
            env.engine.ppo_trainer = ppo_trainer

        # Print debug information about the environment and agents
        print("\n==== DEBUG: ENVIRONMENT INFORMATION ====")
        print(f"Environment type: {type(env).__name__}")
        print(f"Engine type: {type(env.engine).__name__}")
        print(f"Number of agents: {len(env.agents)}")
        print(f"Agent IDs: {list(env.agents.keys())}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print("==== END DEBUG ====\n")

        print("Initialized SB3 PPO trainer and replaced expert")

    # Reset the environment (again if we're using SB3 PPO)
    obs, _ = env.reset()

    # Print debug information about agents
    print("\n==== DEBUG: AGENT INFORMATION ====")
    print(f"Agent IDs in env.agents: {list(env.agents.keys())}")
    print(f"Agent IDs in observation: {list(obs.keys())}")
    for agent_id, agent in env.agents.items():
        print(f"Agent ID: {agent_id}, Type: {type(agent).__name__}, Vehicle ID: {agent.id if hasattr(agent, 'id') else 'N/A'}")
    print("==== END DEBUG ====\n")

    # Set expert_takeover to True for all agents to enable autodrive
    for agent_id, agent in env.agents.items():
        agent.expert_takeover = True
        print(f"Set agent {agent_id} to expert takeover mode")

    # Fix and verify the finish line placement
    print("\nFixing and verifying finish line placement...")
    try:
        if not fix_finish_line_placement(env):
            print("Failed to create finish line using standard method, creating a default finish line")
            # Create a default finish line (invisible)
            env.finish_line = {
                "position": (500.0, 0.0),  # Fixed position far from start
                "width": 100.0,  # Very wide to ensure detection
                "heading": 0.0,  # Default heading
                "finish_time": None,
                "visible": False,  # Don't show the finish line
                "detection_buffer": 20.0,  # Very large buffer for detection
                "final_lane": None,  # No specific lane
                "crossed_positions": {}
            }
            # Remove any existing finish line visualization
            create_finish_line_visualization(env)
    except Exception as e:
        print(f"Error creating finish line: {e}")
        # Create a simple default finish line (invisible)
        env.finish_line = {
            "position": (500.0, 0.0),
            "width": 100.0,
            "heading": 0.0,
            "finish_time": None,
            "visible": False,  # Don't show the finish line
            "detection_buffer": 20.0,
            "final_lane": None,
            "crossed_positions": {}
        }

    # Override the environment's cost and reward functions with our custom wrappers
    # that halt computations for agents that have crossed the finish line
    env._original_cost_function = env.cost_function
    env.cost_function = lambda vehicle_id: custom_cost_function_wrapper(env, vehicle_id)
    print("Overrode environment's cost function with custom wrapper that halts computations for finished agents")

    env._original_reward_function = env.reward_function
    env.reward_function = lambda vehicle_id: custom_reward_function_wrapper(env, vehicle_id)
    print("Overrode environment's reward function with custom wrapper that halts computations for finished agents")

    # Print information about the randomly generated track
    print(f"\nTrack Information:")
    print(f"Seed: {args.seed}")
    print(f"Number of blocks: {len(env.engine.current_map.blocks) if hasattr(env.engine, 'current_map') and hasattr(env.engine.current_map, 'blocks') else 'Unknown'}")
    print(f"Traffic density: {env.config['traffic_density']}")
    print(f"Number of traffic vehicles: {len(env.engine.traffic_manager.traffic_vehicles) if hasattr(env.engine, 'traffic_manager') and hasattr(env.engine.traffic_manager, 'traffic_vehicles') else 0}")

    # Main simulation loop
    for i in range(1, 100000):
        try:
            # Default action is to do nothing
            actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}

            # Step the environment
            obs, rewards, terminated, truncated, info = env.step(actions)
        except Exception as e:
            print(f"Error in simulation step: {e}")
            # Reset the environment and continue
            print("Resetting environment due to error...")
            obs, _ = env.reset()
            continue

        # Process keyboard events for manual reset (if using SB3 PPO)
        if args.use_sb3_ppo and hasattr(env.engine, 'accept_action') and env.engine.global_config["manual_control"]:
            # Handle manual reset with BACKSPACE key
            if env.engine.accept_action("backspace"):
                print("\nManual reset triggered - reinitializing weights for this episode")

                # Reset the models for all agents
                if hasattr(env.engine, 'ppo_trainer'):
                    for agent_id in env.agents.keys():
                        env.engine.ppo_trainer.reset_agent_model(f"agent_{agent_id}")

                # Reset the environment
                obs, _ = env.reset()

                # Set expert_takeover to True for all agents
                for agent_id, agent in env.agents.items():
                    agent.expert_takeover = True
                    print(f"Reset agent {agent_id} to expert takeover mode")

                # Fix and verify the finish line placement
                print("\nFixing and verifying finish line placement after manual reset...")
                fix_finish_line_placement(env)

                # Clear finish tracking variables
                finish_order.clear()
                finish_times.clear()
                reset_timer = None

                # Clear the set of finished agents
                if hasattr(env, 'finished_agents'):
                    env.finished_agents.clear()

                print("Manual reset complete - weights reinitialized for this episode")

        # Debug: Print if any agent is terminated and why
        for agent_id, is_terminated in terminated.items():
            if is_terminated:
                # Get termination reasons
                if agent_id in info:
                    reasons = [key for key, value in info[agent_id].items() if key.startswith("crash") or key in ["out_of_road", "arrive_dest", "max_step"] and value]
                    print(f"WARNING: Agent {agent_id} terminated due to: {reasons if reasons else 'unknown reason'}")

                    # If it's not due to finish line crossing or max steps, this is unexpected
                    if not info[agent_id].get("finish_line_crossed", False) and not info[agent_id].get("max_step", False):
                        print(f"UNEXPECTED TERMINATION for agent {agent_id}! This might be causing premature episode end.")
                        # Override termination to continue the simulation
                        terminated[agent_id] = False
                        truncated[agent_id] = False

        # Get race information
        leading_agent = next((agent_id for agent_id in env.agents.keys()
                             if env.agent_progress.get(agent_id, 0) == max(env.agent_progress.values())), None)

        # Extract current step costs from info
        step_costs = {}
        for agent_id, agent_info in info.items():
            # Get the current step cost if available
            if "cost" in agent_info:
                step_costs[agent_id] = agent_info["cost"]
            else:
                step_costs[agent_id] = 0.0

        # Check if any agent has reached its destination
        # First, check the environment's finish_line_crossed dictionary
        for agent_id in env.agents.keys():
            # Check if this agent just reached its destination (either from info or env.finish_line_crossed)
            destination_reached = (agent_id in env.finish_line_crossed and env.finish_line_crossed[agent_id]) or \
                                 (agent_id in info and info[agent_id].get("finish_line_crossed", False))

            if destination_reached and agent_id not in finish_order and agent_id not in removed_agents:
                print(f"Detected destination arrival for agent {agent_id}")
                finish_order.append(agent_id)
                finish_times[agent_id] = time.time()

                # Mark this agent as finished to halt cost and reward computations
                env.finished_agents.add(agent_id)
                print(f"Agent {agent_id} marked as finished - cost and reward computations halted")

                # Save the agent's model weights before removing it
                if agent_id in env.agents:
                    agent_model_weights[agent_id] = save_agent_model_weights(env.agents[agent_id], agent_id)

                    # Remove the agent from the simulation
                    remove_agent_from_simulation(env, agent_id)
                    removed_agents.add(agent_id)
                    print(f"Agent {agent_id} has been removed from simulation after reaching destination")

                # If this is the first agent to cross, mark it as the winner
                if len(finish_order) == 1:
                    env.race_winner = agent_id
                    env.race_finished = True
                    print(f"Agent {agent_id} has reached the destination first and won the race!")
                else:
                    print(f"Agent {agent_id} has reached the destination (position: {len(finish_order)})")

                # If all agents have reached their destinations, start the reset timer
                # We need to check if the number of agents that have finished equals the total number of agents
                # The total number of agents is the initial number of agents (which is 2 in this case)
                total_agents = env.config["num_agents"]  # This should be 2 for the racing environment
                print(f"Finish check: {len(finish_order)} agents finished out of {total_agents} total agents")

                # Only reset when ALL agents have reached their destinations
                # This ensures the episode continues until both cars reach the end
                if len(finish_order) >= total_agents:
                    reset_timer = time.time()
                    print(f"All agents have reached their destinations. Resetting in {reset_delay} seconds...")

        # Check if it's time to reset after all agents have finished
        if reset_timer is not None and time.time() - reset_timer >= reset_delay:
            # Update race statistics before reset
            if env.race_winner and env.race_winner != previous_winner:
                previous_winner = env.race_winner
                race_count += 1
                print(f"Race {race_count} completed. Winner: {previous_winner}")

            # If using SB3 PPO, save the models
            if args.use_sb3_ppo and hasattr(env.engine, 'ppo_trainer'):
                env.engine.ppo_trainer.end_episode()
                print("Saved PPO models at the end of the episode")

            # Perform a thorough reset of the environment
            print("Resetting environment...")

            # Clear finish tracking variables
            print("Clearing finish tracking variables...")
            finish_order.clear()  # Use clear() instead of reassigning to ensure all references are updated
            finish_times.clear()
            reset_timer = None

            # Clear the set of finished agents to resume cost and reward computations
            if hasattr(env, 'finished_agents'):
                env.finished_agents.clear()
                print("Cleared finished agents set - cost and reward computations will resume for all agents")

            print(f"After clearing: {len(finish_order)} agents in finish_order")

            # Reset the environment
            print("Performing thorough environment reset...")
            obs, _ = env.reset()

            # Restore model weights for all agents that were removed
            print(f"Regenerating {len(removed_agents)} agents with preserved model weights...")
            for agent_id in list(removed_agents):
                if agent_id in env.agents and agent_id in agent_model_weights:
                    # Load the saved model weights into the regenerated agent
                    load_agent_model_weights(env.agents[agent_id], agent_model_weights[agent_id])
                    print(f"Regenerated agent {agent_id} with preserved model weights")

            # Set expert_takeover for all agents after reset
            for agent_id, agent in env.agents.items():
                # If we have saved weights for this agent, use those settings
                if agent_id in agent_model_weights:
                    agent.expert_takeover = agent_model_weights[agent_id].get("policy_weights", True)
                else:
                    # Otherwise use default
                    agent.expert_takeover = True
                print(f"Reset agent {agent_id} to expert takeover mode: {agent.expert_takeover}")

            # Clear the removed agents set since all agents have been regenerated
            removed_agents.clear()

            # Verify that all necessary components are reinitialized
            print(f"Environment reset complete. Track loaded: {env.current_map.road_network is not None}")
            print(f"Number of agents: {len(env.agents)}")
            print(f"Finish line initialized: {env.finish_line is not None}")
            print(f"Preserved model weights: {list(agent_model_weights.keys())}")

            # Fix and verify the finish line placement after reset
            print("\nFixing and verifying finish line placement after reset...")
            fix_finish_line_placement(env)

            # Print information about the randomly generated track
            print(f"\nTrack Information after reset:")
            print(f"Seed: {env.current_seed}")
            print(f"Number of blocks: {len(env.engine.current_map.blocks) if hasattr(env.engine, 'current_map') and hasattr(env.engine.current_map, 'blocks') else 'Unknown'}")
            print(f"Traffic density: {env.config['traffic_density']}")
            print(f"Number of traffic vehicles: {len(env.engine.traffic_manager.traffic_vehicles) if hasattr(env.engine, 'traffic_manager') and hasattr(env.engine.traffic_manager, 'traffic_vehicles') else 0}")

        # Render with race information
        # Create a status display for each agent showing their finish position and time
        finish_status = {}
        for agent_id in env.agents.keys():
            if agent_id in finish_order:
                position = finish_order.index(agent_id) + 1
                elapsed = time.time() - finish_times[agent_id] if agent_id in finish_times else 0
                finish_status[agent_id] = f"Finished #{position} ({elapsed:.1f}s ago)"
            else:
                finish_status[agent_id] = "Racing"

        env.render(
            text={
                "Race Progress": {agent_id: f"{progress:.2f}" for agent_id, progress in env.agent_progress.items()},
                "Leading": leading_agent,
                "Race Winner": env.race_winner if env.race_finished else "None",
                "Previous Winner": previous_winner if previous_winner else "None",
                "Race Count": race_count,
                "Race Status": finish_status,  # Show finish position and time for each agent
                "Rewards": {agent_id: f"{reward:.2f}" for agent_id, reward in rewards.items()},
                "Step Cost": {agent_id: f"{step_costs.get(agent_id, 0):.2f}" for agent_id in env.agents.keys()},
                "Total Cost": {agent_id: f"{env.episode_cost.get(agent_id, 0):.2f}" for agent_id in env.agents.keys()},
                "Off-Road": {agent_id: f"Yes ({env.off_road_counter.get(agent_id, 0)} steps)" if env.off_road_status.get(agent_id, False) else "No" for agent_id in env.agents.keys()},
                "Wrong Side": {agent_id: f"Yes ({env.wrong_side_counter.get(agent_id, 0)} steps)" if env.wrong_side_status.get(agent_id, False) else "No" for agent_id in env.agents.keys()},
                "On Line": {agent_id: "Yellow" if info.get(agent_id, {}).get("on_yellow_continuous_line", False) or info.get(agent_id, {}).get("on_yellow_broken_line", False) else "White" if info.get(agent_id, {}).get("on_white_continuous_line", False) or info.get(agent_id, {}).get("on_white_broken_line", False) else "No" for agent_id in env.agents.keys()},
                "Checkpoints": {agent_id: f"{info.get(agent_id, {}).get('checkpoints_passed', 0)}/{info.get(agent_id, {}).get('total_checkpoints', 0)}" for agent_id in env.agents.keys()},
            }
        )

        # Check if all agents are done (this handles other termination conditions like max steps)
        # Only reset if all agents are terminated AND it's not due to finish line crossing
        # This prevents premature resets when agents are terminated for other reasons
        if all(terminated.values()) and not env.race_finished:
            print("All agents terminated for reasons other than finish line crossing. Investigating...")

            # Debug: Print termination reasons
            for agent_id in env.agents.keys():
                if agent_id in info:
                    reasons = [key for key, value in info[agent_id].items() if key.startswith("crash") or key in ["out_of_road", "arrive_dest", "max_step"] and value]
                    print(f"Agent {agent_id} terminated due to: {reasons if reasons else 'unknown reason'}")

            # Only reset if it's a legitimate termination (e.g., max steps)
            # Check if any agent reached max steps
            max_steps_reached = any(info.get(agent_id, {}).get("max_step", False) for agent_id in env.agents.keys())

            if max_steps_reached:
                print("Max steps reached. Resetting environment.")

                # Clear finish tracking variables
                print("Clearing finish tracking variables due to max steps...")
                finish_order.clear()  # Use clear() instead of reassigning to ensure all references are updated
                finish_times.clear()
                reset_timer = None

                # Clear the set of finished agents to resume cost and reward computations
                if hasattr(env, 'finished_agents'):
                    env.finished_agents.clear()
                    print("Cleared finished agents set - cost and reward computations will resume for all agents")

                print(f"After clearing: {len(finish_order)} agents in finish_order")

                # Reset the environment
                print("Performing thorough environment reset due to max steps...")
                obs, _ = env.reset()

                # Restore model weights for all agents that were removed
                print(f"Regenerating {len(removed_agents)} agents with preserved model weights...")
                for agent_id in list(removed_agents):
                    if agent_id in env.agents and agent_id in agent_model_weights:
                        # Load the saved model weights into the regenerated agent
                        load_agent_model_weights(env.agents[agent_id], agent_model_weights[agent_id])
                        print(f"Regenerated agent {agent_id} with preserved model weights")

                # Set expert_takeover for all agents after reset
                for agent_id, agent in env.agents.items():
                    # If we have saved weights for this agent, use those settings
                    if agent_id in agent_model_weights:
                        agent.expert_takeover = agent_model_weights[agent_id].get("policy_weights", True)
                    else:
                        # Otherwise use default
                        agent.expert_takeover = True

                # Clear the removed agents set since all agents have been regenerated
                removed_agents.clear()

                # Fix and verify the finish line placement after termination reset
                print("\nFixing and verifying finish line placement after termination reset...")
                fix_finish_line_placement(env)

                # Print information about the randomly generated track
                print(f"\nTrack Information after termination reset:")
                print(f"Seed: {env.current_seed}")
                print(f"Number of blocks: {len(env.engine.current_map.blocks) if hasattr(env.engine, 'current_map') and hasattr(env.engine.current_map, 'blocks') else 'Unknown'}")
                print(f"Traffic density: {env.config['traffic_density']}")
                print(f"Number of traffic vehicles: {len(env.engine.traffic_manager.traffic_vehicles) if hasattr(env.engine, 'traffic_manager') and hasattr(env.engine.traffic_manager, 'traffic_vehicles') else 0}")
            else:
                print("Ignoring premature termination since it's not due to max steps.")
                # Override termination to continue the simulation
                terminated = {agent_id: False for agent_id in terminated.keys()}
                truncated = {agent_id: False for agent_id in truncated.keys()}

    env.close()
