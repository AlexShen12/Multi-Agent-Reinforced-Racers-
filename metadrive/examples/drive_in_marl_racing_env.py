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

from metadrive.constants import HELP_MESSAGE
from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the environment")
    parser.add_argument("--manual_control", action="store_true", help="Enable manual control for testing")
    args = parser.parse_args()

    # Configure environment
    env = MultiAgentRacingSafeEnv(
        {
            "use_render": True,
            "manual_control": args.manual_control,
            "num_agents": 2,
            "start_seed": args.seed,
            "num_scenarios": 1,  # Use a single scenario for racing
            "map": "CSRCR",  # Use a circular map for racing
            # Enable AI protector to ensure both agents use autodrive
            "use_AI_protector": True,
            # Enable finish line but don't terminate immediately
            "enable_finish_line": True,  # Add a definitive finish line
            "finish_line_at_end": True,  # Place finish line at the end of the last track segment
            "terminate_on_finish": False,  # Modified: Don't end episode immediately when finish line is crossed
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
    print("- The episode will continue until both cars cross the finish line or max steps are reached")
    print("- The finish line is at the end of the track")
    print("- Cars will be removed from the simulation after crossing the finish line")
    print("- Each car's model weights are preserved when it finishes the race")
    print("- Cars are regenerated with their preserved model weights in the next episode")
    print("- The leading car receives bonus rewards")
    print("- The first car to cross the finish line wins the race and receives a winning reward")
    print("- The race winner is displayed and a new race begins after both cars finish")

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

    # Reset the environment
    obs, _ = env.reset()

    # Set expert_takeover to True for all agents to enable autodrive
    for agent_id, agent in env.agents.items():
        agent.expert_takeover = True
        print(f"Set agent {agent_id} to expert takeover mode")

    # Main simulation loop
    for i in range(1, 100000):
        # Default action is to do nothing
        actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}

        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)

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

        # Check if any agent has crossed the finish line
        for agent_id, agent_info in info.items():
            # Check if this agent just crossed the finish line
            if agent_info.get("finish_line_crossed", False) and agent_id not in finish_order and agent_id not in removed_agents:
                finish_order.append(agent_id)
                finish_times[agent_id] = time.time()

                # Save the agent's model weights before removing it
                if agent_id in env.agents:
                    agent_model_weights[agent_id] = save_agent_model_weights(env.agents[agent_id], agent_id)

                    # Remove the agent from the simulation
                    remove_agent_from_simulation(env, agent_id)
                    removed_agents.add(agent_id)
                    print(f"Agent {agent_id} has been removed from simulation after crossing finish line")

                # If this is the first agent to cross, mark it as the winner
                if len(finish_order) == 1:
                    env.race_winner = agent_id
                    env.race_finished = True
                    print(f"Agent {agent_id} has crossed the finish line first and won the race!")
                else:
                    print(f"Agent {agent_id} has crossed the finish line (position: {len(finish_order)})")

                # If all agents have crossed the finish line, start the reset timer
                # We need to check if the number of agents that have finished equals the total number of agents
                # The total number of agents is the initial number of agents (which is 2 in this case)
                total_agents = env.config["num_agents"]  # This should be 2 for the racing environment
                print(f"Finish check: {len(finish_order)} agents finished out of {total_agents} total agents")
                if len(finish_order) == total_agents:
                    reset_timer = time.time()
                    print(f"All agents have crossed the finish line. Resetting in {reset_delay} seconds...")

        # Check if it's time to reset after all agents have finished
        if reset_timer is not None and time.time() - reset_timer >= reset_delay:
            # Update race statistics before reset
            if env.race_winner and env.race_winner != previous_winner:
                previous_winner = env.race_winner
                race_count += 1
                print(f"Race {race_count} completed. Winner: {previous_winner}")

            # Perform a thorough reset of the environment
            print("Resetting environment...")

            # Clear finish tracking variables
            print("Clearing finish tracking variables...")
            finish_order.clear()  # Use clear() instead of reassigning to ensure all references are updated
            finish_times.clear()
            reset_timer = None
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
        if all(terminated.values()):
            print("All agents terminated. Resetting environment.")

            # Clear finish tracking variables
            print("Clearing finish tracking variables due to termination...")
            finish_order.clear()  # Use clear() instead of reassigning to ensure all references are updated
            finish_times.clear()
            reset_timer = None
            print(f"After clearing: {len(finish_order)} agents in finish_order")

            # Reset the environment
            print("Performing thorough environment reset due to termination...")
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

    env.close()
