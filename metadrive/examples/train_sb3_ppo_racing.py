#!/usr/bin/env python
"""
Training script for Stable-Baselines3 PPO in MetaDrive Racing Environment

This script trains PPO agents using Stable-Baselines3 in the MetaDrive
multi-agent racing environment. It handles:
1. Environment setup and configuration
2. Training loop with episode-by-episode updates
3. Model persistence and loading
4. Handling of manual resets
"""

import argparse
import numpy as np

from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv
from metadrive.examples.sb3_ppo_policy import MultiAgentPPOTrainer, replace_expert


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agents for MetaDrive racing")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--models-dir", type=str, default="trained_models", help="Directory to save models")
    parser.add_argument("--manual-control", action="store_true", help="Enable manual control for testing")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    return parser.parse_args()


# Enhanced custom cost function with heavier penalties for wrong-side driving
def custom_cost_function(env, vehicle_id):
    """
    Enhanced custom cost function that applies:
    1. Same penalty for sidewalks and white lines as for going off-road
    2. Heavier penalty for wrong-side driving (yellow line crossing)
    3. Escalating penalties for continued violations

    Args:
        env: The environment object
        vehicle_id: The ID of the vehicle to calculate cost for

    Returns:
        tuple: (cost, step_info)
    """
    # Call the original cost function
    cost, step_info = env._original_cost_function(vehicle_id)

    # Get the vehicle
    vehicle = env.vehicles[vehicle_id]

    # Apply sidewalk penalty (same as out_of_road_cost)
    if hasattr(vehicle, 'crash_sidewalk') and vehicle.crash_sidewalk:
        sidewalk_cost = env.config["out_of_road_cost"]
        cost += sidewalk_cost
        step_info["sidewalk_cost"] = sidewalk_cost
        step_info["crash_sidewalk"] = True

    # Apply white line penalty (same as out_of_road_cost for solid lines)
    if hasattr(vehicle, 'on_white_continuous_line') and vehicle.on_white_continuous_line:
        white_line_cost = env.config["out_of_road_cost"]
        cost += white_line_cost
        step_info["white_line_cost"] = white_line_cost
        step_info["on_white_continuous_line"] = True

    # Apply enhanced wrong-side driving penalty (yellow line crossing)
    # Initialize wrong-side tracking if not already done
    if not hasattr(env, 'wrong_side_counter'):
        env.wrong_side_counter = {}
    if not hasattr(env, 'wrong_side_status'):
        env.wrong_side_status = {}

    # Initialize for this vehicle if needed
    if vehicle_id not in env.wrong_side_counter:
        env.wrong_side_counter[vehicle_id] = 0
    if vehicle_id not in env.wrong_side_status:
        env.wrong_side_status[vehicle_id] = False

    # Check if on yellow line (wrong side of road)
    on_wrong_side = False
    yellow_line_cost = 0.0

    if hasattr(vehicle, 'on_yellow_continuous_line') and vehicle.on_yellow_continuous_line:
        # Apply a much heavier penalty for continuous yellow line (2x the out_of_road_cost)
        yellow_line_cost += env.config["out_of_road_cost"] * 2.0
        step_info["on_yellow_continuous_line"] = True
        on_wrong_side = True
    if hasattr(vehicle, 'on_yellow_broken_line') and vehicle.on_yellow_broken_line:
        # Apply a slightly lower penalty for broken yellow line (1.5x the out_of_road_cost)
        yellow_line_cost += env.config["out_of_road_cost"] * 1.5
        step_info["on_yellow_broken_line"] = True
        on_wrong_side = True

    if on_wrong_side:
        # Check if this is a new wrong-side event or continuing
        if not env.wrong_side_status[vehicle_id]:
            # First time on wrong side
            env.wrong_side_status[vehicle_id] = True
            env.wrong_side_counter[vehicle_id] = 1
        else:
            # Continuing on wrong side, increment counter
            env.wrong_side_counter[vehicle_id] += 1

        # Apply exponential cost increase for staying on wrong side
        # More aggressive escalation than the default
        escalation_factor = 1.0 + 0.5 * (env.wrong_side_counter[vehicle_id] ** 1.8)
        yellow_line_cost *= escalation_factor

        cost += yellow_line_cost
        step_info["yellow_line_cost"] = yellow_line_cost
        step_info["wrong_side"] = True
        step_info["wrong_side_steps"] = env.wrong_side_counter[vehicle_id]

        # Print diagnostic information for significant wrong-side events
        if env.wrong_side_counter[vehicle_id] % 10 == 0:  # Every 10 steps
            print(f"WARNING: Agent {vehicle_id} has been on the wrong side for {env.wrong_side_counter[vehicle_id]} steps!")
            print(f"Applied escalating penalty: {yellow_line_cost:.2f} (factor: {escalation_factor:.2f})")
    else:
        # Reset wrong-side status when back on correct side
        if env.wrong_side_status[vehicle_id]:
            if env.wrong_side_counter[vehicle_id] > 10:  # Only print for significant events
                print(f"Agent {vehicle_id} returned to correct side after {env.wrong_side_counter[vehicle_id]} steps")
            env.wrong_side_status[vehicle_id] = False

    # Make sure the main cost field is updated with our total cost
    # This is critical for proper cost extraction in the training loop
    step_info["cost"] = cost

    # Update total cost
    if vehicle_id in env.episode_cost:
        env.episode_cost[vehicle_id] = env.episode_cost.get(vehicle_id, 0) + cost
        step_info["total_cost"] = env.episode_cost[vehicle_id]

    # Print diagnostic information for significant costs
    if cost > 5.0:  # Only print for significant costs
        print(f"[COST FUNCTION] Agent {vehicle_id} incurred significant cost: {cost:.2f}")
        # Print breakdown of cost components
        components = []
        if "sidewalk_cost" in step_info:
            components.append(f"sidewalk: {step_info['sidewalk_cost']:.2f}")
        if "white_line_cost" in step_info:
            components.append(f"white line: {step_info['white_line_cost']:.2f}")
        if "yellow_line_cost" in step_info:
            components.append(f"yellow line: {step_info['yellow_line_cost']:.2f}")
        if "off_road_cost" in step_info:
            components.append(f"off-road: {step_info['off_road_cost']:.2f}")
        if components:
            print(f"[COST BREAKDOWN] {', '.join(components)}")

    return cost, step_info


# Custom racing rewards function that penalizes the trailing agent
def custom_racing_rewards(env, rewards, info):
    """
    Custom racing rewards function that adds a penalty for the trailing agent.
    The trailing agent gets a penalty of 1/5th the reward that the leading agent receives.

    Args:
        env: The environment object
        rewards: The original rewards dictionary
        info: The info dictionary

    Returns:
        dict: Modified rewards dictionary
    """
    # Call the original racing rewards function
    modified_rewards = env._original_apply_racing_rewards(rewards, info)

    # Find the leading agent
    if len(env.agent_progress) > 1:  # Only apply if we have multiple agents
        # Get the agent with the most progress
        progress_items = sorted(env.agent_progress.items(), key=lambda x: x[1], reverse=True)

        if len(progress_items) >= 2:
            # Get the leading agent and trailing agent
            leading_agent_id, leading_progress = progress_items[0]
            trailing_agent_id, trailing_progress = progress_items[1]

            # Calculate the leading bonus that was applied
            leading_bonus = env.config["leading_reward_factor"]
            lead_margin = leading_progress - trailing_progress
            margin_bonus = min(lead_margin * 0.01, 0.1)  # Cap at 0.1
            total_leading_bonus = leading_bonus + margin_bonus

            # Apply a penalty to the trailing agent (1/5th of the leading bonus)
            trailing_penalty = -total_leading_bonus / 5.0
            modified_rewards[trailing_agent_id] += trailing_penalty

            # Add info about the trailing penalty
            if trailing_agent_id in info:
                info[trailing_agent_id]["trailing_penalty"] = trailing_penalty

            print(f"Applied trailing penalty of {trailing_penalty:.4f} to agent {trailing_agent_id}")

    return modified_rewards


def create_racing_env(args):
    """Create and configure the racing environment."""
    env = MultiAgentRacingSafeEnv(
        {
            "use_render": args.render,
            "manual_control": args.manual_control,
            "num_agents": 2,
            "start_seed": args.seed,
            "horizon": 2000,

            # ===== Random Track Generation =====
            "num_scenarios": 100,
            "map": 5,
            "random_agent_model": False,
            "random_lane_width": False,
            "random_lane_num": False,

            # ===== Traffic Settings =====
            "traffic_density": 0.15,
            "traffic_mode": "trigger",
            "random_traffic": True,

            # ===== Agent Settings =====
            "use_AI_protector": True,

            # ===== Finish Line Settings =====
            "enable_finish_line": True,
            "finish_line_at_end": True,
            "terminate_on_finish": False,

            # ===== Termination Settings =====
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "out_of_road_done": False,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,
            "out_of_route_done": False,
            "crash_done": False,
            "truncate_as_terminate": False,

            # ===== Cost Settings =====
            "out_of_road_cost": 20.0,  # Base cost for going off-road

            # ===== Vehicle Configuration =====
            "vehicle_config": {
                "enable_reverse": False,  # Disable reverse driving
                "show_lidar": False,
                "show_side_detector": False,
                "show_lane_line_detector": False,
            }
        }
    )

    # Initialize the environment by calling reset
    # This ensures the engine is created and ready to use
    print("Initializing environment...")
    env.reset()
    print("Environment initialized successfully.")

    # Verify that the engine is properly initialized
    if hasattr(env, 'engine') and env.engine is not None:
        print("Engine is properly initialized.")
    else:
        print("Warning: Engine is not properly initialized!")

    # Override the environment's racing rewards function with our custom one
    # that penalizes the trailing agent
    env._original_apply_racing_rewards = env._apply_racing_rewards
    env._apply_racing_rewards = lambda rewards, info: custom_racing_rewards(env, rewards, info)
    print("Applied custom racing rewards function with trailing agent penalty")

    # Override the environment's cost function with our enhanced version
    # that applies heavier penalties for wrong-side driving
    env._original_cost_function = env.cost_function
    env.cost_function = lambda vehicle_id: custom_cost_function(env, vehicle_id)
    print("Applied enhanced cost function with heavier wrong-side driving penalties")

    return env


def setup_ppo_trainer(env, args):
    """Set up the PPO trainer for multi-agent training."""
    # Get first agent's observation and action spaces
    first_agent_id = next(iter(env.observation_space.spaces))
    observation_space = env.observation_space.spaces[first_agent_id]
    action_space = env.action_space.spaces[first_agent_id]

    # Create the PPO trainer
    ppo_trainer = MultiAgentPPOTrainer(
        observation_space=observation_space,
        action_space=action_space,
        num_agents=env.config["num_agents"],
        models_dir=args.models_dir,
        verbose=args.verbose,
        use_expert_weights=True  # Use expert weights instead of random initialization
    )

    # Set the global PPO trainer
    import metadrive.examples.sb3_ppo_policy as sb3_policy
    sb3_policy.global_ppo_trainer = ppo_trainer
    print("Set global PPO trainer successfully")

    # Attach trainer to environment engine if available
    if getattr(env, 'engine', None) is not None:
        env.engine.ppo_trainer = ppo_trainer
        print("Attached PPO trainer to environment engine")
    else:
        # Try initializing the engine
        env.reset()
        if getattr(env, 'engine', None) is not None:
            env.engine.ppo_trainer = ppo_trainer
            print("Attached PPO trainer to environment engine after reset")
        else:
            print("Note: Using global trainer only (engine not available)")

    return ppo_trainer


def collect_episode_experiences(env, ppo_trainer, episode_num):
    """
    Run an episode and collect experiences for training.

    Returns:
        episode_info: Dictionary with episode statistics
        experiences: Dictionary mapping agent_ids to lists of (obs, action, reward, done, next_obs)
    """
    # Reset the environment
    obs, _ = env.reset()

    # Enable autodrive for all agents
    for agent in env.agents.values():
        agent.expert_takeover = True

    # Initialize tracking variables
    experiences = {agent_id: [] for agent_id in obs}
    episode_rewards = {agent_id: 0 for agent_id in obs}
    episode_costs = {agent_id: 0 for agent_id in obs}
    step_rewards = {agent_id: 0 for agent_id in obs}
    step_costs = {agent_id: 0 for agent_id in obs}
    episode_steps = 0

    # Initialize race status tracking
    race_status = {agent_id: "Racing" for agent_id in obs}
    agent_progress = {}
    finish_order = []
    finish_times = {}
    leading_agent = None

    # Run the episode until done
    while True:
        episode_steps += 1

        # Get actions from PPO models
        actions = {}
        for agent_id, agent_obs in obs.items():
            # Get action from the PPO trainer
            action, _ = ppo_trainer.predict(f"agent_{agent_id}", agent_obs, deterministic=False)

            # Ensure action is in the correct format (1D array)
            import numpy as np
            if isinstance(action, np.ndarray) and len(action.shape) > 1:
                # If it's a 2D array (e.g., [[steer, throttle]]), flatten it
                action = action.flatten()

            # Print debug info occasionally
            if np.random.random() < 0.01:  # 1% of the time
                print(f"Action for agent {agent_id}: {action}, shape: {action.shape if isinstance(action, np.ndarray) else 'not numpy'}")

            actions[agent_id] = action

        # Step the environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        # Update step rewards and costs
        step_rewards = {agent_id: rewards.get(agent_id, 0) for agent_id in env.agents.keys()}
        step_costs = {}

        # Extract costs from info with improved handling
        for agent_id, agent_info in info.items():
            # Initialize step cost for this agent
            current_step_cost = 0

            # Extract the main cost value
            if "cost" in agent_info:
                current_step_cost = agent_info["cost"]

            # Also check for individual cost components that might not be included in the main cost
            # This ensures we capture all costs from our custom cost function
            if "sidewalk_cost" in agent_info:
                current_step_cost += agent_info["sidewalk_cost"]
            if "white_line_cost" in agent_info:
                current_step_cost += agent_info["white_line_cost"]
            if "yellow_line_cost" in agent_info:
                current_step_cost += agent_info["yellow_line_cost"]
            if "off_road_cost" in agent_info:
                current_step_cost += agent_info["off_road_cost"]

            # Update step and episode costs
            step_costs[agent_id] = current_step_cost
            episode_costs[agent_id] = episode_costs.get(agent_id, 0) + current_step_cost

            # Print diagnostic information for significant costs
            if current_step_cost > 0:
                print(f"Agent {agent_id} incurred cost: {current_step_cost:.2f} at step {episode_steps}")

        # Update race status
        for agent_id in env.agents.keys():
            # Check if agent has crossed finish line
            if hasattr(env, 'finish_line_crossed') and agent_id in env.finish_line_crossed and env.finish_line_crossed[agent_id]:
                if agent_id not in finish_order:
                    finish_order.append(agent_id)
                    finish_times[agent_id] = episode_steps
                race_status[agent_id] = f"Finished #{finish_order.index(agent_id) + 1}"
            elif terminated.get(agent_id, False) or truncated.get(agent_id, False):
                race_status[agent_id] = "Terminated"
            else:
                race_status[agent_id] = "Racing"

            # Update progress
            if hasattr(env, 'agent_progress'):
                agent_progress = env.agent_progress
            elif hasattr(env.agents[agent_id], 'navigation') and hasattr(env.agents[agent_id].navigation, 'progress'):
                agent_progress[agent_id] = env.agents[agent_id].navigation.progress

        # Determine leading agent
        if agent_progress:
            max_progress = -1
            for agent_id, progress in agent_progress.items():
                if progress > max_progress and race_status[agent_id] == "Racing":
                    max_progress = progress
                    leading_agent = agent_id

        # Store experiences for active agents
        for agent_id in obs:
            if agent_id in next_obs:  # Agent still active
                # Get the cost for this agent
                agent_cost = step_costs.get(agent_id, 0.0)

                # Adjust reward based on cost (penalize the agent for incurring costs)
                # This ensures costs are factored into the learning process
                adjusted_reward = rewards[agent_id] - agent_cost

                # Store (obs, action, adjusted_reward, done, next_obs) tuple
                experiences[agent_id].append((
                    obs[agent_id],
                    actions[agent_id],
                    adjusted_reward,  # Use the cost-adjusted reward
                    terminated.get(agent_id, False) or truncated.get(agent_id, False),
                    next_obs[agent_id]
                ))

                # Update episode rewards (using the adjusted reward)
                episode_rewards[agent_id] += adjusted_reward

                # Print diagnostic information for significant cost adjustments
                if agent_cost > 0:
                    print(f"Adjusted reward for agent {agent_id}: {rewards[agent_id]:.2f} - {agent_cost:.2f} = {adjusted_reward:.2f}")

        # Render with diagnostic information
        if hasattr(env, 'render') and callable(env.render):
            try:
                # Calculate adjusted step rewards for display
                adjusted_step_rewards = {}
                for agent_id in env.agents.keys():
                    raw_reward = step_rewards.get(agent_id, 0.0)
                    cost = step_costs.get(agent_id, 0.0)
                    adjusted_step_rewards[agent_id] = raw_reward - cost

                # Prepare text display with enhanced reward and cost information
                render_text = {
                    "Episode": episode_num,
                    "Step": episode_steps,
                    "Race Status": race_status,
                    "Leading": leading_agent if leading_agent else "None",
                    "Race Winner": getattr(env, "race_winner", None) if getattr(env, "race_finished", False) else "None",
                    "Progress": {agent_id: f"{progress:.1f}%" for agent_id, progress in agent_progress.items()},
                    "Raw Reward": {agent_id: f"{reward:.2f}" for agent_id, reward in step_rewards.items()},
                    "Step Cost": {agent_id: f"{cost:.2f}" for agent_id, cost in step_costs.items()},
                    "Adjusted Reward": {agent_id: f"{reward:.2f}" for agent_id, reward in adjusted_step_rewards.items()},
                    "Total Reward": {agent_id: f"{reward:.2f}" for agent_id, reward in episode_rewards.items()},
                    "Total Cost": {agent_id: f"{cost:.2f}" for agent_id, cost in episode_costs.items()},
                }

                # Add trailing penalty info if available
                trailing_penalties = {}
                for agent_id in env.agents.keys():
                    if agent_id in info and "trailing_penalty" in info[agent_id]:
                        trailing_penalties[agent_id] = f"{info[agent_id]['trailing_penalty']:.2f}"
                if trailing_penalties:
                    render_text["Trailing Penalty"] = trailing_penalties

                # Add enhanced off-road and wrong-side status with step counters if available
                if hasattr(env, 'off_road_status'):
                    render_text["Off-Road"] = {agent_id: f"Yes ({env.off_road_counter.get(agent_id, 0)} steps)" if env.off_road_status.get(agent_id, False) else "No" for agent_id in env.agents.keys()}

                if hasattr(env, 'wrong_side_status'):
                    render_text["Wrong Side"] = {agent_id: f"Yes ({env.wrong_side_counter.get(agent_id, 0)} steps)" if env.wrong_side_status.get(agent_id, False) else "No" for agent_id in env.agents.keys()}

                # Add detailed cost breakdown
                cost_breakdown = {}
                for agent_id in env.agents.keys():
                    if agent_id in info:
                        costs = []
                        if "off_road_cost" in info[agent_id]:
                            costs.append(f"Off-road: {info[agent_id]['off_road_cost']:.2f}")
                        if "yellow_line_cost" in info[agent_id]:
                            costs.append(f"Wrong-side: {info[agent_id]['yellow_line_cost']:.2f}")
                        if "white_line_cost" in info[agent_id]:
                            costs.append(f"White-line: {info[agent_id]['white_line_cost']:.2f}")
                        if "sidewalk_cost" in info[agent_id]:
                            costs.append(f"Sidewalk: {info[agent_id]['sidewalk_cost']:.2f}")
                        if "crash_vehicle_cost" in info[agent_id]:
                            costs.append(f"Crash: {info[agent_id]['crash_vehicle_cost']:.2f}")
                        if costs:
                            cost_breakdown[agent_id] = ", ".join(costs)
                if cost_breakdown:
                    render_text["Cost Breakdown"] = cost_breakdown

                # Add line crossing status
                render_text["On Line"] = {}
                for agent_id in env.agents.keys():
                    if agent_id in info:
                        if info[agent_id].get("on_yellow_continuous_line", False) or info[agent_id].get("on_yellow_broken_line", False):
                            render_text["On Line"][agent_id] = "Yellow"
                        elif info[agent_id].get("on_white_continuous_line", False) or info[agent_id].get("on_white_broken_line", False):
                            render_text["On Line"][agent_id] = "White"
                        else:
                            render_text["On Line"][agent_id] = "No"
                    else:
                        render_text["On Line"][agent_id] = "No"

                # Render with text display
                env.render(text=render_text)
            except Exception as e:
                print(f"Error rendering diagnostics: {e}")

        # Update observations
        obs = next_obs

        # Check if episode is done
        if all(terminated.values()) or all(truncated.values()) or episode_steps >= env.config["horizon"]:
            break

    # Return episode statistics and collected experiences
    return {
        "episode_num": episode_num,
        "episode_steps": episode_steps,
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
        "race_winner": getattr(env, "race_winner", None),
        "race_status": race_status,
        "finish_order": finish_order
    }, experiences


def train_on_episode(ppo_trainer, experiences):
    """Train the PPO models on collected experiences."""
    for agent_id, agent_experiences in experiences.items():
        # Skip empty experiences or reset models
        trainer_id = f"agent_{agent_id}"
        if not agent_experiences or ppo_trainer.reset_status.get(trainer_id, False):
            continue

        # Convert experience tuples to numpy arrays
        # Each experience is (obs, action, reward, done, next_obs)
        obs_batch = np.array([exp[0] for exp in agent_experiences])

        # Train the agent's model
        ppo_trainer.train_on_batch(
            trainer_id,
            obs_batch,
            # Pass other data as keyword arguments
            actions=np.array([exp[1] for exp in agent_experiences]),
            rewards=np.array([exp[2] for exp in agent_experiences]),
            dones=np.array([exp[3] for exp in agent_experiences]),
            next_observations=np.array([exp[4] for exp in agent_experiences])
        )


def main():
    """Main training loop."""
    args = parse_args()
    replace_expert()  # Replace original expert with SB3 PPO expert

    # Setup environment and trainer
    env = create_racing_env(args)
    ppo_trainer = setup_ppo_trainer(env, args)

    # Initialize tracking variables for overall statistics
    total_rewards = {}
    total_costs = {}
    wins_by_agent = {}

    # Training loop
    print(f"Starting training for {args.episodes} episodes")
    for episode in range(1, args.episodes + 1):
        print(f"\nEpisode {episode}/{args.episodes}")

        # Run episode and collect experiences
        stats, experiences = collect_episode_experiences(env, ppo_trainer, episode)

        # Print episode results
        print(f"Steps: {stats['episode_steps']}")

        # Print race status
        print("\nRace Status:")
        for agent_id, status in stats.get('race_status', {}).items():
            print(f"  {agent_id}: {status}")
        if stats['race_winner']:
            print(f"Winner: {stats['race_winner']}")

        # Print reward and cost information
        print("\nRewards:")
        for agent_id, reward in stats['episode_rewards'].items():
            print(f"  {agent_id}: {reward:.2f}")
        print("Costs:")
        for agent_id, cost in stats.get('episode_costs', {}).items():
            print(f"  {agent_id}: {cost:.2f}")

        # Print finish order if available
        if 'finish_order' in stats and stats['finish_order']:
            print("\nFinish Order:")
            for i, agent_id in enumerate(stats['finish_order']):
                print(f"  {i+1}. {agent_id}")

        # Update overall statistics
        for agent_id, reward in stats['episode_rewards'].items():
            if agent_id not in total_rewards:
                total_rewards[agent_id] = 0
            total_rewards[agent_id] += reward

        for agent_id, cost in stats.get('episode_costs', {}).items():
            if agent_id not in total_costs:
                total_costs[agent_id] = 0
            total_costs[agent_id] += cost

        if stats['race_winner']:
            winner = stats['race_winner']
            if winner not in wins_by_agent:
                wins_by_agent[winner] = 0
            wins_by_agent[winner] += 1

        # Train models and save progress
        train_on_episode(ppo_trainer, experiences)
        ppo_trainer.end_episode()

        # Print training progress every 5 episodes
        if episode % 5 == 0 or episode == args.episodes:
            print(f"\n===== Training Progress after {episode} Episodes =====")
            print("Average Rewards:")
            for agent_id, reward in total_rewards.items():
                avg_reward = reward / episode
                print(f"  {agent_id}: {avg_reward:.2f}")

            print("Average Costs:")
            for agent_id, cost in total_costs.items():
                avg_cost = cost / episode
                print(f"  {agent_id}: {avg_cost:.2f}")

            # Calculate reward-to-cost ratio
            print("Reward-to-Cost Ratio:")
            for agent_id in total_rewards.keys():
                if agent_id in total_costs and total_costs[agent_id] > 0:
                    ratio = total_rewards[agent_id] / max(total_costs[agent_id], 0.1)  # Avoid division by zero
                    print(f"  {agent_id}: {ratio:.2f}")
                else:
                    print(f"  {agent_id}: N/A (no costs)")

            print("Wins by Agent:")
            for agent_id, wins in wins_by_agent.items():
                win_rate = (wins / episode) * 100
                print(f"  {agent_id}: {wins} wins ({win_rate:.1f}%)")

    # Print final training statistics
    print("\n========== Final Training Statistics ===========")
    print(f"Total Episodes: {args.episodes}")
    print("Average Rewards:")
    for agent_id, reward in total_rewards.items():
        avg_reward = reward / args.episodes
        print(f"  {agent_id}: {avg_reward:.2f}")

    print("Average Costs:")
    for agent_id, cost in total_costs.items():
        avg_cost = cost / args.episodes
        print(f"  {agent_id}: {avg_cost:.2f}")

    print("Wins by Agent:")
    for agent_id, wins in wins_by_agent.items():
        win_rate = (wins / args.episodes) * 100
        print(f"  {agent_id}: {wins} wins ({win_rate:.1f}%)")

    # Cleanup
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
