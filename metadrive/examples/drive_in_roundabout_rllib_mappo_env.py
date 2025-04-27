import torch
import numpy as np
from metadrive.envs.marl_envs.roundabout_rllib_delegator_env import RoundaboutRLLibDelegatorEnv 
import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import RLModule
import os

# THIS FILE IS FOR USING THE TRAINED RLLib MAPPO ON METADRIVE's GIVEN ROUNDABOUT ENV

checkpoint_base_directory ="/Users/jameswalker/Programming/Checkpoints3/MappoCheckpoint"
checkpoint_number = 2
checkpoint_directory = checkpoint_base_directory + str(checkpoint_number)

agent0_rl_module = RLModule.from_checkpoint(
        os.path.join(
            checkpoint_directory,
            "learner_group",
            "learner",
            "rl_module",
            "agent0",
        )
    )
agent1_rl_module = RLModule.from_checkpoint(
        os.path.join(
            checkpoint_directory,
            "learner_group",
            "learner",
            "rl_module",
            "agent1",
        )
    )

# modules that were created with RLModule.from_checkpoint(...)
agent_modules = {
    "agent0": agent0_rl_module,
    "agent1": agent1_rl_module,
}
action_dist_cls = {
    agent_id: rl_module.get_inference_action_dist_cls() for agent_id, rl_module in agent_modules.items()
}

env = RoundaboutRLLibDelegatorEnv(
    {
        "use_render": True,
        # "manual_control": True, # If you want to drive agent0 by keyboard
    }
)
obs, info = env.reset()

try:
    for step in range(1, 10_000):
        if step % 100 == 0:
            print(f"Iteration {step}")

        # build a separate forward-pass dict for every agent
        actions = {}
        with torch.no_grad():
            for agent_id, single_obs in obs.items():
                fwd_inputs = {"obs": torch.as_tensor(single_obs).unsqueeze(0)}
                out = agent_modules[agent_id].forward_inference(fwd_inputs)
                dist = action_dist_cls[agent_id].from_logits(out["action_dist_inputs"])
                actions[agent_id] = dist.sample()[0].cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(actions)

        # Check if all agents are done
        if all(terminated.values()) or all(truncated.values()):
            print("All agents terminated. Resetting environment.")
            obs, _ = env.reset()
finally:
    env.close()
print("Finished")