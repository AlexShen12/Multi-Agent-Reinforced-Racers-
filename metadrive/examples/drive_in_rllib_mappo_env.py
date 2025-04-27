import torch
import numpy as np
from metadrive.envs.marl_envs.rllib_mappo_env import RLLibMappoEnv
import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import RLModule
import os
from metadrive.envs.rllib_delegator_env import RLLibDelegatorEnv 

checkpoint_base_directory ="/Users/jameswalker/Programming/Checkpoints/MappoCheckpoint"
checkpoint_number = 6
checkpoint_directory = checkpoint_base_directory + str(checkpoint_number)

iterations=1_000
# iterations=10_000

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
    aid: mod.get_inference_action_dist_cls() for aid, mod in agent_modules.items()
}
# Training uses RLLibDelegatorEnv because it defines its actions and observations and possible agents and agents
# env = RLLibDelegatorEnv({ "use_render": True,})
env = RLLibMappoEnv(
    {
        "use_render": True,
        # "manual_control": True,
    }
)
obs, info = env.reset(seed=0)

episode_cost = {"agent0": 0, "agent1": 0}
# print(f"Printing leading_reward_factor: {env.config['leading_reward_factor']}")

try:
    for step in range(1, iterations):
        if step % 100 == 0:
            print(f"Iteration {step}")

        # build a separate forward-pass dict for every agent
        actions = {}
        with torch.no_grad():
            # aid is short for agent_id
            for aid, ob in obs.items():
                # ChatGPT says this way is faster/better
                fwd_ins = {"obs": torch.as_tensor(ob).unsqueeze(0)}
                out = agent_modules[aid].forward_exploration(fwd_ins)
                dist = action_dist_cls[aid].from_logits(out["action_dist_inputs"])
                actions[aid] = dist.sample()[0].cpu().numpy()

                # Compare with old
                # old_fwd_ins = {"obs": torch.Tensor([ob])}
                # old_fwd_outputs = agent_modules[aid].forward_exploration(old_fwd_ins)
                # old_action_dist = action_dist_cls[aid].from_logits(old_fwd_outputs["action_dist_inputs"])
                # old_action = old_action_dist.sample()[0].numpy()
        print("Actions: ", actions)
        obs, reward, terminated, truncated, info = env.step(actions)

        # if env.config["use_render"]:
        #     # Render with race information
        #     env.render(
        #         text={
        #             "Race Progress": {agent_id: f"{progress:.2f}" for agent_id, progress in env.agent_progress.items()},
        #             "Leading": next((agent_id for agent_id in env.agents.keys() if env.agent_progress.get(agent_id, 0) == max(env.agent_progress.values())), None),
        #             "Race Winner": env.race_winner if env.race_finished else "None",
        #         }
        #     )

        # Check if all agents are done
        if all(terminated.values()):
            print("All agents terminated. Resetting environment.")
            obs, _ = env.reset()
finally:
    env.close()
print("Finished")