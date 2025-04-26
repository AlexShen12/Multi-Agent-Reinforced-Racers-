from metadrive.envs.hazardous_metadrive_env import HazardousMetaDriveEnv
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.policy.idm_policy import IDMPolicy

if __name__ == "__main__":
    # env = SafeMetaDriveEnv({
    #     # "force_render_fps": None,
    #     # "out_of_route_done":True, # I don't think this changed anything
    #     # "accident_prob": 0.4,
    #     "map": 4,
    #     # "use_mesh_terrain": True,
    #     "traffic_mode": "trigger",
    #     "traffic_density": 0.1, # 0.15-0.2 is good for training
    #     "out_of_route_done":True, # We don't want them cutting corners
    #     "on_continuous_line_done":True, # Good practice probably
    #     "agent_policy": IDMPolicy,
    #     # 'num_scenarios': 100,
    #     # "start_seed": 129,
    #     # "use_render": True,
    #     # "manual_control": False,
    #     # "vehicle_config": {
    #     #     "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 2),
    #     #     "show_dest_mark": True,
    #     #     "show_navi_mark": True,
    #     #     # Whether to draw a line from current vehicle position to the designation point
    #     #     "show_line_to_dest":True,
    #     #     # Whether to draw a line from current vehicle position to the next navigation point
    #     #     "show_line_to_navi_mark":True,
    #     #     # Whether to draw left / right arrow in the interface to denote the navigation direction
    #     #     "show_navigation_arrow":True,
    #     #     # If set to True, the vehicle will be in color green in top-down renderer or MARL setting
    #     #     "use_special_color":True,
    #     # },
    #     })
    env = HazardousMetaDriveEnv({
        "use_render": True,
    })
    # print(env.vehicle.lane_index)
    obs, info = env.reset(seed=0)
    total_cost = 0
    print(env.agent)
    try:
        for i in range(1, 10_000):
            if i % 100 == 0:
                print(f"Iteration {i}")
            obs, reward, terminated, truncated, info = env.step([0, 0])
            total_cost += info["cost"]
            ret = env.render(
                text={
                    "cost": total_cost,
                    "seed": env.current_seed,
                    "reward": reward,
                    "total_cost": info["total_cost"],
                }
            )
            if info["crash_vehicle"]:
                print("crash_vehicle:cost {}, reward {}".format(info["cost"], reward))
            if info["crash_object"]:
                print("crash_object:cost {}, reward {}".format(info["cost"], reward))
            if (terminated or truncated) and info["arrive_dest"]:
                total_cost = 0
                print("done_cost:{}".format(info["cost"]), "done_reward;{}".format(reward))
                print("Reset")
                env.reset(env.current_seed + 1)
    finally:
        env.close()
    print("Finished")