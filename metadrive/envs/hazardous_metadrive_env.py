from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.utils import Config
from metadrive.component.sensors.lidar import Lidar
from metadrive.component.sensors.distance_detector import LaneLineDetector, SideDetector
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT

import logging

HAZARDOUS_DEFAULT_CONFIG = dict(
    # ===== safe env ===== FROM SafeMetaDriveEnv
    start_seed=129, # From SafeMetaDriveEnv
    num_scenarios=100,
    accident_prob=0.5, # From SafeMetaDriveEnv
    # accident_prob=0.15,
    traffic_density=0.15, # From SafeMetaDriveEnv
    crash_vehicle_done=False,
    crash_object_done=False,
    cost_to_reward=False,
    horizon=1000,

    # ===== Traffic ===== FROM MetaDriveEnv
    # traffic_density=0.1,
    need_inverse_traffic=False,
    traffic_mode="trigger",  # "Respawn", "Trigger"
    random_traffic=False,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    # traffic_vehicle_config=dict(
    #     show_navi_mark=False,
    #     show_dest_mark=False,
    #     enable_reverse=False,
    #     show_lidar=False,
    #     show_lane_line_detector=False,
    #     show_side_detector=False,
    # ),

    # ===== PG Map Config ===== FROM MetaDriveEnv
    map=3,  # int or string: an easy way to fill map_config
    # map="CSRCR" # This is that one map in drive_in_safe_metadrive_env
    block_dist_config=PGBlockDistConfig,
    random_lane_width=False,
    random_lane_num=False,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 50,
        "start_position": [0, 0],
    },
    store_map=True,

    # ===== agent =====
    # Whether randomize the car model for the agent, randomly choosing from 4 types of cars
    random_agent_model=False,

    # ===== multi-agent =====
    # This should be >1 in MARL envs, or set to -1 for spawning as many vehicles as possible.
    num_agents=1,
    # Turn on this to notify the simulator that it is MARL env
    is_multi_agent=False,
    # The number of agent will be fixed adn determined at the start of the episode, if set to False
    allow_respawn=False,
    # How many substeps for the agent to stay static at the death place after done. (Default for MARL: 25)
    delay_done=0,

    # ===== Action/Control =====
    # Please see Documentation: Action and Policy for more details
    # What policy to use for controlling agents
    agent_policy=EnvInputPolicy,
    # If set to True, agent_policy will be overriden and change to ManualControlPolicy
    manual_control=False,
    # What interfaces to use for manual control, options: "steering_wheel" or "keyboard" or "xbos"
    controller="keyboard",
    # Used with EnvInputPolicy. If set to True, the env.action_space will be discrete
    discrete_action=False,
    # If True, use MultiDiscrete action space. Otherwise, use Discrete.
    use_multi_discrete=False,
    # How many discrete actions are used for steering dim
    discrete_steering_dim=5,
    # How many discrete actions are used for throttle/brake dim
    discrete_throttle_dim=5,
    # Check if the action is contained in gym.space. Usually turned off to speed up simulation
    action_check=False,

    # ===== Observation =====
    # Please see Documentation: Observation for more details
    # Whether to normalize the pixel value from 0-255 to 0-1
    norm_pixel=True,
    # The number of timesteps for stacking image observation
    stack_size=3,
    # Whether to use image observation or lidar. It takes effect in get_single_observation
    image_observation=False,
    # Like agent_policy, users can use customized observation class through this field
    agent_observation=None,

    # ===== Termination =====
    # The maximum length of each agent episode. Set to None to remove this constraint
    # horizon=None,
    # If set to True, the terminated will be True as well when the length of agent episode exceeds horizon
    truncate_as_terminate=False,

    # ===== Main Camera =====
    # A True value makes the camera follow the reference line instead of the vehicle, making its movement smooth
    use_chase_camera_follow_lane=False,
    # Height of the main camera
    camera_height=2.2,
    # Distance between the camera and the vehicle. It is the distance projecting to the x-y plane.
    camera_dist=7.5,
    # Pitch of main camera. If None, this will be automatically calculated
    camera_pitch=None,  # degree
    # Smooth the camera movement
    camera_smooth=True,
    # How many frames used to smooth the camera
    camera_smooth_buffer_size=20,
    # FOV of main camera
    camera_fov=65,
    # Only available in MARL setting, choosing which agent to track. Values should be "agent0", "agent1" or so on
    prefer_track_agent=None,
    # Setting the camera position for the Top-down Camera for 3D viewer (pressing key "B" to activate it)
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=200,

    # ===== Vehicle =====
    vehicle_config=dict(
        # Vehicle model. Candidates: "s", "m", "l", "xl", "default". random_agent_model makes this config invalid
        vehicle_model="default",
        # If set to True, the vehicle can go backwards with throttle/brake < -1
        enable_reverse=True,
        # Whether to show the box as navigation points
        show_navi_mark=False,
        # Whether to show a box mark at the destination
        show_dest_mark=False,
        # Whether to draw a line from current vehicle position to the designation point
        show_line_to_dest=False,
        # Whether to draw a line from current vehicle position to the next navigation point
        show_line_to_navi_mark=False,
        # Whether to draw left / right arrow in the interface to denote the navigation direction
        show_navigation_arrow=False,
        # If set to True, the vehicle will be in color green in top-down renderer or MARL setting
        use_special_color=True,
        # Clear wheel friction, so it can not move by setting steering and throttle/brake. Used for ReplayPolicy
        no_wheel_friction=False,

        # ===== image capturing =====
        # Which camera to use for image observation. It should be a sensor registered in sensor config.
        image_source="rgb_camera", # TODO Lidar? 

        # ===== vehicle spawn and navigation =====
        # A BaseNavigation instance. It should match the road network type.
        navigation_module=None, # TODO Figure out what this is for
        # A lane id specifies which lane to spawn this vehicle
        spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 2), # TODO From SafeMetaDriveEnv
        # destination lane id. Required only when navigation module is not None.
        destination=None, # What is a destination lane id
        # the longitudinal and lateral position on the spawn lane
        spawn_longitude=5.0,
        spawn_lateral=0.0,

        # If the following items is assigned, the vehicle will be spawn at the specified position with certain speed
        spawn_position_heading=None,
        spawn_velocity=None,  # m/s
        spawn_velocity_car_frame=False,

        # ==== others ====
        # How many cars the vehicle has overtaken. It is deprecated due to bug.
        overtake_stat=False,
        # If set to True, the default texture for the vehicle will be replaced with a pure color one.
        random_color=False,
        # The shape of vehicle are predefined by its class. But in special scenario (WaymoVehicle) we might want to
        # set to arbitrary shape.
        width=None,
        length=None,
        height=None,
        mass=None,
        scale=None,  # triplet (x, y, z)

        # Set the vehicle size only for pygame top-down renderer. It doesn't affect the physical size!
        top_down_width=None,
        top_down_length=None,

        # ===== vehicle module config =====
        lidar=dict(
            num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0, add_others_navi=False
        ),
        side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
        lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
        show_lidar=False,
        show_side_detector=False,
        show_lane_line_detector=False,
        # Whether to turn on vehicle light, only available when enabling render-pipeline
        light=False,
    ),

    # ===== Sensors =====
    sensors=dict(lidar=(Lidar, ), side_detector=(SideDetector, ), lane_line_detector=(LaneLineDetector, )),

    # ===== Engine Core config =====
    # If true pop a window to render
    use_render=False,
    # (width, height), if set to None, it will be automatically determined
    window_size=(1200, 900),
    # Physics world step is 0.02s and will be repeated for decision_repeat times per env.step()
    physics_world_step_size=2e-2,
    decision_repeat=5,
    # This is an advanced feature for accessing image without moving them to ram!
    image_on_cuda=False, # TODO Make sure we use this
    # Don't set this config. We will determine the render mode automatically, it runs at physics-only mode by default.
    _render_mode=RENDER_MODE_NONE,
    # If set to None: the program will run as fast as possible. Otherwise, the fps will be limited under this value
    force_render_fps=None, # TODO make sure this is None at runtime
    # We will maintain a set of buffers in the engine to store the used objects and can reuse them when possible
    # enhancing the efficiency. If set to True, all objects will be force destroyed when call clear()
    force_destroy=False,
    # Number of buffering objects for each class.
    num_buffering_objects=200,
    # Turn on it to use render pipeline, which provides advanced rendering effects (Beta)
    render_pipeline=False,
    # daytime is only available when using render-pipeline
    daytime="19:00",  # use string like "13:40", We usually set this by editor in toolkit
    # Shadow range, unit: [m]
    shadow_range=50,
    # Whether to use multi-thread rendering
    multi_thread_render=True,
    multi_thread_render_mode="Cull",  # or "Cull/Draw"
    # Model loading optimization. Preload pedestrian for avoiding lagging when creating it for the first time
    preload_models=True,
    # model compression increasing the launch time
    disable_model_compression=True,
    # Whether to disable the collision detection (useful for debugging / replay logged scenarios)
    disable_collision=False,

    # ===== Terrain =====
    # The size of the square map region, which is centered at [0, 0]. The map objects outside it are culled.
    map_region_size=2048,
    # Whether to remove lanes outside the map region. If True, lane localization only applies to map region
    cull_lanes_outside_map=False,
    # Road will have a flat marin whose width is determined by this value, unit: [m]
    drivable_area_extension=7,
    # Height scale for mountains, unit: [m]. 0 height makes the terrain flat
    height_scale=50,
    # If using mesh collision, mountains will have physics body and thus interact with vehicles.
    use_mesh_terrain=True, # TODO: Should we use mesh collision for mountains
    # If set to False, only the center region of the terrain has the physics body
    full_size_mesh=True,
    # Whether to show crosswalk
    show_crosswalk=True, # TODO: Could we disable this for speed in headless running?
    # Whether to show sidewalk
    show_sidewalk=True, # TODO: Could we disable this for speed in headless running?

    # ===== Debug =====
    # Please see Documentation: Debug for more details
    pstats=False,  # turn on to profile the efficiency
    debug=False,  # debug, output more messages
    debug_panda3d=False,  # debug panda3d
    debug_physics_world=False,  # only render physics world without model, a special debug option
    debug_static_world=False,  # debug static world
    log_level=logging.INFO,  # log level. logging.DEBUG/logging.CRITICAL or so on
    show_coordinates=False,  # show coordinates for maps and objects for debug

    # ===== GUI =====
    # Please see Documentation: GUI for more details
    # Whether to show these elements in the 3D scene
    show_fps=True, # TODO: Could we disable this for speed in headless running?
    show_logo=True, # TODO: Could we disable this for speed in headless running?
    show_mouse=True, # TODO: Could we disable this for speed in headless running?
    show_skybox=True, # TODO: Could we disable this for speed in headless running?
    show_terrain=True, # TODO: Could we disable this for speed in headless running?
    show_interface=True, # TODO: What does this do
    # Show marks for policies for debugging multi-policy setting
    show_policy_mark=False, # TODO: What about multi-agent
    # Show an arrow marks for providing navigation information
    show_interface_navi_mark=True, # TODO: Green arrows
    # A list showing sensor output on window. Its elements are chosen from sensors.keys() + "dashboard"
    interface_panel=["dashboard"],

    # ===== Record/Replay Metadata =====
    # Please see Documentation: Record and Replay for more details
    # When replay_episode is True, the episode metadata will be recorded
    record_episode=False,
    # The value should be None or the log data. If it is the later one, the simulator will replay logged scenario
    replay_episode=None,
    # When set to True, the replay system will only reconstruct the first frame from the logged scenario metadata
    only_reset_when_replay=False,
    # If True, when creating and replaying object trajectories, use the same ID as in dataset
    force_reuse_object_name=False,
)

NEW_HAZARDOUS_CONFIG = dict(
    map= 4,
    traffic_mode= "trigger",
    traffic_density= 0.1, # 0.15-0.2 is good for training
    out_of_route_done=True, # We don't want them cutting corners
    on_continuous_line_done=True, # Good practice probably
    agent_policy= IDMPolicy,
)

class HazardousMetaDriveEnv(SafeMetaDriveEnv):
    # TODO: From ComplexEnv
    # def setup_engine(self):
        # super(HazardousMetaDriveEnv, self).setup_engine()
        # self.engine.register_manager("object_manager", ComplexObjectManager())

    # The method below comes from MetaDriveEnv
    # def default_config(cls) -> Config:
    #     config = super(MetaDriveEnv, cls).default_config()
    #     config.update(METADRIVE_DEFAULT_CONFIG)
    #     config.register_type("map", str, int)
    #     config["map_config"].register_type("config", None)
    #     return config
    
    # The methods below come from SafeMetaDriveEnv
    def default_config(self) -> Config:
        config = super(HazardousMetaDriveEnv, self).default_config()
        config.update(
            NEW_HAZARDOUS_CONFIG,
            allow_add_new_key=True # TODO: What does this do?
        )
        return config

    def __init__(self, config=None):
        super(HazardousMetaDriveEnv, self).__init__(config)
        self.episode_cost = 0

    def reset(self, *args, **kwargs):
        self.episode_cost = 0
        return super(HazardousMetaDriveEnv, self).reset(*args, **kwargs)

    def cost_function(self, vehicle_id: str):
        cost, step_info = super(HazardousMetaDriveEnv, self).cost_function(vehicle_id)
        self.episode_cost += cost
        step_info["total_cost"] = self.episode_cost
        return cost, step_info