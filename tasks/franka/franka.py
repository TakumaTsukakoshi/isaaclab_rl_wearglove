# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Shared Franka parent environment
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    # TiledCamera,
    # TiledCameraCfg,
)
from isaaclab.sim import SimulationContext
from isaaclab.sim.simulation_cfg import RenderCfg

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
)
import isaaclab.utils.math as math_utils
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from assets.franka import FRANKA_PANDA_CFG  # isort: skip

from pxr import Sdf
from isaaclab.sim import SimulationContext

import sys
sys.path.append("tasks/franka")
from insert_rew import InsertReward

def ensure_xform_prim(prim_path: str) -> bool:
    sim = SimulationContext.instance()
    if sim is None or getattr(sim, "stage", None) is None:
        return False
    stage = sim.stage
    if not stage.GetPrimAtPath(prim_path):
        stage.DefinePrim(Sdf.Path(prim_path), "Xform")
    return True



@configclass
class FrankaEnvCfg(DirectRLEnvCfg):
    # physics sim
    physics_dt = 1 / 120  # 0.002 #1 / 500 # 120 # 500 Hz

    # number of physics step per control step
    decimation = 2  # 10 # # 50 Hz

    # the number of physics simulation steps per rendering steps (default=1)
    render_interval = 2
    episode_length_s = 5.0  # 5 * 120 / 2 = 300 timesteps

    num_observations = 0
    num_actions = 9
    num_states = 0

    # isaac 4.5 stuff
    action_space = num_actions
    observation_space = num_observations
    state_space = num_states

    # configure this to get the right dimensions for fusion network
    obs_stack = 1

    # reset config
    reset_object_position_noise = 0.05
    # lift stuff
    minimal_height = 0.04
    minimal_dense = 0.02
    reaching_object_scale = 1
    contact_reward_scale = 10
    lift_object_scale = 15.0
    object_goal_tracking_scale = 16.0
    joint_vel_penalty_scale = 0  # -0.01
    object_out_of_bounds = 1.5
    rotation_object_scale = 5

    # reach stuff
    min_reach_dist = 0.05

    # simulation
    # sim: SimulationCfg = SimulationCfg(
    #     dt=physics_dt,
    #     render_interval=decimation,
    #     physics_material=RigidBodyMaterialCfg(
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     physx=PhysxCfg(
    #         bounce_threshold_velocity=0.2,
    #         # gpu_max_rigid_contact_count=2**25, # default 2**23
    #         # gpu_max_rigid_patch_count=2**25, #23, default 5 * 2 ** 15.
    #         gpu_temp_buffer_capacity=2**20, # default 2**20
    #         gpu_max_soft_body_contacts= 2**20, # default 2**20 
    #         gpu_collision_stack_size=2**26, # default 2**26
    #     ),
    # )
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            # gpu_max_rigid_contact_count=2**25, # default 2**23
            # gpu_max_rigid_patch_count=2**25, #23, default 5 * 2 ** 15.
            # gpu_temp_buffer_capacity=2**20, # default 2**20
            # gpu_max_soft_body_contacts= 2**23, # default 2**20 
            # gpu_collision_stack_size=2**26, # default 2**26
            gpu_temp_buffer_capacity=2**20, # default 2**20
            gpu_max_soft_body_contacts= 2**18, # default 2**20 
            gpu_collision_stack_size=2**20, # default 2**26
        ),
        render=RenderCfg(
            antialiasing_mode="DLAA",
        )
    )

    # temp
    replicate_physics = False
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8, env_spacing=5, replicate_physics=replicate_physics
    )

    # default_object_pos = [0.5, 0, 0.20]  # 0.055
    eye = (3, 3, 3)
    lookat = (0, 0, 0)

    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    # robot
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # panda finger bodies are called : 'panda_leftfinger', 'panda_rightfinger``
    # We set the update period to 0 to update the sensor at the same frequency as the simulation
    # contact sensors are called 'left_contact_sensor' and 'right_contact_sensor'
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/ContactCfg"
    left_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/left_contact_sensor",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        # filter_prim_paths_expr=["/World/envs/env_.*/Cube"],  # ["/World/envs/env_.*/Object"]
    )
    right_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_contact_sensor",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        # filter_prim_paths_expr=["/World/envs/env_.*/Cube"],  # ["/World/envs/env_.*/Object"]
    )

    wholebody_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        # filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )

    # Normalisation numbers
    tactile_min_val = 0
    tactile_max_val = 20.0
    vel_max_magnitude = 3

    actuated_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"
    ee_config: FrameTransformerCfg = FrameTransformerCfg(
        # source frame
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                    # rot=[-0.7071, 0.0, -0.7071, 0.0] 
                    rot=[0.0, -0.7071, 0.0, -0.7071]
                    # rot=[0.7071, 0.0, 0.7071, 0.0] 
                ),
            )
        ],
    )

    right_tool_config: FrameTransformerCfg = FrameTransformerCfg(
        # source frame
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.046],
                    rot=[0.0, -0.7071, 0.0, -0.7071]
                ),
            ),
        ],
    )

    left_tool_config: FrameTransformerCfg = FrameTransformerCfg(
        # source frame
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.046],
                    rot=[0.0, -0.7071, 0.0, -0.7071]
                ),
            )
        ],
    )

    # contact sensors - ONLY RECORD FORCES FROM THE OBJECT FOR NOW
    left_sensor_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_contact_sensor",
                name="left_sensor",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    )
    right_sensor_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_contact_sensor",
                name="right_sensor",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    )

    anchor_east_marker_cfg = FRAME_MARKER_CFG.copy()
    anchor_east_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    anchor_east_marker_cfg.prim_path = "/World/Visuals/AnchorEastMarker"

    anchor_east_tf_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",  # regex OK for sensors
        debug_vis=False,
        visualizer_cfg=anchor_east_marker_cfg,                  # concrete prim_path
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Visuals/AnchorEast/Geom",  # target can be regex
                name="anchor_east",
            )
        ],
    )

    img_dim = 84
    eye = [1.2, -0.3, 0.5]
    target = [0, 0, 0]
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.0, 0.0), rot=(1, 0, 0, 0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 3.2)
        ),
        width=img_dim,
        height=img_dim,
        debug_vis=False,
    )

    # defaults to be overwritten
    write_image_to_file = False
    obs_list = ["prop", "gt"] # add pixels later
    aux_list = []
    normalise_prop = True
    normalise_pixels = True
    num_cameras = 1
    object_type = "deformable"  # "rigid" or "deformable"


class FrankaEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaEnvCfg

    def __init__(self, cfg: FrankaEnvCfg, render_mode: str | None = None, **kwargs):

        self.obs_stack = cfg.obs_stack
        super().__init__(cfg, render_mode, **kwargs)

        self.dtype = torch.float32
        self.binary_tactile = cfg.binary_tactile

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # create empty tensors
        self.actions = torch.zeros((self.num_envs, 9), device=self.device)

        self.joint_pos = torch.zeros((self.num_envs, 9), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, 9), device=self.device)
        self.normalised_joint_pos = torch.zeros((self.num_envs, 9), device=self.device)
        self.normalised_joint_vel = torch.zeros((self.num_envs, 9), device=self.device)
        self.aperture = torch.zeros((self.num_envs,), device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.normalised_forces = torch.zeros((self.num_envs, 2), device=self.device)
        self.unnormalised_forces = torch.zeros((self.num_envs, 2), device=self.device)
        self.in_contact = torch.zeros((self.num_envs, 1), device=self.device)
        self.tactile = torch.zeros((self.num_envs, 2), device=self.device)

        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_rot = torch.zeros((self.num_envs, 4), device=self.device)

        # tool_rightfinger
        self.tool_rfinger_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.tool_rfinger_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # tool_leftfinger
        self.tool_lfinger_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.tool_lfinger_rot = torch.zeros((self.num_envs, 4), device=self.device)

        self.east_edge_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.east_edge_rot = torch.zeros((self.num_envs, 4), device=self.device)

        self.object_ee_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_ee_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_ee_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        self.object_ee_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)

        # save reward weights so they can be adjusted online
        self.reaching_object_scale = cfg.reaching_object_scale
        self.contact_reward_scale = cfg.contact_reward_scale
        self.lift_object_scale = cfg.lift_object_scale
        self.joint_vel_penalty_scale = cfg.joint_vel_penalty_scale
        self.object_goal_tracking_scale = cfg.object_goal_tracking_scale
        self.rotation_object_scale = cfg.rotation_object_scale

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # camera stuff
        self.count = 0

        # target node
        self.anchor_idx = None
        self.prev_anchor_idx = None           
        self.max_nodes = None
        self.nodal_state = None  
            
        self.extras["log"] = {
            "reach_reward": None,
            "lift_reward": None,
            "dist_reward": None,
            "contact_reward": None,
            "joint_vel_penalty": None,
            "object_ee_distance": None,
            "object_goal_tracking": None,
            "object_goal_tracking_finegrained": None,
            "tactile": None,
            "unnormalised_forces_left_x": None,
            "unnormalised_forces_right_x": None,
            "normalised_forces_left_x": None,
            "normalised_forces_right_x": None,
            "r_angular_ee_object": None,
            "r_insert": None
        }

        self.extras["counters"] = {
            "timesteps_to_find_object_easy": None,
            "timesteps_to_find_object_med": None,
            "timesteps_to_find_object_hard": None,
            "object_found_easy": None,
            "object_found_med": None,
            "object_found_hard": None,
        }

        self._vis_enabled = False

        # insert controller
        self.insert_reward = InsertReward(self.num_envs, device=self.device, inward_assume="+x")
        self.insert_success = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.insert_dwell = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )

    def _configure_gym_env_spaces(self):
        pass
    
    def set_spaces(self, single_obs, obs, single_action, action):
        self.single_observation_space = single_obs
        self.observation_space = obs
        self.single_action_space = single_action
        self.action_space = action

    def _add_object_to_scene(self):
        if self.cfg.object_type == "rigid":
            print("SETTING UP RIGID OBJECT", self.cfg.object_cfg)
            self.object = RigidObject(self.cfg.object_cfg)
            self.scene.rigid_objects["object"] = self.object

        elif self.cfg.object_type == "deformable":
            print("SETTING UP DEFORMABLE OBJECT", self.cfg.object_cfg)
            self.object = DeformableObject(self.cfg.object_cfg)
            self.scene.deformable_objects["object"] = self.object

    def _setup_scene(self):

        self.robot = Articulation(self.cfg.robot_cfg)

        self._add_object_to_scene()

        # FrameTransformer provides interface for reporting the transform of
        # one or more frames (target frames) wrt to another frame (source frame)
        self.ee_frame = FrameTransformer(self.cfg.ee_config)
        self.ee_frame.set_debug_vis(False)
        self.left_sensor_frame = FrameTransformer(self.cfg.left_sensor_cfg)
        self.right_sensor_frame = FrameTransformer(self.cfg.right_sensor_cfg)

        # add right_tool effector frame
        self.right_tool_frame = FrameTransformer(self.cfg.right_tool_config)
        self.right_tool_frame.set_debug_vis(False)
        
        rb_path_env0 = "/World/envs/env_0/Visuals/AnchorEast/Geom"
        stage = SimulationContext.instance().stage
        ensure_xform_prim("/World/Visuals") 
        if not stage.GetPrimAtPath(rb_path_env0):
            anchor_rb_cfg = sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                physics_material=RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
            anchor_rb_cfg.func(rb_path_env0, anchor_rb_cfg)
        if not stage.GetPrimAtPath("/World/ground"):
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(10000, 10000)))

        self.scene.clone_environments(copy_from_source=False)
        self._anchor_rb_path = "/World/envs/env_.*/Visuals/AnchorEast/Geom"
        self.anchor_rb = RigidObject(RigidObjectCfg(prim_path=self._anchor_rb_path))
        self.scene.rigid_objects["anchor_east"] = self.anchor_rb

        self.anchor_east_tf = FrameTransformer(self.cfg.anchor_east_tf_cfg)

        # register to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["ee_frame"] = self.ee_frame
        self.scene.sensors["left_sensor_frame"] = self.left_sensor_frame
        self.scene.sensors["right_sensor_frame"] = self.right_sensor_frame
        self.scene.sensors["anchor_east_tf"] = self.anchor_east_tf 
        self.scene.sensors["right_tool_frame"] = self.right_tool_frame

        yellow = (1.0, 0.96, 0.0)
        orange = (1.0, 0.5, 0.0)
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        light_cfg_1 = sim_utils.SphereLightCfg(intensity=10000.0, color=yellow)
        light_cfg_1.func("/World/ds", light_cfg_1, translation=(1, 0, 1))
        light_cfg_2 = sim_utils.SphereLightCfg(intensity=10000.0, color=orange)
        light_cfg_2.func("/World/disk", light_cfg_2, translation=(-1, 0, 1))

        if "pixels" in self.cfg.obs_list or "pixels" in self.cfg.aux_list:
            print("Using Isaac Lab camera stack")
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera
            

        if "tactile" in self.cfg.obs_list:
            self.left_contact_sensor = ContactSensor(self.cfg.left_contact_cfg)
            self.scene.sensors["left_contact_sensor"] = self.left_contact_sensor

            self.right_contact_sensor = ContactSensor(self.cfg.right_contact_cfg)
            self.scene.sensors["right_contact_sensor"] = self.right_contact_sensor

            self.wholebody_contact_sensor = ContactSensor(self.cfg.wholebody_contact_cfg)
            self.scene.sensors["wholebody_contact_sensor"] = self.wholebody_contact_sensor

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Store actions from policy in a class variable
        """
        self.last_action = self.robot_dof_targets[:, self.actuated_dof_indices]
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """
        The _apply_action(self) API is called decimation number of times for each RL step, prior to taking each physics step.
        This provides more flexibility for environments where actions should be applied for each physics step.
        """
        scaled_actions = self.scale_action(self.actions)

        self.robot.set_joint_position_target(scaled_actions, joint_ids=self.actuated_dof_indices)

    def scale_action(self, action):
        self.robot_dof_targets[:, self.actuated_dof_indices] = scale(
            action,
            self.robot_dof_lower_limits[self.actuated_dof_indices],
            self.robot_dof_upper_limits[self.actuated_dof_indices],
        )

        self.robot_dof_targets[:, self.actuated_dof_indices] = saturate(
            self.robot_dof_targets[:, self.actuated_dof_indices],
            self.robot_dof_lower_limits[self.actuated_dof_indices],
            self.robot_dof_upper_limits[self.actuated_dof_indices],
        )
        return self.robot_dof_targets[:, self.actuated_dof_indices]

    def get_observations(self):
        # public method
        return self._get_observations()

    def _get_observations(self) -> dict:

        obs_dict = {}
        for k in self.cfg.obs_list:
            if k == "prop":
                obs_dict[k] = self._get_proprioception()
            elif k == "pixels":
                obs_dict[k] = self._get_images()
            elif k == "gt":
                obs_dict[k] = self._get_gt()
            elif k == "tactile":
                obs_dict[k] = self._get_tactile()
            else:
                print("Unknown observations type!")

        obs_dict = {"policy": obs_dict}

        return obs_dict

    def _get_proprioception(self):
        prop = torch.cat(
            (
                self.normalised_joint_pos,
                self.normalised_joint_vel,
                self.aperture.unsqueeze(1),
                self.ee_pos,
                self.ee_rot,
                self.actions,
            ),
            dim=-1,
        )

        return prop

    def _get_gt(self):

        gt = torch.cat(
            (
                # xyz diffs (3,)
                self.object_ee_distance,
                # rotation quaternion (4,)
                self.object_ee_rotation,
                # rotation difference (1,)
                self.object_ee_angular_distance.unsqueeze(1),
                # euclidean distances (1,) [transform from (num_envs,) to (num_envs,1)]
                self.object_ee_euclidean_distance.unsqueeze(1),
            ),
            dim=-1,
        )
        return gt

    def _read_force_matrix(self, filter=False):
        # separate into left and right for frame transform force_matrix_w net_forces_w
        if filter:
            forcesL_world = self.left_contact_sensor.data.force_matrix_w[:].clone().reshape(self.num_envs, 3)
            forcesR_world = self.right_contact_sensor.data.force_matrix_w[:].clone().reshape(self.num_envs, 3)
        else:
            forcesL_world = self.left_contact_sensor.data.net_forces_w[:].clone().reshape(self.num_envs, 3)
            forcesR_world = self.right_contact_sensor.data.net_forces_w[:].clone().reshape(self.num_envs, 3)

        return forcesL_world, forcesR_world

    def _normalise_forces(self, forcesL, forcesR):
        # only return the normal component
        return_forces = torch.abs(torch.cat((forcesL, forcesR), dim=1))

        # Clip the tensor values and normalise 0 to 1
        self.unnormalised_forces = torch.clamp(
            return_forces, min=self.cfg.tactile_min_val, max=self.cfg.tactile_max_val
        )
        self.normalised_forces = (self.unnormalised_forces - self.cfg.tactile_min_val) / (
            self.cfg.tactile_max_val - self.cfg.tactile_min_val
        )

        return self.normalised_forces

    def _get_tactile(self):
        # contact sensor data is [num_envs, 2, 3]
        forcesL_world, forcesR_world = self._read_force_matrix()

        # absolute value the whole thing, and sum it
        forcesL_net = torch.linalg.vector_norm(forcesL_world, dim=1, keepdim=True)
        forcesR_net = torch.linalg.vector_norm(forcesR_world, dim=1, keepdim=True)

        if self.binary_tactile:
            if self.dtype == torch.float16:
                forcesL_norm = (forcesL_net > 0).half()
                forcesR_norm = (forcesR_net > 0).half()
            else:
                forcesL_norm = (forcesL_net > 0).float()
                forcesR_norm = (forcesR_net > 0).float()

            tactile = torch.cat(
                (
                    forcesL_norm,
                    forcesR_norm,
                ),
                dim=-1,
            )
            self.tactile = tactile
            return tactile
        else:

            self.normalised_forces = self._normalise_forces(forcesL_net, forcesR_net)
            self.tactile = self.normalised_forces
            return self.normalised_forces

    def _get_images(self):

        camera_data = self._tiled_camera.data.output["rgb"].clone()

        # normalize the camera data for better training results
        # convert to float 32 to subtract mean, then back to uint8 for memory storage
        if self.cfg.normalise_pixels:
            camera_data = camera_data / 255.0
            mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            camera_data -= mean_tensor
            camera_data *= 255
            camera_data = camera_data.to(torch.uint8)

        return camera_data

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
            
        super()._reset_idx(env_ids)    
        
        self.anchor_idx = self._choose_mouth_nodes_4dirs()
        self.prev_anchor_idx = self.anchor_idx
        print("[INFO]: RESET_IDX")
        # print(f"[INFO] anchor_idx: {self.anchor_idx}")
        # if self.anchor_idx != self.prev_anchor_idx:
        #     print(f"[INFO] anchor_idx: {self.anchor_idx}")
        #     return

           
        if self.cfg.object_type == "deformable" and self.anchor_idx is None:
            self.anchor_idx = self._choose_mouth_nodes_4dirs()

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        if self.cfg.object_type == "rigid":
             self._reset_object_pose(env_ids)
        elif self.cfg.object_type == "deformable":
            self._reset_deformable_pose(env_ids)

        # reset robot
        self._reset_robot(env_ids)
        # refresh intermediate values for _get_observations()
        self._compute_intermediate_values(env_ids=env_ids)
        

    def _reset_deformable_pose(self, env_ids):
        print("[INFO]: RESET_DEFORMABLE_POSE")

        nodal_state = self.object.data.default_nodal_state_w.clone()[env_ids]  
        N = int(nodal_state.shape[1])

        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        pos_w = self.cfg.reset_object_position_noise * pos_noise * 0
        
        # ±10° = ±(10 * π/180)
        # angle_limit = 10.0 * math.pi / 180.0
        # rand_z = torch.empty(len(env_ids), device=self.device).uniform_(-angle_limit, angle_limit)

        # rand_z = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi/3, math.pi/3)
        # quat_w = math_utils.quat_from_euler_xyz(
        #     torch.zeros(len(env_ids), device=self.device),
        #     torch.zeros(len(env_ids), device=self.device),
        #     rand_z
        # )
        # quat_w = math_utils.quat_from_euler_xyz(
        #     torch.rand(len(env_ids), device=self.device)* math.pi * 2,
        #     torch.rand(len(env_ids), device=self.device)* math.pi * 2,
        #     torch.rand(len(env_ids), device=self.device) * math.pi / 1.5
        # )
        quat_w = math_utils.quat_from_euler_xyz(
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device)*math.pi,
        )
        nodal_state[..., :3] = self.object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
        self.object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)

        nodal_kinematic_target = self.object.data.nodal_kinematic_target.clone()[env_ids]
        nodal_kinematic_target[..., :3] = nodal_state[..., :3]
        nodal_kinematic_target[..., 3]  = 1.0

        nodal_kinematic_target[:, self.anchor_idx["east"], :3] = 0.0
        nodal_kinematic_target[:, self.anchor_idx["east"], :3] = nodal_state[:, self.anchor_idx["east"], :3]

        self.object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target, env_ids=env_ids)
        if (self.nodal_state is None) or (self.nodal_state.shape[1] != N):
            self.max_nodes = N
            self.nodal_state = torch.zeros(
                (self.num_envs, self.max_nodes, 6),
                device=self.device, dtype=torch.float32
            )
        self.nodal_state[env_ids] = nodal_state
        self.object.reset()

    def _reset_object_pose(self, env_ids):
        object_default_state = self.object.data.default_root_state.clone()[env_ids]

        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        # global object positions (for writing to sim)
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3]
            + self.cfg.reset_object_position_noise * pos_noise
            + self.scene.env_origins[env_ids]
        )
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)


    def _reset_robot(self, env_ids):
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _reset_target_pose(self, env_ids):
        pass

    def _choose_single_mouth_node(self, end_slice_ratio=0.06, axis_hint=None):
        P = self.object.data.default_nodal_state_w[0, :, :3].clone()  # (V,3)
        mean = P.mean(0)
        X = P - mean 
        import torch

        if axis_hint is None:
            U, S, Vt = torch.pca_lowrank(X, q=3, center=False)  
            a = Vt[:, 0]
            u = Vt[:, 1]
            v = Vt[:, 2]
        else:
            a = torch.tensor(axis_hint, dtype=X.dtype, device=X.device)
            a = a / a.norm()
            g = torch.tensor([0.0, 0.0, 1.0], dtype=X.dtype, device=X.device)
            if torch.abs(g @ a) > 0.9:
                g = torch.tensor([0.0, 1.0, 0.0], dtype=X.dtype, device=X.device)
            u = torch.linalg.cross(g, a)
            u = u / u.norm()
            v = torch.linalg.cross(a, u)

        t = X @ a
        t_min, t_max = t.min(), t.max()
        L = float(t_max - t_min)
        sl = end_slice_ratio * L
        mask_plus = t >= (t_max - sl)
        mask_minus = t <= (t_min + sl)

        X_perp = X - t.unsqueeze(1) * a
        r = torch.linalg.norm(X_perp, dim=1)

        end_mask = mask_plus if r[mask_plus].median() >= r[mask_minus].median() else mask_minus
        idx_end = torch.nonzero(end_mask, as_tuple=False).squeeze(1)

        d = (-u - v)
        d = d / d.norm()
        Xp = X_perp[idx_end] 
        score_dir = Xp @ d
        score_rad = r[idx_end]
        score = score_dir + 0.1 * score_rad 
        anchor_idx = int(idx_end[torch.argmax(score)])
        return anchor_idx

    def _choose_mouth_nodes_4dirs(
        self, end_slice_ratio=0.05, axis_hint=None
    ):
        """
        グローブ入口（mouth）近傍から、上下左右の4ノードのインデックスを返す。

        Args:
            end_slice_ratio (float): 全長に対して何割のスライスを入口側として扱うか。
            axis_hint (torch.Tensor or None): メッシュの主軸方向のヒント (例: +Z)。無指定ならPCAで求める。
        Returns:
            dict: {"up": idx_up, "down": idx_down, "left": idx_left, "right": idx_right}
        """
        import torch
        P = self.object.data.default_nodal_state_w[0, :, :3].clone()  # (V,3)
        mean = P.mean(0)
        X = P - mean

        # --- 主軸推定 ---
        if axis_hint is None:
            U, S, Vt = torch.pca_lowrank(X, q=3, center=False)
            a = Vt[:, 0]   # 主軸（長手方向、奥行き方向）
            u = Vt[:, 1]   # 接線1（横方向）
            v = Vt[:, 2]   # 接線2（縦方向）
            if np.dot(a.cpu().numpy(), np.array([1, 0, 0])) > 0:  # 例えば +Z が口方向なら
                a = -a
        else:
            a = torch.tensor(axis_hint, dtype=X.dtype, device=X.device)
            a = a / a.norm()

            g = torch.tensor([0.0, 0.0, 1.0], dtype=X.dtype, device=X.device)
            if torch.abs(g @ a) > 0.9:
                g = torch.tensor([0.0, 1.0, 0.0], dtype=X.dtype, device=X.device)
            u = torch.linalg.cross(g, a); u = u / u.norm()
            v = torch.linalg.cross(a, u)

        # --- 入口スライスの抽出 ---
        t = X @ a
        t_min, t_max = t.min(), t.max()
        L = float(t_max - t_min)
        sl = end_slice_ratio * L
        mask_plus = t >= (t_max - sl)
        mask_minus = t <= (t_min + sl)


        # どちらが入口側か判定（半径が大きい方を入口とみなす）
        X_perp = X - t.unsqueeze(1) * a
        r = torch.linalg.norm(X_perp, dim=1)
        mask_entry = mask_plus if r[mask_plus].median() >= r[mask_minus].median() else mask_minus
        idx_entry = torch.nonzero(mask_entry, as_tuple=False).squeeze(1)

        # --- 入口パッチ中での極値方向を算出 ---
        X_entry = X[idx_entry]
        alpha = X_entry @ u
        beta  = X_entry @ v

        idx_right = idx_entry[torch.argmax(alpha)]
        idx_left  = idx_entry[torch.argmin(alpha)]
        idx_up    = idx_entry[torch.argmax(beta)]
        idx_down  = idx_entry[torch.argmin(beta)]

        return {
            "north": int(idx_up),
            "south": int(idx_down),
            "west": int(idx_left),
            "east": int(idx_right),
        }

    def _nearest_patch_indices(self, node_idx: int, k: int = 16):
        # 基準（リセット/デフォルト）形状の全ノード座標
        Q0 = self.object.data.default_nodal_state_w[0, :, :3]  # (V,3), CPU/Tensor
        a = Q0[node_idx:node_idx+1, :]                          # (1,3)
        # ユークリッド距離で最近傍 K 点（node_idx自身を含む）
        d2 = torch.sum((Q0 - a)**2, dim=1)                      # (V,)
        idx = torch.topk(-d2, k=k, largest=True).indices        # 近い=距離小→-d2を大きい順
        return idx                                              # (k,)

    def _kabsch_rotation(self, P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor):
        """
        P: 現在 (B,k,3), Q: 基準 (B,k,3), w: 重み (B,k)
        返: R (B,3,3)  反射補正済み
        """
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        muP = torch.sum(P * w.unsqueeze(-1), dim=1, keepdim=True)  # (B,1,3)
        muQ = torch.sum(Q * w.unsqueeze(-1), dim=1, keepdim=True)
        X = P - muP
        Y = Q - muQ
        H = torch.matmul((Y * w.unsqueeze(-1)).transpose(1, 2), X)   # (B,3,3)
        U, S, Vt = torch.linalg.svd(H)
        V = Vt.transpose(-2, -1); Ut = U.transpose(-2, -1)
        det = torch.sign(torch.linalg.det(V @ Ut)).unsqueeze(-1).unsqueeze(-1)
        M = torch.eye(3, device=P.device).unsqueeze(0).repeat(P.shape[0],1,1)
        M[:, 2, 2] = det.squeeze(-1).squeeze(-1)
        R = V @ M @ Ut
        return R

    def _rotmat_to_quat_wxyz(self, R: torch.Tensor):
        """
        R: (B,3,3) → quat (B,4) in (w,x,y,z)
        安定版（対角最大成分で場合分け）
        """
        B = R.shape[0]
        q = torch.empty(B, 4, device=R.device, dtype=R.dtype)
        trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
        # ブランチごとに計算
        mask = trace > 0
        if mask.any():
            t = torch.sqrt(trace[mask] + 1.0) * 2.0
            q[mask,0] = 0.25 * t
            q[mask,1] = (R[mask,2,1] - R[mask,1,2]) / t
            q[mask,2] = (R[mask,0,2] - R[mask,2,0]) / t
            q[mask,3] = (R[mask,1,0] - R[mask,0,1]) / t
        if (~mask).any():
            Rsub = R[~mask]
            # 対角の最大要素で枝分かれ
            idx = torch.argmax(torch.stack([Rsub[:,0,0], Rsub[:,1,1], Rsub[:,2,2]], dim=1), dim=1)
            q_sub = torch.empty(Rsub.shape[0], 4, device=R.device, dtype=R.dtype)
            for a in range(3):
                sel = idx == a
                if not sel.any(): continue
                Rs = Rsub[sel]
                i = a
                j = (a+1)%3; k = (a+2)%3
                t = torch.sqrt(1.0 + Rs[:,i,i] - Rs[:,j,j] - Rs[:,k,k]) * 2.0
                q_sub[sel,0] = (Rs[:,k,j] - Rs[:,j,k]) / t
                q_sub[sel,1+i] = 0.25 * t
                q_sub[sel,1+j] = (Rs[:,j,i] + Rs[:,i,j]) / t
                q_sub[sel,1+k] = (Rs[:,k,i] + Rs[:,i,k]) / t
            q[~mask] = q_sub
        # 正規化
        q = q / (torch.linalg.norm(q, dim=1, keepdim=True) + 1e-8)
        return q

    def estimate_node_quat(self, node_idx: int, k: int = 16, env_ids=None, use_tactile=False):
        """
        指定ノードの局所パッチから Kabsch で回転 R* を推定し、そのノードの quat(w,x,y,z) を返す。
        返り値: (B,4)  B=len(env_ids) 省略時は全env
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.long().cpu()

        # 近傍パッチ（基準）インデックス
        idx = self._nearest_patch_indices(node_idx, k=k)         # (k,)
        idx_dev = idx.to(self.device)

        # 基準（テンプレ）座標（全envで同じ）
        Q = self.object.data.default_nodal_state_w[0, idx, :3].to(self.device)  # (k,3)
        Q = Q.unsqueeze(0).expand(len(env_ids), -1, -1)                          # (B,k,3)

        # 現在座標
        P = self.nodal_state[env_ids][:, idx_dev, :3]             # (B,k,3)

        with torch.no_grad():
            # 基準形状で node_idx からの距離
            a0 = self.object.data.default_nodal_state_w[0, node_idx, :3].to(self.device)  # (3,)
            dist = torch.linalg.norm(Q - a0.view(1,1,3), dim=-1)                          # (B,k)
            w = 1.0 / (dist + 1e-3)                                                       # 逆距離重み
            if use_tactile and "tactile" in self.cfg.obs_list:
                tactile_gain = 1.0  # 必要なら各点に係数をかける
                w = w * tactile_gain
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

        R = self._kabsch_rotation(P, Q, w)              # (B,3,3)
        check_rotmat(R=R)
        rms_reprojection_error(P=P, Q=Q, w=w, R=R)
        q = self._rotmat_to_quat_wxyz(R)                # (B,4)
        return q


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.long().cpu()  
        assert env_ids.min().item() >= 0
        assert env_ids.max().item() < self.num_envs, f"env_ids={env_ids.max().item()} >= num_envs={self.num_envs}"

        # get robot data
        self.joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        self.ee_pos[env_ids] = self.ee_frame.data.target_pos_source[..., 0, :][env_ids]
        self.ee_rot[env_ids] = self.ee_frame.data.target_quat_source[..., 0, :][env_ids]

        # get right tool frame pos/rot
        self.tool_rfinger_pos[env_ids] = self.right_tool_frame.data.target_pos_source[..., 0, :][env_ids]
        self.tool_rfinger_rot[env_ids] = self.right_tool_frame.data.target_quat_source[..., 0, :][env_ids]

        # aperture between 0-0.08, lets scale to 0-1
        max_aperture = 0.08
        self.aperture = (self.joint_pos[:, 7] + self.joint_pos[:, 8]) / max_aperture

        # normalise joint pos
        self.normalised_joint_pos[env_ids] = unscale(
            self.joint_pos[env_ids], self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

        # joint vel roughly between -2.5, 2.5, so dividing by 3.
        self.normalised_joint_vel[env_ids] = self.joint_vel[env_ids] / self.cfg.vel_max_magnitude
        self.nodal_state[env_ids] = self.object.data.nodal_state_w[env_ids]
      
        self.object_pos[env_ids] = self.nodal_state[:, self.anchor_idx["east"], :3] - self.scene.env_origins[env_ids]
        
        # deformable doesn't have quat
        if self.cfg.object_type == "rigid":
            self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        if self.cfg.object_type == "deformable":
            self.object_rot[env_ids] = self.estimate_node_quat(node_idx=self.anchor_idx["east"],env_ids=env_ids)
        # import ipdb; ipdb.set_trace()

        # relative distances
        self.object_ee_distance[env_ids] = self.object_pos[env_ids] - self.ee_pos[env_ids] 
        self.object_ee_euclidean_distance[env_ids] = torch.norm(self.object_ee_distance[env_ids], dim=1)
        
        self.object_ee_rotation[env_ids] = quat_mul(self.object_rot[env_ids], quat_conjugate(self.ee_rot[env_ids]))
        self.object_ee_angular_distance[env_ids] = rotation_distance(self.object_rot[env_ids], self.ee_rot[env_ids])
        # print("object_ee_angular_distance:", self.object_ee_angular_distance)

        # === ここから追加: east ノードの Pose を各 env のダミー Prim へ反映 ===
        env_all = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        root = self.anchor_rb.data.default_root_state.clone().to(self.device)  
        east_idx = self.anchor_idx["east"]

        pos_w_all = self.nodal_state[:, east_idx, :3].to(self.device)          
        quat_wxyz  = self.estimate_node_quat(node_idx=east_idx).to(self.device) 

        root[:, 0:3] = pos_w_all
        root[:, 3:7] = quat_wxyz
        self.east_edge_pos[env_ids] = self.anchor_east_tf.data.target_pos_source[..., 0, :][env_ids]
        self.east_edge_rot[env_ids] = self.anchor_east_tf.data.target_quat_source[..., 0, :][env_ids]

      
        self.tool_rfinger_pos[env_ids] = self.anchor_east_tf.data.target_pos_source[..., 0, :][env_ids]
        self.tool_rfinger_rot[env_ids] = self.anchor_east_tf.data.target_quat_source[..., 0, :][env_ids]


        B = len(env_ids)
        dt_b = torch.full((B,), float(self.cfg.physics_dt), device=self.device)
        import ipdb; ipdb.set_trace()
    
        out = self.insert_reward.step(
            pos_edge_s=self.east_edge_pos[env_ids],
            quat_edge_s=self.east_edge_rot[env_ids],
            pos_grip_s=self.tool_rfinger_pos[env_ids],
            quat_grip_s=self.tool_rfinger_rot[env_ids],
            dt=dt_b,
            idx=env_ids.to(self.device),
        )

        # (c) 成功フラグと dwell をクラスバッファへ反映
        self.insert_success[env_ids] = out["success"]       # (B,) bool
        self.insert_dwell[env_ids]   = out["dwell"]         # (B,) float
        # self.r_sparse = insert_success_reward(out["success"])

        self.anchor_rb.write_root_state_to_sim(root, env_ids=env_all)

        if not self._vis_enabled:
            if (self.ee_frame.data.target_pos_source is not None
                and self.anchor_east_tf.data is not None):
                print("[INFO] Enabling debug visualization for ee_frame and anchor_east_tf")
                self.anchor_east_tf.set_debug_vis(True)
                self.ee_frame.set_debug_vis(False) 
                self.right_tool_frame.set_debug_vis(True) 
                self._vis_enabled = True

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # no termination at the moment
        out_of_reach = torch.norm(self.object_pos, dim=1) >= self.cfg.object_out_of_bounds
        termination = out_of_reach

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out

def check_rotmat(R, atol=1e-4):
    I = torch.eye(3, device=R.device, dtype=R.dtype)
    ortho = torch.max(torch.abs(R.transpose(1,2) @ R - I).reshape(R.shape[0], -1), dim=1).values
    det = torch.linalg.det(R)
    ok = (ortho < atol) & (torch.abs(det - 1.0) < 1e-3)
    return ok, ortho, det

def rms_reprojection_error(P, Q, w, R):
    # P,Q: (B,k,3), w: (B,k), R: (B,3,3)
    w = w / (w.sum(dim=1, keepdim=True)+1e-8)
    muP = torch.sum(P * w.unsqueeze(-1), dim=1, keepdim=True)
    muQ = torch.sum(Q * w.unsqueeze(-1), dim=1, keepdim=True)
    t = muP - (R @ muQ.transpose(1,2)).transpose(1,2)  # (B,1,3)
    P_hat = (R @ Q.transpose(1,2)).transpose(1,2) + t  # (B,k,3)
    err = torch.linalg.norm(P_hat - P, dim=-1)         # (B,k)
    rms = torch.sqrt(torch.sum(w * err**2, dim=1))     # (B,)
    return rms, t

def angle_between_R(R1, R2):
    # acos( (trace(R1^T R2) - 1)/2 )
    Rt = R1.transpose(1,2) @ R2
    tr = Rt[:,0,0] + Rt[:,1,1] + Rt[:,2,2]
    cosang = torch.clamp((tr - 1.0)*0.5, -1.0, 1.0)
    return torch.arccos(cosang)  # (B,)

# scales an input between lower and upper
@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


# scales an input between 1 and -1
@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


### reward functions
@torch.jit.script
def angular_distance_reward(object_ee_angular_distance, std: float = 0.1):
    r_rot = 1 - torch.tanh(object_ee_angular_distance / std)
    # print("r_rot:", r_rot)
    return r_rot


@torch.jit.script
def distance_reward(object_ee_distance, std: float = 0.1):
    r_reach = 1 - torch.tanh(object_ee_distance / std)
    return r_reach


@torch.jit.script
def lift_reward(object_pos, minimal_height: float, episode_timestep_counter):
    # reward for lifting object
    object_height = object_pos[:, 2]
    is_lifted = torch.where(object_height > minimal_height, 1.0, 0.0)
    is_lifted *= (episode_timestep_counter > 50).float()
    return is_lifted

@torch.jit.script
def insert_success_reward(success: torch.Tensor) -> torch.Tensor:
    return success.to(torch.float32)  # (N,) bool → (N,) float


@torch.jit.script
def object_goal_reward(object_goal_distance, r_lift, std: float = 0.1):
    # tracking
    std = 0.3
    object_goal_tracking = 1 - torch.tanh(object_goal_distance / std)
    # only recieve reward if object is lifted
    object_goal_tracking *= (r_lift > 0).float()
    return object_goal_tracking


@torch.jit.script
def joint_vel_penalty(robot_joint_vel):
    r_joint_vel = torch.sum(torch.square(robot_joint_vel), dim=1)
    return r_joint_vel
