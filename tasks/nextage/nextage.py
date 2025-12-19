# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Shared Nextage parent environment
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
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg, DeformableBodyMaterialCfg
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
from assets.nextage import NEXTAGE_CFG, ROBOTIQ_CFG  # isort: skip

from pxr import Sdf
from isaaclab.sim import SimulationContext
def ensure_xform_prim(prim_path: str) -> bool:
    sim = SimulationContext.instance()
    if sim is None or getattr(sim, "stage", None) is None:
        return False
    stage = sim.stage
    if not stage.GetPrimAtPath(prim_path):
        stage.DefinePrim(Sdf.Path(prim_path), "Xform")
    return True

@configclass
class NextageEnvCfg(DirectRLEnvCfg):
    # physics sim
    physics_dt = 1 / 120  # 0.002 #1 / 500 # 120 # 500 Hz

    # number of physics step per control step
    decimation = 2  # 10 # # 50 Hz

    # the number of physics simulation steps per rendering steps (default=1)
    render_interval = 2
    episode_length_s = 5.0  # 5 * 120 / 2 = 300 timesteps

    num_observations = 0
    num_actions = 16
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
    minimal_width = 0.04
    minimal_aperture = 0.05
    maximum_width = 0.148 ########## need changed from measurements

    act_moving_average = 0.05
    minimal_angular = 10.0 # degree
    minimal_dense = 0.02 # added for dense reward 10/20
    reaching_object_scale = 1
    contact_reward_scale = 10
    stretch_object_scale = 15.0
    object_goal_tracking_scale = 16.0
    joint_vel_penalty_scale = 0  # -0.01
    object_out_of_bounds = 5
    rotation_ee_object_scale = 1.0
    rotation_object_goal_scale = 10.0

    # reach stuff
    min_reach_dist = 0.05

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        # physics_material=RigidBodyMaterialCfg(
        #     static_friction=1.0,
        #     dynamic_friction=1.0,
        # ),
        physics_material=DeformableBodyMaterialCfg(
            youngs_modulus=8.0e7,     # 例: 2e5
            poissons_ratio=0.48,      # 例: 0.35
            density=1000.0,            # 例: 300 kg/m^3
            damping_scale=1.0,
            elasticity_damping=0.012,       # 例: 0.02
            dynamic_friction=0.6,     # 例: 0.6
            ),
            
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            # gpu_max_rigid_contact_count=2**25, # default 2**23
            # gpu_max_rigid_patch_count=2**25, #23, default 5 * 2 ** 15.
            gpu_temp_buffer_capacity=2**20, # default 2**20
            gpu_max_soft_body_contacts= 2**23, # default 2**20 
            gpu_collision_stack_size=2**26, # default 2**26
            # gpu_temp_buffer_capacity=2**19, # default 2**20
            # gpu_max_soft_body_contacts= 2**17, # default 2**20 
            # gpu_collision_stack_size=2**19, # default 2**26
        ),
        render=RenderCfg(
            antialiasing_mode="DLAA",
        )
    )

    # temp
    replicate_physics = False
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8, env_spacing=3, replicate_physics=replicate_physics
    )

    # default_object_pos = [0.5, 0, 0.20]  # 0.055
    eye = (3, 3, 3)
    lookat = (0, 0, 0)

    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    # robot
    robot_cfg: ArticulationCfg = NEXTAGE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Normalisation numbers
    vel_max_magnitude = 3

    # chest_head_name
    chest_head_names = [
        "CHEST_JOINT0",
        "HEAD_JOINT0",
        "HEAD_JOINT1",
    ]
    # arm_joint_name
    arm_names = [
        "LARM_JOINT0","LARM_JOINT1","LARM_JOINT2","LARM_JOINT3","LARM_JOINT4","LARM_JOINT5",
        "RARM_JOINT0","RARM_JOINT1","RARM_JOINT2","RARM_JOINT3","RARM_JOINT4","RARM_JOINT5",
    ]
    # gripper_joint_name
    gripper_names = [
        "finger_joint_L",
        "right_outer_knuckle_joint_L",
        "finger_joint", 
        "right_outer_knuckle_joint",
    ]
    ######################### actuated_joint_names/manual_joint_names ######################### 
    actuated_joint_names = []
    manual_joint_names = []
    actuated_joint_names = arm_names + gripper_names
    manual_joint_names = chest_head_names 
    ###########################################################################################

    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"
    left_l_ee_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_inner_finger_L",
                name="left_l_end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot=[0.7071, -0.7071,0.0 , 0.0]
                    # rot=[0.0, -0.7071, 0.0, -0.7071]
                    # rot=[0.7071, 0.0, 0.7071, 0.0]
                    # rot=[0.0, -0.7071, 0.0, -0.7071]
                ),
            )
        ],
    )

    left_r_ee_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_inner_finger_L",
                name="left_r_end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot=[0.7071, -0.7071,0.0 , 0.0]
                ),
            )
        ],
    )


    right_l_ee_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_inner_finger",
                name="right_l_end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot=[0.7071, -0.7071,0.0 , 0.0]
                ),
            )
        ],
    )

    right_r_ee_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_inner_finger",
                name="right_r_end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot=[0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    anchor_east_marker_cfg = FRAME_MARKER_CFG.copy()
    anchor_east_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    anchor_east_marker_cfg.prim_path = "/World/Visuals/AnchorEastMarker"
    anchor_east_tf_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",  
        debug_vis=False,
        visualizer_cfg=anchor_east_marker_cfg,                  
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Visuals/AnchorEast/Geom",  # target can be regex
                name="anchor_east",
                offset=OffsetCfg(
                    rot=[0.7071, 0.0, 0.0, -0.7071]
                )

            )
        ],
    )

    anchor_west_marker_cfg = FRAME_MARKER_CFG.copy()
    anchor_west_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    anchor_west_marker_cfg.prim_path = "/World/Visuals/AnchorWestMarker"
    anchor_west_tf_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",  
        debug_vis=False,
        visualizer_cfg=anchor_west_marker_cfg,              
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Visuals/AnchorWest/Geom",  # target can be regex
                name="anchor_west",
                offset=OffsetCfg(
                    rot=[0.7071, 0.0, 0.0, -0.7071]
                )
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
    binary_tactile = False
    OPEN_IS_UPPER = True  # True: 上限=開, 下限=閉 / False: 上限=閉, 下限=開


class NextageEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: NextageEnvCfg

    def __init__(self, cfg: NextageEnvCfg, render_mode: str | None = None, **kwargs):

        self.obs_stack = cfg.obs_stack
        super().__init__(cfg, render_mode, **kwargs)

        self.dtype = torch.float32
        self.binary_tactile = cfg.binary_tactile

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_joint_pos_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_joint_pos_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_joint_vel_limits = self.robot.data.joint_vel_limits[0, :].to(device=self.device)


        self.joint_pos_cmd = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.prev_joint_pos_cmd = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.joint_pos_cmd  = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        ############################# list of actuated joints/manual joints #################################
        self.actuated_dof_indices = list()
        self.manual_dof_indices = list()
        policy_set = set(self.cfg.actuated_joint_names)
        manual_set = set(self.cfg.manual_joint_names)
        self.actuated_dof_indices = [
            i for i, n in enumerate(self.robot.joint_names) if n in policy_set
        ]
        self.manual_dof_indices = [
            i for i, n in enumerate(self.robot.joint_names) if n in manual_set
        ]
        # verify joint names
        if len(self.actuated_dof_indices) != len(self.cfg.actuated_joint_names):
            missing = sorted(set(self.cfg.actuated_joint_names) - set(self.robot.joint_names))
            raise RuntimeError(f"actuated_joint_names not found in USD: {missing}")
        if len(self.manual_dof_indices) != len(self.cfg.manual_joint_names):
            missing = sorted(set(self.cfg.manual_joint_names) - set(self.robot.joint_names))
            raise RuntimeError(f"manual_joint_names not found in USD: {missing}")
        #####################################################################################################

        self._all_actuated_dof_indices = self.actuated_dof_indices + self.manual_dof_indices
        self._actuated_names_in_order = [self.robot.joint_names[i] for i in self._all_actuated_dof_indices]
        self._actuated_name_to_col = {self.robot.joint_names[i]: i for i in self._all_actuated_dof_indices}

        self._arm_cols = [self._actuated_name_to_col[n] for n in self.cfg.arm_names]
        self._chest_head_cols = [self._actuated_name_to_col[n] for n in self.cfg.chest_head_names]
        self._grip_cols = [self._actuated_name_to_col[n] for n in self.cfg.gripper_names]
        self.actuated_idx = torch.tensor(self.actuated_dof_indices, device=self.device, dtype=torch.long)
        self._grip_step_count = 0

        # create empty tensors
        n_actuated_policy = len(self.actuated_dof_indices)
        self.actions = torch.zeros((self.num_envs, n_actuated_policy), device=self.device)
        default_joint_pos = self.robot.data.default_joint_pos
        self.joint_pos_cmd[:, self.actuated_dof_indices] = default_joint_pos[:, self.actuated_dof_indices]
        self.prev_joint_pos_cmd[:, self.actuated_dof_indices] = default_joint_pos[:, self.actuated_dof_indices]
        
        self.joint_pos = torch.zeros((self.num_envs, n_actuated_policy), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, n_actuated_policy), device=self.device)
        self.normalised_joint_pos = torch.zeros((self.num_envs, n_actuated_policy), device=self.device)
        self.normalised_joint_vel = torch.zeros((self.num_envs, n_actuated_policy), device=self.device)
        
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.normalised_forces = torch.zeros((self.num_envs, 2), device=self.device)
        self.unnormalised_forces = torch.zeros((self.num_envs, 2), device=self.device)
        self.in_contact = torch.zeros((self.num_envs, 1), device=self.device)
        self.tactile = torch.zeros((self.num_envs, 2), device=self.device)
        
        # glove specific
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # r_gripper right_ee
        self.right_r_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_r_ee_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # l_gripper left_ee
        self.left_l_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_l_ee_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # l_gripper right_ee
        self.right_l_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_l_ee_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # r_gripper left_ee
        self.left_r_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_r_ee_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # east and west edges
        self.east_edge_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.east_edge_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.west_edge_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.west_edge_rot = torch.zeros((self.num_envs, 4), device=self.device)

        # for stretch reward
        self.right_ee_left_ee_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_left_ee_rotation = torch.zeros((self.num_envs, 4), device=self.device)

        self.right_ee_object_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_object_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.right_ee_object_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        self.right_ee_object_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.left_ee_object_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ee_object_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.left_ee_object_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        self.left_ee_object_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
     
        self.ee_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        self.ee_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)

        # for validation of aperture
        self.right_ee_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.left_ee_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ee_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)

        # save reward weights so they can be adjusted online
        self.reaching_object_scale = cfg.reaching_object_scale
        self.contact_reward_scale = cfg.contact_reward_scale
        self.joint_vel_penalty_scale = cfg.joint_vel_penalty_scale
        self.object_goal_tracking_scale = cfg.object_goal_tracking_scale
        self.rotation_ee_object_scale = cfg.rotation_ee_object_scale
        self.rotation_object_goal_scale = cfg.rotation_object_goal_scale
        self.stretch_object_scale = cfg.stretch_object_scale

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
            "right_reach_reward": None,
            "left_reach_reward": None,
            "right_angular_reward": None,
            "left_angular_reward": None,
            "right_insert_reward": None,
            "left_insert_reward": None,
            "stretch_reward": None,
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

        # frag of starting policy
        # self._policy_enabled = False
        # self._phase = "pregrasp"
        # self._pregrasp_steps = 0
        # threshold of grasping
        self._grasp_min_steps = 20
        self._pregrasp_duration_s = 0.5
        self._pregrasp_total_steps = int(
            self._pregrasp_duration_s / (self.cfg.physics_dt * self.cfg.decimation)
        )
        self._grip_latched_q = None # (num_envs, num_grip)
        # --- cache finger joint indices from robot.joint_names (global USD順) ---
        _open_target_by_name = {
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "finger_joint_L": 0.0,
            "right_outer_knuckle_joint_L": 0.0,
        }
        _close_target_by_name = {
            "finger_joint": 0.725,                 # 右: 閉
            "right_outer_knuckle_joint": -0.75,   # 右: 閉（符号反転）
            "finger_joint_L": 0.725,               # 左: 閉
            "right_outer_knuckle_joint_L": -0.75, # 左: 閉（符号反転）
        }

        def _vec_from_name_map(name_map: dict):
            vals = []
            for jname in self.cfg.gripper_names:  # 例: ["finger_joint", "finger_joint_L"]
                if jname not in name_map:
                    raise RuntimeError(f"'{jname}' が name_map にありません")
                vals.append(name_map[jname])
            return torch.tensor(vals, device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, G)


        self._grip_open_vec  = _vec_from_name_map(_open_target_by_name).expand(self.num_envs, -1)   # (N, G)
        self._grip_close_vec = _vec_from_name_map(_close_target_by_name).expand(self.num_envs, -1)  # (N, G)
        self._grasp = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

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
        self.left_l_ee_frame = FrameTransformer(self.cfg.left_l_ee_config)
        self.left_l_ee_frame.set_debug_vis(False)
        self.right_r_ee_frame = FrameTransformer(self.cfg.right_r_ee_config)
        self.right_r_ee_frame.set_debug_vis(False)
        self.left_r_ee_frame = FrameTransformer(self.cfg.left_r_ee_config)
        self.left_r_ee_frame.set_debug_vis(False)
        self.right_l_ee_frame = FrameTransformer(self.cfg.right_l_ee_config)
        self.right_l_ee_frame.set_debug_vis(False)

        rb_east_path_env0 = "/World/envs/env_0/Visuals/AnchorEast/Geom"
        rb_west_path_env0 = "/World/envs/env_0/Visuals/AnchorWest/Geom"

        stage = SimulationContext.instance().stage
        ensure_xform_prim("/World/Visuals") 
        if not stage.GetPrimAtPath(rb_east_path_env0):
            anchor_rb_cfg = sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                physics_material=RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
            anchor_rb_cfg.func(rb_east_path_env0, anchor_rb_cfg)
        
        if not stage.GetPrimAtPath(rb_west_path_env0):
            anchor_rb_cfg = sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                physics_material=RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
            anchor_rb_cfg.func(rb_west_path_env0, anchor_rb_cfg)

        if not stage.GetPrimAtPath("/World/ground"):
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(10000, 10000)))

        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False) # change False -> True 10/23
        
        self._anchor_east_rb_path = "/World/envs/env_.*/Visuals/AnchorEast/Geom"
        self.anchor_east_rb = RigidObject(RigidObjectCfg(prim_path=self._anchor_east_rb_path))
        self.scene.rigid_objects["anchor_east"] = self.anchor_east_rb
        self.anchor_east_tf = FrameTransformer(self.cfg.anchor_east_tf_cfg)
        self._anchor_west_rb_path = "/World/envs/env_.*/Visuals/AnchorWest/Geom"
        self.anchor_west_rb = RigidObject(RigidObjectCfg(prim_path=self._anchor_west_rb_path))
        self.scene.rigid_objects["anchor_west"] = self.anchor_west_rb
        self.anchor_west_tf = FrameTransformer(self.cfg.anchor_west_tf_cfg)
        # register to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["left_l_ee_frame"] = self.left_l_ee_frame
        self.scene.sensors["right_r_ee_frame"] = self.right_r_ee_frame
        self.scene.sensors["left_r_ee_frame"] = self.left_r_ee_frame
        self.scene.sensors["right_l_ee_frame"] = self.right_l_ee_frame
        self.scene.sensors["anchor_east_tf"] = self.anchor_east_tf
        self.scene.sensors["anchor_west_tf"] = self.anchor_west_tf
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
            

        # if "tactile" in self.cfg.obs_list:
        #     self.left_contact_sensor = ContactSensor(self.cfg.left_contact_cfg)
        #     self.scene.sensors["left_contact_sensor"] = self.left_contact_sensor

        #     self.right_contact_sensor = ContactSensor(self.cfg.right_contact_cfg)
        #     self.scene.sensors["right_contact_sensor"] = self.right_contact_sensor

        #     self.wholebody_contact_sensor = ContactSensor(self.cfg.wholebody_contact_cfg)
        #     self.scene.sensors["wholebody_contact_sensor"] = self.wholebody_contact_sensor
        self.anchor_east_tf.set_debug_vis(True)
        self.anchor_west_tf.set_debug_vis(True)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Store actions from policy in a class variable
        """
        self.last_action = self.joint_pos_cmd[:, self.actuated_dof_indices]
        self.actions = actions.clone()

    # def _apply_action(self) -> None:
    #     """
    #     The _apply_action(self) API is called decimation number of times for each RL step, prior to taking each physics step.
    #     This provides more flexibility for environments where actions should be applied for each physics step.
    #     """
    #     scaled_actions = self.scale_action(self.actions)

    #     self.robot.set_joint_position_target(scaled_actions, joint_ids=self.actuated_dof_indices)

    def _apply_action(self) -> None:
        """
        Apply actions to the robot. Called multiple times per RL step for decimation.
        """
        self.joint_pos_cmd[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
            self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
        )
        self.joint_pos_cmd[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.joint_pos_cmd[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_joint_pos_cmd[:, self.actuated_dof_indices]
        )
        self.joint_pos_cmd[:, self.actuated_dof_indices] = saturate(
            self.joint_pos_cmd[:, self.actuated_dof_indices],
            self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
            self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
        )

        self.prev_joint_pos_cmd[:, self.actuated_dof_indices] = self.joint_pos_cmd[:, self.actuated_dof_indices]

        self.robot.set_joint_position_target(
            self.joint_pos_cmd[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def scale_action(self, action):
        self.joint_pos_cmd[:, self.actuated_dof_indices] = scale(
            action,
            self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
            self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
        )

        self.joint_pos_cmd[:, self.actuated_dof_indices] = saturate(
            self.joint_pos_cmd[:, self.actuated_dof_indices],
            self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
            self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
        )
        return self.joint_pos_cmd[:, self.actuated_dof_indices]
    
    # def _is_grasped(self) -> torch.Tensor:
    #     """
    #     各envごとに把持が成立したかのboolテンソルを返す (num_envs,)
    #     1) tactile(接触)があるならそれを優先
    #     2) ない場合は EE間距離 と 開口量 からの簡易判定
    #     """
    #     grasp = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    #     use_tactile = hasattr(self, "tactile") and (self.tactile is not None)
    #     if use_tactile and "tactile" in self.cfg.obs_list:
    #         # 例: 左右の接触がともに閾値超え
    #         # self.tactile: (num_envs, 2) を想定 [L, R]（あなたの実装に合わせて変更）
    #         left_hit  = self.tactile[:, 0] > 0.5
    #         right_hit = self.tactile[:, 1] > 0.5
    #         grasp = left_hit & right_hit
    #     else:
    #         left_closed  = self.left_aperture  < self._grasp_min_aperture
    #         right_closed = self.right_aperture < self._grasp_min_aperture
    #         print(f"left_aperture: {self.left_aperture[0]} left closed : {left_closed}")
    #         print(f"right_aperture: {self.right_aperture[0]}  right closed : {right_closed}")

    #         grasp = left_closed & right_closed

    #     if self._pregrasp_steps < self._grasp_min_steps:
    #         grasp[:] = False
    #     return grasp

    # def _scripted_pregrasp(self, q_cmd):
    #     p = min(1.0, float(self._pregrasp_steps) / max(1, self._pregrasp_total_steps))
    #     grip_q = self._grip_q_from_ratio(p)  # (N,G)

    #     # まず現在の全関節にベースを書き、グリッパ列だけ上書き
    #     q_cmd = self.robot.data.joint_pos[:, :].clone()
    #     for i, col in enumerate(self._grip_cols):
    #         q_cmd[:, col] = grip_q[:, i]

    #     # クランプ＆部分ターゲット
    #     lower_all = self.robot_joint_pos_lower_limits[:]
    #     upper_all = self.robot_joint_pos_upper_limits[:]
    #     q_cmd = torch.clamp(q_cmd, lower_all, upper_all)
    #     q_cmd_gripper = q_cmd[:, self._grip_cols]
    #     self.robot.set_joint_position_target(q_cmd_gripper, joint_ids=self._grip_cols)

    #     self._pregrasp_steps += 1

    # def _grip_limits(self):
    #     all_idx = torch.tensor(self._all_actuated_dof_indices, device=self.device, dtype=torch.long)
    #     lower_all = self.robot_joint_pos_lower_limits[all_idx]
    #     upper_all = self.robot_joint_pos_upper_limits[all_idx]
    #     lower_g = self.robot_joint_pos_lower_limits[self._grip_cols].unsqueeze(0).expand(self.num_envs, -1)
    #     upper_g = self.robot_joint_pos_upper_limits[self._grip_cols].unsqueeze(0).expand(self.num_envs, -1)
        
    #     return lower_g, upper_g, lower_all, upper_all

    # def _grip_open_q(self):
    #     # (num_envs, num_grip)
    #     return self._grip_open_vec

    # def _grip_close_q(self):
    #     return self._grip_close_vec

    # def _grip_q_from_ratio(self, s: torch.Tensor):
    #     """
    #     s: (num_envs,) or スカラー in [0,1]  — 0=開, 1=閉
    #     返り値: (num_envs, num_grip)
    #     """
    #     if not torch.is_tensor(s):
    #         s = torch.tensor(float(s), device=self.device)
    #     if s.dim() == 0:
    #         s = s.expand(self.num_envs)
    #     s = torch.clamp(s, 0.0, 1.0).view(-1, 1)
    #     return (1.0 - s) * self._grip_open_vec + s * self._grip_close_vec

    # def _set_grip_q(self, q_cmd, grip_q):
    #     for i, col in enumerate(self._grip_cols):
    #         q_cmd[:, col] = grip_q[:, i]
    #     return q_cmd

    # def _pre_physics_step(self, actions: torch.Tensor) -> None:
    #     """
    #     Store actions from policy in a class variable
    #     """
    #     if not self._policy_enabled:
    #         self.actions = torch.zeros_like(actions)
    #     else:
    #         self.actions = actions.clone()
    #     self.last_action = self.joint_pos_cmd[:, self.actuated_dof_indices]

    # def _apply_action(self):
    #     q_cmd = self.scale_action(self.actions)

    #     if not self._policy_enabled:
    #         self._scripted_pregrasp(q_cmd)

    #         if torch.all(self._grasp):
    #             # 実測角（関節読み出し）でラッチ（コンプライアンス考慮）
    #             # robot.data.joint_pos は全DOF順なので、_all_actuated_dof_indices→_grip_colsで拾う
    #             all_idx = torch.tensor(self._all_actuated_dof_indices, device=self.device, dtype=torch.long)
    #             pos_full = self.robot.data.joint_pos[:, all_idx]  # (num_envs, actuated)
    #             latched = torch.stack([pos_full[:, col] for col in self._grip_cols], dim=1)  # (num_envs, num_grip)
    #             self._grip_latched_q = latched.clone()

    #             self._policy_enabled = True
    #             self._phase = "policy"
    #             print("[INFO]: POLICY (grip latched)")
    #         else:
    #             print("[INFO] PREGRASP...")
    #         return

    #     # ←←← ここから policy フェーズ
    #     # 1) 以前のサイン波開閉は削除（保持のため）
    #     _, _, lower_all, upper_all = self._grip_limits()

    #     # 2) 腕などは policy 出力を使い、grip はラッチ値で上書き
    #     if self._grip_latched_q is None:
    #         self._grip_latched_q = self._grip_close_q()

    #     self._set_grip_q(q_cmd, self._grip_latched_q)

    #     q_cmd = torch.clamp(q_cmd, lower_all, upper_all)
    #     self.robot.set_joint_position_target(q_cmd, joint_ids=self._all_actuated_dof_indices)
    #     # self.robot.set_joint_position_target(q_cmd)
    #     print("[INFO] POLICY (grip hold)")
    #     self._grip_step_count += 1

    # def scale_action(self, action):
    #     self.joint_pos_cmd[:, self.actuated_dof_indices] = scale(
    #         action,
    #         self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
    #         self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
    #     )

    #     self.joint_pos_cmd[:, self.actuated_dof_indices] = saturate(
    #         self.joint_pos_cmd[:, self.actuated_dof_indices],
    #         self.robot_joint_pos_lower_limits[self.actuated_dof_indices],
    #         self.robot_joint_pos_upper_limits[self.actuated_dof_indices],
    #     )
    #     return self.joint_pos_cmd[:, self._all_actuated_dof_indices]

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
                self.right_ee_euclidean_distance.unsqueeze(1),# right ee aperture
                self.left_ee_euclidean_distance.unsqueeze(1), # light ee aperture
                self.right_l_ee_pos,
                self.right_l_ee_rot,
                self.right_r_ee_pos,
                self.right_r_ee_rot,
                self.left_l_ee_pos,
                self.left_l_ee_rot,
                self.left_r_ee_pos,
                self.left_r_ee_rot,
                self.actions,
            ),
            dim=-1,
        )

        return prop

    def _get_gt(self):

        gt = torch.cat(
            (
                # xyz diffs (3,)
                self.right_ee_object_distance,
                # rotation quaternion (4,)
                self.right_ee_object_rotation,
                # xyz diffs (3,)
                self.left_ee_object_distance,
                # rotation quaternion (4,)
                self.left_ee_object_rotation,
                # euclidean distance (1,)
                self.right_ee_euclidean_distance.unsqueeze(1),
                # euclidean distance (1,)
                self.left_ee_euclidean_distance.unsqueeze(1),
                # rotation difference (1,)
                self.ee_angular_distance.unsqueeze(1),
                # euclidean distance (1,)
                self.ee_euclidean_distance.unsqueeze(1),
                # rotation difference (1,)
                self.right_ee_object_angular_distance.unsqueeze(1),
                # euclidean distance (1,)
                self.right_ee_object_euclidean_distance.unsqueeze(1),
                # rotation difference (1,)
                self.left_ee_object_angular_distance.unsqueeze(1),
                # euclidean distance (1,)
                self.left_ee_object_euclidean_distance.unsqueeze(1),
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
        _ = self._get_rewards()

        # self._policy_enabled = False
        # self._phase = "pregrasp"
        # self._pregrasp_steps = 0
        # self._grip_latched_q = None
        # print("[INFO]: PHASE RESET → pregrasp")

    def _reset_deformable_pose(self, env_ids):

        nodal_state = self.object.data.default_nodal_state_w.clone()[env_ids]  
        N = int(nodal_state.shape[1])

        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        pos_w = self.cfg.reset_object_position_noise * pos_noise * 0
        
        quat_w = math_utils.quat_from_euler_xyz(
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
            torch.zeros(len(env_ids), device=self.device),
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
        # all_idx = torch.tensor(self._all_actuated_dof_indices, device=self.device, dtype=torch.long)
        # default_q = self.robot.data.default_joint_pos[env_ids][:, all_idx]
        default_q = self.robot.data.default_joint_pos[env_ids][:, :]
        joint_vel = torch.zeros_like(default_q)

        self.robot.set_joint_position_target(
            default_q, env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            default_q, joint_vel, env_ids=env_ids
        )
        q_cmd = default_q.clone()
        self.robot.set_joint_position_target(q_cmd, env_ids=env_ids)

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

        try:
            self._idx_left_finger  = self.robot.joint_names.index("finger_joint_L")
            self._idx_right_finger = self.robot.joint_names.index("finger_joint")
        except ValueError as e:
            raise RuntimeError(f"Finger joint name not found in USD: {e}")
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
        pos_full = self.robot.data.joint_pos[env_ids]
        vel_full = self.robot.data.joint_vel[env_ids]
        self.joint_pos[env_ids] = pos_full[:, self.actuated_idx]
        self.joint_vel[env_ids] = vel_full[:, self.actuated_idx]

        lower = self.robot_joint_pos_lower_limits[self.actuated_idx]
        upper = self.robot_joint_pos_upper_limits[self.actuated_idx]
        self.normalised_joint_pos[env_ids] = unscale(self.joint_pos[env_ids], lower, upper)

        self.right_r_ee_pos[env_ids] = self.right_r_ee_frame.data.target_pos_source[..., 0, :][env_ids]
        self.right_r_ee_rot[env_ids] = self.right_r_ee_frame.data.target_quat_source[..., 0, :][env_ids]
        self.left_l_ee_pos[env_ids] = self.left_l_ee_frame.data.target_pos_source[..., 0, :][env_ids]
        self.left_l_ee_rot[env_ids] = self.left_l_ee_frame.data.target_quat_source[..., 0, :][env_ids]

        # validate aperature  
        self.right_l_ee_pos[env_ids] = self.right_l_ee_frame.data.target_pos_source[..., 0, :][env_ids]
        self.right_l_ee_rot[env_ids] = self.right_l_ee_frame.data.target_quat_source[..., 0, :][env_ids]
        self.left_r_ee_pos[env_ids] = self.left_r_ee_frame.data.target_pos_source[..., 0, :][env_ids]
        self.left_r_ee_rot[env_ids] = self.left_r_ee_frame.data.target_quat_source[..., 0, :][env_ids]

        # joint vel roughly between -2.5, 2.5, so dividing by 3.
        self.normalised_joint_vel[env_ids] = self.joint_vel[env_ids] / self.cfg.vel_max_magnitude
        self.nodal_state[env_ids] = self.object.data.nodal_state_w[env_ids]
        self.object_pos[env_ids] = self.nodal_state[:, self.anchor_idx["east"], :3] - self.scene.env_origins[env_ids]
        ################ need ????????????#######################
        # self.object_pos[env_ids] = self.nodal_state[:, self.anchor_idx["west"], :3] - self.scene.env_origins[env_ids]
        
        # deformable doesn't have quat
        if self.cfg.object_type == "rigid":
            self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        if self.cfg.object_type == "deformable":
            self.object_rot[env_ids] = self.estimate_node_quat(node_idx=self.anchor_idx["east"],env_ids=env_ids)

        #################### validate aperture ############ 
        self.right_ee_distance[env_ids] = self.right_r_ee_pos[env_ids] - self.right_l_ee_pos[env_ids]
        self.right_ee_euclidean_distance[env_ids] = torch.norm(self.right_ee_distance[env_ids], dim=1)
        self.left_ee_distance[env_ids] = self.left_r_ee_pos[env_ids] - self.left_l_ee_pos[env_ids]
        self.left_ee_euclidean_distance[env_ids] = torch.norm(self.left_ee_distance[env_ids], dim=1)

        # relative distances
        ################################## conform of left/right ee pos ##########################
        self.right_ee_left_ee_distance[env_ids] = self.right_r_ee_pos[env_ids] - self.left_l_ee_pos[env_ids]
        self.ee_euclidean_distance[env_ids] = torch.norm(self.right_ee_left_ee_distance[env_ids], dim=1)
        # print("ee_euclidian_distance:",   self.ee_euclidean_distance[0])
        self.right_ee_left_ee_rotation[env_ids] = quat_mul(self.right_r_ee_rot[env_ids], quat_conjugate(self.left_l_ee_rot[env_ids]))
        self.ee_angular_distance[env_ids] = rotation_distance(self.right_r_ee_rot[env_ids], self.left_l_ee_rot[env_ids])
        # print("ee_angular_distance:",   self.ee_angular_distance[0])

        env_all = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        
        root_east = self.anchor_east_rb.data.default_root_state.clone().to(self.device)  
        east_idx = self.anchor_idx["east"]
        pos_w_east = self.nodal_state[:, east_idx, :3].to(self.device)          
        quat_wxyz_east  = self.estimate_node_quat(node_idx=east_idx).to(self.device)  
        root_east[:, 0:3] = pos_w_east
        root_east[:, 3:7] = quat_wxyz_east
        self.anchor_east_rb.write_root_state_to_sim(root_east, env_ids=env_all)

        root_west = self.anchor_west_rb.data.default_root_state.clone().to(self.device)
        west_idx = self.anchor_idx["west"]
        pos_w_west = self.nodal_state[:, west_idx, :3].to(self.device)
        quat_w_west = self.estimate_node_quat(node_idx=west_idx).to(self.device)
        root_west[:, 0:3] = pos_w_west
        root_west[:, 3:7] = quat_w_west
        self.anchor_west_rb.write_root_state_to_sim(root_west, env_ids=env_all)

        # add insert reward computation 10/20
        self.east_edge_pos[env_ids] = self.anchor_east_tf.data.target_pos_source[..., 0, :][env_ids]
        self.east_edge_rot[env_ids] = self.anchor_east_tf.data.target_quat_source[..., 0, :][env_ids]
        self.west_edge_pos[env_ids] = self.anchor_west_tf.data.target_pos_source[..., 0, :][env_ids]
        self.west_edge_rot[env_ids] = self.anchor_west_tf.data.target_quat_source[..., 0, :][env_ids]

        # gripper and ee distances #############################################################checkn direction!!!!!!!!!
        self.right_ee_object_distance[env_ids] = self.right_l_ee_pos[env_ids] -  self.west_edge_pos[env_ids]
        self.right_ee_object_euclidean_distance[env_ids] = torch.norm(self.right_ee_object_distance[env_ids], dim=1)
        self.right_ee_object_rotation[env_ids] = quat_mul(self.right_l_ee_rot[env_ids], quat_conjugate(self.west_edge_rot[env_ids]))
        self.right_ee_object_angular_distance[env_ids] = rotation_distance(self.right_l_ee_rot[env_ids], self.west_edge_rot[env_ids])

        self.left_ee_object_distance[env_ids] = self.left_r_ee_pos[env_ids] -  self.east_edge_pos[env_ids]
        self.left_ee_object_euclidean_distance[env_ids] = torch.norm(self.left_ee_object_distance[env_ids], dim=1)
        self.left_ee_object_rotation[env_ids] = quat_mul(self.left_r_ee_rot[env_ids], quat_conjugate(self.east_edge_rot[env_ids]))
        self.left_ee_object_angular_distance[env_ids] = rotation_distance(self.left_r_ee_rot[env_ids], self.east_edge_rot[env_ids])

        if not self._vis_enabled:
            if (self.right_r_ee_frame.data.target_pos_source is not None
                and self.anchor_east_tf.data is not None and self.anchor_west_tf.data is not None):
                # self.anchor_east_tf.set_debug_vis(True) 
                self.anchor_west_tf.set_debug_vis(True) 
                # self.right_r_ee_frame.set_debug_vis(True) 
                # self.right_l_ee_frame.set_debug_vis(True) # right-west
                # self.left_l_ee_frame.set_debug_vis(True) 
                # self.left_r_ee_frame.set_debug_vis(True)   # left-east
                self._vis_enabled = True

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # no termination at the moment
        out_of_reach =self.object_pos[:,2] < 0.5
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

# def distance_reward(object_ee_distance, std: float = 0.1):
#     r_reach = 1 - torch.tanh(object_ee_distance / std)
#     return r_reach
@torch.jit.script
def distance_reward(ee_goal_distance, ee_distance, ee_object_distance, minimal_aperture: float, minimal_width: float, std: float = 0.1):
    minimal_aperture = torch.tensor(minimal_aperture, device=ee_distance.device)
    minimal_width = torch.tensor(minimal_width, device=ee_distance.device)
    # print(f"ee_distance: {ee_distance} minimal_aperture: {minimal_aperture}")

    condition = (ee_object_distance < minimal_width) #& (ee_distance < minimal_aperture) 
    ee_goal_distance = torch.where(condition, ee_goal_distance, torch.tensor(1e6, device=ee_goal_distance.device))
    r_reach = 1 - torch.tanh(ee_goal_distance / std) 
    return r_reach

@torch.jit.script
def stretch_reward(ee_distance, episode_timestep_counter, min_width: float = 0.05, max_width: float = 0.148):
    condition = (ee_distance > min_width) & (ee_distance < max_width)
    # reward for stretching gripper
    is_stretched = torch.where(condition, 1.0, 0.0)
    # is_stretched *= (episode_timestep_counter > 10).float()
    is_stretched *= 1.0

    return is_stretched

@torch.jit.script
def lift_reward(object_pos, object_ee_angular_distance, minimal_height: float, minimal_angular: float, episode_timestep_counter):
    # reward for lifting object
    object_height = object_pos[:, 2]
    condition = (object_height > minimal_height) & (torch.abs(object_ee_angular_distance) < torch.deg2rad(torch.tensor(minimal_angular)))
    is_lifted = torch.where(condition, 1.0, 0.0)
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
    # object_goal_tracking *= (r_lift > 0).float()
    return object_goal_tracking

@torch.jit.script
def joint_vel_penalty(robot_joint_vel):
    r_joint_vel = torch.sum(torch.square(robot_joint_vel), dim=1)
    return r_joint_vel
