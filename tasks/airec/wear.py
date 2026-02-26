# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, DeformableObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialCfg,   
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, DeformableBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
    sample_uniform,
    saturate,
)
from tasks.airec.airec import AIRECEnv, AIRECEnvCfg, insert_success_reward, randomize_rotation, rotation_distance, scale
from isaaclab.sensors import (
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    # TiledCamera,
    # TiledCameraCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
import sys
sys.path.append("tasks/airec")
from insert_rew import InsertReward

@configclass
class WearEnvCfg(AIRECEnvCfg):

    # reset config
    reset_object_position_noise = 0.05
    reset_goal_position_noise = 0.01  # scale factor for -1 to 1 m
    default_goal_pos = [0.5, 0.5, 0.4]
    default_right_goal_pos = [0.70, -0.050, 0.607]
    default_left_goal_pos = [0.70, 0.050, 0.607]
    default_object_pos = [0.09, 0.00, 1.25] # 0.13 # 1.07

    object_goal_tracking_scale = 16.0
    object_goal_tracking_finegrained_scale = 5.0

    object_usd = '/home/tamon/code/isaaclab_rl_wearglove/assets/Glove/GL_Gloves068/GL_Gloves068_obj_revise.usd'

    object_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[0.7071, 0.0, 0.7071, 0.0]),#rot=[0.7071, 0.0, 0.7071, 0.0] rot=[1.0, 0.0, 0.0, 0.0]
        spawn=UsdFileCfg(
            usd_path=object_usd,
            copy_from_source=True,
            visible=True,
            scale=(1.0, 1.4, 1.3), # internship:scale=(1.0, 1.4, 1.3), (1.0, 1.3, 1.2)
            # scale=(1.0, 1.5, 1.5),

            deformable_props=DeformableBodyPropertiesCfg(
                deformable_enabled=True,
                kinematic_enabled=False,
                self_collision=True,
                simulation_hexahedral_resolution=45,  # default 10 
                collision_simplification=True,
                collision_simplification_remeshing=True,
                collision_simplification_remeshing_resolution=30, # 40
                collision_simplification_target_triangle_count=0,
                collision_simplification_force_conforming=True,
                # contact_offset=0.005, # default
                # rest_offset=0.003, # default
                # contact_offset=0.015,
                # rest_offset=0.01,
                contact_offset=0.025,
                rest_offset=0.015,
                # contact_offset=0.01,
                # rest_offset=-0.01,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.2, 0.2),
            opacity=1.0,             
        ),   
        ),
        debug_vis=False,
    )
    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"

    # goal frame transformers
    goal_marker_cfg = FRAME_MARKER_CFG.copy()
    goal_marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    goal_marker_cfg.prim_path = "/World/Visuals/GoalMarker"
    # goal_marker_cfg.prim_path = "/World/envs/env_.*/Visuals/GoalMarker"


    right_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/world",
        debug_vis=True,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # prim_path="/World/envs/env_.*/Visuals/RightGoal/Geom",
                prim_path="/World/envs/env_.*/ShadowHand/robot0_thdistal",
                name="right_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    # rot=[0.7071, 0.0, 0.0, -0.7071]
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    left_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/world",
        debug_vis=True,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # prim_path="/World/envs/env_.*/Visuals/LeftGoal/Geom",
                prim_path="/World/envs/env_.*/ShadowHand/robot0_lfdistal",
                name="left_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    wrist_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/world",
        debug_vis=True,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/ShadowHand/robot0_wrist",
                name="wrist_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    # glove edge point(N, S, E, W) 
    glove_north: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    glove_south: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    glove_east: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )

    glove_west: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            ),
        }
    )

    glove_cent: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            ),
        }
    )


class WearEnv(AIRECEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: WearEnvCfg

    def __init__(self, cfg: WearEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Pregrasp phase configuration
        self._pregrasp_total_steps = 500 # Number of steps to close gripper (2 seconds at 60Hz)
        self._pregrasp_distance_threshold = 0.005  # Threshold distance to consider grasp successful (1mm)
        
        # Define open and close positions for grippers
        # These are the joint positions for the robot's finger joints
        num_grip_joints = len(self._grip_cols)
        self._grip_open_vec  = torch.zeros((self.num_envs, num_grip_joints), device=self.device)
        self._grip_close_vec = torch.zeros((self.num_envs, num_grip_joints), device=self.device)

        close_ratio = 0.10  # 0.0=lower (完全に閉じる) / 0.1=少し開けたまま # 0.3 close without penetrating
        for i, col in enumerate(self._grip_cols):
            lo = self.robot_dof_lower_limits[col]  # だいたい0.00
            hi = self.robot_dof_upper_limits[col]  # だいたい0.04

            # open: default（または hi に固定したいなら hi）
            self._grip_open_vec[:, i] = self.robot.data.default_joint_pos[0, col]
            # close: lower側へ
            self._grip_close_vec[:, i] = lo + (hi - lo) * close_ratio

        # define glove opening edges
        self.goal_north_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_north_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_south_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_south_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_east_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_east_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_west_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_west_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        self.goal_cent_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_cent_pos[:, :] = torch.tensor(self.cfg.default_goal_pos, device=self.device)
        self.goal_cent_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        # right and left goal positions/rotations
        self.right_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.right_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.left_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.left_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_wrist_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_wrist_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        
        # Initialize pregrasp phase variables for each environment
        self._policy_enabled = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._pregrasp_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # goal related tensors
        # self.right_glove_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_glove_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_glove_goal_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.right_glove_goal_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.right_glove_goal_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        # self.left_glove_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_glove_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_glove_goal_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.left_glove_goal_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.left_glove_goal_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        
        # save reward weights so they can be adjusted online
        self.object_goal_tracking_scale = cfg.object_goal_tracking_scale
        self.object_goal_tracking_finegrained_scale = cfg.object_goal_tracking_finegrained_scale

        # default goal positions
        self.default_right_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.default_right_goal_pos[:, :] = torch.tensor(self.cfg.default_right_goal_pos, device=self.device)
        self.default_left_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.default_left_goal_pos[:, :] = torch.tensor(self.cfg.default_left_goal_pos, device=self.device)

        # over/under distance reward
        self.wrist_glove_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.wrist_glove_euclidean_distance = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        self.top_wrist_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.top_wrist_euclidean_distance = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        self.under_wrist_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.under_wrist_euclidean_distance = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)

        # insert controller
        self.insert_reward = InsertReward(self.num_envs, device=self.device, inward_assume="+x")
        self.right_insert_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.right_insert_dwell = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.left_insert_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.left_insert_dwell = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    def _check_grasp_achieved(self) -> torch.Tensor:
        """
        Check if the gripper has successfully grasped the object.
        Returns a boolean tensor indicating which environments have achieved grasp.
        """
        # Use hasattr to avoid errors before first compute_intermediate_values call
        if not hasattr(self, 'east_edge_pos') or not hasattr(self, 'west_edge_pos'):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Check distance between right gripper fingers and east anchor
        # Both right fingers (r and l) should be close to east anchor
        right_r_to_east_dist = torch.norm(self.right_gripper_r_pos - self.east_edge_pos, dim=1)
        right_l_to_east_dist = torch.norm(self.right_gripper_l_pos - self.east_edge_pos, dim=1)
        right_close_to_east = (right_r_to_east_dist < self._pregrasp_distance_threshold) & \
                              (right_l_to_east_dist < self._pregrasp_distance_threshold)
        
        # Check distance between left gripper fingers and west anchor
        # Both left fingers (r and l) should be close to west anchor
        left_r_to_west_dist = torch.norm(self.left_gripper_r_pos - self.west_edge_pos, dim=1)
        left_l_to_west_dist = torch.norm(self.left_gripper_l_pos - self.west_edge_pos, dim=1)
        left_close_to_west = (left_r_to_west_dist < self._pregrasp_distance_threshold) & \
                             (left_l_to_west_dist < self._pregrasp_distance_threshold)
        
        # Both conditions must be satisfied for successful grasp (symmetric closing)
        grasp_achieved = right_close_to_east & left_close_to_west
        return grasp_achieved

    def _apply_action(self) -> None:
        """
        Override parent's _apply_action to handle pregrasp phase.
        During pregrasp phase, close the gripper until grasp is achieved.
        After grasp, enable policy and apply normal actions.
        """
        # Identify environments still in pregrasp phase
        pregrasp_env_ids = (~self._policy_enabled).nonzero(as_tuple=False).squeeze(-1)
        
        if len(pregrasp_env_ids) > 0:
            # Pregrasp phase: gradually close the gripper for environments in pregrasp
            # Calculate progress ratio [0, 1] for each environment
            progress = torch.clamp(
                self._pregrasp_steps[pregrasp_env_ids].float() / max(1, self._pregrasp_total_steps),
                0.0, 1.0
            ).unsqueeze(1)  # (N, 1)
            
            # Interpolate between open and close positions
            grip_q = (1.0 - progress) * self._grip_open_vec[pregrasp_env_ids] + progress * self._grip_close_vec[pregrasp_env_ids]
            
            # Set default positions for all joints first
            default_joint_pos = self.robot.data.default_joint_pos[pregrasp_env_ids]
            joint_vel = torch.zeros_like(default_joint_pos)
            joint_pos_cmd_pregrasp = default_joint_pos.clone()
            
            # Apply gripper commands
            for i, col in enumerate(self._grip_cols):
                joint_pos_cmd_pregrasp[:, col] = grip_q[:, i]
            
            # Clamp to joint limits
            lower_all = self.robot_dof_lower_limits[:]
            upper_all = self.robot_dof_upper_limits[:]
            joint_pos_cmd_pregrasp = torch.clamp(joint_pos_cmd_pregrasp, lower_all, upper_all)
            
            # Set joint targets and write to sim (only for pregrasp envs)
            self.joint_pos_cmd[pregrasp_env_ids] = joint_pos_cmd_pregrasp
            self.prev_joint_pos_cmd[pregrasp_env_ids] = joint_pos_cmd_pregrasp
            self.robot.set_joint_position_target(joint_pos_cmd_pregrasp, env_ids=pregrasp_env_ids)
            self.robot.write_joint_state_to_sim(joint_pos_cmd_pregrasp, joint_vel, env_ids=pregrasp_env_ids)
            
            # Increment step counter for pregrasp environments
            self._pregrasp_steps[pregrasp_env_ids] += 1
            
            # Check if grasp is achieved
            grasp_achieved = self._check_grasp_achieved()
            
            # Enable policy for environments that achieved grasp or reached max steps
            max_steps_reached = self._pregrasp_steps >= self._pregrasp_total_steps
            enable_policy_mask = grasp_achieved | max_steps_reached
            enable_env_ids = enable_policy_mask.nonzero(as_tuple=False).squeeze(-1)
            self._policy_enabled[enable_env_ids] = True
            
        # Policy phase: apply actions from policy for environments with policy enabled
        policy_env_ids = self._policy_enabled.nonzero(as_tuple=False).squeeze(-1)
        
        if len(policy_env_ids) > 0:
            # Scale actions from [-1, 1] to joint limits
            self.joint_pos_cmd[policy_env_ids[:, None], self.actuated_dof_indices] = scale(
                self.actions[policy_env_ids],
                self.robot_dof_lower_limits[self.actuated_dof_indices],
                self.robot_dof_upper_limits[self.actuated_dof_indices],
            )
            
            # Apply moving average for smooth control
            self.joint_pos_cmd[policy_env_ids[:, None], self.actuated_dof_indices] = (
                self.cfg.act_moving_average * self.joint_pos_cmd[policy_env_ids[:, None], self.actuated_dof_indices]
                + (1.0 - self.cfg.act_moving_average) * self.prev_joint_pos_cmd[policy_env_ids[:, None], self.actuated_dof_indices]
            )
            
            # Clamp to joint limits
            self.joint_pos_cmd[policy_env_ids[:, None], self.actuated_dof_indices] = saturate(
                self.joint_pos_cmd[policy_env_ids[:, None], self.actuated_dof_indices],
                self.robot_dof_lower_limits[self.actuated_dof_indices],
                self.robot_dof_upper_limits[self.actuated_dof_indices],
            )
            
            # Update previous command
            self.prev_joint_pos_cmd[policy_env_ids[:, None], self.actuated_dof_indices] = self.joint_pos_cmd[policy_env_ids[:, None], self.actuated_dof_indices]
            
            # Keep gripper joints at their closed position (don't let policy control them)
            for i, col in enumerate(self._grip_cols):
                self.joint_pos_cmd[policy_env_ids, col] = self._grip_close_vec[policy_env_ids, i]
            
            # Set joint position targets (only for policy envs)
            self.robot.set_joint_position_target(
                self.joint_pos_cmd[policy_env_ids],
                env_ids=policy_env_ids
            )

    def _setup_scene(self):
        super()._setup_scene()
        self.goal_north_markers = VisualizationMarkers(self.cfg.glove_north)
        self.goal_south_markers = VisualizationMarkers(self.cfg.glove_south)
        self.goal_east_markers  = VisualizationMarkers(self.cfg.glove_east)
        self.goal_west_markers  = VisualizationMarkers(self.cfg.glove_west)
        self.goal_cent_markers  = VisualizationMarkers(self.cfg.glove_cent)

        self.right_goal_frame = FrameTransformer(self.cfg.right_goal_config)
        self.right_goal_frame.set_debug_vis(True)
        self.left_goal_frame = FrameTransformer(self.cfg.left_goal_config)
        self.left_goal_frame.set_debug_vis(True)
        self.wrist_goal_frame = FrameTransformer(self.cfg.wrist_goal_config)
        self.wrist_goal_frame.set_debug_vis(True)

        self.scene.sensors["left_goal_frame"] = self.left_goal_frame
        self.scene.sensors["right_goal_frame"] = self.right_goal_frame
        self.scene.sensors["wrist_goal_frame"] = self.wrist_goal_frame

        right_goal_path_env0 = "/World/envs/env_0/Visuals/RightGoal/Geom"
        left_goal_path_env0  = "/World/envs/env_0/Visuals/LeftGoal/Geom"

        goal_rb_cfg = sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),  
        )
        goal_rb_cfg.func(right_goal_path_env0, goal_rb_cfg)
        goal_rb_cfg.func(left_goal_path_env0,  goal_rb_cfg)

        self._right_goal_rb_path = "/World/envs/env_.*/Visuals/RightGoal/Geom"
        self._left_goal_rb_path  = "/World/envs/env_.*/Visuals/LeftGoal/Geom"

        self.right_goal_rb = RigidObject(RigidObjectCfg(prim_path=self._right_goal_rb_path))
        self.left_goal_rb  = RigidObject(RigidObjectCfg(prim_path=self._left_goal_rb_path))

        self.scene.rigid_objects["right_goal"] = self.right_goal_rb
        self.scene.rigid_objects["left_goal"]  = self.left_goal_rb

    def _get_gt(self):
        gt = torch.cat(
            (   
                # xyz diffs (3,)
                self.right_glove_goal_distance,
                # rotation quaternion (4,)
                self.right_glove_goal_rotation,
                # xyz diffs (3,)
                self.left_glove_goal_distance,
                # rotation quaternion (4,)
                self.left_glove_goal_rotation,
                # xyz diffs (3,)
                # goal eucledean distance 
                self.goal_euclidean_distance.unsqueeze(1),
                # angular distances (1,)
                # self.right_glove_goal_angular_distance.unsqueeze(1),
                # euclidean distances (1,) [transform from (num_envs,) to (num_envs,1)]
                self.right_glove_goal_euclidean_distance.unsqueeze(1),
                # angular distances (1,)
                # self.left_glove_goal_angular_distance.unsqueeze(1),
                # euclidean distances (1,) [transform from (num_envs,) to (num_envs,1)]
                self.left_glove_goal_euclidean_distance.unsqueeze(1),
                # xyz diffs (3,)
                self.top_wrist_distance,
                # xyz diffs (3,)
                self.under_wrist_distance,
                # euclidean distances (1,) 
                self.top_wrist_euclidean_distance.unsqueeze(1),
                # euclidean distances (1,) 
                self.under_wrist_euclidean_distance.unsqueeze(1),
                # xyz diffs (3,)
                self.wrist_glove_distance,
                # # # euclidean distances (1,) 
                self.wrist_glove_euclidean_distance.unsqueeze(1),
                # xyz diffs (3, )
                # self.right_glove_object_distance,
                # # euclidean distance (1,)
                # self.right_glove_object_euclidean_distance.unsqueeze(1),
                # # xyz diffs (3, )
                # self.left_glove_object_distance,
                # # euclidean distance (1,)
                # self.left_glove_object_euclidean_distance.unsqueeze(1),

            ),
            dim=-1,
        )
        return gt
    
    def _get_rewards(self) -> torch.Tensor:
        (
            rewards,
            r_stretch,
            r_wrist_goal,
            r_right_object_goal,
            r_left_object_goal,
            r_right_glove_goal,
            r_left_glove_goal,
            r_object_goal,
            r_joint_vel,
            r_angular_right_glove_goal,
            r_angular_left_glove_goal,
            r_angular_right_object_goal,
            r_angular_left_object_goal,
            r_right_insert,
            r_left_insert,
            r_success_reward
        ) = compute_rewards(
            self.reaching_object_goal_scale,
            self.reaching_ee_object_scale,
            self.stretch_object_scale,
            self.episode_length_buf,
            self.object_goal_tracking_scale,
            self.joint_vel_penalty_scale,
            self.object_pos,
            self.right_glove_goal_euclidean_distance,
            self.left_glove_goal_euclidean_distance,
            self.right_glove_goal_angular_distance,
            self.left_glove_goal_angular_distance,
            self.joint_vel,
            self.cfg.minimal_distance,
            self.cfg.minimal_width,
            self.ee_euclidean_distance,
            self.ee_goal_euclidean_distance,
            self.right_ee_euclidean_distance,
            self.left_ee_euclidean_distance,
            self.right_ee_object_euclidean_distance,
            self.left_ee_object_euclidean_distance,
            self.right_ee_object_angular_distance,
            self.left_ee_object_angular_distance,
            self.rotation_ee_object_scale,
            self.rotation_object_goal_scale,
            self.right_insert_success,
            self.left_insert_success,
            self.wrist_glove_euclidean_distance,
            self.goal_wrist_pos,
            self.north_edge_pos,
            self.south_edge_pos,
        )

        # Keep logs aligned with what's returned/computed
        self.extras["log"] = {
            "r_stretch": r_stretch,
            "r_wrist_goal": r_wrist_goal,
            "reach_reward_right": r_right_glove_goal,
            "reach_reward_left": r_left_glove_goal,
            "object_goal_tracking": r_object_goal,
            "joint_vel_penalty": r_joint_vel,
            "reach_object_goal_right" : r_right_object_goal,
            "reach_object_goal_left" : r_left_object_goal,
            "r_angular_right_glove_goal": r_angular_right_glove_goal,
            "r_angular_left_glove_goal": r_angular_left_glove_goal,
            "r_angular_right_object_goal": r_angular_right_object_goal,
            "r_angular_left_object_goal": r_angular_left_object_goal,
            "r_right_insert": r_right_insert,
            "r_left_insert": r_left_insert,
            "r_success_reward": r_success_reward
        }

        if "tactile" in self.cfg.obs_list:
            self.extras["log"].update(
                {
                    "normalised_forces_left_x": self.normalised_forces[:, 0],
                    "normalised_forces_right_x": self.normalised_forces[:, 1],
                }
            )

        self.extras["counters"] = {}
        return rewards
    
    def _normalize_env_ids(self, env_ids):
        if isinstance(env_ids, int):
            return torch.tensor([env_ids], dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments to pregrasp phase."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        
        # Call parent reset
        super()._reset_idx(env_ids)
        
        # Reset pregrasp variables for specified environments
        self._policy_enabled[env_ids] = False
        self._pregrasp_steps[env_ids] = 0

    def _reset_target_pose(self, env_ids):
        # Make sure this is already on the right device once
        default_state = self.hand.data.default_root_state.clone()[env_ids]

        pos_noise = sample_uniform(-0.02, 0.02, (len(env_ids), 3), device=self.device)

        init_pos = default_state[0, 0:3].unsqueeze(0).repeat(len(env_ids), 1)
      
        default_state[:, 0:3] = (
            init_pos
            + pos_noise * self.cfg.reset_goal_position_noise
            + self.scene.env_origins[env_ids]
        )

        init_rot = default_state[0, 3:7].unsqueeze(0).repeat(len(env_ids), 1)

        default_state[:, 3:7] = init_rot

        default_state[:, 7:] = 0.0

        joint_pos = self.hand.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)

        self.hand.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.hand.write_root_state_to_sim(default_state, env_ids=env_ids)

    def _compute_intermediate_values(self, reset=False, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values()
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # print("number of nodes", self.object.data.nodal_pos_w.size())
        self.goal_north_pos = self.object.data.nodal_pos_w[:, self.anchor_idx["north"], :]
        self.goal_south_pos = self.object.data.nodal_pos_w[:, self.anchor_idx["south"], :]
        self.goal_east_pos = self.object.data.nodal_pos_w[:, self.anchor_idx["east"], :]
        self.goal_west_pos = self.object.data.nodal_pos_w[:, self.anchor_idx["west"], :]

        self.goal_cent_pos = (self.goal_north_pos+self.goal_south_pos+self.goal_east_pos+self.goal_west_pos)/4.0
        
        self.goal_wrist_pos[env_ids] = self.wrist_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.right_goal_pos[env_ids] = self.right_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.left_goal_pos[env_ids] = self.left_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.right_goal_rot[env_ids] = self.right_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        self.left_goal_rot[env_ids] = self.left_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        # goal_pos_world = self.goal_pos + self.scene.env_origins
        # self.goal_north_markers.visualize(self.goal_north_pos, self.goal_north_rot)
        # self.goal_south_markers.visualize(self.goal_south_pos, self.goal_south_rot)
        # self.goal_east_markers.visualize(self.goal_east_pos, self.goal_east_rot)
        # self.goal_west_markers.visualize(self.goal_west_pos, self.goal_west_rot)
        # self.goal_cent_markers.visualize(self.goal_cent_pos, self.goal_cent_rot)
        
        B = len(env_ids)
        dt_b = torch.full((B,), float(self.cfg.physics_dt), device=self.device)

        right_insert_out = self.insert_reward.step(
            pos_ee_s=self.right_ee_pos[env_ids],
            quat_ee_s=self.right_ee_rot[env_ids],
            pos_goal_s=self.right_goal_pos[env_ids],
            quat_goal_s=self.right_goal_rot[env_ids],
            dt=dt_b,
            idx=env_ids.to(self.device)
        )
        left_insert_out = self.insert_reward.step(
            pos_ee_s=self.left_ee_pos[env_ids],
            quat_ee_s=self.left_ee_rot[env_ids],
            pos_goal_s=self.left_goal_pos[env_ids],
            quat_goal_s=self.left_goal_rot[env_ids],
            dt=dt_b,
            idx=env_ids.to(self.device)
        )
        self.right_insert_success[env_ids] = right_insert_out["success"]       # (B,) bool
        self.right_insert_dwell[env_ids]   = right_insert_out["dwell"]         # (B,) float
        self.left_insert_success[env_ids] = left_insert_out["success"]       # (B,) bool
        self.left_insert_dwell[env_ids]   = left_insert_out["dwell"]         # (B,) float
        # print(f"dwell_right:{right_insert_out['dwell'][0]} dwell_left:{left_insert_out['dwell'][0]}")
        # print(f"state_right:{right_insert_out['state'][0]} state_left:{left_insert_out['state'][0]}")
        # print(f"d_right:{right_insert_out['d'][0:4]} r_right:{right_insert_out['r'][0:4]} c_right:{right_insert_out['c'][0:4]}")
        # print(f"d_left:{left_insert_out['d'][0:4]} r_left:{left_insert_out['r'][0:4]} c_left:{left_insert_out['c'][0:4]}")

        # upper/under distance
        self.wrist_glove_distance[env_ids] = self.right_gripper_r_pos[env_ids] - self.goal_wrist_pos[env_ids]
        self.wrist_glove_euclidean_distance[env_ids] = torch.norm(self.wrist_glove_distance[env_ids], dim=1)

        self.top_wrist_distance[env_ids] = self.north_edge_pos[env_ids] - self.goal_wrist_pos[env_ids]
        self.under_wrist_distance[env_ids] = self.goal_wrist_pos[env_ids] - self.south_edge_pos[env_ids]
        self.top_wrist_euclidean_distance[env_ids] = torch.norm(self.top_wrist_distance[env_ids], dim=1)
        self.under_wrist_euclidean_distance[env_ids] = torch.norm(self.under_wrist_distance[env_ids], dim=1)
        # print(f"east: {self.west_edge_pos[0]} right_goal_pos:{self.right_goal_pos[0]}")
        self.right_glove_goal_distance[env_ids] = self.east_edge_pos[env_ids] - self.right_goal_pos[env_ids]
        self.right_glove_goal_euclidean_distance[env_ids] = torch.norm(self.right_glove_goal_distance[env_ids], dim=1)
        self.right_glove_goal_rotation[env_ids] = quat_mul(self.right_gripper_r_rot[env_ids], quat_conjugate(self.right_goal_rot[env_ids]))
        self.right_glove_goal_angular_distance[env_ids] = rotation_distance(self.right_gripper_r_rot[env_ids], self.right_goal_rot[env_ids])
        # self.left_ee_goal_distance[env_ids] = self.left_l_ee_pos[env_ids] - self.left_goal_pos[env_ids]
        self.left_glove_goal_distance[env_ids] = self.left_gripper_r_pos[env_ids] - self.left_goal_pos[env_ids]
        self.left_glove_goal_euclidean_distance[env_ids] = torch.norm(self.left_glove_goal_distance[env_ids], dim=1)
        self.left_glove_goal_rotation[env_ids] = quat_mul(self.left_gripper_r_rot[env_ids], quat_conjugate(self.left_goal_rot[env_ids]))
        self.left_glove_goal_angular_distance[env_ids] = rotation_distance(self.left_gripper_r_rot[env_ids], self.left_goal_rot[env_ids])
        # print(self.left_glove_goal_euclidean_distance[0], self.right_glove_goal_euclidean_distance[0])

from tasks.airec.airec import distance_reward,distance_cond_reward, joint_vel_penalty, object_goal_reward, angular_distance_reward, insert_success_reward, success_reward, wrist_distance_reward

@torch.jit.script
def compute_rewards(
    reaching_object_goal_scale: float,
    reaching_glove_goal_scale: float,
    stretch_object_scale: float,
    episode_timestep_counter: torch.Tensor,
    object_goal_tracking_scale: float,
    joint_vel_penalty_scale: float,
    object_pos: torch.Tensor,
    right_glove_goal_euclidean_distance: torch.Tensor,
    left_glove_goal_euclidean_distance: torch.Tensor,
    right_glove_goal_angular_distance: torch.Tensor,
    left_glove_goal_angular_distance: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    minimal_distance: float,
    minimal_width: float,
    ee_euclidean_distance: torch.Tensor,
    ee_goal_euclidean_distance: torch.Tensor,
    right_ee_euclidean_distance: torch.Tensor,
    left_ee_euclidean_distance: torch.Tensor,
    right_ee_object_euclidean_distance: torch.Tensor,
    left_ee_object_euclidean_distance: torch.Tensor,
    right_ee_object_angular_distance: torch.Tensor,
    left_ee_object_angular_distance: torch.Tensor,
    rotation_ee_object_scale: float,
    rotation_object_goal_scale: float,
    right_insert_success: torch.Tensor,
    left_insert_success: torch.Tensor,
    wrist_ee_distance: torch.Tensor,
    wrist_pos: torch.Tensor,
    top_pos: torch.Tensor,
    under_pos: torch.Tensor,
):
    joint_vel_penalty_scale = 0
    rotation_ee_object_scale = 0.0 # 1.0
    rotation_object_goal_scale = 0.0 # 10.0
    reaching_ee_object_scale = 0.0
    reaching_object_goal_scale = 2.0
    object_goal_tracking_scale = 0.0
    insert_scale = 0.0
    stretch_object_scale = 1.0

    # reaching reward
    r_wrist_goal = wrist_distance_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, std=0.2) * reaching_object_goal_scale * 0.0
    # r_right_object_goal = distance_cond_reward(right_glove_goal_euclidean_distance, right_glove_goal_euclidean_distance, minimal_width,std=0.03) * reaching_object_goal_scale  
    # r_left_object_goal = distance_cond_reward(left_glove_goal_euclidean_distance, left_glove_goal_euclidean_distance, minimal_width, std=0.02) * reaching_object_goal_scale 
    r_right_glove_goal = distance_reward(right_glove_goal_euclidean_distance, std=0.09) * reaching_object_goal_scale  
    r_left_glove_goal = distance_reward(left_glove_goal_euclidean_distance, std=0.09) * reaching_object_goal_scale 
    
    r_right_ee_object = distance_reward(right_ee_object_euclidean_distance, std=0.2) * reaching_ee_object_scale  
    r_left_ee_object = distance_reward(left_ee_object_euclidean_distance, std=0.2) * reaching_ee_object_scale  
    # angular distance rewards
    r_angular_right_glove_goal = angular_distance_reward(right_glove_goal_angular_distance, std=0.4) * rotation_object_goal_scale
    r_angular_left_glove_goal = angular_distance_reward(left_glove_goal_angular_distance, std=0.4) * rotation_object_goal_scale
    r_angular_right_ee_object = angular_distance_reward(right_ee_object_angular_distance, std=0.2) * rotation_ee_object_scale 
    r_angular_left_ee_object = angular_distance_reward(left_ee_object_angular_distance, std=0.2) * rotation_ee_object_scale
    # joint velocity penalty
    r_joint_vel = joint_vel_penalty(robot_joint_vel) * joint_vel_penalty_scale
    r_stretch = distance_reward(ee_goal_euclidean_distance, std=0.01) * stretch_object_scale *0.0
    # insertion success reward
    r_right_insert = insert_success_reward(right_insert_success) * insert_scale
    r_left_insert = insert_success_reward(left_insert_success) * insert_scale

    # minillion bonus reward
    r_object_goal = object_goal_reward(right_glove_goal_euclidean_distance, r_right_insert, std=0.3) * object_goal_tracking_scale*0.0
    r_successed = success_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, minimal_distance)
    rewards = r_stretch + r_wrist_goal + r_right_glove_goal + r_left_glove_goal + r_right_ee_object + r_left_ee_object + r_object_goal + r_joint_vel + r_angular_right_glove_goal + r_angular_left_glove_goal + r_angular_right_ee_object + r_angular_left_ee_object + r_right_insert + r_left_insert + r_successed

    return (rewards, r_stretch, r_wrist_goal, r_right_glove_goal, r_left_glove_goal, r_right_ee_object, r_left_ee_object, r_object_goal, r_joint_vel, r_angular_right_glove_goal, r_angular_left_glove_goal, r_angular_right_ee_object, r_angular_left_ee_object, r_right_insert, r_left_insert, r_successed)

# scales an input between lower and upper
@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

