# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Task-Space AIREC Environment using Differential IK Controller

Author: Modified from Elle Miller 2025
Reference: run_diff_ik_airec_both_arms.py

This environment uses end-effector control with differential inverse kinematics
following the Isaac Lab DifferentialIKController patterns.

Usage:
    python train.py --task AIREC_Wear_TaskSpace
"""

from __future__ import annotations
from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# Import base AIREC environment
from airec2_finger import AIRECEnvCfg, AIRECEnv, scale, saturate
from assets.airec_finger import AIREC_CFG
from assets.shadow_hand import SHADOW_HAND_CFG


@configclass
class AIRECTaskSpaceEnvCfg(AIRECEnvCfg):
    """Configuration for task-space (IK-based) AIREC environment.
    
    Based on reference: run_diff_ik_airec_both_arms.py
    """
    
    # Override action space - now 12D for two 6D end-effector poses (absolute poses, not velocities)
    num_actions = 14  # 7D pose for each arm (position 3D + quaternion 4D) = 14D total
    action_space = num_actions
    
    # Task-space controller configuration - following Isaac Lab patterns
    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",      # Use "pose" for absolute end-effector poses
        use_relative_mode=False,  # Use absolute poses (not delta)
        ik_method="dls",          # Damped Least Squares method (most stable)
    )
    
    # IK Solver tuning (from reference: default 0.01 is good, lower for better tracking)
    ik_lambda = 0.01            # Damping factor - critical for stability
                                # Lower (0.001-0.005): Better tracking, less stable
                                # Default (0.01): Balanced
                                # Higher (0.05+): More stable, less precise
    
    # Motion control
    act_moving_average = 0.001  # Smoothing (increase to 0.01-0.02 if jerky)


class AIRECTaskSpaceEnv(AIRECEnv):
    """Task-space control using differential IK.
    
    Based on reference implementation: run_diff_ik_airec_both_arms.py
    
    This follows Isaac Lab patterns for proper Jacobian integration.
    """
    
    cfg: AIRECTaskSpaceEnvCfg
    
    def __init__(self, cfg: AIRECTaskSpaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_task_space_controller()
    
    def _init_task_space_controller(self):
        """Initialize differential IK controllers for both arms following reference patterns."""
        print("[INFO] Initializing Task-Space Differential IK Controllers (dual-arm)...")
        
        # Create DifferentialIK controllers for right and left arms
        self.ik_controller_right = DifferentialIKController(
            self.cfg.ik_controller_cfg,
            num_envs=self.num_envs,
            device=self.device
        )
        self.ik_controller_left = DifferentialIKController(
            self.cfg.ik_controller_cfg,
            num_envs=self.num_envs,
            device=self.device
        )
        
        # Define arm-specific entity configurations (following run_diff_ik_airec_both_arms.py pattern)
        self.right_arm_entity_cfg = SceneEntityCfg(
            "robot", 
            joint_names=["right_arm_joint_[1-7]"],
            body_names=["right_hand_wrist"]  # Or adjust to your EE link name
        )
        
        self.left_arm_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["left_arm_joint_[1-7]"],
            body_names=["left_hand_wrist"]  # Or adjust to your EE link name
        )
        
        # Resolve entity references
        self.right_arm_entity_cfg.resolve(self.scene)
        self.left_arm_entity_cfg.resolve(self.scene)
        
        # Get body IDs for end-effector jacobian computation
        right_ee_body_ids = [int(self.right_arm_entity_cfg.body_ids[0])]
        left_ee_body_ids = [int(self.left_arm_entity_cfg.body_ids[0])]
        
        # Compute jacobian indices (for fixed base: jacobian_idx = body_id - 1)
        if self.robot.is_fixed_base:
            self.right_ee_jacobi_idx = right_ee_body_ids[0] - 1
            self.left_ee_jacobi_idx = left_ee_body_ids[0] - 1
        else:
            self.right_ee_jacobi_idx = right_ee_body_ids[0]
            self.left_ee_jacobi_idx = left_ee_body_ids[0]
        
        # Buffers for joint commands (following reference pattern)
        self.joint_pos_des_right = torch.zeros(
            (self.num_envs, len(self.cfg.actuated_rarm_joints)),
            device=self.device
        )
        self.joint_pos_des_left = torch.zeros(
            (self.num_envs, len(self.cfg.actuated_larm_joints)),
            device=self.device
        )
        
        print("[INFO] Task-space controllers initialized successfully!")
        print(f"  Right arm EE body ID: {right_ee_body_ids[0]}, Jacobian index: {self.right_ee_jacobi_idx}")
        print(f"  Left arm EE body ID: {left_ee_body_ids[0]}, Jacobian index: {self.left_ee_jacobi_idx}")
    
    def _apply_action(self) -> None:
        """Apply actions using differential IK controllers.
        
        Actions: 14D = 7D (right arm pose) + 7D (left arm pose)
        Each 7D = 3D position + 4D quaternion (wxyz)
        
        Following reference: run_diff_ik_airec_both_arms.py
        """
        # Parse action: 7D pose for each arm
        right_arm_goal = self.actions[:, 0:7]   # 3D pos + 4D quat
        left_arm_goal = self.actions[:, 7:14]   # 3D pos + 4D quat
        
        # Get robot state in world frame
        root_pose_w = self.robot.data.root_pose_w
        
        # ===== RIGHT ARM IK COMPUTATION =====
        # Get jacobian matrix for right arm (following reference pattern)
        jacobian_right = self.robot.root_physx_view.get_jacobians()[
            :, self.right_ee_jacobi_idx, :, self.right_arm_entity_cfg.joint_ids
        ]
        
        # Get current end-effector pose in world frame
        ee_pose_w_right = self.robot.data.body_pose_w[:, self.right_arm_entity_cfg.body_ids[0]]
        
        # Get current joint positions
        joint_pos_right = self.robot.data.joint_pos[:, self.right_arm_entity_cfg.joint_ids]
        
        # Transform to robot root frame (subtract_frame_transforms)
        ee_pos_b_right, ee_quat_b_right = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7]
        )
        
        # Compute IK - returns desired joint positions
        self.joint_pos_des_right = self.ik_controller_right.compute(
            right_arm_goal[:, 0:3],  # Position command
            right_arm_goal[:, 3:7],  # Quaternion command
            jacobian_right,          # Jacobian
            joint_pos_right          # Current joint positions
        )
        
        # ===== LEFT ARM IK COMPUTATION =====
        jacobian_left = self.robot.root_physx_view.get_jacobians()[
            :, self.left_ee_jacobi_idx, :, self.left_arm_entity_cfg.joint_ids
        ]
        
        ee_pose_w_left = self.robot.data.body_pose_w[:, self.left_arm_entity_cfg.body_ids[0]]
        joint_pos_left = self.robot.data.joint_pos[:, self.left_arm_entity_cfg.joint_ids]
        
        ee_pos_b_left, ee_quat_b_left = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7]
        )
        
        self.joint_pos_des_left = self.ik_controller_left.compute(
            left_arm_goal[:, 0:3],   # Position command
            left_arm_goal[:, 3:7],   # Quaternion command
            jacobian_left,           # Jacobian
            joint_pos_left           # Current joint positions
        )
        
        # ===== APPLY JOINT TARGETS =====
        self._apply_joint_targets_final()
    
    def _apply_joint_targets_final(self):
        """Apply computed joint targets with smoothing and saturation.
        
        Follows the original _apply_action pattern with moving average.
        """
        # Apply moving average smoothing for stability
        self.joint_pos_cmd[:, self.right_arm_entity_cfg.joint_ids] = (
            self.cfg.act_moving_average * self.joint_pos_des_right +
            (1.0 - self.cfg.act_moving_average) * 
            self.joint_pos_cmd[:, self.right_arm_entity_cfg.joint_ids]
        )
        
        self.joint_pos_cmd[:, self.left_arm_entity_cfg.joint_ids] = (
            self.cfg.act_moving_average * self.joint_pos_des_left +
            (1.0 - self.cfg.act_moving_average) *
            self.joint_pos_cmd[:, self.left_arm_entity_cfg.joint_ids]
        )
        
        # Saturate to joint limits
        self.joint_pos_cmd[:, self.right_arm_entity_cfg.joint_ids] = saturate(
            self.joint_pos_cmd[:, self.right_arm_entity_cfg.joint_ids],
            self.robot_dof_lower_limits[:, self.right_arm_entity_cfg.joint_ids],
            self.robot_dof_upper_limits[:, self.right_arm_entity_cfg.joint_ids]
        )
        
        self.joint_pos_cmd[:, self.left_arm_entity_cfg.joint_ids] = saturate(
            self.joint_pos_cmd[:, self.left_arm_entity_cfg.joint_ids],
            self.robot_dof_lower_limits[:, self.left_arm_entity_cfg.joint_ids],
            self.robot_dof_upper_limits[:, self.left_arm_entity_cfg.joint_ids]
        )
        
        # Apply joint position targets
        self.robot.set_joint_position_target(
            self.joint_pos_cmd[:, self.right_arm_entity_cfg.joint_ids],
            joint_ids=self.right_arm_entity_cfg.joint_ids
        )
        self.robot.set_joint_position_target(
            self.joint_pos_cmd[:, self.left_arm_entity_cfg.joint_ids],
            joint_ids=self.left_arm_entity_cfg.joint_ids
        )
        
        # Handle fixed joints (same as parent implementation)
        if self._fixed_joint_indices:
            default_pos = self.robot.data.default_joint_pos
            for idx in self._fixed_joint_indices:
                self.joint_pos_cmd[:, idx] = default_pos[:, idx]
            zero_vel_fixed = torch.zeros(
                (self.num_envs, len(self._fixed_joint_indices)),
                device=self.device
            )
            self.robot.set_joint_velocity_target(
                zero_vel_fixed,
                joint_ids=self._fixed_joint_indices
            )
