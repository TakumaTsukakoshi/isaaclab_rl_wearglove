# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AIREC2 **task-space** RL: 14D dual-arm EE poses (robot root frame) → differential IK.

Joint-space control lives only in :mod:`tasks.airec.airec2_finger`. This module subclasses
:class:`~tasks.airec.airec2_finger.AIRECEnv` and overrides action application so the policy never
commands joint deltas directly for the arms (IK maps poses to arm joints; other actuated DOFs follow
``joint_pos_cmd`` as updated by :func:`apply_task_space_action_on_env`).

Train::

    python train.py --task AIREC2_Finger_TaskSpace
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from types import SimpleNamespace

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import saturate, subtract_frame_transforms

from tasks.airec.airec2_finger import AIRECEnv, AIRECEnvCfg


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def init_task_space_ik_on_env(env: AIRECEnv) -> None:
    """Attach dual-arm :class:`~isaaclab.controllers.DifferentialIKController` instances (``right_arm_link_7`` / ``left_arm_link_7``).

    Uses ``robot.find_joints`` (same idea as ``teleop_joints_wearglove``) instead of
    :class:`~isaaclab.managers.SceneEntityCfg.resolve`, which can shrink the joint pool after
    earlier resolves and report only ``right_arm_joint_*`` even when the full articulation exists.
    """
    robot = env.robot
    r_ids = sorted(list(robot.find_joints("right_arm_joint_.*")[0]))
    l_ids = sorted(list(robot.find_joints("left_arm_joint_.*")[0]))
    if len(r_ids) != 7 or len(l_ids) != 7:
        jn = list(robot.joint_names)
        raise RuntimeError(
            f"Task-space IK needs 7 DOFs per arm; got right={len(r_ids)} left={len(l_ids)}. "
            "If left is 0, PhysX likely bound a right-arm-only subtree — check "
            "``articulation_root_prim_path`` / USD. joint_names="
            f"{jn!r}"
        )

    body_names = list(robot.data.body_names)
    try:
        r_body = body_names.index("right_arm_link_7")
        l_body = body_names.index("left_arm_link_7")
    except ValueError as e:
        raise RuntimeError(
            f"Missing EE link on robot.data.body_names (have {len(body_names)} bodies): {body_names!r}"
        ) from e

    env.ik_controller_right = DifferentialIKController(
        env.cfg.ik_controller_cfg, num_envs=env.num_envs, device=env.device
    )
    env.ik_controller_left = DifferentialIKController(
        env.cfg.ik_controller_cfg, num_envs=env.num_envs, device=env.device
    )
    env.right_arm_entity_cfg = SimpleNamespace(joint_ids=r_ids, body_ids=[r_body])
    env.left_arm_entity_cfg = SimpleNamespace(joint_ids=l_ids, body_ids=[l_body])
    if env.robot.is_fixed_base:
        env.right_ee_jacobi_idx = r_body - 1
        env.left_ee_jacobi_idx = l_body - 1
    else:
        env.right_ee_jacobi_idx = r_body
        env.left_ee_jacobi_idx = l_body


def apply_task_space_action_on_env(env: AIRECEnv) -> None:
    """Map ``env.actions`` ``(N, 14)`` to arm joint targets via IK; other actuated DOFs keep ``joint_pos_cmd``."""
    p_min = torch.tensor(env.cfg.task_space_pos_min, device=env.device, dtype=torch.float32)
    p_max = torch.tensor(env.cfg.task_space_pos_max, device=env.device, dtype=torch.float32)
    r_q = F.normalize(env.actions[:, 3:7], dim=-1, eps=1e-6)
    l_q = F.normalize(env.actions[:, 10:14], dim=-1, eps=1e-6)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
    r_q = torch.where(torch.norm(r_q, dim=-1, keepdim=True) < 1e-3, ident, r_q)
    l_q = torch.where(torch.norm(l_q, dim=-1, keepdim=True) < 1e-3, ident, l_q)
    r_pos = scale(env.actions[:, 0:3], p_min, p_max)
    l_pos = scale(env.actions[:, 7:10], p_min, p_max)
    cmd_r = torch.cat([r_pos, r_q], dim=-1)
    cmd_l = torch.cat([l_pos, l_q], dim=-1)
    env.ik_controller_right.set_command(cmd_r)
    env.ik_controller_left.set_command(cmd_l)

    root = env.robot.data.root_pose_w
    jac_r = env.robot.root_physx_view.get_jacobians()[
        :, env.right_ee_jacobi_idx, :, env.right_arm_entity_cfg.joint_ids
    ]
    ee_r = env.robot.data.body_pose_w[:, env.right_arm_entity_cfg.body_ids[0]]
    q_r = env.robot.data.joint_pos[:, env.right_arm_entity_cfg.joint_ids]
    pos_b, quat_b = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee_r[:, 0:3], ee_r[:, 3:7])
    jpr = env.ik_controller_right.compute(pos_b, quat_b, jac_r, q_r)

    jac_l = env.robot.root_physx_view.get_jacobians()[
        :, env.left_ee_jacobi_idx, :, env.left_arm_entity_cfg.joint_ids
    ]
    ee_l = env.robot.data.body_pose_w[:, env.left_arm_entity_cfg.body_ids[0]]
    q_l = env.robot.data.joint_pos[:, env.left_arm_entity_cfg.joint_ids]
    pos_b, quat_b = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee_l[:, 0:3], ee_l[:, 3:7])
    jpl = env.ik_controller_left.compute(pos_b, quat_b, jac_l, q_l)

    r_ids = env.right_arm_entity_cfg.joint_ids
    l_ids = env.left_arm_entity_cfg.joint_ids
    ma = env.cfg.act_moving_average
    env.joint_pos_cmd[:, r_ids] = ma * jpr + (1.0 - ma) * env.joint_pos_cmd[:, r_ids]
    env.joint_pos_cmd[:, l_ids] = ma * jpl + (1.0 - ma) * env.joint_pos_cmd[:, l_ids]
    env.joint_pos_cmd[:, r_ids] = saturate(
        env.joint_pos_cmd[:, r_ids],
        env.robot_dof_lower_limits[r_ids],
        env.robot_dof_upper_limits[r_ids],
    )
    env.joint_pos_cmd[:, l_ids] = saturate(
        env.joint_pos_cmd[:, l_ids],
        env.robot_dof_lower_limits[l_ids],
        env.robot_dof_upper_limits[l_ids],
    )
    env.prev_joint_pos_cmd[:, env.actuated_dof_indices] = env.joint_pos_cmd[:, env.actuated_dof_indices]

    if env._fixed_joint_indices:
        default_pos = env.robot.data.default_joint_pos
        for idx in env._fixed_joint_indices:
            env.joint_pos_cmd[:, idx] = default_pos[:, idx]
        zv = torch.zeros((env.num_envs, len(env._fixed_joint_indices)), device=env.device)
        env.robot.set_joint_velocity_target(zv, joint_ids=env._fixed_joint_indices)

    env.robot.set_joint_position_target(
        env.joint_pos_cmd[:, env.actuated_dof_indices], joint_ids=env.actuated_dof_indices
    )


@configclass
class AIREC2FingerTaskSpaceEnvCfg(AIRECEnvCfg):
    """Same scene as :class:`~tasks.airec.airec2_finger.AIRECEnvCfg` but policy outputs 14D dual-arm poses (root frame)."""

    num_actions: int = 14
    action_space: int = 14
    task_space_pos_min: tuple[float, float, float] = (-0.55, -0.55, 0.35)
    task_space_pos_max: tuple[float, float, float] = (0.55, 0.55, 1.45)
    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )


class AIREC2FingerTaskSpaceEnv(AIRECEnv):
    """Task-space control: overrides action width, last-action bookkeeping, and :meth:`_apply_action`."""

    cfg: AIREC2FingerTaskSpaceEnvCfg

    @property
    def policy_action_dim(self) -> int:
        return int(self.cfg.num_actions)

    def __init__(self, cfg: AIREC2FingerTaskSpaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        init_task_space_ik_on_env(self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.last_action = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        apply_task_space_action_on_env(self)
