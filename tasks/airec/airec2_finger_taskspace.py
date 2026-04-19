# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AIREC2 **task-space** RL: dual-arm EE poses (robot root frame) → differential IK, plus optional hand joints.

Default policy width is **26D**: 14D dual-arm poses (same layout as before) plus **12D** thumb + first-finger
joints (right block then left, matching ``right_*`` EE then ``left_*`` EE). Those hand targets match the
nominal grasp pose in :mod:`assets.airec_finger` ``joint_pos`` defaults — use them to squeeze the glove and
limit slip. Joint-space control lives in :mod:`tasks.airec.airec2_finger`.

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

# Thumb + first finger per side (same order as ``AIREC_CFG.init_state.joint_pos`` defaults in ``assets/airec_finger``).
# Action layout: [:14] dual-arm pose, [14:20] right hand, [20:26] left hand.
TASKSPACE_HAND_JOINT_NAMES: tuple[str, ...] = (
    "right_hand_thumb_joint_1",
    "right_hand_thumb_joint_2",
    "right_hand_thumb_joint_3",
    "right_hand_thumb_joint_4",
    "right_hand_first_finger_joint_1",
    "right_hand_first_finger_joint_2",
    "left_hand_thumb_joint_1",
    "left_hand_thumb_joint_2",
    "left_hand_thumb_joint_3",
    "left_hand_thumb_joint_4",
    "left_hand_first_finger_joint_1",
    "left_hand_first_finger_joint_2",
)
NUM_TASKSPACE_HAND_ACTIONS: int = len(TASKSPACE_HAND_JOINT_NAMES)
NUM_TASKSPACE_POSE_ACTIONS: int = 14


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def init_task_space_ik_on_env(
    env: AIRECEnv, *, taskspace_hand_joint_names: tuple[str, ...] | None = None
) -> None:
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

    env.taskspace_hand_joint_ids = None
    if taskspace_hand_joint_names:
        jn = list(env.robot.joint_names)
        ids: list[int] = []
        missing: list[str] = []
        for name in taskspace_hand_joint_names:
            try:
                ids.append(jn.index(name))
            except ValueError:
                missing.append(name)
        if missing:
            raise RuntimeError(
                "Task-space hand joints not found on robot (check USD / URDF names). Missing: "
                f"{missing!r}; available (sample)={jn[:40]!r}…"
            )
        env.taskspace_hand_joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        env.joint_pos_cmd[:, ids] = env.robot.data.default_joint_pos[:, ids]
        env.prev_joint_pos_cmd[:, ids] = env.robot.data.default_joint_pos[:, ids]


def sync_taskspace_hand_joint_cmd_from_default(env: AIRECEnv, env_ids: torch.Tensor | slice | None) -> None:
    """After reset, align commanded hand joints with articulation defaults for the given env rows."""
    hid = getattr(env, "taskspace_hand_joint_ids", None)
    if hid is None or hid.numel() == 0:
        return
    if env_ids is None:
        d = env.robot.data.default_joint_pos
        env.joint_pos_cmd[:, hid] = d[:, hid]
        env.prev_joint_pos_cmd[:, hid] = d[:, hid]
        return
    if isinstance(env_ids, slice):
        d = env.robot.data.default_joint_pos[env_ids]
        env.joint_pos_cmd[env_ids, hid] = d[:, hid]
        env.prev_joint_pos_cmd[env_ids, hid] = d[:, hid]
        return
    ei = env_ids.long()
    d = env.robot.data.default_joint_pos[ei]
    env.joint_pos_cmd[ei][:, hid] = d[:, hid]
    env.prev_joint_pos_cmd[ei][:, hid] = d[:, hid]


def apply_task_space_action_on_env(env: AIRECEnv) -> None:
    """Map ``env.actions`` ``(N, 14)`` or ``(N, 26)`` to arm joint targets via IK; optional tail commands hands."""
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

    hid = getattr(env, "taskspace_hand_joint_ids", None)
    if hid is not None and hid.numel() > 0:
        expected = NUM_TASKSPACE_POSE_ACTIONS + hid.numel()
        if env.actions.shape[-1] != expected:
            raise RuntimeError(
                f"Task-space env expects actions[..., {expected}] (pose + hand); got {env.actions.shape[-1]}."
            )
        a_h = env.actions[:, NUM_TASKSPACE_POSE_ACTIONS : expected]
        lower = env.robot_dof_lower_limits[hid]
        upper = env.robot_dof_upper_limits[hid]
        # Centre on articulation defaults (same as ``AIREC_CFG.init_state.joint_pos`` in ``assets/airec_finger``):
        # ``a_h ∈ [-1, 1]`` moves each joint within :attr:`~tasks.airec.airec2_finger.AIRECEnvCfg.taskspace_hand_span_fraction` of the soft limit half-range.
        nom = env.robot.data.default_joint_pos[:, hid]
        half_range = 0.5 * (upper - lower)
        frac = float(env.cfg.taskspace_hand_span_fraction)
        scaled_h = nom + (a_h * frac) * half_range
        scaled_h = saturate(scaled_h, lower, upper)
        ma = env.cfg.act_moving_average
        env.joint_pos_cmd[:, hid] = ma * scaled_h + (1.0 - ma) * env.prev_joint_pos_cmd[:, hid]
        env.joint_pos_cmd[:, hid] = saturate(env.joint_pos_cmd[:, hid], lower, upper)
        env.prev_joint_pos_cmd[:, hid] = env.joint_pos_cmd[:, hid]

    if env._fixed_joint_indices:
        default_pos = env.robot.data.default_joint_pos
        for idx in env._fixed_joint_indices:
            env.joint_pos_cmd[:, idx] = default_pos[:, idx]
        zv = torch.zeros((env.num_envs, len(env._fixed_joint_indices)), device=env.device)
        env.robot.set_joint_velocity_target(zv, joint_ids=env._fixed_joint_indices)

    env.robot.set_joint_position_target(
        env.joint_pos_cmd[:, env.actuated_dof_indices], joint_ids=env.actuated_dof_indices
    )
    hand_ids = getattr(env, "taskspace_hand_joint_ids", None)
    if hand_ids is not None and hand_ids.numel() > 0:
        hl = hand_ids.tolist()
        env.robot.set_joint_position_target(env.joint_pos_cmd[:, hl], joint_ids=hl)


@configclass
class AIREC2FingerTaskSpaceEnvCfg(AIRECEnvCfg):
    """Same scene as :class:`~tasks.airec.airec2_finger.AIRECEnvCfg` but policy outputs 26D (14D poses + 12D hands)."""

    num_actions: int = NUM_TASKSPACE_POSE_ACTIONS + NUM_TASKSPACE_HAND_ACTIONS
    action_space: int = NUM_TASKSPACE_POSE_ACTIONS + NUM_TASKSPACE_HAND_ACTIONS
    #: Joints commanded in ``[-1, 1]`` → offset from :attr:`robot.data.default_joint_pos` (after ``act_moving_average``).
    taskspace_hand_joint_names: tuple[str, ...] = TASKSPACE_HAND_JOINT_NAMES
    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )


class AIREC2FingerTaskSpaceEnv(AIRECEnv):
    """Task-space control: overrides action width, last-action bookkeeping, and :meth:`_apply_action`."""

    cfg: AIREC2FingerTaskSpaceEnvCfg

    def __init__(self, cfg: AIREC2FingerTaskSpaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        init_task_space_ik_on_env(self, taskspace_hand_joint_names=cfg.taskspace_hand_joint_names)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        sync_taskspace_hand_joint_cmd_from_default(self, env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.last_action = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        apply_task_space_action_on_env(self)
