# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Wear glove + Shadow Hand + **6D EE velocity** (integrated) + diff IK + HEPi observations.

Train::

    python train.py --task AIREC_Wear_TaskSpace_HEPi --agent_cfg wear_hepi
    # If CUDA OOM: keep default ``num_envs=2`` or add e.g. ``--num_envs 1``; avoid PathTracing if possible.
"""

from __future__ import annotations

import torch

from isaaclab.assets import DeformableObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import saturate, subtract_frame_transforms

from tasks.airec.airec2_finger_taskspace import init_task_space_ik_on_env
from tasks.airec.mdp import observations as mdp_obs
from tasks.airec.wear_finger import WearEnv, WearEnvCfg
from tasks.airec.wear_finger_taskspace import WearFingerTaskSpaceEnvCfg


def apply_velocity_integrated_ik_on_wear_env(env: WearEnv) -> None:
    """Use integrated root-frame positions ``_ee_cmd_pos_{r,l}`` + current EE orientation → IK → joints."""
    root = env.robot.data.root_pose_w
    ee_r = env.robot.data.body_pose_w[:, env.right_arm_entity_cfg.body_ids[0]]
    ee_l = env.robot.data.body_pose_w[:, env.left_arm_entity_cfg.body_ids[0]]
    pos_br, quat_br = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee_r[:, 0:3], ee_r[:, 3:7])
    pos_bl, quat_bl = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee_l[:, 0:3], ee_l[:, 3:7])

    cmd_r = torch.cat([env._ee_cmd_pos_r, quat_br], dim=-1)
    cmd_l = torch.cat([env._ee_cmd_pos_l, quat_bl], dim=-1)
    env.ik_controller_right.set_command(cmd_r)
    env.ik_controller_left.set_command(cmd_l)

    jac_r = env.robot.root_physx_view.get_jacobians()[
        :, env.right_ee_jacobi_idx, :, env.right_arm_entity_cfg.joint_ids
    ]
    q_r = env.robot.data.joint_pos[:, env.right_arm_entity_cfg.joint_ids]
    jpr = env.ik_controller_right.compute(pos_br, quat_br, jac_r, q_r)

    jac_l = env.robot.root_physx_view.get_jacobians()[
        :, env.left_ee_jacobi_idx, :, env.left_arm_entity_cfg.joint_ids
    ]
    q_l = env.robot.data.joint_pos[:, env.left_arm_entity_cfg.joint_ids]
    jpl = env.ik_controller_left.compute(pos_bl, quat_bl, jac_l, q_l)

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
class WearFingerHepiTaskSpaceEnvCfg(WearFingerTaskSpaceEnvCfg):
    """6D velocity commands, HEPi observation channels, cloth-style ``obs_list``.

    Default ``num_envs=2``: deformable glove + Isaac rendering already consume most of a 32GB GPU; PyTorch
    (HEPi + PPO) needs free VRAM. Override with ``--num_envs`` if you have headroom.
    """

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=3, replicate_physics=False)

    #: Strided glove subsample for ``hepi_glove_*`` (rollout buffer + HEPi graph size).
    hepi_max_graph_nodes: int = 256

    num_actions: int = 6
    action_space: int = 6
    velocity_action_scale: float = 0.35
    obs_list: list = [
        "prop",
        "gt",
        "hepi_glove_pos",
        "hepi_glove_vel",
        "hepi_ee_pos",
        "hepi_ee_vel",
        "hepi_target_pos",
    ]


class WearFingerHepiTaskSpaceEnv(WearEnv):
    """Wear + task-space IK + velocity integration + HEPi geometry observations."""

    cfg: WearFingerHepiTaskSpaceEnvCfg  # type: ignore[assignment]

    def __init__(self, cfg: WearFingerHepiTaskSpaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._hepi_v = int(self.object.data.default_nodal_state_w.shape[1])
        n = int(cfg.num_actions)
        self.actions = torch.zeros((self.num_envs, n), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self._ee_cmd_pos_r = torch.zeros((self.num_envs, 3), device=self.device)
        self._ee_cmd_pos_l = torch.zeros((self.num_envs, 3), device=self.device)
        init_task_space_ik_on_env(self)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        root = self.robot.data.root_pose_w[env_ids]
        ee_r = self.robot.data.body_pose_w[env_ids, self.right_arm_entity_cfg.body_ids[0]]
        ee_l = self.robot.data.body_pose_w[env_ids, self.left_arm_entity_cfg.body_ids[0]]
        pos_br, _ = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee_r[:, 0:3], ee_r[:, 3:7])
        pos_bl, _ = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee_l[:, 0:3], ee_l[:, 3:7])
        self._ee_cmd_pos_r[env_ids] = pos_br
        self._ee_cmd_pos_l[env_ids] = pos_bl

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.last_action = self.actions.clone()
        self.actions.copy_(actions)
        dt = self.cfg.physics_dt * self.cfg.decimation
        a = torch.clamp(actions, -1.0, 1.0) * float(self.cfg.velocity_action_scale)
        p_min = torch.tensor(self.cfg.task_space_pos_min, device=self.device, dtype=torch.float32)
        p_max = torch.tensor(self.cfg.task_space_pos_max, device=self.device, dtype=torch.float32)
        self._ee_cmd_pos_r = self._ee_cmd_pos_r + a[:, :3] * dt
        self._ee_cmd_pos_l = self._ee_cmd_pos_l + a[:, 3:6] * dt
        self._ee_cmd_pos_r = torch.maximum(torch.minimum(self._ee_cmd_pos_r, p_max), p_min)
        self._ee_cmd_pos_l = torch.maximum(torch.minimum(self._ee_cmd_pos_l, p_max), p_min)

    def _apply_action(self) -> None:
        apply_velocity_integrated_ik_on_wear_env(self)

    def _compute_intermediate_values(self, reset=False, env_ids=None):
        # Fill arm-link pose for :class:`InsertReward` before :meth:`WearEnv._compute_intermediate_values` runs it.
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        rb = self.right_arm_entity_cfg.body_ids[0]
        lb = self.left_arm_entity_cfg.body_ids[0]
        self.right_ee_pos[env_ids] = self.robot.data.body_pos_w[env_ids, rb] - self.scene.env_origins[env_ids]
        self.left_ee_pos[env_ids] = self.robot.data.body_pos_w[env_ids, lb] - self.scene.env_origins[env_ids]
        self.right_ee_rot[env_ids] = self.robot.data.body_quat_w[env_ids, rb]
        self.left_ee_rot[env_ids] = self.robot.data.body_quat_w[env_ids, lb]
        super()._compute_intermediate_values()

    def _get_observations(self) -> dict:
        obs_dict = {}
        for k in self.cfg.obs_list:
            if k == "prop":
                obs_dict[k] = self._get_proprioception()
            elif k == "gt":
                obs_dict[k] = self._get_gt()
            elif k == "pixels":
                obs_dict[k] = self._get_images()
            elif k == "tactile":
                obs_dict[k] = self._get_tactile()
            elif k == "hepi_glove_pos":
                g = mdp_obs.glove_nodal_positions_w_minus_origin(self)
                g = mdp_obs.subsample_glove_nodes(g, int(self.cfg.hepi_max_graph_nodes))
                obs_dict[k] = mdp_obs.flatten_spatial(g)
            elif k == "hepi_glove_vel":
                gv = mdp_obs.glove_nodal_velocities_w(self)
                gv = mdp_obs.subsample_glove_nodes(gv, int(self.cfg.hepi_max_graph_nodes))
                obs_dict[k] = mdp_obs.flatten_spatial(gv)
            elif k == "hepi_ee_pos":
                ee = mdp_obs.ee_arm_link_positions_minus_origin(
                    self, ("right_arm_link_7", "left_arm_link_7")
                )
                obs_dict[k] = mdp_obs.flatten_spatial(ee)
            elif k == "hepi_ee_vel":
                rb = self.right_arm_entity_cfg.body_ids[0]
                lb = self.left_arm_entity_cfg.body_ids[0]
                v = torch.stack(
                    (
                        self.robot.data.body_lin_vel_w[:, rb],
                        self.robot.data.body_lin_vel_w[:, lb],
                    ),
                    dim=1,
                )
                obs_dict[k] = mdp_obs.flatten_spatial(v)
            elif k == "hepi_target_pos":
                t = mdp_obs.shadow_hand_goal_positions_minus_origin(self)
                obs_dict[k] = mdp_obs.flatten_spatial(t)
            else:
                print(f"[WearHepi] Unknown observation key: {k}")
        return {"policy": obs_dict}
