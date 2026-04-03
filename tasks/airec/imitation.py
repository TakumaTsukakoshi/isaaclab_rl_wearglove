# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AIREC in empty space (ground plane only): no bed, object, or goal.

Uses :mod:`tasks.airec.airec2_finger` (AIREC2 + Shadow Hand articulation in scene) with ``object_type="none"`` so no **deformable glove** is spawned. For **wearing the glove toward Shadow Hand** (glove + goals on Shadow Hand links), use :class:`tasks.airec.wear_finger.WearEnv` / :class:`tasks.airec.teleop_wearglove.TeleopWearGloveEnv` and ``teleop_joints_wearglove.py`` (or ``teleop_joints.py --task wearglove``).

Base RL rewards are zero unless :attr:`ImitationEnvCfg.imitation_demo_path` is set, in which case the
reward is joint-space imitation. For teleop, use :class:`TeleopImitationEnv` and
``teleop_joints.py --task imitation``.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
import isaaclab.sim as sim_utils

from tasks.airec.airec2_finger import AIRECEnv, AIRECEnvCfg, distance_reward
from tasks.airec.teleop_demonstrations import JOINT_COMMANDS_KEY, load_teleop_demo
from assets.airec import (
    ACTUATED_TORSO_JOINTS,
    ACTUATED_LARM_JOINTS,
    ACTUATED_RARM_JOINTS,
    ACTUATED_LHAND_JOINTS,
    ACTUATED_RHAND_JOINTS,
    ACTUATED_BASE_JOINTS,
    IMITATION_DEFAULT_JOINT_NAMES,
)


@configclass
class ImitationEnvCfg(AIRECEnvCfg):
    """Robot + ground / lighting / sensors only (AIREC2 stack, no scene object)."""

    #: No scene object — freespace / teleop imitation. For a **deformable glove**, use
    #: :class:`tasks.airec.teleop_wearglove.TeleopWearGloveEnv` / ``--task wearglove`` (see :class:`WearEnvCfg`).
    #: Ignored when :attr:`imitation_demo_path` is set (length matches demo).
    episode_length_s = 10.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)

    actuated_body_joint_names =  ACTUATED_LARM_JOINTS + ACTUATED_RARM_JOINTS 
    enable_base_control = False
    if enable_base_control:
        actuated_joint_names = actuated_body_joint_names + list(ACTUATED_BASE_JOINTS)
    else:
        actuated_joint_names = actuated_body_joint_names
    num_actions = len(actuated_joint_names)

    imitation_demo_path: str | None = None
    imitation_joint_indices: tuple[int, ...] | None = None
    imitation_gt_future_steps: int = 1

    #: If set during play, this env index follows the recorded demo open-loop; others use the policy.
    replay_demo_env_id: int | None = None


class ImitationEnv(AIRECEnv):
    """Freespace AIREC2: no deformable task object (``object_type="none"``)."""

    cfg: ImitationEnvCfg

    def __init__(self, cfg: ImitationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_joints = self.robot.num_joints
        jn = self.robot.joint_names
        self.actuated_body_dof_indices = sorted(jn.index(n) for n in self.cfg.actuated_body_joint_names)
        self.actuated_base_dof_indices = [jn.index(n) for n in ACTUATED_BASE_JOINTS if n in jn]
        self.torso_dof_indices = sorted(jn.index(n) for n in ACTUATED_TORSO_JOINTS if n in jn)
        self.larm_dof_indices = sorted(jn.index(n) for n in ACTUATED_LARM_JOINTS if n in jn)
        self.rarm_dof_indices = sorted(jn.index(n) for n in ACTUATED_RARM_JOINTS if n in jn)
        self.hand_dof_indices = sorted(
            jn.index(n) for n in ACTUATED_LHAND_JOINTS + ACTUATED_RHAND_JOINTS if n in jn
        )
        self.base_dof_indices = list(self.actuated_base_dof_indices)

        self.base_link_body_idx = self.robot.data.body_names.index("base_link")
        self.com_pos_w = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)
        self.com_pos_b = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)

        self._imitation_ref_joint_pos: torch.Tensor | None = None
        self._imitation_ref_joint_cmd: torch.Tensor | None = None
        self._imitation_ref_base_vel: torch.Tensor | None = None
        self._imitation_joint_idx_t: torch.Tensor | None = None
        self._imitation_ref_object_pos_local: torch.Tensor | None = None
        self._imitation_ref_object_quat_w: torch.Tensor | None = None

        if cfg.imitation_demo_path:
            self._load_imitation_demo_from_npz()
            with np.load(cfg.imitation_demo_path, allow_pickle=False) as z:
                num_demo_steps = int(z[JOINT_COMMANDS_KEY].shape[0])
            step_dt = float(cfg.sim.dt * cfg.decimation)
            cfg.episode_length_s = num_demo_steps * step_dt
            print("***********************************************")
            print(f"Imitation demo length: {num_demo_steps} steps, {step_dt:.3f} s/step")
            print("***********************************************")

        self._body_dof_t = torch.tensor(self.actuated_body_dof_indices, device=self.device, dtype=torch.long)
        self._base_dof_t = (
            torch.tensor(self.actuated_base_dof_indices, device=self.device, dtype=torch.long)
            if self.actuated_base_dof_indices
            else None
        )
        self.diff = torch.zeros_like(self.joint_pos_cmd)
        self.mse = torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)

    def _load_imitation_demo_from_npz(self) -> None:
        demo = load_teleop_demo(self.cfg.imitation_demo_path)
        ref = torch.as_tensor(demo.measured_joint_pos, dtype=self.dtype, device=self.device)
        if ref.shape[1] != self.num_joints:
            raise ValueError(
                f"imitation_demo_path: measured_joint_pos has {ref.shape[1]} joints, robot has {self.num_joints}"
            )
        self._imitation_ref_joint_pos = ref
        ref_cmd = torch.as_tensor(demo.joint_commands, dtype=self.dtype, device=self.device)
        if ref_cmd.shape != ref.shape:
            raise ValueError(
                f"imitation_demo_path: joint_commands shape {ref_cmd.shape} must match measured_joint_pos {ref.shape}"
            )
        self._imitation_ref_joint_cmd = ref_cmd
        if self.actuated_base_dof_indices:
            ref_bv_np = demo.joint_commands[:, self.actuated_base_dof_indices]
            self._imitation_ref_base_vel = torch.as_tensor(ref_bv_np, dtype=self.dtype, device=self.device)
        else:
            self._imitation_ref_base_vel = None
        if demo.object_pos_local is not None and demo.object_quat_w is not None:
            pos_t = torch.as_tensor(demo.object_pos_local, dtype=self.dtype, device=self.device)
            quat_t = torch.as_tensor(demo.object_quat_w, dtype=self.dtype, device=self.device)
            self._imitation_ref_object_pos_local = pos_t
            self._imitation_ref_object_quat_w = quat_t
            if pos_t.shape[0] != ref.shape[0]:
                raise ValueError(
                    "imitation_demo_path: object_pos_local length must match measured_joint_pos time dimension"
                )
            if quat_t.shape[0] != ref.shape[0]:
                raise ValueError(
                    "imitation_demo_path: object_quat_w length must match measured_joint_pos time dimension"
                )
        print(
            f"Loaded imitation demo: {self.cfg.imitation_demo_path} "
            f"({ref.shape[0]} steps; measured_joint_pos + joint_commands references"
            f"{'; object local pos + quat' if self._imitation_ref_object_pos_local is not None else ''})"
        )
        if self.cfg.imitation_joint_indices is not None:
            self._imitation_joint_idx_t = torch.tensor(
                self.cfg.imitation_joint_indices, device=self.device, dtype=torch.long
            )
        else:
            missing = [n for n in IMITATION_DEFAULT_JOINT_NAMES if n not in self.robot.joint_names]
            if missing:
                raise ValueError(f"imitation default joint names not found on robot: {missing}")
            idxs = [self.robot.joint_names.index(n) for n in IMITATION_DEFAULT_JOINT_NAMES]
            self._imitation_joint_idx_t = torch.tensor(idxs, device=self.device, dtype=torch.long)
            print(f"Imitation MSE: {len(idxs)} joints (torso + arms; head/hands excluded)")

    def _get_imitation_reference_joint_pos(self) -> torch.Tensor:
        ref = self._imitation_ref_joint_pos
        if ref is None:
            raise RuntimeError("Imitation reference requested but imitation_demo_path is not set")
        T = ref.shape[0]
        idx = torch.clamp(self.episode_length_buf, max=T - 1)
        return ref[idx.long()]

    def _get_imitation_reference_joint_pos_horizon(self, horizon: int) -> torch.Tensor:
        ref = self._imitation_ref_joint_pos
        if ref is None:
            raise RuntimeError("Imitation reference requested but imitation_demo_path is not set")
        T = ref.shape[0]
        h = max(int(horizon), 1)
        t0 = torch.clamp(self.episode_length_buf.long(), max=T - 1)
        offsets = torch.arange(h, device=self.device, dtype=torch.long)
        time_idx = torch.clamp(t0.unsqueeze(-1) + offsets, max=T - 1)
        return ref[time_idx]

    def _get_imitation_reference_joint_cmd(self) -> torch.Tensor:
        ref = self._imitation_ref_joint_cmd
        if ref is None:
            raise RuntimeError("Imitation joint_commands reference requested but imitation_demo_path is not set")
        T = ref.shape[0]
        idx = torch.clamp(self.episode_length_buf, max=T - 1)
        return ref[idx.long()]

    def _get_imitation_reference_joint_cmd_horizon(self, horizon: int) -> torch.Tensor:
        ref = self._imitation_ref_joint_cmd
        if ref is None:
            raise RuntimeError("Imitation joint_commands reference requested but imitation_demo_path is not set")
        T = ref.shape[0]
        h = max(int(horizon), 1)
        t0 = torch.clamp(self.episode_length_buf.long(), max=T - 1)
        offsets = torch.arange(h, device=self.device, dtype=torch.long)
        time_idx = torch.clamp(t0.unsqueeze(-1) + offsets, max=T - 1)
        return ref[time_idx]

    def _imitation_update_com(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        body_com_pos = self.robot.data.body_com_pose_w[..., :3]
        mass = self.robot.data.default_mass.to(device=self.device)
        total_mass = mass.sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.com_pos_w[env_ids] = (
            body_com_pos[env_ids] * mass[env_ids].unsqueeze(-1)
        ).sum(dim=1) / total_mass[env_ids]
        base_link_pos_w = self.robot.data.body_link_pose_w[:, self.base_link_body_idx, :3]
        base_quat = self.robot.data.body_link_pose_w[:, self.base_link_body_idx, 3:7]
        com_to_base_w = self.com_pos_w[env_ids] - base_link_pos_w[env_ids]
        self.com_pos_b[env_ids] = quat_apply_inverse(base_quat[env_ids], com_to_base_w)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        eid = self.cfg.replay_demo_env_id
        if eid is None:
            return
        e = int(eid)
        if e < 0 or e >= self.num_envs:
            return
        ref_cmd = self._imitation_ref_joint_cmd
        if ref_cmd is None:
            return
        t = int(torch.clamp(self.episode_length_buf[e], max=ref_cmd.shape[0] - 1).item())
        row = ref_cmd[t]
        self.joint_pos_cmd[e, :] = row
        if self.cfg.enable_base_control and self._base_dof_t is not None:
            low = -self.robot_hard_vel_limits[self._base_dof_t]
            high = self.robot_hard_vel_limits[self._base_dof_t]
            self.joint_pos_cmd[e, self._base_dof_t] = torch.clamp(
                self.joint_pos_cmd[e, self._base_dof_t], low, high
            )

    def _setup_scene(self):
        super()._setup_scene()
        print("***********************************************")
        print("Setting up scene for imitation (AIREC2, object_type=none)...")
        print("***********************************************")
        light = sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=250.0,
            texture_format="latlong",
        )
        light.func("/World/bglight", light)

    def _reset_env(self, env_ids) -> None:
        pass

    def _get_proprioception(self):
        prop = torch.cat(
            (
                self.normalised_joint_pos,
                self.normalised_joint_vel,
                self.joint_pos_cmd[:, self.actuated_dof_indices],
                self.com_pos_b,
                self.diff[:, self.actuated_dof_indices],
            ),
            dim=-1,
        )
        return prop

    def _get_gt(self) -> torch.Tensor:
        n = max(int(getattr(self.cfg, "imitation_gt_future_steps", 1)), 1)
        flat_dim = n * self.num_joints
        if not self.cfg.imitation_demo_path:
            return torch.zeros((self.num_envs, flat_dim), device=self.device, dtype=self.dtype)
        window = self._get_imitation_reference_joint_cmd_horizon(n)
        return window.reshape(self.num_envs, flat_dim)

    def _get_rewards(self) -> torch.Tensor:
        if not self.cfg.imitation_demo_path:
            return torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)
        reward = distance_reward(self.mse, std=0.3)

        def _part_mse(ix: list[int]) -> torch.Tensor:
            if not ix:
                return torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)
            return (self.diff[:, ix] * self.diff[:, ix]).mean(dim=-1)

        self.extras["log"].update({
            "Imitation / imitation_mse": self.mse,
            "Imitation / imitation_reward": reward,
            "Imitation / torso mse": _part_mse(self.torso_dof_indices),
            "Imitation / arm mse": _part_mse(self.larm_dof_indices),
            "Imitation / rarm mse": _part_mse(self.rarm_dof_indices),
            "Imitation / hand mse": _part_mse(self.hand_dof_indices),
            "Imitation / base mse": _part_mse(self.base_dof_indices),
        })

        return reward

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values(env_ids)
        self._imitation_update_com(env_ids)
        if not self.cfg.imitation_demo_path:
            self.diff = torch.zeros_like(self.joint_pos_cmd)
            self.mse = torch.zeros((self.num_envs,), device=self.device, dtype=self.dtype)
            return
        ref_cmd = self._get_imitation_reference_joint_cmd()
        self.diff = self.joint_pos_cmd - ref_cmd
        sq = self.diff * self.diff
        self.mse = sq.mean(dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        com_tip_termination = self.com_pos_b[:, 0] < 0
        too_far_from_ref = self.mse > 0.5
        termination = com_tip_termination | too_far_from_ref
        return termination, time_out
