# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared joint-space teleoperation for any :class:`tasks.airec.airec.AIRECEnv` subclass.

Use multiple inheritance: ``class TeleopFooEnv(TeleopJointSpaceMixin, FooEnv): ...``

Policy actions are ignored; drive :attr:`teleop_base_vel`, :attr:`teleop_arm_joint_target`, and
:attr:`teleop_torso12_target` before each :meth:`step`, or call :meth:`set_joint_pos_replay` with saved
``joint_commands`` (``joint_pos_cmd`` before each recorded step: base = velocity, body = position);
see :mod:`tasks.airec.teleop_demonstrations`.
For block demos with ``object_pos_local`` / ``object_quat_w``, call :meth:`set_object_pose_replay` after
``set_joint_pos_replay`` and :meth:`write_object_replay_to_sim` after each env step (sphere pose follows
the same replay index and hold as ``joint_commands``).
Call :meth:`set_measured_joint_replay` with ``measured_joint_pos`` and :meth:`write_measured_robot_replay_to_sim`
after each step so the **simulated** robot matches recorded poses (especially base translation/yaw), because
``joint_commands`` stores base **velocities** which are often zero in sparse pose captures.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from assets.airec import ACTUATED_LHAND_JOINTS, ACTUATED_RHAND_JOINTS


class TeleopJointSpaceMixin:
    """Inject joint-space teleop command buffers + replay; must be listed *before* the task env base in MRO."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        robot = self.robot
        la = list(robot.find_joints("left_arm_joint_.*")[0])
        ra = list(robot.find_joints("right_arm_joint_.*")[0])
        self.teleop_arm_joint_ids = sorted(la + ra)
        n_arm = len(self.teleop_arm_joint_ids)
        self._ik_joint_ids_t = torch.tensor(self.teleop_arm_joint_ids, device=self.device, dtype=torch.long)

        t1 = robot.joint_names.index("torso_joint_1")
        t2 = robot.joint_names.index("torso_joint_2")
        t3 = robot.joint_names.index("torso_joint_3")
        self._torso12_joint_ids_t = torch.tensor([t1, t2], device=self.device, dtype=torch.long)
        lhi = [robot.joint_names.index(n) for n in ACTUATED_LHAND_JOINTS]
        rhi = [robot.joint_names.index(n) for n in ACTUATED_RHAND_JOINTS]
        self._locked_body_dof_indices = sorted([t3] + lhi + rhi)

        self.teleop_base_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.robot.data.default_joint_pos.dtype)
        self.teleop_arm_joint_target = torch.zeros((self.num_envs, n_arm), device=self.device, dtype=self.robot.data.default_joint_pos.dtype)
        self.teleop_torso12_target = torch.zeros((self.num_envs, 2), device=self.device, dtype=self.robot.data.default_joint_pos.dtype)
        self._sync_arm_targets_from_default()
        self.teleop_torso12_target[:] = self.robot.data.default_joint_pos[:, self._torso12_joint_ids_t]

        self._joint_cmd_replay_traj: np.ndarray | None = None
        self._base_vel_replay_traj: np.ndarray | None = None
        self._joint_cmd_replay_t: int = 0
        self._replay_base_vel_cmd: torch.Tensor | None = None
        #: Number of control steps to hold each replay row before advancing (1 = every step, previous behavior).
        self._replay_hold_steps: int = 1
        self._replay_hold_counter: int = 0
        self._replay_log_pose_index: bool = False
        #: Optional (T, 3) / (T, 4) from demo; same row index and hold as ``joint_commands`` replay.
        self._object_replay_pos_local: np.ndarray | None = None
        self._object_replay_quat_w: np.ndarray | None = None
        #: Optional (T, num_joints) — snap robot to this after each step when replaying (pose / base motion).
        self._measured_joint_replay_traj: np.ndarray | None = None

    def set_joint_pos_replay(self, joint_commands: np.ndarray) -> None:
        """Play back saved teleop from ``joint_commands`` (T, num_joints): full ``joint_pos_cmd`` per step.

        Same layout as live teleop: ``actuated_base_dof_indices`` hold velocity commands (vx, vy, yaw_rate);
        body actuated indices hold position targets. Replay applies these each control step (not measured
        ``robot.data.joint_pos`` / ``joint_vel``).
        """
        jc = np.asarray(joint_commands, dtype=np.float32)
        if jc.ndim != 2 or jc.shape[1] != self.num_joints:
            raise ValueError(f"joint_commands must have shape (T, {self.num_joints}), got {jc.shape}")

        base_traj: np.ndarray | None = None
        if self.cfg.enable_base_control and self.actuated_base_dof_indices:
            ab = self.actuated_base_dof_indices
            base_traj = jc[:, ab]

        self._joint_cmd_replay_traj = jc
        self._base_vel_replay_traj = base_traj
        self._joint_cmd_replay_t = 0
        self._replay_hold_counter = 0
        self._object_replay_pos_local = None
        self._object_replay_quat_w = None
        self._measured_joint_replay_traj = None

    def set_measured_joint_replay(self, measured_joint_pos: np.ndarray) -> None:
        """Register ``measured_joint_pos`` (T, num_joints) for :meth:`write_measured_robot_replay_to_sim`.

        Call after :meth:`set_joint_pos_replay` with the same ``T``. Replay still applies ``joint_commands``
        each control step; after physics, the robot is snapped to the matching measured row so base pose
        matches sparse pose files (where base velocities in ``joint_commands`` are often zero).
        """
        if self._joint_cmd_replay_traj is None:
            raise RuntimeError("Call set_joint_pos_replay before set_measured_joint_replay.")
        mj = np.asarray(measured_joint_pos, dtype=np.float32)
        t = len(self._joint_cmd_replay_traj)
        n = int(self.num_joints)
        if mj.shape != (t, n):
            raise ValueError(f"measured_joint_pos must have shape ({t}, {n}), got {mj.shape}")
        self._measured_joint_replay_traj = mj

    def set_object_pose_replay(self, object_pos_local: np.ndarray, object_quat_w: np.ndarray) -> None:
        """Replay rigid object pose from a demo; must follow :meth:`set_joint_pos_replay` with matching ``T``."""
        if self._joint_cmd_replay_traj is None:
            raise RuntimeError("Call set_joint_pos_replay before set_object_pose_replay.")
        pl = np.asarray(object_pos_local, dtype=np.float32)
        qw = np.asarray(object_quat_w, dtype=np.float32)
        if pl.ndim != 2 or pl.shape[1] != 3 or qw.shape != (pl.shape[0], 4):
            raise ValueError(f"Expected object_pos_local (T, 3) and object_quat_w (T, 4); got {pl.shape}, {qw.shape}")
        t = len(self._joint_cmd_replay_traj)
        if pl.shape[0] != t:
            raise ValueError(f"object trajectory length {pl.shape[0]} must match joint_commands {t}")
        self._object_replay_pos_local = pl
        self._object_replay_quat_w = qw

    def write_object_replay_to_sim(self) -> None:
        """Set sphere root state from the current replay row (call after ``env.step``; zeros root velocity)."""
        if self._joint_cmd_replay_traj is None or self._object_replay_pos_local is None or self._object_replay_quat_w is None:
            return
        obj = getattr(self, "object", None)
        if obj is None:
            return
        t = self._joint_cmd_replay_t
        pl = self._object_replay_pos_local[t]
        qw = self._object_replay_quat_w[t]
        env_origin = self.scene.env_origins[0]
        dtype = obj.data.root_state_w.dtype
        pos_w = torch.as_tensor(pl, device=self.device, dtype=dtype) + env_origin.to(dtype=dtype)
        quat = torch.as_tensor(qw, device=self.device, dtype=dtype)
        root = obj.data.root_state_w[0:1].clone()
        root[0, 0:3] = pos_w
        root[0, 3:7] = quat
        root[0, 7:13] = 0.0
        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        obj.write_root_state_to_sim(root, env_ids)

    def write_measured_robot_replay_to_sim(self) -> None:
        """If :meth:`set_measured_joint_replay` was used, snap articulation to the current replay row."""
        if self._measured_joint_replay_traj is None:
            return
        t = self._joint_cmd_replay_t
        row = self._measured_joint_replay_traj[t]
        dtype = self.robot.data.default_joint_pos.dtype
        device = self.device
        joint_rows = torch.as_tensor(row, device=device, dtype=dtype).unsqueeze(0)
        joint_vel = torch.zeros_like(joint_rows)
        env_ids_t = torch.tensor([0], device=device, dtype=torch.long)
        self.robot.write_joint_state_to_sim(joint_rows, joint_vel, env_ids=env_ids_t)
        self.robot.set_joint_position_target(joint_rows, env_ids=env_ids_t)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((1, 6), dtype=dtype, device=device),
            env_ids=env_ids_t,
        )

    def apply_begin_from_teleop_demo_frame(
        self,
        measured_joint_pos_row: np.ndarray,
        *,
        object_pos_local: np.ndarray | None = None,
        object_quat_w: np.ndarray | None = None,
    ) -> None:
        """Set simulation and teleop buffers from one row of a teleop ``.npz`` (``measured_joint_pos``).

        Use with live teleop or ``--record-poses`` (not with ``--replay``): after :meth:`env.reset`, applies
        measured joint positions/velocities to the robot, optional rigid object root pose, and syncs
        ``teleop_*`` targets so the next :meth:`_pre_physics_step` matches this pose.
        """
        row = np.asarray(measured_joint_pos_row, dtype=np.float32).reshape(-1)
        n = int(self.num_joints)
        if row.shape[0] != n:
            raise ValueError(f"measured_joint_pos row must have length {n}, got {row.shape[0]}")

        dtype = self.robot.data.default_joint_pos.dtype
        device = self.device
        joint_rows = torch.as_tensor(row, device=device, dtype=dtype).unsqueeze(0)
        joint_vel = torch.zeros_like(joint_rows)
        env_ids_t = torch.tensor([0], device=device, dtype=torch.long)

        self.robot.set_joint_position_target(joint_rows, env_ids=env_ids_t)
        self.robot.write_joint_state_to_sim(joint_rows, joint_vel, env_ids=env_ids_t)
        self.joint_pos_cmd[:] = joint_rows
        self.prev_joint_pos_cmd[:] = joint_rows
        self.actions.zero_()
        self.prev_actions.zero_()

        self.robot.write_root_velocity_to_sim(
            torch.zeros((1, 6), dtype=dtype, device=device),
            env_ids=env_ids_t,
        )

        self.teleop_base_vel.zero_()
        self.teleop_arm_joint_target[:] = joint_rows[:, self._ik_joint_ids_t]
        self.teleop_torso12_target[:] = joint_rows[:, self._torso12_joint_ids_t]

        if self.cfg.enable_base_control and self.actuated_base_dof_indices:
            ab = self.actuated_base_dof_indices
            z = torch.zeros((self.num_envs, len(ab)), device=device, dtype=dtype)
            self.robot.set_joint_velocity_target(z, joint_ids=ab, env_ids=None)

        has_obj_pose = object_pos_local is not None and object_quat_w is not None
        if has_obj_pose:
            obj = getattr(self, "object", None)
            if obj is None:
                print("[begin-from] Demo includes object pose but this env has no rigid object; skipping object.")
            else:
                pl = np.asarray(object_pos_local, dtype=np.float32).reshape(3)
                qw = np.asarray(object_quat_w, dtype=np.float32).reshape(4)
                env_origin = self.scene.env_origins[0]
                pos_w = torch.as_tensor(pl, device=device, dtype=dtype) + env_origin.to(dtype=dtype)
                quat = torch.as_tensor(qw, device=device, dtype=dtype)
                root = obj.data.root_state_w[0:1].clone()
                root[0, 0:3] = pos_w
                root[0, 3:7] = quat
                root[0, 7:13] = 0.0
                obj.write_root_state_to_sim(root, env_ids_t)
        elif object_pos_local is not None or object_quat_w is not None:
            raise ValueError("object_pos_local and object_quat_w must both be set or both be None")

    def clear_joint_pos_replay(self) -> None:
        self._joint_cmd_replay_traj = None
        self._base_vel_replay_traj = None
        self._joint_cmd_replay_t = 0
        self._replay_base_vel_cmd = None
        self._replay_hold_counter = 0
        self._object_replay_pos_local = None
        self._object_replay_quat_w = None
        self._measured_joint_replay_traj = None

    def configure_replay_pose_hold(self, seconds: float, control_step_dt: float) -> None:
        """Hold each replay row for ``seconds`` before advancing; ``seconds <= 0`` advances every control step."""
        if seconds <= 0:
            self._replay_hold_steps = 1
            self._replay_log_pose_index = False
        else:
            self._replay_hold_steps = max(1, int(round(float(seconds) / float(control_step_dt))))
            self._replay_log_pose_index = True
        self._replay_hold_counter = 0

    def _sync_arm_targets_from_default(self) -> None:
        self.teleop_arm_joint_target[:] = self.robot.data.default_joint_pos[:, self._ik_joint_ids_t]

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self.teleop_base_vel.zero_()
        d = self.robot.data.default_joint_pos
        self.teleop_arm_joint_target[:] = d[:, self._ik_joint_ids_t]
        self.teleop_torso12_target[:] = d[:, self._torso12_joint_ids_t]

        if self.cfg.enable_base_control and self.actuated_base_dof_indices:
            ab = self.actuated_base_dof_indices
            n_ab = len(ab)
            self.joint_pos_cmd[:, ab] = 0
            z = torch.zeros((self.num_envs, n_ab), device=self.device, dtype=self.joint_pos_cmd.dtype)
            self.robot.set_joint_velocity_target(z, joint_ids=ab, env_ids=None)

        if self._joint_cmd_replay_traj is not None:
            self._joint_cmd_replay_t = 0
            self._replay_hold_counter = 0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_joint_pos_cmd[:] = self.joint_pos_cmd
        self.prev_actions[:] = self.actions
        self.actions = actions.clone()

        if self._joint_cmd_replay_traj is not None:
            traj = self._joint_cmd_replay_traj
            t = self._joint_cmd_replay_t
            if self._replay_hold_counter == 0 and self._replay_log_pose_index:
                print(f"[replay] pose {t + 1}/{len(traj)}")
            row = traj[t]
            q = torch.as_tensor(row, device=self.device, dtype=self.joint_pos_cmd.dtype).unsqueeze(0).expand(self.num_envs, -1)
            self.joint_pos_cmd[:] = q

            if self.cfg.enable_base_control and self.actuated_base_dof_indices and self._base_vel_replay_traj is not None:
                ab = self.actuated_base_dof_indices
                bv_np = self._base_vel_replay_traj[t]
                bv = torch.as_tensor(bv_np, device=self.device, dtype=self.joint_pos_cmd.dtype).unsqueeze(0).expand(self.num_envs, -1)
                low = -self.robot_hard_vel_limits[ab]
                high = self.robot_hard_vel_limits[ab]
                self._replay_base_vel_cmd = torch.clamp(bv, low, high)
            else:
                self._replay_base_vel_cmd = None

            self._replay_hold_counter += 1
            hs = max(1, int(self._replay_hold_steps))
            if self._replay_hold_counter >= hs:
                self._replay_hold_counter = 0
                self._joint_cmd_replay_t = (t + 1) % len(traj)
            return

        if self.cfg.enable_base_control and self.actuated_base_dof_indices:
            low = -self.robot_hard_vel_limits[self.actuated_base_dof_indices]
            high = self.robot_hard_vel_limits[self.actuated_base_dof_indices]
            bv = torch.clamp(self.teleop_base_vel, low, high)
            self.joint_pos_cmd[:, self.actuated_base_dof_indices] = bv

        self.joint_pos_cmd[:, self._locked_body_dof_indices] = self.robot.data.default_joint_pos[:, self._locked_body_dof_indices]
        self.joint_pos_cmd[:, self._torso12_joint_ids_t] = self.teleop_torso12_target
        self.joint_pos_cmd[:, self._ik_joint_ids_t] = self.teleop_arm_joint_target

    def _apply_action(self) -> None:
        if self._joint_cmd_replay_traj is not None:
            self.robot.set_joint_position_target(
                self.joint_pos_cmd[:, self.actuated_body_dof_indices],
                joint_ids=self.actuated_body_dof_indices,
                env_ids=None,
            )
            if self.cfg.enable_base_control and self.actuated_base_dof_indices and self._replay_base_vel_cmd is not None:
                self.robot.set_joint_velocity_target(
                    self._replay_base_vel_cmd,
                    joint_ids=self.actuated_base_dof_indices,
                    env_ids=None,
                )
            return
        # Live teleop: :meth:`_pre_physics_step` already filled ``joint_pos_cmd`` from keyboard buffers.
        # Do **not** call :meth:`AIRECEnv._apply_action` here: it overwrites ``joint_pos_cmd`` with
        # ``scale(self.actions, ...)`` where policy actions are zeros. ``scale`` maps [-1, 1] to joint limits,
        # so action 0 is the **midpoint** of each range — not the default pose — and the robot drifts.
        if self._fixed_joint_indices:
            default_pos = self.robot.data.default_joint_pos
            for idx in self._fixed_joint_indices:
                self.joint_pos_cmd[:, idx] = default_pos[:, idx]
            zf = torch.zeros((self.num_envs, len(self._fixed_joint_indices)), device=self.device)
            self.robot.set_joint_velocity_target(zf, joint_ids=self._fixed_joint_indices)
        self.robot.set_joint_position_target(
            self.joint_pos_cmd[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
            env_ids=None,
        )
        ab = getattr(self, "actuated_base_dof_indices", None)
        if self.cfg.enable_base_control and ab:
            self.robot.set_joint_velocity_target(
                self.joint_pos_cmd[:, ab],
                joint_ids=ab,
                env_ids=None,
            )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """When :attr:`_teleop_suppress_dones` is set (pose capture), never end episodes so the sim does not reset."""
        if getattr(self, "_teleop_suppress_dones", False):
            z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return z, z
        return super()._get_dones()