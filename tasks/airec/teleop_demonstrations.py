# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record / load teleop trajectories: measured state, commanded targets, optional object pose."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DEFAULT_DEMO_DIRNAME = "demonstrations"
DEMO_FILE_EXT = ".npz"
# ``--record-poses``: rewritten on every snapshot so data survives closing the sim without Delete.
POSE_LIVE_FILENAME = "poses_live.npz"

MEASURED_JOINT_POS_KEY = "measured_joint_pos"
MEASURED_JOINT_VEL_KEY = "measured_joint_vel"
JOINT_COMMANDS_KEY = "joint_commands"
# Object root position relative to the recording env origin (translation); same axes as world.
OBJECT_POS_LOCAL_KEY = "object_pos_local"
OBJECT_QUAT_W_KEY = "object_quat_w"


@dataclass(frozen=True)
class TeleopDemo:
    """One recorded episode.

    ``measured_joint_pos`` / ``measured_joint_vel``: (T, num_joints) — ``robot.data.joint_pos`` /
    ``joint_vel`` **after** each physics step (simulated state).

    ``joint_commands``: (T, num_joints) — full ``env.joint_pos_cmd`` **before** each step.
    Indices in ``actuated_base_dof_indices`` hold **velocity** commands (vx, vy, yaw_rate); body /
    arm / torso actuated indices hold **position** targets (same semantics as live teleop).

    ``object_pos_local``: optional (T, 3) — object root position in the recording env's local frame
    (world position minus ``scene.env_origins[recording_env]``).

    ``object_quat_w``: optional (T, 4) root orientation in world frame (w, x, y, z).
    """

    measured_joint_pos: np.ndarray
    measured_joint_vel: np.ndarray
    joint_commands: np.ndarray
    object_pos_local: np.ndarray | None = None
    object_quat_w: np.ndarray | None = None


def default_demonstrations_dir(repo_root: str | None = None) -> str:
    """Absolute path to ``<repo_root>/demonstrations`` (folder is created when recording)."""
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, DEFAULT_DEMO_DIRNAME)


class JointPosEpisodeRecorder:
    """Append measured state, joint_commands, and optional object pose; save ``demo_<i>.npz`` on episode end."""

    def __init__(self, out_dir: str, *, record_object: bool = False, live_pose_path: str | None = None):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.record_object = record_object
        self._live_pose_path = live_pose_path
        self.episode_index = 0
        self._joint_buffer: list[np.ndarray] = []
        self._joint_vel_buffer: list[np.ndarray] = []
        self._joint_commands_buffer: list[np.ndarray] = []
        self._object_pos_buffer: list[np.ndarray] = []
        self._object_quat_buffer: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._joint_buffer)

    def discard_episode(self) -> None:
        """Clear buffered steps without writing a file (does not advance ``episode_index``)."""
        self._joint_buffer.clear()
        self._joint_vel_buffer.clear()
        self._joint_commands_buffer.clear()
        self._object_pos_buffer.clear()
        self._object_quat_buffer.clear()
        if self._live_pose_path is not None and os.path.isfile(self._live_pose_path):
            try:
                os.remove(self._live_pose_path)
            except OSError:
                pass

    def observe_step(
        self,
        measured_joint_pos: np.ndarray,
        measured_joint_vel: np.ndarray,
        *,
        joint_commands: np.ndarray,
        object_pos_local: np.ndarray | None = None,
        object_quat_w: np.ndarray | None = None,
    ) -> None:
        self._joint_buffer.append(np.asarray(measured_joint_pos, dtype=np.float32).reshape(-1).copy())
        self._joint_vel_buffer.append(np.asarray(measured_joint_vel, dtype=np.float32).reshape(-1).copy())
        self._joint_commands_buffer.append(np.asarray(joint_commands, dtype=np.float32).reshape(-1).copy())
        if self.record_object:
            if object_pos_local is None or object_quat_w is None:
                raise ValueError("object_pos_local and object_quat_w are required when record_object=True")
            self._object_pos_buffer.append(np.asarray(object_pos_local, dtype=np.float32).reshape(3).copy())
            self._object_quat_buffer.append(np.asarray(object_quat_w, dtype=np.float32).reshape(4).copy())

    def flush_live_to_disk(self) -> str | None:
        """If ``live_pose_path`` was set (``--record-poses``), write current buffers there without clearing them."""
        if self._live_pose_path is None or not self._joint_buffer:
            return None
        measured_joint_pos = np.stack(self._joint_buffer, axis=0)
        measured_joint_vel = np.stack(self._joint_vel_buffer, axis=0)
        joint_commands = np.stack(self._joint_commands_buffer, axis=0)
        payload: dict[str, np.ndarray] = {
            MEASURED_JOINT_POS_KEY: measured_joint_pos,
            MEASURED_JOINT_VEL_KEY: measured_joint_vel,
            JOINT_COMMANDS_KEY: joint_commands,
        }
        if self.record_object:
            payload[OBJECT_POS_LOCAL_KEY] = np.stack(self._object_pos_buffer, axis=0)
            payload[OBJECT_QUAT_W_KEY] = np.stack(self._object_quat_buffer, axis=0)
        np.savez_compressed(self._live_pose_path, **payload)
        return self._live_pose_path

    def close_episode(self) -> str | None:
        """Write ``demo_{episode_index}.npz`` if any steps were recorded; increment episode index."""
        if not self._joint_buffer:
            return None
        measured_joint_pos = np.stack(self._joint_buffer, axis=0)
        measured_joint_vel = np.stack(self._joint_vel_buffer, axis=0)
        joint_commands = np.stack(self._joint_commands_buffer, axis=0)
        payload: dict[str, np.ndarray] = {
            MEASURED_JOINT_POS_KEY: measured_joint_pos,
            MEASURED_JOINT_VEL_KEY: measured_joint_vel,
            JOINT_COMMANDS_KEY: joint_commands,
        }
        if self.record_object:
            payload[OBJECT_POS_LOCAL_KEY] = np.stack(self._object_pos_buffer, axis=0)
            payload[OBJECT_QUAT_W_KEY] = np.stack(self._object_quat_buffer, axis=0)
        path = os.path.join(self.out_dir, f"demo_{self.episode_index}{DEMO_FILE_EXT}")
        np.savez_compressed(path, **payload)
        self.episode_index += 1
        self._joint_buffer.clear()
        self._joint_vel_buffer.clear()
        self._joint_commands_buffer.clear()
        self._object_pos_buffer.clear()
        self._object_quat_buffer.clear()
        return path


def load_teleop_demo(path: str) -> TeleopDemo:
    """Load ``.npz`` written by :class:`JointPosEpisodeRecorder`."""
    with np.load(path, allow_pickle=False) as data:
        for key in (MEASURED_JOINT_POS_KEY, MEASURED_JOINT_VEL_KEY, JOINT_COMMANDS_KEY):
            if key not in data.files:
                raise ValueError(f"Demo archive must contain '{key}'")

        measured_joint_pos = np.asarray(data[MEASURED_JOINT_POS_KEY], dtype=np.float32)
        measured_joint_vel = np.asarray(data[MEASURED_JOINT_VEL_KEY], dtype=np.float32)
        joint_commands = np.asarray(data[JOINT_COMMANDS_KEY], dtype=np.float32)

        has_local = OBJECT_POS_LOCAL_KEY in data.files
        has_quat = OBJECT_QUAT_W_KEY in data.files
        if has_local != has_quat:
            raise ValueError(
                f"Object pose requires both '{OBJECT_POS_LOCAL_KEY}' and '{OBJECT_QUAT_W_KEY}' or neither"
            )

        object_pos_local = None
        object_quat_w = None
        if has_quat:
            object_pos_local = np.asarray(data[OBJECT_POS_LOCAL_KEY], dtype=np.float32)
            object_quat_w = np.asarray(data[OBJECT_QUAT_W_KEY], dtype=np.float32)

    if measured_joint_pos.ndim != 2:
        raise ValueError(f"Expected measured_joint_pos shape (T, num_joints), got {measured_joint_pos.shape}")
    if measured_joint_vel.ndim != 2 or measured_joint_vel.shape != measured_joint_pos.shape:
        raise ValueError(
            f"Expected measured_joint_vel shape {measured_joint_pos.shape}, got {measured_joint_vel.shape}"
        )
    t, n_j = measured_joint_pos.shape
    if joint_commands.shape != (t, n_j):
        raise ValueError(f"Expected joint_commands shape {(t, n_j)}, got {joint_commands.shape}")
    if object_quat_w is not None:
        assert object_pos_local is not None
        if object_pos_local.shape != (t, 3):
            raise ValueError(f"Expected object_pos_local shape ({t}, 3), got {object_pos_local.shape}")
        if object_quat_w.shape != (t, 4):
            raise ValueError(f"Expected object_quat_w shape ({t}, 4), got {object_quat_w.shape}")
    return TeleopDemo(
        measured_joint_pos=measured_joint_pos,
        measured_joint_vel=measured_joint_vel,
        joint_commands=joint_commands,
        object_pos_local=object_pos_local,
        object_quat_w=object_quat_w,
    )