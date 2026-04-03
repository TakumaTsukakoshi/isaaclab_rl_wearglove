# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop + demo recording for AIREC: base SE(2) **velocity** + arm/torso **position** targets.

Demonstration ``.npz`` files store ``measured_joint_pos`` / ``measured_joint_vel`` (simulated state after each step),
``joint_commands`` (full ``joint_pos_cmd`` before each step: velocity at base DOFs, position targets elsewhere),
and for manipulation tasks (e.g. ``--task block``) ``object_pos_local`` (T, 3) relative to the recording env origin
and ``object_quat_w`` (T, 4) world orientation (w, x, y, z).

``--replay`` applies ``joint_commands`` each control step, then snaps the articulation to ``measured_joint_pos`` so the **base** (and whole pose) matches the recording; sparse pose captures often have zero base **velocities** in ``joint_commands`` while ``measured_joint_pos`` still reflects where the robot was.

Keyboard (when not in ``--replay``):

- Base: see printed ``Se2Keyboard`` legend (numpad / arrows, **Z**/**X** yaw).
- Right arm: **A–J** select joint 1–7, **K** / **L** decrease / increase angle.
- Left arm: **Q–U** select joint 1–7, **O** / **P** decrease / increase.
- Torso: **V** / **B** for joint 1, **N** / **M** for joint 2.
- **Backspace**: force an episode reset (save demo if ``--record-demonstrations``; with ``--record-poses`` the pose buffer is cleared and ``poses_live.npz`` is removed, then reset).
- With ``--record-poses``: press **9** in the **simulator** window (viewport focused) to append one snapshot (each snapshot is written immediately to ``<demo-dir>/poses_live.npz``). Episodes do not end on timeout or task termination while capturing poses. **Delete** exits teleop (optional; data is already on disk after each snapshot).
- With ``--task block`` and ``--control-object``: move the rigid sphere in **world** frame (m/s from ``--object_speed``): **1**/ **2** = −X / +X, **3**/ **4** = −Y / +Y, **5**/ **6** = −Z / +Z. Orientation unchanged. A cached pose is updated from keys, then **re-written after each physics step** with zero velocity so the sphere **stays** where you leave it (gravity does not accumulate).

Arm + torso position targets reset to defaults when an **episode ends** (timeout / termination), together with base velocity clearing — except in ``--record-poses`` mode, where episode boundaries are suppressed so the scene does not reset until **Backspace** or you exit.

To add another scene, register it in ``_TELEOP_JOINT_TASKS`` below (after the app starts).

Run::

    ./isaaclab.sh -p teleop_joints.py --task imitation
    ./isaaclab.sh -p teleop_joints.py --task wearglove
    ./isaaclab.sh -p teleop_joints.py --task block --viewer external
    ./isaaclab.sh -p teleop_joints.py --task block --record-demonstrations
    ./isaaclab.sh -p teleop_joints.py --task block --record-poses
    ./isaaclab.sh -p teleop_joints.py --task block --viewer external --record-poses --begin-from demonstrations/beautiful_grasp.npz
    ./isaaclab.sh -p teleop_joints.py --task block --control-object --object_speed 0.5
    ./isaaclab.sh -p teleop_joints.py --task block --replay demonstrations/demo_4.npz --video --video_length 1000
    ./isaaclab.sh -p teleop_joints.py --task block --replay demonstrations/poses.npz --replay-pose-hold 1
    ./isaaclab.sh -p teleop_joints.py --task block --viewer external --replay demonstrations/beautiful_grasp.npz --replay-hold-sec 10
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_PROJECT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TELEOP_JOINT_TASK_CHOICES = ("imitation", "block", "wearglove")


def build_parser(*, include_task: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint-space teleop for AIREC tasks: base + 7+7 arm joint positions.")
    if include_task:
        parser.add_argument(
            "--task",
            type=str,
            choices=_TELEOP_JOINT_TASK_CHOICES,
            default="imitation",
            help="Task / scene: imitation = AIREC2 floor only (no glove); block = sphere; wearglove = deformable glove + Shadow Hand (wear task).",
        )
    parser.add_argument("--joint_speed", type=float, default=1.2, help="Rad/s while holding +/- keys.")
    parser.add_argument(
        "--couple-arms",
        action="store_true",
        help="Apply each arm joint delta to the same-index joint on the other arm as well (same sign).",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        choices=("head", "external", "none"),
        default="head",
        help="Viewport: head | external | none.",
    )
    _demo = parser.add_mutually_exclusive_group()
    _demo.add_argument(
        "--record-demonstrations",
        action="store_true",
        help="Save each step to <demo-dir>/demo_<episode>.npz: joint_pos (T,N), joint_vel (T,N); block task also object_pos_local (T,3) env-local + object_quat_w (T,4) wxyz.",
    )
    _demo.add_argument(
        "--record-poses",
        action="store_true",
        dest="record_poses",
        help="Append one pose row when you press 9 in the sim window; each row is saved immediately to <demo-dir>/poses_live.npz. Same .npz keys as demos. Delete exits (optional).",
    )
    _demo.add_argument(
        "--replay",
        type=str,
        default=None,
        metavar="PATH",
        help="Replay a .npz from --record-demonstrations (loops the clip).",
    )
    parser.add_argument(
        "--replay-pose-hold",
        "--replay-hold-sec",
        type=float,
        default=0.0,
        metavar="SEC",
        dest="replay_pose_hold",
        help="With --replay: hold each saved row for SEC seconds (e.g. 1 or 10 for pose-by-pose slideshow). 0 = continuous (default). Alias: --replay-hold-sec.",
    )
    parser.add_argument(
        "--begin-from",
        type=str,
        default=None,
        metavar="PATH",
        help="Live teleop / --record-poses only: after reset, load a teleop .npz and apply measured_joint_pos "
        "(and object pose if keys present) from --begin-from-index. Incompatible with --replay.",
    )
    parser.add_argument(
        "--begin-from-index",
        type=int,
        default=0,
        metavar="N",
        help="Frame index in --begin-from demo (default: 0).",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default=None,
        help="Output directory when recording (default: <repo_root>/demonstrations).",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record the viewport to an mp4 (uses gymnasium RecordVideo; enables cameras).",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Number of env steps to record (same convention as scripts/play.py).",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory for mp4 files (default: <repo_root>/videos).",
    )
    parser.add_argument(
        "--control-object",
        action="store_true",
        help="Block task only: nudge the sphere (rigid body) in world XYZ with keys 1–6 (see module docstring).",
    )
    parser.add_argument(
        "--object_speed",
        type=float,
        default=0.35,
        metavar="MPS",
        help="With --control-object: world-frame translation speed (m/s) while holding 1–6.",
    )
    return parser


from isaaclab.app import AppLauncher

parser = build_parser(include_task=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import weakref  # noqa: E402

import gymnasium as gym  # noqa: E402


def _unwrap_task_env(env: gym.Env) -> gym.Env:
    """Strip gymnasium wrappers (e.g. RecordVideo) to reach the Isaac Lab task env."""
    e: gym.Env = env
    while isinstance(e, gym.Wrapper):
        e = e.env
    return e


import carb  # noqa: E402
import omni  # noqa: E402
import torch  # noqa: E402
from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg  # noqa: E402

import tasks.airec  # noqa: F401, E402
from tasks.airec.teleop_block import TeleopBlockEnv, TeleopBlockEnvCfg  # noqa: E402
from tasks.airec.teleop_demonstrations import (  # noqa: E402
    POSE_LIVE_FILENAME,
    JointPosEpisodeRecorder,
    default_demonstrations_dir,
    load_teleop_demo,
)
from tasks.airec.teleop_imitation import TeleopImitationEnv, TeleopImitationEnvCfg  # noqa: E402
from tasks.airec.teleop_wearglove import TeleopWearGloveEnv, TeleopWearGloveEnvCfg  # noqa: E402

_TELEOP_JOINT_TASKS: dict[str, tuple[type, type]] = {
    "imitation": (TeleopImitationEnv, TeleopImitationEnvCfg),
    "block": (TeleopBlockEnv, TeleopBlockEnvCfg),
    "wearglove": (TeleopWearGloveEnv, TeleopWearGloveEnvCfg),
}

_LEFT_SELECT_KEYS = {"Q": 0, "W": 1, "E": 2, "R": 3, "T": 4, "Y": 5, "U": 6}
_RIGHT_SELECT_KEYS = {"A": 0, "S": 1, "D": 2, "F": 3, "G": 4, "H": 5, "J": 6}


class TeleopBodyTargets:
    """Teleop command buffers: joint **positions** (arms + torso) and base **velocities** ``(vx, vy, ωz)``."""

    def __init__(self, robot, env, device: torch.device, dtype: torch.dtype):
        left_ids = sorted(list(robot.find_joints("left_arm_joint_.*")[0]))
        right_ids = sorted(list(robot.find_joints("right_arm_joint_.*")[0]))
        if len(left_ids) != 7 or len(right_ids) != 7:
            raise RuntimeError(f"Expected 7 left + 7 right arm joints, got {len(left_ids)} + {len(right_ids)}")

        self.arm_joint_ids: list[int] = list(env.teleop_arm_joint_ids)
        t1 = robot.joint_names.index("torso_joint_1")
        t2 = robot.joint_names.index("torso_joint_2")
        self.torso_joint_ids = (t1, t2)
        self.all_joint_ids: list[int] = [*self.arm_joint_ids, t1, t2]
        self._jid_t = torch.tensor(self.all_joint_ids, device=device, dtype=torch.long)

        arm_to_col = {jid: k for k, jid in enumerate(self.arm_joint_ids)}
        self.left_cols = torch.tensor([arm_to_col[j] for j in left_ids], device=device, dtype=torch.long)
        self.right_cols = torch.tensor([arm_to_col[j] for j in right_ids], device=device, dtype=torch.long)
        self._n_arm = len(self.arm_joint_ids)
        self.torso_cols = torch.tensor([self._n_arm, self._n_arm + 1], device=device, dtype=torch.long)

        self.targets = torch.zeros((1, len(self.all_joint_ids)), device=device, dtype=dtype)
        self.base_vel = torch.zeros((1, 3), device=device, dtype=dtype)
        self.reset_joint_targets_from_default(robot)

    def reset_joint_targets_from_default(self, robot) -> None:
        d = robot.data.default_joint_pos[0]
        self.targets[0] = d[self._jid_t]
        self.base_vel.zero_()

    def sync_targets_from_robot(self, robot) -> None:
        """Set arm/torso command targets from current simulated joint positions (e.g. after ``--begin-from``)."""
        self.targets[0] = robot.data.joint_pos[0, self._jid_t].to(
            device=self.targets.device, dtype=self.targets.dtype
        )
        self.base_vel.zero_()

    def clamp_joint_targets_to_limits(self, robot) -> None:
        lim = robot.data.soft_joint_pos_limits[0]
        lo = lim[self._jid_t, 0]
        hi = lim[self._jid_t, 1]
        self.targets[0] = torch.clamp(self.targets[0], lo, hi)

    def write_to_env(self, env) -> None:
        env.teleop_arm_joint_target[:] = self.targets[:, : self._n_arm]
        env.teleop_torso12_target[:] = self.targets[:, self._n_arm :]
        env.teleop_base_vel[:] = self.base_vel


def _carb_key_name(event) -> str:
    inp = event.input
    if isinstance(inp, str):
        return inp
    for attr in ("name", "value"):
        v = getattr(inp, attr, None)
        if v is not None:
            return str(v)
    return str(inp)


def _carb_key_token(name: str) -> str:
    """Normalize carb / Isaac keyboard names to tokens used in :class:`BodyJointKeyboard` (e.g. ``A``, ``K``, ``BACKSPACE``)."""
    s = str(name).strip()
    if "." in s:
        s = s.split(".")[-1]
    nu = s.upper()
    for prefix in ("KEY_", "INPUT_", "KEYBOARD_"):
        if nu.startswith(prefix):
            nu = nu[len(prefix) :]
            break
    if nu.startswith("KEY_"):
        nu = nu[4:]
    aliases = {
        "BACK_SPACE": "BACKSPACE",
        "NUMPAD_ENTER": "ENTER",
    }
    return aliases.get(nu, nu)


class BodyJointKeyboard:
    """Carb keyboard for arm/torso joint deltas + :class:`Se2Keyboard` for base velocity into :class:`TeleopBodyTargets`."""

    def __init__(
        self,
        joint_speed: float,
        couple_arms: bool,
        *,
        sim_device: str,
        command_device: torch.device,
        command_dtype: torch.dtype,
        record_pose_snapshots: bool = False,
        stop_recording_on_delete: bool = False,
        control_object: bool = False,
        v_x_sensitivity: float = 0.6,
        v_y_sensitivity: float = 0.35,
        omega_z_sensitivity: float = 0.9,
    ):
        self.joint_speed = float(joint_speed)
        self.couple_arms = bool(couple_arms)
        self._device = command_device
        self._dtype = command_dtype
        self._se2 = Se2Keyboard(
            Se2KeyboardCfg(
                sim_device=sim_device,
                # v_x_sensitivity=v_x_sensitivity,
                # v_y_sensitivity=v_y_sensitivity,
                # omega_z_sensitivity=omega_z_sensitivity,
            )
        )
        self._se2.reset()

        self.sel_r = 0
        self.sel_l = 0
        self._r_dec = False
        self._r_inc = False
        self._l_dec = False
        self._l_inc = False
        self._t1_dec = False
        self._t1_inc = False
        self._t2_dec = False
        self._t2_inc = False
        self._force_reset = False
        self._stop_recording = False
        self._stop_recording_on_delete = bool(stop_recording_on_delete)
        self._record_pose_snapshots = bool(record_pose_snapshots)
        self._pose_snapshots_pending = 0
        self._control_object = bool(control_object)
        self._obj_x_dec = self._obj_x_inc = False
        self._obj_y_dec = self._obj_y_inc = False
        self._obj_z_dec = self._obj_z_inc = False
        # Cached root state (1, 13) for --control-object: re-written every step so the sphere does not fall when keys are released.
        self._obj_hold_root: torch.Tensor | None = None

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def consume_force_reset(self) -> bool:
        """Return True once if Backspace was pressed since the last call."""
        if self._force_reset:
            self._force_reset = False
            return True
        return False

    def consume_stop_recording(self) -> bool:
        """Return True once if Delete was pressed (only when ``stop_recording_on_delete`` was True)."""
        if self._stop_recording:
            self._stop_recording = False
            return True
        return False

    def consume_pose_snapshot_requests(self) -> int:
        """Return how many pose snapshots were requested (key **9**) since the last call."""
        n = self._pose_snapshots_pending
        self._pose_snapshots_pending = 0
        return n

    def reset_object_hold(self) -> None:
        """Forget cached sphere pose; next ``apply_object_hold_pre_step`` copies from sim (call after episode reset)."""
        self._obj_hold_root = None

    @property
    def base_se2_keyboard(self) -> Se2Keyboard:
        return self._se2

    def __del__(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _apply_joint_deltas(self, dt: float, body: TeleopBodyTargets) -> None:
        step = self.joint_speed * dt
        t = body.targets[0]
        lc, rc, tc = body.left_cols, body.right_cols, body.torso_cols
        if self._r_dec:
            t[int(rc[self.sel_r].item())] -= step
            if self.couple_arms:
                t[int(lc[self.sel_r].item())] -= step
        if self._r_inc:
            t[int(rc[self.sel_r].item())] += step
            if self.couple_arms:
                t[int(lc[self.sel_r].item())] += step
        if self._l_dec:
            t[int(lc[self.sel_l].item())] -= step
            if self.couple_arms:
                t[int(rc[self.sel_l].item())] -= step
        if self._l_inc:
            t[int(lc[self.sel_l].item())] += step
            if self.couple_arms:
                t[int(rc[self.sel_l].item())] += step
        if self._t1_dec:
            t[int(tc[0].item())] -= step
        if self._t1_inc:
            t[int(tc[0].item())] += step
        if self._t2_dec:
            t[int(tc[1].item())] -= step
        if self._t2_inc:
            t[int(tc[1].item())] += step


    def apply_step(self, sim_dt: float, body: TeleopBodyTargets) -> None:
        """Integrate joint deltas from held keys; sample base velocity from ``Se2Keyboard`` into ``body.base_vel``."""
        self._apply_joint_deltas(sim_dt, body)
        body.base_vel[:] = self._se2.advance().to(device=self._device, dtype=self._dtype)

    def apply_object_hold_pre_step(self, sim_dt: float, object_speed: float, task_env) -> None:
        """Update cached sphere pose from held 1–6 keys (before ``env.step``)."""
        if not self._control_object or not hasattr(task_env, "object"):
            return
        obj = task_env.object
        cfg = task_env.cfg
        if self._obj_hold_root is None:
            self._obj_hold_root = obj.data.root_state_w[0:1].clone()

        step = float(object_speed) * float(sim_dt)
        if step > 0.0:
            dx = dy = dz = 0.0
            if self._obj_x_dec:
                dx -= step
            if self._obj_x_inc:
                dx += step
            if self._obj_y_dec:
                dy -= step
            if self._obj_y_inc:
                dy += step
            if self._obj_z_dec:
                dz -= step
            if self._obj_z_inc:
                dz += step
            if dx != 0.0 or dy != 0.0 or dz != 0.0:
                self._obj_hold_root[0, 0] += dx
                self._obj_hold_root[0, 1] += dy
                self._obj_hold_root[0, 2] += dz

        z_floor = float(cfg.bed_height + cfg.bed_depth / 2.0 + cfg.object_radius) - 0.02
        z_floor_t = torch.tensor(z_floor, device=self._obj_hold_root.device, dtype=self._obj_hold_root.dtype)
        self._obj_hold_root[0, 2] = torch.maximum(self._obj_hold_root[0, 2], z_floor_t)

    def apply_object_hold_post_step(self, task_env) -> None:
        """Re-write cached pose with zero root velocity after ``env.step`` so gravity does not drift the sphere."""
        if not self._control_object or not hasattr(task_env, "object"):
            return
        if self._obj_hold_root is None:
            return
        obj = task_env.object
        env_ids = torch.tensor([0], device=task_env.device, dtype=torch.long)
        self._obj_hold_root[0, 7:13] = 0.0
        obj.write_root_state_to_sim(self._obj_hold_root, env_ids)

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        name = _carb_key_name(event)
        tok = _carb_key_token(name)
        if self._control_object:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if tok == "1":
                    self._obj_x_dec = True
                    return True
                if tok == "2":
                    self._obj_x_inc = True
                    return True
                if tok == "3":
                    self._obj_y_dec = True
                    return True
                if tok == "4":
                    self._obj_y_inc = True
                    return True
                if tok == "5":
                    self._obj_z_dec = True
                    return True
                if tok == "6":
                    self._obj_z_inc = True
                    return True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                if tok == "1":
                    self._obj_x_dec = False
                    return True
                if tok == "2":
                    self._obj_x_inc = False
                    return True
                if tok == "3":
                    self._obj_y_dec = False
                    return True
                if tok == "4":
                    self._obj_y_inc = False
                    return True
                if tok == "5":
                    self._obj_z_dec = False
                    return True
                if tok == "6":
                    self._obj_z_inc = False
                    return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if tok == "BACKSPACE":
                self._force_reset = True
                return True
            if self._stop_recording_on_delete and tok in ("DELETE", "FORWARD_DELETE", "DEL"):
                self._stop_recording = True
                return True
            if self._record_pose_snapshots and tok == "9":
                self._pose_snapshots_pending += 1
                return True
            if tok in _RIGHT_SELECT_KEYS:
                self.sel_r = _RIGHT_SELECT_KEYS[tok]
                return True
            if tok in _LEFT_SELECT_KEYS:
                self.sel_l = _LEFT_SELECT_KEYS[tok]
                return True
            if tok == "K":
                self._r_dec = True
            elif tok == "L":
                self._r_inc = True
            elif tok == "O":
                self._l_dec = True
            elif tok == "P":
                self._l_inc = True
            elif tok == "V":
                self._t1_dec = True
            elif tok == "B":
                self._t1_inc = True
            elif tok == "N":
                self._t2_dec = True
            elif tok == "M":
                self._t2_inc = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if tok == "K":
                self._r_dec = False
            elif tok == "L":
                self._r_inc = False
            elif tok == "O":
                self._l_dec = False
            elif tok == "P":
                self._l_inc = False
            elif tok == "V":
                self._t1_dec = False
            elif tok == "B":
                self._t1_inc = False
            elif tok == "N":
                self._t2_dec = False
            elif tok == "M":
                self._t2_inc = False
        return True


def run_teleop_joint_loop(args_cli, simulation_app, project_root: str) -> None:
    """Run until the app exits. ``args_cli`` must include ``task`` when using the unified parser."""
    task_key = getattr(args_cli, "task", "imitation")
    if task_key is None:
        task_key = "imitation"
    task_key = str(task_key).lower()

    if getattr(args_cli, "control_object", False) and task_key != "block":
        print("[WARN] --control-object only applies to --task block; ignoring.")
    control_object = bool(getattr(args_cli, "control_object", False)) and task_key == "block"

    replay_mode = args_cli.replay is not None
    begin_from_path = getattr(args_cli, "begin_from", None)
    if begin_from_path is not None and replay_mode:
        raise RuntimeError("--begin-from cannot be used with --replay")
    demo_dir = args_cli.demo_dir if args_cli.demo_dir is not None else default_demonstrations_dir(project_root)
    want_record = args_cli.record_demonstrations or args_cli.record_poses
    recorder: JointPosEpisodeRecorder | None = None
    total_poses_session = 0

    EnvCls, CfgCls = _TELEOP_JOINT_TASKS[task_key]
    cfg = CfgCls(viewer_mode=args_cli.viewer)
    render_mode = "rgb_array" if args_cli.video else None
    env = EnvCls(cfg, render_mode=render_mode)

    if replay_mode:
        demo = load_teleop_demo(args_cli.replay)
        env.set_joint_pos_replay(demo.joint_commands)
        env.set_measured_joint_replay(demo.measured_joint_pos)
        if demo.object_pos_local is not None and demo.object_quat_w is not None:
            env.set_object_pose_replay(demo.object_pos_local, demo.object_quat_w)
            _obj = (
                f" object_pos_local {demo.object_pos_local.shape} object_quat_w {demo.object_quat_w.shape} "
                f"(sphere pose replayed after each step)"
            )
        else:
            _obj = ""
        print(
            f"Replay: loaded {args_cli.replay} joint_commands {demo.joint_commands.shape}; "
            f"measured_joint_pos {demo.measured_joint_pos.shape} (robot snapped to measured state after each step){_obj}"
        )

    if args_cli.video:
        video_folder = args_cli.video_dir if args_cli.video_dir is not None else os.path.join(project_root, "videos")
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix=f"teleop_{task_key}_replay" if replay_mode else f"teleop_{task_key}",
            disable_logger=True,
        )
        print(f"[INFO] Recording video: {video_folder}/  length={args_cli.video_length} steps")

    env.reset()
    print(f"Teleop joint-space: task={task_key} env={EnvCls.__name__}")

    task_env = _unwrap_task_env(env)
    if args_cli.record_poses:
        task_env._teleop_suppress_dones = True
    if want_record:
        _live = (
            os.path.join(demo_dir, POSE_LIVE_FILENAME) if args_cli.record_poses else None
        )
        recorder = JointPosEpisodeRecorder(
            demo_dir,
            record_object=hasattr(task_env, "object"),
            live_pose_path=_live,
        )
    robot = task_env.robot
    decimation = int(getattr(task_env.cfg, "decimation", 1))
    physics_dt = float(getattr(task_env, "physics_dt", None) or task_env.cfg.physics_dt)
    sim_dt = physics_dt * decimation

    if replay_mode and args_cli.replay_pose_hold > 0:
        task_env.configure_replay_pose_hold(args_cli.replay_pose_hold, sim_dt)
        task_env._teleop_suppress_dones = True
        print(
            f"[replay] holding each row {args_cli.replay_pose_hold:g}s "
            f"({task_env._replay_hold_steps} control steps @ sim_dt={sim_dt:.4f}s); looping clip"
        )

    raw_dev = task_env.device
    device = torch.device(raw_dev) if isinstance(raw_dev, str) else raw_dev
    dtype = robot.data.default_joint_pos.dtype
    dev_str = str(device)

    if begin_from_path is not None:
        demo_bf = load_teleop_demo(begin_from_path)
        idx = int(getattr(args_cli, "begin_from_index", 0))
        t_demo = int(demo_bf.measured_joint_pos.shape[0])
        if not (0 <= idx < t_demo):
            raise ValueError(f"--begin-from-index {idx} out of range for demo length T={t_demo}")
        row = demo_bf.measured_joint_pos[idx]
        opl = oqw = None
        if demo_bf.object_pos_local is not None and demo_bf.object_quat_w is not None:
            opl = demo_bf.object_pos_local[idx]
            oqw = demo_bf.object_quat_w[idx]
        task_env.apply_begin_from_teleop_demo_frame(row, object_pos_local=opl, object_quat_w=oqw)
        print(
            f"[begin-from] {begin_from_path} frame {idx}/{t_demo} "
            f"(object pose: {'yes' if opl is not None else 'no'})"
        )

    body = TeleopBodyTargets(robot, task_env, device, dtype)
    if begin_from_path is not None:
        body.sync_targets_from_robot(robot)

    kb: BodyJointKeyboard | None = None
    if not replay_mode:
        kb = BodyJointKeyboard(
            args_cli.joint_speed,
            args_cli.couple_arms,
            sim_device=dev_str,
            command_device=device,
            command_dtype=dtype,
            record_pose_snapshots=args_cli.record_poses,
            stop_recording_on_delete=args_cli.record_poses,
            control_object=control_object,
        )
        task_env.teleop_base_vel.zero_()
        body.base_vel.zero_()


        print(kb.base_se2_keyboard)
        print("Arm/torso joint teleop (see module docstring). joint_speed=", args_cli.joint_speed, "rad/s", end="")
        if args_cli.couple_arms:
            print("  couple_arms=True (same-index deltas on both arms)", end="")
        print()
        print("  Right: A-J pick joint, K/L -/+  |  Left: Q-U, O/P  |  Torso: V/B (j1), N/M (j2)  |  Backspace: reset episode")
        if args_cli.record_poses:
            print(
                f"  --record-poses: 9 in sim = snapshot (saved to {demo_dir}/{POSE_LIVE_FILENAME})  |  Delete = exit"
            )
        if control_object:
            print(
                f"  --control-object: 1/2 = −X/+X  3/4 = −Y/+Y  5/6 = −Z/+Z "
                f"(world m/s={args_cli.object_speed}); pose re-written each step (no drop when keys up)"
            )
        print(f"  Right: selected joint {kb.sel_r + 1}/7  |  Left: selected joint {kb.sel_l + 1}/7")
    if recorder is not None:
        _extra = " + object pose (local pos + world quat)" if recorder.record_object else ""
        mel = int(getattr(task_env, "max_episode_length", -1))
        if args_cli.record_poses:
            print(
                f"Recording poses: key 9 appends one row to {demo_dir}/{POSE_LIVE_FILENAME} immediately "
                f"(measured_joint_pos/vel + joint_commands{_extra}). "
                f"Episode timeout disabled; Backspace clears buffer and removes {POSE_LIVE_FILENAME}; "
                f"Delete exits (data already on disk)."
            )
        else:
            print(
                f"Recording to {demo_dir} as demo_<episode>.npz "
                f"(measured_joint_pos/vel + joint_commands{_extra}; "
                f"one row per control step; timeout at max_episode_length={mel})"
            )

    zero_action = torch.zeros((1, task_env.cfg.num_actions), device=device, dtype=dtype)
    pending_ep_resync = False
    video_timestep = 0
    pose_exit_via_delete = False

    while simulation_app.is_running():
        if not replay_mode:
            assert kb is not None
            if args_cli.record_poses and kb.consume_stop_recording():
                pose_exit_via_delete = True
                if recorder is not None and len(recorder) > 0:
                    recorder.flush_live_to_disk()
                print(
                    "Stopping pose recording (Delete); exiting teleop loop "
                    f"(poses: {demo_dir}/{POSE_LIVE_FILENAME})."
                )
                break
            if kb.consume_force_reset():
                if recorder is not None:
                    if args_cli.record_poses:
                        recorder.discard_episode()
                        print(
                            "Backspace: pose buffer cleared and poses_live.npz removed (if present); resetting episode."
                        )
                    else:
                        n_bs = len(recorder)
                        saved = recorder.close_episode()
                        if saved:
                            print(f"Saved {saved} (Backspace; T={n_bs})")
                try:
                    env.reset(hard=True)
                except TypeError:
                    env.reset()
                kb.reset_object_hold()
                kb.base_se2_keyboard.reset()
                body.reset_joint_targets_from_default(robot)
                body.clamp_joint_targets_to_limits(robot)
                body.write_to_env(task_env)
                pending_ep_resync = False
                print("Episode reset (Backspace)")
            elif pending_ep_resync:
                pending_ep_resync = False
                if control_object:
                    kb.reset_object_hold()
                kb.base_se2_keyboard.reset()
                body.reset_joint_targets_from_default(robot)

            kb.apply_step(sim_dt, body)
            body.clamp_joint_targets_to_limits(robot)
            body.write_to_env(task_env)
            if control_object:
                kb.apply_object_hold_pre_step(sim_dt, args_cli.object_speed, task_env)

        joint_commands = None
        if recorder is not None and not replay_mode:
            joint_commands = task_env.joint_pos_cmd[0].detach().cpu().numpy().copy()

        ep_len_before = (
            int(task_env.episode_length_buf[0].item()) if recorder is not None else 0
        )

        _, _, terminated, truncated, _ = env.step(zero_action)

        if replay_mode:
            task_env.write_object_replay_to_sim()
            task_env.write_measured_robot_replay_to_sim()
        elif control_object:
            assert kb is not None
            kb.apply_object_hold_post_step(task_env)

        if recorder is not None:
            jp = task_env.robot.data.joint_pos[0].detach().cpu().numpy()
            jv = task_env.robot.data.joint_vel[0].detach().cpu().numpy()
            if args_cli.record_poses:
                assert kb is not None
                n_pose = kb.consume_pose_snapshot_requests()
                for _ in range(n_pose):
                    if recorder.record_object:
                        obj = task_env.object
                        env_origin = task_env.scene.env_origins[0]
                        pos_w = obj.data.root_pos_w[0]
                        pos_local = (pos_w - env_origin).detach().cpu().numpy()
                        recorder.observe_step(
                            jp,
                            jv,
                            joint_commands=joint_commands,
                            object_pos_local=pos_local,
                            object_quat_w=obj.data.root_quat_w[0].detach().cpu().numpy(),
                        )
                    else:
                        recorder.observe_step(jp, jv, joint_commands=joint_commands)
                    total_poses_session += 1
                    _flushed = recorder.flush_live_to_disk()
                    _path_note = f" -> {_flushed}" if _flushed else ""
                    print(
                        f"Recorded pose {len(recorder)} in buffer; session total: {total_poses_session}{_path_note}"
                    )
            else:
                if recorder.record_object:
                    obj = task_env.object
                    env_origin = task_env.scene.env_origins[0]
                    pos_w = obj.data.root_pos_w[0]
                    pos_local = (pos_w - env_origin).detach().cpu().numpy()
                    recorder.observe_step(
                        jp,
                        jv,
                        joint_commands=joint_commands,
                        object_pos_local=pos_local,
                        object_quat_w=obj.data.root_quat_w[0].detach().cpu().numpy(),
                    )
                else:
                    recorder.observe_step(jp, jv, joint_commands=joint_commands)
        if (
            recorder is not None
            and not args_cli.record_poses
            and bool(torch.logical_or(terminated, truncated).any().item())
        ):
            n = len(recorder)
            # First control step after reset sometimes spuriously reports done (physics / done flags);
            # do not write a 1-step demo for that (Backspace still saves via close_episode above).
            if n == 1 and ep_len_before == 0:
                recorder.discard_episode()
                print(
                    "[teleop record] Discarded 1-step episode (done on first step after reset); "
                    "not saving demo_*.npz."
                )
            else:
                saved = recorder.close_episode()
                if saved:
                    mel = int(getattr(task_env, "max_episode_length", -1))
                    print(
                        f"Saved {saved}  (T={n} control steps; "
                        f"episode_length_buf before terminal step was {ep_len_before}; "
                        f"max_episode_length={mel})"
                    )

        if not replay_mode and not args_cli.record_poses:
            pending_ep_resync = bool(torch.logical_or(terminated, truncated).any().item())

        if args_cli.video:
            if video_timestep == args_cli.video_length:
                if recorder is not None and len(recorder) > 0 and not args_cli.record_poses:
                    n_vid = len(recorder)
                    saved = recorder.close_episode()
                    if saved:
                        print(
                            f"Saved {saved} (stopped at --video_length={args_cli.video_length}; T={n_vid})"
                        )
                break
        video_timestep += 1

    if recorder is not None and len(recorder) > 0:
        if args_cli.record_poses:
            recorder.flush_live_to_disk()
            n_exit = len(recorder)
            print(
                f"Pose capture: {n_exit} row(s) in buffer; on-disk file "
                f"{os.path.join(demo_dir, POSE_LIVE_FILENAME)}"
                + (" (Delete exit)" if pose_exit_via_delete else " (window closed or other exit)")
            )
        else:
            n_exit = len(recorder)
            saved = recorder.close_episode()
            if saved:
                print(f"Saved {saved} (flush at exit; T={n_exit})")

    if args_cli.record_poses:
        print(f"Total poses recorded this session: {total_poses_session}")

    try:
        env.close()
    except Exception:
        pass
    simulation_app.close()


def main() -> None:
    run_teleop_joint_loop(args_cli, simulation_app, _PROJECT_ROOT)


if __name__ == "__main__":
    main()