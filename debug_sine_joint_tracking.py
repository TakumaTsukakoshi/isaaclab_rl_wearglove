# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone joint-tracking diagnostic with configurable target waveforms.

No checkpoint / policy is loaded. The robot is spawned alone (``scene_mode="free_space"``)
and each actuated joint receives an absolute position target::

    q_target_raw(t) = q_init + A * sin(2*pi*f*t)      (A, f from CLI)
    q_cmd(t)        = clamp(q_target_raw(t), hard joint limits)   # sent to Isaac

The existing environment class is reused unmodified for robot init, controller
(PD via ``set_joint_position_target``), decimation stepping and resets; only
``_pre_physics_step`` is overridden in a local subclass so the "action" is an
absolute joint target in radians (no EMA / residual scaling).

Example::

    python debug_sine_joint_tracking.py --amplitude-deg 5 --frequency-hz 0.2 \\
        --duration-sec 20 --output-dir outputs/sine_tracking
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Waveform joint tracking diagnostic (robot only, no policy).")
parser.add_argument("--task", type=str, default="AIREC_Reach_Deformable_Bracelet", help="Registered task name (env cfg source).")
parser.add_argument(
    "--waveform",
    choices=("sine", "multisine", "ramp", "square", "step"),
    default="sine",
    help="Joint target waveform (default: sine).",
)
parser.add_argument("--amplitude-deg", type=float, default=5.0, help="Maximum target displacement from initial pose [deg].")
parser.add_argument("--frequency-hz", type=float, default=0.2, help="Base frequency for sine/multisine/square [Hz].")
parser.add_argument("--multisine-components", type=int, default=3, help="Number of integer harmonics in multisine.")
parser.add_argument("--step-time-sec", type=float, default=1.0, help="Step onset time [s].")
parser.add_argument("--duration-sec", type=float, default=20.0, help="Total run time [s] (control-step resolution).")
parser.add_argument("--output-dir", type=str, default="outputs/sine_tracking", help="Directory for CSV/NPZ/PNG outputs.")
parser.add_argument("--env-id", type=int, default=0, help="Env index to record (default 0).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (recording uses --env-id).")
parser.add_argument(
    "--disable-self-collision",
    action="store_true",
    help="Disable AIREC self-collision (same semantics as play.py).",
)
parser.add_argument("--no-plots", action="store_true", help="Save CSV/NPZ only, skip PNG plots.")
parser.add_argument(
    "--diagnostics",
    action="store_true",
    help="Run the per-joint diagnostic suite (mapping/limits/drive/sign/offset/saturation) instead of a waveform sweep.",
)
parser.add_argument(
    "--diag-step-deg",
    type=float,
    default=5.0,
    help="Per-joint isolated step magnitude for --diagnostics [deg] (applied as +deg and -deg).",
)
parser.add_argument(
    "--diag-settle-sec",
    type=float,
    default=3.0,
    help="Settle/hold time per direction for --diagnostics step tests [s].",
)
parser.add_argument("--video", action="store_true", help="Record the simulation to MP4.")
parser.add_argument(
    "--video-name",
    type=str,
    default="joint_tracking",
    help="Video filename or stem (default: joint_tracking.mp4).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from tasks import airec  # noqa: F401  (registers AIREC tasks)

from joint_tracking_debug import _group_joint_indices, _short_joint_label, read_joint_tracking_from_env

# Series recorded per control step; each expands to one CSV column per joint.
_SERIES = (
    "q_target_raw",
    "q_cmd",
    "q_act",
    "q_current_measured",
    "q_vel",
    "q_error_cmd",
    "applied_torque",
    "computed_torque",
    "torque_limit_reached",
    "velocity_limit_reached",
    "position_limit_reached",
)


def _make_debug_env(task: str, env_cfg):
    """Subclass the registered env so actions are absolute joint targets [rad].

    Everything else (scene setup, PD controller, decimation, resets, joint-state
    access) is inherited unchanged from the existing environment class.
    """
    entry_point = gym.spec(task).entry_point
    module_name, class_name = entry_point.split(":")
    base_cls = getattr(importlib.import_module(module_name), class_name)

    class SineJointDebugEnv(base_cls):
        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            # ``actions`` = absolute joint targets [rad] for actuated DOFs;
            # bypass EMA / residual scaling used by the policy pipeline.
            self.last_action = self.joint_pos_cmd[:, self.actuated_dof_indices]
            self.prev_joint_pos_cmd[:] = self.joint_pos_cmd
            q_target = actions.to(device=self.device, dtype=self.joint_pos_cmd.dtype)
            self.joint_pos_cmd[:, self.actuated_dof_indices] = q_target
            self.joint_pos_policy[:, self.actuated_dof_indices] = q_target
            # Hard-limit clamp (same clamp _apply_action re-applies each substep).
            self._clamp_actuated_joint_pos_cmd_inplace()

    SineJointDebugEnv.__name__ = f"SineJointDebug_{class_name}"
    return SineJointDebugEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)


def _waveform_offset(t: float, duration: float) -> float:
    """Return normalized displacement; the caller applies the amplitude."""
    waveform = str(args_cli.waveform)
    frequency = float(args_cli.frequency_hz)
    if waveform == "sine":
        return math.sin(2.0 * math.pi * frequency * t)
    if waveform == "multisine":
        count = int(args_cli.multisine_components)
        # Equal-weight integer harmonics; division guarantees |offset| <= 1.
        return sum(
            math.sin(2.0 * math.pi * frequency * harmonic * t)
            for harmonic in range(1, count + 1)
        ) / count
    if waveform == "square":
        return 1.0 if math.sin(2.0 * math.pi * frequency * t) >= 0.0 else -1.0
    if waveform == "ramp":
        return min(max(t / max(duration, 1.0e-12), 0.0), 1.0)
    if waveform == "step":
        return 0.0 if t < float(args_cli.step_time_sec) else 1.0
    raise ValueError(f"unsupported waveform: {waveform!r}")


def _save_csv(path: str, joint_names: list[str], time_s: np.ndarray, data: dict[str, np.ndarray]) -> str:
    fields = ["step", "simulation_time"]
    for prefix in _SERIES:
        for name in joint_names:
            fields.append(f"{prefix}__{name}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for k in range(len(time_s)):
            row = {"step": k, "simulation_time": float(time_s[k])}
            for prefix in _SERIES:
                arr = data.get(prefix)
                if arr is None:
                    continue
                for j, name in enumerate(joint_names):
                    row[f"{prefix}__{name}"] = arr[k, j]
            writer.writerow(row)
    return path


def _plot_group(
    group: str,
    joint_names: list[str],
    time_s: np.ndarray,
    data: dict[str, np.ndarray],
    save_path: str,
) -> str | None:
    idxs = _group_joint_indices(joint_names, group)
    if not idxs:
        return None
    n = len(idxs)
    fig_w = max(2.6 * n, 12.0)
    fig, axes = plt.subplots(3, n, figsize=(fig_w, 10.0), sharex="col", squeeze=False, constrained_layout=True)
    fig.suptitle(
        f"{group.capitalize()} — {args_cli.waveform} joint tracking",
        fontsize=14,
        fontweight="bold",
    )

    for col, ji in enumerate(idxs):
        label = _short_joint_label(joint_names[ji]).replace("_", " ")
        ax_q, ax_tau, ax_err = axes[0, col], axes[1, col], axes[2, col]

        ax_q.plot(
            time_s,
            data["q_target_raw"][:, ji],
            color="#9467bd",
            linestyle=":",
            linewidth=1.4,
            label=f"q_target_raw ({args_cli.waveform})",
        )
        ax_q.plot(time_s, data["q_cmd"][:, ji], color="#1f77b4", linestyle="--", linewidth=1.4, label="q_cmd → Isaac")
        ax_q.plot(time_s, data["q_act"][:, ji], color="#ff7f0e", linestyle="-", linewidth=1.6, label="q_act (after step)")
        ax_q.plot(time_s, data["q_current_measured"][:, ji], color="#2ca02c", linestyle="-.", linewidth=1.1, label="q_current_measured")
        ax_q.set_title(f"{group} {label}", fontsize=10)
        ax_q.grid(True, alpha=0.35)

        applied = data.get("applied_torque")
        computed = data.get("computed_torque")
        if applied is not None:
            ax_tau.plot(time_s, applied[:, ji], color="#1f77b4", linewidth=1.5, label="applied torque")
        if computed is not None:
            ax_tau.plot(time_s, computed[:, ji], color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8, label="computed torque")
        ax_tau.grid(True, alpha=0.35)

        ax_err.plot(time_s, data["q_error_cmd"][:, ji], color="#d62728", linewidth=1.4, label="q_cmd − q_act")
        ax_err.axhline(0.0, color="k", linewidth=0.7, alpha=0.5)
        ax_err.grid(True, alpha=0.35)
        ax_err.set_xlabel("time (s)", fontsize=9)

        if col == 0:
            ax_q.set_ylabel("angle [rad]", fontsize=10)
            ax_q.legend(fontsize=7, loc="best")
            ax_tau.set_ylabel("torque [N·m]", fontsize=10)
            ax_tau.legend(fontsize=7, loc="best")
            ax_err.set_ylabel("tracking error [rad]", fontsize=10)
            ax_err.legend(fontsize=7, loc="best")

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# =====================================================================================
# Diagnostics mode (--diagnostics)
# =====================================================================================

# Verdict labels emitted per joint.
VERDICT_PASS = "PASS"
VERDICT_LIMIT = "LIMIT_SUSPECTED"
VERDICT_SIGN = "SIGN_MISMATCH"
VERDICT_OFFSET = "OFFSET_SUSPECTED"
VERDICT_TORQUE = "TORQUE_SATURATION"
VERDICT_LOCK = "LOCK_OR_CONSTRAINT_SUSPECTED"


def _control_mode_from_gains(stiffness: float, damping: float) -> str:
    """Derive control mode (no explicit 'mode' field exists in robot.data)."""
    if stiffness > 0.0:
        return "position (implicit PD)"
    if damping > 0.0:
        return "velocity"
    return "effort/none"


def _inspect_usd_joint(stage, prim_path: str) -> dict:
    """Best-effort USD introspection of one joint prim.

    Any attribute we cannot resolve is reported as the string '取得不可'.
    """
    out = {
        "usd_prim_path": prim_path,
        "joint_enabled": "取得不可",
        "excluded_from_articulation": "取得不可",
        "usd_drive_present": "取得不可",
        "usd_drive_type": "取得不可",
        "usd_drive_stiffness": "取得不可",
        "usd_drive_damping": "取得不可",
        "usd_drive_max_force": "取得不可",
        "mimic": "取得不可",
    }
    if stage is None or not prim_path:
        return out
    try:
        from pxr import PhysxSchema, UsdPhysics  # noqa: F401
    except Exception:
        return out
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return out
    except Exception:
        return out

    def _attr(name):
        try:
            a = prim.GetAttribute(name)
            if a and a.IsValid():
                v = a.Get()
                return v
        except Exception:
            return None
        return None

    je = _attr("physics:jointEnabled")
    out["joint_enabled"] = bool(je) if je is not None else True  # default enabled
    exc = _attr("physics:excludeFromArticulation")
    out["excluded_from_articulation"] = bool(exc) if exc is not None else False

    # Angular / linear drive API (multiple-apply schema).
    try:
        from pxr import UsdPhysics

        drive = None
        for token in ("angular", "linear"):
            try:
                d = UsdPhysics.DriveAPI.Get(prim, token)
                if d and d.GetStiffnessAttr():
                    drive = (token, d)
                    break
            except Exception:
                continue
        if drive is not None:
            token, d = drive
            out["usd_drive_present"] = True
            try:
                out["usd_drive_stiffness"] = float(d.GetStiffnessAttr().Get())
            except Exception:
                pass
            try:
                out["usd_drive_damping"] = float(d.GetDampingAttr().Get())
            except Exception:
                pass
            try:
                out["usd_drive_max_force"] = float(d.GetMaxForceAttr().Get())
            except Exception:
                pass
            try:
                out["usd_drive_type"] = str(d.GetTypeAttr().Get())
            except Exception:
                pass
        else:
            out["usd_drive_present"] = False
    except Exception:
        pass

    # Mimic joint (PhysX).
    try:
        from pxr import PhysxSchema

        has_mimic = False
        for axis in ("rotX", "rotY", "rotZ", "transX", "transY", "transZ", ""):
            try:
                if axis:
                    has_mimic = has_mimic or bool(prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis))
                else:
                    has_mimic = has_mimic or bool(prim.HasAPI(PhysxSchema.PhysxMimicJointAPI))
            except Exception:
                continue
        out["mimic"] = bool(has_mimic)
    except Exception:
        pass

    return out


def _dof_prim_path(robot, dof_index: int) -> str:
    """USD prim path for a DOF (authoritative DOF->USD mapping), or '取得不可'."""
    try:
        return str(robot.root_physx_view.dof_paths[0][dof_index])
    except Exception:
        return "取得不可"


def _run_diagnostics(raw_env, env_cfg, out_dir: str, eid: int) -> None:
    from isaaclab.sim import SimulationContext

    robot = raw_env.robot
    data = robot.data
    all_names = list(robot.joint_names)
    actuated = list(raw_env.actuated_dof_indices)
    actuated_set = set(actuated)
    name_to_actcol = {all_names[i]: k for k, i in enumerate(actuated)}
    try:
        stage = SimulationContext.instance().stage
    except Exception:
        stage = None

    def _row_scalar(t, j):
        try:
            v = t[eid, j] if t.ndim > 1 else t[j]
            return float(v.detach().cpu().item())
        except Exception:
            return float("nan")

    stiffness = getattr(data, "joint_stiffness", None)
    damping = getattr(data, "joint_damping", None)
    effort_lim = getattr(data, "joint_effort_limits", None)
    vel_lim = getattr(data, "joint_vel_limits", None)
    pos_lim = getattr(data, "joint_pos_limits", None)
    default_pos = getattr(data, "default_joint_pos", None)

    # -------- 1. joint_index_mapping.txt --------
    expected_order = (
        [f"torso_joint_{i}" for i in range(1, 4)]
        + [f"left_arm_joint_{i}" for i in range(1, 8)]
        + [f"right_arm_joint_{i}" for i in range(1, 8)]
    )
    mapping_path = os.path.join(out_dir, "joint_index_mapping.txt")
    with open(mapping_path, "w") as f:
        f.write("# Joint name / DOF index / articulation index / USD prim path\n")
        f.write("# DOF index = index into robot.joint_names (PhysX ArticulationView DOF order).\n")
        f.write("# articulation_index is identical to DOF index for this 1-DOF-per-joint articulation.\n\n")
        f.write(f"{'dof_idx':>7}  {'art_idx':>7}  {'actuated':>8}  {'joint_name':32s}  usd_prim_path\n")
        for dof_idx, name in enumerate(all_names):
            act = "yes" if dof_idx in actuated_set else "no"
            f.write(f"{dof_idx:7d}  {dof_idx:7d}  {act:>8}  {name:32s}  {_dof_prim_path(robot, dof_idx)}\n")
        f.write("\n# Expected order check (torso 1-3, left arm 1-7, right arm 1-7):\n")
        for name in expected_order:
            if name in all_names:
                f.write(f"  {name:32s} present at DOF index {all_names.index(name)}\n")
            else:
                f.write(f"  {name:32s} MISSING (取得不可)\n")
    print(f"[diag] wrote {mapping_path}")

    # -------- 2/3. joint_config_summary.csv --------
    diag_joints = [n for n in expected_order if n in all_names]
    diag_dof = [all_names.index(n) for n in diag_joints]

    config_path = os.path.join(out_dir, "joint_config_summary.csv")
    config_fields = [
        "joint_name", "dof_index", "actuated", "control_mode",
        "stiffness", "damping", "max_effort_Nm", "max_velocity_rad_s",
        "lower_limit_rad", "upper_limit_rad", "init_angle_rad",
        "dist_to_lower_rad", "dist_to_upper_rad",
        "usd_drive_present", "usd_drive_type", "usd_drive_max_force",
        "joint_enabled", "mimic", "excluded_from_articulation", "usd_prim_path",
    ]
    config_rows = []
    for name, dof in zip(diag_joints, diag_dof):
        stf = _row_scalar(stiffness, dof) if stiffness is not None else float("nan")
        dmp = _row_scalar(damping, dof) if damping is not None else float("nan")
        eff = _row_scalar(effort_lim, dof) if effort_lim is not None else float("nan")
        vel = _row_scalar(vel_lim, dof) if vel_lim is not None else float("nan")
        lo = _row_scalar(pos_lim[..., 0], dof) if pos_lim is not None else float("nan")
        hi = _row_scalar(pos_lim[..., 1], dof) if pos_lim is not None else float("nan")
        init = _row_scalar(default_pos, dof) if default_pos is not None else float("nan")
        usd = _inspect_usd_joint(stage, _dof_prim_path(robot, dof))
        row = {
            "joint_name": name,
            "dof_index": dof,
            "actuated": "yes" if dof in actuated_set else "no",
            "control_mode": _control_mode_from_gains(stf, dmp),
            "stiffness": stf,
            "damping": dmp,
            "max_effort_Nm": eff,
            "max_velocity_rad_s": vel,
            "lower_limit_rad": lo,
            "upper_limit_rad": hi,
            "init_angle_rad": init,
            "dist_to_lower_rad": init - lo,
            "dist_to_upper_rad": hi - init,
            "usd_drive_present": usd["usd_drive_present"],
            "usd_drive_type": usd["usd_drive_type"],
            "usd_drive_max_force": usd["usd_drive_max_force"],
            "joint_enabled": usd["joint_enabled"],
            "mimic": usd["mimic"],
            "excluded_from_articulation": usd["excluded_from_articulation"],
            "usd_prim_path": usd["usd_prim_path"],
        }
        config_rows.append(row)
    with open(config_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=config_fields)
        writer.writeheader()
        writer.writerows(config_rows)
    print(f"[diag] wrote {config_path}")

    # joint 1-4 vs 5-7 comparison table (stdout).
    def _group_gain(rows, side, lo_i, hi_i):
        vals = [r for r in rows if r["joint_name"].startswith(f"{side}_arm_joint_")
                and lo_i <= int(r["joint_name"].split("_")[-1]) <= hi_i]
        return vals
    print("\n[diag] drive-setting comparison (arm joint 1-4 vs 5-7):")
    print(f"  {'group':22s} {'stiffness':>12} {'damping':>10} {'max_effort':>11} {'max_vel':>9}")
    for side in ("left", "right"):
        for lo_i, hi_i in ((1, 4), (5, 7)):
            g = _group_gain(config_rows, side, lo_i, hi_i)
            if not g:
                continue
            s = np.nanmean([r["stiffness"] for r in g])
            d = np.nanmean([r["damping"] for r in g])
            e = np.nanmean([r["max_effort_Nm"] for r in g])
            v = np.nanmean([r["max_velocity_rad_s"] for r in g])
            print(f"  {side+' arm '+str(lo_i)+'-'+str(hi_i):22s} {s:12.2f} {d:10.2f} {e:11.2f} {v:9.3f}")

    # -------- 4/5/6. per-joint +deg / -deg isolated step tests --------
    control_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    settle_steps = max(2, int(round(float(args_cli.diag_settle_sec) / control_dt)))
    step_rad = math.radians(float(args_cli.diag_step_deg))
    lower_t = raw_env.robot_hard_dof_lower_limits[actuated]
    upper_t = raw_env.robot_hard_dof_upper_limits[actuated]

    # Command center = default pose in actuated order.
    q_center = data.default_joint_pos[0, actuated].clone()

    limit_tol = math.radians(1.0)  # 1 deg near-limit band
    deadband = math.radians(0.5)   # motion deadband

    diag_rows = []
    step_traces: dict[str, dict] = {}

    def _hold_center(n):
        cmd = q_center.unsqueeze(0).clone()
        for _ in range(n):
            raw_env.step(cmd)

    with torch.inference_mode():
        for name, dof in zip(diag_joints, diag_dof):
            col = name_to_actcol[name]
            eff = _row_scalar(effort_lim, dof) if effort_lim is not None else float("nan")
            lo = _row_scalar(pos_lim[..., 0], dof) if pos_lim is not None else float("nan")
            hi = _row_scalar(pos_lim[..., 1], dof) if pos_lim is not None else float("nan")
            usd = _inspect_usd_joint(stage, _dof_prim_path(robot, dof))
            per_dir = {}
            trace = {"time": np.arange(settle_steps) * control_dt}

            for direction, tag in ((+1.0, "plus"), (-1.0, "minus")):
                # Return to center first for a clean baseline.
                _hold_center(settle_steps)
                baseline_qact = _row_scalar(data.joint_pos, dof)

                cmd = q_center.unsqueeze(0).clone()
                target = float(q_center[col].item()) + direction * step_rad
                target_clamped = float(min(max(target, float(lower_t[col].item())), float(upper_t[col].item())))
                cmd[0, col] = target_clamped

                qcmd_s = np.zeros(settle_steps)
                qact_s = np.zeros(settle_steps)
                applied_s = np.zeros(settle_steps)
                computed_s = np.zeros(settle_steps)
                for s in range(settle_steps):
                    raw_env.step(cmd)
                    qcmd_s[s] = _row_scalar(raw_env.joint_pos_cmd, dof)
                    qact_s[s] = _row_scalar(data.joint_pos, dof)
                    applied_s[s] = _row_scalar(getattr(data, "applied_torque", None), dof) if getattr(data, "applied_torque", None) is not None else float("nan")
                    computed_s[s] = _row_scalar(getattr(data, "computed_torque", None), dof) if getattr(data, "computed_torque", None) is not None else float("nan")

                tail = slice(max(0, settle_steps - max(3, settle_steps // 5)), settle_steps)
                cmd_delta = target_clamped - float(q_center[col].item())
                act_delta = float(np.mean(qact_s[tail])) - baseline_qact
                offset_start = qcmd_s[0] - qact_s[0]
                offset_steady = float(np.mean(qcmd_s[tail] - qact_s[tail]))
                if np.isfinite(eff) and eff > 0.0:
                    sat_rate = float(np.mean(np.abs(applied_s) >= 0.999 * eff))
                else:
                    sat_rate = float("nan")
                requested = direction * step_rad
                clamp_loss = abs(requested) - abs(cmd_delta)  # how much command was clipped by limit

                per_dir[tag] = {
                    "cmd_delta": cmd_delta,
                    "act_delta": act_delta,
                    "offset_start": offset_start,
                    "offset_steady": offset_steady,
                    "sat_rate": sat_rate,
                    "target_clamped": target_clamped,
                    "clamp_loss": clamp_loss,
                    "baseline_qact": baseline_qact,
                    "final_qact": float(np.mean(qact_s[tail])),
                }
                trace[f"{tag}_q_cmd"] = qcmd_s
                trace[f"{tag}_q_act"] = qact_s
                trace[f"{tag}_applied"] = applied_s
                trace[f"{tag}_computed"] = computed_s

            step_traces[name] = trace

            # ----- verdict -----
            flags = []
            # sign
            sign_mismatch = False
            for tag in ("plus", "minus"):
                pd = per_dir[tag]
                if abs(pd["cmd_delta"]) > deadband and abs(pd["act_delta"]) > deadband:
                    if math.copysign(1.0, pd["cmd_delta"]) != math.copysign(1.0, pd["act_delta"]):
                        sign_mismatch = True
            # limit
            near_limit = False
            for tag in ("plus", "minus"):
                pd = per_dir[tag]
                if pd["clamp_loss"] > limit_tol:
                    near_limit = True
                fq = pd["final_qact"]
                if (np.isfinite(lo) and abs(fq - lo) < limit_tol) or (np.isfinite(hi) and abs(fq - hi) < limit_tol):
                    near_limit = True
            # saturation
            max_sat = np.nanmax([per_dir["plus"]["sat_rate"], per_dir["minus"]["sat_rate"]])
            max_abs_offset = max(abs(per_dir["plus"]["offset_steady"]), abs(per_dir["minus"]["offset_steady"]))
            saturated = np.isfinite(max_sat) and max_sat >= 0.5 and max_abs_offset > deadband
            # structural lock
            locked = False
            if usd["joint_enabled"] is False:
                locked = True
            if usd["excluded_from_articulation"] is True:
                locked = True
            if usd["mimic"] is True:
                locked = True
            stf = _row_scalar(stiffness, dof) if stiffness is not None else float("nan")
            dmp = _row_scalar(damping, dof) if damping is not None else float("nan")
            no_motion = all(abs(per_dir[t]["act_delta"]) < deadband for t in ("plus", "minus"))
            near_zero_torque = np.nanmax(np.abs([
                np.nanmax(np.abs(step_traces[name]["plus_applied"])),
                np.nanmax(np.abs(step_traces[name]["minus_applied"])),
            ])) < 1.0
            if (stf <= 0.0 and dmp <= 0.0) or (no_motion and near_zero_torque):
                locked = True
            # offset (~90/180 or generic large steady offset not explained by saturation)
            offset_suspect = (not saturated) and (max_abs_offset > math.radians(20.0))

            # precedence: LOCK -> LIMIT -> TORQUE -> SIGN -> OFFSET -> PASS
            # Torque saturation is reported before sign mismatch because gravity +
            # effort clipping often produces unidirectional collapse that looks like
            # a sign error (especially on arm joints 5-7).
            if locked:
                verdict = VERDICT_LOCK
            elif near_limit:
                verdict = VERDICT_LIMIT
            elif saturated:
                verdict = VERDICT_TORQUE
            elif sign_mismatch:
                verdict = VERDICT_SIGN
            elif offset_suspect:
                verdict = VERDICT_OFFSET
            else:
                verdict = VERDICT_PASS

            if sign_mismatch:
                flags.append("sign_mismatch")
            if near_limit:
                flags.append("near_limit")
            if saturated:
                flags.append(f"saturation({max_sat:.0%})")
            for special, lbl in ((math.pi / 2, "~90deg_offset"), (math.pi, "~180deg_offset")):
                if abs(max_abs_offset - special) < math.radians(12.0):
                    flags.append(lbl)
            if offset_suspect and not flags:
                flags.append("steady_offset")

            diag_rows.append({
                "joint_name": name,
                "dof_index": dof,
                "stiffness": stf,
                "damping": dmp,
                "max_effort_Nm": eff,
                "lower_limit_rad": lo,
                "upper_limit_rad": hi,
                "init_angle_rad": float(q_center[col].item()),
                "plus_cmd_delta_rad": per_dir["plus"]["cmd_delta"],
                "plus_act_delta_rad": per_dir["plus"]["act_delta"],
                "minus_cmd_delta_rad": per_dir["minus"]["cmd_delta"],
                "minus_act_delta_rad": per_dir["minus"]["act_delta"],
                "offset_start_rad": per_dir["plus"]["offset_start"],
                "offset_steady_rad": per_dir["plus"]["offset_steady"],
                "offset_steady_minus_rad": per_dir["minus"]["offset_steady"],
                "max_saturation_rate": max_sat,
                "sign_ok": not sign_mismatch,
                "joint_enabled": usd["joint_enabled"],
                "mimic": usd["mimic"],
                "excluded_from_articulation": usd["excluded_from_articulation"],
                "flags": ";".join(flags),
                "verdict": verdict,
            })

    diag_path = os.path.join(out_dir, "joint_diagnostics.csv")
    diag_fields = list(diag_rows[0].keys()) if diag_rows else ["joint_name", "verdict"]
    with open(diag_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diag_fields)
        writer.writeheader()
        writer.writerows(diag_rows)
    print(f"[diag] wrote {diag_path}")

    # -------- 7. per-joint step response plots --------
    if not args_cli.no_plots:
        for name, trace in step_traces.items():
            fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
            fig.suptitle(f"{name} — isolated ±{args_cli.diag_step_deg:.0f} deg step response", fontsize=13, fontweight="bold")
            for c, tag in enumerate(("plus", "minus")):
                ax_q, ax_t = axes[0, c], axes[1, c]
                ax_q.plot(trace["time"], trace[f"{tag}_q_cmd"], color="#1f77b4", linestyle="--", label="q_cmd")
                ax_q.plot(trace["time"], trace[f"{tag}_q_act"], color="#ff7f0e", label="q_act")
                ax_q.set_title(f"{'+' if tag == 'plus' else '-'}{args_cli.diag_step_deg:.0f} deg", fontsize=10)
                ax_q.grid(True, alpha=0.35)
                ax_t.plot(trace["time"], trace[f"{tag}_applied"], color="#1f77b4", label="applied τ")
                ax_t.plot(trace["time"], trace[f"{tag}_computed"], color="#d62728", linestyle="--", alpha=0.8, label="computed τ")
                ax_t.grid(True, alpha=0.35)
                ax_t.set_xlabel("time (s)")
                if c == 0:
                    ax_q.set_ylabel("angle [rad]")
                    ax_q.legend(fontsize=8)
                    ax_t.set_ylabel("torque [N·m]")
                    ax_t.legend(fontsize=8)
            path = os.path.join(out_dir, f"step_response_{name}.png")
            fig.savefig(path, dpi=130)
            plt.close(fig)
        print(f"[diag] wrote per-joint step-response plots to {out_dir}")

    # -------- verdict summary (stdout) --------
    print("\n[diag] per-joint verdict:")
    print(f"  {'joint_name':22s} {'verdict':30s} flags")
    for r in diag_rows:
        print(f"  {r['joint_name']:22s} {r['verdict']:30s} {r['flags']}")


def main() -> None:
    if not args_cli.diagnostics:
        if float(args_cli.duration_sec) <= 0.0:
            parser.error("--duration-sec must be positive")
        if float(args_cli.amplitude_deg) < 0.0:
            parser.error("--amplitude-deg must be non-negative")
        if args_cli.waveform in ("sine", "multisine", "square") and float(args_cli.frequency_hz) <= 0.0:
            parser.error("--frequency-hz must be positive for sine, multisine and square")
        if int(args_cli.multisine_components) < 1:
            parser.error("--multisine-components must be at least 1")
    else:
        if float(args_cli.diag_settle_sec) <= 0.0:
            parser.error("--diag-settle-sec must be positive")
        if float(args_cli.diag_step_deg) <= 0.0:
            parser.error("--diag-step-deg must be positive")

    out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Same env cfg source as play.py (robot cfg, actuated joints, dt, decimation).
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene_mode = "free_space"
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    control_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    # Avoid mid-run timeout resets: one continuous "episode" spans the whole run.
    if args_cli.diagnostics:
        n_dirs = 2  # +deg and -deg
        n_settle = max(2, int(round(float(args_cli.diag_settle_sec) / control_dt)))
        n_diag_joints = 17  # torso 3 + arms 14 (upper bound)
        # Each joint test: return-to-center + directed hold, times both directions.
        est_steps = 2 * n_dirs * n_settle * n_diag_joints
        env_cfg.episode_length_s = (est_steps + 50) * control_dt
    else:
        env_cfg.episode_length_s = float(args_cli.duration_sec) + 10.0 * control_dt
    env_cfg.debug_joint_cmd_vs_actual = False
    if args_cli.disable_self_collision:
        env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = False

    eid = int(args_cli.env_id)
    if args_cli.diagnostics:
        raw_env = _make_debug_env(args_cli.task, env_cfg)
        raw_env.reset()
        print(f"[diag] task={args_cli.task} scene_mode=free_space num_envs={raw_env.num_envs}")
        print(f"[diag] control_dt={control_dt:.6g} s, settle={float(args_cli.diag_settle_sec)} s, step=±{args_cli.diag_step_deg} deg")
        print(f"[diag] output dir: {out_dir}")
        _run_diagnostics(raw_env, env_cfg, out_dir, eid)
        raw_env.close()
        return

    amplitude_rad = math.radians(float(args_cli.amplitude_deg))
    frequency_hz = float(args_cli.frequency_hz)
    n_steps = max(1, int(round(float(args_cli.duration_sec) / control_dt)))
    eid = int(args_cli.env_id)

    raw_env = _make_debug_env(args_cli.task, env_cfg)
    env = raw_env
    video_stem = os.path.splitext(os.path.basename(str(args_cli.video_name)))[0] or "joint_tracking"
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=out_dir,
            step_trigger=lambda step: step == 0,
            video_length=n_steps,
            name_prefix=video_stem,
            disable_logger=True,
        )

    env.reset()
    robot = raw_env.robot
    actuated = list(raw_env.actuated_dof_indices)
    n_act = len(actuated)
    # Waveform center = joint pose right after reset (default pose).
    q_center = robot.data.joint_pos[:, actuated].clone()
    lower = raw_env.robot_hard_dof_lower_limits[actuated]
    upper = raw_env.robot_hard_dof_upper_limits[actuated]

    # Recording order = read_joint_tracking_from_env order (torso first, then arms).
    snap0 = read_joint_tracking_from_env(raw_env, env_id=eid)
    joint_names = list(snap0["joint_names"])
    name_to_cmd_col = {robot.joint_names[i]: k for k, i in enumerate(actuated)}
    cmd_cols = [name_to_cmd_col[n] for n in joint_names]  # tracking joints ⊆ actuated

    print(f"[waveform debug] task={args_cli.task} scene_mode=free_space num_envs={raw_env.num_envs}")
    print(
        f"[waveform debug] waveform={args_cli.waveform}, amplitude={args_cli.amplitude_deg} deg, "
        f"frequency={frequency_hz} Hz, duration={args_cli.duration_sec} s"
    )
    print(f"[waveform debug] control_dt={control_dt:.6g} s (physics_dt={float(env_cfg.sim.dt):.6g} x decimation={int(env_cfg.decimation)}) -> {n_steps} control steps")
    print(f"[waveform debug] {n_act} actuated joints, recording {len(joint_names)}: {joint_names}")
    print(f"[waveform debug] output dir: {out_dir}")
    if args_cli.video:
        print(f"[waveform debug] recording video: {os.path.join(out_dir, video_stem + '.mp4')}")

    n_rec = len(joint_names)
    time_s = np.zeros(n_steps, dtype=np.float64)
    data = {key: np.zeros((n_steps, n_rec), dtype=np.float64) for key in _SERIES}

    with torch.inference_mode():
        for k in range(n_steps):
            t = k * control_dt
            offset = _waveform_offset(t, float(args_cli.duration_sec))
            q_target_raw = q_center + amplitude_rad * offset
            q_cmd = torch.clamp(q_target_raw, lower, upper)

            # Measured joint angle available at command time (before this step).
            q_meas = robot.data.joint_pos[eid, actuated].detach().cpu().numpy()

            env.step(q_cmd)

            snap = read_joint_tracking_from_env(raw_env, env_id=eid)
            time_s[k] = t
            raw_np = q_target_raw[eid].detach().cpu().numpy()
            data["q_target_raw"][k] = raw_np[cmd_cols]
            data["q_current_measured"][k] = q_meas[cmd_cols]
            data["q_cmd"][k] = np.asarray(snap["q_cmd"])
            data["q_act"][k] = np.asarray(snap["q_act"])
            data["q_vel"][k] = np.asarray(snap["q_vel"])
            data["q_error_cmd"][k] = np.asarray(snap["q_err"])  # q_cmd - q_act
            if snap.get("applied_torque") is not None:
                data["applied_torque"][k] = np.asarray(snap["applied_torque"])
            if snap.get("computed_torque") is not None:
                data["computed_torque"][k] = np.asarray(snap["computed_torque"])
            for key in ("torque_limit_reached", "velocity_limit_reached", "position_limit_reached"):
                if snap.get(key) is not None:
                    data[key][k] = np.asarray(snap[key], dtype=np.float64)

            if (k + 1) % max(1, n_steps // 10) == 0:
                err = data["q_error_cmd"][k]
                print(f"[waveform debug] step {k + 1}/{n_steps} t={t:.2f}s max|q_cmd-q_act|={np.abs(err).max():.4f} rad")

            if not simulation_app.is_running():
                n_steps = k + 1
                time_s = time_s[:n_steps]
                data = {key: arr[:n_steps] for key, arr in data.items()}
                print("[waveform debug] simulation app closed early; truncating recording")
                break

    # ---- save ----
    output_stem = f"{args_cli.waveform}_joint_tracking"
    npz_path = os.path.join(out_dir, f"{output_stem}.npz")
    np.savez_compressed(
        npz_path,
        joint_names=np.asarray(joint_names),
        simulation_time=time_s,
        control_dt=control_dt,
        amplitude_rad=amplitude_rad,
        frequency_hz=frequency_hz,
        waveform=str(args_cli.waveform),
        multisine_components=int(args_cli.multisine_components),
        step_time_sec=float(args_cli.step_time_sec),
        **data,
    )
    csv_path = _save_csv(os.path.join(out_dir, f"{output_stem}.csv"), joint_names, time_s, data)
    print(f"[waveform debug] saved: {npz_path}")
    print(f"[waveform debug] saved: {csv_path}")

    # ---- per-joint error summary ----
    err = data["q_error_cmd"]
    rms = np.sqrt(np.mean(err**2, axis=0))
    mx = np.abs(err).max(axis=0)
    print("[waveform debug] tracking error (q_cmd - q_act) per joint:")
    for j, name in enumerate(joint_names):
        flags = []
        if data["torque_limit_reached"][:, j].any():
            flags.append("torque_limit")
        if data["velocity_limit_reached"][:, j].any():
            flags.append("velocity_limit")
        if data["position_limit_reached"][:, j].any():
            flags.append("position_limit")
        flag_txt = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {name:32s} RMS={rms[j]:+.5f} rad  max={mx[j]:+.5f} rad{flag_txt}")

    # ---- plots (left / right / torso) ----
    if not args_cli.no_plots:
        for group in ("right", "left", "torso"):
            path = _plot_group(
                group,
                joint_names,
                time_s,
                data,
                os.path.join(out_dir, f"{group}_{args_cli.waveform}_tracking.png"),
            )
            if path:
                print(f"[waveform debug] plot: {path}")

    env.close()
    if args_cli.video:
        final_video = os.path.join(out_dir, f"{video_stem}.mp4")
        candidates = [
            os.path.join(out_dir, name)
            for name in os.listdir(out_dir)
            if name.startswith(f"{video_stem}-") and name.endswith(".mp4")
        ]
        if candidates:
            recorded_video = max(candidates, key=os.path.getmtime)
            os.replace(recorded_video, final_video)
            print(f"[waveform debug] video: {final_video}")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        raise
    finally:
        print("CLOSING")
        simulation_app.close()
