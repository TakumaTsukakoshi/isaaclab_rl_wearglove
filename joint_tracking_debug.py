"""Record and plot joint targets / measured state and torque diagnostics.

Naming (joint space, radians unless noted)::

    action          raw policy output ∈ [-1, 1]
    q_policy        scale(action) — policy target **before** EMA
    q_cmd           after EMA + hard clamp — sent to Isaac via set_joint_position_target
    q_act / q_meas  robot.data.joint_pos — measured sim state (not a command)

Typical usage (via ``play.py``)::

    python play.py --task ... --checkpoint ... --num_envs 1 \\
        --record-joint-tracking joint_tracking_plots --joint-tracking-no-plots

    Per episode this also writes ``episode_XXX_q_cmd.npy`` (``(T, N)`` rad) and
    ``episode_XXX_q_cmd.npz`` (``q_cmd``, ``joint_names``, ``dt``, ``simulation_time``).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class JointTrackingStep:
    step: int
    global_step: int
    joint_names: list[str]
    action: list[float]
    q_cmd: list[float]
    q_act: list[float]
    q_err: list[float]
    q_vel: list[float]
    q_policy: list[float] | None = None  # scaled policy, before EMA
    simulation_time: float = 0.0
    q_actual: list[float] | None = None  # explicit alias of q_act
    q_error_cmd: list[float] | None = None  # q_cmd - q_actual
    q_error_policy: list[float] | None = None  # q_policy - q_actual
    applied_torque: list[float] | None = None
    computed_torque: list[float] | None = None
    torque_err: list[float] | None = None  # computed - applied
    torque_limit_reached: list[bool] | None = None
    velocity_limit_reached: list[bool] | None = None
    position_limit_reached: list[bool] | None = None


@dataclass
class JointTrackingEpisodeTrace:
    episode_index: int
    env_id: int
    joint_names: list[str] = field(default_factory=list)
    steps: list[JointTrackingStep] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "env_id": self.env_id,
            "joint_names": self.joint_names,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "steps": [asdict(s) for s in self.steps],
        }


def _to_list(t: torch.Tensor | None) -> list[float] | None:
    if t is None:
        return None
    return [float(x) for x in t.detach().cpu().reshape(-1).tolist()]


def _to_bool_list(t: torch.Tensor | None) -> list[bool] | None:
    if t is None:
        return None
    return [bool(x) for x in t.detach().cpu().reshape(-1).tolist()]


def _unwrap(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _tracking_joint_indices(env: Any) -> list[int]:
    """Actuated DOFs plus torso joints (for viz), unique and ordered."""
    unwrapped = _unwrap(env)
    robot_names = list(unwrapped.robot.joint_names)
    name_to_i = {n: i for i, n in enumerate(robot_names)}

    ordered: list[int] = []
    seen: set[int] = set()

    # Prefer torso first so plots align with reference PDFs (torso then arms).
    torso_names = list(getattr(getattr(unwrapped, "cfg", None), "actuated_torso_joints", None) or [])
    if not torso_names:
        torso_names = [n for n in robot_names if n.startswith("torso_joint_")]
    for n in torso_names:
        if n in name_to_i and name_to_i[n] not in seen:
            ordered.append(name_to_i[n])
            seen.add(name_to_i[n])

    for i in unwrapped.actuated_dof_indices:
        ii = int(i)
        if ii not in seen:
            ordered.append(ii)
            seen.add(ii)
    return ordered


def read_joint_tracking_from_env(env: Any, env_id: int = 0) -> dict[str, Any]:
    """Snapshot policy / cmd / measured joint state and torques for one env."""
    unwrapped = _unwrap(env)
    if not hasattr(unwrapped, "actuated_dof_indices") or not hasattr(unwrapped, "joint_pos_cmd"):
        raise AttributeError("env missing actuated_dof_indices / joint_pos_cmd (not an AIREC joint env?)")

    idx = _tracking_joint_indices(unwrapped)
    names = [unwrapped.robot.joint_names[i] for i in idx]
    e = int(env_id)

    q_cmd = unwrapped.joint_pos_cmd[e, idx]
    q_act = unwrapped.robot.data.joint_pos[e, idx]
    q_vel = unwrapped.robot.data.joint_vel[e, idx]
    q_err = q_cmd - q_act

    q_policy = None
    if hasattr(unwrapped, "joint_pos_policy") and unwrapped.joint_pos_policy is not None:
        q_policy = unwrapped.joint_pos_policy[e, idx]
    q_error_policy = q_policy - q_act if q_policy is not None else None

    # Policy action only spans actuated DOFs; pad non-actuated (e.g. torso) with 0.
    # Prefer raw (pre-EMA) actions when available.
    action_full = [0.0] * len(names)
    act_src = getattr(unwrapped, "_raw_actions", None)
    if act_src is None:
        act_src = getattr(unwrapped, "actions", None)
    if act_src is not None:
        act_list = act_src[e].detach().cpu().tolist()
        actuated_set = {int(i) for i in unwrapped.actuated_dof_indices}
        act_pos = 0
        for j, ji in enumerate(idx):
            if ji in actuated_set and act_pos < len(act_list):
                action_full[j] = float(act_list[act_pos])
                act_pos += 1

    applied = computed = torque_err = None
    data = unwrapped.robot.data
    if hasattr(data, "applied_torque") and data.applied_torque is not None:
        applied = data.applied_torque[e, idx]
    if hasattr(data, "computed_torque") and data.computed_torque is not None:
        computed = data.computed_torque[e, idx]
    if applied is not None and computed is not None:
        torque_err = computed - applied

    effort_limits = getattr(data, "joint_effort_limits", None)
    velocity_limits = getattr(data, "joint_vel_limits", None)
    position_limits = getattr(data, "joint_pos_limits", None)
    torque_limit_reached = velocity_limit_reached = position_limit_reached = None
    if applied is not None and effort_limits is not None:
        effort = effort_limits[e, idx].abs()
        torque_limit_reached = (
            torch.isfinite(effort) & (effort > 0.0) & (applied.abs() >= 0.999 * effort)
        )
    if velocity_limits is not None:
        velocity = velocity_limits[e, idx].abs()
        velocity_limit_reached = (
            torch.isfinite(velocity) & (velocity > 0.0) & (q_vel.abs() >= 0.999 * velocity)
        )
    if position_limits is not None:
        lower = position_limits[e, idx, 0]
        upper = position_limits[e, idx, 1]
        tolerance = torch.maximum(
            torch.full_like(lower, 1.0e-4), 1.0e-3 * (upper - lower).abs()
        )
        position_limit_reached = (q_act <= lower + tolerance) | (q_act >= upper - tolerance)

    step_dt = float(
        getattr(
            unwrapped,
            "step_dt",
            float(unwrapped.cfg.sim.dt) * int(unwrapped.cfg.decimation),
        )
    )
    simulation_time = float(getattr(unwrapped, "common_step_counter", 0)) * step_dt

    return {
        "env_id": e,
        "joint_names": names,
        "simulation_time": simulation_time,
        "action": action_full,
        "q_policy": _to_list(q_policy),
        "q_cmd": _to_list(q_cmd) or [],
        "q_act": _to_list(q_act) or [],
        "q_actual": _to_list(q_act) or [],
        "q_err": _to_list(q_err) or [],
        "q_error_cmd": _to_list(q_err) or [],
        "q_error_policy": _to_list(q_error_policy),
        "q_vel": _to_list(q_vel) or [],
        "applied_torque": _to_list(applied),
        "computed_torque": _to_list(computed),
        "torque_err": _to_list(torque_err),
        "torque_limit_reached": _to_bool_list(torque_limit_reached),
        "velocity_limit_reached": _to_bool_list(velocity_limit_reached),
        "position_limit_reached": _to_bool_list(position_limit_reached),
    }


def _stack_field(trace: JointTrackingEpisodeTrace, key: str) -> np.ndarray | None:
    rows = []
    for s in trace.steps:
        val = getattr(s, key)
        if val is None:
            return None
        rows.append(val)
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def _group_joint_indices(joint_names: list[str], group: str) -> list[int]:
    """Indices of joints for ``left`` / ``right`` arm or ``torso``, sorted by joint number."""
    group = group.lower().strip()
    if group in ("left", "right"):
        prefix = f"{group}_arm_joint_"
    elif group == "torso":
        prefix = "torso_joint_"
    else:
        raise ValueError(f"group must be 'left', 'right', or 'torso', got {group!r}")
    matched: list[tuple[int, int]] = []
    for i, name in enumerate(joint_names):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            try:
                num = int(suffix)
            except ValueError:
                num = i
            matched.append((num, i))
    matched.sort(key=lambda t: t[0])
    return [i for _, i in matched]


def _arm_joint_indices(joint_names: list[str], side: str) -> list[int]:
    """Backward-compatible alias for :func:`_group_joint_indices`."""
    return _group_joint_indices(joint_names, side)


def _short_joint_label(name: str) -> str:
    """``left_arm_joint_3`` → ``joint_3``; ``torso_joint_1`` → ``joint_1``."""
    for prefix in ("left_arm_", "right_arm_", "torso_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def plot_arm_joint_states(
    trace: JointTrackingEpisodeTrace,
    save_path: str,
    *,
    side: str,
    dt: float | None = 0.1,
    title: str | None = None,
    angle_unit: str = "rad",
) -> str:
    """PDF-style per-joint panels: torque (actual) on top, angles on bottom.

    Bottom row compares:
    - ``q_policy`` — scaled policy (before EMA), if recorded
    - ``q_cmd`` — after EMA+clamp, sent to Isaac
    - ``q_act`` — measured ``joint_pos``
    """
    if not trace.steps:
        raise ValueError("empty joint tracking episode")
    side = side.lower().strip()
    if side not in ("left", "right", "torso"):
        raise ValueError(f"side must be 'left', 'right', or 'torso', got {side!r}")
    angle_unit = angle_unit.lower().strip()
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got {angle_unit!r}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    names = list(trace.joint_names or trace.steps[0].joint_names)
    idxs = _group_joint_indices(names, side)
    if not idxs:
        raise ValueError(f"no {side} joints found in {names}")

    arm_names = [names[i] for i in idxs]
    n = len(arm_names)
    q_cmd_all = _stack_field(trace, "q_cmd")
    q_act_all = _stack_field(trace, "q_act")
    q_policy_all = _stack_field(trace, "q_policy")
    applied_all = _stack_field(trace, "applied_torque")
    q_cmd = q_cmd_all[:, idxs].copy()
    q_act = q_act_all[:, idxs].copy()
    q_policy = q_policy_all[:, idxs].copy() if q_policy_all is not None else None
    applied = applied_all[:, idxs] if applied_all is not None else None
    if angle_unit == "deg":
        q_cmd *= 180.0 / np.pi
        q_act *= 180.0 / np.pi
        if q_policy is not None:
            q_policy *= 180.0 / np.pi
        angle_ylabel = "position (deg)"
    else:
        angle_ylabel = "Joint Angle [rad]"

    n_steps = len(trace.steps)
    if dt is not None and dt > 0.0:
        t = np.arange(n_steps, dtype=np.float64) * float(dt)
        xlabel = "time (s)"
    else:
        t = np.arange(n_steps, dtype=np.float64)
        xlabel = "sample index"

    color_policy = "#2ca02c"
    color_cmd = "#1f77b4"
    color_act = "#ff7f0e"
    color_tau = "#1f77b4"

    fig_w = max(2.4 * n, 12.0)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(fig_w, 7.5),
        sharex="col",
        squeeze=False,
        constrained_layout=True,
    )
    if title is None:
        if side == "torso":
            title = "Torso Joint States"
        elif side == "right":
            title = "Right Arm Joint States"
        else:
            title = "Left Arm Joint States"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for j, name in enumerate(arm_names):
        label = _short_joint_label(name).replace("_", " ")
        ax_tau = axes[0, j]
        ax_pos = axes[1, j]
        col_title = f"{side} {label}"

        if applied is not None:
            ax_tau.plot(t, applied[:, j], color=color_tau, linestyle="-", linewidth=1.6, label="actual torque")
        else:
            ax_tau.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax_tau.transAxes, fontsize=9)
        ax_tau.set_title(col_title, fontsize=10)
        ax_tau.grid(True, alpha=0.35)
        if j == 0:
            ax_tau.set_ylabel("torque (Nm)", fontsize=10)
            ax_tau.legend(fontsize=7, loc="best")

        if q_policy is not None:
            ax_pos.plot(
                t,
                q_policy[:, j],
                color=color_policy,
                linestyle=":",
                linewidth=1.5,
                label="q_policy (scaled)",
            )
        ax_pos.plot(t, q_cmd[:, j], color=color_cmd, linestyle="--", linewidth=1.4, label="q_cmd → Isaac")
        ax_pos.plot(t, q_act[:, j], color=color_act, linestyle="-", linewidth=1.6, label="q_act (measured)")
        ax_pos.set_xlabel(xlabel, fontsize=9)
        ax_pos.grid(True, alpha=0.35)
        if j == 0:
            ax_pos.set_ylabel(angle_ylabel, fontsize=10)
            ax_pos.legend(fontsize=7, loc="best")

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_left_right_arm_joint_states(
    trace: JointTrackingEpisodeTrace,
    out_dir: str,
    *,
    dt: float | None = 0.1,
    angle_unit: str = "rad",
) -> dict[str, str]:
    """Save arm (+ torso if present) PNGs with the same PDF-style layout."""
    os.makedirs(out_dir, exist_ok=True)
    names = list(trace.joint_names or (trace.steps[0].joint_names if trace.steps else []))
    paths: dict[str, str] = {
        "right": plot_arm_joint_states(
            trace,
            os.path.join(out_dir, "right_arm_joint_states.png"),
            side="right",
            dt=dt,
            title="Right Arm Joint States",
            angle_unit=angle_unit,
        ),
        "left": plot_arm_joint_states(
            trace,
            os.path.join(out_dir, "left_arm_joint_states.png"),
            side="left",
            dt=dt,
            title="Left Arm Joint States",
            angle_unit=angle_unit,
        ),
    }
    if _group_joint_indices(names, "torso"):
        paths["torso"] = plot_arm_joint_states(
            trace,
            os.path.join(out_dir, "torso_joint_states.png"),
            side="torso",
            dt=dt,
            title="Torso Joint States",
            angle_unit=angle_unit,
        )
    return paths


def plot_joint_tracking_episode(
    trace: JointTrackingEpisodeTrace,
    save_path: str,
    *,
    max_joints: int | None = None,
    title: str | None = None,
) -> str:
    """Legacy multi-panel diagnostic plot (prefer :func:`plot_left_right_arm_joint_states`)."""
    if not trace.steps:
        raise ValueError("empty joint tracking episode")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    names = trace.joint_names or trace.steps[0].joint_names
    n = len(names)
    if max_joints is not None:
        n = min(n, int(max_joints))
        names = names[:n]

    steps = np.arange(len(trace.steps))
    q_cmd = _stack_field(trace, "q_cmd")[:, :n]
    q_act = _stack_field(trace, "q_act")[:, :n]
    q_err = _stack_field(trace, "q_err")[:, :n]
    q_policy = _stack_field(trace, "q_policy")
    if q_policy is not None:
        q_policy = q_policy[:, :n]
    applied = _stack_field(trace, "applied_torque")
    computed = _stack_field(trace, "computed_torque")
    torque_err = _stack_field(trace, "torque_err")
    if applied is not None:
        applied = applied[:, :n]
    if computed is not None:
        computed = computed[:, :n]
    if torque_err is not None:
        torque_err = torque_err[:, :n]

    has_torque = applied is not None or computed is not None
    n_panels = 4 if has_torque else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.6 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    try:
        cmap = plt.colormaps["tab20"].resampled(max(n, 1))
    except Exception:
        cmap = plt.cm.get_cmap("tab20", max(n, 1))

    ax = axes[0]
    for j, name in enumerate(names):
        c = cmap(j)
        if q_policy is not None:
            ax.plot(
                steps,
                q_policy[:, j],
                color=c,
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
                label=f"{name} policy",
            )
        ax.plot(steps, q_cmd[:, j], color=c, linestyle="--", linewidth=1.2, alpha=0.85, label=f"{name} cmd")
        ax.plot(steps, q_act[:, j], color=c, linestyle="-", linewidth=1.4, alpha=0.95, label=f"{name} act")
    ax.set_ylabel("q [rad]")
    ax.set_title(title or f"Joint tracking (episode {trace.episode_index}, env {trace.env_id})")
    ax.grid(True, alpha=0.3)
    if n <= 6:
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    ax = axes[1]
    for j, name in enumerate(names):
        ax.plot(steps, q_err[:, j], color=cmap(j), linewidth=1.3, label=name)
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("q_cmd − q_act [rad]")
    ax.set_title("Position tracking error (Isaac target − measured)")
    ax.grid(True, alpha=0.3)
    if n <= 12:
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    if has_torque:
        ax = axes[2]
        for j, name in enumerate(names):
            c = cmap(j)
            if applied is not None:
                ax.plot(steps, applied[:, j], color=c, linestyle="-", linewidth=1.3, label=f"{name} applied")
            if computed is not None:
                ax.plot(
                    steps, computed[:, j], color=c, linestyle="--", linewidth=1.0, alpha=0.8, label=f"{name} computed"
                )
        ax.set_ylabel("τ [N·m]")
        ax.set_title("Applied vs computed torque (PD effort)")
        ax.grid(True, alpha=0.3)
        if n <= 6:
            ax.legend(fontsize=7, ncol=2, loc="upper right")

        ax = axes[3]
        if torque_err is not None:
            for j, name in enumerate(names):
                ax.plot(steps, torque_err[:, j], color=cmap(j), linewidth=1.3, label=name)
            ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
            ax.set_ylabel("τ_computed − τ_applied [N·m]")
            ax.set_title("Torque difference (saturation / clipping)")
            ax.grid(True, alpha=0.3)
            if n <= 12:
                ax.legend(fontsize=7, ncol=2, loc="upper right")
        else:
            ax.set_visible(False)

    axes[-1].set_xlabel("episode step")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def plot_joint_tracking_summary(
    trace: JointTrackingEpisodeTrace,
    save_path: str,
    *,
    title: str | None = None,
) -> str:
    """Bar summary: RMS / max |q_err| and |torque_err| per joint."""
    if not trace.steps:
        raise ValueError("empty joint tracking episode")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    names = trace.joint_names or trace.steps[0].joint_names
    q_err = _stack_field(trace, "q_err")
    torque_err = _stack_field(trace, "torque_err")
    n = len(names)
    x = np.arange(n)

    rms_q = np.sqrt(np.mean(q_err**2, axis=0))
    max_q = np.max(np.abs(q_err), axis=0)

    n_panels = 2 if torque_err is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(8, 0.45 * n), 3.2 * n_panels))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    w = 0.38
    ax.bar(x - w / 2, rms_q, width=w, label="RMS |q_err|")
    ax.bar(x + w / 2, max_q, width=w, label="max |q_err|")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("[rad]")
    ax.set_title(title or f"Position error summary (ep {trace.episode_index})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    if torque_err is not None:
        rms_t = np.sqrt(np.mean(torque_err**2, axis=0))
        max_t = np.max(np.abs(torque_err), axis=0)
        ax = axes[1]
        ax.bar(x - w / 2, rms_t, width=w, label="RMS |τ_err|")
        ax.bar(x + w / 2, max_t, width=w, label="max |τ_err|")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("[N·m]")
        ax.set_title("Torque difference summary (computed − applied)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


_CSV_SERIES = (
    "action",
    "q_policy",
    "q_cmd",
    "q_act",
    "q_actual",
    "q_err",
    "q_error_cmd",
    "q_error_policy",
    "q_vel",
    "applied_torque",
    "computed_torque",
    "torque_err",
    "torque_limit_reached",
    "velocity_limit_reached",
    "position_limit_reached",
)


def save_joint_tracking_csv(trace: JointTrackingEpisodeTrace, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    names = trace.joint_names or (trace.steps[0].joint_names if trace.steps else [])
    fields = ["step", "global_step", "simulation_time"]
    for prefix in _CSV_SERIES:
        for name in names:
            fields.append(f"{prefix}__{name}")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in trace.steps:
            row: dict[str, Any] = {
                "step": s.step,
                "global_step": s.global_step,
                "simulation_time": s.simulation_time,
            }
            mapping = {
                "action": s.action,
                "q_policy": s.q_policy,
                "q_cmd": s.q_cmd,
                "q_act": s.q_act,
                "q_actual": s.q_actual if s.q_actual is not None else s.q_act,
                "q_err": s.q_err,
                "q_error_cmd": s.q_error_cmd if s.q_error_cmd is not None else s.q_err,
                "q_error_policy": s.q_error_policy,
                "q_vel": s.q_vel,
                "applied_torque": s.applied_torque,
                "computed_torque": s.computed_torque,
                "torque_err": s.torque_err,
                "torque_limit_reached": s.torque_limit_reached,
                "velocity_limit_reached": s.velocity_limit_reached,
                "position_limit_reached": s.position_limit_reached,
            }
            for prefix, vals in mapping.items():
                if vals is None:
                    continue
                for i, name in enumerate(names):
                    if i < len(vals):
                        row[f"{prefix}__{name}"] = vals[i]
            writer.writerow(row)
    return path


def save_joint_tracking_json(trace: JointTrackingEpisodeTrace, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(trace.to_dict(), f, indent=2)
    return path


def save_joint_cmd_npy(
    trace: JointTrackingEpisodeTrace,
    path: str,
    *,
    dt: float | None = None,
) -> dict[str, str]:
    """Save commanded joint angles (rad) for real-world playback.

    Writes:
      ``*.npy``  — ``q_cmd`` array shaped ``(T, N)``, radians, column order = ``joint_names``
      ``*.npz`` — same ``q_cmd`` plus ``joint_names``, ``simulation_time``, ``dt``

    Returns paths for ``npy`` and ``npz``.
    """
    if not trace.steps:
        raise ValueError("empty joint tracking episode")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    stem, _ = os.path.splitext(path)
    npy_path = f"{stem}.npy"
    npz_path = f"{stem}.npz"

    names = list(trace.joint_names or trace.steps[0].joint_names)
    q_cmd = _stack_field(trace, "q_cmd")
    if q_cmd is None:
        raise ValueError("trace has no q_cmd")
    times = np.asarray([float(s.simulation_time) for s in trace.steps], dtype=np.float64)
    if dt is None and len(times) >= 2:
        dt = float(np.median(np.diff(times)))
    if dt is None:
        dt = 0.0

    np.save(npy_path, q_cmd.astype(np.float64, copy=False))
    np.savez_compressed(
        npz_path,
        q_cmd=q_cmd.astype(np.float64, copy=False),
        joint_names=np.asarray(names, dtype=object),
        simulation_time=times,
        dt=np.float64(dt),
        episode_index=np.int64(trace.episode_index),
        env_id=np.int64(trace.env_id),
    )
    # Sidecar names for the plain .npy (same column order).
    meta_path = f"{stem}_joint_names.json"
    with open(meta_path, "w") as f:
        json.dump({"joint_names": names, "dt": float(dt), "unit": "rad", "shape": list(q_cmd.shape)}, f, indent=2)
    return {"npy": npy_path, "npz": npz_path, "meta": meta_path}


def finalize_joint_tracking_episode(
    trace: JointTrackingEpisodeTrace,
    out_dir: str,
    *,
    save_plots: bool = True,
    max_joints: int | None = None,
    dt: float | None = 0.1,
    angle_unit: str = "deg",
) -> dict[str, str]:
    """Write JSON/CSV/NPY and arm (+ torso) PNGs for one episode. Returns path map.

    Bottom angle panel: ``q_policy`` (if present), ``q_cmd`` → Isaac, ``q_act`` measured.
    """
    del max_joints  # unused; arm plots include all arm joints
    os.makedirs(out_dir, exist_ok=True)
    stem = f"episode_{trace.episode_index:03d}_joint_tracking"
    paths: dict[str, str] = {}
    paths["json"] = save_joint_tracking_json(trace, os.path.join(out_dir, f"{stem}.json"))
    paths["csv"] = save_joint_tracking_csv(trace, os.path.join(out_dir, f"{stem}.csv"))
    if trace.steps:
        npy_paths = save_joint_cmd_npy(
            trace,
            os.path.join(out_dir, f"episode_{trace.episode_index:03d}_q_cmd.npy"),
            dt=dt,
        )
        paths.update(npy_paths)
    if save_plots and trace.steps:
        arm_paths = plot_left_right_arm_joint_states(trace, out_dir, dt=dt, angle_unit=angle_unit)
        paths["right_png"] = arm_paths["right"]
        paths["left_png"] = arm_paths["left"]
        if "torso" in arm_paths:
            paths["torso_png"] = arm_paths["torso"]
        paths["png"] = arm_paths["right"]
    return paths


def load_joint_tracking_json(path: str) -> JointTrackingEpisodeTrace:
    """Load a previously saved episode JSON into a trace."""
    with open(path) as f:
        data = json.load(f)
    steps = [
        JointTrackingStep(
            step=int(s["step"]),
            global_step=int(s["global_step"]),
            joint_names=list(s["joint_names"]),
            action=list(s["action"]),
            q_cmd=list(s["q_cmd"]),
            q_act=list(s["q_act"]),
            q_err=list(s["q_err"]),
            q_vel=list(s["q_vel"]),
            q_policy=s.get("q_policy"),
            simulation_time=float(s.get("simulation_time", 0.0)),
            q_actual=s.get("q_actual", s.get("q_act")),
            q_error_cmd=s.get("q_error_cmd", s.get("q_err")),
            q_error_policy=s.get("q_error_policy"),
            applied_torque=s.get("applied_torque"),
            computed_torque=s.get("computed_torque"),
            torque_err=s.get("torque_err"),
            torque_limit_reached=s.get("torque_limit_reached"),
            velocity_limit_reached=s.get("velocity_limit_reached"),
            position_limit_reached=s.get("position_limit_reached"),
        )
        for s in data.get("steps", [])
    ]
    return JointTrackingEpisodeTrace(
        episode_index=int(data.get("episode_index", 0)),
        env_id=int(data.get("env_id", 0)),
        joint_names=list(data.get("joint_names") or (steps[0].joint_names if steps else [])),
        steps=steps,
        terminated=bool(data.get("terminated", False)),
        truncated=bool(data.get("truncated", False)),
    )


def save_all_joint_tracking_csv(traces: list[JointTrackingEpisodeTrace], path: str) -> str:
    """Concatenate episodes into one CSV with an episode_index column."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not traces or not traces[0].steps:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["episode_index", "step"])
        return path

    names = traces[0].joint_names or traces[0].steps[0].joint_names
    fields = ["episode_index", "step", "global_step", "simulation_time"]
    for prefix in _CSV_SERIES:
        for name in names:
            fields.append(f"{prefix}__{name}")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tr in traces:
            for s in tr.steps:
                row: dict[str, Any] = {
                    "episode_index": tr.episode_index,
                    "step": s.step,
                    "global_step": s.global_step,
                    "simulation_time": s.simulation_time,
                }
                mapping = {
                    "action": s.action,
                    "q_policy": s.q_policy,
                    "q_cmd": s.q_cmd,
                    "q_act": s.q_act,
                    "q_actual": s.q_actual if s.q_actual is not None else s.q_act,
                    "q_err": s.q_err,
                    "q_error_cmd": s.q_error_cmd if s.q_error_cmd is not None else s.q_err,
                    "q_error_policy": s.q_error_policy,
                    "q_vel": s.q_vel,
                    "applied_torque": s.applied_torque,
                    "computed_torque": s.computed_torque,
                    "torque_err": s.torque_err,
                    "torque_limit_reached": s.torque_limit_reached,
                    "velocity_limit_reached": s.velocity_limit_reached,
                    "position_limit_reached": s.position_limit_reached,
                }
                for prefix, vals in mapping.items():
                    if vals is None:
                        continue
                    for i, name in enumerate(names):
                        if i < len(vals):
                            row[f"{prefix}__{name}"] = vals[i]
                writer.writerow(row)
    return path
