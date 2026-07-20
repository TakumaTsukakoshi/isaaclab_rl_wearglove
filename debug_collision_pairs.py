# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-collision pair finder: report which collision meshes touch / interpenetrate.

The robot is spawned alone (``scene_mode="free_space"``) with contact reporting
enabled on every rigid body link. PhysX contact-report events are collected each
physics step and aggregated per (collider mesh, collider mesh) pair, including
contact impulse and penetration depth (negative separation). Self-collision is
kept ON by default because interpenetrating pairs are exactly what we want to see.

The robot can either hold its default pose or sweep all joints with a sine wave
so that pose-dependent interferences also show up.

PhysX only delivers contact-report events with CPU dynamics, so the script runs
physics on the CPU unless ``--device`` is given explicitly.

Example::

    # Which meshes touch at the default pose?
    python debug_collision_pairs.py --duration-sec 5 --headless

    # Which meshes touch anywhere along a +-10 deg all-joint sine sweep?
    python debug_collision_pairs.py --waveform sine --amplitude-deg 10 \\
        --frequency-hz 0.2 --duration-sec 10 --headless
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Report contacting/interpenetrating collision-mesh pairs (robot only).")
parser.add_argument("--task", type=str, default="AIREC_Reach_Deformable_Bracelet", help="Registered task name (env cfg source).")
parser.add_argument(
    "--self-collision",
    choices=("on", "off"),
    default="on",
    help="Robot self-collision (default: on — interpenetrating pairs only report contacts when on).",
)
parser.add_argument(
    "--waveform",
    choices=("hold", "sine"),
    default="hold",
    help="hold = keep default pose; sine = sweep all joints around the default pose.",
)
parser.add_argument("--amplitude-deg", type=float, default=10.0, help="Sine amplitude [deg] (waveform=sine).")
parser.add_argument("--frequency-hz", type=float, default=0.2, help="Sine frequency [Hz] (waveform=sine).")
parser.add_argument("--duration-sec", type=float, default=5.0, help="Total run time [s].")
parser.add_argument("--output-dir", type=str, default="outputs/collision_pairs", help="Directory for CSV output.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (pair report uses env 0 paths).")
parser.add_argument("--top", type=int, default=20, help="How many pairs to print to stdout (CSV always has all).")

AppLauncher.add_app_launcher_args(parser)
# Contact-report events are not delivered with GPU dynamics (verified empirically);
# default to CPU physics unless the user explicitly picked a device.
_user_set_device = any(a == "--device" or a.startswith("--device=") for a in sys.argv[1:])
args_cli, hydra_args = parser.parse_known_args()
if not _user_set_device:
    args_cli.device = "cpu"
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from tasks import airec  # noqa: F401  (registers AIREC tasks)


def _make_debug_env(task: str, env_cfg):
    """Subclass the registered env so actions are absolute joint targets [rad]."""
    entry_point = gym.spec(task).entry_point
    module_name, class_name = entry_point.split(":")
    base_cls = getattr(importlib.import_module(module_name), class_name)

    class CollisionDebugEnv(base_cls):
        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            self.last_action = self.joint_pos_cmd[:, self.actuated_dof_indices]
            self.prev_joint_pos_cmd[:] = self.joint_pos_cmd
            q_target = actions.to(device=self.device, dtype=self.joint_pos_cmd.dtype)
            self.joint_pos_cmd[:, self.actuated_dof_indices] = q_target
            self.joint_pos_policy[:, self.actuated_dof_indices] = q_target
            self._clamp_actuated_joint_pos_cmd_inplace()

    CollisionDebugEnv.__name__ = f"CollisionDebug_{class_name}"
    return CollisionDebugEnv(cfg=env_cfg, render_mode=None)


def _short(path: str) -> str:
    """Readable link/mesh label: strip the per-env robot prefix."""
    for marker in ("/Robot/", "/robot/"):
        if marker in path:
            return path.split(marker, 1)[1]
    return path


class ContactPairRecorder:
    """Aggregate PhysX contact-report events per collider-mesh pair."""

    def __init__(self):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self._decode = PhysicsSchemaTools.intToSdfPath
        self._iface = get_physx_simulation_interface()
        self._sub = self._iface.subscribe_contact_report_events(self._on_subscribed_report)
        self.sim_time = 0.0  # updated by the stepping loop (control-step resolution)
        self.n_callbacks = 0    # subscription callbacks (every physics substep)
        self.n_polls_hit = 0    # manual get_contact_report() polls that returned data
        # key = sorted (collider0, collider1); value = aggregate dict
        self.pairs: dict[tuple[str, str], dict] = {}

    def _on_subscribed_report(self, contact_headers, contact_data) -> None:
        self.n_callbacks += 1
        self._ingest(contact_headers, contact_data)

    def poll(self) -> None:
        """Fallback: fetch the latest report directly. Only used while the
        subscription stays silent (it can be, when physics is stepped through
        the tensor API instead of the timeline)."""
        if self.n_callbacks > 0:
            return
        try:
            contact_headers, contact_data = self._iface.get_contact_report()
        except Exception:
            return
        if contact_headers:
            self.n_polls_hit += 1
            self._ingest(contact_headers, contact_data)

    def close(self):
        self._sub = None

    def _ingest(self, contact_headers, contact_data) -> None:
        for header in contact_headers:
            event = str(header.type).rsplit(".", 1)[-1]  # CONTACT_FOUND / _PERSIST / _LOST
            if event == "CONTACT_LOST":
                continue
            c0 = str(self._decode(header.collider0))
            c1 = str(self._decode(header.collider1))
            key = (c0, c1) if c0 <= c1 else (c1, c0)
            rec = self.pairs.get(key)
            if rec is None:
                rec = {
                    "collider0": key[0],
                    "collider1": key[1],
                    "actor0": str(self._decode(header.actor0)),
                    "actor1": str(self._decode(header.actor1)),
                    "events": 0,
                    "contact_points": 0,
                    "max_impulse_N_s": 0.0,
                    "sum_impulse_N_s": 0.0,
                    "max_penetration_mm": 0.0,
                    "first_seen_s": self.sim_time,
                    "last_seen_s": self.sim_time,
                }
                self.pairs[key] = rec
            rec["events"] += 1
            rec["last_seen_s"] = self.sim_time
            start = header.contact_data_offset
            for datum in contact_data[start : start + header.num_contact_data]:
                rec["contact_points"] += 1
                ix, iy, iz = datum.impulse.x, datum.impulse.y, datum.impulse.z
                imp = math.sqrt(ix * ix + iy * iy + iz * iz)
                rec["max_impulse_N_s"] = max(rec["max_impulse_N_s"], imp)
                rec["sum_impulse_N_s"] += imp
                # separation < 0 means interpenetration.
                pen_mm = max(0.0, -float(datum.separation)) * 1000.0
                rec["max_penetration_mm"] = max(rec["max_penetration_mm"], pen_mm)


def main() -> None:
    if float(args_cli.duration_sec) <= 0.0:
        parser.error("--duration-sec must be positive")

    out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene_mode = "free_space"
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if "cpu" not in str(env_cfg.sim.device):
        print("[collision debug] WARNING: GPU dynamics delivers no contact reports; results will likely be empty.")
    control_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    env_cfg.episode_length_s = float(args_cli.duration_sec) + 10.0 * control_dt
    env_cfg.debug_joint_cmd_vs_actual = False
    # Contact reports are only emitted for bodies carrying PhysxContactReportAPI;
    # this makes Isaac Lab apply it (threshold 0) to every robot link at spawn.
    env_cfg.robot_cfg.spawn.activate_contact_sensors = True
    self_collision = args_cli.self_collision == "on"
    env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = self_collision

    env = _make_debug_env(args_cli.task, env_cfg)
    # Isaac Lab's SimulationContext sets /physics/disableContactProcessing=True by
    # default (only ContactSensor re-enables it). Without this, no contact-report
    # events are ever delivered.
    import carb

    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    env.reset()
    recorder = ContactPairRecorder()

    # Sanity check: contact reports require PhysxContactReportAPI on the links.
    from isaaclab.sim import SimulationContext
    from pxr import PhysxSchema, Usd

    stage = SimulationContext.instance().stage
    robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    n_report_api = sum(
        1 for p in Usd.PrimRange(robot_root) if p.HasAPI(PhysxSchema.PhysxContactReportAPI)
    ) if robot_root and robot_root.IsValid() else 0
    print(f"[collision debug] links with PhysxContactReportAPI: {n_report_api}")
    if n_report_api == 0:
        print("[collision debug] WARNING: no contact-report API found on robot links; no pairs will be reported.")

    robot = env.robot
    actuated = list(env.actuated_dof_indices)
    q_center = robot.data.joint_pos[:, actuated].clone()
    lower = env.robot_hard_dof_lower_limits[actuated]
    upper = env.robot_hard_dof_upper_limits[actuated]

    n_steps = max(1, int(round(float(args_cli.duration_sec) / control_dt)))
    amplitude_rad = math.radians(float(args_cli.amplitude_deg))
    frequency_hz = float(args_cli.frequency_hz)

    print(f"[collision debug] task={args_cli.task} scene_mode=free_space num_envs={env.num_envs}")
    print(f"[collision debug] self_collision={'ON' if self_collision else 'OFF'} waveform={args_cli.waveform}")
    print(f"[collision debug] control_dt={control_dt:.6g} s -> {n_steps} control steps ({args_cli.duration_sec} s)")
    print(f"[collision debug] output dir: {out_dir}")
    if not self_collision:
        print("[collision debug] NOTE: self-collision OFF — robot-internal pairs will NOT appear in the report.")

    with torch.inference_mode():
        for k in range(n_steps):
            t = k * control_dt
            recorder.sim_time = t
            if args_cli.waveform == "sine":
                offset = math.sin(2.0 * math.pi * frequency_hz * t)
                cmd = torch.clamp(q_center + amplitude_rad * offset, lower, upper)
            else:
                cmd = q_center
            env.step(cmd)
            recorder.poll()
            if (k + 1) % max(1, n_steps // 5) == 0:
                print(
                    f"[collision debug] step {k + 1}/{n_steps} t={t:.2f}s "
                    f"pairs_so_far={len(recorder.pairs)} "
                    f"callbacks={recorder.n_callbacks} polls_hit={recorder.n_polls_hit}"
                )
            if not simulation_app.is_running():
                print("[collision debug] simulation app closed early; stopping")
                break

    recorder.close()

    rows = sorted(
        recorder.pairs.values(),
        key=lambda r: (r["max_penetration_mm"], r["max_impulse_N_s"]),
        reverse=True,
    )

    csv_path = os.path.join(out_dir, "collision_pairs.csv")
    fields = [
        "link_pair", "max_penetration_mm", "max_impulse_N_s", "sum_impulse_N_s",
        "events", "contact_points", "first_seen_s", "last_seen_s",
        "collider0", "collider1", "actor0", "actor1",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "link_pair": f"{_short(r['actor0'])} <-> {_short(r['actor1'])}",
                "max_penetration_mm": f"{r['max_penetration_mm']:.3f}",
                "max_impulse_N_s": f"{r['max_impulse_N_s']:.5f}",
                "sum_impulse_N_s": f"{r['sum_impulse_N_s']:.5f}",
                "events": r["events"],
                "contact_points": r["contact_points"],
                "first_seen_s": f"{r['first_seen_s']:.3f}",
                "last_seen_s": f"{r['last_seen_s']:.3f}",
                "collider0": r["collider0"],
                "collider1": r["collider1"],
                "actor0": r["actor0"],
                "actor1": r["actor1"],
            })
    print(f"[collision debug] wrote {csv_path} ({len(rows)} pairs)")

    if recorder.n_callbacks == 0 and recorder.n_polls_hit == 0:
        print(
            "[collision debug] WARNING: no contact reports were received at all "
            "(subscription silent and polling empty). "
            "If this persists, retry with '--device cpu'."
        )
    if not rows:
        print("[collision debug] no contacting pairs detected.")
    else:
        print(f"\n[collision debug] top {min(args_cli.top, len(rows))} contacting pairs "
              f"(sorted by max penetration, then impulse):")
        print(f"  {'pen[mm]':>8} {'max_imp':>9} {'events':>7} {'first[s]':>8} {'last[s]':>8}  link pair")
        for r in rows[: int(args_cli.top)]:
            pair = f"{_short(r['actor0'])} <-> {_short(r['actor1'])}"
            print(
                f"  {r['max_penetration_mm']:8.3f} {r['max_impulse_N_s']:9.5f} {r['events']:7d} "
                f"{r['first_seen_s']:8.3f} {r['last_seen_s']:8.3f}  {pair}"
            )
        persistent = [r for r in rows if r["first_seen_s"] < 2.0 * control_dt and r["events"] >= 5]
        if persistent:
            print("\n[collision debug] pairs already in contact at the initial pose (likely permanent interpenetration):")
            for r in persistent:
                print(f"  - {_short(r['actor0'])} <-> {_short(r['actor1'])} (pen {r['max_penetration_mm']:.2f} mm)")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        raise
    finally:
        print("CLOSING")
        simulation_app.close()
