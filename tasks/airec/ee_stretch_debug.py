"""Evaluation-only EE stretch logging, analysis, and command-side distance clamp.

Cartesian EE command targets are not exposed by the AIREC stack; clamp operates on
joint_pos_cmd before set_joint_position_target. ``remove_outward_relative_command``
uses arm-wide joint command deltas as a scalar proxy projected onto the stretch axis
(diagnostic, not Jacobian-exact). ``joint7_fallback`` only stops outward
left/right_arm_joint_7 increments when outward direction is configured or inferred.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sign_to_scalar(direction: str) -> float:
    if direction == "positive":
        return 1.0
    if direction == "negative":
        return -1.0
    return 0.0


@dataclass
class EEStretchDebugCfg:
    enabled: bool = False
    log_dir: str = "logs/ee_stretch_debug"
    watch_distance: float = 0.25
    clamp_enabled: bool = False
    clamp_limit: float = 0.30
    clamp_activation_distance: float = 0.295
    clamp_mode: str = "remove_outward_relative_command"
    target_object: str = "deformable_bracelet"
    joint7_fallback: bool = False
    left_joint7_outward_direction: str = "none"
    right_joint7_outward_direction: str = "none"
    evaluation_mode: bool = False


@dataclass
class _EpisodeState:
    episode_id: int
    env_id: int
    reference_ee_distance: float = 0.0
    prev_ee_distance: float | None = None
    prev_left_j7_cmd: float | None = None
    prev_right_j7_cmd: float | None = None
    step: int = -1
    first_watch_step: int | None = None
    max_ee_distance: float = 0.0
    step_of_max_ee_distance: int = 0
    max_stretch_ratio: float = 0.0
    first_clamp_step: int | None = None
    clamp_activation_count: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    snapshot_at_watch: dict[str, Any] | None = None
    snapshot_at_max: dict[str, Any] | None = None


class EEStretchDebugLogger:
    """Per-env step logger + aggregate analysis for Deformable Bracelet evaluation."""

    OBJECT_CATEGORY = "deformable_bracelet"

    def __init__(self, cfg: EEStretchDebugCfg, num_envs: int):
        self.cfg = cfg
        self.num_envs = num_envs
        self.log_dir = Path(cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._episode_counter = 0
        self._episodes: dict[int, _EpisodeState] = {}
        self._finished_episodes: list[dict[str, Any]] = []
        self._all_step_records: list[dict[str, Any]] = []

    @staticmethod
    def from_env_cfg(env_cfg) -> EEStretchDebugLogger | None:
        if not getattr(env_cfg, "ee_stretch_log_enabled", False):
            return None
        if getattr(env_cfg, "debug_target_object", "deformable_bracelet") != "deformable_bracelet":
            return None
        cfg = EEStretchDebugCfg(
            enabled=True,
            log_dir=getattr(env_cfg, "ee_stretch_log_dir", "logs/ee_stretch_debug"),
            watch_distance=float(getattr(env_cfg, "debug_ee_watch_distance", 0.25)),
            clamp_enabled=bool(getattr(env_cfg, "debug_enable_ee_distance_clamp", False)),
            clamp_limit=float(getattr(env_cfg, "debug_ee_clamp_limit", 0.30)),
            clamp_activation_distance=float(getattr(env_cfg, "debug_ee_clamp_activation_distance", 0.295)),
            clamp_mode=str(getattr(env_cfg, "debug_ee_clamp_mode", "remove_outward_relative_command")),
            target_object=str(getattr(env_cfg, "debug_target_object", "deformable_bracelet")),
            joint7_fallback=bool(getattr(env_cfg, "debug_joint7_fallback_clamp", False)),
            left_joint7_outward_direction=str(
                getattr(env_cfg, "debug_left_joint7_outward_direction", "none")
            ),
            right_joint7_outward_direction=str(
                getattr(env_cfg, "debug_right_joint7_outward_direction", "none")
            ),
            evaluation_mode=bool(getattr(env_cfg, "evaluation_mode", False)),
        )
        if cfg.joint7_fallback and cfg.clamp_mode != "joint7_fallback":
            cfg.clamp_mode = "joint7_fallback"
        return EEStretchDebugLogger(cfg, num_envs=1)  # num_envs patched in bind_env

    def bind_env(self, env) -> None:
        self.env = env
        self.num_envs = env.num_envs

    def on_reset(self, env_ids: torch.Tensor | list[int]) -> None:
        if env_ids is None:
            ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.detach().cpu().tolist()
        else:
            ids = list(env_ids)
        env = self.env
        left = env.left_upper_ee_pos
        right = env.right_upper_ee_pos
        ee_delta = right - left
        ee_distance = torch.linalg.norm(ee_delta, dim=-1)
        for eid in ids:
            prev = self._episodes.get(eid)
            if prev is not None and prev.step >= 0:
                last_success = False
                if prev.records:
                    last_success = bool(prev.records[-1].get("success", False))
                self._finish_episode(eid, prev, last_success)
            self._episode_counter += 1
            ref = float(ee_distance[eid].item())
            self._episodes[eid] = _EpisodeState(
                episode_id=self._episode_counter,
                env_id=int(eid),
                reference_ee_distance=ref,
                prev_ee_distance=ref,
            )
            lidx = env._left_arm_joint_7_robot_idx
            ridx = env._right_arm_joint_7_robot_idx
            self._episodes[eid].prev_left_j7_cmd = float(env.joint_pos_cmd[eid, lidx].item())
            self._episodes[eid].prev_right_j7_cmd = float(env.joint_pos_cmd[eid, ridx].item())

    def log_step(
        self,
        env_ids: torch.Tensor | None,
        rewards: torch.Tensor,
        log_extras: dict[str, Any],
        success: torch.Tensor,
        done: torch.Tensor,
        clamp_meta: dict[str, Any] | None = None,
    ) -> None:
        env = self.env
        if rewards.ndim > 1 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if env_ids is None:
            ids = list(range(self.num_envs))
        else:
            ids = env_ids.detach().cpu().tolist()

        left = env.left_upper_ee_pos
        right = env.right_upper_ee_pos
        ee_delta = right - left
        ee_distance = torch.linalg.norm(ee_delta, dim=-1)
        ee_axis = ee_delta / ee_distance.unsqueeze(-1).clamp_min(1e-6)

        lidx = env._left_arm_joint_7_robot_idx
        ridx = env._right_arm_joint_7_robot_idx
        lcol = env._left_arm_joint_7_policy_col
        rcol = env._right_arm_joint_7_policy_col
        act_idx = env.actuated_dof_indices

        joint_pos = env.robot.data.joint_pos
        joint_vel = env.robot.data.joint_vel
        left_j7_act = joint_pos[:, lidx]
        right_j7_act = joint_pos[:, ridx]
        left_j7_vel = joint_vel[:, lidx]
        right_j7_vel = joint_vel[:, ridx]
        left_j7_cmd = env.joint_pos_cmd[:, lidx]
        right_j7_cmd = env.joint_pos_cmd[:, ridx]

        clamp_meta = clamp_meta or {}

        for eid in ids:
            ep = self._episodes.get(eid)
            if ep is None:
                continue
            ep.step += 1
            step = ep.step
            dist = float(ee_distance[eid].item())
            ref = ep.reference_ee_distance
            stretch = dist / max(ref, 1e-6)
            prev_d = ep.prev_ee_distance if ep.prev_ee_distance is not None else dist
            ee_dist_vel = dist - prev_d
            ep.prev_ee_distance = dist

            left_cmd = float(left_j7_cmd[eid].item())
            right_cmd = float(right_j7_cmd[eid].item())
            prev_lcmd = ep.prev_left_j7_cmd if ep.prev_left_j7_cmd is not None else left_cmd
            prev_rcmd = ep.prev_right_j7_cmd if ep.prev_right_j7_cmd is not None else right_cmd
            left_cmd_change = left_cmd - prev_lcmd
            right_cmd_change = right_cmd - prev_rcmd
            ep.prev_left_j7_cmd = left_cmd
            ep.prev_right_j7_cmd = right_cmd

            watch = self.cfg.watch_distance
            if ep.first_watch_step is None and dist >= watch:
                ep.first_watch_step = step

            if dist > ep.max_ee_distance:
                ep.max_ee_distance = dist
                ep.step_of_max_ee_distance = step
            if stretch > ep.max_stretch_ratio:
                ep.max_stretch_ratio = stretch

            clamp_active = bool(clamp_meta.get("clamp_active", [False] * self.num_envs)[eid])
            if clamp_active:
                ep.clamp_activation_count += 1
                if ep.first_clamp_step is None:
                    ep.first_clamp_step = step

            total_r = float(rewards[eid].item()) if rewards.ndim > 0 else float(rewards.item())
            rec: dict[str, Any] = {
                "episode_id": ep.episode_id,
                "step": step,
                "env_id": eid,
                "object_category": self.OBJECT_CATEGORY,
                "success": bool(success[eid].item()),
                "done": bool(done[eid].item()),
                "total_reward": total_r,
                "left_ee_position_x": float(left[eid, 0].item()),
                "left_ee_position_y": float(left[eid, 1].item()),
                "left_ee_position_z": float(left[eid, 2].item()),
                "right_ee_position_x": float(right[eid, 0].item()),
                "right_ee_position_y": float(right[eid, 1].item()),
                "right_ee_position_z": float(right[eid, 2].item()),
                "ee_distance": dist,
                "reference_ee_distance": ref,
                "stretch_ratio": stretch,
                "ee_distance_velocity": ee_dist_vel,
                "left_joint7_actual_position": float(left_j7_act[eid].item()),
                "right_joint7_actual_position": float(right_j7_act[eid].item()),
                "left_joint7_actual_velocity": float(left_j7_vel[eid].item()),
                "right_joint7_actual_velocity": float(right_j7_vel[eid].item()),
                "left_joint7_command_position_or_target": left_cmd,
                "right_joint7_command_position_or_target": right_cmd,
                "left_joint7_command_change": left_cmd_change,
                "right_joint7_command_change": right_cmd_change,
                "left_joint7_policy_action_component_if_identifiable": float(
                    env.actions[eid, lcol].item()
                ),
                "right_joint7_policy_action_component_if_identifiable": float(
                    env.actions[eid, rcol].item()
                ),
                "clamp_enabled": self.cfg.clamp_enabled,
                "clamp_active": clamp_active,
                "clamp_activation_distance": self.cfg.clamp_activation_distance,
                "clamp_limit": self.cfg.clamp_limit,
            }
            for k, v in log_extras.items():
                if isinstance(v, torch.Tensor):
                    if v.ndim == 0:
                        rec[f"reward_{k}"] = float(v.item())
                    elif v.shape[0] > eid:
                        rec[f"reward_{k}"] = float(v[eid].item())
                elif isinstance(v, (int, float)):
                    rec[f"reward_{k}"] = float(v)
            for extra_k in (
                "outward_command_before_clamp",
                "outward_command_after_clamp",
            ):
                if extra_k in clamp_meta:
                    val = clamp_meta[extra_k]
                    rec[extra_k] = val[eid] if isinstance(val, (list, np.ndarray)) else val

            if hasattr(env, "fingers_inside_soft_gate"):
                rec["fingers_inside_soft_gate"] = float(env.fingers_inside_soft_gate[eid].item())
            if hasattr(env, "per_finger_insert_margin"):
                pfm = env.per_finger_insert_margin[eid].detach().cpu().tolist()
                for fi, val in enumerate(pfm):
                    rec[f"per_finger_insert_margin_{fi}"] = float(val)
            if hasattr(env, "per_finger_soft_inside"):
                pfs = env.per_finger_soft_inside[eid].detach().cpu().tolist()
                for fi, val in enumerate(pfs):
                    rec[f"per_finger_soft_inside_{fi}"] = float(val)
            if hasattr(env, "goal_south_pos"):
                rec["goal_south_pos_x"] = float(env.goal_south_pos[eid, 0].item())
                rec["goal_south_pos_y"] = float(env.goal_south_pos[eid, 1].item())
                rec["goal_south_pos_z"] = float(env.goal_south_pos[eid, 2].item())
            if hasattr(env, "goal_north_pos"):
                rec["goal_north_pos_x"] = float(env.goal_north_pos[eid, 0].item())
                rec["goal_north_pos_y"] = float(env.goal_north_pos[eid, 1].item())
                rec["goal_north_pos_z"] = float(env.goal_north_pos[eid, 2].item())
            if hasattr(env, "per_finger_height_z"):
                for fi, val in enumerate(env.per_finger_height_z[eid].detach().cpu().tolist()):
                    rec[f"finger_height_{fi}"] = float(val)

            ep.records.append(rec)
            self._all_step_records.append(rec)

            snap = {
                "total_reward": rec["total_reward"],
                "ee_distance": dist,
                "left_joint7_actual_position": rec["left_joint7_actual_position"],
                "right_joint7_actual_position": rec["right_joint7_actual_position"],
                "left_joint7_command_position_or_target": left_cmd,
                "right_joint7_command_position_or_target": right_cmd,
                "reward_components": {k: v for k, v in rec.items() if k.startswith("reward_")},
            }
            if ep.first_watch_step == step:
                ep.snapshot_at_watch = snap
            if ep.step_of_max_ee_distance == step:
                ep.snapshot_at_max = snap

            if done[eid].item():
                self._finish_episode(eid, ep, success[eid].item())

    def _finish_episode(self, env_id: int, ep: _EpisodeState, success: bool) -> None:
        summary = {
            "episode_id": ep.episode_id,
            "env_id": env_id,
            "success": bool(success),
            "first_step_ee_distance_reaches_watch": ep.first_watch_step,
            "first_step_ee_distance_reaches_0_25": ep.first_watch_step
            if self.cfg.watch_distance == 0.25
            else ep.first_watch_step,
            "ee_distance_at_episode_end": ep.prev_ee_distance,
            "max_ee_distance": ep.max_ee_distance,
            "step_of_max_ee_distance": ep.step_of_max_ee_distance,
            "max_stretch_ratio": ep.max_stretch_ratio,
            "first_clamp_step": ep.first_clamp_step,
            "clamp_activation_count": ep.clamp_activation_count,
            "reference_ee_distance": ep.reference_ee_distance,
        }
        if ep.snapshot_at_watch:
            summary.update({f"at_watch_{k}": v for k, v in _flatten_snapshot(ep.snapshot_at_watch).items()})
        if ep.snapshot_at_max:
            summary.update({f"at_max_{k}": v for k, v in _flatten_snapshot(ep.snapshot_at_max).items()})
        self._finished_episodes.append(summary)
        del self._episodes[env_id]

    def finalize(self) -> None:
        self._write_step_csv()
        self._write_step_npz()
        analysis = self._compute_analysis()
        self._write_json(self.log_dir / "analysis.json", analysis)
        self._write_json(self.log_dir / "episode_summaries.json", self._finished_episodes)
        agg = self._aggregate_summary()
        self._write_json(self.log_dir / "aggregate_summary.json", agg)
        if _HAS_MPL and self._all_step_records:
            self._write_plots()
        self._write_readme_interpretation()

    def _write_step_csv(self) -> None:
        if not self._all_step_records:
            return
        path = self.log_dir / "step_log.csv"
        keys = sorted({k for r in self._all_step_records for k in r.keys()})
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._all_step_records)

    def _write_step_npz(self) -> None:
        if not self._all_step_records:
            return
        path = self.log_dir / "step_log.npz"
        keys = sorted({k for r in self._all_step_records for k in r.keys()})
        arr = {k: [] for k in keys}
        for r in self._all_step_records:
            for k in keys:
                arr[k].append(r.get(k, np.nan))
        np.savez(path, **{k: np.array(v) for k, v in arr.items()})

    def _compute_analysis(self) -> dict[str, Any]:
        if not self._all_step_records:
            return {}
        recs = self._all_step_records
        ee_vel = np.array([r["ee_distance_velocity"] for r in recs], dtype=np.float64)
        ljv = np.array([r["left_joint7_actual_velocity"] for r in recs], dtype=np.float64)
        rjv = np.array([r["right_joint7_actual_velocity"] for r in recs], dtype=np.float64)
        lcc = np.array([r["left_joint7_command_change"] for r in recs], dtype=np.float64)
        rcc = np.array([r["right_joint7_command_change"] for r in recs], dtype=np.float64)

        def mean_vel_when(cond):
            m = cond
            return float(ee_vel[m].mean()) if m.any() else float("nan")

        analysis = {
            "correlation_left_joint7_actual_velocity_vs_ee_distance_velocity": _safe_corr(ljv, ee_vel),
            "correlation_right_joint7_actual_velocity_vs_ee_distance_velocity": _safe_corr(rjv, ee_vel),
            "correlation_left_joint7_command_change_vs_ee_distance_velocity": _safe_corr(lcc, ee_vel),
            "correlation_right_joint7_command_change_vs_ee_distance_velocity": _safe_corr(rcc, ee_vel),
            "mean_ee_distance_velocity_when_left_joint7_command_change_gt_0": mean_vel_when(lcc > 0),
            "mean_ee_distance_velocity_when_left_joint7_command_change_lt_0": mean_vel_when(lcc < 0),
            "mean_ee_distance_velocity_when_right_joint7_command_change_gt_0": mean_vel_when(rcc > 0),
            "mean_ee_distance_velocity_when_right_joint7_command_change_lt_0": mean_vel_when(rcc < 0),
        }
        analysis["candidate_outward_direction_left_joint7"] = _infer_outward(
            lcc, ee_vel, analysis["mean_ee_distance_velocity_when_left_joint7_command_change_gt_0"],
            analysis["mean_ee_distance_velocity_when_left_joint7_command_change_lt_0"],
        )
        analysis["candidate_outward_direction_right_joint7"] = _infer_outward(
            rcc, ee_vel, analysis["mean_ee_distance_velocity_when_right_joint7_command_change_gt_0"],
            analysis["mean_ee_distance_velocity_when_right_joint7_command_change_lt_0"],
        )
        analysis["evidence"] = (
            "Candidate directions are inferred from mean ee_distance_velocity when joint7 "
            "command change is positive vs negative. Multiple joints move simultaneously; "
            "this is not causal proof."
        )
        return analysis

    def _aggregate_summary(self) -> dict[str, Any]:
        eps = self._finished_episodes
        if not eps:
            return {}
        max_dists = [e["max_ee_distance"] for e in eps if e.get("max_ee_distance") is not None]
        stretch = [e["max_stretch_ratio"] for e in eps if e.get("max_stretch_ratio") is not None]
        watch_steps = [
            e["first_step_ee_distance_reaches_watch"]
            for e in eps
            if e.get("first_step_ee_distance_reaches_watch") is not None
        ]
        clamp_eps = sum(1 for e in eps if (e.get("clamp_activation_count") or 0) > 0)
        clamp_counts = [e.get("clamp_activation_count", 0) for e in eps]
        exceed_30 = sum(1 for d in max_dists if d > 0.30)
        reach_25 = sum(
            1 for e in eps if e.get("first_step_ee_distance_reaches_watch") is not None
        )
        rewards = []
        for r in self._all_step_records:
            if r.get("step") == max(
                (x["step"] for x in self._all_step_records if x["episode_id"] == r["episode_id"]),
                default=0,
            ):
                rewards.append(r.get("total_reward", 0.0))
        successes = [e.get("success", False) for e in eps]
        fig = [
            r.get("fingers_inside_soft_gate", np.nan)
            for r in self._all_step_records
            if "fingers_inside_soft_gate" in r
        ]
        return {
            "num_episodes": len(eps),
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "mean_total_reward": float(np.mean(rewards)) if rewards else float("nan"),
            "mean_max_ee_distance": float(np.mean(max_dists)) if max_dists else float("nan"),
            "median_max_ee_distance": float(np.median(max_dists)) if max_dists else float("nan"),
            "max_ee_distance_overall": float(np.max(max_dists)) if max_dists else float("nan"),
            "num_episodes_reaching_ee_distance_watch": reach_25,
            "watch_distance": self.cfg.watch_distance,
            "mean_first_step_reaching_watch": float(np.mean(watch_steps)) if watch_steps else float("nan"),
            "num_episodes_exceeding_ee_distance_0_30": exceed_30,
            "clamp_limit": self.cfg.clamp_limit,
            "mean_max_stretch_ratio": float(np.mean(stretch)) if stretch else float("nan"),
            "max_stretch_ratio_overall": float(np.max(stretch)) if stretch else float("nan"),
            "num_clamp_activated_episodes": clamp_eps,
            "mean_clamp_activation_count_per_episode": float(np.mean(clamp_counts)) if clamp_counts else 0.0,
            "mean_fingers_inside_soft_gate": float(np.nanmean(fig)) if fig else float("nan"),
            "clamp_enabled": self.cfg.clamp_enabled,
        }

    def _write_plots(self) -> None:
        plot_dir = self.log_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        by_ep: dict[int, list[dict]] = {}
        for r in self._all_step_records:
            by_ep.setdefault(r["episode_id"], []).append(r)
        for eid, recs in by_ep.items():
            recs = sorted(recs, key=lambda x: x["step"])
            steps = [r["step"] for r in recs]
            ee_d = [r["ee_distance"] for r in recs]
            stretch = [r["stretch_ratio"] for r in recs]
            watch_step = next(
                (r["step"] for r in recs if r["ee_distance"] >= self.cfg.watch_distance), None
            )
            max_step = recs[int(np.argmax(ee_d))]["step"] if ee_d else None
            clamp_step = next((r["step"] for r in recs if r.get("clamp_active")), None)

            def _mark(ax):
                for s, lbl in ((watch_step, "watch"), (max_step, "max_ee"), (clamp_step, "clamp")):
                    if s is not None:
                        ax.axvline(s, linestyle="--", alpha=0.6, label=lbl)

            fig, ax = plt.subplots()
            ax.plot(steps, ee_d)
            _mark(ax)
            ax.set_xlabel("step")
            ax.set_ylabel("ee_distance (m)")
            ax.legend()
            fig.savefig(plot_dir / f"ep{eid}_ee_distance.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(steps, stretch)
            _mark(ax)
            ax.set_xlabel("step")
            ax.set_ylabel("stretch_ratio")
            fig.savefig(plot_dir / f"ep{eid}_stretch_ratio.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(steps, [r["left_joint7_actual_position"] for r in recs], label="left")
            ax.plot(steps, [r["right_joint7_actual_position"] for r in recs], label="right")
            _mark(ax)
            ax.legend()
            ax.set_xlabel("step")
            ax.set_ylabel("joint7 actual position (rad)")
            fig.savefig(plot_dir / f"ep{eid}_joint7_actual.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(steps, [r["left_joint7_command_position_or_target"] for r in recs], label="left")
            ax.plot(steps, [r["right_joint7_command_position_or_target"] for r in recs], label="right")
            _mark(ax)
            ax.legend()
            ax.set_xlabel("step")
            ax.set_ylabel("joint7 command (rad)")
            fig.savefig(plot_dir / f"ep{eid}_joint7_command.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(steps, [r["ee_distance_velocity"] for r in recs])
            _mark(ax)
            ax.set_xlabel("step")
            ax.set_ylabel("ee_distance_velocity (m/step)")
            fig.savefig(plot_dir / f"ep{eid}_ee_distance_velocity.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.scatter(
                [r["left_joint7_command_change"] for r in recs],
                [r["ee_distance_velocity"] for r in recs],
                s=8,
                alpha=0.6,
            )
            ax.set_xlabel("left joint7 command change")
            ax.set_ylabel("ee_distance_velocity")
            fig.savefig(plot_dir / f"ep{eid}_scatter_left_cmd_vs_ee_vel.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.scatter(
                [r["right_joint7_command_change"] for r in recs],
                [r["ee_distance_velocity"] for r in recs],
                s=8,
                alpha=0.6,
            )
            ax.set_xlabel("right joint7 command change")
            ax.set_ylabel("ee_distance_velocity")
            fig.savefig(plot_dir / f"ep{eid}_scatter_right_cmd_vs_ee_vel.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(steps, [r["total_reward"] for r in recs])
            _mark(ax)
            ax.set_xlabel("step")
            ax.set_ylabel("total_reward")
            fig.savefig(plot_dir / f"ep{eid}_total_reward.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            reward_keys = sorted({k for r in recs for k in r if k.startswith("reward_")})
            if reward_keys:
                fig, ax = plt.subplots()
                for rk in reward_keys:
                    ax.plot(steps, [r.get(rk, np.nan) for r in recs], label=rk.replace("reward_", ""))
                _mark(ax)
                ax.legend(fontsize=6)
                ax.set_xlabel("step")
                ax.set_ylabel("reward component")
                fig.savefig(plot_dir / f"ep{eid}_reward_components.png", dpi=120, bbox_inches="tight")
                plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(steps, [r["left_joint7_actual_velocity"] for r in recs], label="left")
            ax.plot(steps, [r["right_joint7_actual_velocity"] for r in recs], label="right")
            _mark(ax)
            ax.legend()
            ax.set_xlabel("step")
            ax.set_ylabel("joint7 actual velocity (rad/s)")
            fig.savefig(plot_dir / f"ep{eid}_joint7_velocity.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    def _write_readme_interpretation(self) -> None:
        text = """# EE Stretch Debug — Interpretation

1. Baseline: if a joint_7 command change direction consistently pairs with ee_distance_velocity > 0,
   that direction is a **candidate** outward movement (not causal proof).

2. Clamp lowers max_ee_distance / stretch_ratio while success_rate is maintained:
   excess EE separation was likely unnecessary policy behavior; consider stretch penalty in training.

3. Clamp lowers max_ee_distance but success_rate drops sharply:
   policy may rely on excess separation; consider stretch-aware fine-tuning.

4. Clamp applied but actual ee_distance still exceeds 0.30 by a large margin:
   re-check controller tracking, contact forces, deformable physics, attachments, solver.

5. joint_7 vs ee_distance_velocity unclear:
   investigate full arm / Cartesian command contributions.

## Clamp implementation note

True Cartesian EE command targets are not available. ``remove_outward_relative_command`` adjusts
``joint_pos_cmd`` on both arms using summed joint deltas as a stretch-axis proxy (diagnostic).
``joint7_fallback`` only blocks outward ``left/right_arm_joint_7`` increments when direction is set.
"""
        (self.log_dir / "INTERPRETATION.md").write_text(text)

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        with path.open("w") as f:
            json.dump(obj, f, indent=2, default=str)


def _flatten_snapshot(snap: dict) -> dict[str, Any]:
    out = {}
    for k, v in snap.items():
        if k == "reward_components" and isinstance(v, dict):
            for rk, rv in v.items():
                out[f"reward_{rk}"] = rv
        else:
            out[k] = v
    return out


def _infer_outward(
    cmd_change: np.ndarray,
    ee_vel: np.ndarray,
    mean_pos: float,
    mean_neg: float,
) -> str:
    if np.isnan(mean_pos) and np.isnan(mean_neg):
        return "unclear"
    if not np.isnan(mean_pos) and not np.isnan(mean_neg):
        if mean_pos > mean_neg + 1e-5 and mean_pos > 0:
            return "positive"
        if mean_neg > mean_pos + 1e-5 and mean_neg > 0:
            return "negative"
    corr = _safe_corr(cmd_change, ee_vel)
    if corr > 0.2:
        return "positive"
    if corr < -0.2:
        return "negative"
    return "unclear"


def apply_ee_distance_clamp(env, cfg: EEStretchDebugCfg) -> dict[str, Any]:
    """Modify env.joint_pos_cmd in-place before set_joint_position_target. Returns per-env meta."""
    num_envs = env.num_envs
    meta = {
        "clamp_active": [False] * num_envs,
        "outward_command_before_clamp": [0.0] * num_envs,
        "outward_command_after_clamp": [0.0] * num_envs,
    }
    if not cfg.clamp_enabled or not cfg.evaluation_mode:
        return meta
    if cfg.target_object != "deformable_bracelet":
        return meta

    pre_cmd = env._ee_stretch_pre_joint_pos_cmd
    left_dofs = env._left_arm_robot_dof_indices
    right_dofs = env._right_arm_robot_dof_indices
    lidx = env._left_arm_joint_7_robot_idx
    ridx = env._right_arm_joint_7_robot_idx

    left = env.left_upper_ee_pos
    right = env.right_upper_ee_pos
    ee_delta = right - left
    ee_distance = torch.linalg.norm(ee_delta, dim=-1)
    stretch_axis = ee_delta / ee_distance.unsqueeze(-1).clamp_min(1e-6)

    delta_left = env.joint_pos_cmd[:, left_dofs] - pre_cmd[:, left_dofs]
    delta_right = env.joint_pos_cmd[:, right_dofs] - pre_cmd[:, right_dofs]

    # Scalar proxy: relative outward joint command (diagnostic, not Jacobian-exact Cartesian)
    outward_cmd = delta_right.sum(dim=-1) - delta_left.sum(dim=-1)

    activation = cfg.clamp_activation_distance
    clamp_mask = (ee_distance >= activation) & (outward_cmd > 0)

    mode = cfg.clamp_mode
    if cfg.joint7_fallback:
        mode = "joint7_fallback"

    left_dir = cfg.left_joint7_outward_direction
    right_dir = cfg.right_joint7_outward_direction
    if left_dir == "none" and hasattr(env, "_ee_stretch_inferred_left_outward"):
        left_dir = env._ee_stretch_inferred_left_outward
    if right_dir == "none" and hasattr(env, "_ee_stretch_inferred_right_outward"):
        right_dir = env._ee_stretch_inferred_right_outward

    for i in range(num_envs):
        meta["outward_command_before_clamp"][i] = float(outward_cmd[i].item())

    if not clamp_mask.any():
        meta["outward_command_after_clamp"] = meta["outward_command_before_clamp"].copy()
        return meta

    if mode == "joint7_fallback":
        ls = _sign_to_scalar(left_dir)
        rs = _sign_to_scalar(right_dir)
        for i in torch.where(clamp_mask)[0].tolist():
            dl7 = env.joint_pos_cmd[i, lidx] - pre_cmd[i, lidx]
            dr7 = env.joint_pos_cmd[i, ridx] - pre_cmd[i, ridx]
            outward_scalar = 0.0
            if ls != 0.0 and dl7 * ls > 0:
                outward_scalar += float((dl7 * ls).item())
            if rs != 0.0 and dr7 * rs > 0:
                outward_scalar += float((dr7 * rs).item())
            if outward_scalar <= 0:
                continue
            if ls != 0.0 and dl7 * ls > 0:
                env.joint_pos_cmd[i, lidx] = pre_cmd[i, lidx]
            if rs != 0.0 and dr7 * rs > 0:
                env.joint_pos_cmd[i, ridx] = pre_cmd[i, ridx]
            meta["clamp_active"][i] = True
            meta["outward_command_after_clamp"][i] = 0.0
    else:
        # remove_outward_relative_command via arm-sum proxy (50/50 split on joint deltas)
        for i in torch.where(clamp_mask)[0].tolist():
            oc = float(outward_cmd[i].item())
            half = oc / 2.0
            env.joint_pos_cmd[i, left_dofs] = pre_cmd[i, left_dofs] + delta_left[i] + half
            env.joint_pos_cmd[i, right_dofs] = pre_cmd[i, right_dofs] + delta_right[i] - half
            meta["clamp_active"][i] = True
            meta["outward_command_after_clamp"][i] = 0.0

    # Re-apply joint limits on actuated DOFs (same as parent after command update)
    if hasattr(env, "_clamp_actuated_joint_pos_cmd_inplace"):
        env._clamp_actuated_joint_pos_cmd_inplace()

    return meta


def compare_aggregate_summaries(path_a: Path, path_b: Path, out_path: Path) -> dict[str, Any]:
    """Compare baseline vs clamp aggregate_summary.json files."""
    with path_a.open() as f:
        a = json.load(f)
    with path_b.open() as f:
        b = json.load(f)
    cmp_out = {"baseline": a, "clamp": b, "delta": {}}
    for k in a:
        if isinstance(a[k], (int, float)) and isinstance(b.get(k), (int, float)):
            cmp_out["delta"][k] = b[k] - a[k]
    with out_path.open("w") as f:
        json.dump(cmp_out, f, indent=2)
    return cmp_out
