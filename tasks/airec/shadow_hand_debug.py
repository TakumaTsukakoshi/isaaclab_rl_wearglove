"""Shadow Hand debug modes for reach-deformable-bracelet causal analysis.

Coordinate convention (matches ``ReachDeformableBraceletEnv``):
  - Task targets in ``thumb_target``, ``goal_wrist_pos``, etc. are **env-local** (meters).
  - GUI / ``VisualizationMarkers`` use **sim world** = env-local + ``scene.env_origins``.
  - User-captured constants below are env-local positions (e.g. env 0 print before ``+ env_origins``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal

import torch
from isaaclab.utils import configclass

try:
    from pxr import Sdf, Usd, UsdPhysics
except ImportError:  # pragma: no cover - Isaac Sim runtime only
    Sdf = None
    Usd = None
    UsdPhysics = None

DebugTargetMode = Literal[
    "baseline",
    "no_shadow_collision_fixed_targets",
    "no_shadow_actor_fixed_targets",
]

DEBUG_TARGET_MODES: tuple[str, ...] = (
    "baseline",
    "no_shadow_collision_fixed_targets",
    "no_shadow_actor_fixed_targets",
)

# AIREC gripper links that can interfere with Shadow Hand during insertion (env-local frame irrelevant).
ROBOT_HAND_COLLISION_LINK_NAMES: tuple[str, ...] = (
    "left_hand_first_finger_link_1",
    "left_hand_first_finger_link_2",
    "right_hand_first_finger_link_1",
    "right_hand_first_finger_link_2",
    "left_hand_thumb_link_1",
    "left_hand_thumb_link_2",
    "right_hand_thumb_link_1",
    "right_hand_thumb_link_2",
    "left_hand_palm_link",
    "right_hand_palm_link",
)


@configclass
class DebugFixedTargetCfg:
    """Env-local fixed virtual targets (same offset in every cloned env)."""

    thumb_target: tuple[float, float, float] = (0.5321, -0.0864, 0.7985)
    fore_goal_pos: tuple[float, float, float] = (0.4587, -0.0330, 0.8100)
    middle_goal_pos: tuple[float, float, float] = (0.4557, -0.0110, 0.8100)
    ring_goal_pos: tuple[float, float, float] = (0.4587, 0.0110, 0.8100)
    pinky_target: tuple[float, float, float] = (0.4542, 0.0513, 0.8117)
    goal_wrist_pos: tuple[float, float, float] = (0.6080, 0.0000, 0.8100)


def uses_fixed_task_targets(mode: str) -> bool:
    return mode in (
        "no_shadow_collision_fixed_targets",
        "no_shadow_actor_fixed_targets",
    )


def expand_fixed_positions(
    env_ids: torch.Tensor | Sequence[int],
    num_envs: int,
    device: torch.device,
    cfg: DebugFixedTargetCfg,
) -> dict[str, torch.Tensor]:
    """Build per-env tensors (env-local) for all fixed task targets."""
    if isinstance(env_ids, torch.Tensor):
        eids = env_ids.long().reshape(-1)
    else:
        eids = torch.as_tensor(list(env_ids), device=device, dtype=torch.long)
    n = int(eids.numel())

    def _tile(name: str) -> torch.Tensor:
        base = torch.tensor(getattr(cfg, name), device=device, dtype=torch.float32)
        out = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        out[eids] = base.unsqueeze(0).expand(n, 3)
        return out

    thumb = _tile("thumb_target")
    pinky = _tile("pinky_target")
    return {
        "thumb_target": thumb,
        "pinky_target": pinky,
        "fore_goal_pos": _tile("fore_goal_pos"),
        "middle_goal_pos": _tile("middle_goal_pos"),
        "ring_goal_pos": _tile("ring_goal_pos"),
        "goal_wrist_pos": _tile("goal_wrist_pos"),
        # Reach rewards use thumb/pinky goals then outward offsets; fixed mode sets both to captured tips.
        "thumb_goal_pos": thumb.clone(),
        "pinky_goal_pos": pinky.clone(),
    }


def _iter_collision_prim_paths(stage, root_path: str) -> list[str]:
    if Usd is None:
        return []
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    paths: list[str] = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            paths.append(prim.GetPath().pathString)
    return paths


def _add_filtered_pair(stage, prim_path_a: str, prim_path_b: str) -> bool:
    if UsdPhysics is None:
        return False
    prim_a = stage.GetPrimAtPath(prim_path_a)
    if not prim_a.IsValid():
        return False
    prim_b = stage.GetPrimAtPath(prim_path_b)
    if not prim_b.IsValid():
        return False
    fp_api = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
    rel = fp_api.GetFilteredPairsRel()
    if not rel:
        rel = fp_api.CreateFilteredPairsRel()
    rel.AddTarget(Sdf.Path(prim_path_b))
    return True


def apply_shadow_hand_collision_filters(
    stage,
    num_envs: int,
    *,
    disable_shadow_bracelet_collision: bool,
    disable_shadow_robot_collision: bool,
    disable_all_shadow_collision: bool = False,
) -> dict[str, int]:
    """Disable contact between Shadow Hand colliders and bracelet / AIREC hand links.

    Uses USD ``FilteredPairsAPI`` (one-sided pair definition is sufficient).

    Collision pairs disabled when flags are set:
      - ``disable_shadow_bracelet_collision``: every ShadowHand collider vs ``.../Object`` colliders
        (deformable bracelet mesh).
      - ``disable_shadow_robot_collision``: every ShadowHand collider vs AIREC
        ``*_hand_*finger*`` / ``*_palm*`` / ``*_thumb*`` link colliders.
      - ``disable_all_shadow_collision``: all colliders under ShadowHand (mode 3 ghost hand).
    """
    stats = {"pairs_added": 0, "shadow_colliders": 0, "envs": 0}
    if Usd is None:
        return stats

    from isaaclab.sim.schemas import modify_collision_properties
    from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg

    for env_i in range(num_envs):
        env_root = f"/World/envs/env_{env_i}"
        shadow_root = f"{env_root}/ShadowHand"
        shadow_colliders = _iter_collision_prim_paths(stage, shadow_root)
        stats["shadow_colliders"] += len(shadow_colliders)
        if not shadow_colliders:
            continue
        stats["envs"] += 1

        if disable_all_shadow_collision:
            modify_collision_properties(
                shadow_root,
                CollisionPropertiesCfg(collision_enabled=False),
                stage=stage,
            )
            continue

        other_paths: list[str] = []
        if disable_shadow_bracelet_collision:
            object_root = f"{env_root}/Object"
            other_paths.extend(_iter_collision_prim_paths(stage, object_root))
            if not other_paths and stage.GetPrimAtPath(object_root).IsValid():
                other_paths.append(object_root)

        if disable_shadow_robot_collision:
            robot_root = f"{env_root}/Robot"
            for link_name in ROBOT_HAND_COLLISION_LINK_NAMES:
                link_path = f"{robot_root}/{link_name}"
                other_paths.extend(_iter_collision_prim_paths(stage, link_path))
                if stage.GetPrimAtPath(link_path).IsValid() and link_path not in other_paths:
                    if stage.GetPrimAtPath(link_path).HasAPI(UsdPhysics.CollisionAPI):
                        other_paths.append(link_path)

        for sc in shadow_colliders:
            for oc in other_paths:
                if _add_filtered_pair(stage, sc, oc):
                    stats["pairs_added"] += 1

    return stats


try:
    from pxr import UsdGeom
except ImportError:
    UsdGeom = None  # type: ignore


def hide_shadow_hand_visual(stage, num_envs: int) -> None:
    """Hide Shadow Hand visuals (physics unchanged unless collision filters applied)."""
    if Usd is None or UsdGeom is None:
        return
    for env_i in range(num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/ShadowHand")
        if not root.IsValid():
            continue
        UsdGeom.Imageable(root).MakeInvisible()


class DebugRolloutLogger:
    """Accumulate per-step scalars and save ``.pt`` on episode end (eval / play)."""

    def __init__(
        self,
        save_path: str | None,
        enabled: bool,
        mode: str,
        num_envs: int,
        device: torch.device,
        episode_ids: torch.Tensor | None = None,
    ) -> None:
        self.enabled = bool(enabled and save_path)
        self.save_path = save_path
        self.mode = mode
        self.device = device
        self.records: list[dict[str, Any]] = []
        self._prev_ee_dist = torch.zeros((num_envs,), device=device, dtype=torch.float32)
        self._prev_right_upper = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self._prev_left_upper = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self._initialized = torch.zeros((num_envs,), device=device, dtype=torch.bool)
        if episode_ids is None:
            self.episode_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        else:
            self.episode_ids = episode_ids.long()

    def log_step(
        self,
        env,
        *,
        step: int,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        r_wrist_center: torch.Tensor | None = None,
    ) -> None:
        if not self.enabled:
            return

        ee_dist = env.ee_euclidean_distance.detach()
        delta_ee = ee_dist - self._prev_ee_dist

        right_upper = env.right_upper_ee_pos.detach()
        left_upper = env.left_upper_ee_pos.detach()
        opening_axis = right_upper - left_upper
        axis_norm = torch.norm(opening_axis, dim=-1, keepdim=True).clamp_min(1e-6)
        opening_axis_n = opening_axis / axis_norm

        step_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        actual_open_vel = torch.zeros_like(ee_dist)
        right_vel = torch.zeros_like(right_upper)
        left_vel = torch.zeros_like(left_upper)
        mask_init = self._initialized
        if mask_init.any():
            right_vel = (right_upper - self._prev_right_upper) / step_dt
            left_vel = (left_upper - self._prev_left_upper) / step_dt
            actual_open_vel = torch.sum((right_vel - left_vel) * opening_axis_n, dim=-1)

        term_log = getattr(env, "_term_log", {}) or {}
        term_reason = _termination_reason_str(term_log, terminated, truncated, env.task_success)

        only_eval = bool(getattr(env.cfg, "debug_log_eval_envs_only", True))
        num_eval = int(getattr(env.cfg, "num_eval_envs", 0))

        for e in range(env.num_envs):
            if only_eval and num_eval > 0 and e >= num_eval:
                continue
            rec = {
                "episode_id": int(self.episode_ids[e].item()),
                "step": int(step),
                "debug_target_mode": self.mode,
                "ee_euclidean_distance": float(ee_dist[e].item()),
                "delta_ee_euclidean_distance": float(delta_ee[e].item()),
                "actions": env.actions[e].detach().cpu(),
                "joint_pos_cmd": env.joint_pos_cmd[e, env.actuated_dof_indices].detach().cpu(),
                "joint_pos": env.joint_pos[e].detach().cpu(),
                "joint_vel": env.joint_vel[e].detach().cpu(),
                "left_ee_pos": env.left_ee_pos[e].detach().cpu(),
                "right_ee_pos": env.right_ee_pos[e].detach().cpu(),
                "left_upper_ee_pos": left_upper[e].detach().cpu(),
                "right_upper_ee_pos": right_upper[e].detach().cpu(),
                "left_ee_vel": left_vel[e].detach().cpu() if mask_init[e] else None,
                "right_ee_vel": right_vel[e].detach().cpu() if mask_init[e] else None,
                "goal_wrist_pos": env.goal_wrist_pos[e].detach().cpu(),
                "thumb_target": env.thumb_target[e].detach().cpu(),
                "pinky_target": env.pinky_target[e].detach().cpu(),
                "fingers_inside_soft_gate": float(env.fingers_inside_soft_gate[e].item()),
                "reward": float(rewards[e].item()),
                "terminated": bool(terminated[e].item()),
                "truncated": bool(truncated[e].item()),
                "termination_reason": term_reason[e],
                "command_opening_velocity": None,
                "command_opening_velocity_note": (
                    "Policy uses joint-position actions only; no Cartesian opening command."
                ),
                "actual_opening_velocity": float(actual_open_vel[e].item()) if mask_init[e] else None,
                "shadow_bracelet_contact_force": None,
                "shadow_robot_contact_force": None,
                "contact_force_note": "ContactSensor not enabled in AIRECEnv (commented out in _setup_scene).",
            }
            if r_wrist_center is not None:
                rec["r_wrist_center_distance"] = float(r_wrist_center[e].item())
            self.records.append(rec)

        self._prev_ee_dist = ee_dist.clone()
        self._prev_right_upper = right_upper.clone()
        self._prev_left_upper = left_upper.clone()
        self._initialized[:] = True

    def save_final(self) -> None:
        """Write accumulated steps to disk (call at end of rollout script)."""
        self.flush()

    def flush(self) -> None:
        if not self.enabled or not self.records:
            return
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        payload = {
            "debug_target_mode": self.mode,
            "records": self.records,
            "summary": summarize_episodes(self.records),
        }
        torch.save(payload, self.save_path)
        print(f"[DebugRolloutLogger] Saved {len(self.records)} steps to {self.save_path}")
        self.records.clear()


def _termination_reason_str(
    term_log: dict,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    task_success: torch.Tensor,
) -> list[str]:
    n = terminated.shape[0]
    reasons = ["none"] * n
    for i in range(n):
        if truncated[i]:
            reasons[i] = "timeout"
        elif not terminated[i]:
            continue
        elif task_success[i]:
            reasons[i] = "task_success"
        elif term_log.get("term_too_far") is not None and term_log["term_too_far"][i] > 0.5:
            reasons[i] = "too_far"
        elif term_log.get("term_grasp_right") is not None and term_log["term_grasp_right"][i] > 0.5:
            reasons[i] = "grasp_right"
        elif term_log.get("term_grasp_left") is not None and term_log["term_grasp_left"][i] > 0.5:
            reasons[i] = "grasp_left"
        elif term_log.get("term_out_of_reach") is not None and term_log["term_out_of_reach"][i] > 0.5:
            reasons[i] = "out_of_reach"
        else:
            reasons[i] = "terminated_other"
    return reasons


def summarize_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-episode max ee distance and threshold crossing rates."""
    from collections import defaultdict

    by_ep: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_ep[int(r["episode_id"])].append(r)

    summary = {}
    for ep_id, steps in by_ep.items():
        dists = [s["ee_euclidean_distance"] for s in steps]
        summary[ep_id] = {
            "max_ee_euclidean_distance": max(dists),
            "final_ee_euclidean_distance": dists[-1],
            "frac_ee_gt_0.25": sum(d > 0.25 for d in dists) / len(dists),
            "frac_ee_gt_0.30": sum(d > 0.30 for d in dists) / len(dists),
            "num_steps": len(steps),
        }
    return summary


def compare_debug_logs(path_a: str, path_b: str) -> dict[str, Any]:
    """Load two ``.pt`` logs from baseline vs no-shadow rollout."""
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    return {
        "mode_a": a.get("debug_target_mode"),
        "mode_b": b.get("debug_target_mode"),
        "summary_a": a.get("summary"),
        "summary_b": b.get("summary"),
    }
