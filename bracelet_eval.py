"""Bracelet-task evaluation: motion-lock success, finger insertion, joint deviation.

Insertion is a per-finger crossing state machine, not a final-frame or last-window
point-in-opening test.

Geometry (env-local, same live buffers as ``reach_*_bracelet``):
  * opening center ``c_t`` = ``goal_cent_pos`` (rigid: root + rim offsets; deformable: live rim)
  * opening plane normal ``n`` = env +X
  * opening boundary = live Y-Z ellipse from N/S/E/W rim goals
  * finger point ``p_i`` = finger-base / knuckle COM (not the fingertip)

Signed distance:
  ``d_i = n · (p_i - c_t) = p_i.x - c_t.x``
  pre-insertion (hand / +X) = ``d > +delta``
  post-insertion (through / -X) = ``d < -delta``

A finger becomes inserted after a confirmed forward crossing through the
ellipse. ``PRE → BAND → POST`` counts: the last clear hand-side sample is
used, so the ±delta deadband does not swallow the event. It stays inserted
if the bracelet later slides to the wrist or deforms. A confirmed reverse
crossing (``POST → PRE`` through the ellipse) clears the flag.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from play_common import PlaySession, unwrap_env
from play_output_utils import control_dt_from_env_cfg

FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
FINGER_LABELS = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "little": "Pinky",
}

# Actuated Shadow Hand finger joints (matches Reach*BraceletEnvCfg.finger_joint_names).
# Wrist WRJ* and coupled DIP *J0 are excluded.
FINGER_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "thumb": ("robot0_THJ4", "robot0_THJ3", "robot0_THJ2", "robot0_THJ1", "robot0_THJ0"),
    "index": ("robot0_FFJ3", "robot0_FFJ2", "robot0_FFJ1"),
    "middle": ("robot0_MFJ3", "robot0_MFJ2", "robot0_MFJ1"),
    "ring": ("robot0_RFJ3", "robot0_RFJ2", "robot0_RFJ1"),
    "little": ("robot0_LFJ4", "robot0_LFJ3", "robot0_LFJ2", "robot0_LFJ1"),
}

# Palm-side finger bases (first match wins). Tips / middle phalanges are not used.
BASE_BODY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "thumb": ("robot0_thbase", "robot0_thproximal"),
    "index": ("robot0_ffknuckle", "robot0_ffproximal"),
    "middle": ("robot0_mfknuckle", "robot0_mfproximal"),
    "ring": ("robot0_rfknuckle", "robot0_rfproximal"),
    "little": ("robot0_lfknuckle", "robot0_lfmetacarpal", "robot0_lfproximal"),
}

INSERTION_OUTCOMES = (
    "none",
    "partial",
    "all_exited",
    "all_retained",
    "all_retained_and_success",
)

CSV_FIELDS = [
    "episode",
    "env_id",
    "success",
    "terminated",
    "truncated",
    "episode_length_steps",
    "episode_length_seconds",
    "motion_lock_step",
    "motion_lock_time_s",
    "thumb_inserted",
    "index_inserted",
    "middle_inserted",
    "ring_inserted",
    "little_inserted",
    "pinky_inserted",
    "final_inserted_fingers",
    "max_inserted_fingers",
    "ever_all_inserted",
    "final_all_inserted",
    "insertion_outcome",
    "thumb_first_insert_time",
    "index_first_insert_time",
    "middle_first_insert_time",
    "ring_first_insert_time",
    "little_first_insert_time",
    "pinky_first_insert_time",
    "thumb_insert_ratio",
    "index_insert_ratio",
    "middle_insert_ratio",
    "ring_insert_ratio",
    "little_insert_ratio",
    "thumb_insert_steps",
    "index_insert_steps",
    "middle_insert_steps",
    "ring_insert_steps",
    "little_insert_steps",
    "num_inserted_fingers",
    "thumb_rms",
    "index_rms",
    "middle_rms",
    "ring_rms",
    "little_rms",
    "thumb_peak",
    "index_peak",
    "middle_peak",
    "ring_peak",
    "little_peak",
    "hand_rms",
    "worst_finger_peak",
    "worst_finger",
    "return",
    "final_success",
    "wrist_distance_at_success",
    "inserted_fingers_at_success",
    "all_5_inserted_at_success",
    "thumb_inserted_at_success",
    "index_inserted_at_success",
    "middle_inserted_at_success",
    "ring_inserted_at_success",
    "little_inserted_at_success",
    "episode_done_step",
    "episode_done_reason",
    "first_wrist_goal_step",
    "first_wrist_goal_time_s",
    "inserted_fingers_at_first_wrist_goal",
    "missing_fingers_at_first_wrist_goal",
]


def _rad_to_deg(value: float) -> float:
    return float(value) * (180.0 / math.pi)


def sample_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def _as_bool_scalar(value: Any, env_id: int) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        if value.ndim == 0:
            return bool(value.item())
        if env_id >= int(value.shape[0]):
            return False
        return bool(value[env_id].reshape(-1)[0].item() > 0.5)
    if isinstance(value, (list, tuple)) and env_id < len(value):
        return bool(value[env_id])
    return bool(value)


def read_motion_locked(infos: Any, raw_env: Any, env_id: int) -> bool:
    """Read the pre-reset motion-lock flag written in ``_get_rewards``.

    After ``step()``, Isaac Lab has already reset done envs, so
    ``_task_success_bonus_awarded`` is False. ``extras['log']['motion_locked']``
    (and ``infos['log']``) still hold the value from that control step.
    """
    log: dict[str, Any] | None = None
    if isinstance(infos, dict):
        maybe = infos.get("log")
        if isinstance(maybe, dict):
            log = maybe
    if log is None:
        extras = getattr(raw_env, "extras", None) or {}
        maybe = extras.get("log") if isinstance(extras, dict) else None
        if isinstance(maybe, dict):
            log = maybe
    if log is None:
        return False
    return _as_bool_scalar(log.get("motion_locked"), env_id)


def _resolve_body_index(body_names: list[str], candidates: tuple[str, ...]) -> int | None:
    lower = {name.lower(): i for i, name in enumerate(body_names)}
    for cand in candidates:
        idx = lower.get(cand.lower())
        if idx is not None:
            return idx
    return None


def opening_radii(
    east: torch.Tensor,
    west: torch.Tensor,
    north: torch.Tensor,
    south: torch.Tensor,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Live Y (E/W) and Z (N/S) ellipse semi-axes. Shape ``(num_envs,)``."""
    min_r = torch.as_tensor(eps, device=east.device, dtype=east.dtype)
    radius_y = 0.5 * torch.abs(east[:, 1] - west[:, 1]).clamp_min(min_r)
    radius_z = 0.5 * torch.abs(north[:, 2] - south[:, 2]).clamp_min(min_r)
    return radius_y, radius_z


def ellipse_value_yz(
    point: torch.Tensor,
    center: torch.Tensor,
    radius_y: torch.Tensor,
    radius_z: torch.Tensor,
) -> torch.Tensor:
    """Normalized Y-Z ellipse value. ``point`` is ``(N, 5, 3)`` or ``(N, 3)``."""
    if point.ndim == 2:
        dy = (point[:, 1] - center[:, 1]) / radius_y
        dz = (point[:, 2] - center[:, 2]) / radius_z
        return dy.pow(2) + dz.pow(2)
    dy = (point[..., 1] - center[:, 1].unsqueeze(1)) / radius_y.unsqueeze(1)
    dz = (point[..., 2] - center[:, 2].unsqueeze(1)) / radius_z.unsqueeze(1)
    return dy.pow(2) + dz.pow(2)


def classify_insertion_outcome(
    *,
    max_inserted: int,
    ever_all: bool,
    final_all: bool,
    success: bool,
) -> str:
    """Most specific episode insertion case. Task success is only used for the last label."""
    if ever_all or final_all:
        if final_all and success:
            return "all_retained_and_success"
        if final_all:
            return "all_retained"
        return "all_exited"
    if max_inserted <= 0:
        return "none"
    return "partial"


class FingerCrossingTracker:
    """Per-env, per-finger insertion latch with forward/reverse crossing confirmation."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        delta: float,
        confirm_frames: int,
        ellipse_threshold: float,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = dtype
        self.delta = float(delta)
        self.confirm_frames = max(1, int(confirm_frames))
        self.ellipse_threshold = float(ellipse_threshold)

        z5 = (self.num_envs, 5)
        self.inserted = torch.zeros(z5, dtype=torch.bool, device=device)
        # Last clearly-outside-band side: +1 PRE, -1 POST, 0 unknown. BAND does not update this.
        self.last_clear_side = torch.zeros(z5, dtype=torch.int8, device=device)
        self.last_clear_pos = torch.zeros((self.num_envs, 5, 3), dtype=dtype, device=device)
        self.last_clear_center = torch.zeros((self.num_envs, 5, 3), dtype=dtype, device=device)
        self.last_clear_radius_y = torch.zeros((self.num_envs, 5), dtype=dtype, device=device)
        self.last_clear_radius_z = torch.zeros((self.num_envs, 5), dtype=dtype, device=device)
        self.has_clear = torch.zeros(z5, dtype=torch.bool, device=device)
        self.fwd_pending = torch.zeros(z5, dtype=torch.bool, device=device)
        self.rev_pending = torch.zeros(z5, dtype=torch.bool, device=device)
        self.fwd_count = torch.zeros(z5, dtype=torch.int32, device=device)
        self.rev_count = torch.zeros(z5, dtype=torch.int32, device=device)
        self.first_insert_step = torch.full(z5, -1, dtype=torch.int32, device=device)
        self.step_count = torch.zeros((self.num_envs,), dtype=torch.int32, device=device)
        self.max_inserted = torch.zeros((self.num_envs,), dtype=torch.int32, device=device)
        self.ever_all = torch.zeros((self.num_envs,), dtype=torch.bool, device=device)
        self.inserted_steps = torch.zeros(z5, dtype=torch.int32, device=device)

    def reset_envs(self, env_ids: list[int] | torch.Tensor) -> None:
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self.inserted[ids] = False
        self.last_clear_side[ids] = 0
        self.last_clear_pos[ids] = 0.0
        self.last_clear_center[ids] = 0.0
        self.last_clear_radius_y[ids] = 0.0
        self.last_clear_radius_z[ids] = 0.0
        self.has_clear[ids] = False
        self.fwd_pending[ids] = False
        self.rev_pending[ids] = False
        self.fwd_count[ids] = 0
        self.rev_count[ids] = 0
        self.first_insert_step[ids] = -1
        self.step_count[ids] = 0
        self.max_inserted[ids] = 0
        self.ever_all[ids] = False
        self.inserted_steps[ids] = 0

    def update(
        self,
        distal: torch.Tensor,
        center: torch.Tensor,
        radius_y: torch.Tensor,
        radius_z: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the state machine. ``active`` skips post-reset / inactive envs.

        Crossing uses the last *clear* side (outside ±delta), so PRE → BAND → POST
        counts as a forward cross. The ±delta band is only a deadzone, not a veto.
        """
        active = active.to(device=self.device, dtype=torch.bool)
        if not bool(active.any()):
            return self.inserted

        d_curr = distal[..., 0] - center[:, 0].unsqueeze(1)
        is_pre = d_curr > self.delta
        is_post = d_curr < -self.delta
        is_clear = is_pre | is_post
        side = torch.where(is_pre, torch.ones_like(self.last_clear_side), torch.where(is_post, -torch.ones_like(self.last_clear_side), torch.zeros_like(self.last_clear_side)))

        d_clear = self.last_clear_pos[..., 0] - self.last_clear_center[..., 0]
        denom = d_clear - d_curr
        t = torch.where(denom.abs() > 1e-8, d_clear / denom, torch.full_like(d_curr, 0.5))
        t = t.clamp(0.0, 1.0)
        t3 = t.unsqueeze(-1)
        cross_p = self.last_clear_pos + t3 * (distal - self.last_clear_pos)
        cross_c = self.last_clear_center + t3 * (center.unsqueeze(1) - self.last_clear_center)
        cross_ry = self.last_clear_radius_y + t * (radius_y.unsqueeze(1) - self.last_clear_radius_y)
        cross_rz = self.last_clear_radius_z + t * (radius_z.unsqueeze(1) - self.last_clear_radius_z)
        ev = ((cross_p[..., 1] - cross_c[..., 1]) / cross_ry.clamp_min(1e-6)).pow(2) + (
            (cross_p[..., 2] - cross_c[..., 2]) / cross_rz.clamp_min(1e-6)
        ).pow(2)
        through_opening = ev <= self.ellipse_threshold

        from_pre = self.has_clear & (self.last_clear_side > 0)
        from_post = self.has_clear & (self.last_clear_side < 0)
        active_f = active.unsqueeze(1)
        fwd_cand = active_f & is_post & from_pre & (~self.inserted) & through_opening
        rev_cand = active_f & is_pre & from_post & self.inserted & through_opening

        ones = torch.ones_like(self.fwd_count)
        zeros = torch.zeros_like(self.fwd_count)

        fwd_pending = ((self.fwd_pending & is_post) | fwd_cand) & (~self.inserted) & active_f
        fwd_count = torch.where(~fwd_pending, zeros, torch.where(fwd_cand, ones, self.fwd_count + 1))
        commit_fwd = fwd_pending & (fwd_count >= self.confirm_frames)
        inserted = self.inserted | commit_fwd
        fwd_pending = fwd_pending & (~commit_fwd)
        fwd_count = torch.where(fwd_pending, fwd_count, zeros)

        first = self.first_insert_step
        step_i = (self.step_count + 1).unsqueeze(1).expand_as(first)
        self.first_insert_step = torch.where((first < 0) & commit_fwd, step_i.to(first.dtype), first)

        rev_pending = ((self.rev_pending & is_pre) | rev_cand) & inserted & active_f
        rev_count = torch.where(~rev_pending, zeros, torch.where(rev_cand, ones, self.rev_count + 1))
        commit_rev = rev_pending & (rev_count >= self.confirm_frames)
        inserted = inserted & (~commit_rev)
        rev_pending = rev_pending & (~commit_rev) & inserted
        rev_count = torch.where(rev_pending, rev_count, zeros)

        active_i = active.to(dtype=self.step_count.dtype)
        self.step_count = self.step_count + active_i
        n_now = inserted.sum(dim=1).to(dtype=self.max_inserted.dtype)
        self.max_inserted = torch.maximum(self.max_inserted, torch.where(active, n_now, self.max_inserted))
        self.ever_all = self.ever_all | (active & (n_now >= 5))
        self.inserted_steps = self.inserted_steps + (inserted & active_f).to(self.inserted_steps.dtype)

        clear_f = active_f & is_clear
        self.last_clear_side = torch.where(clear_f, side, self.last_clear_side)
        self.last_clear_pos = torch.where(clear_f.unsqueeze(-1), distal, self.last_clear_pos)
        self.last_clear_center = torch.where(clear_f.unsqueeze(-1), center.unsqueeze(1).expand_as(distal), self.last_clear_center)
        self.last_clear_radius_y = torch.where(clear_f, radius_y.unsqueeze(1).expand_as(d_curr), self.last_clear_radius_y)
        self.last_clear_radius_z = torch.where(clear_f, radius_z.unsqueeze(1).expand_as(d_curr), self.last_clear_radius_z)
        self.has_clear = self.has_clear | clear_f

        self.inserted = torch.where(active_f, inserted, self.inserted)
        self.fwd_pending = torch.where(active_f, fwd_pending, self.fwd_pending)
        self.rev_pending = torch.where(active_f, rev_pending, self.rev_pending)
        self.fwd_count = torch.where(active_f, fwd_count, self.fwd_count)
        self.rev_count = torch.where(active_f, rev_count, self.rev_count)
        return self.inserted


def _finger_names_from_flags(flags: list[bool]) -> str:
    return ",".join(name for name, on in zip(FINGER_ORDER, flags) if on)


def _missing_finger_names(flags: list[bool]) -> str:
    return ",".join(name for name, on in zip(FINGER_ORDER, flags) if not on)


def _read_float_env(value: Any, env_id: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        if value.ndim == 0:
            return float(value.item())
        if env_id >= int(value.shape[0]):
            return None
        return float(value[env_id].reshape(-1)[0].item())
    if isinstance(value, (list, tuple)) and env_id < len(value):
        return float(value[env_id])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _episode_done_reason(terminated: bool, truncated: bool, infos: Any, env_id: int) -> str:
    if truncated:
        return "timeout"
    if not terminated:
        return "incomplete"
    log = infos.get("log") if isinstance(infos, dict) else None
    if isinstance(log, dict):
        for key in (
            "term_com_tip",
            "term_too_far",
            "term_out_of_reach",
            "term_grasp_right",
            "term_grasp_left",
        ):
            if _as_bool_scalar(log.get(key), env_id):
                return key
    return "terminated"


@dataclass
class _RunningEpisode:
    env_id: int
    steps: int = 0
    episode_return: float = 0.0
    motion_locked: bool = False
    motion_lock_step: int | None = None
    inserted_state: list[list[bool]] = field(default_factory=list)
    d_finger: list[list[float]] = field(default_factory=list)
    sum_sq_all_joints: float = 0.0
    n_joint_samples: int = 0
    wrist_distance_at_success: float | None = None
    inserted_flags_at_success: list[bool] | None = None
    first_wrist_goal_step: int | None = None
    first_wrist_incomplete: bool = False
    inserted_flags_at_first_wrist_goal: list[bool] | None = None
    last_done_reason: str = "incomplete"


@dataclass
class EpisodeMetrics:
    episode: int
    env_id: int
    success: bool
    terminated: bool
    truncated: bool
    episode_length_steps: int
    episode_length_seconds: float
    motion_lock_step: int | None
    motion_lock_time_s: float | None
    inserted: dict[str, bool]
    insert_ratio: dict[str, float]
    insert_steps: dict[str, int]
    first_insert_time_s: dict[str, float | None]
    num_inserted_fingers: int
    final_inserted_fingers: int
    max_inserted_fingers: int
    ever_all_inserted: bool
    final_all_inserted: bool
    insertion_outcome: str
    finger_rms: dict[str, float]
    finger_peak: dict[str, float]
    hand_rms: float
    worst_finger_peak: float
    worst_finger: str
    episode_return: float
    wrist_distance_at_success: float | None = None
    inserted_fingers_at_success: int | None = None
    all_5_inserted_at_success: bool | None = None
    per_finger_inserted_at_success: dict[str, bool] | None = None
    episode_done_step: int | None = None
    episode_done_reason: str = "incomplete"
    first_wrist_goal_step: int | None = None
    first_wrist_goal_time_s: float | None = None
    inserted_fingers_at_first_wrist_goal: int | None = None
    missing_fingers_at_first_wrist_goal: str = ""

    def to_csv_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "episode": self.episode,
            "env_id": self.env_id,
            "success": int(self.success),
            "terminated": int(self.terminated),
            "truncated": int(self.truncated),
            "episode_length_steps": self.episode_length_steps,
            "episode_length_seconds": f"{self.episode_length_seconds:.6g}",
            "motion_lock_step": "" if self.motion_lock_step is None else self.motion_lock_step,
            "motion_lock_time_s": "" if self.motion_lock_time_s is None else f"{self.motion_lock_time_s:.6g}",
            "final_inserted_fingers": self.final_inserted_fingers,
            "max_inserted_fingers": self.max_inserted_fingers,
            "ever_all_inserted": int(self.ever_all_inserted),
            "final_all_inserted": int(self.final_all_inserted),
            "insertion_outcome": self.insertion_outcome,
            "num_inserted_fingers": self.num_inserted_fingers,
            "hand_rms": f"{self.hand_rms:.8g}",
            "worst_finger_peak": f"{self.worst_finger_peak:.8g}",
            "worst_finger": self.worst_finger,
            "return": f"{self.episode_return:.6g}",
        }
        for name in FINGER_ORDER:
            t0 = self.first_insert_time_s[name]
            row[f"{name}_inserted"] = int(self.inserted[name])
            row[f"{name}_first_insert_time"] = "" if t0 is None else f"{t0:.6g}"
            row[f"{name}_insert_ratio"] = f"{self.insert_ratio[name]:.6g}"
            row[f"{name}_insert_steps"] = self.insert_steps[name]
            row[f"{name}_rms"] = f"{self.finger_rms[name]:.8g}"
            row[f"{name}_peak"] = f"{self.finger_peak[name]:.8g}"
        row["pinky_inserted"] = row["little_inserted"]
        row["pinky_first_insert_time"] = row["little_first_insert_time"]
        at_success = self.per_finger_inserted_at_success or {}
        row["final_success"] = int(self.success)
        row["wrist_distance_at_success"] = (
            "" if self.wrist_distance_at_success is None else f"{self.wrist_distance_at_success:.8g}"
        )
        row["inserted_fingers_at_success"] = (
            "" if self.inserted_fingers_at_success is None else self.inserted_fingers_at_success
        )
        row["all_5_inserted_at_success"] = (
            "" if self.all_5_inserted_at_success is None else int(self.all_5_inserted_at_success)
        )
        for name in FINGER_ORDER:
            row[f"{name}_inserted_at_success"] = (
                "" if name not in at_success else int(at_success[name])
            )
        row["episode_done_step"] = "" if self.episode_done_step is None else self.episode_done_step
        row["episode_done_reason"] = self.episode_done_reason
        row["first_wrist_goal_step"] = (
            "" if self.first_wrist_goal_step is None else self.first_wrist_goal_step
        )
        row["first_wrist_goal_time_s"] = (
            "" if self.first_wrist_goal_time_s is None else f"{self.first_wrist_goal_time_s:.6g}"
        )
        row["inserted_fingers_at_first_wrist_goal"] = (
            ""
            if self.inserted_fingers_at_first_wrist_goal is None
            else self.inserted_fingers_at_first_wrist_goal
        )
        row["missing_fingers_at_first_wrist_goal"] = self.missing_fingers_at_first_wrist_goal
        return row


class BraceletEvalCollector:
    """Per-control-step bracelet evaluation; write CSV incrementally."""

    def __init__(
        self,
        raw_env: Any,
        *,
        output_dir: Path,
        control_dt: float,
        max_episodes: int,
        insertion_delta_m: float,
        insertion_confirm_frames: int,
        insertion_ellipse_threshold: float,
        eval_env_ids: list[int],
        task: str | None,
        checkpoint: str,
        executed_at: str,
        log_prefix: str = "play_eval",
        debug_insertion: bool = False,
        debug_insertion_interval: int = 10,
    ) -> None:
        self.raw_env = raw_env
        self.output_dir = Path(output_dir)
        self.control_dt = float(control_dt)
        self.max_episodes = int(max_episodes)
        self.insertion_delta_m = float(insertion_delta_m)
        self.insertion_confirm_frames = max(1, int(insertion_confirm_frames))
        self.insertion_ellipse_threshold = float(insertion_ellipse_threshold)
        self.eval_env_ids = list(eval_env_ids)
        self.task = task
        self.checkpoint = checkpoint
        self.executed_at = executed_at
        self.log_prefix = log_prefix
        self.debug_insertion = bool(debug_insertion)
        self.debug_insertion_interval = max(1, int(debug_insertion_interval))
        self._debug_prev_inserted: dict[int, list[bool]] = {eid: [False] * 5 for eid in eval_env_ids}
        self._debug_prev_side: dict[int, list[str]] = {eid: ["?"] * 5 for eid in eval_env_ids}

        self.csv_path = self.output_dir / "episode_metrics.csv"
        self.summary_path = self.output_dir / "evaluation_summary.json"
        self.partial_path = self.output_dir / "evaluation_summary.partial.json"

        self.episodes: list[EpisodeMetrics] = []
        self._running = {eid: _RunningEpisode(env_id=eid) for eid in self.eval_env_ids}
        self._tracker: FingerCrossingTracker | None = None

        self.hand = getattr(raw_env, "hand", None)
        self.finger_joint_ids: dict[str, list[int]] = {name: [] for name in FINGER_ORDER}
        self.all_finger_joint_ids: list[int] = []
        self.resolved_joint_groups: dict[str, list[str]] = {name: [] for name in FINGER_ORDER}
        self.base_body_ids: dict[str, int | None] = {name: None for name in FINGER_ORDER}
        self.resolved_base_bodies: dict[str, str | None] = {name: None for name in FINGER_ORDER}
        self.q_default: torch.Tensor | None = None
        self._bind_hand()
        self._write_csv_header()

    @classmethod
    def from_session(cls, session: PlaySession) -> BraceletEvalCollector:
        args = session.args_cli
        raw = unwrap_env(session.env)
        n_envs = int(getattr(session.env, "num_envs", 1))
        eval_env_id = getattr(args, "eval_env_id", None)
        if eval_env_id is None:
            eval_env_ids = list(range(n_envs))
        else:
            eid = int(eval_env_id)
            if eid < 0 or eid >= n_envs:
                raise ValueError(f"--eval-env-id {eid} is out of range for num_envs={n_envs}")
            eval_env_ids = [eid]

        control_dt = control_dt_from_env_cfg(session.env_cfg)
        ellipse_thr = getattr(args, "insertion_ellipse_threshold", None)
        if ellipse_thr is None:
            ellipse_thr = float(getattr(session.env_cfg, "eval_opening_ellipse_threshold", 1.0))
        delta = float(getattr(args, "insertion_delta_m", 0.003))
        confirm = int(getattr(args, "insertion_confirm_frames", 4))

        collector = cls(
            raw,
            output_dir=session.output_paths.evaluation_dir,
            control_dt=control_dt,
            max_episodes=int(args.max_episodes),
            insertion_delta_m=delta,
            insertion_confirm_frames=confirm,
            insertion_ellipse_threshold=float(ellipse_thr),
            eval_env_ids=eval_env_ids,
            task=getattr(args, "task", None),
            checkpoint=session.resume_path,
            executed_at=session.output_paths.executed_at,
            log_prefix=session.log_prefix,
            debug_insertion=bool(getattr(args, "debug_insertion", False)),
            debug_insertion_interval=int(getattr(args, "debug_insertion_interval", 10)),
        )
        require_all = bool(getattr(args, "complete_dressing_success", True))
        if hasattr(raw, "cfg") and hasattr(raw.cfg, "eval_success_requires_all_fingers"):
            raw.cfg.eval_success_requires_all_fingers = require_all
            raw.cfg.eval_insertion_delta_m = collector.insertion_delta_m
            raw.cfg.eval_insertion_confirm_frames = collector.insertion_confirm_frames
            if hasattr(raw.cfg, "eval_opening_ellipse_threshold"):
                raw.cfg.eval_opening_ellipse_threshold = collector.insertion_ellipse_threshold
        print(
            f"[{session.log_prefix}] bracelet eval: max_episodes={collector.max_episodes} "
            f"control={1.0 / collector.control_dt:.1f} Hz "
            f"crossing delta={collector.insertion_delta_m:.4g} m "
            f"confirm={collector.insertion_confirm_frames} frames "
            f"ellipse<={collector.insertion_ellipse_threshold:.3g} "
            f"success={'wrist+all_5' if require_all else 'wrist_only'} "
            f"envs={eval_env_ids}"
            + (
                f" debug_insertion every {collector.debug_insertion_interval} steps"
                if collector.debug_insertion
                else ""
            )
        )
        return collector

    def _bind_hand(self) -> None:
        hand = self.hand
        if hand is None:
            print(f"[{self.log_prefix}] WARNING: no Shadow Hand on env; insertion/deviation will be empty.")
            return
        names = list(getattr(hand, "joint_names", []) or getattr(hand.data, "joint_names", []) or [])
        name_to_idx = {n: i for i, n in enumerate(names)}
        missing: list[str] = []
        for finger, joint_names in FINGER_JOINT_NAMES.items():
            resolved = []
            ids = []
            for jn in joint_names:
                if jn in name_to_idx:
                    resolved.append(jn)
                    ids.append(name_to_idx[jn])
                else:
                    missing.append(jn)
            self.finger_joint_ids[finger] = ids
            self.resolved_joint_groups[finger] = resolved
            self.all_finger_joint_ids.extend(ids)
        if missing:
            print(f"[{self.log_prefix}] WARNING: missing Shadow Hand joints (skipped): {missing}")

        default = getattr(hand.data, "default_joint_pos", None)
        if default is not None:
            self.q_default = default[0].detach().clone()

        body_names = list(
            getattr(hand, "body_names", None) or getattr(hand.data, "body_names", None) or []
        )
        missing_base: list[str] = []
        for finger, candidates in BASE_BODY_CANDIDATES.items():
            idx = _resolve_body_index(body_names, candidates)
            self.base_body_ids[finger] = idx
            if idx is None:
                missing_base.append(finger)
                self.resolved_base_bodies[finger] = None
            else:
                self.resolved_base_bodies[finger] = body_names[idx]
        if missing_base:
            print(
                f"[{self.log_prefix}] WARNING: missing finger-base bodies for {missing_base}; "
                "those fingers will not be tracked"
            )
        print(
            f"[{self.log_prefix}] insertion points: finger-base COM "
            f"{self.resolved_base_bodies}"
        )

    def _ensure_tracker(self, like: torch.Tensor) -> FingerCrossingTracker:
        if self._tracker is None:
            n = int(getattr(self.raw_env, "num_envs", like.shape[0]))
            self._tracker = FingerCrossingTracker(
                n,
                like.device,
                like.dtype,
                delta=self.insertion_delta_m,
                confirm_frames=self.insertion_confirm_frames,
                ellipse_threshold=self.insertion_ellipse_threshold,
            )
        return self._tracker

    def _stack_finger_base_env_local(self) -> torch.Tensor | None:
        """Return ``(num_envs, 5, 3)`` finger-base COMs in env-local frame, or None."""
        hand = self.hand
        if hand is None or any(self.base_body_ids[name] is None for name in FINGER_ORDER):
            return None
        body_pos_w = getattr(hand.data, "body_pos_w", None)
        if body_pos_w is None:
            return None
        origins = getattr(self.raw_env, "env_origins", None)
        if origins is None:
            scene = getattr(self.raw_env, "scene", None)
            origins = getattr(scene, "env_origins", None) if scene is not None else None
        if origins is None:
            origins = body_pos_w.new_zeros((body_pos_w.shape[0], 3))
        origins = origins.to(device=body_pos_w.device, dtype=body_pos_w.dtype)
        cols = []
        for name in FINGER_ORDER:
            idx = self.base_body_ids[name]
            cols.append(body_pos_w[:, idx] - origins)
        return torch.stack(cols, dim=1)

    def _write_csv_header(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()

    def _append_csv(self, ep: EpisodeMetrics) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writerow(ep.to_csv_row())

    def is_complete(self) -> bool:
        return len(self.episodes) >= self.max_episodes

    def on_after_step(
        self,
        *,
        timestep: int,
        rewards: Any,
        terminated: Any,
        truncated: Any,
        infos: Any,
    ) -> None:
        if self.is_complete():
            return
        raw = self.raw_env
        done = torch.logical_or(terminated, truncated)
        inserted = self._update_insertion(done)
        d_finger, sum_sq, n_j = self._compute_joint_deviation()

        for env_id in self.eval_env_ids:
            if self.is_complete():
                break
            run = self._running[env_id]
            run.steps += 1
            if isinstance(rewards, torch.Tensor) and env_id < rewards.shape[0]:
                run.episode_return += float(rewards[env_id].reshape(-1)[0].item())

            if inserted is not None:
                flags = [bool(inserted[env_id, i].item()) for i in range(5)]
                run.inserted_state.append(flags)
                if self.debug_insertion and env_id == self.eval_env_ids[0]:
                    self._debug_insertion_env(env_id, run.steps, flags)
            else:
                flags = [False] * 5
                run.inserted_state.append(flags)

            log = infos.get("log") if isinstance(infos, dict) else None
            if not isinstance(log, dict):
                extras = getattr(raw, "extras", None) or {}
                log = extras.get("log") if isinstance(extras, dict) else {}
            wrist_ok = _as_bool_scalar((log or {}).get("wrist_within_goal"), env_id)
            if not wrist_ok:
                dist = getattr(raw, "wrist_center_euclidean_distance", None)
                wrist_dist = _read_float_env(dist, env_id)
                thr = float(getattr(getattr(raw, "cfg", None), "bracelet_success_threshold", 0.01))
                if wrist_dist is not None:
                    wrist_ok = wrist_dist < thr
            if wrist_ok and run.first_wrist_goal_step is None and not all(flags):
                run.first_wrist_goal_step = run.steps - 1
                run.first_wrist_incomplete = True
                run.inserted_flags_at_first_wrist_goal = list(flags)

            if (not run.motion_locked) and read_motion_locked(infos, raw, env_id):
                run.motion_locked = True
                run.motion_lock_step = run.steps - 1
                run.inserted_flags_at_success = list(flags)
                dist = None
                if isinstance(log, dict):
                    dist = _read_float_env(log.get("wrist_center_distance"), env_id)
                if dist is None:
                    dist = _read_float_env(getattr(raw, "wrist_center_euclidean_distance", None), env_id)
                run.wrist_distance_at_success = dist

            if d_finger is not None:
                run.d_finger.append([float(d_finger[env_id, i].item()) for i in range(5)])
                run.sum_sq_all_joints += float(sum_sq[env_id].item())
                run.n_joint_samples += int(n_j)
            else:
                run.d_finger.append([0.0] * 5)

            env_done = bool(done[env_id].item())
            if env_done:
                run.last_done_reason = _episode_done_reason(
                    bool(terminated[env_id].item()),
                    bool(truncated[env_id].item()),
                    infos,
                    env_id,
                )
                self._finalize_env(
                    env_id,
                    terminated=bool(terminated[env_id].item()),
                    truncated=bool(truncated[env_id].item()),
                )

    def on_periodic_hard_reset(self) -> None:
        for env_id in self.eval_env_ids:
            if self.is_complete():
                break
            if self._running[env_id].steps > 0:
                self._running[env_id].last_done_reason = "timeout"
                self._finalize_env(env_id, terminated=False, truncated=True)

    def _finalize_env(self, env_id: int, *, terminated: bool, truncated: bool) -> None:
        if self.is_complete():
            return
        run = self._running[env_id]
        if run.steps <= 0:
            self._running[env_id] = _RunningEpisode(env_id=env_id)
            self._debug_prev_inserted[env_id] = [False] * 5
            self._debug_prev_side[env_id] = ["?"] * 5
            if self._tracker is not None:
                self._tracker.reset_envs([env_id])
            return
        ep = self._build_episode(run, terminated=terminated, truncated=truncated)
        self.episodes.append(ep)
        self._append_csv(ep)
        self._write_summary(partial=True)
        print(
            f"[{self.log_prefix}] episode {ep.episode}: success={int(ep.success)} "
            f"final={ep.final_inserted_fingers}/5 max={ep.max_inserted_fingers}/5 "
            f"outcome={ep.insertion_outcome} lock_step={ep.motion_lock_step} "
            f"done={ep.episode_done_reason} steps={ep.episode_length_steps} env={env_id}"
            + (
                f" first_wrist_incomplete={ep.missing_fingers_at_first_wrist_goal}"
                if ep.first_wrist_goal_step is not None
                else ""
            )
        )
        self._running[env_id] = _RunningEpisode(env_id=env_id)
        self._debug_prev_inserted[env_id] = [False] * 5
        self._debug_prev_side[env_id] = ["?"] * 5
        if self._tracker is not None:
            self._tracker.reset_envs([env_id])

    def _snapshot_tracker(self, env_id: int) -> tuple[int, bool, dict[str, float | None], dict[str, int]]:
        first_t = {name: None for name in FINGER_ORDER}
        insert_steps = {name: 0 for name in FINGER_ORDER}
        max_n = 0
        ever_all = False
        raw = self.raw_env
        end_max = getattr(raw, "_episode_end_eval_max_inserted", None)
        end_ever = getattr(raw, "_episode_end_eval_ever_all", None)
        end_first = getattr(raw, "_episode_end_eval_first_insert_step", None)
        end_steps = getattr(raw, "_episode_end_eval_insert_steps", None)
        if end_max is not None and env_id < int(end_max.shape[0]):
            max_n = int(end_max[env_id].item())
            ever_all = bool(end_ever[env_id].item()) if end_ever is not None else False
            for i, name in enumerate(FINGER_ORDER):
                if end_first is not None:
                    step_i = int(end_first[env_id, i].item())
                    if step_i >= 0:
                        first_t[name] = step_i * self.control_dt
                if end_steps is not None:
                    insert_steps[name] = int(end_steps[env_id, i].item())
            return max_n, ever_all, first_t, insert_steps

        tracker = self._tracker
        if tracker is None:
            return max_n, ever_all, first_t, insert_steps
        max_n = int(tracker.max_inserted[env_id].item())
        ever_all = bool(tracker.ever_all[env_id].item())
        for i, name in enumerate(FINGER_ORDER):
            step_i = int(tracker.first_insert_step[env_id, i].item())
            if step_i >= 0:
                first_t[name] = step_i * self.control_dt
            insert_steps[name] = int(tracker.inserted_steps[env_id, i].item())
        return max_n, ever_all, first_t, insert_steps

    def _build_episode(self, run: _RunningEpisode, *, terminated: bool, truncated: bool) -> EpisodeMetrics:
        max_n, ever_all, first_t, insert_steps = self._snapshot_tracker(run.env_id)
        inserted = {name: False for name in FINGER_ORDER}
        insert_ratio = {name: 0.0 for name in FINGER_ORDER}
        if run.inserted_state:
            last = run.inserted_state[-1]
            for i, name in enumerate(FINGER_ORDER):
                inserted[name] = bool(last[i])
                insert_ratio[name] = insert_steps[name] / float(max(run.steps, 1))
        if max_n == 0:
            max_n = max((sum(1 for v in row if v) for row in run.inserted_state), default=0)
            ever_all = ever_all or any(all(row) for row in run.inserted_state)
        for i, name in enumerate(FINGER_ORDER):
            if first_t[name] is None:
                for step_i, row in enumerate(run.inserted_state):
                    if row[i]:
                        first_t[name] = step_i * self.control_dt
                        break
            if insert_steps[name] == 0 and run.inserted_state:
                insert_steps[name] = sum(1 for row in run.inserted_state if row[i])

        finger_rms = {name: 0.0 for name in FINGER_ORDER}
        finger_peak = {name: 0.0 for name in FINGER_ORDER}
        if run.d_finger:
            t = len(run.d_finger)
            for i, name in enumerate(FINGER_ORDER):
                series = [row[i] for row in run.d_finger]
                finger_rms[name] = _rad_to_deg(math.sqrt(sum(v * v for v in series) / t))
                finger_peak[name] = _rad_to_deg(max(series))

        hand_rms = 0.0
        if run.n_joint_samples > 0:
            hand_rms = _rad_to_deg(math.sqrt(run.sum_sq_all_joints / float(run.n_joint_samples)))

        worst_finger = max(FINGER_ORDER, key=lambda n: finger_peak[n])
        lock_t = None if run.motion_lock_step is None else run.motion_lock_step * self.control_dt
        final_n = sum(1 for name in FINGER_ORDER if inserted[name])
        final_all = final_n == 5
        success = bool(run.motion_locked)
        outcome = classify_insertion_outcome(
            max_inserted=max_n,
            ever_all=ever_all,
            final_all=final_all,
            success=success,
        )
        flags_at_success = run.inserted_flags_at_success
        per_success = (
            {name: bool(flags_at_success[i]) for i, name in enumerate(FINGER_ORDER)}
            if flags_at_success is not None
            else None
        )
        first_wrist_t = (
            None if run.first_wrist_goal_step is None else run.first_wrist_goal_step * self.control_dt
        )
        first_flags = run.inserted_flags_at_first_wrist_goal
        return EpisodeMetrics(
            episode=len(self.episodes),
            env_id=run.env_id,
            success=success,
            terminated=terminated,
            truncated=truncated,
            episode_length_steps=run.steps,
            episode_length_seconds=run.steps * self.control_dt,
            motion_lock_step=run.motion_lock_step,
            motion_lock_time_s=lock_t,
            inserted=inserted,
            insert_ratio=insert_ratio,
            insert_steps=insert_steps,
            first_insert_time_s=first_t,
            num_inserted_fingers=final_n,
            final_inserted_fingers=final_n,
            max_inserted_fingers=max_n,
            ever_all_inserted=ever_all,
            final_all_inserted=final_all,
            insertion_outcome=outcome,
            finger_rms=finger_rms,
            finger_peak=finger_peak,
            hand_rms=hand_rms,
            worst_finger_peak=finger_peak[worst_finger],
            worst_finger=worst_finger,
            episode_return=run.episode_return,
            wrist_distance_at_success=run.wrist_distance_at_success,
            inserted_fingers_at_success=(
                None if flags_at_success is None else sum(1 for v in flags_at_success if v)
            ),
            all_5_inserted_at_success=(
                None if flags_at_success is None else all(flags_at_success)
            ),
            per_finger_inserted_at_success=per_success,
            episode_done_step=max(run.steps - 1, 0),
            episode_done_reason=run.last_done_reason if run.last_done_reason else (
                "timeout" if truncated else ("terminated" if terminated else "incomplete")
            ),
            first_wrist_goal_step=run.first_wrist_goal_step if run.first_wrist_incomplete else None,
            first_wrist_goal_time_s=first_wrist_t if run.first_wrist_incomplete else None,
            inserted_fingers_at_first_wrist_goal=(
                None if first_flags is None else sum(1 for v in first_flags if v)
            ),
            missing_fingers_at_first_wrist_goal=(
                _missing_finger_names(first_flags) if first_flags is not None else ""
            ),
        )

    def _update_insertion(self, done: torch.Tensor) -> torch.Tensor | None:
        """Update the crossing tracker. Skip envs that just reset (``done``)."""
        raw = self.raw_env
        if getattr(raw, "_is_free_space_mode", lambda: False)():
            return None
        cfg = getattr(raw, "cfg", None)
        if bool(getattr(cfg, "eval_success_requires_all_fingers", True)):
            env_tracker = getattr(raw, "_eval_finger_crossing_tracker", None)
            if env_tracker is not None:
                self._tracker = env_tracker
            end = getattr(raw, "_episode_end_eval_inserted", None)
            if end is not None:
                return end
        distal = self._stack_finger_base_env_local()
        cent = getattr(raw, "goal_cent_pos", None)
        east = getattr(raw, "goal_east_pos", None)
        west = getattr(raw, "goal_west_pos", None)
        north = getattr(raw, "goal_north_pos", None)
        south = getattr(raw, "goal_south_pos", None)
        if distal is None or cent is None or east is None or west is None or north is None or south is None:
            return None

        radius_y, radius_z = opening_radii(east, west, north, south)
        tracker = self._ensure_tracker(distal)
        active = torch.zeros((tracker.num_envs,), dtype=torch.bool, device=distal.device)
        for env_id in self.eval_env_ids:
            if env_id < tracker.num_envs and not bool(done[env_id].item()):
                active[env_id] = True
        return tracker.update(distal, cent, radius_y, radius_z, active)

    def _debug_insertion_env(self, env_id: int, steps: int, flags: list[bool]) -> None:
        """Print per-finger signed distance / ellipse / latch for one env."""
        raw = self.raw_env
        distal = self._stack_finger_base_env_local()
        cent = getattr(raw, "goal_cent_pos", None)
        east = getattr(raw, "goal_east_pos", None)
        west = getattr(raw, "goal_west_pos", None)
        north = getattr(raw, "goal_north_pos", None)
        south = getattr(raw, "goal_south_pos", None)
        if distal is None or cent is None or east is None or west is None or north is None or south is None:
            return
        if env_id >= int(distal.shape[0]):
            return

        radius_y, radius_z = opening_radii(east, west, north, south)
        p = distal[env_id]
        c = cent[env_id]
        d = p[:, 0] - c[0]
        ev = ellipse_value_yz(p.unsqueeze(0), c.unsqueeze(0), radius_y[env_id : env_id + 1], radius_z[env_id : env_id + 1])[0]
        delta = self.insertion_delta_m
        prev = self._debug_prev_inserted.get(env_id, [False] * 5)
        prev_side = self._debug_prev_side.get(env_id, ["?"] * 5)
        sides: list[str] = []
        for i in range(5):
            di = float(d[i].item())
            if di > delta:
                sides.append("PRE")
            elif di < -delta:
                sides.append("POST")
            else:
                sides.append("BAND")
        latch_changed = [flags[i] != prev[i] for i in range(5)]
        side_changed = [sides[i] != prev_side[i] for i in range(5)]
        periodic = (steps % self.debug_insertion_interval) == 0
        if not periodic and not any(latch_changed) and not any(side_changed):
            self._debug_prev_inserted[env_id] = list(flags)
            self._debug_prev_side[env_id] = sides
            return

        n_in = sum(1 for v in flags if v)
        t_s = steps * self.control_dt
        print(
            f"[{self.log_prefix} insert] ep={len(self.episodes)} env={env_id} "
            f"t={t_s:.2f}s step={steps}  inserted={n_in}/5  "
            f"c=({c[0].item():.3f},{c[1].item():.3f},{c[2].item():.3f}) "
            f"r_yz=({radius_y[env_id].item():.3f},{radius_z[env_id].item():.3f})"
        )
        tracker = self._tracker
        for i, name in enumerate(FINGER_ORDER):
            di = float(d[i].item())
            evi = float(ev[i].item())
            hole = "in " if evi <= self.insertion_ellipse_threshold else "out"
            latch = "IN" if flags[i] else "--"
            pend = ""
            if tracker is not None:
                if bool(tracker.fwd_pending[env_id, i].item()):
                    pend = f" fwd {int(tracker.fwd_count[env_id, i].item())}/{tracker.confirm_frames}"
                elif bool(tracker.rev_pending[env_id, i].item()):
                    pend = f" rev {int(tracker.rev_count[env_id, i].item())}/{tracker.confirm_frames}"
            event = ""
            if side_changed[i] and prev_side[i] != "?":
                event = f"  << CROSS {prev_side[i]}->{sides[i]}"
            if latch_changed[i]:
                event += "  << INSERT" if flags[i] else "  << EXIT"
            print(
                f"  {FINGER_LABELS[name]:<6} {latch}  {sides[i]:<4}  "
                f"d={di:+.4f}m  ev={evi:.2f} {hole}{pend}{event}"
            )
        self._debug_prev_inserted[env_id] = list(flags)
        self._debug_prev_side[env_id] = sides

    def _compute_joint_deviation(self) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        """Per-finger RMS ``D_f`` and per-env sum of squared joint errors.

        ``D_f(t) = sqrt(mean_j (q_j(t) - q_j_default)^2)`` for joints of finger f.
        """
        hand = self.hand
        if hand is None or self.q_default is None or not self.all_finger_joint_ids:
            return None, None, 0
        q = hand.data.joint_pos
        q0 = self.q_default.to(device=q.device, dtype=q.dtype)
        d_cols = []
        sum_sq = torch.zeros((q.shape[0],), device=q.device, dtype=q.dtype)
        n_j = 0
        for finger in FINGER_ORDER:
            ids = self.finger_joint_ids[finger]
            if not ids:
                d_cols.append(torch.zeros((q.shape[0],), device=q.device, dtype=q.dtype))
                continue
            idx = torch.as_tensor(ids, device=q.device, dtype=torch.long)
            delta = q[:, idx] - q0[idx]
            d_cols.append(torch.sqrt(torch.mean(delta * delta, dim=-1)))
            sum_sq = sum_sq + torch.sum(delta * delta, dim=-1)
            n_j += len(ids)
        return torch.stack(d_cols, dim=1), sum_sq, n_j

    def build_summary(self) -> dict[str, Any]:
        eps = self.episodes
        n = len(eps)
        success_count = sum(1 for ep in eps if ep.success)
        inserted_counts = {name: sum(1 for ep in eps if ep.inserted[name]) for name in FINGER_ORDER}
        ever_counts = {
            name: sum(1 for ep in eps if ep.first_insert_time_s[name] is not None) for name in FINGER_ORDER
        }
        n_final = [float(ep.final_inserted_fingers) for ep in eps]
        n_max = [float(ep.max_inserted_fingers) for ep in eps]
        hand_rms = [ep.hand_rms for ep in eps]
        outcome_counts = {key: sum(1 for ep in eps if ep.insertion_outcome == key) for key in INSERTION_OUTCOMES}

        deform_fingers: dict[str, Any] = {}
        insert_fingers: dict[str, Any] = {}
        for name in FINGER_ORDER:
            rms = [ep.finger_rms[name] for ep in eps]
            peak = [ep.finger_peak[name] for ep in eps]
            times = [t for t in (ep.first_insert_time_s[name] for ep in eps) if t is not None]
            deform_fingers[name] = {
                "mean_rms": (sum(rms) / n) if n else 0.0,
                "std_rms": sample_std(rms),
                "mean_peak": (sum(peak) / n) if n else 0.0,
                "max_peak": max(peak) if peak else 0.0,
            }
            insert_fingers[name] = {
                "ever_count": ever_counts[name],
                "ever_rate": (ever_counts[name] / n) if n else 0.0,
                "final_count": inserted_counts[name],
                "final_rate": (inserted_counts[name] / n) if n else 0.0,
                "mean_first_insert_time_s": (sum(times) / len(times)) if times else None,
            }
        insert_fingers["pinky"] = dict(insert_fingers["little"])

        worst_ep = max(eps, key=lambda e: e.worst_finger_peak) if eps else None
        episode_debug = [
            {
                "env_id": ep.env_id,
                "episode": ep.episode,
                "motion_lock_step": ep.motion_lock_step,
                "motion_lock_time_s": ep.motion_lock_time_s,
                "wrist_distance_at_success": ep.wrist_distance_at_success,
                "inserted_fingers_at_success": ep.inserted_fingers_at_success,
                "all_5_inserted_at_success": ep.all_5_inserted_at_success,
                "per_finger_inserted_at_success": ep.per_finger_inserted_at_success,
                "episode_done_step": ep.episode_done_step,
                "episode_done_reason": ep.episode_done_reason,
                "final_success": ep.success,
                "first_wrist_goal_step": ep.first_wrist_goal_step,
                "first_wrist_goal_time_s": ep.first_wrist_goal_time_s,
                "inserted_fingers_at_first_wrist_goal": ep.inserted_fingers_at_first_wrist_goal,
                "missing_fingers_at_first_wrist_goal": ep.missing_fingers_at_first_wrist_goal,
            }
            for ep in eps
        ]
        return {
            "schema_version": 4,
            "task": self.task,
            "checkpoint": self.checkpoint,
            "executed_at": self.executed_at,
            "control_frequency_hz": 1.0 / self.control_dt if self.control_dt else None,
            "control_dt": self.control_dt,
            "config": {
                "max_episodes": self.max_episodes,
                "num_envs": int(getattr(self.raw_env, "num_envs", 1)),
                "eval_env_ids": self.eval_env_ids,
                "insertion_delta_m": self.insertion_delta_m,
                "insertion_confirm_frames": self.insertion_confirm_frames,
                "insertion_ellipse_threshold": self.insertion_ellipse_threshold,
                "success_definition": (
                    "wrist_within_goal_and_all_5_fingers_inserted"
                    if bool(getattr(getattr(self.raw_env, "cfg", None), "eval_success_requires_all_fingers", True))
                    else "motion_lock_triggered"
                ),
                "motion_lock_definition": (
                    "wrist_within_goal_and_all_5_fingers_inserted"
                    if bool(getattr(getattr(self.raw_env, "cfg", None), "eval_success_requires_all_fingers", True))
                    else "wrist_within_goal"
                ),
                "insertion_definition": "last_clear_pre_to_post_through_live_yz_ellipse",
                "insertion_normal": "+x",
                "insertion_pre_side": "d > +delta (hand / +X of opening)",
                "insertion_post_side": "d < -delta (through / -X of opening)",
                "finger_representative_point": "finger_base_com",
                "finger_base_bodies": dict(self.resolved_base_bodies),
                "deformation_definition": "rms_joint_deviation_from_hand_default_joint_pos",
                "finger_joint_groups": self.resolved_joint_groups,
            },
            "success": {
                "num_episodes": n,
                "num_success": success_count,
                "num_failed": n - success_count,
                "success_rate": (success_count / n) if n else 0.0,
            },
            "insertion": {
                "mean_final_inserted_fingers": (sum(n_final) / n) if n else 0.0,
                "std_final_inserted_fingers": sample_std(n_final),
                "mean_max_inserted_fingers": (sum(n_max) / n) if n else 0.0,
                "std_max_inserted_fingers": sample_std(n_max),
                "mean_inserted_fingers": (sum(n_final) / n) if n else 0.0,
                "std_inserted_fingers": sample_std(n_final),
                "ever_all_inserted_count": sum(1 for ep in eps if ep.ever_all_inserted),
                "ever_all_inserted_rate": (sum(1 for ep in eps if ep.ever_all_inserted) / n) if n else 0.0,
                "final_all_inserted_count": sum(1 for ep in eps if ep.final_all_inserted),
                "final_all_inserted_rate": (sum(1 for ep in eps if ep.final_all_inserted) / n) if n else 0.0,
                "outcomes": {
                    key: {
                        "count": outcome_counts[key],
                        "rate": (outcome_counts[key] / n) if n else 0.0,
                    }
                    for key in INSERTION_OUTCOMES
                },
                "fingers": insert_fingers,
            },
            "deformation": {
                "unit": "deg",
                "fingers": deform_fingers,
                "hand": {
                    "mean_rms": (sum(hand_rms) / n) if n else 0.0,
                    "std_rms": sample_std(hand_rms),
                    "max_worst_finger_peak": worst_ep.worst_finger_peak if worst_ep else 0.0,
                    "max_worst_finger": worst_ep.worst_finger if worst_ep else "",
                    "max_worst_finger_episode": worst_ep.episode if worst_ep else None,
                },
            },
            "episode_debug": episode_debug,
        }

    def _write_summary(self, *, partial: bool) -> None:
        path = self.partial_path if partial else self.summary_path
        path.write_text(json.dumps(self.build_summary(), indent=2) + "\n", encoding="utf-8")

    def finalize(self) -> dict[str, Any]:
        for env_id in self.eval_env_ids:
            if self.is_complete():
                break
            if self._running[env_id].steps > 0:
                self._finalize_env(env_id, terminated=False, truncated=False)
        summary = self.build_summary()
        self._write_summary(partial=False)
        self.partial_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print_evaluation_summary(summary, output_dir=self.output_dir)
        print(f"[{self.log_prefix}] episode metrics -> {self.csv_path}")
        print(f"[{self.log_prefix}] evaluation summary -> {self.summary_path}")
        return summary


def print_evaluation_summary(summary: dict[str, Any], *, output_dir: Path | None = None) -> None:
    cfg = summary.get("config") or {}
    success = summary.get("success") or {}
    insertion = summary.get("insertion") or {}
    deform = summary.get("deformation") or {}
    fingers_ins = insertion.get("fingers") or {}
    fingers_def = deform.get("fingers") or {}
    hand = deform.get("hand") or {}
    outcomes = insertion.get("outcomes") or {}
    n = int(success.get("num_episodes") or 0)
    n_ok = int(success.get("num_success") or 0)
    n_fail = int(success.get("num_failed") or 0)
    rate = float(success.get("success_rate") or 0.0) * 100.0
    freq = summary.get("control_frequency_hz")
    delta = cfg.get("insertion_delta_m")
    confirm = cfg.get("insertion_confirm_frames")

    print("")
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    if output_dir is not None:
        print(f"Output : {output_dir}")
    if freq and delta is not None:
        print(
            f"Control: {float(freq):.1f} Hz   crossing: |d| > {float(delta):.4g} m "
            f"confirm {int(confirm)} frames   ellipse <= {float(cfg.get('insertion_ellipse_threshold') or 1.0):.3g}"
        )
    print("")
    print("Episodes")
    print(f"  Success definition : {cfg.get('success_definition')}")
    print(f"  Motion lock        : {cfg.get('motion_lock_definition')}")
    print(f"  Total   : {n}")
    print(f"  Success : {n_ok}")
    print(f"  Failed  : {n_fail}")
    print(f"  Rate    : {rate:.1f} %")
    print("")
    print("-" * 60)
    print("Finger Insertion  (finger-base crossing through live opening)")
    print("-" * 60)
    print(f"{'Finger':<10}  {'Ever':>14}     {'Final':>14}")
    for name in FINGER_ORDER:
        block = fingers_ins.get(name) or {}
        ever_c = int(block.get("ever_count") or 0)
        ever_r = float(block.get("ever_rate") or 0.0) * 100.0
        final_c = int(block.get("final_count") or block.get("count") or 0)
        final_r = float(block.get("final_rate") or block.get("rate") or 0.0) * 100.0
        print(
            f"{FINGER_LABELS.get(name, name.capitalize()):<10}  "
            f"{ever_c:4d} / {n:<4d} {ever_r:5.1f}%   "
            f"{final_c:4d} / {n:<4d} {final_r:5.1f}%"
        )
    print("")
    print(
        f"Mean final inserted fingers : "
        f"{float(insertion.get('mean_final_inserted_fingers') or insertion.get('mean_inserted_fingers') or 0.0):.2f} "
        f"± {float(insertion.get('std_final_inserted_fingers') or insertion.get('std_inserted_fingers') or 0.0):.2f} / 5"
    )
    print(
        f"Mean max inserted fingers   : "
        f"{float(insertion.get('mean_max_inserted_fingers') or 0.0):.2f} "
        f"± {float(insertion.get('std_max_inserted_fingers') or 0.0):.2f} / 5"
    )
    print(
        f"Ever all five               : "
        f"{int(insertion.get('ever_all_inserted_count') or 0)} / {n}   "
        f"{float(insertion.get('ever_all_inserted_rate') or 0.0) * 100.0:.1f} %"
    )
    print(
        f"Final all five              : "
        f"{int(insertion.get('final_all_inserted_count') or 0)} / {n}   "
        f"{float(insertion.get('final_all_inserted_rate') or 0.0) * 100.0:.1f} %"
    )
    print("")
    print("Insertion outcomes")
    outcome_labels = {
        "none": "no insertion",
        "partial": "partial insertion",
        "all_exited": "all five then one+ exited",
        "all_retained": "all five retained to end",
        "all_retained_and_success": "all five retained + wrist success",
    }
    for key in INSERTION_OUTCOMES:
        block = outcomes.get(key) or {}
        print(
            f"  {outcome_labels[key]:<34}  "
            f"{int(block.get('count') or 0):4d} / {n:<4d}  "
            f"{float(block.get('rate') or 0.0) * 100.0:5.1f} %"
        )
    print("")
    print("-" * 60)
    print("Finger Joint Deviation  (from hand.data.default_joint_pos, deg)")
    print("-" * 60)
    print(f"{'Finger':<10}  {'Mean RMS':>10}  {'Std RMS':>10}  {'Mean Peak':>10}  {'Max Peak':>10}")
    for name in FINGER_ORDER:
        block = fingers_def.get(name) or {}
        print(
            f"{FINGER_LABELS.get(name, name.capitalize()):<10}  "
            f"{float(block.get('mean_rms') or 0.0):8.3f}    "
            f"{float(block.get('std_rms') or 0.0):8.3f}    "
            f"{float(block.get('mean_peak') or 0.0):8.3f}    "
            f"{float(block.get('max_peak') or 0.0):8.3f}"
        )
    print("")
    print("Hand RMS")
    print(f"  {float(hand.get('mean_rms') or 0.0):.3f} ± {float(hand.get('std_rms') or 0.0):.3f} deg")
    print("")
    print("Worst observed finger peak")
    print(
        f"  {float(hand.get('max_worst_finger_peak') or 0.0):.3f} deg   "
        f"finger={hand.get('max_worst_finger') or '-'}   "
        f"episode={hand.get('max_worst_finger_episode')}"
    )
    print("=" * 60)
    print("")
