"""Bracelet-task evaluation: motion-lock success, finger insertion, joint deviation.

Insertion geometry (env-local, same convention as reach_*_bracelet):
  * opening plane: X = goal_cent.x, normal = +X (insertion axis)
  * opening boundary: live Y-Z ellipse from N/S/E/W rim goals
  * finger segment: middle-phalanx COM -> distal COM (short distal-phalanx centerline)
  * inserted_raw iff the A->B line hits the plane at P inside the ellipse and
    the distal has reached or passed the plane (signed X >= 0)
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

# Actuated Shadow Hand finger joints (matches Reach*BraceletEnvCfg.finger_joint_names).
# Wrist WRJ* and coupled DIP *J0 are excluded.
FINGER_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "thumb": ("robot0_THJ4", "robot0_THJ3", "robot0_THJ2", "robot0_THJ1", "robot0_THJ0"),
    "index": ("robot0_FFJ3", "robot0_FFJ2", "robot0_FFJ1"),
    "middle": ("robot0_MFJ3", "robot0_MFJ2", "robot0_MFJ1"),
    "ring": ("robot0_RFJ3", "robot0_RFJ2", "robot0_RFJ1"),
    "little": ("robot0_LFJ4", "robot0_LFJ3", "robot0_LFJ2", "robot0_LFJ1"),
}

DISTAL_POS_ATTRS: dict[str, str] = {
    "thumb": "thumb_goal_pos",
    "index": "fore_goal_pos",
    "middle": "middle_goal_pos",
    "ring": "ring_goal_pos",
    "little": "pinky_goal_pos",
}

# Shadow Hand USD body names; first match wins. Looked up at runtime.
MIDDLE_BODY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "thumb": ("robot0_thmiddle", "robot0_thhub", "robot0_thproximal"),
    "index": ("robot0_ffmiddle", "robot0_ffproximal"),
    "middle": ("robot0_mfmiddle", "robot0_mfproximal"),
    "ring": ("robot0_rfmiddle", "robot0_rfproximal"),
    "little": ("robot0_lfmiddle", "robot0_lfproximal"),
}

# When no middle body exists, step this far from distal toward the wrist (meters).
_FALLBACK_SEGMENT_M = 0.02
_PLANE_PARALLEL_EPS = 1e-8

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
    "num_inserted_fingers",
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
    "insertion_window_steps",
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


def _stack_distal_env_local(raw_env: Any) -> torch.Tensor | None:
    """Return ``(num_envs, 5, 3)`` distal positions in env-local frame, or None."""
    cols = []
    for name in FINGER_ORDER:
        attr = DISTAL_POS_ATTRS[name]
        tensor = getattr(raw_env, attr, None)
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return None
        cols.append(tensor)
    return torch.stack(cols, dim=1)


@dataclass
class _RunningEpisode:
    env_id: int
    steps: int = 0
    episode_return: float = 0.0
    motion_locked: bool = False
    motion_lock_step: int | None = None
    inserted_raw: list[list[bool]] = field(default_factory=list)
    d_finger: list[list[float]] = field(default_factory=list)
    sum_sq_all_joints: float = 0.0
    n_joint_samples: int = 0


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
    num_inserted_fingers: int
    insertion_window_steps: int
    finger_rms: dict[str, float]
    finger_peak: dict[str, float]
    hand_rms: float
    worst_finger_peak: float
    worst_finger: str
    episode_return: float

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
            "num_inserted_fingers": self.num_inserted_fingers,
            "insertion_window_steps": self.insertion_window_steps,
            "hand_rms": f"{self.hand_rms:.8g}",
            "worst_finger_peak": f"{self.worst_finger_peak:.8g}",
            "worst_finger": self.worst_finger,
            "return": f"{self.episode_return:.6g}",
        }
        for name in FINGER_ORDER:
            row[f"{name}_inserted"] = int(self.inserted[name])
            row[f"{name}_insert_ratio"] = f"{self.insert_ratio[name]:.6g}"
            row[f"{name}_insert_steps"] = self.insert_steps[name]
            row[f"{name}_rms"] = f"{self.finger_rms[name]:.8g}"
            row[f"{name}_peak"] = f"{self.finger_peak[name]:.8g}"
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
        insertion_window_sec: float,
        insertion_ratio_threshold: float,
        insertion_ellipse_threshold: float,
        eval_env_ids: list[int],
        task: str | None,
        checkpoint: str,
        executed_at: str,
        log_prefix: str = "play_eval",
    ) -> None:
        self.raw_env = raw_env
        self.output_dir = Path(output_dir)
        self.control_dt = float(control_dt)
        self.max_episodes = int(max_episodes)
        self.insertion_window_sec = float(insertion_window_sec)
        self.insertion_ratio_threshold = float(insertion_ratio_threshold)
        self.insertion_ellipse_threshold = float(insertion_ellipse_threshold)
        self.eval_env_ids = list(eval_env_ids)
        self.task = task
        self.checkpoint = checkpoint
        self.executed_at = executed_at
        self.log_prefix = log_prefix

        self.insertion_window_steps = max(1, int(round(self.insertion_window_sec / max(self.control_dt, 1e-9))))
        self.csv_path = self.output_dir / "episode_metrics.csv"
        self.summary_path = self.output_dir / "evaluation_summary.json"
        self.partial_path = self.output_dir / "evaluation_summary.partial.json"

        self.episodes: list[EpisodeMetrics] = []
        self._running = {eid: _RunningEpisode(env_id=eid) for eid in self.eval_env_ids}

        self.hand = getattr(raw_env, "hand", None)
        self.finger_joint_ids: dict[str, list[int]] = {name: [] for name in FINGER_ORDER}
        self.all_finger_joint_ids: list[int] = []
        self.resolved_joint_groups: dict[str, list[str]] = {name: [] for name in FINGER_ORDER}
        self.middle_body_ids: dict[str, int | None] = {name: None for name in FINGER_ORDER}
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

        collector = cls(
            raw,
            output_dir=session.output_paths.evaluation_dir,
            control_dt=control_dt,
            max_episodes=int(args.max_episodes),
            insertion_window_sec=float(args.insertion_window_sec),
            insertion_ratio_threshold=float(args.insertion_ratio_threshold),
            insertion_ellipse_threshold=float(ellipse_thr),
            eval_env_ids=eval_env_ids,
            task=getattr(args, "task", None),
            checkpoint=session.resume_path,
            executed_at=session.output_paths.executed_at,
            log_prefix=session.log_prefix,
        )
        print(
            f"[{session.log_prefix}] bracelet eval: max_episodes={collector.max_episodes} "
            f"control={1.0 / collector.control_dt:.1f} Hz "
            f"window={collector.insertion_window_sec:.2f}s ({collector.insertion_window_steps} steps) "
            f"ratio>={collector.insertion_ratio_threshold:.2f} "
            f"ellipse<={collector.insertion_ellipse_threshold:.3g} "
            f"envs={eval_env_ids}"
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

        body_names = list(getattr(hand, "body_names", []) or getattr(hand.data, "body_names", []) or [])
        found_bodies: dict[str, str] = {}
        for finger, cands in MIDDLE_BODY_CANDIDATES.items():
            idx = _resolve_body_index(body_names, cands)
            self.middle_body_ids[finger] = idx
            if idx is not None:
                found_bodies[finger] = body_names[idx]
        if found_bodies:
            print(f"[{self.log_prefix}] finger middle bodies: {found_bodies}")
        else:
            print(
                f"[{self.log_prefix}] no middle-phalanx bodies found; "
                f"using distal-to-wrist fallback segment ({_FALLBACK_SEGMENT_M * 100:.0f} mm)"
            )

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
        inserted = self._compute_inserted_raw()
        d_finger, sum_sq, n_j = self._compute_joint_deviation()

        for env_id in self.eval_env_ids:
            if self.is_complete():
                break
            run = self._running[env_id]
            run.steps += 1
            if isinstance(rewards, torch.Tensor) and env_id < rewards.shape[0]:
                run.episode_return += float(rewards[env_id].reshape(-1)[0].item())

            if (not run.motion_locked) and read_motion_locked(infos, raw, env_id):
                run.motion_locked = True
                run.motion_lock_step = run.steps - 1

            if inserted is not None:
                run.inserted_raw.append([bool(inserted[env_id, i].item()) for i in range(5)])
            else:
                run.inserted_raw.append([False] * 5)

            if d_finger is not None:
                run.d_finger.append([float(d_finger[env_id, i].item()) for i in range(5)])
                run.sum_sq_all_joints += float(sum_sq[env_id].item())
                run.n_joint_samples += int(n_j)
            else:
                run.d_finger.append([0.0] * 5)

            done = bool(terminated[env_id].item()) or bool(truncated[env_id].item())
            if done:
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
                self._finalize_env(env_id, terminated=False, truncated=True)

    def _finalize_env(self, env_id: int, *, terminated: bool, truncated: bool) -> None:
        if self.is_complete():
            return
        run = self._running[env_id]
        if run.steps <= 0:
            self._running[env_id] = _RunningEpisode(env_id=env_id)
            return
        ep = self._build_episode(run, terminated=terminated, truncated=truncated)
        self.episodes.append(ep)
        self._append_csv(ep)
        self._write_summary(partial=True)
        print(
            f"[{self.log_prefix}] episode {ep.episode}: success={int(ep.success)} "
            f"inserted={ep.num_inserted_fingers}/5 lock_step={ep.motion_lock_step} "
            f"steps={ep.episode_length_steps} env={env_id}"
        )
        self._running[env_id] = _RunningEpisode(env_id=env_id)

    def _build_episode(self, run: _RunningEpisode, *, terminated: bool, truncated: bool) -> EpisodeMetrics:
        window = min(self.insertion_window_steps, run.steps)
        tail = run.inserted_raw[-window:] if window else []
        insert_steps = {name: 0 for name in FINGER_ORDER}
        insert_ratio = {name: 0.0 for name in FINGER_ORDER}
        inserted = {name: False for name in FINGER_ORDER}
        if tail:
            for i, name in enumerate(FINGER_ORDER):
                count = sum(1 for row in tail if row[i])
                insert_steps[name] = count
                insert_ratio[name] = count / float(window)
                inserted[name] = insert_ratio[name] >= self.insertion_ratio_threshold

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
        return EpisodeMetrics(
            episode=len(self.episodes),
            env_id=run.env_id,
            success=bool(run.motion_locked),
            terminated=terminated,
            truncated=truncated,
            episode_length_steps=run.steps,
            episode_length_seconds=run.steps * self.control_dt,
            motion_lock_step=run.motion_lock_step,
            motion_lock_time_s=lock_t,
            inserted=inserted,
            insert_ratio=insert_ratio,
            insert_steps=insert_steps,
            num_inserted_fingers=sum(1 for name in FINGER_ORDER if inserted[name]),
            insertion_window_steps=window,
            finger_rms=finger_rms,
            finger_peak=finger_peak,
            hand_rms=hand_rms,
            worst_finger_peak=finger_peak[worst_finger],
            worst_finger=worst_finger,
            episode_return=run.episode_return,
        )

    def _compute_inserted_raw(self) -> torch.Tensor | None:
        """Per-env, per-finger raw insertion. Shape ``(num_envs, 5)`` bool.

        Geometric test (see module docstring):
          1. opening center c = goal_cent_pos (live, follows deformable rim)
          2. opening plane X = c_x (env-local +X is the task insertion axis)
          3. ellipse radii from live E/W (Y) and N/S (Z)
          4. segment A (middle) -> B (distal)
          5. line-plane hit P; require distal on/past the plane and ellipse(P) <= thr
        Fully-through fingers still count: after the phalanx is past the plane, t is
        typically negative (the plane lies behind the middle joint).
        """
        raw = self.raw_env
        if getattr(raw, "_is_free_space_mode", lambda: False)():
            return None
        distal = _stack_distal_env_local(raw)
        cent = getattr(raw, "goal_cent_pos", None)
        east = getattr(raw, "goal_east_pos", None)
        west = getattr(raw, "goal_west_pos", None)
        north = getattr(raw, "goal_north_pos", None)
        south = getattr(raw, "goal_south_pos", None)
        if distal is None or cent is None or east is None or west is None or north is None or south is None:
            return None

        n_envs = int(distal.shape[0])
        middle = self._middle_positions(n_envs, distal)
        if middle is None:
            return None

        # A = middle, B = distal. All env-local.
        a = middle
        b = distal
        c = cent
        eps = torch.as_tensor(1e-4, device=cent.device, dtype=cent.dtype)
        radius_y = 0.5 * torch.abs(east[:, 1] - west[:, 1]).clamp_min(eps)
        radius_z = 0.5 * torch.abs(north[:, 2] - south[:, 2]).clamp_min(eps)

        # Line AB vs plane X = c_x:  A_x + t (B_x - A_x) = c_x
        dx = b[..., 0] - a[..., 0]
        parallel = dx.abs() < _PLANE_PARALLEL_EPS
        safe_dx = torch.where(parallel, torch.ones_like(dx), dx)
        t = (c[:, 0].unsqueeze(1) - a[..., 0]) / safe_dx

        p = a + t.unsqueeze(-1) * (b - a)
        dy = (p[..., 1] - c[:, 1].unsqueeze(1)) / radius_y.unsqueeze(1)
        dz = (p[..., 2] - c[:, 2].unsqueeze(1)) / radius_z.unsqueeze(1)
        ellipse = dy.pow(2) + dz.pow(2)
        # Distal has reached / passed the opening plane (env +X). Do not require t>=0:
        # after the whole phalanx is through, t is typically negative (plane is behind A).
        signed_b = b[..., 0] - c[:, 0].unsqueeze(1)
        hit = (ellipse <= self.insertion_ellipse_threshold) & (signed_b >= 0.0)

        # Parallel fallback: distal already on/past the plane and inside the Y-Z ellipse.
        dy_b = (b[..., 1] - c[:, 1].unsqueeze(1)) / radius_y.unsqueeze(1)
        dz_b = (b[..., 2] - c[:, 2].unsqueeze(1)) / radius_z.unsqueeze(1)
        ellipse_b = dy_b.pow(2) + dz_b.pow(2)
        fallback = (signed_b >= 0.0) & (ellipse_b <= self.insertion_ellipse_threshold)
        return torch.where(parallel, fallback, hit)

    def _middle_positions(self, n_envs: int, distal: torch.Tensor) -> torch.Tensor | None:
        """Env-local middle (or fallback) points, shape ``(num_envs, 5, 3)``."""
        raw = self.raw_env
        hand = self.hand
        origins = getattr(getattr(raw, "scene", None), "env_origins", None)
        cols: list[torch.Tensor] = []
        wrist = getattr(raw, "goal_wrist_pos", None)

        body_pos_w = None
        if hand is not None:
            body_pos_w = getattr(hand.data, "body_pos_w", None)

        for i, finger in enumerate(FINGER_ORDER):
            bid = self.middle_body_ids.get(finger)
            if bid is not None and body_pos_w is not None and origins is not None:
                cols.append(body_pos_w[:, bid, :] - origins)
                continue
            # Fallback: short centerline from distal toward the wrist goal.
            b = distal[:, i, :]
            if wrist is not None:
                axis = b - wrist
                norm = torch.norm(axis, dim=-1, keepdim=True).clamp_min(1e-6)
                cols.append(b - _FALLBACK_SEGMENT_M * (axis / norm))
            else:
                fallback = b.clone()
                fallback[:, 0] = fallback[:, 0] - _FALLBACK_SEGMENT_M
                cols.append(fallback)
        return torch.stack(cols, dim=1)

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
        n_ins = [float(ep.num_inserted_fingers) for ep in eps]
        hand_rms = [ep.hand_rms for ep in eps]

        fingers_block: dict[str, Any] = {}
        for name in FINGER_ORDER:
            rms = [ep.finger_rms[name] for ep in eps]
            peak = [ep.finger_peak[name] for ep in eps]
            fingers_block[name] = {
                "mean_rms": (sum(rms) / n) if n else 0.0,
                "std_rms": sample_std(rms),
                "mean_peak": (sum(peak) / n) if n else 0.0,
                "max_peak": max(peak) if peak else 0.0,
            }

        worst_ep = max(eps, key=lambda e: e.worst_finger_peak) if eps else None
        return {
            "schema_version": 1,
            "task": self.task,
            "checkpoint": self.checkpoint,
            "executed_at": self.executed_at,
            "control_frequency_hz": 1.0 / self.control_dt if self.control_dt else None,
            "control_dt": self.control_dt,
            "config": {
                "max_episodes": self.max_episodes,
                "num_envs": int(getattr(self.raw_env, "num_envs", 1)),
                "eval_env_ids": self.eval_env_ids,
                "insertion_window_sec": self.insertion_window_sec,
                "insertion_window_steps": self.insertion_window_steps,
                "insertion_ratio_threshold": self.insertion_ratio_threshold,
                "insertion_ellipse_threshold": self.insertion_ellipse_threshold,
                "success_definition": "motion_lock_triggered",
                "insertion_definition": "middle_to_distal_line_vs_env_x_opening_ellipse",
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
                "mean_inserted_fingers": (sum(n_ins) / n) if n else 0.0,
                "std_inserted_fingers": sample_std(n_ins),
                "fingers": {
                    name: {
                        "count": inserted_counts[name],
                        "rate": (inserted_counts[name] / n) if n else 0.0,
                    }
                    for name in FINGER_ORDER
                },
            },
            "deformation": {
                "unit": "deg",
                "fingers": fingers_block,
                "hand": {
                    "mean_rms": (sum(hand_rms) / n) if n else 0.0,
                    "std_rms": sample_std(hand_rms),
                    "max_worst_finger_peak": worst_ep.worst_finger_peak if worst_ep else 0.0,
                    "max_worst_finger": worst_ep.worst_finger if worst_ep else "",
                    "max_worst_finger_episode": worst_ep.episode if worst_ep else None,
                },
            },
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
    n = int(success.get("num_episodes") or 0)
    n_ok = int(success.get("num_success") or 0)
    n_fail = int(success.get("num_failed") or 0)
    rate = float(success.get("success_rate") or 0.0) * 100.0
    freq = summary.get("control_frequency_hz")
    window_s = cfg.get("insertion_window_sec")
    window_n = cfg.get("insertion_window_steps")
    ratio_thr = cfg.get("insertion_ratio_threshold")

    print("")
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    if output_dir is not None:
        print(f"Output : {output_dir}")
    if freq and window_s is not None:
        print(
            f"Control: {float(freq):.1f} Hz   window: {float(window_s):.2f} s "
            f"({int(window_n)} steps)   ratio >= {float(ratio_thr):.2f}"
        )
    print("")
    print("Episodes")
    print(f"  Total   : {n}")
    print(f"  Success : {n_ok}")
    print(f"  Failed  : {n_fail}")
    print(f"  Rate    : {rate:.1f} %")
    print("")
    print("-" * 60)
    print("Finger Insertion  (plane ∩ opening ellipse, last window)")
    print("-" * 60)
    print(f"{'Finger':<10}  {'Episodes Inserted':>18}     Rate")
    for name in FINGER_ORDER:
        block = fingers_ins.get(name) or {}
        count = int(block.get("count") or 0)
        fr = float(block.get("rate") or 0.0) * 100.0
        print(f"{name.capitalize():<10}  {count:8d} / {n:<6d}   {fr:6.1f} %")
    print("")
    print("Mean inserted fingers :")
    print(
        f"  {float(insertion.get('mean_inserted_fingers') or 0.0):.2f} "
        f"± {float(insertion.get('std_inserted_fingers') or 0.0):.2f} / 5"
    )
    print("")
    print("-" * 60)
    print("Finger Joint Deviation  (from hand.data.default_joint_pos, deg)")
    print("-" * 60)
    print(f"{'Finger':<10}  {'Mean RMS':>10}  {'Std RMS':>10}  {'Mean Peak':>10}  {'Max Peak':>10}")
    for name in FINGER_ORDER:
        block = fingers_def.get(name) or {}
        print(
            f"{name.capitalize():<10}  "
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
