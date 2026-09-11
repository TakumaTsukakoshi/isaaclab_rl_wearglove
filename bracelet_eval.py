"""Bracelet-task evaluation: motion-lock success, finger insertion, joint deviation.

Insertion is a per-finger crossing state machine. Episode passage is that
latch at the last recorded step, not an ever-OR and not a live
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

Evaluation has three layers that must not share the word "success":

  1. **Task success** (official): ``legacy_all_five`` knuckle latch AND
     ``wrist_within_goal`` at the same motion-lock step. Same rule as
     training. Printed as ``Task success``.
  2. **Finger passage** (geometric diagnostic): canonical ``finger_passed``
     vector at the **last recorded step** (not ever-OR). Thumb = ordered
     PRE→POST ``thdistal → thmiddle → thproximal``. A reverse POST→PRE
     clears that landmark and every later one; earlier stations stay True
     if they did not reverse. ``thbase`` is diagnostic only. Other fingers
     knuckle-only. ``all_five_passage`` is not task success.
  3. **Physical / snag diagnostics**: joint deviation, wrist shortfall,
     thumb timing. Outcome C is "possible incomplete advancement", not
     an automatic snag label.

Passage outcome breakdown (not a second success table):

  A no passage / B partial / C all-five passage, wrist incomplete /
  D all-five passage + wrist complete.

``task_success``, ``all_five_passage``, and D are allowed to differ.
Training reward / soft-gate / motion lock / checkpoints are not affected.
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

# Second landmark for live-containment / distal-crossing diagnostics.
DISTAL_BODY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "thumb": ("robot0_thdistal", "robot0_thmiddle", "robot0_thproximal"),
    "index": ("robot0_ffdistal", "robot0_ffmiddle", "robot0_ffproximal"),
    "middle": ("robot0_mfdistal", "robot0_mfmiddle", "robot0_mfproximal"),
    "ring": ("robot0_rfdistal", "robot0_rfmiddle", "robot0_rfproximal"),
    "little": ("robot0_lfdistal", "robot0_lfmiddle", "robot0_lfproximal"),
}

# Eval-only thumb landmarks, tip → base. Passage requires the first 3 to
# complete PRE→POST through the opening in order. A confirmed reverse
# POST→PRE clears that landmark and later ones; earlier stay True if they
# did not reverse. thbase is diagnostic only.
THUMB_SWEEP_BODY_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("robot0_thdistal",),
    ("robot0_thmiddle",),
    ("robot0_thproximal",),
    ("robot0_thbase",),
)
THUMB_SWEEP_REQUIRED_STATIONS = 3
THUMB_SWEEP_MIN_STATIONS = 3
THUMB_SWEEP_HOLE_HALF_WIDTH_M = 0.008
EVAL_DUAL_LANDMARK_FINGERS: tuple[str, ...] = ("thumb",)


def _make_insertion_debug_fields() -> list[str]:
    fields = [
        "episode",
        "env_id",
        "step",
        "t_s",
        "c_x",
        "c_y",
        "c_z",
        "n_x",
        "n_y",
        "n_z",
        "r_y",
        "r_z",
        "wrist_d",
        "wrist_dx",
        "wrist_dy",
        "wrist_dz",
        "latched_n",
        "live_n",
    ]
    for name in FINGER_ORDER:
        fields.extend(
            [
                f"{name}_knuckle_x",
                f"{name}_knuckle_y",
                f"{name}_knuckle_z",
                f"{name}_d_knuckle",
                f"{name}_e_knuckle",
                f"{name}_side_knuckle",
                f"{name}_distal_x",
                f"{name}_distal_y",
                f"{name}_distal_z",
                f"{name}_d_distal",
                f"{name}_e_distal",
                f"{name}_side_distal",
                f"{name}_inserted",
                f"{name}_live_ok",
                f"{name}_fwd_count",
                f"{name}_rev_count",
            ]
        )
    return fields


INSERTION_DEBUG_FIELDS = _make_insertion_debug_fields()

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
    "final_inserted_fingers_latched",
    "max_inserted_fingers",
    "max_passed_fingers",
    "num_ever_passed",
    "final_geometric_overlap",
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
    "all_five_ever",
    "all_five_retained",
    "all_five_retained_latched",
    "first_all_five_frame_latched",
    "wrist_success",
    "task_success",
    "legacy_success",
    "legacy_all_five",
    "all_five_passage",
    "all5_passage_wrist_incomplete",
    "all5_passage_wrist_complete",
    "strict_success",
    "thumb_passed",
    "passed_count",
    "thdistal_frame",
    "thmiddle_frame",
    "thproximal_frame",
    "thbase_frame",
    "thumb_passage_duration_frames",
    "first_all_five_frame",
    "first_wrist_success_frame",
    "wrist_success_threshold_m",
    "wrist_distance_final_m",
    "wrist_distance_best_m",
    "wrist_distance_best_after_all_five_m",
    "wrist_ok_ever",
    "wrist_ok_after_all_five",
    "wrist_ok_at_end",
    "wrist_dx_final_m",
    "wrist_dy_final_m",
    "wrist_dz_final_m",
    "wrist_dx_best_after_all_five_m",
    "wrist_dy_best_after_all_five_m",
    "wrist_dz_best_after_all_five_m",
    "failure_mode",
    "wrist_shortfall_primary",
    "wrist_shortfall_tags",
    "final_inserted_fingers_live",
    "live_all_five",
    "snag_suspect",
    "thumb_live_ok",
    "index_live_ok",
    "middle_live_ok",
    "ring_live_ok",
    "little_live_ok",
    "thumb_inserted_latched",
    "index_inserted_latched",
    "middle_inserted_latched",
    "ring_inserted_latched",
    "little_inserted_latched",
]

# Wrist-only condition is instantaneous: ||goal_wrist - goal_cent|| < threshold.
# Task success latches the first step where that is true AND all five fingers are inserted.
WRIST_NEAR_MULT = 2.0
WRIST_REGRESSION_M = 0.005
FAILURE_MODE_NO_INSERTION = "no_insertion"
FAILURE_MODE_PARTIAL = "partial_finger_insertion"
FAILURE_MODE_FULL_INCOMPLETE_WRIST = "full_finger_insertion_incomplete_wrist_advancement"
FAILURE_MODE_FULL_AND_WRIST = "full_insertion_and_wrist_success"
FAILURE_MODE_SUMMARY_ORDER = (
    FAILURE_MODE_NO_INSERTION,
    FAILURE_MODE_PARTIAL,
    FAILURE_MODE_FULL_INCOMPLETE_WRIST,
    FAILURE_MODE_FULL_AND_WRIST,
)
FAILURE_MODE_LABELS = {
    FAILURE_MODE_NO_INSERTION: "No finger passage",
    FAILURE_MODE_PARTIAL: "Partial finger passage",
    FAILURE_MODE_FULL_INCOMPLETE_WRIST: "All-five passage, wrist incomplete",
    FAILURE_MODE_FULL_AND_WRIST: "All-five passage + wrist complete",
}


def _rad_to_deg(value: float) -> float:
    return float(value) * (180.0 / math.pi)


def sample_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def wrist_best_group_stats(episodes: list[Any]) -> dict[str, Any]:
    """Best wrist distance (m/cm) for task-success vs failure / incomplete placement."""

    def _collect(pred) -> dict[str, Any]:
        vals: list[float] = []
        rows: list[dict[str, Any]] = []
        for ep in episodes:
            if not pred(ep):
                continue
            d = _ep_get(ep, "wrist_distance_best_m")
            if d is None:
                continue
            d = float(d)
            vals.append(d)
            rows.append(
                {
                    "episode": int(ep_i) if (ep_i := _ep_get(ep, "episode", None)) is not None else -1,
                    "env_id": int(eid) if (eid := _ep_get(ep, "env_id", None)) is not None else -1,
                    "wrist_distance_best_m": d,
                    "wrist_distance_best_cm": d * 100.0,
                }
            )
        rows.sort(key=lambda r: (r["env_id"], r["episode"]))
        n = len(vals)
        return {
            "n": n,
            "mean_m": (sum(vals) / n) if n else None,
            "std_m": sample_std(vals) if n else None,
            "mean_cm": (sum(vals) / n * 100.0) if n else None,
            "std_cm": (sample_std(vals) * 100.0) if n else None,
            "episodes": rows,
        }

    return {
        "metric": "wrist_distance_best_m = min_t ||goal_wrist_pos - goal_cent_pos||",
        "successful_dressing": _collect(
            lambda ep: bool(_ep_get(ep, "task_success", _ep_get(ep, "legacy_success", False)))
        ),
        "task_failure": _collect(
            lambda ep: not bool(_ep_get(ep, "task_success", _ep_get(ep, "legacy_success", False)))
        ),
        "incomplete_placement": _collect(
            lambda ep: bool(
                _ep_get(ep, "all5_passage_wrist_incomplete", False)
                or _ep_get(ep, "failure_mode") == FAILURE_MODE_FULL_INCOMPLETE_WRIST
            )
        ),
    }


def inserted_finger_histogram(values: list[int] | list[float], n_fingers: int = 5) -> dict[str, Any]:
    """Episode counts for ``k/5`` inserted fingers. Index ``k`` is the bin."""
    counts = [0] * (n_fingers + 1)
    for raw in values:
        k = int(raw)
        if 0 <= k <= n_fingers:
            counts[k] += 1
    n = len(values)
    return {
        "n_fingers": n_fingers,
        "num_episodes": n,
        "counts": counts,
        "rates": [(c / n) if n else 0.0 for c in counts],
        "labels": [f"{k}/{n_fingers}" for k in range(n_fingers + 1)],
    }


def last_passage_flags(rows: list[list[bool]], n_fingers: int = 5) -> list[bool]:
    """Latch vector at the last recorded step. Empty history is all False."""
    if not rows:
        return [False] * n_fingers
    last = rows[-1]
    return [bool(last[i]) if i < len(last) else False for i in range(n_fingers)]


def combine_eval_passage_flags(
    base_flags: list[bool],
    distal_flags: list[bool] | None,
    *,
    thumb_passed: bool | None = None,
    dual_fingers: tuple[str, ...] = EVAL_DUAL_LANDMARK_FINGERS,
) -> list[bool]:
    """Eval passage. Thumb uses ``thumb_passed`` (ordered tip→proximal crossings) when given.

    Fallback without a sweep result: listed dual-landmark fingers need base
    AND distal latch. Other fingers stay base-only.
    """
    out = [bool(v) for v in base_flags]
    if thumb_passed is not None:
        out[0] = bool(thumb_passed)
        return out
    if distal_flags is None:
        return out
    for i, name in enumerate(FINGER_ORDER):
        if i < len(out) and i < len(distal_flags) and name in dual_fingers:
            out[i] = bool(base_flags[i] and distal_flags[i])
    return out


@dataclass
class EpisodePassageResult:
    """Canonical episode-level finger passage. All passage tables derive from this.

    ``passed`` is the last-step latch, not an ever-OR. ``task_success`` is
    official motion-lock success and is allowed to differ from
    ``all_five_passage`` and ``all5_passage_wrist_complete``.
    """

    passed: dict[str, bool]
    passed_count: int
    all_five_passage: bool
    wrist_ok_ever: bool
    all5_passage_wrist_complete: bool
    all5_passage_wrist_incomplete: bool
    legacy_all_five: bool
    task_success: bool
    failure_mode: str

    @property
    def thumb_passed(self) -> bool:
        return bool(self.passed.get("thumb"))

    @property
    def all_five_passed(self) -> bool:
        return self.all_five_passage

    @property
    def wrist_success(self) -> bool:
        return self.wrist_ok_ever

    @property
    def strict_success(self) -> bool:
        """Deprecated alias of ``all5_passage_wrist_complete`` (not task success)."""
        return self.all5_passage_wrist_complete

    @property
    def legacy_success(self) -> bool:
        return self.task_success


def make_episode_passage_result(
    *,
    passed: dict[str, bool] | list[bool],
    wrist_ok_ever: bool,
    legacy_all_five: bool,
    legacy_success: bool,
) -> EpisodePassageResult:
    """Build the single passage record used by histogram, all-five passage, and A–D."""
    if isinstance(passed, dict):
        flags = [bool(passed.get(name, False)) for name in FINGER_ORDER]
    else:
        flags = [bool(v) for v in passed]
        if len(flags) < 5:
            flags.extend([False] * (5 - len(flags)))
    passed_d = {name: flags[i] for i, name in enumerate(FINGER_ORDER)}
    count = sum(1 for v in flags if v)
    all_five = count == 5
    wrist = bool(wrist_ok_ever)
    return EpisodePassageResult(
        passed=passed_d,
        passed_count=count,
        all_five_passage=all_five,
        wrist_ok_ever=wrist,
        all5_passage_wrist_complete=all_five and wrist,
        all5_passage_wrist_incomplete=all_five and not wrist,
        legacy_all_five=bool(legacy_all_five),
        task_success=bool(legacy_success),
        failure_mode=classify_episode_failure_mode(
            max_passed_fingers=count,
            ever_all_passed=all_five,
            wrist_success=wrist,
        ),
    )


def thumb_sweep_timing(passage_events: dict[str, Any] | None) -> dict[str, Any]:
    """Thumb station frames from ``thumb_sweep.visits``. ``thbase`` is diagnostic."""
    sweep = (passage_events or {}).get("thumb_sweep") or {}
    frames: dict[str, int | None] = {
        "thdistal": None,
        "thmiddle": None,
        "thproximal": None,
        "thbase": None,
    }
    for rec in sweep.get("visits") or []:
        body = str(rec.get("body") or "")
        key = body.rsplit("_", 1)[-1] if body else ""
        if key not in frames:
            continue
        step = rec.get("visit_step")
        frames[key] = None if step is None else int(step)
    distal = frames["thdistal"]
    proximal = frames["thproximal"]
    duration = None if distal is None or proximal is None else int(proximal) - int(distal)
    return {
        "thdistal_frame": frames["thdistal"],
        "thmiddle_frame": frames["thmiddle"],
        "thproximal_frame": frames["thproximal"],
        "thbase_frame": frames["thbase"],
        "thumb_passage_duration_frames": duration,
    }


def _episode_env_ref(ep: Any) -> dict[str, int]:
    episode = _ep_get(ep, "episode", None)
    env_id = _ep_get(ep, "env_id", None)
    return {
        "episode": int(episode) if episode is not None else -1,
        "env_id": int(env_id) if env_id is not None else -1,
    }


def compute_task_success_vs_passage(episodes: list[Any]) -> dict[str, Any]:
    """2x2 of official task success vs strict all-five passage."""
    no_no = no_yes = yes_no = yes_yes = 0
    ts_thumb = ts_all5 = 0
    fail_thumb = fail_all5 = 0
    n_success = 0
    n_fail = 0
    yes_no_envs: list[dict[str, int]] = []
    yes_yes_envs: list[dict[str, int]] = []
    no_yes_envs: list[dict[str, int]] = []
    task_success_envs: list[dict[str, int]] = []
    yes_no_details: list[dict[str, Any]] = []
    for ep in episodes:
        task = bool(_ep_get(ep, "task_success", _ep_get(ep, "legacy_success", False)))
        all5 = bool(
            _ep_get(ep, "all_five_passage", None)
            if _ep_get(ep, "all_five_passage", None) is not None
            else _ep_get(ep, "all_five_passed", _ep_get(ep, "ever_all_inserted", False))
        )
        thumb = _ep_get(ep, "thumb_passed", None)
        if thumb is None:
            passed = _ep_get(ep, "inserted", None) or _ep_get(ep, "ever_passed", None) or {}
            if isinstance(passed, dict):
                thumb = bool(passed.get("thumb"))
            else:
                thumb = False
        thumb = bool(thumb)
        ref = _episode_env_ref(ep)
        if task and all5:
            yes_yes += 1
            yes_yes_envs.append(ref)
        elif task and not all5:
            yes_no += 1
            yes_no_envs.append(ref)
            yes_no_details.append(describe_task_yes_not_all_five(ep))
        elif (not task) and all5:
            no_yes += 1
            no_yes_envs.append(ref)
        else:
            no_no += 1
        if task:
            n_success += 1
            ts_thumb += int(thumb)
            ts_all5 += int(all5)
            task_success_envs.append(ref)
        else:
            n_fail += 1
            fail_thumb += int(thumb)
            fail_all5 += int(all5)
    return {
        "task_no_all5_no": no_no,
        "task_no_all5_yes": no_yes,
        "task_yes_all5_no": yes_no,
        "task_yes_all5_yes": yes_yes,
        "task_yes_all5_no_envs": yes_no_envs,
        "task_yes_all5_yes_envs": yes_yes_envs,
        "task_no_all5_yes_envs": no_yes_envs,
        "task_yes_all5_no_details": yes_no_details,
        "task_success_envs": task_success_envs,
        "among_task_success": {
            "n": n_success,
            "thumb_passed": ts_thumb,
            "all_five_passage": ts_all5,
            "envs": task_success_envs,
        },
        "among_task_failure": {
            "n": n_fail,
            "thumb_passed": fail_thumb,
            "all_five_passage": fail_all5,
        },
    }


def classify_episode_failure_mode(
    *,
    max_passed_fingers: int | None = None,
    ever_all_passed: bool = False,
    wrist_success: bool = False,
    task_success: bool | None = None,
    final_inserted_fingers: int | None = None,
) -> str:
    """Mutually exclusive A–D from the canonical passage vector and wrist.

    ``task_success`` is accepted but ignored: legacy motion-lock must not
    override a 4/5 strict passage into the all-five bucket.
    ``final_inserted_fingers`` is an alias for ``max_passed_fingers``.
    """
    del task_success
    if max_passed_fingers is None:
        max_passed_fingers = 0 if final_inserted_fingers is None else int(final_inserted_fingers)
    passed = int(max_passed_fingers)
    all_five = bool(ever_all_passed) or passed >= 5
    if all_five and wrist_success:
        return FAILURE_MODE_FULL_AND_WRIST
    if all_five:
        return FAILURE_MODE_FULL_INCOMPLETE_WRIST
    if passed <= 0:
        return FAILURE_MODE_NO_INSERTION
    return FAILURE_MODE_PARTIAL


def _ep_get(ep: Any, name: str, default: Any = None) -> Any:
    if isinstance(ep, dict):
        return ep.get(name, default)
    return getattr(ep, name, default)


def _finger_flag_map(value: Any) -> dict[str, bool]:
    if isinstance(value, dict):
        return {name: bool(value.get(name, False)) for name in FINGER_ORDER}
    if isinstance(value, (list, tuple)):
        return {
            name: bool(value[i]) if i < len(value) else False
            for i, name in enumerate(FINGER_ORDER)
        }
    return {name: False for name in FINGER_ORDER}


def describe_task_yes_not_all_five(ep: Any) -> dict[str, Any]:
    """Why a task-success episode is not last-step all-five passage."""
    last = _finger_flag_map(
        _ep_get(ep, "inserted") or _ep_get(ep, "passed") or _ep_get(ep, "finger_passed")
    )
    ever = _finger_flag_map(_ep_get(ep, "ever_passed") or last)
    exited = [name for name in FINGER_ORDER if ever[name] and not last[name]]
    missing = [name for name in FINGER_ORDER if not last[name]]
    events = _ep_get(ep, "passage_events") or {}
    sweep = events.get("thumb_sweep") if isinstance(events, dict) else {}
    sweep = sweep or {}
    visits = list(sweep.get("visits") or [])
    inserted_now = list(sweep.get("inserted") or [])
    stations: list[dict[str, Any]] = []
    labels = ("thdistal", "thmiddle", "thproximal")
    for i, lab in enumerate(labels):
        rec = visits[i] if i < len(visits) else {}
        vis = rec.get("visit_step")
        now = bool(inserted_now[i]) if i < len(inserted_now) else bool(rec.get("inserted_now"))
        if vis is None and not now:
            status = "never"
        elif vis is not None and not now:
            status = "exited"
        else:
            status = "latched" if now else "never"
        stations.append(
            {
                "name": rec.get("body") or lab,
                "visit_step": vis,
                "inserted_now": now,
                "status": status,
            }
        )
    n_last = sum(1 for name in FINGER_ORDER if last[name])
    n_ever = sum(1 for name in FINGER_ORDER if ever[name])
    note = "entered then left" if exited else "never reached last-step 5/5"
    return {
        "episode": int(ep_i) if (ep_i := _ep_get(ep, "episode", None)) is not None else -1,
        "env_id": int(eid) if (eid := _ep_get(ep, "env_id", None)) is not None else -1,
        "last_passed": n_last,
        "ever_passed": n_ever,
        "last_fingers": last,
        "ever_fingers": ever,
        "exited_fingers": exited,
        "missing_last": missing,
        "note": note,
        "thumb_stations": stations,
        "thumb_next_idx": sweep.get("next_idx"),
        "thdistal_frame": _ep_get(ep, "thdistal_frame"),
        "thmiddle_frame": _ep_get(ep, "thmiddle_frame"),
        "thproximal_frame": _ep_get(ep, "thproximal_frame"),
        "wrist_distance_best_m": _ep_get(ep, "wrist_distance_best_m"),
        "wrist_distance_final_m": _ep_get(ep, "wrist_distance_final_m"),
        "wrist_distance_at_success": _ep_get(ep, "wrist_distance_at_success"),
        "legacy_all_five": bool(_ep_get(ep, "legacy_all_five", False)),
        "max_passed_fingers": int(_ep_get(ep, "max_passed_fingers", n_ever) or n_ever),
        "motion_lock_step": _ep_get(ep, "motion_lock_step"),
        "motion_lock_time_s": _ep_get(ep, "motion_lock_time_s"),
    }


def assert_eval_outcome_consistency(episodes: list[Any]) -> dict[str, Any]:
    """Passage tables must agree with the histogram and A–D.

    D is ``all5_passage_wrist_complete``, not official ``task_success``.
    Those two metrics are allowed to differ.
    """
    n = len(episodes)
    counts = {key: 0 for key in FAILURE_MODE_SUMMARY_ORDER}
    hist = [0] * 6
    all_five_passage = 0
    all5_complete = 0
    task_success = 0
    legacy_all_five = 0
    for ep in episodes:
        mode = _ep_get(ep, "failure_mode")
        counts[mode] += 1
        passed = int(_ep_get(ep, "max_passed_fingers", _ep_get(ep, "passed_count", 0)) or 0)
        if 0 <= passed <= 5:
            hist[passed] += 1
        ever = bool(
            _ep_get(ep, "all_five_passage", None)
            if _ep_get(ep, "all_five_passage", None) is not None
            else (
                _ep_get(ep, "all_five_passed", None)
                if _ep_get(ep, "all_five_passed", None) is not None
                else (_ep_get(ep, "ever_all_inserted", False) or _ep_get(ep, "all_five_ever", False))
            )
        )
        if ever:
            all_five_passage += 1
        complete = _ep_get(ep, "all5_passage_wrist_complete", None)
        if complete is None:
            complete = _ep_get(ep, "strict_success", False)
        if bool(complete):
            all5_complete += 1
        task = _ep_get(ep, "task_success", None)
        if task is None:
            task = _ep_get(ep, "legacy_success", False)
        if bool(task):
            task_success += 1
        knuckle = _ep_get(ep, "legacy_all_five", None)
        if knuckle is None:
            knuckle = _ep_get(ep, "ever_all_inserted_knuckle", False)
        if bool(knuckle):
            legacy_all_five += 1
    a = counts[FAILURE_MODE_NO_INSERTION]
    b = counts[FAILURE_MODE_PARTIAL]
    c = counts[FAILURE_MODE_FULL_INCOMPLETE_WRIST]
    d = counts[FAILURE_MODE_FULL_AND_WRIST]
    errors: list[str] = []
    if a + b + c + d != n:
        errors.append(f"categories {a}+{b}+{c}+{d} != total {n}")
    if sum(hist) != n:
        errors.append(f"histogram sum {sum(hist)} != total {n}")
    if a != hist[0]:
        errors.append(f"A {a} != histogram[0] {hist[0]}")
    if b != sum(hist[1:5]):
        errors.append(f"B {b} != histogram[1:5] {sum(hist[1:5])}")
    if all_five_passage != hist[5]:
        errors.append(f"all_five_passage {all_five_passage} != histogram[5] {hist[5]}")
    if c + d != all_five_passage:
        errors.append(f"C+D {c}+{d} != all_five_passage {all_five_passage}")
    if d != all5_complete:
        errors.append(f"D {d} != all5_passage_wrist_complete {all5_complete}")
    if task_success > n:
        errors.append(f"task_success {task_success} > total {n}")
    ever_and_mismatch = 0
    for i, ep in enumerate(episodes):
        task = _ep_get(ep, "task_success", None)
        knuckle = _ep_get(ep, "legacy_all_five", None)
        wrist = _ep_get(ep, "wrist_ok_ever", None)
        if wrist is None:
            wrist = _ep_get(ep, "wrist_success", None)
        if task is None or knuckle is None or wrist is None:
            continue
        if bool(task) and not (bool(knuckle) and bool(wrist)):
            errors.append(
                f"episode {i}: task_success without legacy_all_five and wrist_ok"
            )
        if (bool(knuckle) and bool(wrist)) != bool(task):
            ever_and_mismatch += 1
    payload = {
        "ok": not errors,
        "total": n,
        "A_no_passage": a,
        "B_partial": b,
        "C_all_five_wrist_incomplete": c,
        "D_all5_passage_wrist_complete": d,
        "D_success": d,
        "ever_all_five": all_five_passage,
        "all_five_passage": all_five_passage,
        "all5_passage_wrist_incomplete": c,
        "all5_passage_wrist_complete": all5_complete,
        "strict_success": all5_complete,
        "legacy_success": task_success,
        "legacy_all_five": legacy_all_five,
        "histogram": hist,
        "task_success": task_success,
        "task_success_vs_legacy_and_wrist_mismatches": ever_and_mismatch,
        "errors": errors,
    }
    if errors:
        raise AssertionError("eval outcome consistency failed: " + "; ".join(errors))
    return payload


def classify_wrist_shortfall(
    *,
    threshold_m: float,
    wrist_ok_ever: bool,
    wrist_ok_after_all_five: bool,
    first_wrist_success_frame: int | None,
    first_all_five_frame: int | None,
    best_after_all_five_m: float | None,
    final_distance_m: float | None,
) -> tuple[str | None, list[str]]:
    """Why a 5/5 episode missed the wrist threshold. None if not that case."""
    tags: list[str] = []
    if first_all_five_frame is None:
        return None, ["missing_first_all_five_frame"]
    if best_after_all_five_m is None:
        return None, ["missing_wrist_distance_after_all_five"]

    if wrist_ok_after_all_five:
        primary = "wrist_ok_after_all_five_not_latched"
    elif (
        wrist_ok_ever
        and first_wrist_success_frame is not None
        and first_wrist_success_frame < first_all_five_frame
    ):
        primary = "reached_wrist_criterion_before_all_five"
        tags.append("transient_wrist_not_retained_with_all_five")
    elif best_after_all_five_m < float(threshold_m) * WRIST_NEAR_MULT:
        primary = "reached_near_wrist_threshold_but_did_not_cross"
    else:
        primary = "insufficient_advancement_toward_wrist"

    if (
        final_distance_m is not None
        and (final_distance_m - best_after_all_five_m) >= WRIST_REGRESSION_M
        and final_distance_m > float(threshold_m)
    ):
        tags.append("regression_after_full_finger_insertion")
    return primary, tags


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


def plane_side(d: float, delta: float) -> str:
    if d > delta:
        return "PRE"
    if d < -delta:
        return "POST"
    return "BAND"


def live_containment_flags(
    knuckle: torch.Tensor,
    distal: torch.Tensor | None,
    center: torch.Tensor,
    radius_y: torch.Tensor,
    radius_z: torch.Tensor,
    *,
    delta: float,
    ellipse_threshold: float,
) -> torch.Tensor:
    """Current-state insertion: knuckle (and distal if given) POST and inside the live YZ ellipse.

    ``knuckle`` / ``distal`` are ``(N, 5, 3)``, ``center`` / radii ``(N,)`` or ``(N, 3)``.
    Does not use the crossing latch.
    """
    d_k = knuckle[..., 0] - center[:, 0].unsqueeze(1)
    e_k = ellipse_value_yz(knuckle, center, radius_y, radius_z)
    ok = (d_k < -float(delta)) & (e_k <= float(ellipse_threshold))
    if distal is not None:
        d_d = distal[..., 0] - center[:, 0].unsqueeze(1)
        e_d = ellipse_value_yz(distal, center, radius_y, radius_z)
        ok = ok & (d_d < -float(delta)) & (e_d <= float(ellipse_threshold))
    return ok


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


def thumb_station_occupied(
    nodes: torch.Tensor,
    center: torch.Tensor,
    radius_y: torch.Tensor,
    radius_z: torch.Tensor,
    *,
    hole_half_width: float = THUMB_SWEEP_HOLE_HALF_WIDTH_M,
    ellipse_threshold: float = 1.0,
) -> torch.Tensor:
    """Which thumb stations currently sit in the opening interior.

    ``nodes`` is tip→base, ``(K, 3)`` or ``(N, K, 3)``. A station is occupied
    if its COM is in the opening slab (``|d| <= hole`` and ``e <= threshold``)
    or the adjacent thumb segment pierces the opening disk.
    """
    squeeze = nodes.ndim == 2
    if squeeze:
        nodes = nodes.unsqueeze(0)
        center = center.unsqueeze(0)
        radius_y = radius_y.reshape(-1)
        radius_z = radius_z.reshape(-1)
        if radius_y.numel() == 1:
            radius_y = radius_y.expand(nodes.shape[0])
            radius_z = radius_z.expand(nodes.shape[0])
    d = nodes[..., 0] - center[:, 0].unsqueeze(1)
    e = ellipse_value_yz(nodes, center, radius_y, radius_z)
    occupied = (e <= float(ellipse_threshold)) & (d.abs() <= float(hole_half_width))
    if nodes.shape[1] >= 2:
        p0 = nodes[:, :-1]
        p1 = nodes[:, 1:]
        denom = p1[..., 0] - p0[..., 0]
        valid = denom.abs() > 1e-8
        t = torch.where(
            valid,
            (center[:, 0].unsqueeze(1) - p0[..., 0]) / denom,
            torch.full_like(denom, -1.0),
        )
        hit = valid & (t >= 0.0) & (t <= 1.0)
        ph = p0 + t.unsqueeze(-1) * (p1 - p0)
        through = hit & (ellipse_value_yz(ph, center, radius_y, radius_z) <= float(ellipse_threshold))
        occupied = occupied.clone()
        occupied[:, :-1] = occupied[:, :-1] | (through & (t < 0.5))
        occupied[:, 1:] = occupied[:, 1:] | (through & (t >= 0.5))
    return occupied[0] if squeeze else occupied


def thumb_required_stations_inside(
    nodes: torch.Tensor,
    center: torch.Tensor,
    radius_y: torch.Tensor,
    radius_z: torch.Tensor,
    *,
    n_required: int = THUMB_SWEEP_REQUIRED_STATIONS,
    hole_half_width: float = THUMB_SWEEP_HOLE_HALF_WIDTH_M,
    ellipse_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-station opening occupancy and strict thumb-inside flag.

    A station is inside only while its COM (or the adjacent segment) sits in
    the opening slab: ``|d| <= hole`` and ``e <= threshold``. Thumb LIVE is
    true only when every required tip→proximal station is inside at once
    (cannot skip). This is a current-state test, not the crossing latch.
    """
    occ = thumb_station_occupied(
        nodes,
        center,
        radius_y,
        radius_z,
        hole_half_width=hole_half_width,
        ellipse_threshold=ellipse_threshold,
    )
    n_req = max(1, min(int(n_required), int(occ.shape[-1])))
    inside = occ[..., :n_req].all(dim=-1)
    return occ, inside


def apply_opening_crossing_latch(
    points: torch.Tensor,
    center: torch.Tensor,
    radius_y: torch.Tensor,
    radius_z: torch.Tensor,
    active: torch.Tensor,
    *,
    delta: float,
    confirm_frames: int,
    ellipse_threshold: float,
    inserted: torch.Tensor,
    last_clear_side: torch.Tensor,
    last_clear_pos: torch.Tensor,
    last_clear_center: torch.Tensor,
    last_clear_radius_y: torch.Tensor,
    last_clear_radius_z: torch.Tensor,
    has_clear: torch.Tensor,
    fwd_pending: torch.Tensor,
    rev_pending: torch.Tensor,
    fwd_count: torch.Tensor,
    rev_count: torch.Tensor,
    first_insert_step: torch.Tensor,
    step_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Independent PRE→POST / POST→PRE opening-crossing latch for ``(N, K)`` points.

    Same rule as training knuckle insertion: the interpolated crossing must go
    through the live YZ ellipse. Reverse through the ellipse clears that point.
    Order among the K points is not required.
    """
    active = active.to(device=inserted.device, dtype=torch.bool)
    if not bool(active.any()):
        return {
            "inserted": inserted,
            "last_clear_side": last_clear_side,
            "last_clear_pos": last_clear_pos,
            "last_clear_center": last_clear_center,
            "last_clear_radius_y": last_clear_radius_y,
            "last_clear_radius_z": last_clear_radius_z,
            "has_clear": has_clear,
            "fwd_pending": fwd_pending,
            "rev_pending": rev_pending,
            "fwd_count": fwd_count,
            "rev_count": rev_count,
            "first_insert_step": first_insert_step,
            "step_count": step_count,
        }

    d_curr = points[..., 0] - center[:, 0].unsqueeze(1)
    is_pre = d_curr > float(delta)
    is_post = d_curr < -float(delta)
    is_clear = is_pre | is_post
    side = torch.where(
        is_pre,
        torch.ones_like(last_clear_side),
        torch.where(is_post, -torch.ones_like(last_clear_side), torch.zeros_like(last_clear_side)),
    )

    d_clear = last_clear_pos[..., 0] - last_clear_center[..., 0]
    denom = d_clear - d_curr
    t = torch.where(denom.abs() > 1e-8, d_clear / denom, torch.full_like(d_curr, 0.5))
    t = t.clamp(0.0, 1.0)
    t3 = t.unsqueeze(-1)
    cross_p = last_clear_pos + t3 * (points - last_clear_pos)
    cross_c = last_clear_center + t3 * (center.unsqueeze(1) - last_clear_center)
    cross_ry = last_clear_radius_y + t * (radius_y.unsqueeze(1) - last_clear_radius_y)
    cross_rz = last_clear_radius_z + t * (radius_z.unsqueeze(1) - last_clear_radius_z)
    ev = ((cross_p[..., 1] - cross_c[..., 1]) / cross_ry.clamp_min(1e-6)).pow(2) + (
        (cross_p[..., 2] - cross_c[..., 2]) / cross_rz.clamp_min(1e-6)
    ).pow(2)
    through_opening = ev <= float(ellipse_threshold)

    from_pre = has_clear & (last_clear_side > 0)
    from_post = has_clear & (last_clear_side < 0)
    active_f = active.unsqueeze(1)
    confirm = max(1, int(confirm_frames))
    fwd_cand = active_f & is_post & from_pre & (~inserted) & through_opening
    rev_cand = active_f & is_pre & from_post & inserted & through_opening

    ones = torch.ones_like(fwd_count)
    zeros = torch.zeros_like(fwd_count)

    fwd_pending_n = ((fwd_pending & is_post) | fwd_cand) & (~inserted) & active_f
    fwd_count_n = torch.where(~fwd_pending_n, zeros, torch.where(fwd_cand, ones, fwd_count + 1))
    commit_fwd = fwd_pending_n & (fwd_count_n >= confirm)
    inserted_n = inserted | commit_fwd
    fwd_pending_n = fwd_pending_n & (~commit_fwd)
    fwd_count_n = torch.where(fwd_pending_n, fwd_count_n, zeros)

    first = first_insert_step
    step_i = (step_count + 1).unsqueeze(1).expand_as(first)
    first_n = torch.where((first < 0) & commit_fwd, step_i.to(first.dtype), first)

    rev_pending_n = ((rev_pending & is_pre) | rev_cand) & inserted_n & active_f
    rev_count_n = torch.where(~rev_pending_n, zeros, torch.where(rev_cand, ones, rev_count + 1))
    commit_rev = rev_pending_n & (rev_count_n >= confirm)
    inserted_n = inserted_n & (~commit_rev)
    rev_pending_n = rev_pending_n & (~commit_rev) & inserted_n
    rev_count_n = torch.where(rev_pending_n, rev_count_n, zeros)

    step_n = step_count + active.to(dtype=step_count.dtype)

    clear_f = active_f & is_clear
    return {
        "inserted": torch.where(active_f, inserted_n, inserted),
        "last_clear_side": torch.where(clear_f, side, last_clear_side),
        "last_clear_pos": torch.where(clear_f.unsqueeze(-1), points, last_clear_pos),
        "last_clear_center": torch.where(
            clear_f.unsqueeze(-1), center.unsqueeze(1).expand_as(points), last_clear_center
        ),
        "last_clear_radius_y": torch.where(
            clear_f, radius_y.unsqueeze(1).expand_as(d_curr), last_clear_radius_y
        ),
        "last_clear_radius_z": torch.where(
            clear_f, radius_z.unsqueeze(1).expand_as(d_curr), last_clear_radius_z
        ),
        "has_clear": has_clear | clear_f,
        "fwd_pending": torch.where(active_f, fwd_pending_n, fwd_pending),
        "rev_pending": torch.where(active_f, rev_pending_n, rev_pending),
        "fwd_count": torch.where(active_f, fwd_count_n, fwd_count),
        "rev_count": torch.where(active_f, rev_count_n, rev_count),
        "first_insert_step": first_n,
        "step_count": torch.where(active, step_n, step_count),
    }


def merge_ordered_thumb_inserted(was: torch.Tensor, now: torch.Tensor) -> torch.Tensor:
    """Order-preserving thumb latches: keep earlier, drop the reversed point and after.

    ``was`` / ``now`` are ``(N, K)``. ``was`` is unused; ``now`` already has
    per-station reverse applied. A False station clears every later station
    (cannot skip, and an exit invalidates points after it). Stations before
    that False stay as in ``now`` (True unless they also reversed).
    """
    del was
    return torch.cumprod(now.to(dtype=torch.int32), dim=1).to(dtype=torch.bool)


class ThumbOpeningSweepTracker:
    """Eval-only: required thumb landmarks must first cross tip→base in order.

    Default required stations are ``thdistal → thmiddle → thproximal``. Extra
    stations (usually ``thbase``) stay diagnostic. Each station uses the same
    PRE→POST / POST→PRE ellipse-crossing latch as the fingers. A new latch is
    accepted only when every earlier station is already True. Reverse unlatches
    that station and every later one; earlier stations stay True if they did
    not reverse. Passage resumes from the exited station. ``passed`` is true
    only while all required stations are currently latched.
    """

    def __init__(
        self,
        num_envs: int,
        n_stations: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        delta: float,
        confirm_frames: int,
        ellipse_threshold: float,
        hole_half_width: float = THUMB_SWEEP_HOLE_HALF_WIDTH_M,
        n_required: int | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.n_stations = int(n_stations)
        self.n_required = int(n_stations if n_required is None else n_required)
        self.n_required = max(1, min(self.n_required, self.n_stations))
        self.device = device
        self.dtype = dtype
        self.delta = float(delta)
        self.confirm_frames = max(1, int(confirm_frames))
        self.ellipse_threshold = float(ellipse_threshold)
        self.hole_half_width = float(hole_half_width)
        z = (self.num_envs, self.n_stations)
        self.inserted = torch.zeros(z, dtype=torch.bool, device=device)
        self.last_clear_side = torch.zeros(z, dtype=torch.int8, device=device)
        self.last_clear_pos = torch.zeros((self.num_envs, self.n_stations, 3), dtype=dtype, device=device)
        self.last_clear_center = torch.zeros((self.num_envs, self.n_stations, 3), dtype=dtype, device=device)
        self.last_clear_radius_y = torch.zeros(z, dtype=dtype, device=device)
        self.last_clear_radius_z = torch.zeros(z, dtype=dtype, device=device)
        self.has_clear = torch.zeros(z, dtype=torch.bool, device=device)
        self.fwd_pending = torch.zeros(z, dtype=torch.bool, device=device)
        self.rev_pending = torch.zeros(z, dtype=torch.bool, device=device)
        self.fwd_count = torch.zeros(z, dtype=torch.int32, device=device)
        self.rev_count = torch.zeros(z, dtype=torch.int32, device=device)
        self.next_idx = torch.zeros((self.num_envs,), dtype=torch.int32, device=device)
        self.confirm = torch.zeros((self.num_envs,), dtype=torch.int32, device=device)
        self.passed = torch.zeros((self.num_envs,), dtype=torch.bool, device=device)
        self.visit_step = torch.full(z, -1, dtype=torch.int32, device=device)
        self.step_count = torch.zeros((self.num_envs,), dtype=torch.int32, device=device)
        self.last_occupied = torch.zeros(z, dtype=torch.bool, device=device)

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
        self.next_idx[ids] = 0
        self.confirm[ids] = 0
        self.passed[ids] = False
        self.visit_step[ids] = -1
        self.step_count[ids] = 0
        self.last_occupied[ids] = False

    def update(
        self,
        nodes: torch.Tensor,
        center: torch.Tensor,
        radius_y: torch.Tensor,
        radius_z: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Advance ordered tip→base crossings. ``nodes`` is ``(N, K, 3)`` tip→base."""
        active = active.to(device=self.device, dtype=torch.bool)
        occupied = thumb_station_occupied(
            nodes,
            center,
            radius_y,
            radius_z,
            hole_half_width=self.hole_half_width,
            ellipse_threshold=self.ellipse_threshold,
        )
        latch = apply_opening_crossing_latch(
            nodes,
            center,
            radius_y,
            radius_z,
            active,
            delta=self.delta,
            confirm_frames=self.confirm_frames,
            ellipse_threshold=self.ellipse_threshold,
            inserted=self.inserted,
            last_clear_side=self.last_clear_side,
            last_clear_pos=self.last_clear_pos,
            last_clear_center=self.last_clear_center,
            last_clear_radius_y=self.last_clear_radius_y,
            last_clear_radius_z=self.last_clear_radius_z,
            has_clear=self.has_clear,
            fwd_pending=self.fwd_pending,
            rev_pending=self.rev_pending,
            fwd_count=self.fwd_count,
            rev_count=self.rev_count,
            first_insert_step=self.visit_step,
            step_count=self.step_count,
        )
        raw_inserted = latch["inserted"]
        ordered = merge_ordered_thumb_inserted(self.inserted, raw_inserted)
        rejected = raw_inserted & (~ordered)
        visit = latch["first_insert_step"]
        visit = torch.where(ordered, visit, torch.full_like(visit, -1))
        self.inserted = torch.where(active.unsqueeze(1), ordered, self.inserted)
        self.last_clear_side = latch["last_clear_side"]
        self.last_clear_pos = latch["last_clear_pos"]
        self.last_clear_center = latch["last_clear_center"]
        self.last_clear_radius_y = latch["last_clear_radius_y"]
        self.last_clear_radius_z = latch["last_clear_radius_z"]
        self.has_clear = latch["has_clear"]
        self.fwd_pending = latch["fwd_pending"] & (~rejected)
        self.rev_pending = latch["rev_pending"] & ordered
        self.fwd_count = torch.where(rejected, torch.zeros_like(latch["fwd_count"]), latch["fwd_count"])
        self.rev_count = torch.where(ordered, latch["rev_count"], torch.zeros_like(latch["rev_count"]))
        self.visit_step = torch.where(active.unsqueeze(1), visit, self.visit_step)
        self.step_count = latch["step_count"]
        missing = ~self.inserted
        first_gap = missing.to(dtype=torch.int64).argmax(dim=1).to(dtype=self.next_idx.dtype)
        all_on = ~missing.any(dim=1)
        nxt = torch.where(
            all_on,
            torch.full_like(first_gap, self.n_stations),
            first_gap,
        )
        required = self.inserted[:, : self.n_required].all(dim=1)
        self.next_idx = torch.where(active, nxt, self.next_idx)
        self.passed = torch.where(active, required, self.passed)
        self.last_occupied = torch.where(active.unsqueeze(1), occupied, self.last_occupied)
        return self.passed


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
        A confirmed reverse POST → PRE through the ellipse clears that finger.
        """
        latch = apply_opening_crossing_latch(
            distal,
            center,
            radius_y,
            radius_z,
            active,
            delta=self.delta,
            confirm_frames=self.confirm_frames,
            ellipse_threshold=self.ellipse_threshold,
            inserted=self.inserted,
            last_clear_side=self.last_clear_side,
            last_clear_pos=self.last_clear_pos,
            last_clear_center=self.last_clear_center,
            last_clear_radius_y=self.last_clear_radius_y,
            last_clear_radius_z=self.last_clear_radius_z,
            has_clear=self.has_clear,
            fwd_pending=self.fwd_pending,
            rev_pending=self.rev_pending,
            fwd_count=self.fwd_count,
            rev_count=self.rev_count,
            first_insert_step=self.first_insert_step,
            step_count=self.step_count,
        )
        self.inserted = latch["inserted"]
        self.last_clear_side = latch["last_clear_side"]
        self.last_clear_pos = latch["last_clear_pos"]
        self.last_clear_center = latch["last_clear_center"]
        self.last_clear_radius_y = latch["last_clear_radius_y"]
        self.last_clear_radius_z = latch["last_clear_radius_z"]
        self.has_clear = latch["has_clear"]
        self.fwd_pending = latch["fwd_pending"]
        self.rev_pending = latch["rev_pending"]
        self.fwd_count = latch["fwd_count"]
        self.rev_count = latch["rev_count"]
        self.first_insert_step = latch["first_insert_step"]
        self.step_count = latch["step_count"]
        active = active.to(device=self.device, dtype=torch.bool)
        n_now = self.inserted.sum(dim=1).to(dtype=self.max_inserted.dtype)
        self.max_inserted = torch.maximum(self.max_inserted, torch.where(active, n_now, self.max_inserted))
        self.ever_all = self.ever_all | (active & (n_now >= self.inserted.shape[1]))
        self.inserted_steps = self.inserted_steps + (
            self.inserted & active.unsqueeze(1)
        ).to(self.inserted_steps.dtype)
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


def _read_vec3_env(value: Any, env_id: int) -> tuple[float, float, float] | None:
    if value is None or not isinstance(value, torch.Tensor):
        return None
    if value.ndim < 2 or env_id >= int(value.shape[0]) or int(value.shape[-1]) < 3:
        return None
    row = value[env_id].reshape(-1)
    return float(row[0].item()), float(row[1].item()), float(row[2].item())


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
    distal_inserted_state: list[list[bool]] = field(default_factory=list)
    passage_state: list[list[bool]] = field(default_factory=list)
    d_finger: list[list[float]] = field(default_factory=list)
    sum_sq_all_joints: float = 0.0
    n_joint_samples: int = 0
    wrist_distance_at_success: float | None = None
    inserted_flags_at_success: list[bool] | None = None
    first_wrist_goal_step: int | None = None
    first_wrist_incomplete: bool = False
    inserted_flags_at_first_wrist_goal: list[bool] | None = None
    last_done_reason: str = "incomplete"
    first_all_five_frame: int | None = None
    first_all_five_frame_latched: int | None = None
    first_wrist_success_frame: int | None = None
    wrist_ok_ever: bool = False
    wrist_ok_after_all_five: bool = False
    wrist_distance_best_m: float | None = None
    wrist_distance_best_after_all_five_m: float | None = None
    wrist_distance_final_m: float | None = None
    wrist_vec_final: tuple[float, float, float] | None = None
    wrist_vec_best_after_all_five: tuple[float, float, float] | None = None
    wrist_success_threshold_m: float = 0.01
    last_live_ok: list[bool] = field(default_factory=lambda: [False] * 5)
    debug_rows: list[dict[str, Any]] = field(default_factory=list)
    last_pre_pos: list[tuple[float, float, float] | None] = field(default_factory=lambda: [None] * 5)
    last_pre_center: list[tuple[float, float, float] | None] = field(default_factory=lambda: [None] * 5)
    last_pre_radius_y: list[float | None] = field(default_factory=lambda: [None] * 5)
    last_pre_radius_z: list[float | None] = field(default_factory=lambda: [None] * 5)
    last_pre_pos_distal: list[tuple[float, float, float] | None] = field(default_factory=lambda: [None] * 5)
    last_pre_center_distal: list[tuple[float, float, float] | None] = field(default_factory=lambda: [None] * 5)
    last_pre_radius_y_distal: list[float | None] = field(default_factory=lambda: [None] * 5)
    last_pre_radius_z_distal: list[float | None] = field(default_factory=lambda: [None] * 5)
    knuckle_passage_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    distal_passage_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    passage_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    thumb_sweep_next: int = 0
    thumb_sweep_occupied: list[bool] = field(default_factory=list)
    thumb_sweep_inserted: list[bool] = field(default_factory=list)
    last_thumb_diag: dict[str, Any] = field(default_factory=dict)


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
    all_five_ever: bool = False
    all_five_retained: bool = False
    all_five_retained_latched: bool = False
    wrist_success: bool = False
    task_success: bool = False
    legacy_success: bool = False
    legacy_all_five: bool = False
    all_five_passage: bool = False
    all5_passage_wrist_incomplete: bool = False
    all5_passage_wrist_complete: bool = False
    strict_success: bool = False
    all_five_passed: bool = False
    thumb_passed: bool = False
    thdistal_frame: int | None = None
    thmiddle_frame: int | None = None
    thproximal_frame: int | None = None
    thbase_frame: int | None = None
    thumb_passage_duration_frames: int | None = None
    first_all_five_frame: int | None = None
    first_all_five_frame_latched: int | None = None
    first_wrist_success_frame: int | None = None
    wrist_success_threshold_m: float = 0.01
    wrist_distance_final_m: float | None = None
    wrist_distance_best_m: float | None = None
    wrist_distance_best_after_all_five_m: float | None = None
    wrist_ok_ever: bool = False
    wrist_ok_after_all_five: bool = False
    wrist_ok_at_end: bool = False
    wrist_vec_final: tuple[float, float, float] | None = None
    wrist_vec_best_after_all_five: tuple[float, float, float] | None = None
    failure_mode: str = FAILURE_MODE_NO_INSERTION
    wrist_shortfall_primary: str | None = None
    wrist_shortfall_tags: list[str] = field(default_factory=list)
    final_inserted_fingers_live: int = 0
    live_all_five: bool = False
    snag_suspect: bool = False
    live_ok: dict[str, bool] = field(default_factory=dict)
    inserted_latched: dict[str, bool] = field(default_factory=dict)
    final_inserted_fingers_latched: int = 0
    max_passed_fingers: int = 0
    num_ever_passed: int = 0
    ever_passed: dict[str, bool] = field(default_factory=dict)
    ever_passed_knuckle: dict[str, bool] = field(default_factory=dict)
    ever_passed_distal: dict[str, bool] = field(default_factory=dict)
    ever_all_inserted_knuckle: bool = False
    max_passed_knuckle: int = 0
    final_geometric_overlap: int = 0
    per_finger_geometric_overlap: dict[str, bool] = field(default_factory=dict)
    passage_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_thumb_diag: dict[str, Any] = field(default_factory=dict)

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
            "final_inserted_fingers_latched": self.final_inserted_fingers_latched,
            "max_inserted_fingers": self.max_inserted_fingers,
            "max_passed_fingers": self.max_passed_fingers,
            "num_ever_passed": self.num_ever_passed,
            "final_geometric_overlap": self.final_geometric_overlap,
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
        row["all_five_ever"] = int(self.all_five_ever)
        row["all_five_retained"] = int(self.all_five_retained)
        row["all_five_retained_latched"] = int(self.all_five_retained_latched)
        row["wrist_success"] = int(self.wrist_success)
        row["task_success"] = int(self.task_success)
        row["legacy_success"] = int(self.legacy_success)
        row["legacy_all_five"] = int(self.legacy_all_five)
        row["all_five_passage"] = int(self.all_five_passage)
        row["all5_passage_wrist_incomplete"] = int(self.all5_passage_wrist_incomplete)
        row["all5_passage_wrist_complete"] = int(self.all5_passage_wrist_complete)
        row["strict_success"] = int(self.all5_passage_wrist_complete)
        row["thumb_passed"] = int(self.thumb_passed)
        row["passed_count"] = int(self.max_passed_fingers)
        row["thdistal_frame"] = "" if self.thdistal_frame is None else self.thdistal_frame
        row["thmiddle_frame"] = "" if self.thmiddle_frame is None else self.thmiddle_frame
        row["thproximal_frame"] = "" if self.thproximal_frame is None else self.thproximal_frame
        row["thbase_frame"] = "" if self.thbase_frame is None else self.thbase_frame
        row["thumb_passage_duration_frames"] = (
            "" if self.thumb_passage_duration_frames is None else self.thumb_passage_duration_frames
        )
        row["first_all_five_frame"] = (
            "" if self.first_all_five_frame is None else self.first_all_five_frame
        )
        row["first_all_five_frame_latched"] = (
            "" if self.first_all_five_frame_latched is None else self.first_all_five_frame_latched
        )
        row["first_wrist_success_frame"] = (
            "" if self.first_wrist_success_frame is None else self.first_wrist_success_frame
        )
        row["wrist_success_threshold_m"] = f"{self.wrist_success_threshold_m:.8g}"
        row["wrist_distance_final_m"] = (
            "" if self.wrist_distance_final_m is None else f"{self.wrist_distance_final_m:.8g}"
        )
        row["wrist_distance_best_m"] = (
            "" if self.wrist_distance_best_m is None else f"{self.wrist_distance_best_m:.8g}"
        )
        row["wrist_distance_best_after_all_five_m"] = (
            ""
            if self.wrist_distance_best_after_all_five_m is None
            else f"{self.wrist_distance_best_after_all_five_m:.8g}"
        )
        row["wrist_ok_ever"] = int(self.wrist_ok_ever)
        row["wrist_ok_after_all_five"] = int(self.wrist_ok_after_all_five)
        row["wrist_ok_at_end"] = int(self.wrist_ok_at_end)
        fx = self.wrist_vec_final
        bx = self.wrist_vec_best_after_all_five
        row["wrist_dx_final_m"] = "" if fx is None else f"{fx[0]:.8g}"
        row["wrist_dy_final_m"] = "" if fx is None else f"{fx[1]:.8g}"
        row["wrist_dz_final_m"] = "" if fx is None else f"{fx[2]:.8g}"
        row["wrist_dx_best_after_all_five_m"] = "" if bx is None else f"{bx[0]:.8g}"
        row["wrist_dy_best_after_all_five_m"] = "" if bx is None else f"{bx[1]:.8g}"
        row["wrist_dz_best_after_all_five_m"] = "" if bx is None else f"{bx[2]:.8g}"
        row["failure_mode"] = self.failure_mode
        row["wrist_shortfall_primary"] = self.wrist_shortfall_primary or ""
        row["wrist_shortfall_tags"] = ",".join(self.wrist_shortfall_tags)
        row["final_inserted_fingers_live"] = self.final_geometric_overlap
        row["live_all_five"] = int(self.final_geometric_overlap == 5)
        row["snag_suspect"] = 0
        for name in FINGER_ORDER:
            row[f"{name}_live_ok"] = int(bool((self.live_ok or {}).get(name)))
            row[f"{name}_inserted_latched"] = int(bool((self.inserted_latched or {}).get(name)))
        return row

    def to_failure_mode_record(self) -> dict[str, Any]:
        fx = self.wrist_vec_final
        bx = self.wrist_vec_best_after_all_five
        return {
            "episode": self.episode,
            "env_id": self.env_id,
            "final_inserted_fingers": self.final_inserted_fingers,
            "final_inserted_fingers_latched": self.final_inserted_fingers_latched,
            "max_inserted_fingers": self.max_inserted_fingers,
            "max_passed_fingers": self.max_passed_fingers,
            "num_ever_passed": self.num_ever_passed,
            "all_five_passed": self.all_five_passage,
            "all_five_passage": self.all_five_passage,
            "all5_passage_wrist_incomplete": self.all5_passage_wrist_incomplete,
            "all5_passage_wrist_complete": self.all5_passage_wrist_complete,
            "thumb_passed": self.thumb_passed,
            "passed_count": self.max_passed_fingers,
            "thdistal_frame": self.thdistal_frame,
            "thmiddle_frame": self.thmiddle_frame,
            "thproximal_frame": self.thproximal_frame,
            "thbase_frame": self.thbase_frame,
            "thumb_passage_duration_frames": self.thumb_passage_duration_frames,
            "all_five_ever": self.all_five_ever,
            "all_five_retained": self.all_five_retained,
            "all_five_retained_latched": self.all_five_retained_latched,
            "wrist_success": self.wrist_success,
            "task_success": self.task_success,
            "legacy_success": self.legacy_success,
            "legacy_all_five": self.legacy_all_five,
            "strict_success": self.all5_passage_wrist_complete,
            "first_all_five_frame": self.first_all_five_frame,
            "first_all_five_frame_latched": self.first_all_five_frame_latched,
            "first_wrist_success_frame": self.first_wrist_success_frame,
            "wrist_success_threshold_m": self.wrist_success_threshold_m,
            "wrist_distance_final_m": self.wrist_distance_final_m,
            "wrist_distance_best_m": self.wrist_distance_best_m,
            "wrist_distance_best_after_all_five_m": self.wrist_distance_best_after_all_five_m,
            "wrist_ok_ever": self.wrist_ok_ever,
            "wrist_ok_after_all_five": self.wrist_ok_after_all_five,
            "wrist_ok_at_end": self.wrist_ok_at_end,
            "wrist_center_vector_final_m": None if fx is None else {"x": fx[0], "y": fx[1], "z": fx[2]},
            "wrist_center_vector_best_after_all_five_m": (
                None if bx is None else {"x": bx[0], "y": bx[1], "z": bx[2]}
            ),
            "failure_mode": self.failure_mode,
            "wrist_shortfall_primary": self.wrist_shortfall_primary,
            "wrist_shortfall_tags": list(self.wrist_shortfall_tags),
            "final_geometric_overlap": self.final_geometric_overlap,
            "final_inserted_fingers_live": self.final_geometric_overlap,
            "live_all_five": self.final_geometric_overlap == 5,
            "snag_suspect": False,
            "per_finger_geometric_overlap": dict(self.per_finger_geometric_overlap or self.live_ok),
            "per_finger_live_ok": dict(self.live_ok),
            "per_finger_inserted_latched": dict(self.inserted_latched),
            "per_finger_ever_passed": dict(self.ever_passed or self.inserted),
            "per_finger_ever_passed_knuckle": dict(self.ever_passed_knuckle),
            "per_finger_ever_passed_distal": dict(self.ever_passed_distal),
            "ever_all_inserted_knuckle": self.ever_all_inserted_knuckle,
            "max_passed_knuckle": self.max_passed_knuckle,
            "thumb_eval_requires": (
                "thdistal → thmiddle → thproximal ordered PRE→POST "
                "(reverse clears that station and later ones; earlier stay if they did not reverse); thbase diagnostic)"
            ),
            "hand_rms": self.hand_rms,
            "finger_rms": dict(self.finger_rms),
            "finger_peak": dict(self.finger_peak),
            "passage_events": dict(self.passage_events),
            "last_thumb_diag": dict(self.last_thumb_diag),
        }


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
        record_insertion_debug: bool = False,
        video_dir: Path | None = None,
        video_failure_modes: set[str] | None = None,
        video_env_ids: set[int] | None = None,
        video_shortfalls: set[str] | None = None,
        video_max: int | None = None,
        success_definition: str = "legacy",
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
        self.record_insertion_debug = bool(record_insertion_debug) or self.debug_insertion
        self.video_dir = Path(video_dir) if video_dir is not None else None
        self.video_failure_modes = set(video_failure_modes or ())
        self.video_env_ids = set(video_env_ids or ())
        self.video_shortfalls = set(video_shortfalls or ())
        self.video_max = int(video_max) if video_max is not None else None
        requested = str(success_definition).lower()
        if requested == "strict":
            print(
                f"[{log_prefix}] WARNING: --success-definition strict no longer overrides "
                "Task success. Official task_success stays legacy knuckle all-five + wrist. "
                "Geometric completion is reported separately as all-five passage + wrist complete."
            )
        self.success_definition = "legacy"
        self.kept_failure_videos: list[Path] = []
        self._seen_video_paths: set[Path] = set()
        self._parallel_video = int(getattr(raw_env, "num_envs", 1)) > 1
        self._debug_prev_inserted: dict[int, list[bool]] = {eid: [False] * 5 for eid in eval_env_ids}
        self._debug_prev_side: dict[int, list[str]] = {eid: ["?"] * 5 for eid in eval_env_ids}

        self.csv_path = self.output_dir / "episode_metrics.csv"
        self.summary_path = self.output_dir / "evaluation_summary.json"
        self.partial_path = self.output_dir / "evaluation_summary.partial.json"
        self.histogram_path = self.output_dir / "inserted_finger_histogram.json"
        self.failure_modes_json_path = self.output_dir / "failure_modes.json"
        self.failure_modes_csv_path = self.output_dir / "failure_modes.csv"

        self.episodes: list[EpisodeMetrics] = []
        self._running = {eid: _RunningEpisode(env_id=eid) for eid in self.eval_env_ids}
        self._tracker: FingerCrossingTracker | None = None
        self._distal_tracker: FingerCrossingTracker | None = None
        self._thumb_sweep: ThumbOpeningSweepTracker | None = None
        self.thumb_sweep_ids: list[int] = []
        self.thumb_sweep_names: list[str] = []

        self.hand = getattr(raw_env, "hand", None)
        self.finger_joint_ids: dict[str, list[int]] = {name: [] for name in FINGER_ORDER}
        self.all_finger_joint_ids: list[int] = []
        self.resolved_joint_groups: dict[str, list[str]] = {name: [] for name in FINGER_ORDER}
        self.base_body_ids: dict[str, int | None] = {name: None for name in FINGER_ORDER}
        self.resolved_base_bodies: dict[str, str | None] = {name: None for name in FINGER_ORDER}
        self.distal_body_ids: dict[str, int | None] = {name: None for name in FINGER_ORDER}
        self.resolved_distal_bodies: dict[str, str | None] = {name: None for name in FINGER_ORDER}
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

        from play_common import (
            load_video_from_eval,
            parse_video_env_ids,
            parse_video_failure_modes,
            parse_video_shortfalls,
        )

        video_modes = parse_video_failure_modes(getattr(args, "video_failures", None))
        video_ids = parse_video_env_ids(getattr(args, "video_env_ids", None))
        video_short = parse_video_shortfalls(getattr(args, "video_shortfall", None))
        from_eval = getattr(args, "video_from_eval", None)
        if from_eval:
            prev = load_video_from_eval(from_eval)
            if prev is not None:
                print(f"[{session.log_prefix}] video-from-eval {prev['path']}")
                for mode, ids in prev["by_mode"].items():
                    print(
                        f"[{session.log_prefix}]   {mode}: {len(ids)}  env_ids={','.join(str(i) for i in ids)}"
                    )
                if not video_modes:
                    video_modes = set(prev["by_mode"])
                if not video_ids and n_envs > 1:
                    video_ids = {eid for ids in prev["by_mode"].values() for eid in ids}
                elif video_ids and n_envs == 1:
                    print(
                        f"[{session.log_prefix}] NOTE: --num_envs 1 always has env_id=0, "
                        "so previous env_ids are not replayed. Filtering by failure mode instead."
                    )

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
            record_insertion_debug=bool(getattr(args, "record_insertion_debug", False)),
            video_dir=Path(args.video_dir) if getattr(args, "video", False) else None,
            video_failure_modes=video_modes,
            video_env_ids=video_ids,
            video_shortfalls=video_short,
            video_max=getattr(args, "video_max", None),
            success_definition=str(getattr(args, "success_definition", "legacy") or "legacy"),
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
            f"task_success=legacy_knuckle_all5_and_wrist "
            f"eval_thumb=thdistal->thmiddle->thproximal ordered PRE-POST "
            f"envs={eval_env_ids}"
            + (
                f" debug_insertion every {collector.debug_insertion_interval} steps"
                if collector.debug_insertion
                else ""
            )
            + (
                f" insertion_debug CSV -> {collector.output_dir / 'insertion_debug'}"
                if collector.record_insertion_debug
                else ""
            )
        )
        if str(getattr(args, "success_definition", "legacy") or "legacy").lower() == "strict":
            print(
                f"[{session.log_prefix}] NOTE: --success-definition strict is ignored for "
                "Task success; see Finger Passage / Passage Outcome Breakdown instead."
            )
        if collector.video_filter_active():
            dest = collector.output_dir / "failure_videos"
            if collector._parallel_video:
                print(
                    f"[{session.log_prefix}] failure videos: --num_envs {n_envs} records ONE tiled "
                    f"clip of all envs (not one file per failure). File is saved at the end to {dest}. "
                    f"Use --num_envs 1 --video-failures ... --video-max N for individual failure clips."
                )
            else:
                print(
                    f"[{session.log_prefix}] failure videos: modes="
                    f"{sorted(collector.video_failure_modes) or ['(env-id filter only)']} "
                    f"env_ids={sorted(collector.video_env_ids) or 'any'} "
                    f"shortfall={sorted(collector.video_shortfalls) or 'any'} "
                    f"max={collector.video_max if collector.video_max is not None else 'none'} "
                    f"-> {dest}"
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
        missing_distal: list[str] = []
        for finger, candidates in DISTAL_BODY_CANDIDATES.items():
            idx = _resolve_body_index(body_names, candidates)
            self.distal_body_ids[finger] = idx
            if idx is None:
                missing_distal.append(finger)
                self.resolved_distal_bodies[finger] = None
            else:
                self.resolved_distal_bodies[finger] = body_names[idx]
        if missing_distal:
            print(
                f"[{self.log_prefix}] WARNING: missing finger-distal bodies for {missing_distal}; "
                "live containment will use knuckles only"
            )
        print(
            f"[{self.log_prefix}] insertion points: finger-base COM "
            f"{self.resolved_base_bodies}"
        )
        print(
            f"[{self.log_prefix}] live-containment points: distal COM "
            f"{self.resolved_distal_bodies}"
        )
        self.thumb_sweep_ids = []
        self.thumb_sweep_names = []
        for candidates in THUMB_SWEEP_BODY_CANDIDATES:
            idx = _resolve_body_index(body_names, candidates)
            if idx is None:
                continue
            self.thumb_sweep_ids.append(idx)
            self.thumb_sweep_names.append(body_names[idx])
        if len(self.thumb_sweep_ids) < THUMB_SWEEP_MIN_STATIONS:
            print(
                f"[{self.log_prefix}] WARNING: thumb landmark crossing needs "
                f">={THUMB_SWEEP_MIN_STATIONS} bodies, got {self.thumb_sweep_names}; "
                "falling back to thbase AND thdistal"
            )
        print(
            f"[{self.log_prefix}] eval thumb passage is ordered PRE→POST "
            f"{self.thumb_sweep_names[:THUMB_SWEEP_REQUIRED_STATIONS]} "
            f"(reverse clears that station and later ones; earlier stay if they did not reverse); "
            f"thbase diagnostic only; training latch stays knuckle-only)"
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

    def _ensure_distal_tracker(self, like: torch.Tensor) -> FingerCrossingTracker:
        if self._distal_tracker is None:
            n = int(getattr(self.raw_env, "num_envs", like.shape[0]))
            self._distal_tracker = FingerCrossingTracker(
                n,
                like.device,
                like.dtype,
                delta=self.insertion_delta_m,
                confirm_frames=self.insertion_confirm_frames,
                ellipse_threshold=self.insertion_ellipse_threshold,
            )
        return self._distal_tracker

    def _ensure_thumb_sweep(self, like: torch.Tensor) -> ThumbOpeningSweepTracker | None:
        n_stat = len(self.thumb_sweep_ids)
        if n_stat < THUMB_SWEEP_MIN_STATIONS:
            return None
        if self._thumb_sweep is None or self._thumb_sweep.n_stations != n_stat:
            n = int(getattr(self.raw_env, "num_envs", like.shape[0]))
            self._thumb_sweep = ThumbOpeningSweepTracker(
                n,
                n_stat,
                like.device,
                like.dtype,
                delta=self.insertion_delta_m,
                confirm_frames=self.insertion_confirm_frames,
                ellipse_threshold=self.insertion_ellipse_threshold,
                hole_half_width=THUMB_SWEEP_HOLE_HALF_WIDTH_M,
                n_required=min(THUMB_SWEEP_REQUIRED_STATIONS, n_stat),
            )
        return self._thumb_sweep

    def _stack_body_ids_env_local(self, body_ids: dict[str, int | None]) -> torch.Tensor | None:
        """Return ``(num_envs, 5, 3)`` body COMs in env-local frame, or None."""
        hand = self.hand
        if hand is None or any(body_ids[name] is None for name in FINGER_ORDER):
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
            idx = body_ids[name]
            cols.append(body_pos_w[:, idx] - origins)
        return torch.stack(cols, dim=1)

    def _stack_finger_base_env_local(self) -> torch.Tensor | None:
        """Return ``(num_envs, 5, 3)`` finger-base COMs in env-local frame, or None."""
        return self._stack_body_ids_env_local(self.base_body_ids)

    def _stack_finger_distal_env_local(self) -> torch.Tensor | None:
        """Return ``(num_envs, 5, 3)`` distal COMs in env-local frame, or None."""
        return self._stack_body_ids_env_local(self.distal_body_ids)

    def _insertion_geom(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        knuckle = self._stack_finger_base_env_local()
        raw = self.raw_env
        cent = getattr(raw, "goal_cent_pos", None)
        east = getattr(raw, "goal_east_pos", None)
        west = getattr(raw, "goal_west_pos", None)
        north = getattr(raw, "goal_north_pos", None)
        south = getattr(raw, "goal_south_pos", None)
        if knuckle is None or cent is None or east is None or west is None or north is None or south is None:
            return None
        radius_y, radius_z = opening_radii(east, west, north, south)
        return knuckle, self._stack_finger_distal_env_local(), cent, radius_y, radius_z

    def _stack_thumb_sweep_env_local(self) -> torch.Tensor | None:
        """Return ``(num_envs, K, 3)`` thumb polyline COMs tip→base, or None."""
        hand = self.hand
        if hand is None or len(self.thumb_sweep_ids) < THUMB_SWEEP_MIN_STATIONS:
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
        cols = [body_pos_w[:, idx] - origins for idx in self.thumb_sweep_ids]
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
        if self.video_max is not None and len(self.kept_failure_videos) >= self.video_max:
            return True
        return len(self.episodes) >= self.max_episodes

    def video_filter_active(self) -> bool:
        return bool(self.video_failure_modes or self.video_env_ids or self.video_shortfalls or self.video_max)

    def _episode_matches_video_filter(self, ep: EpisodeMetrics) -> bool:
        if self.video_env_ids and ep.env_id not in self.video_env_ids and ep.episode not in self.video_env_ids:
            return False
        modes = self.video_failure_modes
        if modes:
            matched = False
            if "all_fail" in modes and ep.failure_mode != FAILURE_MODE_FULL_AND_WRIST:
                matched = True
            if ep.failure_mode in modes:
                matched = True
            if "snag_suspect" in modes and ep.failure_mode == FAILURE_MODE_FULL_INCOMPLETE_WRIST:
                matched = True
            if not matched:
                return False
        if self.video_shortfalls:
            tags = set(ep.wrist_shortfall_tags)
            if ep.wrist_shortfall_primary:
                tags.add(ep.wrist_shortfall_primary)
            if not (tags & self.video_shortfalls):
                return False
        return bool(modes or self.video_env_ids or self.video_shortfalls)

    def _iter_new_mp4s(self) -> list[Path]:
        if self.video_dir is None or not self.video_dir.is_dir():
            return []
        found: list[Path] = []
        for p in self.video_dir.glob("*.mp4"):
            if "failure_videos" in p.parts:
                continue
            resolved = p.resolve()
            if resolved in self._seen_video_paths:
                continue
            found.append(p)
        return found

    def _harvest_episode_video(self, ep: EpisodeMetrics) -> None:
        """Keep or drop a per-episode clip. Multi-env tiled recording is saved in ``finalize``."""
        if self.video_dir is None or not self.video_filter_active():
            return
        if self._parallel_video:
            return
        keep = self._episode_matches_video_filter(ep)
        newest = max(self._iter_new_mp4s(), key=lambda p: p.stat().st_mtime, default=None)
        if newest is None:
            if keep:
                print(
                    f"[{self.log_prefix}] WARNING: matched failure ep={ep.episode} "
                    f"mode={ep.failure_mode} but no new mp4 was written yet"
                )
            return
        self._seen_video_paths.add(newest.resolve())
        if keep:
            dest = self._move_to_failure_videos(
                newest,
                f"{ep.failure_mode}_ep{ep.episode:04d}_env{ep.env_id}_"
                f"{(ep.wrist_shortfall_primary or 'na').split('_')[0]}.mp4",
            )
            print(
                f"[{self.log_prefix}] kept failure video ({len(self.kept_failure_videos)}"
                f"{'' if self.video_max is None else f'/{self.video_max}'}) "
                f"-> {dest}"
            )
        else:
            newest.unlink(missing_ok=True)

    def _move_to_failure_videos(self, src: Path, name: str) -> Path:
        dest_dir = self.output_dir / "failure_videos"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        src.replace(dest)
        self._seen_video_paths.add(dest.resolve())
        self.kept_failure_videos.append(dest)
        return dest

    def _harvest_leftover_videos(self) -> None:
        leftovers = self._iter_new_mp4s()
        if not leftovers:
            return
        n_fail = sum(1 for ep in self.episodes if ep.failure_mode != FAILURE_MODE_FULL_AND_WRIST)
        for src in leftovers:
            self._seen_video_paths.add(src.resolve())
            if self._parallel_video:
                dest = self._move_to_failure_videos(src, f"tiled_{self.num_eval_label()}_{src.name}")
                print(
                    f"[{self.log_prefix}] saved tiled playback video "
                    f"({n_fail} failure / {len(self.episodes)} episodes) -> {dest}"
                )
                continue
            dest = self._move_to_failure_videos(src, src.name)
            print(f"[{self.log_prefix}] saved leftover video -> {dest}")

    def num_eval_label(self) -> str:
        return f"{len(self.eval_env_ids)}envs"

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
        distal_inserted = self._update_distal_insertion(done)
        thumb_sweep = self._update_thumb_sweep(done)
        d_finger, sum_sq, n_j = self._compute_joint_deviation()
        geom = self._insertion_geom()
        live = None
        if geom is not None:
            knuckle, distal, cent, radius_y, radius_z = geom
            live = live_containment_flags(
                knuckle,
                distal,
                cent,
                radius_y,
                radius_z,
                delta=self.insertion_delta_m,
                ellipse_threshold=self.insertion_ellipse_threshold,
            )

        for env_id in self.eval_env_ids:
            if self.is_complete():
                break
            run = self._running[env_id]
            run.steps += 1
            if isinstance(rewards, torch.Tensor) and env_id < rewards.shape[0]:
                run.episode_return += float(rewards[env_id].reshape(-1)[0].item())

            if inserted is not None:
                knuckle_flags = [bool(inserted[env_id, i].item()) for i in range(5)]
            else:
                knuckle_flags = [False] * 5
            if distal_inserted is not None and env_id < int(distal_inserted.shape[0]):
                distal_flags = [bool(distal_inserted[env_id, i].item()) for i in range(5)]
            else:
                distal_flags = [False] * 5
            thumb_passed = None
            if thumb_sweep is not None and env_id < int(thumb_sweep.shape[0]):
                thumb_passed = bool(thumb_sweep[env_id].item())
            flags = combine_eval_passage_flags(
                knuckle_flags, distal_flags, thumb_passed=thumb_passed
            )
            if self._thumb_sweep is not None and env_id < int(self._thumb_sweep.next_idx.shape[0]):
                run.thumb_sweep_next = int(self._thumb_sweep.next_idx[env_id].item())
                run.thumb_sweep_occupied = [
                    bool(self._thumb_sweep.last_occupied[env_id, i].item())
                    for i in range(self._thumb_sweep.n_stations)
                ]
                run.thumb_sweep_inserted = [
                    bool(self._thumb_sweep.inserted[env_id, i].item())
                    for i in range(self._thumb_sweep.n_stations)
                ]
            self._capture_thumb_diag(run, env_id, knuckle_flags, flags)
            run.inserted_state.append(knuckle_flags)
            run.distal_inserted_state.append(distal_flags)
            run.passage_state.append(flags)
            live_flags = (
                [bool(live[env_id, i].item()) for i in range(5)]
                if live is not None and env_id < int(live.shape[0])
                else [False] * 5
            )
            run.last_live_ok = live_flags
            if self.debug_insertion and env_id == self.eval_env_ids[0]:
                self._debug_insertion_env(
                    env_id,
                    run.steps,
                    knuckle_flags,
                    live_flags,
                    distal_flags=distal_flags,
                    passage_flags=flags,
                )

            log = infos.get("log") if isinstance(infos, dict) else None
            if not isinstance(log, dict):
                extras = getattr(raw, "extras", None) or {}
                log = extras.get("log") if isinstance(extras, dict) else {}
            thr = float(getattr(getattr(raw, "cfg", None), "bracelet_success_threshold", 0.01))
            run.wrist_success_threshold_m = thr
            wrist_dist = None
            if isinstance(log, dict):
                wrist_dist = _read_float_env(log.get("wrist_center_distance"), env_id)
            if wrist_dist is None:
                wrist_dist = _read_float_env(getattr(raw, "wrist_center_euclidean_distance", None), env_id)
            wrist_vec = _read_vec3_env(getattr(raw, "wrist_center_distance", None), env_id)
            if self.record_insertion_debug:
                run.debug_rows.append(
                    self._insertion_debug_row(
                        env_id,
                        run.steps,
                        flags,
                        live_flags,
                        wrist_dist=wrist_dist,
                        wrist_vec=wrist_vec,
                    )
                )
            wrist_ok = _as_bool_scalar((log or {}).get("wrist_within_goal"), env_id)
            if wrist_dist is not None:
                wrist_ok = wrist_dist < thr
            frame = run.steps - 1
            eval_all_five = all(flags)
            latched_all_five = all(knuckle_flags)
            live_all_five = all(live_flags)
            if wrist_dist is not None:
                run.wrist_distance_final_m = wrist_dist
                if run.wrist_distance_best_m is None or wrist_dist < run.wrist_distance_best_m:
                    run.wrist_distance_best_m = wrist_dist
            if wrist_vec is not None:
                run.wrist_vec_final = wrist_vec
            if wrist_ok:
                run.wrist_ok_ever = True
                if run.first_wrist_success_frame is None:
                    run.first_wrist_success_frame = frame
            if eval_all_five and run.first_all_five_frame is None:
                run.first_all_five_frame = frame
            if latched_all_five and run.first_all_five_frame_latched is None:
                run.first_all_five_frame_latched = frame
            if eval_all_five and wrist_dist is not None:
                if (
                    run.wrist_distance_best_after_all_five_m is None
                    or wrist_dist < run.wrist_distance_best_after_all_five_m
                ):
                    run.wrist_distance_best_after_all_five_m = wrist_dist
                    if wrist_vec is not None:
                        run.wrist_vec_best_after_all_five = wrist_vec
                if wrist_ok:
                    run.wrist_ok_after_all_five = True
            if wrist_ok and run.first_wrist_goal_step is None and not latched_all_five:
                run.first_wrist_goal_step = frame
                run.first_wrist_incomplete = True
                run.inserted_flags_at_first_wrist_goal = list(knuckle_flags)
            if geom is not None:
                self._record_passage_progress(
                    run,
                    env_id,
                    knuckle_flags,
                    distal_flags,
                    flags,
                    geom,
                    frame,
                    wrist_dist,
                    wrist_vec,
                )

            if (not run.motion_locked) and read_motion_locked(infos, raw, env_id):
                run.motion_locked = True
                run.motion_lock_step = run.steps - 1
                run.inserted_flags_at_success = list(knuckle_flags)
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
            if self._distal_tracker is not None:
                self._distal_tracker.reset_envs([env_id])
            if self._thumb_sweep is not None:
                self._thumb_sweep.reset_envs([env_id])
            return
        ep = self._build_episode(run, terminated=terminated, truncated=truncated)
        self.episodes.append(ep)
        self._append_csv(ep)
        self._write_summary(partial=True)
        self._harvest_episode_video(ep)
        print(
            f"[{self.log_prefix}] episode {ep.episode}: success={int(ep.success)} "
            f"passed={ep.max_passed_fingers}/5 ever={int(ep.num_ever_passed)} "
            f"ever_all={int(ep.ever_all_inserted)} "
            f"wrist_ok_after_5={int(ep.wrist_success)} success={int(ep.success)} "
            f"mode={ep.failure_mode} lock_step={ep.motion_lock_step} "
            f"done={ep.episode_done_reason} steps={ep.episode_length_steps} env={env_id}"
            + (
                f" first_wrist_incomplete={ep.missing_fingers_at_first_wrist_goal}"
                if ep.first_wrist_goal_step is not None
                else ""
            )
        )
        self._flush_insertion_debug(ep, run)
        self._running[env_id] = _RunningEpisode(env_id=env_id)
        self._debug_prev_inserted[env_id] = [False] * 5
        self._debug_prev_side[env_id] = ["?"] * 5
        if self._tracker is not None:
            self._tracker.reset_envs([env_id])
        if self._distal_tracker is not None:
            self._distal_tracker.reset_envs([env_id])
        if self._thumb_sweep is not None:
            self._thumb_sweep.reset_envs([env_id])

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

    def _passage_stats_from_rows(
        self,
        rows: list[list[bool]],
    ) -> tuple[int, bool, dict[str, float | None], dict[str, int]]:
        first_t = {name: None for name in FINGER_ORDER}
        insert_steps = {name: 0 for name in FINGER_ORDER}
        max_n = max((sum(1 for v in row if v) for row in rows), default=0)
        ever_all = any(all(row) for row in rows)
        for i, name in enumerate(FINGER_ORDER):
            for step_i, row in enumerate(rows):
                if row[i]:
                    first_t[name] = step_i * self.control_dt
                    break
            insert_steps[name] = sum(1 for row in rows if row[i])
        return max_n, ever_all, first_t, insert_steps

    def _build_episode(self, run: _RunningEpisode, *, terminated: bool, truncated: bool) -> EpisodeMetrics:
        knuckle_max, knuckle_ever, knuckle_first, knuckle_steps = self._snapshot_tracker(run.env_id)
        inserted_latched = {name: False for name in FINGER_ORDER}
        if run.inserted_state:
            last = run.inserted_state[-1]
            for i, name in enumerate(FINGER_ORDER):
                inserted_latched[name] = bool(last[i])
        if knuckle_max == 0 and run.inserted_state:
            knuckle_max = max((sum(1 for v in row if v) for row in run.inserted_state), default=0)
            knuckle_ever = knuckle_ever or any(all(row) for row in run.inserted_state)
        for i, name in enumerate(FINGER_ORDER):
            if knuckle_first[name] is None:
                for step_i, row in enumerate(run.inserted_state):
                    if row[i]:
                        knuckle_first[name] = step_i * self.control_dt
                        break
            if knuckle_steps[name] == 0 and run.inserted_state:
                knuckle_steps[name] = sum(1 for row in run.inserted_state if row[i])

        if run.passage_state:
            max_n, ever_all, first_t, insert_steps = self._passage_stats_from_rows(run.passage_state)
            final_flags = last_passage_flags(run.passage_state)
        else:
            max_n, ever_all, first_t, insert_steps = knuckle_max, knuckle_ever, knuckle_first, knuckle_steps
            final_flags = [bool(inserted_latched[name]) for name in FINGER_ORDER]
        insert_ratio = {
            name: insert_steps[name] / float(max(run.steps, 1)) for name in FINGER_ORDER
        }
        ever_passed_distal = {name: False for name in FINGER_ORDER}
        for i, name in enumerate(FINGER_ORDER):
            ever_passed_distal[name] = any(row[i] for row in run.distal_inserted_state)

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
        live_ok = {name: bool(run.last_live_ok[i]) for i, name in enumerate(FINGER_ORDER)}
        ever_passed = {name: first_t[name] is not None for name in FINGER_ORDER}
        ever_passed_knuckle = {
            name: knuckle_first[name] is not None or inserted_latched[name] for name in FINGER_ORDER
        }
        final_n_latched = sum(1 for name in FINGER_ORDER if inserted_latched[name])
        knuckle_ever = bool(knuckle_ever or knuckle_max >= 5)
        overlap_n = sum(1 for v in live_ok.values() if v)
        passage = make_episode_passage_result(
            passed=final_flags,
            wrist_ok_ever=run.wrist_ok_ever,
            legacy_all_five=knuckle_ever,
            legacy_success=bool(run.motion_locked),
        )
        max_passed = passage.passed_count
        num_ever_passed = sum(1 for name in FINGER_ORDER if ever_passed[name])
        inserted = dict(passage.passed)
        thumb_t = thumb_sweep_timing(run.passage_events)
        success = passage.task_success
        outcome = classify_insertion_outcome(
            max_inserted=max_n,
            ever_all=ever_all,
            final_all=passage.all_five_passage,
            success=passage.legacy_success,
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
        wrist_success = passage.wrist_ok_ever
        task_success = passage.task_success
        first_all_five = run.first_all_five_frame
        first_all_five_latched = run.first_all_five_frame_latched
        if first_all_five is None and ever_all:
            rows = run.passage_state or run.inserted_state
            for step_i, row in enumerate(rows):
                if all(row):
                    first_all_five = step_i
                    break
        if first_all_five_latched is None:
            first_all_five_latched = first_all_five
        failure_mode = passage.failure_mode
        shortfall_primary = None
        shortfall_tags: list[str] = []
        if failure_mode == FAILURE_MODE_FULL_INCOMPLETE_WRIST:
            shortfall_primary, shortfall_tags = classify_wrist_shortfall(
                threshold_m=run.wrist_success_threshold_m,
                wrist_ok_ever=run.wrist_ok_ever,
                wrist_ok_after_all_five=run.wrist_ok_after_all_five,
                first_wrist_success_frame=run.first_wrist_success_frame,
                first_all_five_frame=first_all_five,
                best_after_all_five_m=run.wrist_distance_best_after_all_five_m,
                final_distance_m=run.wrist_distance_final_m,
            )
        final_dist = run.wrist_distance_final_m
        wrist_ok_at_end = (
            final_dist is not None and final_dist < run.wrist_success_threshold_m
        )
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
            inserted_latched=inserted_latched,
            insert_ratio=insert_ratio,
            insert_steps=insert_steps,
            first_insert_time_s=first_t,
            num_inserted_fingers=max_passed,
            final_inserted_fingers=max_passed,
            final_inserted_fingers_latched=final_n_latched,
            max_inserted_fingers=max_n,
            ever_all_inserted=ever_all,
            final_all_inserted=passage.all_five_passage,
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
            all_five_ever=bool(ever_all),
            all_five_retained=passage.all_five_passage,
            all_five_retained_latched=final_n_latched == 5,
            wrist_success=wrist_success,
            task_success=task_success,
            legacy_success=passage.task_success,
            legacy_all_five=passage.legacy_all_five,
            all_five_passage=passage.all_five_passage,
            all5_passage_wrist_incomplete=passage.all5_passage_wrist_incomplete,
            all5_passage_wrist_complete=passage.all5_passage_wrist_complete,
            strict_success=passage.all5_passage_wrist_complete,
            all_five_passed=passage.all_five_passage,
            thumb_passed=passage.thumb_passed,
            thdistal_frame=thumb_t["thdistal_frame"],
            thmiddle_frame=thumb_t["thmiddle_frame"],
            thproximal_frame=thumb_t["thproximal_frame"],
            thbase_frame=thumb_t["thbase_frame"],
            thumb_passage_duration_frames=thumb_t["thumb_passage_duration_frames"],
            first_all_five_frame=first_all_five,
            first_all_five_frame_latched=first_all_five_latched,
            first_wrist_success_frame=run.first_wrist_success_frame,
            wrist_success_threshold_m=run.wrist_success_threshold_m,
            wrist_distance_final_m=run.wrist_distance_final_m,
            wrist_distance_best_m=run.wrist_distance_best_m,
            wrist_distance_best_after_all_five_m=run.wrist_distance_best_after_all_five_m,
            wrist_ok_ever=run.wrist_ok_ever,
            wrist_ok_after_all_five=run.wrist_ok_after_all_five,
            wrist_ok_at_end=wrist_ok_at_end,
            wrist_vec_final=run.wrist_vec_final,
            wrist_vec_best_after_all_five=run.wrist_vec_best_after_all_five,
            failure_mode=failure_mode,
            wrist_shortfall_primary=shortfall_primary,
            wrist_shortfall_tags=shortfall_tags,
            final_inserted_fingers_live=overlap_n,
            live_all_five=overlap_n == 5,
            snag_suspect=False,
            live_ok=live_ok,
            max_passed_fingers=max_passed,
            num_ever_passed=num_ever_passed,
            ever_passed=ever_passed,
            ever_passed_knuckle=ever_passed_knuckle,
            ever_passed_distal=ever_passed_distal,
            ever_all_inserted_knuckle=knuckle_ever,
            max_passed_knuckle=int(knuckle_max),
            final_geometric_overlap=overlap_n,
            per_finger_geometric_overlap=live_ok,
            passage_events=dict(run.passage_events),
            last_thumb_diag=dict(run.last_thumb_diag),
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

    def _update_distal_insertion(self, done: torch.Tensor) -> torch.Tensor | None:
        """Eval-only distal crossing tracker. Does not affect training latch."""
        raw = self.raw_env
        if getattr(raw, "_is_free_space_mode", lambda: False)():
            return None
        distal = self._stack_finger_distal_env_local()
        cent = getattr(raw, "goal_cent_pos", None)
        east = getattr(raw, "goal_east_pos", None)
        west = getattr(raw, "goal_west_pos", None)
        north = getattr(raw, "goal_north_pos", None)
        south = getattr(raw, "goal_south_pos", None)
        if distal is None or cent is None or east is None or west is None or north is None or south is None:
            return None
        radius_y, radius_z = opening_radii(east, west, north, south)
        tracker = self._ensure_distal_tracker(distal)
        active = torch.zeros((tracker.num_envs,), dtype=torch.bool, device=distal.device)
        for env_id in self.eval_env_ids:
            if env_id < tracker.num_envs and not bool(done[env_id].item()):
                active[env_id] = True
        return tracker.update(distal, cent, radius_y, radius_z, active)

    def _update_thumb_sweep(self, done: torch.Tensor) -> torch.Tensor | None:
        """Eval-only independent thumb landmark crossings. Does not affect training latch."""
        raw = self.raw_env
        if getattr(raw, "_is_free_space_mode", lambda: False)():
            return None
        nodes = self._stack_thumb_sweep_env_local()
        cent = getattr(raw, "goal_cent_pos", None)
        east = getattr(raw, "goal_east_pos", None)
        west = getattr(raw, "goal_west_pos", None)
        north = getattr(raw, "goal_north_pos", None)
        south = getattr(raw, "goal_south_pos", None)
        if nodes is None or cent is None or east is None or west is None or north is None or south is None:
            return None
        radius_y, radius_z = opening_radii(east, west, north, south)
        tracker = self._ensure_thumb_sweep(nodes)
        if tracker is None:
            return None
        active = torch.zeros((tracker.num_envs,), dtype=torch.bool, device=nodes.device)
        for env_id in self.eval_env_ids:
            if env_id < tracker.num_envs and not bool(done[env_id].item()):
                active[env_id] = True
        return tracker.update(nodes, cent, radius_y, radius_z, active)

    def _capture_thumb_diag(
        self,
        run: _RunningEpisode,
        env_id: int,
        knuckle_flags: list[bool],
        passage_flags: list[bool],
    ) -> None:
        raw = self.raw_env
        cent = getattr(raw, "goal_cent_pos", None)
        nodes = self._stack_thumb_sweep_env_local()
        diag: dict[str, Any] = {
            "step": run.steps,
            "thumb_passed_strict": bool(passage_flags[0]),
            "thumb_knuckle_latched": bool(knuckle_flags[0]),
            "sweep_next_idx": run.thumb_sweep_next,
            "sweep_occupied": list(run.thumb_sweep_occupied),
            "sweep_inserted": list(run.thumb_sweep_inserted),
            "sweep_bodies": list(self.thumb_sweep_names),
            "opening_normal": {"x": 1.0, "y": 0.0, "z": 0.0},
        }
        if cent is not None and env_id < int(cent.shape[0]):
            c = cent[env_id]
            diag["opening_center"] = {
                "x": float(c[0].item()),
                "y": float(c[1].item()),
                "z": float(c[2].item()),
            }
        if nodes is not None and env_id < int(nodes.shape[0]):
            stations = []
            c_x = float(cent[env_id, 0].item()) if cent is not None and env_id < int(cent.shape[0]) else 0.0
            east = getattr(raw, "goal_east_pos", None)
            west = getattr(raw, "goal_west_pos", None)
            north = getattr(raw, "goal_north_pos", None)
            south = getattr(raw, "goal_south_pos", None)
            evs = None
            if east is not None and west is not None and north is not None and south is not None:
                ry, rz = opening_radii(east, west, north, south)
                evs = ellipse_value_yz(nodes[env_id : env_id + 1], cent[env_id : env_id + 1], ry[env_id : env_id + 1], rz[env_id : env_id + 1])[0]
                diag["ellipse_radius_y"] = float(ry[env_id].item())
                diag["ellipse_radius_z"] = float(rz[env_id].item())
            for i, name in enumerate(self.thumb_sweep_names):
                p = nodes[env_id, i]
                xyz = {"x": float(p[0].item()), "y": float(p[1].item()), "z": float(p[2].item())}
                d = float(p[0].item()) - c_x
                rec = {
                    "body": name,
                    "pos": xyz,
                    "signed_distance_m": d,
                    "side": plane_side(d, self.insertion_delta_m),
                    "occupied": (
                        bool(run.thumb_sweep_occupied[i]) if i < len(run.thumb_sweep_occupied) else False
                    ),
                    "inserted": (
                        bool(run.thumb_sweep_inserted[i]) if i < len(run.thumb_sweep_inserted) else False
                    ),
                }
                if evs is not None and i < int(evs.shape[0]):
                    rec["ellipse_value"] = float(evs[i].item())
                stations.append(rec)
            diag["stations"] = stations
            if stations:
                diag["thumb_tip"] = stations[0]["pos"]
                diag["thumb_base"] = stations[-1]["pos"]
        run.last_thumb_diag = diag

    def _maybe_record_crossing(
        self,
        events: dict[str, dict[str, Any]],
        last_pre_pos: list[tuple[float, float, float] | None],
        last_pre_center: list[tuple[float, float, float] | None],
        last_pre_radius_y: list[float | None],
        last_pre_radius_z: list[float | None],
        points: torch.Tensor,
        flags: list[bool],
        cent: torch.Tensor,
        radius_y: torch.Tensor,
        radius_z: torch.Tensor,
        env_id: int,
        frame: int,
        wrist_dist: float | None,
        wrist_vec: tuple[float, float, float] | None,
        *,
        landmark_bodies: dict[str, str | None],
        landmark_kind: str,
    ) -> None:
        if env_id >= int(points.shape[0]):
            return
        c = cent[env_id]
        ry = float(radius_y[env_id].item())
        rz = float(radius_z[env_id].item())
        c_xyz = (float(c[0].item()), float(c[1].item()), float(c[2].item()))
        d_list, e_list, sides = self._finger_side_and_ellipse(points, cent, radius_y, radius_z, env_id)
        for i, name in enumerate(FINGER_ORDER):
            p = points[env_id, i]
            p_xyz = (float(p[0].item()), float(p[1].item()), float(p[2].item()))
            if sides[i] == "PRE":
                last_pre_pos[i] = p_xyz
                last_pre_center[i] = c_xyz
                last_pre_radius_y[i] = ry
                last_pre_radius_z[i] = rz
            if not flags[i] or name in events:
                continue
            pre = last_pre_pos[i]
            pre_c = last_pre_center[i] or c_xyz
            cross = None
            cross_e = None
            if pre is not None:
                d_pre = pre[0] - pre_c[0]
                d_post = d_list[i]
                denom = d_pre - d_post
                t = 0.5 if abs(denom) <= 1e-8 else max(0.0, min(1.0, d_pre / denom))
                cross = (
                    pre[0] + t * (p_xyz[0] - pre[0]),
                    pre[1] + t * (p_xyz[1] - pre[1]),
                    pre[2] + t * (p_xyz[2] - pre[2]),
                )
                cc = (
                    pre_c[0] + t * (c_xyz[0] - pre_c[0]),
                    pre_c[1] + t * (c_xyz[1] - pre_c[1]),
                    pre_c[2] + t * (c_xyz[2] - pre_c[2]),
                )
                cry = float(last_pre_radius_y[i] or ry)
                crz = float(last_pre_radius_z[i] or rz)
                cry = cry + t * (ry - cry)
                crz = crz + t * (rz - crz)
                if cry > 1e-6 and crz > 1e-6:
                    cross_e = ((cross[1] - cc[1]) / cry) ** 2 + ((cross[2] - cc[2]) / crz) ** 2
            events[name] = {
                "finger": name,
                "landmark_body": landmark_bodies.get(name),
                "landmark_kind": landmark_kind,
                "pre_pos": None if pre is None else {"x": pre[0], "y": pre[1], "z": pre[2]},
                "post_pos": {"x": p_xyz[0], "y": p_xyz[1], "z": p_xyz[2]},
                "signed_distance_post_m": d_list[i],
                "ellipse_value_post": e_list[i],
                "side_post": sides[i],
                "crossing_frame": frame,
                "crossing_time_s": frame * self.control_dt,
                "crossing_point": None if cross is None else {"x": cross[0], "y": cross[1], "z": cross[2]},
                "crossing_ellipse_value": cross_e,
                "opening_center": {"x": c_xyz[0], "y": c_xyz[1], "z": c_xyz[2]},
                "opening_normal": {"x": 1.0, "y": 0.0, "z": 0.0},
                "opening_local_axes": {"y": "env +Y", "z": "env +Z"},
                "ellipse_radius_y": ry,
                "ellipse_radius_z": rz,
                "wrist_distance_m": wrist_dist,
                "wrist_center_vector_m": (
                    None if wrist_vec is None else {"x": wrist_vec[0], "y": wrist_vec[1], "z": wrist_vec[2]}
                ),
            }

    def _record_passage_progress(
        self,
        run: _RunningEpisode,
        env_id: int,
        knuckle_flags: list[bool],
        distal_flags: list[bool],
        passage_flags: list[bool],
        geom: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor],
        frame: int,
        wrist_dist: float | None,
        wrist_vec: tuple[float, float, float] | None,
    ) -> None:
        knuckle, distal, cent, radius_y, radius_z = geom
        if env_id >= int(knuckle.shape[0]):
            return
        self._maybe_record_crossing(
            run.knuckle_passage_events,
            run.last_pre_pos,
            run.last_pre_center,
            run.last_pre_radius_y,
            run.last_pre_radius_z,
            knuckle,
            knuckle_flags,
            cent,
            radius_y,
            radius_z,
            env_id,
            frame,
            wrist_dist,
            wrist_vec,
            landmark_bodies=self.resolved_base_bodies,
            landmark_kind="finger_base_com",
        )
        if distal is not None and env_id < int(distal.shape[0]):
            self._maybe_record_crossing(
                run.distal_passage_events,
                run.last_pre_pos_distal,
                run.last_pre_center_distal,
                run.last_pre_radius_y_distal,
                run.last_pre_radius_z_distal,
                distal,
                distal_flags,
                cent,
                radius_y,
                radius_z,
                env_id,
                frame,
                wrist_dist,
                wrist_vec,
                landmark_bodies=self.resolved_distal_bodies,
                landmark_kind="finger_distal_com",
            )
        for name in FINGER_ORDER:
            if name == "thumb":
                continue
            if name in run.knuckle_passage_events and name not in run.passage_events:
                run.passage_events[name] = run.knuckle_passage_events[name]
        if "thumb" in run.knuckle_passage_events:
            run.passage_events["thumb_knuckle"] = run.knuckle_passage_events["thumb"]
        if "thumb" in run.distal_passage_events:
            run.passage_events["thumb_distal"] = run.distal_passage_events["thumb"]
        sweep = self._thumb_sweep
        if sweep is not None and env_id < int(sweep.next_idx.shape[0]):
            visits = []
            for i, name in enumerate(self.thumb_sweep_names):
                step_i = int(sweep.visit_step[env_id, i].item())
                visits.append(
                    {
                        "station": i,
                        "body": name,
                        "visit_step": None if step_i < 0 else step_i,
                        "occupied_now": bool(sweep.last_occupied[env_id, i].item()),
                        "inserted_now": bool(sweep.inserted[env_id, i].item()),
                    }
                )
            run.passage_events["thumb_sweep"] = {
                "finger": "thumb",
                "landmark_kind": "ordered_thumb_landmark_crossing",
                "stations": list(self.thumb_sweep_names),
                "n_required": int(sweep.n_required),
                "next_idx": int(sweep.next_idx[env_id].item()),
                "passed": bool(sweep.passed[env_id].item()),
                "inserted": [
                    bool(sweep.inserted[env_id, i].item()) for i in range(sweep.n_stations)
                ],
                "visits": visits,
            }
        if passage_flags[0] and "thumb" not in run.passage_events:
            sweep_ev = run.passage_events.get("thumb_sweep") or {}
            last_visit = None
            for rec in reversed(sweep_ev.get("visits") or []):
                if rec.get("visit_step") is not None:
                    last_visit = rec
                    break
            run.passage_events["thumb"] = {
                "finger": "thumb",
                "landmark_kind": "ordered_thumb_landmark_crossing",
                "landmark_body": ", ".join(self.thumb_sweep_names[:THUMB_SWEEP_REQUIRED_STATIONS]),
                "requires": (
                    "thdistal, thmiddle, thproximal each PRE→POST through the ellipse "
                    "(tip→proximal order); reverse POST→PRE clears that landmark and later ones; "
                    "re-enter in the same order; thbase diagnostic"
                ),
                "knuckle": run.knuckle_passage_events.get("thumb"),
                "distal": run.distal_passage_events.get("thumb"),
                "sweep": sweep_ev,
                "crossing_frame": frame if last_visit is None else last_visit.get("visit_step"),
                "crossing_time_s": (
                    frame * self.control_dt
                    if last_visit is None
                    else float(last_visit.get("visit_step") or 0) * self.control_dt
                ),
                "wrist_distance_m": wrist_dist,
                "wrist_center_vector_m": (
                    None if wrist_vec is None else {"x": wrist_vec[0], "y": wrist_vec[1], "z": wrist_vec[2]}
                ),
            }

    def _finger_side_and_ellipse(
        self,
        points: torch.Tensor,
        center: torch.Tensor,
        radius_y: torch.Tensor,
        radius_z: torch.Tensor,
        env_id: int,
    ) -> tuple[list[float], list[float], list[str]]:
        p = points[env_id]
        c = center[env_id]
        d = p[:, 0] - c[0]
        ev = ellipse_value_yz(
            p.unsqueeze(0),
            c.unsqueeze(0),
            radius_y[env_id : env_id + 1],
            radius_z[env_id : env_id + 1],
        )[0]
        d_list = [float(d[i].item()) for i in range(5)]
        e_list = [float(ev[i].item()) for i in range(5)]
        sides = [plane_side(di, self.insertion_delta_m) for di in d_list]
        return d_list, e_list, sides

    def _insertion_debug_row(
        self,
        env_id: int,
        steps: int,
        flags: list[bool],
        live_flags: list[bool],
        *,
        wrist_dist: float | None,
        wrist_vec: tuple[float, float, float] | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "episode": "",
            "env_id": env_id,
            "step": steps,
            "t_s": f"{steps * self.control_dt:.6g}",
            "c_x": "",
            "c_y": "",
            "c_z": "",
            "n_x": 1.0,
            "n_y": 0.0,
            "n_z": 0.0,
            "r_y": "",
            "r_z": "",
            "wrist_d": "" if wrist_dist is None else f"{wrist_dist:.8g}",
            "wrist_dx": "" if wrist_vec is None else f"{wrist_vec[0]:.8g}",
            "wrist_dy": "" if wrist_vec is None else f"{wrist_vec[1]:.8g}",
            "wrist_dz": "" if wrist_vec is None else f"{wrist_vec[2]:.8g}",
            "latched_n": sum(1 for v in flags if v),
            "live_n": sum(1 for v in live_flags if v),
        }
        for name in FINGER_ORDER:
            row[f"{name}_knuckle_x"] = ""
            row[f"{name}_knuckle_y"] = ""
            row[f"{name}_knuckle_z"] = ""
            row[f"{name}_d_knuckle"] = ""
            row[f"{name}_e_knuckle"] = ""
            row[f"{name}_side_knuckle"] = ""
            row[f"{name}_distal_x"] = ""
            row[f"{name}_distal_y"] = ""
            row[f"{name}_distal_z"] = ""
            row[f"{name}_d_distal"] = ""
            row[f"{name}_e_distal"] = ""
            row[f"{name}_side_distal"] = ""
            row[f"{name}_inserted"] = int(flags[FINGER_ORDER.index(name)])
            row[f"{name}_live_ok"] = int(live_flags[FINGER_ORDER.index(name)])
            row[f"{name}_fwd_count"] = 0
            row[f"{name}_rev_count"] = 0
        geom = self._insertion_geom()
        if geom is None:
            return row
        knuckle, distal, cent, radius_y, radius_z = geom
        if env_id >= int(knuckle.shape[0]):
            return row
        c = cent[env_id]
        row["c_x"] = f"{float(c[0].item()):.8g}"
        row["c_y"] = f"{float(c[1].item()):.8g}"
        row["c_z"] = f"{float(c[2].item()):.8g}"
        row["r_y"] = f"{float(radius_y[env_id].item()):.8g}"
        row["r_z"] = f"{float(radius_z[env_id].item()):.8g}"
        d_k, e_k, side_k = self._finger_side_and_ellipse(knuckle, cent, radius_y, radius_z, env_id)
        d_d = e_d = side_d = None
        if distal is not None and env_id < int(distal.shape[0]):
            d_d, e_d, side_d = self._finger_side_and_ellipse(distal, cent, radius_y, radius_z, env_id)
        tracker = self._tracker
        for i, name in enumerate(FINGER_ORDER):
            pk = knuckle[env_id, i]
            row[f"{name}_knuckle_x"] = f"{float(pk[0].item()):.8g}"
            row[f"{name}_knuckle_y"] = f"{float(pk[1].item()):.8g}"
            row[f"{name}_knuckle_z"] = f"{float(pk[2].item()):.8g}"
            row[f"{name}_d_knuckle"] = f"{d_k[i]:.8g}"
            row[f"{name}_e_knuckle"] = f"{e_k[i]:.8g}"
            row[f"{name}_side_knuckle"] = side_k[i]
            if distal is not None and d_d is not None and e_d is not None and side_d is not None:
                pd = distal[env_id, i]
                row[f"{name}_distal_x"] = f"{float(pd[0].item()):.8g}"
                row[f"{name}_distal_y"] = f"{float(pd[1].item()):.8g}"
                row[f"{name}_distal_z"] = f"{float(pd[2].item()):.8g}"
                row[f"{name}_d_distal"] = f"{d_d[i]:.8g}"
                row[f"{name}_e_distal"] = f"{e_d[i]:.8g}"
                row[f"{name}_side_distal"] = side_d[i]
            if tracker is not None and env_id < tracker.num_envs:
                row[f"{name}_fwd_count"] = int(tracker.fwd_count[env_id, i].item())
                row[f"{name}_rev_count"] = int(tracker.rev_count[env_id, i].item())
        return row

    def _flush_insertion_debug(self, ep: EpisodeMetrics, run: _RunningEpisode) -> None:
        if not self.record_insertion_debug or not run.debug_rows:
            return
        debug_dir = self.output_dir / "insertion_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for row in run.debug_rows:
            row["episode"] = ep.episode
        csv_path = debug_dir / f"episode_{ep.episode:04d}_env{ep.env_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=INSERTION_DEBUG_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(run.debug_rows)
        json_path = debug_dir / f"episode_{ep.episode:04d}_env{ep.env_id}.json"
        json_path.write_text(
            json.dumps(
                {
                    "episode": ep.episode,
                    "env_id": ep.env_id,
                    "steps": ep.episode_length_steps,
                    "final_inserted_fingers": ep.final_inserted_fingers,
                    "final_inserted_fingers_latched": ep.final_inserted_fingers_latched,
                    "final_inserted_fingers_live": ep.final_inserted_fingers_live,
                    "live_all_five": ep.live_all_five,
                    "snag_suspect": ep.snag_suspect,
                    "per_finger_inserted": dict(ep.inserted),
                    "per_finger_inserted_latched": dict(ep.inserted_latched),
                    "per_finger_live_ok": dict(ep.live_ok),
                    "wrist_distance_final_m": ep.wrist_distance_final_m,
                    "wrist_distance_best_after_all_five_m": ep.wrist_distance_best_after_all_five_m,
                    "task_success": ep.task_success,
                    "failure_mode": ep.failure_mode,
                    "csv": csv_path.name,
                    "last_step": run.debug_rows[-1],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[{self.log_prefix}] insertion debug -> {csv_path}")

    def _debug_insertion_env(
        self,
        env_id: int,
        steps: int,
        flags: list[bool],
        live_flags: list[bool] | None = None,
        *,
        distal_flags: list[bool] | None = None,
        passage_flags: list[bool] | None = None,
    ) -> None:
        """Print per-finger signed distance / ellipse / latch for one env."""
        geom = self._insertion_geom()
        if geom is None:
            return
        knuckle, distal, cent, radius_y, radius_z = geom
        if env_id >= int(knuckle.shape[0]):
            return

        d_k, e_k, sides = self._finger_side_and_ellipse(knuckle, cent, radius_y, radius_z, env_id)
        d_d = e_d = side_d = None
        if distal is not None and env_id < int(distal.shape[0]):
            d_d, e_d, side_d = self._finger_side_and_ellipse(distal, cent, radius_y, radius_z, env_id)
        live_flags = live_flags if live_flags is not None else [False] * 5
        prev = self._debug_prev_inserted.get(env_id, [False] * 5)
        prev_side = self._debug_prev_side.get(env_id, ["?"] * 5)
        latch_changed = [flags[i] != prev[i] for i in range(5)]
        side_changed = [sides[i] != prev_side[i] for i in range(5)]
        periodic = (steps % self.debug_insertion_interval) == 0
        if not periodic and not any(latch_changed) and not any(side_changed):
            self._debug_prev_inserted[env_id] = list(flags)
            self._debug_prev_side[env_id] = sides
            return

        n_in = sum(1 for v in flags if v)
        n_live = sum(1 for v in live_flags if v)
        n_pass = sum(1 for v in (passage_flags or flags) if v)
        t_s = steps * self.control_dt
        c = cent[env_id]
        print(
            f"[{self.log_prefix} insert] ep={len(self.episodes)} env={env_id} "
            f"t={t_s:.2f}s step={steps}  knuckle={n_in}/5 eval={n_pass}/5 live={n_live}/5  "
            f"c=({c[0].item():.3f},{c[1].item():.3f},{c[2].item():.3f}) "
            f"n=(1,0,0) r_yz=({radius_y[env_id].item():.3f},{radius_z[env_id].item():.3f})"
        )
        tracker = self._tracker
        for i, name in enumerate(FINGER_ORDER):
            hole_k = "in " if e_k[i] <= self.insertion_ellipse_threshold else "out"
            latch = "IN" if flags[i] else "--"
            live_s = "LIVE" if live_flags[i] else "----"
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
            distal_s = ""
            if d_d is not None and e_d is not None and side_d is not None:
                hole_d = "in " if e_d[i] <= self.insertion_ellipse_threshold else "out"
                d_latch = ""
                if distal_flags is not None:
                    d_latch = " IN" if distal_flags[i] else " --"
                distal_s = (
                    f"  distal{d_latch} {side_d[i]:<4} d={d_d[i]:+.4f}m ev={e_d[i]:.2f} {hole_d}"
                )
            if passage_flags is not None and name == "thumb":
                sweep_s = ""
                if self._thumb_sweep is not None and env_id < int(self._thumb_sweep.next_idx.shape[0]):
                    nxt = int(self._thumb_sweep.next_idx[env_id].item())
                    occ = "".join(
                        "1" if bool(self._thumb_sweep.last_occupied[env_id, j].item()) else "0"
                        for j in range(self._thumb_sweep.n_stations)
                    )
                    sweep_s = (
                        f"  sweep={occ} latched={nxt}/{self._thumb_sweep.n_stations}"
                        f" passed={int(bool(self._thumb_sweep.passed[env_id].item()))}"
                    )
                distal_s += sweep_s + "  eval=" + ("PASS" if passage_flags[i] else "----")
            print(
                f"  {FINGER_LABELS[name]:<6} {latch} {live_s}  knuckle {sides[i]:<4}  "
                f"d={d_k[i]:+.4f}m ev={e_k[i]:.2f} {hole_k}{distal_s}{pend}{event}"
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
        task_success_count = sum(1 for ep in eps if ep.task_success)
        all5_wrist_complete_count = sum(1 for ep in eps if ep.all5_passage_wrist_complete)
        all5_wrist_incomplete_count = sum(1 for ep in eps if ep.all5_passage_wrist_incomplete)
        all_five_passage_count = sum(1 for ep in eps if ep.all_five_passage)
        legacy_all_five_count = sum(1 for ep in eps if ep.legacy_all_five)
        wrist_ok_count = sum(1 for ep in eps if ep.wrist_ok_ever)
        cross = compute_task_success_vs_passage(eps)
        inserted_counts = {name: sum(1 for ep in eps if ep.inserted[name]) for name in FINGER_ORDER}
        ever_counts = {
            name: sum(1 for ep in eps if ep.first_insert_time_s[name] is not None) for name in FINGER_ORDER
        }
        n_passed = [int(ep.max_passed_fingers) for ep in eps]
        n_max = [int(ep.max_inserted_fingers) for ep in eps]
        n_overlap = [int(ep.final_geometric_overlap) for ep in eps]
        hist_passed = inserted_finger_histogram(n_passed)
        hist_max = inserted_finger_histogram(n_max)
        hist_overlap = inserted_finger_histogram(n_overlap)
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
                "final_inserted_fingers": ep.final_inserted_fingers,
                "max_passed_fingers": ep.max_passed_fingers,
                "num_ever_passed": ep.num_ever_passed,
                "max_inserted_fingers": ep.max_inserted_fingers,
                "insertion_outcome": ep.insertion_outcome,
                "per_finger_ever_passed": dict(ep.ever_passed or ep.inserted),
                "per_finger_ever_passed_knuckle": dict(ep.ever_passed_knuckle),
                "per_finger_ever_passed_distal": dict(ep.ever_passed_distal),
                "all_five_ever": ep.all_five_ever,
                "ever_all_inserted_knuckle": ep.ever_all_inserted_knuckle,
                "max_passed_knuckle": ep.max_passed_knuckle,
                "all_five_passed": ep.all_five_passage,
                "all_five_passage": ep.all_five_passage,
                "all5_passage_wrist_incomplete": ep.all5_passage_wrist_incomplete,
                "all5_passage_wrist_complete": ep.all5_passage_wrist_complete,
                "thumb_passed": ep.thumb_passed,
                "thdistal_frame": ep.thdistal_frame,
                "thmiddle_frame": ep.thmiddle_frame,
                "thproximal_frame": ep.thproximal_frame,
                "thbase_frame": ep.thbase_frame,
                "thumb_passage_duration_frames": ep.thumb_passage_duration_frames,
                "wrist_success": ep.wrist_success,
                "task_success": ep.task_success,
                "first_all_five_frame": ep.first_all_five_frame,
                "first_wrist_success_frame": ep.first_wrist_success_frame,
                "wrist_success_threshold_m": ep.wrist_success_threshold_m,
                "wrist_distance_final_m": ep.wrist_distance_final_m,
                "wrist_distance_best_m": ep.wrist_distance_best_m,
                "wrist_distance_best_after_all_five_m": ep.wrist_distance_best_after_all_five_m,
                "failure_mode": ep.failure_mode,
                "wrist_shortfall_primary": ep.wrist_shortfall_primary,
                "wrist_shortfall_tags": list(ep.wrist_shortfall_tags),
                "final_geometric_overlap": ep.final_geometric_overlap,
                "per_finger_geometric_overlap": dict(ep.per_finger_geometric_overlap or ep.live_ok),
                "passage_events": dict(ep.passage_events),
            }
            for ep in eps
        ]
        return {
            "schema_version": 15,
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
                "headline_success_definition": "legacy",
                "success_definition": "legacy_knuckle_all5_and_wrist",
                "task_success_definition": (
                    "legacy_all_five knuckle latch AND wrist_within_goal (motion lock / training)"
                ),
                "legacy_success_definition": "wrist_within_goal AND knuckle all_5 (motion lock / training)",
                "finger_passage_definition": (
                    "last-step latch (not ever-OR): thumb = thdistal → thmiddle → thproximal "
                    "ordered PRE→POST (reverse clears that station and later ones; earlier stay "
                    "if they did not reverse); other fingers knuckle-only; thbase diagnostic only"
                ),
                "all5_passage_wrist_complete_definition": (
                    "all_five_passage AND wrist_ok_ever (geometric completion; not task success)"
                ),
                "strict_success_definition": (
                    "deprecated alias of all5_passage_wrist_complete; not official task success"
                ),
                "outcome_breakdown_definition": (
                    "passage outcome A–D from canonical finger_passed + wrist_ok_ever"
                ),
                "metric_layers": ["task_success", "finger_passage", "physical_diagnostics"],
                "motion_lock_requires_all_fingers": bool(
                    getattr(getattr(self.raw_env, "cfg", None), "eval_success_requires_all_fingers", True)
                ),
                "motion_lock_definition": (
                    "wrist_within_goal_and_all_5_fingers_inserted"
                    if bool(getattr(getattr(self.raw_env, "cfg", None), "eval_success_requires_all_fingers", True))
                    else "wrist_within_goal"
                ),
                "insertion_definition": "final_confirmed_opening_crossing_latch",
                "insertion_latch_definition": "last_clear_pre_to_post_through_live_yz_ellipse",
                "eval_final_inserted_source": "last_passage_latch",
                "insertion_normal": "+x",
                "insertion_pre_side": "d > +delta (hand / +X of opening)",
                "insertion_post_side": "d < -delta (through / -X of opening)",
                "finger_representative_point": "finger_base_com",
                "finger_base_bodies": dict(self.resolved_base_bodies),
                "eval_thumb_passage": (
                    "ordered PRE→POST thdistal → thmiddle → thproximal "
                    "(reverse clears that station and later ones; earlier stay if they did not reverse); thbase diagnostic only"
                ),
                "eval_thumb_sweep_bodies": list(self.thumb_sweep_names),
                "eval_thumb_sweep_required_stations": THUMB_SWEEP_REQUIRED_STATIONS,
                "eval_thumb_sweep_hole_half_width_m": THUMB_SWEEP_HOLE_HALF_WIDTH_M,
                "eval_dual_landmark_fingers": list(EVAL_DUAL_LANDMARK_FINGERS),
                "training_latch_point": "finger_base_com",
                "live_containment_point": "finger_base_com_and_distal_com",
                "finger_distal_bodies": dict(self.resolved_distal_bodies),
                "live_containment_definition": (
                    "diagnostic only (final_geometric_overlap): current knuckle AND distal both "
                    "POST and inside live YZ ellipse. Not finger-passage and not task_success."
                ),
                "deformation_definition": "rms_joint_deviation_from_hand_default_joint_pos",
                "finger_joint_groups": self.resolved_joint_groups,
                "wrist_success_condition": (
                    "||goal_wrist_pos - goal_cent_pos|| < bracelet_success_threshold"
                ),
                "wrist_success_threshold_m": float(
                    getattr(getattr(self.raw_env, "cfg", None), "bracelet_success_threshold", 0.01)
                ),
                "wrist_success_latch": "instantaneous_per_step",
                "task_success_latch": "once_true_wrist_and_all_five",
                "wrist_near_band_mult": WRIST_NEAR_MULT,
                "wrist_regression_m": WRIST_REGRESSION_M,
            },
            "failure_modes": self._failure_mode_payload(eps),
            "outcome_consistency": assert_eval_outcome_consistency(eps),
            "success": {
                "num_episodes": n,
                "num_success": task_success_count,
                "num_failed": n - task_success_count,
                "success_rate": (task_success_count / n) if n else 0.0,
                "task_success": task_success_count,
                "task_success_rate": (task_success_count / n) if n else 0.0,
                "task_success_envs": list(cross.get("task_success_envs") or []),
                "wrist_best_distance": wrist_best_group_stats(eps),
                "headline": "legacy",
                "definition": "legacy_all_five knuckle AND wrist_within_goal (motion lock)",
                "legacy_all_five": legacy_all_five_count,
                "wrist_ok_ever": wrist_ok_count,
                "legacy": {
                    "all_five_knuckle": legacy_all_five_count,
                    "wrist_ok_ever": wrist_ok_count,
                    "success": task_success_count,
                    "success_rate": (task_success_count / n) if n else 0.0,
                    "strict_thumb_passed_among_legacy_success": cross["among_task_success"]["thumb_passed"],
                    "strict_all_five_among_legacy_success": cross["among_task_success"]["all_five_passage"],
                },
                "strict": {
                    "all_five_passed": all_five_passage_count,
                    "all_five_passage": all_five_passage_count,
                    "wrist_ok_ever": wrist_ok_count,
                    "all5_passage_wrist_complete": all5_wrist_complete_count,
                    "success": all5_wrist_complete_count,
                    "success_rate": (all5_wrist_complete_count / n) if n else 0.0,
                },
            },
            "finger_passage": {
                "definition": (
                    "last-step latch (not ever-OR): thumb = thdistal → thmiddle → thproximal "
                    "ordered PRE→POST (reverse clears that station and later ones; earlier stay "
                    "if they did not reverse); other fingers knuckle-only; thbase diagnostic only"
                ),
                "thumb_passed": sum(1 for ep in eps if ep.thumb_passed),
                "index_passed": sum(1 for ep in eps if bool(ep.inserted.get("index"))),
                "middle_passed": sum(1 for ep in eps if bool(ep.inserted.get("middle"))),
                "ring_passed": sum(1 for ep in eps if bool(ep.inserted.get("ring"))),
                "pinky_passed": sum(1 for ep in eps if bool(ep.inserted.get("little"))),
                "all_five_passage": all_five_passage_count,
                "all5_passage_wrist_incomplete": all5_wrist_incomplete_count,
                "all5_passage_wrist_complete": all5_wrist_complete_count,
                "mean_max_passed_fingers": (sum(n_passed) / n) if n else 0.0,
                "std_max_passed_fingers": sample_std(n_passed),
                "histogram": hist_passed,
            },
            "task_success_vs_passage": cross,
            "insertion": {
                "mean_max_passed_fingers": (sum(n_passed) / n) if n else 0.0,
                "std_max_passed_fingers": sample_std(n_passed),
                "mean_max_inserted_fingers": (sum(n_max) / n) if n else 0.0,
                "std_max_inserted_fingers": sample_std(n_max),
                "mean_final_inserted_fingers": (sum(n_passed) / n) if n else 0.0,
                "std_final_inserted_fingers": sample_std(n_passed),
                "mean_inserted_fingers": (sum(n_passed) / n) if n else 0.0,
                "std_inserted_fingers": sample_std(n_passed),
                "ever_all_inserted_count": sum(1 for ep in eps if ep.ever_all_inserted),
                "ever_all_inserted_rate": (sum(1 for ep in eps if ep.ever_all_inserted) / n) if n else 0.0,
                "ever_all_inserted_knuckle_count": sum(1 for ep in eps if ep.ever_all_inserted_knuckle),
                "ever_all_inserted_knuckle_rate": (
                    (sum(1 for ep in eps if ep.ever_all_inserted_knuckle) / n) if n else 0.0
                ),
                "mean_max_passed_knuckle": (
                    (sum(int(ep.max_passed_knuckle) for ep in eps) / n) if n else 0.0
                ),
                "final_all_inserted_count": sum(1 for ep in eps if ep.final_all_inserted),
                "final_all_inserted_rate": (sum(1 for ep in eps if ep.final_all_inserted) / n) if n else 0.0,
                "histogram_passed": hist_passed,
                "histogram_final": hist_passed,
                "histogram_max": hist_max,
                "histogram_geometric_overlap": hist_overlap,
                "mean_final_geometric_overlap": (sum(n_overlap) / n) if n else 0.0,
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
            "physical_diagnostics": {
                "incomplete_advancement_definition": (
                    "all_five_passage AND NOT wrist_ok_ever. Possible snag / incomplete "
                    "advancement candidate; not an automatic snag label."
                ),
                "incomplete_advancement_count": all5_wrist_incomplete_count,
                "thumb_passage": (
                    "thdistal → thmiddle → thproximal ordered PRE→POST "
                    "(reverse clears that station and later ones; earlier stay if they did not reverse); "
                    "thbase diagnostic only; duration = proximal_frame - distal_frame"
                ),
                "thumb_passage_duration_frames_among_all_five": [
                    ep.thumb_passage_duration_frames
                    for ep in eps
                    if ep.all_five_passage and ep.thumb_passage_duration_frames is not None
                ],
                "thumb_passage_duration_frames_among_incomplete": [
                    ep.thumb_passage_duration_frames
                    for ep in eps
                    if ep.all5_passage_wrist_incomplete and ep.thumb_passage_duration_frames is not None
                ],
            },
            "episode_debug": episode_debug,
        }

    def _failure_mode_payload(self, eps: list[EpisodeMetrics] | None = None) -> dict[str, Any]:
        eps = self.episodes if eps is None else eps
        n = len(eps)
        counts = {key: sum(1 for ep in eps if ep.failure_mode == key) for key in FAILURE_MODE_SUMMARY_ORDER}
        incomplete = [ep for ep in eps if ep.failure_mode == FAILURE_MODE_FULL_INCOMPLETE_WRIST]
        shortfall_counts: dict[str, int] = {}
        for ep in incomplete:
            key = ep.wrist_shortfall_primary or "unclassified"
            shortfall_counts[key] = shortfall_counts.get(key, 0) + 1
        cfg = getattr(self.raw_env, "cfg", None)
        return {
            "wrist_metric": "||goal_wrist_pos - goal_cent_pos||",
            "wrist_success_threshold_m": float(getattr(cfg, "bracelet_success_threshold", 0.01)),
            "wrist_success": "instantaneous dist < threshold; wrist_success flag is once-true while all 5 inserted",
            "task_success": "official: legacy motion lock, first step with wrist_ok AND knuckle all_5",
            "all5_passage_wrist_complete": (
                "all_five_passage AND wrist_ok_ever (geometric completion; not task success)"
            ),
            "strict_success": "deprecated alias of all5_passage_wrist_complete; not official task success",
            "outcome_uses": "canonical finger passage + wrist_ok_ever; not official task success",
                "eval_finger_passage": (
                "last-step latch (not ever-OR): thumb = thdistal → thmiddle → thproximal "
                "ordered PRE→POST (reverse clears that station and later ones; earlier stay "
                "if they did not reverse); other fingers knuckle-only; thbase diagnostic only"
            ),
            "near_band": f"threshold <= best_after_all_five < {WRIST_NEAR_MULT} * threshold",
            "regression_threshold_m": WRIST_REGRESSION_M,
            "summary": {
                key: {
                    "count": counts[key],
                    "rate": (counts[key] / n) if n else 0.0,
                    "label": FAILURE_MODE_LABELS[key],
                }
                for key in FAILURE_MODE_SUMMARY_ORDER
            },
            "full_insertion_incomplete_wrist": {
                "count": len(incomplete),
                "shortfall_primary_counts": shortfall_counts,
                "episodes": [ep.to_failure_mode_record() for ep in incomplete],
            },
            "outcome_consistency": assert_eval_outcome_consistency(eps),
            "episodes": [ep.to_failure_mode_record() for ep in eps],
            "additional_signals_not_logged": [
                "per-step wrist-center time series (only min/final/first-frame scalars are stored)",
                "separate opening-center vs wrist trajectories (only the difference vector at snapshots)",
            ],
        }

    def _write_failure_modes(self, payload: dict[str, Any] | None = None) -> None:
        payload = payload if payload is not None else self._failure_mode_payload()
        self.failure_modes_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        episodes = payload.get("episodes") or []
        fieldnames = [
            "episode",
            "env_id",
            "final_inserted_fingers",
            "final_inserted_fingers_latched",
            "max_inserted_fingers",
            "max_passed_fingers",
            "num_ever_passed",
            "all_five_ever",
            "all_five_retained",
            "all_five_retained_latched",
            "wrist_success",
            "task_success",
            "all_five_passage",
            "all5_passage_wrist_complete",
            "thumb_passed",
            "first_all_five_frame",
            "first_all_five_frame_latched",
            "first_wrist_success_frame",
            "wrist_success_threshold_m",
            "wrist_distance_final_m",
            "wrist_distance_best_m",
            "wrist_distance_best_after_all_five_m",
            "wrist_ok_ever",
            "wrist_ok_after_all_five",
            "wrist_ok_at_end",
            "wrist_dx_final_m",
            "wrist_dy_final_m",
            "wrist_dz_final_m",
            "wrist_dx_best_after_all_five_m",
            "wrist_dy_best_after_all_five_m",
            "wrist_dz_best_after_all_five_m",
            "failure_mode",
            "wrist_shortfall_primary",
            "wrist_shortfall_tags",
            "final_inserted_fingers_live",
            "live_all_five",
            "snag_suspect",
        ]
        with self.failure_modes_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rec in episodes:
                fx = rec.get("wrist_center_vector_final_m") or {}
                bx = rec.get("wrist_center_vector_best_after_all_five_m") or {}
                writer.writerow(
                    {
                        **{k: rec.get(k) for k in fieldnames},
                        "all_five_ever": int(bool(rec.get("all_five_ever"))),
                        "all_five_retained": int(bool(rec.get("all_five_retained"))),
                        "all_five_retained_latched": int(bool(rec.get("all_five_retained_latched"))),
                        "wrist_success": int(bool(rec.get("wrist_success"))),
                        "task_success": int(bool(rec.get("task_success"))),
                        "wrist_ok_ever": int(bool(rec.get("wrist_ok_ever"))),
                        "wrist_ok_after_all_five": int(bool(rec.get("wrist_ok_after_all_five"))),
                        "wrist_ok_at_end": int(bool(rec.get("wrist_ok_at_end"))),
                        "live_all_five": int(bool(rec.get("live_all_five"))),
                        "snag_suspect": int(bool(rec.get("snag_suspect"))),
                        "wrist_dx_final_m": None if not fx else fx.get("x"),
                        "wrist_dy_final_m": None if not fx else fx.get("y"),
                        "wrist_dz_final_m": None if not fx else fx.get("z"),
                        "wrist_dx_best_after_all_five_m": None if not bx else bx.get("x"),
                        "wrist_dy_best_after_all_five_m": None if not bx else bx.get("y"),
                        "wrist_dz_best_after_all_five_m": None if not bx else bx.get("z"),
                        "wrist_shortfall_tags": ",".join(rec.get("wrist_shortfall_tags") or []),
                    }
                )

    def _write_wrist_incomplete_diagnostics(self, summary: dict[str, Any]) -> Path | None:
        incomplete = [
            ep for ep in self.episodes if ep.failure_mode == FAILURE_MODE_FULL_INCOMPLETE_WRIST
        ]
        path = self.output_dir / "wrist_incomplete_diagnostics.json"
        payload = {
            "note": (
                "all_five_passage AND NOT wrist_ok_ever. Passage-outcome C / possible "
                "incomplete-advancement candidate — not official task success and not an "
                "automatic snag label. Thumb passage is ordered PRE→POST "
                "thdistal → thmiddle → thproximal (reverse clears that station and later ones; "
                "earlier stay if they did not reverse); "
                "thbase is diagnostic only."
            ),
            "thumb_landmark": ", ".join(self.thumb_sweep_names) or self.resolved_base_bodies.get("thumb"),
            "thumb_knuckle_landmark": self.resolved_base_bodies.get("thumb"),
            "thumb_distal_landmark": self.resolved_distal_bodies.get("thumb"),
            "finger_base_bodies": dict(self.resolved_base_bodies),
            "opening": {
                "center": "(goal_north + goal_south) / 2",
                "normal": "env +X (not live PCA)",
                "ellipse": "env Y/Z from live N/S/E/W nodes",
                "nsew_mode": str(
                    getattr(getattr(self.raw_env, "cfg", None), "deformable_bracelet_nsew_geom_mode", "")
                ),
            },
            "count": len(incomplete),
            "episodes": [
                {
                    **ep.to_failure_mode_record(),
                    "hand_rms_deg": ep.hand_rms,
                    "finger_rms_deg": dict(ep.finger_rms),
                }
                for ep in incomplete
            ],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def _write_legacy_success_thumb_diagnostics(self) -> Path | None:
        legacy = [ep for ep in self.episodes if ep.legacy_success]
        failed = [ep for ep in legacy if not bool(ep.thumb_passed)]
        path = self.output_dir / "legacy_success_thumb_diagnostics.json"
        payload = {
            "note": (
                "Official task-success episodes (knuckle 5/5 + wrist) where geometric "
                "thumb passage (thdistal → thmiddle → thproximal ordered PRE→POST) "
                "did not pass. Diagnostic only; not a second success definition."
            ),
            "legacy_success_count": len(legacy),
            "strict_thumb_passed": len(legacy) - len(failed),
            "strict_thumb_failed": len(failed),
            "strict_all_five_among_legacy_success": sum(1 for ep in legacy if ep.all_five_passed),
            "episodes": [
                {
                    "episode": ep.episode,
                    "env_id": ep.env_id,
                    "legacy_success": ep.legacy_success,
                    "legacy_all_five": ep.legacy_all_five,
                    "strict_success": ep.strict_success,
                    "strict_thumb_passed": bool(ep.thumb_passed),
                    "passed_count": ep.max_passed_fingers,
                    "wrist_ok_ever": ep.wrist_ok_ever,
                    "wrist_distance_final_m": ep.wrist_distance_final_m,
                    "wrist_distance_best_m": ep.wrist_distance_best_m,
                    "thumb_knuckle_latched": bool((ep.inserted_latched or {}).get("thumb")),
                    "thumb_sweep": (ep.passage_events or {}).get("thumb_sweep"),
                    "thumb_knuckle_crossing": (ep.passage_events or {}).get("thumb_knuckle"),
                    "thumb_distal_crossing": (ep.passage_events or {}).get("thumb_distal"),
                    "last_thumb_diag": dict(ep.last_thumb_diag),
                }
                for ep in failed
            ],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def _write_histogram(self, summary: dict[str, Any]) -> None:
        insertion = summary.get("insertion") or {}
        payload = {
            "checkpoint": summary.get("checkpoint"),
            "executed_at": summary.get("executed_at"),
            "num_episodes": (summary.get("success") or {}).get("num_episodes"),
            "histogram_passed": insertion.get("histogram_passed") or insertion.get("histogram_max"),
            "histogram_geometric_overlap": insertion.get("histogram_geometric_overlap"),
            "eval_final_inserted_source": "last_passage_latch",
        }
        self.histogram_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _write_summary(self, *, partial: bool) -> None:
        summary = self.build_summary()
        path = self.partial_path if partial else self.summary_path
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not partial:
            self._write_histogram(summary)
            self._write_failure_modes(summary.get("failure_modes"))

    def finalize(self) -> dict[str, Any]:
        for env_id in self.eval_env_ids:
            if self.is_complete():
                break
            if self._running[env_id].steps > 0:
                self._finalize_env(env_id, terminated=False, truncated=False)
        summary = self.build_summary()
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.partial_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self._write_histogram(summary)
        self._write_failure_modes(summary.get("failure_modes"))
        diag_path = self._write_wrist_incomplete_diagnostics(summary)
        thumb_diag = self._write_legacy_success_thumb_diagnostics()
        self._harvest_leftover_videos()
        print_evaluation_summary(summary, output_dir=self.output_dir)
        print(f"[{self.log_prefix}] episode metrics -> {self.csv_path}")
        print(f"[{self.log_prefix}] evaluation summary -> {self.summary_path}")
        print(f"[{self.log_prefix}] passed-finger histogram -> {self.histogram_path}")
        print(f"[{self.log_prefix}] failure modes -> {self.failure_modes_json_path}")
        print(f"[{self.log_prefix}] failure modes csv -> {self.failure_modes_csv_path}")
        if diag_path is not None:
            print(f"[{self.log_prefix}] 5/5 wrist-incomplete diagnostics -> {diag_path}")
        if thumb_diag is not None:
            print(f"[{self.log_prefix}] legacy-success thumb diagnostics -> {thumb_diag}")
        if self.record_insertion_debug:
            print(f"[{self.log_prefix}] insertion debug -> {self.output_dir / 'insertion_debug'}")
        if self.kept_failure_videos:
            print(f"[{self.log_prefix}] kept {len(self.kept_failure_videos)} failure video(s):")
            for path in self.kept_failure_videos:
                print(f"  {path}")
        return summary


def _fmt_m(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_frame(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(value))


def _fmt_cm_pm(block: dict[str, Any] | None) -> str:
    if not block or block.get("mean_cm") is None:
        return "-"
    n = int(block.get("n") or 0)
    return f"{float(block['mean_cm']):.1f} ± {float(block.get('std_cm') or 0.0):.1f} cm  (n={n})"


def _print_wrist_best_distance(groups: dict[str, Any]) -> None:
    if not groups:
        return
    print("")
    print("Best wrist distance  (min ||wrist − opening center|| during episode)")
    print(f"  Successful dressing     : {_fmt_cm_pm(groups.get('successful_dressing'))}")
    print(f"  Task failure            : {_fmt_cm_pm(groups.get('task_failure'))}")
    print(f"  Incomplete placement    : {_fmt_cm_pm(groups.get('incomplete_placement'))}")
    print("    (all-five last-step passage, wrist never within threshold)")
    for key, title in (
        ("successful_dressing", "Successful dressing envs"),
        ("task_failure", "Task-failure envs"),
        ("incomplete_placement", "Incomplete-placement envs"),
    ):
        rows = (groups.get(key) or {}).get("episodes") or []
        if not rows:
            continue
        print(f"  {title}:")
        for rec in rows:
            print(
                f"    env={rec.get('env_id')} (ep {rec.get('episode')})  "
                f"{float(rec.get('wrist_distance_best_cm') or 0.0):.1f} cm"
            )


def _fmt_env_refs(items: list[Any] | None) -> str:
    if not items:
        return "(none)"
    parts = []
    for it in items:
        if isinstance(it, dict):
            parts.append(f"env={it.get('env_id')} (ep {it.get('episode')})")
        else:
            parts.append(str(it))
    return ", ".join(parts)


def _fmt_finger_yn(flags: dict[str, bool] | None) -> str:
    flags = flags or {}
    abbrev = {"thumb": "T", "index": "I", "middle": "M", "ring": "R", "little": "P"}
    return " ".join(
        f"{abbrev[name]}={'YES' if flags.get(name) else 'NO'}" for name in FINGER_ORDER
    )


def _print_task_yes_not_all_five_details(details: list[Any]) -> None:
    if not details:
        return
    print("")
    print(f"Task-success / not last-step all-five ({len(details)}):")
    print(
        "  Task locked, but last-step passage was not 5/5 "
        "(often a finger entered then reversed)."
    )
    for rec in details:
        if not isinstance(rec, dict):
            continue
        exited = rec.get("exited_fingers") or []
        exited_s = ",".join(exited) if exited else "(none)"
        print("")
        print(f"  env={rec.get('env_id')} (ep {rec.get('episode')})")
        print(
            f"    last passage : {int(rec.get('last_passed') or 0)}/5  "
            f"{_fmt_finger_yn(rec.get('last_fingers'))}"
        )
        print(
            f"    ever passage : {int(rec.get('ever_passed') or 0)}/5  "
            f"{_fmt_finger_yn(rec.get('ever_fingers'))}"
        )
        print(
            f"    max passed   : {int(rec.get('max_passed_fingers') or 0)}/5  "
            f"entered-then-left: {exited_s}  missing-at-end: "
            f"{','.join(rec.get('missing_last') or []) or '(none)'}"
        )
        st_parts = []
        for st in rec.get("thumb_stations") or []:
            vis = st.get("visit_step")
            vis_s = "-" if vis is None else str(int(vis))
            st_parts.append(
                f"{st.get('name')}={st.get('status')}@f{vis_s}"
            )
        if st_parts:
            print(f"    thumb order  : {'  '.join(st_parts)}")
        print(
            f"    thumb frames : distal@{_fmt_frame(rec.get('thdistal_frame'))}  "
            f"middle@{_fmt_frame(rec.get('thmiddle_frame'))}  "
            f"proximal@{_fmt_frame(rec.get('thproximal_frame'))}"
        )
        lock_t = rec.get("motion_lock_time_s")
        lock_s = f"{float(lock_t):.2f}s" if lock_t is not None else "-"
        print(
            f"    wrist best={_fmt_m(rec.get('wrist_distance_best_m'))}  "
            f"final={_fmt_m(rec.get('wrist_distance_final_m'))}  "
            f"at_lock={_fmt_m(rec.get('wrist_distance_at_success'))}  "
            f"lock@{lock_s}"
        )
        print(
            f"    knuckle-ever-5={bool(rec.get('legacy_all_five'))}  "
            f"note={rec.get('note')}"
        )


def print_evaluation_summary(summary: dict[str, Any], *, output_dir: Path | None = None) -> None:
    cfg = summary.get("config") or {}
    success = summary.get("success") or {}
    insertion = summary.get("insertion") or {}
    deform = summary.get("deformation") or {}
    passage = summary.get("finger_passage") or {}
    cross = summary.get("task_success_vs_passage") or {}
    physical = summary.get("physical_diagnostics") or {}
    fingers_ins = insertion.get("fingers") or {}
    fingers_def = deform.get("fingers") or {}
    hand = deform.get("hand") or {}
    n = int(success.get("num_episodes") or 0)
    n_ok = int(success.get("task_success") or success.get("num_success") or 0)
    rate = float(success.get("task_success_rate") or success.get("success_rate") or 0.0) * 100.0
    freq = summary.get("control_frequency_hz")
    delta = cfg.get("insertion_delta_m")
    confirm = cfg.get("insertion_confirm_frames")
    fm = summary.get("failure_modes") or {}
    fm_summary = fm.get("summary") or {}
    checks = summary.get("outcome_consistency") or fm.get("outcome_consistency") or {}
    legacy = success.get("legacy") or {}
    among_ok = cross.get("among_task_success") or {}
    among_fail = cross.get("among_task_failure") or {}

    print("")
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    if output_dir is not None:
        print(f"Output : {output_dir}")
    print("")
    print(f"Total episodes : {n}")
    if freq and delta is not None:
        print(
            f"Control: {float(freq):.1f} Hz   crossing: |d| > {float(delta):.4g} m "
            f"confirm {int(confirm)} frames   ellipse <= {float(cfg.get('insertion_ellipse_threshold') or 1.0):.3g}"
        )
    print("")
    print("-" * 60)
    print("Task Success")
    print("-" * 60)
    print("Task success = legacy knuckle all-five + wrist")
    print(
        f"Legacy all-five (knuckle) : "
        f"{int(success.get('legacy_all_five') or legacy.get('all_five_knuckle') or 0)} / {n}"
    )
    print(
        f"Wrist goal reached        : "
        f"{int(success.get('wrist_ok_ever') or legacy.get('wrist_ok_ever') or 0)} / {n}"
    )
    print(f"Task success              : {n_ok} / {n}   {rate:.1f} %")
    print(
        f"  envs                    : "
        f"{_fmt_env_refs(success.get('task_success_envs') or cross.get('task_success_envs'))}"
    )
    _print_wrist_best_distance(success.get("wrist_best_distance") or {})
    print("")
    print("-" * 60)
    print("Finger Passage")
    print("-" * 60)
    print(
        "Last-step latch (not ever-OR). Thumb: thdistal → thmiddle → thproximal "
        "ordered PRE→POST (reverse clears that station and later ones; earlier stay "
        "if they did not reverse; thbase diagnostic only)"
    )
    print("Other fingers: knuckle crossing")
    for name in FINGER_ORDER:
        key = "pinky_passed" if name == "little" else f"{name}_passed"
        count = passage.get(key)
        if count is None:
            block = fingers_ins.get(name) or {}
            count = int(block.get("final_count") if block.get("final_count") is not None else block.get("ever_count") or 0)
        print(
            f"{FINGER_LABELS.get(name, name.capitalize())} passed              : "
            f"{int(count)} / {n}"
        )
    mean_p = float(passage.get("mean_max_passed_fingers") or insertion.get("mean_final_inserted_fingers") or 0.0)
    std_p = float(passage.get("std_max_passed_fingers") or insertion.get("std_final_inserted_fingers") or 0.0)
    print(f"Mean passed fingers       : {mean_p:.2f} ± {std_p:.2f} / 5")
    all5_raw = passage.get("all_five_passage")
    if all5_raw is None:
        all5_raw = (success.get("strict") or {}).get("all_five_passage")
    all5 = int(all5_raw or 0)
    print(f"All-five passage          : {all5} / {n}")
    hist = passage.get("histogram") or insertion.get("histogram_passed") or {}
    counts = hist.get("counts")
    if isinstance(counts, list) and counts:
        labels = hist.get("labels") or [f"{k}/5" for k in range(len(counts))]
        print("")
        print("Passed-finger histogram:")
        print(f"               {''.join(f'{lab:>6}' for lab in labels)}")
        print(f"episodes       {''.join(f'{int(c):6d}' for c in counts)}")
    print("")
    print("-" * 60)
    print("Passage Outcome Breakdown")
    print("-" * 60)
    print("Not official task success. D is geometric completion only.")
    for key in FAILURE_MODE_SUMMARY_ORDER:
        block = fm_summary.get(key) or {}
        print(
            f"{FAILURE_MODE_LABELS[key]:<38} : "
            f"{int(block.get('count') or 0):4d} / {n}"
        )
    print("")
    print("Passage checks:")
    d_complete = checks.get("D_all5_passage_wrist_complete", checks.get("D_success", "-"))
    print(
        f"  histogram.sum == total: {checks.get('histogram', '-')}  "
        f"ok={checks.get('ok')}"
    )
    print(
        f"  A+B+C+D = {n}: "
        f"{checks.get('A_no_passage', '-')}+{checks.get('B_partial', '-')}+"
        f"{checks.get('C_all_five_wrist_incomplete', '-')}+{d_complete}"
    )
    print(
        f"  C+D == all-five passage: "
        f"{checks.get('C_all_five_wrist_incomplete', '-')}+{d_complete} "
        f"= {checks.get('all_five_passage', checks.get('ever_all_five', '-'))}"
    )
    print(f"  Task success (separate): {checks.get('task_success', n_ok)} / {n}")
    print("")
    print("-" * 60)
    print("Task Success vs Strict Passage")
    print("-" * 60)
    print("                           strict all-five")
    print("                         no             yes")
    print(
        f"task success = no      {int(cross.get('task_no_all5_no') or 0):6d}         "
        f"{int(cross.get('task_no_all5_yes') or 0):6d}"
    )
    print(
        f"task success = yes     {int(cross.get('task_yes_all5_no') or 0):6d}         "
        f"{int(cross.get('task_yes_all5_yes') or 0):6d}"
    )
    print(f"  yes / not all-five envs : {_fmt_env_refs(cross.get('task_yes_all5_no_envs'))}")
    print(f"  yes / all-five envs     : {_fmt_env_refs(cross.get('task_yes_all5_yes_envs'))}")
    _print_task_yes_not_all_five_details(cross.get("task_yes_all5_no_details") or [])
    n_ts = int(among_ok.get("n") or 0)
    n_tf = int(among_fail.get("n") or 0)
    print("")
    print("Among task-success episodes:")
    print(f"    strict thumb passed : {int(among_ok.get('thumb_passed') or 0)} / {n_ts}")
    print(f"    strict all-five     : {int(among_ok.get('all_five_passage') or 0)} / {n_ts}")
    print("Among task-failure episodes:")
    print(f"    strict thumb passed : {int(among_fail.get('thumb_passed') or 0)} / {n_tf}")
    print(f"    strict all-five     : {int(among_fail.get('all_five_passage') or 0)} / {n_tf}")
    print("")
    print("-" * 60)
    print("Physical Diagnostics")
    print("-" * 60)
    print("These are not task success and not an automatic snag label.")
    incomplete_eps = (fm.get("full_insertion_incomplete_wrist") or {}).get("episodes") or []
    print(
        f"Possible incomplete-advancement / snag candidates: "
        f"{int(physical.get('incomplete_advancement_count') or len(incomplete_eps))} / {n}"
    )
    print("  = all-five passage, wrist incomplete")
    durs_all5 = physical.get("thumb_passage_duration_frames_among_all_five") or []
    durs_c = physical.get("thumb_passage_duration_frames_among_incomplete") or []
    if durs_all5:
        print(
            f"Thumb passage duration (proximal − distal): "
            f"all-five mean {sum(durs_all5) / len(durs_all5):.1f} frames"
            + (
                f"; incomplete-advancement mean {sum(durs_c) / len(durs_c):.1f} frames"
                if durs_c
                else ""
            )
        )
    if incomplete_eps:
        thr = float(fm.get("wrist_success_threshold_m") or 0.01)
        print(f"wrist metric {fm.get('wrist_metric')}  threshold={thr:.4g} m")
        for rec in incomplete_eps:
            print("")
            print(f"Episode {int(rec.get('episode'))}  env={int(rec.get('env_id'))}")
            print(
                f"  passed={int(rec.get('passed_count') or rec.get('max_passed_fingers') or 0)}/5  "
                f"wrist_final={_fmt_m(rec.get('wrist_distance_final_m'))}  "
                f"wrist_best={_fmt_m(rec.get('wrist_distance_best_m'))}  "
                f"5@={_fmt_frame(rec.get('first_all_five_frame'))}  "
                f"w@={_fmt_frame(rec.get('first_wrist_success_frame'))}"
            )
            print(
                f"  thumb  distal@{_fmt_frame(rec.get('thdistal_frame'))}  "
                f"middle@{_fmt_frame(rec.get('thmiddle_frame'))}  "
                f"proximal@{_fmt_frame(rec.get('thproximal_frame'))}  "
                f"base@{_fmt_frame(rec.get('thbase_frame'))} (diag)  "
                f"dt={_fmt_frame(rec.get('thumb_passage_duration_frames'))}"
            )
            print(
                f"  hand_rms={float((rec.get('last_thumb_diag') or {}).get('hand_rms') or rec.get('hand_rms') or 0.0):.3f}  "
                f"shortfall={rec.get('wrist_shortfall_primary') or 'unclassified'}"
            )
    print("")
    print("Finger Joint Deviation  (from hand.data.default_joint_pos, deg)")
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
