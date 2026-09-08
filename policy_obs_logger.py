"""Log policy-input ``gt`` / ``prop`` timeseries for deploy comparison.

Records the tensors actually passed to ``encoder(states)`` — not a recompute
after the subsequent physics step. Optional per-field arrays are sliced from
those same vectors using the schema built from the live env.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
_EE_SOURCE_FRAME = "/World/envs/env_*/Robot/world (FrameTransformer source)"
_QUAT_WXYZ = "wxyz"


def _unwrap(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        nxt = getattr(current, "env", None) or getattr(current, "unwrapped", None)
        if nxt is None or nxt is current:
            break
        current = nxt
    return getattr(current, "unwrapped", current)


def _to_1d(value: Any, env_id: int) -> np.ndarray:
    if hasattr(value, "detach"):
        row = value[int(env_id)].detach().cpu().numpy()
    else:
        row = np.asarray(value)
    return np.asarray(row, dtype=np.float32).reshape(-1)


def _policy_obs_dict(states: Any) -> dict[str, Any]:
    if isinstance(states, dict) and "policy" in states:
        return states["policy"]
    if not isinstance(states, dict):
        raise TypeError(f"expected observation dict, got {type(states)}")
    return states


def _field(
    name: str,
    start: int,
    width: int,
    *,
    unit: str,
    frame: str,
    meaning: str,
    quaternion_convention: str | None = None,
    joint_names: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name,
        "slice": [start, start + width],
        "shape": [width],
        "dtype": "float32",
        "unit": unit,
        "frame": frame,
        "meaning": meaning,
        "quaternion_convention": quaternion_convention,
        "joint_names": joint_names,
    }
    if extra:
        out.update(extra)
    return out


def _actuated_joint_names(raw: Any) -> list[str]:
    ids = list(getattr(raw, "actuated_dof_indices", []) or [])
    names = list(getattr(getattr(raw, "robot", None), "joint_names", []) or [])
    return [names[int(i)] for i in ids] if names else list(getattr(raw.cfg, "actuated_joint_names", []) or [])


def _prop_schema(raw: Any) -> dict[str, Any]:
    joints = _actuated_joint_names(raw)
    n = len(joints)
    vel_max = float(getattr(raw.cfg, "vel_max_magnitude", 3.0))
    i = 0
    fields = [
        _field(
            "normalised_joint_pos",
            i, n,
            unit="normalized",
            frame="joint",
            meaning="unscale(q_act, soft_lower, soft_upper) = (2*q - upper - lower) / (upper - lower)",
            joint_names=joints,
            extra={"normalization": "soft_joint_limits_to_minus1_plus1", "source_buffer": "normalised_joint_pos"},
        ),
    ]
    i += n
    fields.append(
        _field(
            "normalised_joint_vel",
            i, n,
            unit="normalized",
            frame="joint",
            meaning=f"q_vel / vel_max_magnitude ({vel_max:g})",
            joint_names=joints,
            extra={"normalization": "divide_by_vel_max_magnitude", "vel_max_magnitude": vel_max, "source_buffer": "normalised_joint_vel"},
        )
    )
    i += n
    fields.append(
        _field(
            "joint_pos_error",
            i, n,
            unit="rad",
            frame="joint",
            meaning="q_cmd - q_act (not normalized). q_cmd is EMA-filtered residual target.",
            joint_names=joints,
            extra={"normalization": "none", "source_buffer": "joint_pos_error"},
        )
    )
    i += n
    fields.append(
        _field(
            "right_upper_ee_pos",
            i, 3,
            unit="m",
            frame=_EE_SOURCE_FRAME,
            meaning="right_hand_first_finger_link_2 position from FrameTransformer.target_pos_source",
            extra={"source_buffer": "right_upper_ee_pos"},
        )
    )
    i += 3
    fields.append(
        _field(
            "right_upper_ee_rot",
            i, 4,
            unit="quaternion",
            frame=_EE_SOURCE_FRAME,
            meaning="right_hand_first_finger_link_2 orientation from FrameTransformer.target_quat_source",
            quaternion_convention=_QUAT_WXYZ,
            extra={"source_buffer": "right_upper_ee_rot"},
        )
    )
    i += 4
    fields.append(
        _field(
            "left_upper_ee_pos",
            i, 3,
            unit="m",
            frame=_EE_SOURCE_FRAME,
            meaning="left_hand_first_finger_link_2 position from FrameTransformer.target_pos_source",
            extra={"source_buffer": "left_upper_ee_pos"},
        )
    )
    i += 3
    fields.append(
        _field(
            "left_upper_ee_rot",
            i, 4,
            unit="quaternion",
            frame=_EE_SOURCE_FRAME,
            meaning="left_hand_first_finger_link_2 orientation from FrameTransformer.target_quat_source",
            quaternion_convention=_QUAT_WXYZ,
            extra={"source_buffer": "left_upper_ee_rot"},
        )
    )
    i += 4
    fields.append(
        _field(
            "actions",
            i, n,
            unit="normalized",
            frame="action",
            meaning="EMA-filtered policy action (self.actions), not a_raw. a_t = tau*a_raw + (1-tau)*a_{t-1}",
            joint_names=joints,
            extra={
                "normalization": "policy_tanh_action_in_minus1_plus1",
                "source_buffer": "actions",
                "ema_tau": float(getattr(raw.cfg, "act_moving_average", 0.0)),
            },
        )
    )
    i += n
    return {
        "total_dim": i,
        "concat_order": [f["name"] for f in fields],
        "source_method": "AIRECEnv._get_proprioception",
        "policy_input": "this concatenated vector is states['policy']['prop'] with no further scaling",
        "fields": fields,
    }


def _gt_schema_deformable() -> dict[str, Any]:
    fields = [
        _field("ee_distance", 0, 3, unit="m", frame=_EE_SOURCE_FRAME,
               meaning="right_upper_ee_pos - left_upper_ee_pos (signed xyz; abs() only in free_space)",
               extra={"source_buffer": "ee_distance"}),
        _field("ee_euclidean_distance", 3, 1, unit="m", frame=_EE_SOURCE_FRAME,
               meaning="||ee_distance||", extra={"source_buffer": "ee_euclidean_distance"}),
        _field("right_ee_thumb_distance", 4, 3, unit="m", frame="env-local (same as EE / goal buffers)",
               meaning="right_upper_ee_pos - thumb_target", extra={"source_buffer": "right_ee_thumb_distance"}),
        _field("right_ee_thumb_euclidean_distance", 7, 1, unit="m", frame="env-local (same as EE / goal buffers)",
               meaning="||right_ee_thumb_distance||", extra={"source_buffer": "right_ee_thumb_euclidean_distance"}),
        _field("left_ee_pinky_distance", 8, 3, unit="m", frame="env-local (same as EE / goal buffers)",
               meaning="left_upper_ee_pos - pinky_target", extra={"source_buffer": "left_ee_pinky_distance"}),
        _field("left_ee_pinky_euclidean_distance", 11, 1, unit="m", frame="env-local (same as EE / goal buffers)",
               meaning="||left_ee_pinky_distance||", extra={"source_buffer": "left_ee_pinky_euclidean_distance"}),
        _field("wrist_center_distance", 12, 3, unit="m", frame="env-local",
               meaning="goal_wrist_pos - goal_cent_pos", extra={"source_buffer": "wrist_center_distance"}),
        _field("wrist_center_euclidean_distance", 15, 1, unit="m", frame="env-local",
               meaning="||wrist_center_distance||", extra={"source_buffer": "wrist_center_euclidean_distance"}),
        _field("per_finger_soft_inside", 16, 5, unit="unitless", frame="n/a",
               meaning="sigmoid(min(z-south, north-z) / insertion_gate_temperature) per finger",
               extra={"finger_names": list(_FINGER_ORDER), "source_buffer": "per_finger_soft_inside"}),
        _field("com_pos_b", 21, 3, unit="m", frame="robot chassis body (quat_apply_inverse of world CoM-to-base)",
               meaning="mass-weighted body CoM expressed in the chassis / base body frame",
               extra={"source_buffer": "com_pos_b"}),
    ]
    return {
        "total_dim": 24,
        "concat_order": [f["name"] for f in fields],
        "source_method": "ReachDeformableBraceletEnv._get_gt",
        "policy_input": "this concatenated vector is states['policy']['gt'] with no further scaling",
        "fields": fields,
    }


def _gt_schema_rigid_bracelet() -> dict[str, Any]:
    schema = _gt_schema_deformable()
    schema["source_method"] = "ReachBraceletEnv._get_gt"
    rename = {
        "right_ee_thumb_distance": "right_upper_ee_thumb_distance",
        "right_ee_thumb_euclidean_distance": "right_upper_ee_thumb_euclidean_distance",
        "left_ee_pinky_distance": "left_upper_ee_pinky_distance",
        "left_ee_pinky_euclidean_distance": "left_upper_ee_pinky_euclidean_distance",
    }
    for item in schema["fields"]:
        if item["name"] in rename:
            item["name"] = rename[item["name"]]
            item["source_buffer"] = rename.get(item.get("source_buffer", ""), item["name"])
    schema["concat_order"] = [f["name"] for f in schema["fields"]]
    return schema


def _gt_schema_parent() -> dict[str, Any]:
    fields = [
        _field("ee_distance", 0, 3, unit="m", frame=_EE_SOURCE_FRAME,
               meaning="right_upper_ee_pos - left_upper_ee_pos", extra={"source_buffer": "ee_distance"}),
        _field("ee_euclidean_distance", 3, 1, unit="m", frame=_EE_SOURCE_FRAME,
               meaning="||ee_distance||", extra={"source_buffer": "ee_euclidean_distance"}),
        _field("com_pos_b", 4, 3, unit="m", frame="robot chassis body",
               meaning="mass-weighted CoM in chassis frame", extra={"source_buffer": "com_pos_b"}),
    ]
    return {
        "total_dim": 7,
        "concat_order": [f["name"] for f in fields],
        "source_method": "AIRECEnv._get_gt",
        "policy_input": "this concatenated vector is states['policy']['gt'] with no further scaling",
        "fields": fields,
    }


def build_observation_schema(env: Any) -> dict[str, Any]:
    raw = _unwrap(env)
    cls = type(raw).__name__
    if cls == "ReachDeformableBraceletEnv":
        gt = _gt_schema_deformable()
    elif cls == "ReachBraceletEnv":
        gt = _gt_schema_rigid_bracelet()
    else:
        gt = _gt_schema_parent()
        gt["note"] = f"generic parent layout; env class is {cls}"

    return {
        "env_class": cls,
        "obs_list": list(getattr(raw.cfg, "obs_list", ["prop", "gt"])),
        "encoder": {
            "method": "early",
            "state_preprocessor": None,
            "raw_state_concat_order": "sorted keys among {gt, prop, tactile} => gt then prop",
            "additional_normalization_after_env": False,
        },
        "gt": gt,
        "prop": _prop_schema(raw),
    }


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"isaac_sim": None, "isaac_lab": None}
    try:
        import isaacsim  # type: ignore

        versions["isaac_sim"] = getattr(isaacsim, "__version__", None)
    except Exception:
        pass
    try:
        import isaaclab  # type: ignore

        versions["isaac_lab"] = getattr(isaaclab, "__version__", None)
    except Exception:
        pass
    return versions


def extract_policy_vectors(states: Any, env_id: int) -> dict[str, np.ndarray]:
    obs = _policy_obs_dict(states)
    out: dict[str, np.ndarray] = {}
    if "gt" in obs:
        out["gt"] = _to_1d(obs["gt"], env_id)
    if "prop" in obs:
        out["prop"] = _to_1d(obs["prop"], env_id)
    if "tactile" in obs:
        out["tactile"] = _to_1d(obs["tactile"], env_id)
    return out


def _sim_time(raw: Any, env_id: int, control_dt: float) -> float:
    buf = getattr(raw, "episode_length_buf", None)
    if buf is not None:
        return float(buf[int(env_id)].item()) * float(control_dt)
    return float(getattr(raw, "common_step_counter", 0)) * float(control_dt)


@dataclass
class PolicyObsEpisode:
    episode_index: int
    env_id: int
    gt: list[np.ndarray] = field(default_factory=list)
    prop: list[np.ndarray] = field(default_factory=list)
    control_step: list[int] = field(default_factory=list)
    sim_time: list[float] = field(default_factory=list)
    global_step: list[int] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    task_success: bool | None = None
    motion_locked: bool | None = None


class PolicyObsRecorder:
    """Accumulate per-step policy observations and write one episode folder at a time."""

    def __init__(
        self,
        env: Any,
        out_root: Path,
        *,
        env_id: int = 0,
        control_dt: float,
        physics_dt: float,
        metadata_base: dict[str, Any] | None = None,
        max_episodes: int | None = None,
    ) -> None:
        self.env = env
        self.raw = _unwrap(env)
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.env_id = int(env_id)
        self.control_dt = float(control_dt)
        self.physics_dt = float(physics_dt)
        self.metadata_base = dict(metadata_base or {})
        self.max_episodes = max_episodes
        self.schema = build_observation_schema(env)
        self._episode_count = 0
        self.current = PolicyObsEpisode(episode_index=0, env_id=self.env_id)
        self.written: list[Path] = []

    @property
    def active(self) -> bool:
        if self.max_episodes is None:
            return True
        return self._episode_count < int(self.max_episodes)

    def record(self, states: Any, *, global_step: int) -> None:
        if not self.active:
            return
        vecs = extract_policy_vectors(states, self.env_id)
        if "gt" not in vecs or "prop" not in vecs:
            raise KeyError(
                f"policy obs missing gt/prop; keys={list(_policy_obs_dict(states).keys())}"
            )
        self.current.gt.append(vecs["gt"])
        self.current.prop.append(vecs["prop"])
        self.current.control_step.append(len(self.current.control_step))
        self.current.sim_time.append(_sim_time(self.raw, self.env_id, self.control_dt))
        self.current.global_step.append(int(global_step))

    def finalize(self, *, terminated: bool, truncated: bool) -> Path | None:
        ep = self.current
        if not ep.gt:
            self.current = PolicyObsEpisode(episode_index=self._episode_count, env_id=self.env_id)
            return None
        if not self.active:
            self.current = PolicyObsEpisode(episode_index=self._episode_count, env_id=self.env_id)
            return None
        ep.terminated = bool(terminated)
        ep.truncated = bool(truncated)
        eid = self.env_id
        # After env.step() Isaac Lab may already have reset this env; prefer
        # the snapshot taken at episode end (before `_reset_idx`).
        end_success = getattr(self.raw, "_episode_end_task_success", None)
        awarded = getattr(self.raw, "_task_success_bonus_awarded", None)
        src = end_success if end_success is not None else awarded
        if src is not None:
            ep.task_success = bool(src[eid].item())
            ep.motion_locked = bool(src[eid].item())

        self._episode_count += 1
        dest = self.out_root / f"episode_{self._episode_count:06d}"
        dest.mkdir(parents=True, exist_ok=True)
        _write_episode(dest, ep, self.schema, self)
        self.written.append(dest)
        self.current = PolicyObsEpisode(episode_index=self._episode_count, env_id=self.env_id)
        return dest


def _split_by_schema(matrix: np.ndarray, fields: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for item in fields:
        lo, hi = item["slice"]
        if hi <= matrix.shape[1]:
            out[item["name"]] = matrix[:, lo:hi]
    return out


def _write_episode(
    dest: Path,
    ep: PolicyObsEpisode,
    schema: dict[str, Any],
    recorder: PolicyObsRecorder,
) -> None:
    gt = np.stack(ep.gt, axis=0).astype(np.float32)
    prop = np.stack(ep.prop, axis=0).astype(np.float32)
    control_step = np.asarray(ep.control_step, dtype=np.int32)
    sim_time = np.asarray(ep.sim_time, dtype=np.float64)
    global_step = np.asarray(ep.global_step, dtype=np.int32)

    payload: dict[str, np.ndarray] = {
        "gt": gt,
        "prop": prop,
        "gt_raw": gt,
        "prop_raw": prop,
        "control_step": control_step,
        "sim_time": sim_time,
        "global_step": global_step,
        "encoder_early_fusion": np.concatenate([gt, prop], axis=1),
    }
    for name, arr in _split_by_schema(gt, schema["gt"]["fields"]).items():
        payload[f"gt_{name}"] = arr
    for name, arr in _split_by_schema(prop, schema["prop"]["fields"]).items():
        payload[f"prop_{name}"] = arr

    np.savez_compressed(dest / "data.npz", **payload)

    schema_path = dest / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    term_reason = "unknown"
    if ep.terminated and not ep.truncated:
        term_reason = "terminated"
    elif ep.truncated and not ep.terminated:
        term_reason = "truncated"
    elif ep.terminated and ep.truncated:
        term_reason = "terminated_and_truncated"

    meta = {
        **recorder.metadata_base,
        "episode_id": recorder._episode_count,
        "episode_index_0based": ep.episode_index,
        "env_id": ep.env_id,
        "num_steps": int(gt.shape[0]),
        "gt_dim": int(gt.shape[1]),
        "prop_dim": int(prop.shape[1]),
        "control_frequency_hz": (1.0 / recorder.control_dt) if recorder.control_dt else None,
        "physics_frequency_hz": (1.0 / recorder.physics_dt) if recorder.physics_dt else None,
        "control_dt": recorder.control_dt,
        "physics_dt": recorder.physics_dt,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "success": ep.task_success,
        "motion_locked": ep.motion_locked,
        "terminated": ep.terminated,
        "truncated": ep.truncated,
        "episode_termination_reason": term_reason,
        "recorded_tensor": "states['policy'][gt|prop] immediately before encoder(states) / policy.act",
        "git_commit": _git_commit(Path(__file__).resolve().parents[0]),
        **_package_versions(),
        "arrays_in_npz": sorted(payload.keys()),
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
