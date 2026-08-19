"""Helpers for organizing ``play.py`` evaluation outputs under ``outputs/``."""

from __future__ import annotations

import ast
import inspect
import json
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"
_TRAINING_RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
_WANDB_RUN_DIR_RE = re.compile(r"^run-\d{8}_\d{6}-(.+)$")


@dataclass(frozen=True)
class CheckpointInfo:
    checkpoint_path: Path
    checkpoint_name: str
    checkpoint_stem: str
    experiment: str | None
    training_run: str | None
    training_run_dir: Path | None
    logs_directory: str | None
    parsed: bool


@dataclass(frozen=True)
class EvalOutputPaths:
    executed_at: str
    outputs_root: Path
    checkpoint_root: Path
    wandb_id_file: Path
    evaluation_dir: Path
    metadata_file: Path
    video_file: Path | None


def execution_timestamp() -> str:
    return datetime.now().strftime(_TIMESTAMP_FMT)


def parse_checkpoint_path(checkpoint_path: str | Path) -> CheckpointInfo:
    path = Path(checkpoint_path).expanduser().resolve()
    parts = path.parts
    checkpoint_name = path.name
    checkpoint_stem = path.stem

    if "checkpoints" not in parts or path.suffix != ".pt":
        return CheckpointInfo(
            checkpoint_path=path,
            checkpoint_name=checkpoint_name,
            checkpoint_stem=checkpoint_stem,
            experiment=None,
            training_run=None,
            training_run_dir=None,
            logs_directory=None,
            parsed=False,
        )

    ckpt_idx = parts.index("checkpoints")
    if ckpt_idx < 2:
        parsed = False
        experiment = None
        training_run = None
        training_run_dir = None
        logs_directory = None
    else:
        training_run = parts[ckpt_idx - 1]
        experiment = parts[ckpt_idx - 2]
        logs_directory = parts[ckpt_idx - 3] if ckpt_idx >= 3 else None
        training_run_dir = Path(*parts[:ckpt_idx])
        parsed = _TRAINING_RUN_RE.match(training_run) is not None

    return CheckpointInfo(
        checkpoint_path=path,
        checkpoint_name=checkpoint_name,
        checkpoint_stem=checkpoint_stem,
        experiment=experiment,
        training_run=training_run,
        training_run_dir=training_run_dir,
        logs_directory=logs_directory,
        parsed=parsed,
    )


def build_eval_output_paths(
    checkpoint_info: CheckpointInfo,
    *,
    executed_at: str,
    repo_root: str | Path,
    record_video: bool,
) -> EvalOutputPaths:
    root = Path(repo_root).resolve()
    if checkpoint_info.parsed and checkpoint_info.experiment and checkpoint_info.training_run:
        checkpoint_root = (
            root
            / "outputs"
            / checkpoint_info.experiment
            / checkpoint_info.training_run
            / checkpoint_info.checkpoint_stem
        )
    else:
        checkpoint_root = root / "outputs" / "_unparsed_checkpoints" / checkpoint_info.checkpoint_stem

    evaluation_dir = checkpoint_root / "evaluations" / executed_at
    video_file = evaluation_dir / f"{executed_at}.mp4" if record_video else None
    return EvalOutputPaths(
        executed_at=executed_at,
        outputs_root=root / "outputs",
        checkpoint_root=checkpoint_root,
        wandb_id_file=checkpoint_root / "wandb_id.txt",
        evaluation_dir=evaluation_dir,
        metadata_file=evaluation_dir / "metadata.json",
        video_file=video_file,
    )


def _normalize_training_run_timestamp(training_run: str | None) -> str | None:
    if training_run is None:
        return None
    return training_run.replace("-", "").replace("_", "")


def _normalize_wandb_dir_timestamp(run_dir_name: str) -> str | None:
    if not run_dir_name.startswith("run-"):
        return None
    body = run_dir_name[len("run-") :]
    if "-" not in body:
        return None
    ts_part = body.rsplit("-", 1)[0]
    return ts_part.replace("_", "")


def lookup_wandb_run_id(
    *,
    training_run: str | None,
    search_roots: list[str | Path] | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(run_id, source_path)`` from local W&B metadata if available."""
    roots = [Path(p) for p in (search_roots or [Path.cwd() / "wandb", Path.cwd()])]
    target_ts = _normalize_training_run_timestamp(training_run)

    best: tuple[str, str, str] | None = None
    for root in roots:
        wandb_dir = root if root.name == "wandb" else root / "wandb"
        if not wandb_dir.is_dir():
            continue
        for run_dir in wandb_dir.iterdir():
            if not run_dir.is_dir():
                continue
            match = _WANDB_RUN_DIR_RE.match(run_dir.name)
            if match is None:
                continue
            run_id = match.group(1)
            metadata_file = run_dir / "files" / "wandb-metadata.json"
            source = str(metadata_file if metadata_file.is_file() else run_dir)
            if metadata_file.is_file():
                try:
                    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
                    run_id = str(payload.get("id") or run_id)
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass

            run_ts = _normalize_wandb_dir_timestamp(run_dir.name)
            if target_ts is not None and run_ts == target_ts:
                return run_id, source
            if best is None:
                best = (run_id, source, run_ts or "")

    if best is not None and target_ts is None:
        return best[0], best[1]
    return None, None


def ensure_wandb_id_file(
    wandb_id_file: Path,
    *,
    training_run: str | None,
    search_roots: list[str | Path] | None = None,
) -> tuple[str | None, str | None]:
    wandb_id_file.parent.mkdir(parents=True, exist_ok=True)
    if wandb_id_file.is_file():
        existing = wandb_id_file.read_text(encoding="utf-8").strip()
        if existing and not existing.startswith("#"):
            return existing, str(wandb_id_file)

    run_id, source = lookup_wandb_run_id(training_run=training_run, search_roots=search_roots)
    if run_id:
        wandb_id_file.write_text(f"{run_id}\n", encoding="utf-8")
    else:
        wandb_id_file.write_text("# wandb run id unavailable\n", encoding="utf-8")
    return run_id, source


def _cfg_section(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    return None


def _spawn_section(cfg_obj: Any, attr: str) -> dict[str, Any] | None:
    spawn = getattr(cfg_obj, "spawn", None)
    if spawn is None:
        return None
    return _cfg_section(getattr(spawn, attr, None))


def _unwrap_env(env) -> Any:
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        next_env = getattr(current, "env", None)
        if next_env is None:
            break
        current = next_env
    return getattr(current, "unwrapped", current)


def _cfg_named(env_cfg, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: getattr(env_cfg, name, None) for name in names}


def _cfg_scale_fields(env_cfg) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in dir(env_cfg):
        if name.startswith("_"):
            continue
        if not (name.endswith("_scale") or name.endswith("_bonus")):
            continue
        value = getattr(env_cfg, name, None)
        if callable(value):
            continue
        out[name] = value
    return out


def _compute_rewards_literal_scales(env) -> dict[str, Any] | None:
    """Numeric ``*_scale`` assignments inside the task ``compute_rewards`` function."""
    raw = _unwrap_env(env)
    fn = getattr(inspect.getmodule(type(raw)), "compute_rewards", None)
    if fn is None:
        return None
    try:
        src = inspect.getsource(fn)
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return None

    scales: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.endswith("_scale"):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
            scales[target.id] = node.value.value
    return scales or None


def _ast_numeric(node: ast.AST) -> float | int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _ast_numeric(node.operand)
        return None if value is None else -value
    if isinstance(node, ast.Call):
        args = list(node.args)
        if not args:
            return None
        return _ast_numeric(args[0])
    return None


def _ast_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _reset_target_pose_randomization(env) -> dict[str, Any] | None:
    """Ranges actually used in ``_reset_target_pose`` (Shadow Hand goal pose)."""
    raw = _unwrap_env(env)
    fn = getattr(raw, "_reset_target_pose", None)
    if fn is None or not callable(fn):
        return None
    try:
        src = inspect.getsource(fn)
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return None

    position_m: dict[str, Any] = {}
    orientation_deg: dict[str, Any] = {}
    xyz = ("x", "y", "z")
    rpy = {"yaw_rad": "yaw", "pitch_rad": "pitch", "roll_rad": "roll"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.value, ast.Call) or _ast_call_name(node.value) != "sample_uniform":
            continue
        if len(node.value.args) < 2:
            continue
        low = _ast_numeric(node.value.args[0])
        high = _ast_numeric(node.value.args[1])
        if low is None or high is None:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in rpy:
            orientation_deg[rpy[target.id]] = {"low_deg": low, "high_deg": high}
            continue
        if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Tuple) and len(target.slice.elts) >= 2:
            idx = _ast_numeric(target.slice.elts[1])
            if isinstance(idx, int) and 0 <= idx < 3:
                position_m[xyz[idx]] = {"low_m": low, "high_m": high}

    if not position_m and not orientation_deg:
        return None
    return {
        "source": f"{type(raw).__name__}._reset_target_pose",
        "position_m": position_m or None,
        "orientation_deg": orientation_deg or None,
    }


def _collect_robot_runtime(env) -> dict[str, Any] | None:
    raw = env
    while getattr(raw, "env", None) is not None:
        raw = raw.env
    raw = getattr(raw, "unwrapped", raw)
    robot = getattr(raw, "robot", None)
    if robot is None:
        return None

    joint_ids = list(getattr(raw, "actuated_dof_indices", []))
    if not joint_ids:
        return None

    data = robot.data
    idx = torch.as_tensor(joint_ids, device=data.joint_pos.device, dtype=torch.long)

    def _selected(value):
        if value is None:
            return None
        tensor = value[0] if value.ndim > 1 else value
        return tensor[idx].detach().cpu().tolist()

    return {
        "joint_names": [robot.joint_names[i] for i in joint_ids],
        "stiffness": _selected(getattr(data, "joint_stiffness", None)),
        "damping": _selected(getattr(data, "joint_damping", None)),
        "effort_limits": _selected(getattr(data, "joint_effort_limits", None)),
        "velocity_limits": _selected(getattr(data, "joint_vel_limits", None)),
    }


def build_play_metadata(
    *,
    args_cli,
    env,
    env_cfg,
    checkpoint_info: CheckpointInfo,
    output_paths: EvalOutputPaths,
    wandb_run_id: str | None,
    wandb_run_id_source: str | None,
    seed: int | None,
    joint_tracking_plot_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    physics_dt = float(getattr(env_cfg, "physics_dt", env_cfg.sim.dt))
    decimation = int(getattr(env_cfg, "decimation", 1))
    control_dt = physics_dt * decimation

    metadata: dict[str, Any] = {
        "evaluation": {
            "executed_at": output_paths.executed_at,
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
            "task": args_cli.task,
            "num_envs": int(env_cfg.scene.num_envs),
            "scene_mode": str(getattr(env_cfg, "scene_mode", "full")),
            "video": bool(args_cli.video),
            "video_length": int(args_cli.video_length),
            "seed": seed,
            "record_joint_tracking": bool(args_cli.record_joint_tracking),
            "joint_tracking_no_plots": bool(getattr(args_cli, "joint_tracking_no_plots", False)),
            "joint_tracking_max_episodes": int(args_cli.joint_tracking_max_episodes),
        },
        "checkpoint": {
            "experiment": checkpoint_info.experiment,
            "training_run": checkpoint_info.training_run,
            "checkpoint_name": checkpoint_info.checkpoint_name,
            "checkpoint_stem": checkpoint_info.checkpoint_stem,
            "checkpoint_path": str(checkpoint_info.checkpoint_path),
            "training_run_dir": str(checkpoint_info.training_run_dir) if checkpoint_info.training_run_dir else None,
            "logs_directory": checkpoint_info.logs_directory,
            "parsed": checkpoint_info.parsed,
        },
        "wandb": {
            "run_id": wandb_run_id,
            "run_id_source": wandb_run_id_source,
        },
        "simulation": {
            "physics_dt": physics_dt,
            "physics_frequency_hz": 1.0 / physics_dt if physics_dt else None,
            "decimation": decimation,
            "control_dt": control_dt,
            "control_frequency_hz": 1.0 / control_dt if control_dt else None,
            "render_interval": int(getattr(env_cfg.sim, "render_interval", decimation)),
            "gravity": list(getattr(env_cfg.sim, "gravity", (0.0, 0.0, -9.81))),
            "device": str(getattr(env_cfg.sim, "device", "")),
            "scene_mode": str(getattr(env_cfg, "scene_mode", "full")),
            "object_type": getattr(env_cfg, "object_type", None),
            "episode_length_s": getattr(env_cfg, "episode_length_s", None),
            "adaptive_physics_on_success": getattr(env_cfg, "adaptive_physics_on_success", None),
            "fine_physics_dt": getattr(env_cfg, "fine_physics_dt", None),
            "fine_decimation": getattr(env_cfg, "fine_decimation", None),
        },
        "insertion_gate": _cfg_named(
            env_cfg,
            (
                "insertion_gate_mode",
                "insertion_gate_temperature",
                "eval_opening_ellipse_threshold",
                "bracelet_desired_insert_depth",
                "bracelet_inside_opening_std",
            ),
        ),
        "randomization": {
            "reset_target_pose": _reset_target_pose_randomization(env),
        },
        "reward_scales": {
            "from_env_cfg": _cfg_scale_fields(env_cfg),
            "from_compute_rewards": _compute_rewards_literal_scales(env),
        },
        "physx": _cfg_section(getattr(env_cfg.sim, "physx", None)),
        "materials": {
            "sim_physics_material": _cfg_section(getattr(env_cfg.sim, "physics_material", None)),
        },
        "contact": {
            "deformable_object": _spawn_section(getattr(env_cfg, "object_cfg", None), "deformable_props"),
            "robot": {
                "articulation": _spawn_section(getattr(env_cfg, "robot_cfg", None), "articulation_props"),
                "rigid_body": _spawn_section(getattr(env_cfg, "robot_cfg", None), "rigid_props"),
                "collision": _spawn_section(getattr(env_cfg, "robot_cfg", None), "collision_props"),
            },
            "shadow_hand": {
                "articulation": _spawn_section(getattr(env_cfg, "hand_cfg", None), "articulation_props"),
                "rigid_body": _spawn_section(getattr(env_cfg, "hand_cfg", None), "rigid_props"),
                "collision": _spawn_section(getattr(env_cfg, "hand_cfg", None), "collision_props"),
            },
        },
        "robot_runtime": _collect_robot_runtime(env),
        "output": {
            "evaluation_dir": str(output_paths.evaluation_dir),
            "metadata_file": str(output_paths.metadata_file),
            "video_file": output_paths.video_file.name if output_paths.video_file else None,
            "wandb_id_file": str(output_paths.wandb_id_file),
            "joint_tracking_dir": str(default_joint_tracking_dir(output_paths.evaluation_dir)),
        },
    }
    if joint_tracking_plot_paths:
        metadata["output"]["joint_tracking_plots"] = joint_tracking_plot_paths
    return metadata


class _JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


def write_metadata_json(metadata: dict[str, Any], metadata_file: Path) -> None:
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=False, cls=_JsonEncoder) + "\n",
        encoding="utf-8",
    )


def finalize_recorded_video(evaluation_dir: Path, executed_at: str) -> Path | None:
    """Rename ``RecordVideo`` output to ``{executed_at}.mp4`` if present."""
    target = evaluation_dir / f"{executed_at}.mp4"
    if target.is_file():
        return target

    candidates = sorted(evaluation_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None

    recorded = candidates[-1]
    if recorded != target:
        recorded.rename(target)
    return target


def control_dt_from_env_cfg(env_cfg) -> float:
    physics_dt = float(getattr(env_cfg, "physics_dt", env_cfg.sim.dt))
    decimation = int(getattr(env_cfg, "decimation", 1))
    return physics_dt * decimation


def default_joint_tracking_dir(evaluation_dir: Path) -> Path:
    return evaluation_dir / "joint_tracking"


def render_joint_tracking_plots(
    traces: list[Any],
    out_dir: str | Path,
    *,
    control_dt: float,
    angle_unit: str = "deg",
) -> dict[str, str]:
    """Save left/right/torso arm PNGs (PDF-style layout) for each recorded episode."""
    from joint_tracking_debug import plot_left_right_arm_joint_states

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for trace in traces:
        if not getattr(trace, "steps", None):
            continue
        episode_paths = plot_left_right_arm_joint_states(
            trace,
            str(out),
            dt=float(control_dt),
            angle_unit=angle_unit,
        )
        ep = int(getattr(trace, "episode_index", 0))
        for key, path in episode_paths.items():
            paths[f"episode_{ep:03d}_{key}"] = path
    return paths


def render_joint_tracking_plots_from_json_dir(
    joint_tracking_dir: str | Path,
    *,
    control_dt: float,
    angle_unit: str = "deg",
) -> dict[str, str]:
    """Regenerate arm PNGs from saved ``episode_*_joint_tracking.json`` files."""
    from joint_tracking_debug import load_joint_tracking_json

    out = Path(joint_tracking_dir)
    traces = []
    for json_path in sorted(out.glob("episode_*_joint_tracking.json")):
        traces.append(load_joint_tracking_json(str(json_path)))
    return render_joint_tracking_plots(traces, out, control_dt=control_dt, angle_unit=angle_unit)
