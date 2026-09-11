"""Shared playback CLI and runtime for ``play.py`` / ``play_eval.py``.

This module is imported *before* ``AppLauncher`` starts (for argparse helpers),
so the top level must not import Isaac / torch. Heavy imports live inside the
setup and loop functions, which run after SimulationApp is created.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def add_play_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """CLI flags shared by playback and evaluation scripts (``play.py`` baseline)."""
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
    parser.add_argument("--video_length", type=int, default=100, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--checkpoint",
        "--checkpoints",
        type=str,
        default=None,
        dest="checkpoint",
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to save recorded videos.")
    parser.add_argument("--agent_cfg", type=str, default=None, help="Name of the agent configuration.")
    parser.add_argument(
        "--scene-mode",
        type=str,
        choices=("full", "free_space"),
        default="full",
        help="Scene contents: full task scene (default) or robot-only free-space diagnostics.",
    )
    parser.add_argument(
        "--disable-self-collision",
        action="store_true",
        help="Disable AIREC self-collision. Not implied by --scene-mode free_space.",
    )
    parser.add_argument(
        "--show-task-markers",
        action="store_true",
        help=(
            "Show N/S/E/W/C rim goal spheres (+ thumb/pinky targets). "
            "Requires a GUI viewport (do not use --headless). Also prints frozen NSEW indices at init."
        ),
    )
    parser.add_argument(
        "--show-knuckle-markers",
        action="store_true",
        help=(
            "Show spheres on the eval knuckle COMs (thumb thbase, index/middle/ring/little *knuckle). "
            "Color is insertion state: red=PRE, yellow=BAND, blue=POST, green=LIVE. "
            "Other fingers LIVE = knuckle+distal POST inside the ellipse. Thumb LIVE "
            "only when thdistal+thmiddle+thproximal all sit in the opening slab. "
            "Also draws smaller Td/Tm/Tp station spheres. Prints T/I/M/R/P and "
            "thumb-station PRE/POST/IN. Requires GUI (do not use --headless). "
            "Limit to one clone with --knuckle-marker-env-id."
        ),
    )
    parser.add_argument(
        "--knuckle-marker-env-id",
        type=int,
        default=None,
        help="With --show-knuckle-markers, draw only this env (e.g. 4). Default: all parallel envs.",
    )
    parser.add_argument(
        "--knuckle-status-interval",
        type=int,
        default=None,
        help=(
            "With --show-knuckle-markers: print PRE/POST/LIVE/LATCH every N control steps "
            "(and on any state change). Default: 5. 1 = every step. 0 = change-only."
        ),
    )
    parser.add_argument(
        "--show-com-marker",
        action="store_true",
        help="Show a large sphere at the robot full-body CoM. Requires a GUI viewport (do not use --headless).",
    )
    parser.add_argument(
        "--debug-com",
        action="store_true",
        help="Periodically print com_pos_b / tip-band status (for tuning com_tip_x_min/max).",
    )
    parser.add_argument(
        "--debug-opening-pca",
        action="store_true",
        help=(
            "Draw opening-ring PCA axes (e1/e2/n) on reset via debug_draw. "
            "Requires GUI; useful to inspect how NSEW axes are chosen."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--print-eval-episode-returns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print mean and per-env returns when an eval env finishes (terminated/truncated). "
            "On by default for play; pass --no-print-eval-episode-returns to silence."
        ),
    )
    parser.add_argument(
        "--debug-joints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print policy action vs joint_pos_cmd vs measured joint_pos (overrides env cfg).",
    )
    parser.add_argument(
        "--debug-joint-env-id",
        type=int,
        default=None,
        help="Env index for joint debug stdout (default: env cfg, usually 0).",
    )
    parser.add_argument(
        "--debug-joint-interval",
        type=int,
        default=None,
        help="Print joint debug every N control steps (default: env cfg).",
    )
    parser.add_argument(
        "--residual-scale-mult",
        type=float,
        default=None,
        help=(
            "Multiply residual action scale: q = q_ref + a * s * mult. "
            "Default: env cfg (0.25 on AIREC_Reach_Deformable_Bracelet). "
            "Use 1.0 to restore the training-time inferred half-range."
        ),
    )
    parser.add_argument(
        "--record-joint-tracking",
        nargs="?",
        const="joint_tracking_plots",
        default=None,
        help=(
            "Record q_policy / q_cmd / q_act + applied/computed torque each step for --joint-tracking-env-id. "
            "Optional DIR (default: joint_tracking_plots). Writes JSON/CSV/PNG per episode."
        ),
    )
    parser.add_argument(
        "--joint-tracking-no-plots",
        action="store_true",
        help="With --record-joint-tracking: save JSON/CSV only, skip PNG plots.",
    )
    parser.add_argument(
        "--joint-tracking-env-id",
        type=int,
        default=0,
        help="Env index for joint tracking recording (default: 0).",
    )
    parser.add_argument(
        "--joint-tracking-max-episodes",
        type=int,
        default=3,
        help="Stop joint tracking recording after this many completed episodes.",
    )
    parser.add_argument(
        "--joint-tracking-max-joints",
        type=int,
        default=None,
        help="Optional: plot only the first N actuated joints (less clutter).",
    )
    parser.add_argument(
        "--record-policy-observations",
        nargs="?",
        const="policy_observations",
        default=None,
        help=(
            "Save per-step policy-input gt/prop for deploy comparison. "
            "Optional DIR (default: <eval_dir>/policy_observations). "
            "Writes episode_XXXXXX/{data.npz,schema.json,metadata.json}."
        ),
    )
    parser.add_argument(
        "--policy-obs-env-id",
        type=int,
        default=0,
        help="Env index for --record-policy-observations (default: 0; use with --num_envs 1).",
    )
    parser.add_argument(
        "--policy-obs-max-episodes",
        type=int,
        default=None,
        help="Stop observation logging after this many episodes (default: no extra cap).",
    )
    parser.add_argument(
        "--record-finger-insertion-gate",
        nargs="?",
        const="finger_insertion_gate_records",
        default=None,
        help=(
            "Record per_finger_soft_inside each step for --finger-gate-env-id. "
            "Optional DIR (default: finger_insertion_gate_records). Writes JSON + CSV per episode."
        ),
    )
    parser.add_argument(
        "--plot-finger-insertion-gate",
        type=str,
        default=None,
        help="Alias for --record-finger-insertion-gate DIR (also saves PNG unless --finger-gate-no-plots).",
    )
    parser.add_argument(
        "--finger-gate-no-plots",
        action="store_true",
        help="With recording enabled: save JSON/CSV only, skip PNG plots.",
    )
    parser.add_argument(
        "--finger-gate-env-id",
        type=int,
        default=0,
        help="Env index to log (use 0 with --num_envs 1 for animation).",
    )
    parser.add_argument(
        "--finger-gate-max-episodes",
        type=int,
        default=10,
        help="Stop after this many episode plots (insertion gate logging).",
    )
    parser.add_argument(
        "--finger-gate-show-ellipse",
        action="store_true",
        help="With --finger-gate-per-finger: also plot per_finger_inside_ellipse (dotted).",
    )
    parser.add_argument(
        "--finger-gate-per-finger",
        action="store_true",
        help="Plot each per_finger_soft_inside curve; default is fingers_inside_soft_gate only.",
    )
    parser.add_argument(
        "--num-fingers",
        type=int,
        default=5,
        help="Number of fingers for insertion gate logging.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="PathTracing",
        choices=["RayTracedLighting", "PathTracing"],
        help="Renderer to use.",
    )
    parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1, help="Number of samples per pixel per frame.")
    return parser


def add_play_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Evaluation-only flags for ``play_eval.py``."""
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=50,
        help="Stop evaluation after this many completed episodes.",
    )
    parser.add_argument(
        "--insertion-delta-m",
        type=float,
        default=0.003,
        help="Signed-distance hysteresis (m) for opening-plane crossing (default: 0.003).",
    )
    parser.add_argument(
        "--insertion-confirm-frames",
        type=int,
        default=4,
        help="Consecutive destination-side frames required to confirm a crossing (default: 4 at 50 Hz = 80 ms).",
    )
    parser.add_argument(
        "--insertion-window-sec",
        type=float,
        default=1.0,
        help="Deprecated. Ignored by the crossing state machine; kept so old commands still parse.",
    )
    parser.add_argument(
        "--insertion-ratio-threshold",
        type=float,
        default=0.8,
        help="Deprecated. Ignored by the crossing state machine; kept so old commands still parse.",
    )
    parser.add_argument(
        "--insertion-ellipse-threshold",
        type=float,
        default=None,
        help=(
            "Opening-ellipse inside threshold at the interpolated crossing (default: env "
            "eval_opening_ellipse_threshold, usually 1.0). "
            "Inside when ((y-c_y)/r_y)^2 + ((z-c_z)/r_z)^2 <= threshold."
        ),
    )
    parser.add_argument(
        "--eval-env-id",
        type=int,
        default=None,
        help="If set, only this env_id contributes evaluation episodes. Default: all parallel envs.",
    )
    parser.add_argument(
        "--success-definition",
        choices=("legacy", "strict"),
        default="legacy",
        help=(
            "Deprecated. Official Task success is always training motion-lock "
            "(wrist-within-goal AND knuckle all-five). 'strict' is ignored for "
            "Task success; geometric completion is reported separately as "
            "all-five passage + wrist complete. Training is unchanged."
        ),
    )
    parser.add_argument(
        "--complete-dressing-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Success / motion lock require wrist-within-1cm AND all five confirmed "
            "finger insertions (default: on; same rule as training). "
            "Use --no-complete-dressing-success for the old wrist-only criterion."
        ),
    )
    parser.add_argument(
        "--debug-insertion",
        action="store_true",
        default=False,
        help=(
            "Print per-finger insertion debug (signed distance, ellipse, latch vs live knuckle+distal) "
            "and write insertion_debug/ CSVs. Best with --num_envs 1."
        ),
    )
    parser.add_argument(
        "--debug-insertion-interval",
        type=int,
        default=10,
        help="With --debug-insertion: print every N control steps (default: 10 = 0.2 s at 50 Hz). 1 = every step.",
    )
    parser.add_argument(
        "--record-insertion-debug",
        action="store_true",
        default=False,
        help=(
            "Write per-step insertion debug CSV/JSON under <eval_dir>/insertion_debug/ "
            "(center, radii, knuckle+distal xyz/d/e/side, latch vs live). "
            "Implied by --debug-insertion."
        ),
    )
    parser.add_argument(
        "--legacy-dressing-eval",
        "--dressing-eval",
        action="store_true",
        dest="legacy_dressing_eval",
        default=False,
        help="Run the previous dressing_eval path (strict success / final-frame ellipse) and exit.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="With --legacy-dressing-eval: label for evaluation tables.",
    )
    parser.add_argument(
        "--object-type",
        type=str,
        default=None,
        help="With --legacy-dressing-eval: object category for tables.",
    )
    parser.add_argument(
        "--eval-save-dir",
        type=str,
        default=None,
        help="With --legacy-dressing-eval: save the old evaluation JSON.",
    )
    parser.add_argument(
        "--insertion-gate-analysis-dir",
        type=str,
        default=None,
        help="With --legacy-dressing-eval: insertion-gate sensitivity analysis directory.",
    )
    parser.add_argument(
        "--video-failures",
        type=str,
        default=None,
        help=(
            "Record only selected failure videos (implies --video). Use with --num_envs 1. "
            "Comma-separated: no_insertion, partial, incomplete_wrist, all_fail. "
            "snag_suspect is an alias for incomplete_wrist (5/5 passed, wrist not done). "
            "Each kept episode is one mp4; successes are discarded."
        ),
    )
    parser.add_argument(
        "--video-env-ids",
        type=str,
        default=None,
        help=(
            "Keep videos only for these env_id or episode numbers (comma-separated, e.g. 5,20,44). "
            "With --num_envs 1 the running env_id is 0; prefer --video-failures for that case."
        ),
    )
    parser.add_argument(
        "--video-shortfall",
        type=str,
        default=None,
        help=(
            "Further filter incomplete-wrist videos: insufficient, near, before_all_five, regression. "
            "Comma-separated."
        ),
    )
    parser.add_argument(
        "--video-max",
        type=int,
        default=None,
        help="Stop after this many kept failure videos (default: no extra cap; --max-episodes still applies).",
    )
    parser.add_argument(
        "--video-from-eval",
        type=str,
        default=None,
        help=(
            "Path to a previous evaluation_summary.json / failure_modes.json / eval directory. "
            "Prints those failure env_ids and, if --video-failures is omitted, records the same modes."
        ),
    )
    return parser


VIDEO_FAILURE_ALIASES = {
    "no_insertion": "no_insertion",
    "none": "no_insertion",
    "partial": "partial_finger_insertion",
    "partial_finger_insertion": "partial_finger_insertion",
    "incomplete_wrist": "full_finger_insertion_incomplete_wrist_advancement",
    "wrist": "full_finger_insertion_incomplete_wrist_advancement",
    "full_finger_insertion_incomplete_wrist_advancement": (
        "full_finger_insertion_incomplete_wrist_advancement"
    ),
    "success": "full_insertion_and_wrist_success",
    "full_insertion_and_wrist_success": "full_insertion_and_wrist_success",
    "all_fail": "all_fail",
    "failures": "all_fail",
    "all": "all_fail",
    "snag": "snag_suspect",
    "snag_suspect": "snag_suspect",
}

VIDEO_SHORTFALL_ALIASES = {
    "insufficient": "insufficient_advancement_toward_wrist",
    "insufficient_advancement_toward_wrist": "insufficient_advancement_toward_wrist",
    "near": "reached_near_wrist_threshold_but_did_not_cross",
    "reached_near_wrist_threshold_but_did_not_cross": "reached_near_wrist_threshold_but_did_not_cross",
    "before_all_five": "reached_wrist_criterion_before_all_five",
    "reached_wrist_criterion_before_all_five": "reached_wrist_criterion_before_all_five",
    "regression": "regression_after_full_finger_insertion",
    "regression_after_full_finger_insertion": "regression_after_full_finger_insertion",
}


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).replace(" ", ",").split(",") if part.strip()]


def parse_video_env_ids(raw: str | None) -> set[int]:
    ids: set[int] = set()
    for tok in _parse_csv_tokens(raw):
        ids.add(int(tok))
    return ids


def parse_video_failure_modes(raw: str | None) -> set[str]:
    modes: set[str] = set()
    for tok in _parse_csv_tokens(raw):
        key = tok.lower().replace("-", "_")
        if key not in VIDEO_FAILURE_ALIASES:
            known = ", ".join(sorted(VIDEO_FAILURE_ALIASES))
            raise ValueError(f"Unknown --video-failures value {tok!r}. Expected one of: {known}")
        modes.add(VIDEO_FAILURE_ALIASES[key])
    return modes


def parse_video_shortfalls(raw: str | None) -> set[str]:
    tags: set[str] = set()
    for tok in _parse_csv_tokens(raw):
        key = tok.lower().replace("-", "_")
        if key not in VIDEO_SHORTFALL_ALIASES:
            known = ", ".join(sorted(VIDEO_SHORTFALL_ALIASES))
            raise ValueError(f"Unknown --video-shortfall value {tok!r}. Expected one of: {known}")
        tags.add(VIDEO_SHORTFALL_ALIASES[key])
    return tags


def wants_failure_videos(args_cli: Any) -> bool:
    return bool(
        getattr(args_cli, "video_failures", None)
        or getattr(args_cli, "video_env_ids", None)
        or getattr(args_cli, "video_from_eval", None)
        or getattr(args_cli, "video_shortfall", None)
        or getattr(args_cli, "video_max", None)
    )


def load_video_from_eval(path_raw: str | None) -> dict[str, Any] | None:
    """Read a previous eval JSON and return failure-mode env/episode ids."""
    if not path_raw:
        return None
    path = Path(path_raw)
    if path.is_dir():
        for name in ("failure_modes.json", "evaluation_summary.json"):
            cand = path / name
            if cand.is_file():
                path = cand
                break
    if not path.is_file():
        raise FileNotFoundError(f"--video-from-eval not found: {path_raw}")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data.get("failure_modes") if isinstance(data.get("failure_modes"), dict) else data
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list):
        raise ValueError(f"--video-from-eval has no failure-mode episodes: {path}")
    by_mode: dict[str, list[int]] = {}
    for rec in episodes:
        mode = str(rec.get("failure_mode") or "")
        if mode == "full_insertion_and_wrist_success":
            continue
        env_id = rec.get("env_id", rec.get("episode"))
        if env_id is None:
            continue
        by_mode.setdefault(mode, []).append(int(env_id))
    return {"path": str(path), "by_mode": by_mode, "episodes": episodes}


def unwrap_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        next_env = getattr(current, "env", None)
        if next_env is None:
            break
        current = next_env
    return getattr(current, "unwrapped", current)


def resolve_num_eval_envs(env: Any, num_eval_envs: int | None = None) -> int:
    n_envs = int(getattr(env, "num_envs", 1))
    if num_eval_envs is None:
        cfg = getattr(unwrap_env(env), "cfg", None)
        num_eval_envs = int(getattr(cfg, "num_eval_envs", n_envs)) if cfg is not None else n_envs
    requested = int(num_eval_envs)
    effective = min(requested, n_envs)
    if effective < requested:
        print(
            f"[play] WARNING: config requests {requested} eval env(s) but only "
            f"{n_envs} parallel env(s) are running; using env_id 0..{effective - 1}."
        )
    return effective


@dataclass
class PlaySession:
    args_cli: Any
    env: Any
    env_cfg: Any
    agent_cfg: dict[str, Any]
    agent: Any
    encoder: Any
    resume_path: str
    output_paths: Any
    checkpoint_info: Any
    wandb_run_id: str | None
    wandb_run_id_source: str | None
    log_prefix: str = "play"


@dataclass
class PlaybackResult:
    timestep: int = 0
    joint_tracking_plot_paths: dict[str, str] = field(default_factory=dict)
    finger_gate_dir: str | None = None
    joint_track_dir: str | None = None
    policy_obs_dir: str | None = None


def _resolve_default_record_dir(record_arg: str | None, default_name: str, evaluation_dir: Path) -> str | None:
    from play_output_utils import default_joint_tracking_dir

    if not record_arg:
        return None
    if record_arg == default_name:
        return str(default_joint_tracking_dir(evaluation_dir))
    return record_arg


def print_runtime_diagnostics(env, env_cfg, checkpoint_path: str) -> None:
    """Print the comparison contract shared by full and free-space runs."""
    import torch

    raw = unwrap_env(env)
    scene_mode = str(getattr(env_cfg, "scene_mode", "full"))
    free_space = scene_mode == "free_space"
    obs_space = env.observation_space["policy"]
    obs_shapes = {key: tuple(space.shape) for key, space in obs_space.spaces.items()}
    obs_keys = list(obs_space.spaces.keys())
    action_shape = tuple(env.action_space.shape)
    expected_keys = list(getattr(env_cfg, "obs_list", []))
    assert set(obs_keys) == set(expected_keys), (
        f"observation keys changed: runtime={sorted(obs_keys)}, expected={sorted(expected_keys)}"
    )
    assert action_shape[-1] == int(env_cfg.num_actions), (
        f"action width changed: runtime={action_shape}, expected={env_cfg.num_actions}"
    )

    robot = raw.robot
    data = robot.data
    joint_ids = list(raw.actuated_dof_indices)
    joint_order = [robot.joint_names[i] for i in joint_ids]
    self_collision = bool(
        getattr(
            getattr(getattr(env_cfg.robot_cfg, "spawn", None), "articulation_props", None),
            "enabled_self_collisions",
            False,
        )
    )

    scene = raw.scene
    articulation_keys = list(getattr(scene, "articulations", {}).keys())
    rigid_keys = list(getattr(scene, "rigid_objects", {}).keys())
    deformable_keys = list(getattr(scene, "deformable_objects", {}).keys())
    if free_space:
        assert articulation_keys == ["robot"], f"free_space contains unexpected articulations: {articulation_keys}"
        assert not rigid_keys, f"free_space contains rigid objects: {rigid_keys}"
        assert not deformable_keys, f"free_space contains deformable objects: {deformable_keys}"

    idx = torch.as_tensor(joint_ids, device=data.joint_pos.device, dtype=torch.long)
    stiffness = getattr(data, "joint_stiffness", None)
    damping = getattr(data, "joint_damping", None)
    effort_limits = getattr(data, "joint_effort_limits", None)
    velocity_limits = getattr(data, "joint_vel_limits", None)

    def _selected(value):
        if value is None:
            return "unavailable"
        tensor = value[0] if value.ndim > 1 else value
        return tensor[idx].detach().cpu().tolist()

    print(f"Scene mode: {scene_mode}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Robot asset: enabled")
    print(f"Bracelet asset: {'disabled' if free_space else 'enabled'}")
    print(f"Shadow Hand asset: {'disabled' if free_space else 'enabled'}")
    print("Human asset: disabled (this task has no separate Human articulation)")
    print(f"External collision: {'disabled' if free_space else 'enabled'}")
    print(f"Robot self-collision: {'enabled' if self_collision else 'disabled'}")
    print("Observation schema unchanged: yes")
    print("Action schema unchanged: yes")
    print(f"Observation keys: {obs_keys}")
    print(f"Observation shapes: {obs_shapes}")
    print(f"Dummy observations: {getattr(raw, 'free_space_dummy_observations', [])}")
    print(f"Action shape: {action_shape}")
    print(f"Joint order: {joint_order}")
    print(f"Physics timestep: {float(env_cfg.sim.dt):.9g} s")
    print(f"Control/policy timestep: {float(env_cfg.sim.dt) * int(env_cfg.decimation):.9g} s")
    print(f"Decimation: {int(env_cfg.decimation)}")
    print(f"Stiffness: {_selected(stiffness)}")
    print(f"Damping: {_selected(damping)}")
    print(f"Torque limits: {_selected(effort_limits)}")
    print(f"Velocity limits: {_selected(velocity_limits)}")
    scale_mult = float(getattr(env_cfg, "residual_action_scale_mult", 1.0))
    live_scale = getattr(raw, "_body_residual_scale", None)
    print(f"Residual action scale mult: {scale_mult:g}")
    print(
        "Residual action scale live: "
        f"{live_scale.detach().cpu().tolist() if live_scale is not None else 'unavailable'}"
    )
    print(f"Scene articulations: {articulation_keys}")
    print(f"Scene rigid objects: {rigid_keys}")
    print(f"Scene deformable objects: {deformable_keys}")


def setup_play_session(
    parser: argparse.ArgumentParser,
    args_cli: Any,
    *,
    log_prefix: str = "play",
) -> PlaySession:
    """Resolve checkpoint, output dir, env, and PPO agent. Requires SimulationApp."""
    import torch
    import isaaclab_tasks  # noqa: F401
    from common_utils import (
        LOG_PATH,
        load_play_checkpoint,
        make_env,
        make_models,
        resolve_checkpoint_path,
        set_seed,
        update_env_cfg,
    )
    from isaaclab.utils import update_dict
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
    from multimodal_rl.tools.writer import Writer
    from play_output_utils import (
        build_eval_output_paths,
        ensure_wandb_id_file,
        execution_timestamp,
        parse_checkpoint_path,
    )

    if not args_cli.checkpoint:
        parser.error(
            "Missing checkpoint path. Use --checkpoint /path/to/best_agent.pt "
            "(not --checkpoints unless you use the plural alias we accept)."
        )
    try:
        resume_path = resolve_checkpoint_path(args_cli.checkpoint)
    except (ValueError, FileNotFoundError) as err:
        parser.error(str(err))

    if wants_failure_videos(args_cli):
        args_cli.video = True
    executed_at = execution_timestamp()
    checkpoint_info = parse_checkpoint_path(resume_path)
    output_paths = build_eval_output_paths(
        checkpoint_info,
        executed_at=executed_at,
        repo_root=LOG_PATH,
        record_video=bool(args_cli.video),
    )
    if output_paths.evaluation_dir.exists():
        raise FileExistsError(
            f"Evaluation output directory already exists: {output_paths.evaluation_dir}. "
            "Wait one second and rerun to get a new timestamp, or remove the existing directory."
        )
    output_paths.evaluation_dir.mkdir(parents=True, exist_ok=True)

    wandb_run_id, wandb_run_id_source = ensure_wandb_id_file(
        output_paths.wandb_id_file,
        training_run=checkpoint_info.training_run,
        search_roots=[LOG_PATH, checkpoint_info.training_run_dir or LOG_PATH],
    )
    print(f"[{log_prefix}] evaluation output -> {output_paths.evaluation_dir}")

    env_cfg, agent_cfg = register_task_to_hydra(args_cli.task, "skrl_cfg_entry_point")
    if args_cli.agent_cfg is not None:
        specialised_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent_cfg)
        agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    if wants_failure_videos(args_cli):
        args_cli.video = True
        n_envs = int(getattr(args_cli, "num_envs", 1) or 1)
        if n_envs != 1:
            print(
                f"[{log_prefix}] WARNING: --video-failures is cheapest with --num_envs 1 "
                f"(you passed {n_envs}; the camera records the tiled view of every env)."
            )
    if args_cli.video:
        args_cli.video_dir = str(output_paths.evaluation_dir)
    elif args_cli.video_dir is None:
        args_cli.video_dir = agent_cfg["experiment"].get("video_dir") or os.path.join(LOG_PATH, "videos")
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    if getattr(env_cfg, "debug_joint_cmd_vs_actual", False):
        print(
            f"[{log_prefix}] joint debug ON: "
            f"env_id={getattr(env_cfg, 'debug_joint_print_env_id', 0)}, "
            f"interval={getattr(env_cfg, 'debug_joint_print_interval', 10)}"
        )

    writer = Writer(agent_cfg, play=True)
    env = make_env(
        agent_cfg,
        env_cfg,
        writer,
        args_cli,
        video_name_prefix=executed_at if args_cli.video else None,
    )
    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    ppo_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_agent_cfg.update(agent_cfg["agent"])
    value.value_preprocessor = value_preprocessor
    agent = PPO(
        encoder,
        policy,
        value,
        value_preprocessor,
        memory=None,
        cfg=ppo_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer,
        ssl_task=None,
        dtype=dtype,
        debug=agent_cfg["experiment"]["debug"],
    )

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    load_play_checkpoint(agent, resume_path)
    print_runtime_diagnostics(env, env_cfg, resume_path)
    modules = torch.load(resume_path, map_location=env.device)
    if isinstance(modules, dict):
        for name in modules.keys():
            print(f"  - {name}")

    return PlaySession(
        args_cli=args_cli,
        env=env,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        agent=agent,
        encoder=encoder,
        resume_path=resume_path,
        output_paths=output_paths,
        checkpoint_info=checkpoint_info,
        wandb_run_id=wandb_run_id,
        wandb_run_id_source=wandb_run_id_source,
        log_prefix=log_prefix,
    )


def run_playback_loop(
    session: PlaySession,
    simulation_app: Any,
    *,
    on_after_step: Callable[..., None] | None = None,
    on_periodic_hard_reset: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    stop_on_video_length: bool = True,
    stop_on_recorder_caps: bool = True,
) -> PlaybackResult:
    """Deterministic policy playback with optional recorders and step hooks."""
    import torch
    from finger_insertion_gate_viz import (
        FingerInsertionEpisodeTrace,
        FingerInsertionStep,
        finalize_finger_gate_episode,
        read_finger_insertion_from_env,
        save_all_traces_csv,
    )
    from joint_tracking_debug import (
        JointTrackingEpisodeTrace,
        JointTrackingStep,
        finalize_joint_tracking_episode,
        read_joint_tracking_from_env,
        save_all_joint_tracking_csv,
    )
    from play_output_utils import control_dt_from_env_cfg
    from policy_obs_logger import PolicyObsRecorder

    args_cli = session.args_cli
    env = session.env
    env_cfg = session.env_cfg
    agent = session.agent
    encoder = session.encoder
    log_prefix = session.log_prefix

    timestep = 0
    ep_length = env.env.unwrapped.max_episode_length - 1
    returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
    mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)
    num_eval_envs = resolve_num_eval_envs(env, int(session.agent_cfg["trainer"]["num_eval_envs"]))
    episode_return_sum = torch.zeros((num_eval_envs, 1), device=env.device)
    print_eval_episode_returns = bool(args_cli.print_eval_episode_returns)

    states, infos = env.reset(hard=True)

    finger_gate_dir = args_cli.record_finger_insertion_gate or args_cli.plot_finger_insertion_gate
    finger_gate_save_plots = finger_gate_dir is not None and not bool(args_cli.finger_gate_no_plots)
    finger_gate_env_id = int(args_cli.finger_gate_env_id)
    finger_gate_traces: list = []
    finger_gate_current: FingerInsertionEpisodeTrace | None = None
    finger_gate_episode_cap = int(args_cli.finger_gate_max_episodes)
    if finger_gate_dir:
        finger_gate_dir = os.path.abspath(finger_gate_dir)
        os.makedirs(finger_gate_dir, exist_ok=True)
        finger_gate_current = FingerInsertionEpisodeTrace(episode_index=0, env_id=finger_gate_env_id)
        print(f"[{log_prefix}] recording finger insertion soft gate: env_id={finger_gate_env_id} -> {finger_gate_dir}")

    joint_track_dir = _resolve_default_record_dir(
        args_cli.record_joint_tracking,
        "joint_tracking_plots",
        session.output_paths.evaluation_dir,
    )
    joint_track_save_plots = joint_track_dir is not None and not bool(args_cli.joint_tracking_no_plots)
    joint_track_control_dt = control_dt_from_env_cfg(env_cfg)
    joint_track_env_id = int(args_cli.joint_tracking_env_id)
    joint_track_traces: list = []
    joint_track_current: JointTrackingEpisodeTrace | None = None
    joint_track_episode_cap = int(args_cli.joint_tracking_max_episodes)
    joint_tracking_plot_paths: dict[str, str] = {}
    if joint_track_dir:
        joint_track_dir = os.path.abspath(joint_track_dir)
        os.makedirs(joint_track_dir, exist_ok=True)
        joint_track_current = JointTrackingEpisodeTrace(episode_index=0, env_id=joint_track_env_id)
        print(
            f"[{log_prefix}] recording joint tracking (q_policy / q_cmd / q_act + torque): "
            f"env_id={joint_track_env_id} -> {joint_track_dir} "
            "(no stdout joint dump; use --debug-joints to print actions)"
        )

    policy_obs_dir = _resolve_default_record_dir(
        getattr(args_cli, "record_policy_observations", None),
        "policy_observations",
        session.output_paths.evaluation_dir,
    )
    policy_obs_recorder: PolicyObsRecorder | None = None
    if policy_obs_dir:
        policy_obs_dir = os.path.abspath(policy_obs_dir)
        physics_dt = float(getattr(env_cfg, "physics_dt", getattr(getattr(env_cfg, "sim", None), "dt", 0.002)))
        policy_obs_recorder = PolicyObsRecorder(
            env,
            Path(policy_obs_dir),
            env_id=int(getattr(args_cli, "policy_obs_env_id", 0)),
            control_dt=joint_track_control_dt,
            physics_dt=physics_dt,
            metadata_base={
                "task": getattr(args_cli, "task", None),
                "checkpoint": session.resume_path,
                "scene_mode": str(getattr(env_cfg, "scene_mode", "full")),
                "policy": {
                    "play_mode": "deterministic mean (GaussianPolicy.act(z, deterministic=True))",
                    "encoder_method": (session.agent_cfg.get("encoder") or {}).get("method"),
                    "encoder_state_preprocessor": (session.agent_cfg.get("encoder") or {}).get("state_preprocessor"),
                    "obs_list": list(getattr(env_cfg, "obs_list", [])),
                },
            },
            max_episodes=getattr(args_cli, "policy_obs_max_episodes", None),
        )
        print(
            f"[{log_prefix}] recording policy observations (gt/prop before encoder): "
            f"env_id={policy_obs_recorder.env_id} -> {policy_obs_dir}"
        )

    def _finalize_finger_gate_episode(*, terminated: bool, truncated: bool) -> None:
        nonlocal finger_gate_current
        if finger_gate_current is None or not finger_gate_current.steps:
            finger_gate_current = FingerInsertionEpisodeTrace(
                episode_index=len(finger_gate_traces), env_id=finger_gate_env_id
            )
            return
        finger_gate_current.terminated = terminated
        finger_gate_current.truncated = truncated
        ep = finger_gate_current.episode_index
        paths = finalize_finger_gate_episode(
            finger_gate_current,
            finger_gate_dir,
            num_fingers=args_cli.num_fingers,
            save_plots=finger_gate_save_plots,
            plot_per_finger=bool(args_cli.finger_gate_per_finger),
            show_ellipse=bool(args_cli.finger_gate_show_ellipse),
        )
        print(
            f"[{log_prefix}] finger gate episode {ep}: {len(finger_gate_current.steps)} steps -> "
            f"{paths.get('png', paths.get('csv', paths.get('json')))}"
        )
        finger_gate_traces.append(finger_gate_current)
        finger_gate_current = FingerInsertionEpisodeTrace(
            episode_index=len(finger_gate_traces), env_id=finger_gate_env_id
        )

    def _finalize_joint_track_episode(*, terminated: bool, truncated: bool) -> None:
        nonlocal joint_track_current, joint_tracking_plot_paths
        if joint_track_current is None or not joint_track_current.steps:
            joint_track_current = JointTrackingEpisodeTrace(
                episode_index=len(joint_track_traces), env_id=joint_track_env_id
            )
            return
        joint_track_current.terminated = terminated
        joint_track_current.truncated = truncated
        ep = joint_track_current.episode_index
        paths = finalize_joint_tracking_episode(
            joint_track_current,
            joint_track_dir,
            save_plots=joint_track_save_plots,
            max_joints=args_cli.joint_tracking_max_joints,
            dt=joint_track_control_dt,
            angle_unit="deg",
        )
        for key in ("left_png", "right_png", "torso_png"):
            if paths.get(key):
                joint_tracking_plot_paths[f"episode_{ep:03d}_{key.replace('_png', '')}"] = paths[key]
        print(
            f"[{log_prefix}] joint tracking episode {ep}: {len(joint_track_current.steps)} steps -> "
            f"{paths.get('right_png', '')}, {paths.get('left_png', '')}"
            + (f", {paths['torso_png']}" if paths.get("torso_png") else "")
            + (f" / {paths.get('csv', paths.get('json'))}" if not paths.get("right_png") else "")
        )
        joint_track_traces.append(joint_track_current)
        joint_track_current = JointTrackingEpisodeTrace(
            episode_index=len(joint_track_traces),
            env_id=joint_track_env_id,
            joint_names=list(joint_track_traces[-1].joint_names),
        )

    while simulation_app.is_running():
        with torch.inference_mode():
            if policy_obs_recorder is not None:
                policy_obs_recorder.record(states, global_step=timestep)
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)
            states, rewards, terminated, truncated, infos = env.step(actions)

            if finger_gate_current is not None and len(finger_gate_traces) < finger_gate_episode_cap:
                eid = finger_gate_env_id
                done_e = bool(terminated[eid].item()) or bool(truncated[eid].item())
                snap = read_finger_insertion_from_env(
                    env,
                    env_id=eid,
                    num_fingers=args_cli.num_fingers,
                    use_episode_end_snapshot=done_e,
                )
                finger_gate_current.steps.append(
                    FingerInsertionStep(
                        step=len(finger_gate_current.steps),
                        global_step=timestep,
                        per_finger_soft_inside=snap.get("per_finger_soft_inside") or [0.0] * args_cli.num_fingers,
                        fingers_inside_soft_gate=float(snap.get("fingers_inside_soft_gate", 0.0)),
                        per_finger_inside_ellipse=snap.get("per_finger_inside_ellipse"),
                        per_finger_insert_margin=snap.get("per_finger_insert_margin"),
                    )
                )
                if done_e:
                    _finalize_finger_gate_episode(
                        terminated=bool(terminated[eid].item()),
                        truncated=bool(truncated[eid].item()),
                    )

            if policy_obs_recorder is not None:
                oid = policy_obs_recorder.env_id
                done_o = bool(terminated[oid].item()) or bool(truncated[oid].item())
                if done_o:
                    dest = policy_obs_recorder.finalize(
                        terminated=bool(terminated[oid].item()),
                        truncated=bool(truncated[oid].item()),
                    )
                    if dest is not None:
                        print(f"[{log_prefix}] policy observations -> {dest}")

            if joint_track_current is not None and len(joint_track_traces) < joint_track_episode_cap:
                eid = joint_track_env_id
                done_e = bool(terminated[eid].item()) or bool(truncated[eid].item())
                snap = read_joint_tracking_from_env(env, env_id=eid)
                if not joint_track_current.joint_names:
                    joint_track_current.joint_names = list(snap["joint_names"])
                joint_track_current.steps.append(
                    JointTrackingStep(
                        step=len(joint_track_current.steps),
                        global_step=timestep,
                        joint_names=list(snap["joint_names"]),
                        action=snap["action"],
                        q_cmd=snap["q_cmd"],
                        q_act=snap["q_act"],
                        q_err=snap["q_err"],
                        q_vel=snap["q_vel"],
                        q_policy=snap.get("q_policy"),
                        simulation_time=float(snap.get("simulation_time", 0.0)),
                        q_actual=snap.get("q_actual"),
                        q_error_cmd=snap.get("q_error_cmd"),
                        q_error_policy=snap.get("q_error_policy"),
                        applied_torque=snap.get("applied_torque"),
                        computed_torque=snap.get("computed_torque"),
                        torque_err=snap.get("torque_err"),
                        torque_limit_reached=snap.get("torque_limit_reached"),
                        velocity_limit_reached=snap.get("velocity_limit_reached"),
                        position_limit_reached=snap.get("position_limit_reached"),
                    )
                )
                if done_e:
                    _finalize_joint_track_episode(
                        terminated=bool(terminated[eid].item()),
                        truncated=bool(truncated[eid].item()),
                    )

            if on_after_step is not None:
                on_after_step(
                    timestep=timestep,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                )

            mask_update = 1 - torch.logical_or(terminated, truncated).float()
            returns += rewards * mask
            mask *= mask_update

            r_ev = rewards[:num_eval_envs]
            done_ev = torch.logical_or(terminated[:num_eval_envs], truncated[:num_eval_envs])
            episode_return_sum += r_ev
            done_flat = done_ev.squeeze(-1)
            if print_eval_episode_returns and done_flat.any():
                idx = done_flat.nonzero(as_tuple=False).view(-1)
                mean_episode_return = episode_return_sum[idx].mean().item()
                print(
                    f"[{log_prefix} eval episode end] timestep={timestep} "
                    f"mean_returns (over {idx.numel()} completed eval env(s))={mean_episode_return:.4f}"
                )
                for j in idx.tolist():
                    print(f"  eval_env_id={j} return={episode_return_sum[j, 0].item():.4f}")
            episode_return_sum *= 1.0 - done_ev.float()

            if timestep > 0 and timestep % ep_length == 0:
                if finger_gate_current is not None and len(finger_gate_traces) < finger_gate_episode_cap:
                    _finalize_finger_gate_episode(terminated=False, truncated=True)
                if joint_track_current is not None and len(joint_track_traces) < joint_track_episode_cap:
                    _finalize_joint_track_episode(terminated=False, truncated=True)
                if policy_obs_recorder is not None:
                    dest = policy_obs_recorder.finalize(terminated=False, truncated=True)
                    if dest is not None:
                        print(f"[{log_prefix}] policy observations -> {dest}")
                if on_periodic_hard_reset is not None:
                    on_periodic_hard_reset()
                states, infos = env.reset(hard=True)
                returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
                mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)
                episode_return_sum.zero_()

        if should_stop is not None and should_stop():
            break
        if stop_on_video_length and args_cli.video and timestep == args_cli.video_length:
            break
        if stop_on_recorder_caps and finger_gate_dir and len(finger_gate_traces) >= finger_gate_episode_cap:
            break
        if stop_on_recorder_caps and joint_track_dir and len(joint_track_traces) >= joint_track_episode_cap:
            break

        timestep += 1

    if finger_gate_current is not None and finger_gate_current.steps and len(finger_gate_traces) < finger_gate_episode_cap:
        _finalize_finger_gate_episode(terminated=False, truncated=False)
    if finger_gate_dir and finger_gate_traces:
        combined = os.path.join(finger_gate_dir, "all_episodes_finger_insertion_gate.csv")
        save_all_traces_csv(finger_gate_traces, combined, num_fingers=args_cli.num_fingers)
        print(f"[{log_prefix}] combined finger gate CSV ({len(finger_gate_traces)} episode(s)): {combined}")

    if joint_track_current is not None and joint_track_current.steps and len(joint_track_traces) < joint_track_episode_cap:
        _finalize_joint_track_episode(terminated=False, truncated=False)
    if policy_obs_recorder is not None and policy_obs_recorder.current.gt:
        dest = policy_obs_recorder.finalize(terminated=False, truncated=False)
        if dest is not None:
            print(f"[{log_prefix}] policy observations -> {dest}")
    if joint_track_dir and joint_track_traces:
        combined = os.path.join(joint_track_dir, "all_episodes_joint_tracking.csv")
        save_all_joint_tracking_csv(joint_track_traces, combined)
        print(f"[{log_prefix}] combined joint tracking CSV ({len(joint_track_traces)} episode(s)): {combined}")
        if joint_tracking_plot_paths:
            print(
                f"[{log_prefix}] joint tracking plots -> "
                f"{joint_track_dir}/left_arm_joint_states.png, "
                f"{joint_track_dir}/right_arm_joint_states.png"
            )

    return PlaybackResult(
        timestep=timestep,
        joint_tracking_plot_paths=joint_tracking_plot_paths,
        finger_gate_dir=finger_gate_dir,
        joint_track_dir=joint_track_dir,
        policy_obs_dir=policy_obs_dir,
    )


def finalize_play_session(session: PlaySession, playback: PlaybackResult | None = None) -> None:
    from play_output_utils import build_play_metadata, finalize_recorded_video, write_metadata_json

    args_cli = session.args_cli
    if args_cli.video:
        finalized = finalize_recorded_video(session.output_paths.evaluation_dir, session.output_paths.executed_at)
        if finalized is not None:
            print(f"[{session.log_prefix}] saved video -> {finalized}")

    plot_paths = playback.joint_tracking_plot_paths if playback is not None else None
    metadata = build_play_metadata(
        args_cli=args_cli,
        env=session.env,
        env_cfg=session.env_cfg,
        checkpoint_info=session.checkpoint_info,
        output_paths=session.output_paths,
        wandb_run_id=session.wandb_run_id,
        wandb_run_id_source=session.wandb_run_id_source,
        seed=session.agent_cfg["seed"],
        joint_tracking_plot_paths=plot_paths or None,
    )
    eval_csv = session.output_paths.evaluation_dir / "episode_metrics.csv"
    eval_json = session.output_paths.evaluation_dir / "evaluation_summary.json"
    if eval_csv.is_file():
        metadata["output"]["episode_metrics_csv"] = str(eval_csv)
    if eval_json.is_file():
        metadata["output"]["evaluation_summary_json"] = str(eval_json)
    if playback is not None and playback.policy_obs_dir:
        metadata["output"]["policy_observations_dir"] = playback.policy_obs_dir
    write_metadata_json(metadata, session.output_paths.metadata_file)
    print(f"[{session.log_prefix}] saved metadata -> {session.output_paths.metadata_file}")
    session.env.close()
