"""Dressing-task rollout evaluation for play / checkpoint evaluation.

Aggregates strict success, inserted finger count at episode end (inside opening Y-Z ellipse), and minimum wrist distance
from Isaac Lab envs that expose reach / wear bracelet metrics (e.g.
:class:`~tasks.airec.reach_deformable_bracelet.ReachDeformableBraceletEnv`).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import torch

# Matches ReachDeformableBraceletEnvCfg.insertion_gate_temperature: g_i = sigmoid(m_i / k).
DEFAULT_INSERTION_GATE_K = 0.01
# Normalized ellipse in Y-Z: inside when ellipse_value <= threshold (default unit ellipse).
DEFAULT_OPENING_ELLIPSE_THRESHOLD = 1.0
FINGER_NAMES = ("thumb", "fore", "middle", "ring", "pinky")


@dataclass
class DressingEpisodeData:
    """Per-episode metrics collected at episode end."""

    env_id: int
    episode_index: int
    strict_success: bool
    final_inserted_fingers: int
    min_wrist_distance_m: float
    terminated: bool
    truncated: bool
    task_success_flag: bool
    orientation_ok: bool
    no_entanglement: bool
    simulation_stable: bool
    insertion_gate: dict[str, Any] | None = None


@dataclass
class _RunningEpisode:
    min_wrist_distance_m: float = math.inf
    steps: int = 0


def _tensor_scalar(env: Any, name: str, env_id: int, default: float = 0.0) -> float:
    if not hasattr(env, name):
        return default
    value = getattr(env, name)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value[env_id].item())
    return float(value)


def _tensor_bool(env: Any, name: str, env_id: int, default: bool = False) -> bool:
    if not hasattr(env, name):
        return default
    value = getattr(env, name)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return bool(value[env_id].item())
    return bool(value)


def resolve_num_eval_envs(env, num_eval_envs: int | None = None) -> int:
    """Clamp eval-env slice to the number of parallel envs actually running.

    Training yaml may set ``trainer.num_eval_envs: 50`` while play uses ``--num_envs 1``.
    """
    n_envs = int(getattr(env, "num_envs", 1))
    if num_eval_envs is None:
        if hasattr(env, "num_eval_envs"):
            num_eval_envs = int(env.num_eval_envs)
        else:
            cfg = getattr(getattr(env, "unwrapped", None), "cfg", None)
            num_eval_envs = int(getattr(cfg, "num_eval_envs", n_envs)) if cfg is not None else n_envs
    requested = int(num_eval_envs)
    effective = min(requested, n_envs)
    if effective < requested:
        print(
            f"[dressing_eval] WARNING: config requests {requested} eval env(s) but only "
            f"{n_envs} parallel env(s) are running; evaluating env_id 0..{effective - 1}."
        )
    return effective


def finger_inside_opening_ellipse(ellipse_value: float, threshold: float = DEFAULT_OPENING_ELLIPSE_THRESHOLD) -> bool:
    """True iff fingertip lies inside the opening ellipse in the Y-Z plane (``ellipse_value <= threshold``)."""
    return float(ellipse_value) <= float(threshold)


def count_fingers_inside_ellipse(
    ellipse_values: list[float],
    num_fingers: int = 5,
    ellipse_threshold: float = DEFAULT_OPENING_ELLIPSE_THRESHOLD,
) -> int:
    """Number of fingers inside the opening ellipse at the final step."""
    return sum(
        1
        for v in ellipse_values[:num_fingers]
        if finger_inside_opening_ellipse(float(v), ellipse_threshold)
    )


def sigmoid_gate_from_margin(margin: float, k: float = DEFAULT_INSERTION_GATE_K) -> float:
    """``g = sigmoid(m / k)`` as in reach_deformable_bracelet; ``g > 0.5`` iff ``m > 0``."""
    x = margin / k
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def extract_insertion_gate_final_step(
    env: Any,
    env_id: int,
    *,
    k: float = DEFAULT_INSERTION_GATE_K,
    num_fingers: int = 5,
    ellipse_threshold: float = DEFAULT_OPENING_ELLIPSE_THRESHOLD,
) -> dict[str, Any]:
    """Final-step finger metrics (pre-reset snapshot from ``_get_dones``).

    **Primary eval insertion rule:** fingertip inside opening Y-Z ellipse
    (``per_finger_inside_ellipse`` / ``ellipse_value <= threshold``).

    Ellipse (same as ``_ellipse_soft_gate_zy`` / ``_per_finger_ellipse_value_zy``):
    ``((y-c_y)/r_y)^2 + ((z-c_z)/r_z)^2 <= 1`` with rim semi-axes from E/W and N/S goals.

    Also logs height margin and sigmoid gate for ablation sensitivity analysis.
    """
    margin_row = _episode_end_tensor(env, "per_finger_insert_margin", env_id)
    g_row = _episode_end_tensor(env, "per_finger_soft_inside", env_id)
    s_row = _episode_end_tensor(env, "per_finger_height_z", env_id)
    ell_row = _episode_end_tensor(env, "per_finger_ellipse_value", env_id)
    inside_ell_row = _episode_end_tensor(env, "per_finger_inside_ellipse", env_id)
    tau_south_t = _episode_end_tensor(env, "opening_south_z", env_id)
    tau_north_t = _episode_end_tensor(env, "opening_north_z", env_id)

    margins: list[float] = []
    g_vals: list[float] = []
    s_vals: list[float] = []
    ellipse_vals: list[float] = []
    inside_ellipse: list[bool] = []

    for i in range(num_fingers):
        m_i = float(margin_row[i].item()) if margin_row is not None and margin_row.numel() > i else 0.0
        if g_row is not None and g_row.numel() > i:
            g_i = float(g_row[i].item())
        else:
            g_i = sigmoid_gate_from_margin(m_i, k)
        s_i = float(s_row[i].item()) if s_row is not None and s_row.numel() > i else 0.0
        e_i = float(ell_row[i].item()) if ell_row is not None and ell_row.numel() > i else 0.0
        if inside_ell_row is not None and inside_ell_row.numel() > i:
            in_ell = bool(inside_ell_row[i].item() > 0.5)
        else:
            in_ell = finger_inside_opening_ellipse(e_i, ellipse_threshold)
        margins.append(m_i)
        g_vals.append(g_i)
        s_vals.append(s_i)
        ellipse_vals.append(e_i)
        inside_ellipse.append(in_ell)

    tau_south = float(tau_south_t.item()) if tau_south_t is not None else 0.0
    tau_north = float(tau_north_t.item()) if tau_north_t is not None else 0.0
    tau_mid = 0.5 * (tau_south + tau_north)
    inserted_by_ellipse = sum(inside_ellipse)

    def _count_gate(threshold: float) -> int:
        return sum(1 for g in g_vals if g > threshold)

    return {
        "s_fingers": s_vals,
        "tau": tau_mid,
        "tau_south": tau_south,
        "tau_north": tau_north,
        "margin_fingers": margins,
        "g_fingers": g_vals,
        "ellipse_value_fingers": ellipse_vals,
        "inside_ellipse": inside_ellipse,
        "inserted_by_ellipse": inserted_by_ellipse,
        "ellipse_threshold": ellipse_threshold,
        "k": k,
        "inserted_045": _count_gate(0.45),
        "inserted_050": _count_gate(0.50),
        "inserted_055": _count_gate(0.55),
    }


def dressing_episode_to_analysis_dict(ep: DressingEpisodeData) -> dict[str, Any]:
    """Convert one episode to the analysis-friendly dict structure."""
    out: dict[str, Any] = {
        "episode_index": ep.episode_index,
        "env_id": ep.env_id,
        "strict_success": ep.strict_success,
        "final_inserted_fingers": ep.final_inserted_fingers,
        "min_wrist_distance_m": ep.min_wrist_distance_m,
        "terminated": ep.terminated,
        "truncated": ep.truncated,
    }
    if ep.insertion_gate is not None:
        out["final"] = ep.insertion_gate
    return out


def save_dressing_eval_results(result: dict[str, Any], save_path: str) -> str:
    """Write evaluation dict (including per-episode insertion gates) to JSON."""
    save_path = os.path.abspath(save_path)
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    episodes = result.get("insertion_gate_episodes") or []
    payload = {k: v for k, v in result.items() if k != "episodes"}
    payload["insertion_gate_episodes"] = episodes
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return save_path


def sample_std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1); returns 0.0 for n < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def _episode_end_tensor(env: Any, live_name: str, env_id: int):
    """Prefer pre-reset snapshot written in ``_get_dones`` (see reach_*_bracelet envs)."""
    snap_name = f"_episode_end_{live_name}"
    if hasattr(env, snap_name):
        value = getattr(env, snap_name)
        if isinstance(value, torch.Tensor) and value.numel() > env_id:
            return value[env_id]
    if hasattr(env, live_name):
        value = getattr(env, live_name)
        if isinstance(value, torch.Tensor) and value.numel() > env_id:
            return value[env_id]
    return None


def extract_env_step_snapshot(env: Any, env_id: int) -> dict[str, Any]:
    """Read dressing-related tensors for episode finalization.

    After ``env.step()``, Isaac Lab has usually already reset done envs, so finger / success
    metrics must come from ``_episode_end_*`` buffers captured in ``_get_dones``.
    """
    snap: dict[str, Any] = {"env_id": env_id}

    task_success_t = _episode_end_tensor(env, "task_success", env_id)
    snap["task_success"] = bool(task_success_t.item()) if task_success_t is not None else _tensor_bool(
        env, "task_success", env_id
    )

    wrist_t = _episode_end_tensor(env, "wrist_center_euclidean_distance", env_id)
    snap["wrist_center_euclidean_distance_m"] = (
        float(wrist_t.item())
        if wrist_t is not None
        else _tensor_scalar(env, "wrist_center_euclidean_distance", env_id, default=math.inf)
    )

    for key in (
        "right_ee_thumb_angular_distance",
        "left_ee_pinky_angular_distance",
        "insert_depth",
        "inside_opening_soft",
        "wrist_inside_ellipse",
        "fingers_inside_soft_gate",
    ):
        if hasattr(env, key):
            snap[key] = _tensor_scalar(env, key, env_id)

    pf_row = _episode_end_tensor(env, "per_finger_soft_inside", env_id)
    if pf_row is not None and pf_row.ndim == 1:
        snap["per_finger_soft_inside"] = pf_row.detach().cpu().tolist()
    elif hasattr(env, "per_finger_soft_inside"):
        pf = env.per_finger_soft_inside[env_id]
        if isinstance(pf, torch.Tensor):
            snap["per_finger_soft_inside"] = pf.detach().cpu().tolist()

    term_log = getattr(env, "_term_log", None) or {}
    snap["term_log"] = {}
    for key, tensor in term_log.items():
        if isinstance(tensor, torch.Tensor) and tensor.numel() > env_id:
            snap["term_log"][key] = float(tensor[env_id].item())

    return snap


def check_strict_success(
    episode_data: dict[str, Any] | DressingEpisodeData,
    *,
    env_cfg: Any | None = None,
    wrist_success_threshold_m: float | None = None,
    max_thumb_angular_distance_rad: float = 1.4,
    max_pinky_angular_distance_rad: float = 0.8,
    min_inside_opening_soft: float = 0.25,
) -> tuple[bool, dict[str, bool]]:
    """Strict dressing success from env metrics (reuse task success + stability flags).

    Components:
    - **Target region**: ``task_success`` or wrist distance below ``bracelet_success_threshold``.
    - **Orientation**: thumb / pinky angular distance below thresholds (if logged).
    - **No entanglement**: no early termination from grasp-loss flags in ``_term_log``.
    - **Stable**: no ``term_out_of_reach`` / ``term_too_far`` at episode end.

    Returns:
        (success, component_flags dict)
    """
    if isinstance(episode_data, DressingEpisodeData):
        return (
            episode_data.strict_success,
            {
                "task_success": episode_data.task_success_flag,
                "orientation_ok": episode_data.orientation_ok,
                "no_entanglement": episode_data.no_entanglement,
                "simulation_stable": episode_data.simulation_stable,
            },
        )

    snap = episode_data
    threshold = wrist_success_threshold_m
    if threshold is None and env_cfg is not None:
        threshold = float(getattr(env_cfg, "bracelet_success_threshold", 0.01))
    if threshold is None:
        threshold = 0.01

    in_region = snap.get("task_success", False)
    if not in_region and math.isfinite(snap.get("wrist_center_euclidean_distance_m", math.inf)):
        in_region = snap["wrist_center_euclidean_distance_m"] <= threshold

    orientation_ok = True
    if "right_ee_thumb_angular_distance" in snap:
        orientation_ok = orientation_ok and snap["right_ee_thumb_angular_distance"] <= max_thumb_angular_distance_rad
    if "left_ee_pinky_angular_distance" in snap:
        orientation_ok = orientation_ok and snap["left_ee_pinky_angular_distance"] <= max_pinky_angular_distance_rad
    if "wrist_inside_ellipse" in snap:
        orientation_ok = orientation_ok and snap["wrist_inside_ellipse"] >= min_inside_opening_soft

    term = snap.get("term_log", {})
    no_entanglement = (
        term.get("term_grasp_right", 0.0) < 0.5 and term.get("term_grasp_left", 0.0) < 0.5
    )
    simulation_stable = (
        term.get("term_out_of_reach", 0.0) < 0.5 and term.get("term_too_far", 0.0) < 0.5
    )

    success = bool(in_region and orientation_ok and no_entanglement and simulation_stable)
    return success, {
        "task_success": bool(in_region),
        "orientation_ok": bool(orientation_ok),
        "no_entanglement": bool(no_entanglement),
        "simulation_stable": bool(simulation_stable),
    }


def count_inserted_fingers(
    step_data: dict[str, Any] | DressingEpisodeData,
    num_fingers: int = 5,
    inside_threshold: float = 0.5,
    ellipse_threshold: float = DEFAULT_OPENING_ELLIPSE_THRESHOLD,
) -> int:
    """Count inserted fingertips at the final step (inside opening Y-Z ellipse)."""
    if isinstance(step_data, DressingEpisodeData):
        return int(step_data.final_inserted_fingers)

    block = step_data.get("final", step_data)
    if "inserted_by_ellipse" in block:
        return int(block["inserted_by_ellipse"])
    if "inside_ellipse" in block:
        return sum(1 for b in block["inside_ellipse"][:num_fingers] if b)
    if "ellipse_value_fingers" in block:
        return count_fingers_inside_ellipse(
            block["ellipse_value_fingers"], num_fingers=num_fingers, ellipse_threshold=ellipse_threshold
        )

    pf = step_data.get("per_finger_soft_inside") or block.get("g_fingers")
    if pf is None:
        return 0
    return sum(1 for v in pf[:num_fingers] if float(v) >= inside_threshold)


def count_inserted_fingers_from_env(
    env: Any,
    env_id: int,
    num_fingers: int = 5,
    ellipse_threshold: float = DEFAULT_OPENING_ELLIPSE_THRESHOLD,
) -> int:
    """Insertion count at episode end: inside opening Y-Z ellipse."""
    gate = extract_insertion_gate_final_step(
        env, env_id, num_fingers=num_fingers, ellipse_threshold=ellipse_threshold
    )
    return int(gate["inserted_by_ellipse"])


def count_final_inserted_fingers(
    episode_data: dict[str, Any] | DressingEpisodeData,
    num_fingers: int = 5,
    ellipse_threshold: float = DEFAULT_OPENING_ELLIPSE_THRESHOLD,
    inside_threshold: float = 0.5,
) -> int:
    """Inserted finger count at episode end (inside opening ellipse)."""
    return count_inserted_fingers(
        episode_data,
        num_fingers=num_fingers,
        inside_threshold=inside_threshold,
        ellipse_threshold=ellipse_threshold,
    )


def compute_min_wrist_distance(episode_data: dict[str, Any] | DressingEpisodeData) -> float:
    """Minimum wrist-to-opening-center distance over the episode, in meters.

    Uses ``wrist_center_euclidean_distance`` (opening rim center to wrist goal).
    This is **center distance**, not mesh surface distance.
    """
    if isinstance(episode_data, DressingEpisodeData):
        return episode_data.min_wrist_distance_m
    return float(episode_data.get("min_wrist_distance_m", math.inf))


def min_wrist_distance_cm(episode_data: dict[str, Any] | DressingEpisodeData) -> float:
    """Same as :func:`compute_min_wrist_distance`, converted to centimeters."""
    return compute_min_wrist_distance(episode_data) * 100.0


def finalize_episode(
    env: Any,
    env_id: int,
    episode_index: int,
    running: _RunningEpisode,
    *,
    terminated: bool,
    truncated: bool,
    env_cfg: Any | None = None,
    num_fingers: int = 5,
    inside_threshold: float = 0.5,
) -> DressingEpisodeData:
    # ``step()`` returns after reset; use ``_episode_end_*`` snapshots from ``_get_dones``.
    snap = extract_env_step_snapshot(env, env_id)
    strict, components = check_strict_success(snap, env_cfg=env_cfg)
    gate_k = float(getattr(env_cfg, "insertion_gate_temperature", DEFAULT_INSERTION_GATE_K)) if env_cfg else DEFAULT_INSERTION_GATE_K
    ell_thr = float(getattr(env_cfg, "eval_opening_ellipse_threshold", DEFAULT_OPENING_ELLIPSE_THRESHOLD)) if env_cfg else DEFAULT_OPENING_ELLIPSE_THRESHOLD
    insertion_gate = extract_insertion_gate_final_step(
        env, env_id, k=gate_k, num_fingers=num_fingers, ellipse_threshold=ell_thr
    )
    final_inserted = int(insertion_gate["inserted_by_ellipse"])
    end_wrist_m = snap.get("wrist_center_euclidean_distance_m", math.inf)
    min_m = running.min_wrist_distance_m
    if math.isfinite(end_wrist_m):
        min_m = min(min_m, end_wrist_m) if math.isfinite(min_m) else end_wrist_m
    if not math.isfinite(min_m):
        min_m = end_wrist_m
    return DressingEpisodeData(
        env_id=env_id,
        episode_index=episode_index,
        strict_success=strict,
        final_inserted_fingers=final_inserted,
        min_wrist_distance_m=min_m,
        terminated=terminated,
        truncated=truncated,
        task_success_flag=components["task_success"],
        orientation_ok=components["orientation_ok"],
        no_entanglement=components["no_entanglement"],
        simulation_stable=components["simulation_stable"],
        insertion_gate=insertion_gate,
    )


def aggregate_dressing_results(
    episodes: list[DressingEpisodeData],
    *,
    object_name: str,
    object_type: str,
    max_episodes: int = 30,
    num_fingers: int = 5,
) -> dict[str, Any]:
    n = len(episodes)
    strict_count = sum(1 for ep in episodes if ep.strict_success)
    fingers = [float(ep.final_inserted_fingers) for ep in episodes]
    wrist_cm = [ep.min_wrist_distance_m * 100.0 for ep in episodes]

    f_mean = sum(fingers) / n if n else 0.0
    f_std = sample_std(fingers)
    w_mean = sum(wrist_cm) / n if n else 0.0
    w_std = sample_std(wrist_cm)

    return {
        "object_name": object_name,
        "object_type": object_type,
        "max_episodes_requested": max_episodes,
        "num_episodes": n,
        "strict_success_count": strict_count,
        "strict_success_rate": (strict_count / n) if n else 0.0,
        "final_inserted_fingers_mean": f_mean,
        "final_inserted_fingers_std": f_std,
        "min_wrist_distance_cm_mean": w_mean,
        "min_wrist_distance_cm_std": w_std,
        "num_fingers": num_fingers,
        "insertion_gate_episodes": [dressing_episode_to_analysis_dict(ep) for ep in episodes],
        "episodes": episodes,
    }


def format_latex_row(result: dict[str, Any]) -> str:
    """One LaTeX table row for the main performance table."""
    name = result["object_name"].replace("_", r"\_")
    n = result["num_episodes"]
    sc = result["strict_success_count"]
    f_mean = result["final_inserted_fingers_mean"]
    f_std = result["final_inserted_fingers_std"]
    nf = result["num_fingers"]
    w_mean = result["min_wrist_distance_cm_mean"]
    w_std = result["min_wrist_distance_cm_std"]
    return (
        f"{name} ({result['object_type']}) & "
        f"{sc}/{n} & "
        f"${f_mean:.1f} \\pm {f_std:.1f}$ / {nf} & "
        f"${w_mean:.2f} \\pm {w_std:.2f}$ \\\\"
    )


def print_evaluation_summary(result: dict[str, Any]) -> None:
    """Human-readable summary + LaTeX row to stdout."""
    n = result["num_episodes"]
    max_n = result["max_episodes_requested"]
    sc = result["strict_success_count"]
    f_mean = result["final_inserted_fingers_mean"]
    f_std = result["final_inserted_fingers_std"]
    nf = result["num_fingers"]
    w_mean = result["min_wrist_distance_cm_mean"]
    w_std = result["min_wrist_distance_cm_std"]

    print("\n" + "=" * 60)
    print("Dressing evaluation summary")
    print("=" * 60)
    print(f"Object: {result['object_name']} ({result['object_type']})")
    print(f"Episodes evaluated: {n}" + (f" (requested {max_n})" if n != max_n else ""))
    if n < max_n:
        print(
            f"WARNING: Only {n} episode(s) completed (requested {max_n}). "
            "Statistics use available episodes only."
        )
    print(f"Strict Success: {sc}/{n}")
    print(f"Final Inserted Fingers (inside opening ellipse, final step): {f_mean:.1f} ± {f_std:.1f} / {nf}")
    print(f"Minimum Wrist Distance: {w_mean:.2f} ± {w_std:.2f} cm")
    print("  (center distance: wrist goal vs opening center, not mesh surface)")
    print("\nLaTeX row:")
    print(format_latex_row(result))
    print("=" * 60 + "\n")


@torch.no_grad()
def run_dressing_evaluation_rollouts(
    env,
    agent,
    encoder,
    *,
    simulation_app,
    checkpoint_path: str | None = None,
    object_name: str = "wearable_objects",
    object_type: str = "deformable",
    max_episodes: int = 30,
    num_fingers: int = 5,
    finger_inside_threshold: float = 0.5,
    num_eval_envs: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Roll out a deterministic policy and aggregate dressing metrics.

    Args:
        env: Wrapped gym env (``IsaacLabWrapper`` + optional ``FrameStack``).
        agent: PPO agent with loaded policy.
        encoder: Observation encoder.
        simulation_app: Isaac Sim app (stop when ``max_episodes`` reached).
        checkpoint_path: Optional path string for logging only.
        object_name: Label for tables (e.g. ``wearable_objects``).
        object_type: ``deformable`` / ``rigid`` / etc.
        max_episodes: Stop after this many completed eval episodes.
        num_fingers: Expected fingertip count (5 for thumb…pinky).
        finger_inside_threshold: ``per_finger_soft_inside`` cutoff for insertion count at episode end.
        num_eval_envs: Eval env slice size (default: trainer config on env).
        verbose: Print per-episode lines.

    Returns:
        Aggregated metrics dict from :func:`aggregate_dressing_results`.
    """
    unwrapped = env.unwrapped
    env_cfg = getattr(unwrapped, "cfg", None)
    num_eval_envs = resolve_num_eval_envs(env, num_eval_envs)

    completed: list[DressingEpisodeData] = []
    running: dict[int, _RunningEpisode] = {i: _RunningEpisode() for i in range(num_eval_envs)}
    episode_counter = 0

    states, _ = env.reset(hard=True)
    timestep = 0

    if checkpoint_path and verbose:
        print(f"[dressing_eval] checkpoint: {checkpoint_path}")
    if verbose:
        print(
            f"[dressing_eval] object={object_name} type={object_type} "
            f"max_episodes={max_episodes} num_eval_envs={num_eval_envs}"
        )

    while simulation_app.is_running() and len(completed) < max_episodes:
        z = encoder(states)
        actions, _, _ = agent.policy.act(z, deterministic=True)
        states, _, terminated, truncated, _ = env.step(actions)

        for env_id in range(num_eval_envs):
            wrist_row = _episode_end_tensor(unwrapped, "wrist_center_euclidean_distance", env_id)
            if wrist_row is not None and (
                bool(terminated[env_id].item()) or bool(truncated[env_id].item())
            ):
                dist_m = float(wrist_row.item())
            else:
                dist_m = _tensor_scalar(
                    unwrapped, "wrist_center_euclidean_distance", env_id, default=math.inf
                )
            if math.isfinite(dist_m):
                running[env_id].min_wrist_distance_m = min(running[env_id].min_wrist_distance_m, dist_m)
            running[env_id].steps += 1

        done_ev = torch.logical_or(terminated[:num_eval_envs], truncated[:num_eval_envs])
        if done_ev.any():
            for env_id in done_ev.squeeze(-1).nonzero(as_tuple=False).view(-1).tolist():
                if len(completed) >= max_episodes:
                    break
                ep = finalize_episode(
                    unwrapped,
                    env_id,
                    episode_counter,
                    running[env_id],
                    terminated=bool(terminated[env_id].item()),
                    truncated=bool(truncated[env_id].item()),
                    env_cfg=env_cfg,
                    num_fingers=num_fingers,
                    inside_threshold=finger_inside_threshold,
                )
                completed.append(ep)
                episode_counter += 1
                running[env_id] = _RunningEpisode()
                if verbose:
                    ig = ep.insertion_gate or {}
                    inside = ig.get("inside_ellipse", [])
                    ell = ig.get("ellipse_value_fingers", [])
                    in_str = "".join("1" if b else "0" for b in inside[:num_fingers]) if inside else "n/a"
                    ell_str = ", ".join(f"{v:.2f}" for v in ell[:num_fingers]) if ell else "n/a"
                    print(
                        f"[dressing_eval] episode {ep.episode_index}: "
                        f"strict={ep.strict_success} final_fingers={ep.final_inserted_fingers}/{num_fingers} "
                        f"(inside_ellipse=[{in_str}] ellipse_val=[{ell_str}]) "
                        f"end={'term' if ep.terminated else 'timeout' if ep.truncated else '?'} "
                        f"min_wrist={ep.min_wrist_distance_m * 100:.2f} cm "
                        f"(env_id={env_id}, step={timestep})"
                    )

        timestep += 1

    result = aggregate_dressing_results(
        completed,
        object_name=object_name,
        object_type=object_type,
        max_episodes=max_episodes,
        num_fingers=num_fingers,
    )
    print_evaluation_summary(result)
    return result


def run_dressing_evaluation_rollouts_with_save(
    *args,
    eval_save_path: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Like :func:`run_dressing_evaluation_rollouts`, optionally writing JSON artifacts."""
    result = run_dressing_evaluation_rollouts(*args, **kwargs)
    if eval_save_path:
        path = save_dressing_eval_results(result, eval_save_path)
        print(f"[dressing_eval] saved results to {path}")
    return result


def evaluate_dressing_rollouts(
    env,
    agent,
    encoder,
    *,
    simulation_app=None,
    checkpoint_path: str | None = None,
    environment=None,
    environment_config=None,
    object_name: str | None = None,
    object_type: str | None = None,
    max_episodes: int = 30,
    num_fingers: int = 5,
    **kwargs,
) -> dict[str, Any]:
    """Public API: resolve names from env cfg and run rollouts."""
    unwrapped = env.unwrapped if environment is None else environment
    cfg = environment_config if environment_config is not None else getattr(unwrapped, "cfg", None)

    if object_type is None and cfg is not None:
        object_type = str(getattr(cfg, "object_type", "unknown"))
    if object_type is None:
        object_type = "unknown"

    if object_name is None:
        object_name = getattr(cfg, "experiment_name", None) or object_type

    if simulation_app is None:
        raise ValueError("simulation_app is required for evaluate_dressing_rollouts")

    return run_dressing_evaluation_rollouts(
        env,
        agent,
        encoder,
        simulation_app=simulation_app,
        checkpoint_path=checkpoint_path,
        object_name=object_name,
        object_type=object_type,
        max_episodes=max_episodes,
        num_fingers=num_fingers,
        **kwargs,
    )
