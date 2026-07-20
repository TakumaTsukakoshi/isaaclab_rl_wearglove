"""Record and plot finger insertion soft gates for a single env during play / animation.

Metrics from :class:`~tasks.airec.reach_deformable_bracelet.ReachDeformableBraceletEnv`:
- ``per_finger_soft_inside[i]`` = g_i = sigmoid(m_i / k)  (height margin through opening)
- ``fingers_inside_soft_gate`` = mean(g_i)
- ``per_finger_inside_ellipse[i]`` = inside opening Y-Z ellipse (optional overlay)
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

FINGER_NAMES = ("thumb", "fore", "middle", "ring", "pinky")
INSERTION_GATE_THRESHOLD = 0.5


@dataclass
class FingerInsertionStep:
    step: int
    global_step: int
    per_finger_soft_inside: list[float]
    fingers_inside_soft_gate: float
    per_finger_inside_ellipse: list[float] | None = None
    per_finger_insert_margin: list[float] | None = None


@dataclass
class FingerInsertionEpisodeTrace:
    episode_index: int
    env_id: int
    steps: list[FingerInsertionStep] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "env_id": self.env_id,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "steps": [asdict(s) for s in self.steps],
        }


def _row_to_list(tensor_row: torch.Tensor | None, n: int) -> list[float] | None:
    if tensor_row is None:
        return None
    return [float(tensor_row[i].item()) for i in range(min(n, tensor_row.numel()))]


def read_finger_insertion_from_env(
    env: Any,
    env_id: int = 0,
    num_fingers: int = 5,
    *,
    use_episode_end_snapshot: bool = False,
) -> dict[str, Any]:
    """Read insertion-related tensors from unwrapped env at the current control step.

    After ``env.step()``, terminated envs are already reset; set
    ``use_episode_end_snapshot=True`` on the final step of an episode so values
    come from ``_episode_end_*`` buffers cloned in ``_get_dones()``.
    """
    unwrapped = getattr(env, "unwrapped", env)
    out: dict[str, Any] = {"env_id": env_id}

    if use_episode_end_snapshot and hasattr(unwrapped, "_episode_end_per_finger_soft_inside"):
        pf = unwrapped._episode_end_per_finger_soft_inside[env_id]
        out["per_finger_soft_inside"] = _row_to_list(pf, num_fingers)
        if out["per_finger_soft_inside"] is not None:
            out["fingers_inside_soft_gate"] = float(sum(out["per_finger_soft_inside"]) / len(out["per_finger_soft_inside"]))
        if hasattr(unwrapped, "_episode_end_per_finger_inside_ellipse"):
            out["per_finger_inside_ellipse"] = _row_to_list(
                unwrapped._episode_end_per_finger_inside_ellipse[env_id], num_fingers
            )
        if hasattr(unwrapped, "_episode_end_per_finger_insert_margin"):
            out["per_finger_insert_margin"] = _row_to_list(
                unwrapped._episode_end_per_finger_insert_margin[env_id], num_fingers
            )
        return out

    if hasattr(unwrapped, "per_finger_soft_inside"):
        out["per_finger_soft_inside"] = _row_to_list(unwrapped.per_finger_soft_inside[env_id], num_fingers)
    if hasattr(unwrapped, "fingers_inside_soft_gate"):
        out["fingers_inside_soft_gate"] = float(unwrapped.fingers_inside_soft_gate[env_id].item())
    if hasattr(unwrapped, "per_finger_inside_ellipse"):
        out["per_finger_inside_ellipse"] = _row_to_list(unwrapped.per_finger_inside_ellipse[env_id], num_fingers)
    if hasattr(unwrapped, "per_finger_insert_margin"):
        out["per_finger_insert_margin"] = _row_to_list(unwrapped.per_finger_insert_margin[env_id], num_fingers)

    return out


def plot_finger_insertion_episode(
    trace: FingerInsertionEpisodeTrace,
    save_path: str,
    *,
    num_fingers: int = 5,
    gate_threshold: float = INSERTION_GATE_THRESHOLD,
    plot_per_finger: bool = False,
    show_ellipse: bool = False,
    title: str | None = None,
) -> str:
    """Plot insertion soft gate vs episode step for one rollout.

    By default plots only ``fingers_inside_soft_gate`` (mean over fingers). Set
    ``plot_per_finger=True`` to overlay each ``per_finger_soft_inside`` curve.
    """
    if not trace.steps:
        raise ValueError("empty episode trace")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    t = np.arange(len(trace.steps))
    mean_gate = [s.fingers_inside_soft_gate for s in trace.steps]

    fig, ax = plt.subplots(figsize=(10, 4))

    if plot_per_finger:
        colors = plt.cm.tab10(np.linspace(0, 0.5, num_fingers))
        for i in range(num_fingers):
            gi = [s.per_finger_soft_inside[i] if i < len(s.per_finger_soft_inside) else 0.0 for s in trace.steps]
            ax.plot(
                t,
                gi,
                label=FINGER_NAMES[i] if i < len(FINGER_NAMES) else f"finger_{i}",
                color=colors[i],
                linewidth=1.8,
            )
        ax.plot(
            t,
            mean_gate,
            label="fingers_inside_soft_gate",
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
        if show_ellipse and trace.steps[0].per_finger_inside_ellipse is not None:
            for i in range(num_fingers):
                inside = [
                    float(s.per_finger_inside_ellipse[i]) if s.per_finger_inside_ellipse else 0.0
                    for s in trace.steps
                ]
                ax.plot(
                    t,
                    inside,
                    label=f"{FINGER_NAMES[i]} in ellipse" if i < len(FINGER_NAMES) else f"f{i} ellipse",
                    color=colors[i],
                    linestyle=":",
                    alpha=0.5,
                    linewidth=1.0,
                )
        ylabel = "Insertion soft gate"
        legend_ncol = 2
    else:
        ax.plot(t, mean_gate, label="fingers_inside_soft_gate", color="tab:blue", linewidth=2.0)
        ylabel = "fingers_inside_soft_gate"
        legend_ncol = 1

    ax.axhline(gate_threshold, color="gray", linestyle=":", linewidth=1.0, label=f"threshold = {gate_threshold}")

    end_tag = "term" if trace.terminated else ("timeout" if trace.truncated else "end")
    ax.set_xlabel("Step in episode")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=legend_ncol)
    if title is None:
        title = f"Episode {trace.episode_index} (env {trace.env_id}, {end_tag}, {len(trace.steps)} steps)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_episode_trace(trace: FingerInsertionEpisodeTrace, path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, indent=2)
    return path


def _finger_gate_csv_fieldnames(num_fingers: int) -> list[str]:
    names = [FINGER_NAMES[i] if i < len(FINGER_NAMES) else f"finger_{i}" for i in range(num_fingers)]
    return [
        "episode",
        "env_id",
        "step",
        "global_step",
        "fingers_inside_soft_gate",
        *[f"g_{n}" for n in names],
        *[f"in_ellipse_{n}" for n in names],
        *[f"insert_margin_{n}" for n in names],
    ]


def save_episode_trace_csv(
    trace: FingerInsertionEpisodeTrace,
    path: str,
    *,
    num_fingers: int = 5,
) -> str:
    """Write one row per control step (per-finger soft gates + mean)."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = _finger_gate_csv_fieldnames(num_fingers)
    finger_labels = [FINGER_NAMES[i] if i < len(FINGER_NAMES) else f"finger_{i}" for i in range(num_fingers)]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in trace.steps:
            row: dict[str, Any] = {
                "episode": trace.episode_index,
                "env_id": trace.env_id,
                "step": s.step,
                "global_step": s.global_step,
                "fingers_inside_soft_gate": s.fingers_inside_soft_gate,
            }
            for i, label in enumerate(finger_labels):
                gi = s.per_finger_soft_inside[i] if i < len(s.per_finger_soft_inside) else 0.0
                row[f"g_{label}"] = gi
                if s.per_finger_inside_ellipse is not None and i < len(s.per_finger_inside_ellipse):
                    row[f"in_ellipse_{label}"] = s.per_finger_inside_ellipse[i]
                if s.per_finger_insert_margin is not None and i < len(s.per_finger_insert_margin):
                    row[f"insert_margin_{label}"] = s.per_finger_insert_margin[i]
            writer.writerow(row)
    return path


def save_all_traces_csv(
    traces: list[FingerInsertionEpisodeTrace],
    path: str,
    *,
    num_fingers: int = 5,
) -> str:
    """Append all completed episodes into a single CSV."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = _finger_gate_csv_fieldnames(num_fingers)
    finger_labels = [FINGER_NAMES[i] if i < len(FINGER_NAMES) else f"finger_{i}" for i in range(num_fingers)]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trace in traces:
            for s in trace.steps:
                row: dict[str, Any] = {
                    "episode": trace.episode_index,
                    "env_id": trace.env_id,
                    "step": s.step,
                    "global_step": s.global_step,
                    "fingers_inside_soft_gate": s.fingers_inside_soft_gate,
                }
                for i, label in enumerate(finger_labels):
                    gi = s.per_finger_soft_inside[i] if i < len(s.per_finger_soft_inside) else 0.0
                    row[f"g_{label}"] = gi
                    if s.per_finger_inside_ellipse is not None and i < len(s.per_finger_inside_ellipse):
                        row[f"in_ellipse_{label}"] = s.per_finger_inside_ellipse[i]
                    if s.per_finger_insert_margin is not None and i < len(s.per_finger_insert_margin):
                        row[f"insert_margin_{label}"] = s.per_finger_insert_margin[i]
                writer.writerow(row)
    return path


def finalize_finger_gate_episode(
    trace: FingerInsertionEpisodeTrace,
    out_dir: str,
    *,
    num_fingers: int = 5,
    save_plots: bool = True,
    plot_per_finger: bool = False,
    show_ellipse: bool = False,
) -> dict[str, str]:
    """Save JSON + per-episode CSV; optionally PNG plot. Returns paths written."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ep = trace.episode_index
    paths: dict[str, str] = {}
    json_path = os.path.join(out_dir, f"episode_{ep:03d}_finger_insertion_gate.json")
    csv_path = os.path.join(out_dir, f"episode_{ep:03d}_finger_insertion_gate.csv")
    paths["json"] = save_episode_trace(trace, json_path)
    paths["csv"] = save_episode_trace_csv(trace, csv_path, num_fingers=num_fingers)
    if save_plots:
        png = os.path.join(out_dir, f"episode_{ep:03d}_finger_insertion_gate.png")
        paths["png"] = plot_finger_insertion_episode(
            trace,
            png,
            num_fingers=num_fingers,
            plot_per_finger=plot_per_finger,
            show_ellipse=show_ellipse,
        )
    return paths


@torch.no_grad()
def run_finger_insertion_gate_animation_logging(
    env,
    agent,
    encoder,
    *,
    simulation_app,
    env_id: int = 0,
    max_episodes: int = 10,
    num_fingers: int = 5,
    save_dir: str = "finger_insertion_gate_plots",
    plot_ellipse_overlay: bool = False,
    plot_per_finger: bool = False,
    verbose: bool = True,
) -> list[FingerInsertionEpisodeTrace]:
    """Roll out policy on ``env_id`` (default 0) and save one plot per completed episode.

    Use with ``--num_envs 1`` (or ensure ``env_id`` is in the eval slice) while animating / recording video.
    """
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    traces: list[FingerInsertionEpisodeTrace] = []
    current = FingerInsertionEpisodeTrace(episode_index=0, env_id=env_id)
    global_step = 0

    states, _ = env.reset(hard=True)
    if verbose:
        print(f"[finger_gate_viz] logging env_id={env_id} -> {save_dir} (max_episodes={max_episodes})")

    while simulation_app.is_running() and len(traces) < max_episodes:
        z = encoder(states)
        actions, _, _ = agent.policy.act(z, deterministic=True)
        states, _, terminated, truncated, _ = env.step(actions)

        done = bool(terminated[env_id].item()) or bool(truncated[env_id].item())
        snap = read_finger_insertion_from_env(
            env, env_id=env_id, num_fingers=num_fingers, use_episode_end_snapshot=done
        )
        gate_mean = float(snap.get("fingers_inside_soft_gate", 0.0))
        pf = snap.get("per_finger_soft_inside") or [0.0] * num_fingers
        pe = snap.get("per_finger_inside_ellipse")
        pm = snap.get("per_finger_insert_margin")

        current.steps.append(
            FingerInsertionStep(
                step=len(current.steps),
                global_step=global_step,
                per_finger_soft_inside=pf,
                fingers_inside_soft_gate=gate_mean,
                per_finger_inside_ellipse=pe,
                per_finger_insert_margin=pm,
            )
        )
        global_step += 1

        if done:
            current.terminated = bool(terminated[env_id].item())
            current.truncated = bool(truncated[env_id].item())
            paths = finalize_finger_gate_episode(
                current,
                save_dir,
                num_fingers=num_fingers,
                save_plots=True,
                plot_per_finger=plot_per_finger,
                show_ellipse=plot_ellipse_overlay,
            )
            if verbose:
                final_g = current.steps[-1].per_finger_soft_inside if current.steps else []
                print(
                    f"[finger_gate_viz] episode {current.episode_index}: {len(current.steps)} steps -> "
                    f"{paths.get('csv', paths.get('json'))}\n"
                    f"  final g_i={[f'{x:.2f}' for x in final_g]}"
                )
            traces.append(current)
            if len(traces) >= max_episodes:
                break
            current = FingerInsertionEpisodeTrace(episode_index=len(traces), env_id=env_id)

    if verbose:
        print(f"[finger_gate_viz] saved {len(traces)} episode plot(s) under {save_dir}")
    return traces
