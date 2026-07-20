"""Analysis utilities for deformable-bracelet finger insertion gate sensitivity.

Gate definition (reach_deformable_bracelet):
    m_i = min(z_i - z_south, z_north - z_i)
    g_i = sigmoid(m_i / k),  k = 0.01 by default
    inserted at threshold t  <=>  g_i > t  (equivalently m_i > 0 when t = 0.5)
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from dressing_eval import DEFAULT_INSERTION_GATE_K, sigmoid_gate_from_margin


def _coerce_episode_list(results: Any) -> list[dict[str, Any]]:
    """Normalize eval outputs to a list of per-episode dicts."""
    if results is None:
        return []
    if isinstance(results, list):
        return [e for e in results if isinstance(e, dict)]
    if not isinstance(results, dict):
        return []

    for key in ("insertion_gate_episodes", "episodes", "episode_results"):
        if key in results and isinstance(results[key], list):
            out: list[dict[str, Any]] = []
            for ep in results[key]:
                if isinstance(ep, dict):
                    out.append(ep)
                else:
                    # DressingEpisodeData or similar dataclass
                    insertion_gate = getattr(ep, "insertion_gate", None)
                    out.append(
                        {
                            "episode_index": getattr(ep, "episode_index", None),
                            "strict_success": getattr(ep, "strict_success", None),
                            "final_inserted_fingers": getattr(ep, "final_inserted_fingers", None),
                            "final": insertion_gate,
                        }
                    )
            return out
    return []


def _final_block(ep: dict[str, Any]) -> dict[str, Any]:
    if "final" in ep and isinstance(ep["final"], dict):
        return ep["final"]
    return ep


def _resolve_g_fingers(final: dict[str, Any], k: float) -> list[float]:
    if "g_fingers" in final:
        return [float(x) for x in final["g_fingers"]]
    margins = final.get("margin_fingers")
    if margins is not None:
        return [sigmoid_gate_from_margin(float(m), k) for m in margins]
    s = final.get("s_fingers")
    tau = final.get("tau", 0.0)
    if s is not None:
        return [sigmoid_gate_from_margin(float(si) - float(tau), k) for si in s]
    return []


def _resolve_margins(final: dict[str, Any], k: float) -> list[float]:
    if "margin_fingers" in final:
        return [float(x) for x in final["margin_fingers"]]
    g = final.get("g_fingers")
    if g is not None:
        # invert sigmoid only for logging fallback; prefer stored margins
        return [k * math.log(float(gi) / max(1.0 - float(gi), 1e-12)) for gi in g]
    s = final.get("s_fingers")
    tau = final.get("tau", 0.0)
    if s is not None:
        return [float(si) - float(tau) for si in s]
    return []


def flatten_finger_samples(
    episodes: Iterable[dict[str, Any]],
    *,
    k: float = DEFAULT_INSERTION_GATE_K,
) -> dict[str, list[float]]:
    """Flatten per-episode finger data into lists over all episodes × fingers."""
    margins: list[float] = []
    gates: list[float] = []
    for ep in episodes:
        final = _final_block(ep)
        m = _resolve_margins(final, k)
        g = _resolve_g_fingers(final, k)
        n = max(len(m), len(g))
        for i in range(n):
            mi = m[i] if i < len(m) else (k * math.log(g[i] / max(1.0 - g[i], 1e-12)) if i < len(g) else 0.0)
            gi = g[i] if i < len(g) else sigmoid_gate_from_margin(mi, k)
            margins.append(mi)
            gates.append(gi)
    return {"margin": margins, "gate": gates}


def per_episode_inserted_counts(
    episodes: Iterable[dict[str, Any]],
    *,
    k: float = DEFAULT_INSERTION_GATE_K,
) -> dict[str, list[int]]:
    """Per-episode inserted finger counts at thresholds 0.45, 0.50, 0.55."""
    out: dict[str, list[int]] = {"045": [], "050": [], "055": []}
    for ep in episodes:
        final = _final_block(ep)
        if all(k in final for k in ("inserted_045", "inserted_050", "inserted_055")):
            out["045"].append(int(final["inserted_045"]))
            out["050"].append(int(final["inserted_050"]))
            out["055"].append(int(final["inserted_055"]))
            continue
        g = _resolve_g_fingers(final, k)
        out["045"].append(sum(1 for gi in g if gi > 0.45))
        out["050"].append(sum(1 for gi in g if gi > 0.50))
        out["055"].append(sum(1 for gi in g if gi > 0.55))
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def summarize_method(
    method_name: str,
    episodes: list[dict[str, Any]],
    *,
    k: float = DEFAULT_INSERTION_GATE_K,
) -> dict[str, Any]:
    flat = flatten_finger_samples(episodes, k=k)
    counts = per_episode_inserted_counts(episodes, k=k)
    margins = flat["margin"]
    gates = flat["gate"]
    m_mean, m_std = _mean_std(margins)
    g_mean, g_std = _mean_std(gates)
    ambiguous_pct = 100.0 * sum(1 for g in gates if 0.45 <= g <= 0.55) / max(len(gates), 1)

    strict_count = sum(1 for ep in episodes if ep.get("strict_success") is True)
    num_episodes = len(episodes)

    row: dict[str, Any] = {
        "method": method_name,
        "num_episodes": num_episodes,
        "num_finger_samples": len(gates),
        "margin_mean": m_mean,
        "margin_std": m_std,
        "gate_mean": g_mean,
        "gate_std": g_std,
        "ambiguous_gate_pct_045_055": ambiguous_pct,
        "inserted_mean_045": _mean_std([float(x) for x in counts["045"]])[0],
        "inserted_std_045": _mean_std([float(x) for x in counts["045"]])[1],
        "inserted_mean_050": _mean_std([float(x) for x in counts["050"]])[0],
        "inserted_std_050": _mean_std([float(x) for x in counts["050"]])[1],
        "inserted_mean_055": _mean_std([float(x) for x in counts["055"]])[0],
        "inserted_std_055": _mean_std([float(x) for x in counts["055"]])[1],
        "strict_success_count": strict_count,
        "strict_success_rate": strict_count / num_episodes if num_episodes else 0.0,
        "k": k,
    }
    return row


def _plot_margin_histogram(
    results_by_method: dict[str, Any],
    save_dir: str,
    *,
    k: float,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, results in results_by_method.items():
        episodes = _coerce_episode_list(results)
        margins = flatten_finger_samples(episodes, k=k)["margin"]
        if margins:
            ax.hist(margins, bins=30, alpha=0.5, label=method, density=True)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1, label="m=0 (g=0.5)")
    ax.set_xlabel("Margin m_i = s_i - tau  [m]")
    ax.set_ylabel("Density")
    ax.set_title("Final-step insertion margin by method")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, "margin_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_gate_histogram(
    results_by_method: dict[str, Any],
    save_dir: str,
    *,
    k: float,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, results in results_by_method.items():
        episodes = _coerce_episode_list(results)
        gates = flatten_finger_samples(episodes, k=k)["gate"]
        if gates:
            ax.hist(gates, bins=30, alpha=0.5, label=method, density=True)
    ax.axvspan(0.45, 0.55, color="gray", alpha=0.15, label="ambiguous [0.45, 0.55]")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold 0.5")
    ax.set_xlabel("Gate value g_i")
    ax.set_ylabel("Density")
    ax.set_title("Final-step insertion gate by method")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, "gate_value_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_threshold_sensitivity(
    summaries: list[dict[str, Any]],
    save_dir: str,
) -> str:
    methods = [s["method"] for s in summaries]
    thresholds = ("045", "050", "055")
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, th in enumerate(thresholds):
        means = [s[f"inserted_mean_{th}"] for s in summaries]
        stds = [s[f"inserted_std_{th}"] for s in summaries]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=f"g > 0.{th[1:]}")

    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Mean final inserted fingers")
    ax.set_title("Threshold sensitivity (final step)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, "threshold_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_summary_csv(summaries: list[dict[str, Any]], save_dir: str) -> str:
    path = os.path.join(save_dir, "insertion_gate_summary.csv")
    if not summaries:
        return path
    fieldnames = list(summaries[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return path


def analyze_insertion_gate_sensitivity(
    results_by_method: dict[str, Any],
    save_dir: str,
    *,
    k: float = DEFAULT_INSERTION_GATE_K,
) -> dict[str, Any]:
    """Analyze and plot final-step insertion gate sensitivity across ablation methods.

    Args:
        results_by_method: ``{"Full": eval_dict_or_episode_list, ...}``
        save_dir: Directory for CSV and PNG outputs.
        k: Gate temperature (must match env ``insertion_gate_temperature``).

    Returns:
        Dict with ``summaries`` list and paths to saved artifacts.
    """
    os.makedirs(save_dir, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for method, results in results_by_method.items():
        episodes = _coerce_episode_list(results)
        summaries.append(summarize_method(method, episodes, k=k))

    csv_path = write_summary_csv(summaries, save_dir)
    margin_path = _plot_margin_histogram(results_by_method, save_dir, k=k)
    gate_path = _plot_gate_histogram(results_by_method, save_dir, k=k)
    threshold_path = _plot_threshold_sensitivity(summaries, save_dir)

    print("\n" + "=" * 60)
    print("Insertion gate sensitivity analysis")
    print("=" * 60)
    print(f"k = {k}  (g_i = sigmoid(m_i / k); g_i > 0.5  <=>  m_i > 0)")
    for s in summaries:
        print(
            f"\n{s['method']}: episodes={s['num_episodes']} "
            f"margin={s['margin_mean']:.4f}±{s['margin_std']:.4f} m  "
            f"gate={s['gate_mean']:.3f}±{s['gate_std']:.3f}  "
            f"ambiguous={s['ambiguous_gate_pct_045_055']:.1f}%"
        )
        print(
            f"  inserted @0.45/0.50/0.55: "
            f"{s['inserted_mean_045']:.2f}±{s['inserted_std_045']:.2f} / "
            f"{s['inserted_mean_050']:.2f}±{s['inserted_std_050']:.2f} / "
            f"{s['inserted_mean_055']:.2f}±{s['inserted_std_055']:.2f}"
        )
        if s["num_episodes"]:
            print(
                f"  strict success: {s['strict_success_count']}/{s['num_episodes']} "
                f"({100.0 * s['strict_success_rate']:.1f}%)"
            )
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {margin_path}")
    print(f"Saved: {gate_path}")
    print(f"Saved: {threshold_path}")
    print("=" * 60 + "\n")

    return {
        "summaries": summaries,
        "insertion_gate_summary_csv": csv_path,
        "margin_histogram_png": margin_path,
        "gate_value_histogram_png": gate_path,
        "threshold_sensitivity_png": threshold_path,
    }


def load_eval_results_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
