"""Shadow Hand debug CLI (no Isaac Sim imports — safe before AppLauncher)."""

from __future__ import annotations

import argparse


def add_shadow_debug_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_debug_target_mode: str | None = None,
) -> None:
    """CLI flags for ReachDeformableBracelet Shadow Hand debug (train / play / debug_rollout)."""
    parser.add_argument(
        "--debug-target-mode",
        type=str,
        default=default_debug_target_mode,
        choices=["baseline", "no_shadow_collision_fixed_targets", "no_shadow_actor_fixed_targets"],
        help=(
            "Shadow Hand debug mode. baseline=live Shadow Hand; "
            "no_shadow_collision_fixed_targets / no_shadow_actor_fixed_targets=remove Shadow Hand "
            "from scene and use fixed env-local targets from debug_fixed_targets (constants)."
        ),
    )
    parser.add_argument(
        "--save-debug-log",
        type=str,
        default=None,
        metavar="PATH",
        help="Enable per-step debug log and write PATH (.pt) at end of run.",
    )
    parser.add_argument(
        "--disable-shadow-bracelet-collision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="FilteredPairs: ShadowHand vs deformable Object (bracelet).",
    )
    parser.add_argument(
        "--disable-shadow-robot-collision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="FilteredPairs: ShadowHand vs AIREC hand/finger/palm links.",
    )
    parser.add_argument(
        "--show-debug-fixed-target-markers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Visualize fixed env-local targets (world = local + env_origins).",
    )
    parser.add_argument(
        "--debug-log-all-envs",
        action="store_true",
        default=False,
        help="Log every env each step (default: eval envs only when training with many envs).",
    )
