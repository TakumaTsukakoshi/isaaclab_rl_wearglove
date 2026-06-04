"""EE stretch debug CLI (no Isaac Sim imports — safe before AppLauncher)."""

from __future__ import annotations

import argparse


def add_ee_stretch_cli_args(parser: argparse.ArgumentParser) -> None:
    """CLI for Deformable Bracelet evaluation-only EE distance logging and clamp."""
    parser.add_argument(
        "--debug-ee-stretch-log",
        action="store_true",
        default=False,
        help="Enable per-step EE stretch debug log (evaluation / rollout only).",
    )
    parser.add_argument(
        "--debug-ee-stretch-log-dir",
        type=str,
        default="logs/ee_stretch_debug",
        help="Output directory for EE stretch CSV/NPZ/plots/summary.",
    )
    parser.add_argument(
        "--debug-ee-watch-distance",
        type=float,
        default=0.25,
        help="Analysis threshold (m): first step where ee_distance >= this value.",
    )
    parser.add_argument(
        "--debug-enable-ee-distance-clamp",
        action="store_true",
        default=False,
        help="Evaluation-only: suppress outward commands when ee_distance nears limit.",
    )
    parser.add_argument(
        "--debug-ee-clamp-limit",
        type=float,
        default=0.30,
        help="Safety upper limit (m) on ee_euclidean_distance (right_upper - left_upper norm).",
    )
    parser.add_argument(
        "--debug-ee-clamp-activation-distance",
        type=float,
        default=0.295,
        help="Start removing outward commands when ee_distance >= this (m).",
    )
    parser.add_argument(
        "--debug-ee-clamp-mode",
        type=str,
        default="remove_outward_relative_command",
        choices=["remove_outward_relative_command", "joint7_fallback"],
        help=(
            "remove_outward_relative_command: arm joint-pos cmd sum proxy (diagnostic, not Jacobian-exact); "
            "joint7_fallback: limit only left/right_arm_joint_7 cmd increments."
        ),
    )
    parser.add_argument(
        "--debug-target-object",
        type=str,
        default="deformable_bracelet",
        help="Restrict logging/clamp to this object category (only deformable_bracelet supported).",
    )
    parser.add_argument(
        "--debug-joint7-fallback-clamp",
        action="store_true",
        default=False,
        help="Use joint7-only fallback clamp (same as --debug-ee-clamp-mode joint7_fallback).",
    )
    parser.add_argument(
        "--debug-left-joint7-outward-direction",
        type=str,
        default="none",
        choices=["positive", "negative", "none"],
        help="Outward direction for left_arm_joint_7 cmd increment (joint7_fallback mode).",
    )
    parser.add_argument(
        "--debug-right-joint7-outward-direction",
        type=str,
        default="none",
        choices=["positive", "negative", "none"],
        help="Outward direction for right_arm_joint_7 cmd increment (joint7_fallback mode).",
    )
