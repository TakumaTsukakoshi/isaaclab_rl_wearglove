#!/usr/bin/env python3
"""Compare insertion-gate sensitivity across Deformable Bracelet ablation eval runs.

Run dressing evaluation for each checkpoint (saves JSON), then analyze:

    python play_eval.py --task AIREC_Reach_Deformable_Bracelet \\
        --checkpoint /path/to/full.pt --dressing-eval \\
        --eval-save-dir log_eval/ablation/full_eval.json --headless

    python analyze_insertion_gate_ablation.py \\
        --full log_eval/ablation/full_eval.json \\
        --no-curriculum log_eval/ablation/no_curriculum_eval.json \\
        --no-finger-gate log_eval/ablation/no_finger_gate_eval.json \\
        --save-dir log_eval/ablation/gate_analysis
"""

from __future__ import annotations

import argparse
import os

from insertion_gate_analysis import analyze_insertion_gate_sensitivity, load_eval_results_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze final-step finger insertion gate sensitivity for ablation methods."
    )
    parser.add_argument(
        "--full",
        type=str,
        required=True,
        help="JSON from play_eval --dressing-eval --eval-save-dir (Full method).",
    )
    parser.add_argument(
        "--no-curriculum",
        type=str,
        required=True,
        help="JSON for No Curriculum Learning ablation.",
    )
    parser.add_argument(
        "--no-finger-gate",
        type=str,
        required=True,
        help="JSON for No Finger Gate ablation.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="log_eval/insertion_gate_analysis",
        help="Output directory for CSV and PNG plots.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.01,
        help="Gate temperature k in g_i = sigmoid(m_i / k).",
    )
    args = parser.parse_args()

    results_by_method = {
        "Full": load_eval_results_json(os.path.abspath(args.full)),
        "No Curriculum Learning": load_eval_results_json(os.path.abspath(args.no_curriculum)),
        "No Finger Gate": load_eval_results_json(os.path.abspath(args.no_finger_gate)),
    }
    analyze_insertion_gate_sensitivity(
        results_by_method,
        os.path.abspath(args.save_dir),
        k=args.k,
    )


if __name__ == "__main__":
    main()
