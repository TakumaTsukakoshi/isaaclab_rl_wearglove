# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RL agent: motion-lock success, finger insertion, joint deviation.

Shares playback / CLI / output layout with ``play.py``. See ``bracelet_eval.py``
for metric definitions.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

from play_common import add_play_common_args, add_play_eval_args

parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from skrl.")
add_play_common_args(parser)
add_play_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from bracelet_eval import BraceletEvalCollector
from play_common import finalize_play_session, run_playback_loop, setup_play_session


def _run_legacy_dressing_eval(session) -> None:
    from dressing_eval import evaluate_dressing_rollouts, save_dressing_eval_results
    from insertion_gate_analysis import analyze_insertion_gate_sensitivity

    exp_name = (session.agent_cfg.get("experiment") or {}).get("experiment_name")
    result = evaluate_dressing_rollouts(
        session.env,
        session.agent,
        session.encoder,
        simulation_app=simulation_app,
        checkpoint_path=session.resume_path,
        environment_config=session.env_cfg,
        object_name=args_cli.object_name or exp_name,
        object_type=args_cli.object_type or getattr(session.env_cfg, "object_type", None),
        max_episodes=args_cli.max_episodes,
        num_fingers=args_cli.num_fingers,
        num_eval_envs=None,
    )
    if args_cli.eval_save_dir:
        save_path = args_cli.eval_save_dir
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, f"{result.get('object_name', 'eval')}_dressing_eval.json")
        save_dressing_eval_results(result, os.path.abspath(save_path))
        print(f"[play_eval] saved dressing eval JSON to {os.path.abspath(save_path)}")
    if args_cli.insertion_gate_analysis_dir:
        method_label = args_cli.object_name or exp_name or "eval"
        analyze_insertion_gate_sensitivity(
            {method_label: result},
            os.path.abspath(args_cli.insertion_gate_analysis_dir),
        )


def main():
    """Play a trained RL agent and write bracelet evaluation metrics."""
    session = setup_play_session(parser, args_cli, log_prefix="play_eval")
    playback = None
    collector = None
    try:
        if bool(getattr(args_cli, "legacy_dressing_eval", False)):
            _run_legacy_dressing_eval(session)
            return

        collector = BraceletEvalCollector.from_session(session)
        playback = run_playback_loop(
            session,
            simulation_app,
            on_after_step=collector.on_after_step,
            on_periodic_hard_reset=collector.on_periodic_hard_reset,
            should_stop=collector.is_complete,
            stop_on_video_length=False,
            stop_on_recorder_caps=False,
        )
    finally:
        if collector is not None:
            try:
                collector.finalize()
            except Exception as dump_err:
                print(f"[play_eval] WARNING: could not flush evaluation artifacts: {dump_err}")
        finalize_play_session(session, playback)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        raise
    finally:
        print("CLOSING")
        simulation_app.close()
