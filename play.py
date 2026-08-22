# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained RL agent with multimodal_rl.

Author: Elle Miller
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from play_common import add_play_common_args

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
add_play_common_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from play_common import finalize_play_session, run_playback_loop, setup_play_session


def main():
    """Play a trained RL agent from a checkpoint."""
    session = setup_play_session(parser, args_cli, log_prefix="play")
    playback = run_playback_loop(session, simulation_app)
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
