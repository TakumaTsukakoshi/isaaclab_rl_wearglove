# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug rollout: compare baseline vs fixed-target / no-Shadow-collision modes with the same checkpoint."""

import argparse
import os
import sys
import traceback

from shadow_debug_cli import add_shadow_debug_cli_args
from ee_stretch_cli import add_ee_stretch_cli_args

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Debug rollout for ReachDeformableBracelet (Shadow Hand causal analysis)."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during rollout.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_dir", type=str, default=None, help="Directory to save recorded videos.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel environments (use 1 for GUI markers).")
parser.add_argument("--task", type=str, default="AIREC_Reach_Deformable_Bracelet")
parser.add_argument("--checkpoint", type=str, required=True, help="Trained policy checkpoint (.pt).")
parser.add_argument("--agent_cfg", type=str, default=None, help="Optional agent YAML name under tasks/airec/agents/.")
parser.add_argument("--seed", type=int, default=0)
add_shadow_debug_cli_args(parser, default_debug_target_mode="baseline")
add_ee_stretch_cli_args(parser)
parser.add_argument("--episodes", type=int, default=1, help="Number of full episodes to run.")
parser.add_argument(
    "--compare-with",
    type=str,
    default=None,
    help="Optional second .pt log; print side-by-side summary vs this run.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.evaluation_mode = True
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from common_utils import LOG_PATH, get_unwrapped_env, make_env, make_models, set_seed, update_env_cfg  # noqa: E402
from isaaclab.utils import update_dict  # noqa: E402
from isaaclab_tasks.utils.hydra import register_task_to_hydra  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG  # noqa: E402
from multimodal_rl.tools.writer import Writer  # noqa: E402
from tasks.airec.shadow_hand_debug import compare_debug_logs  # noqa: E402


def main() -> None:
    env_cfg, agent_cfg = register_task_to_hydra(args_cli.task, "skrl_cfg_entry_point")
    if args_cli.agent_cfg is not None:
        specialised_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent_cfg)
        agent_cfg = update_dict(agent_cfg, specialised_cfg)

    agent_cfg["seed"] = args_cli.seed
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    if args_cli.video_dir is None:
        args_cli.video_dir = agent_cfg["experiment"].get("video_dir") or os.path.join(LOG_PATH, "videos")
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    save_log = getattr(args_cli, "save_debug_log", None)
    if save_log:
        os.makedirs(os.path.dirname(os.path.abspath(save_log)) or ".", exist_ok=True)
    if getattr(args_cli, "debug_ee_stretch_log", False):
        os.makedirs(getattr(args_cli, "debug_ee_stretch_log_dir", "logs/ee_stretch_debug"), exist_ok=True)

    writer = Writer(agent_cfg, play=True)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    print(
        f"[debug_rollout] creating env: num_envs={env_cfg.scene.num_envs} "
        f"num_eval_envs={env_cfg.num_eval_envs} debug_target_mode={env_cfg.debug_target_mode}"
    )
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    dtype = torch.float32
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
    agent.load(os.path.abspath(args_cli.checkpoint))

    unwrapped = get_unwrapped_env(env)
    print(
        f"[debug_rollout] mode={unwrapped.cfg.debug_target_mode} "
        f"num_envs={env.num_envs} seed={args_cli.seed} "
        "(targets are env-local; markers use +env_origins for world display)"
    )

    ep_length = unwrapped.max_episode_length - 1
    states, _ = env.reset(hard=True)
    global_step = 0

    for ep in range(args_cli.episodes):
        if ep > 0:
            states, _ = env.reset(hard=True)
            global_step = 0
        for _ in range(ep_length):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                z = encoder(states)
                actions, _, _ = agent.policy.act(z, deterministic=True)
                states, rewards, terminated, truncated, _ = env.step(actions)
            global_step += 1
            if (terminated | truncated).any():
                break

    if getattr(unwrapped, "_debug_rollout_logger", None) is not None:
        unwrapped._debug_rollout_logger.save_final()

    if hasattr(unwrapped, "finalize_ee_stretch_debug"):
        unwrapped.finalize_ee_stretch_debug()

    env.close()
    log_path = getattr(unwrapped.cfg, "debug_rollout_log_path", None)
    if log_path and os.path.isfile(log_path):
        print(f"[debug_rollout] Wrote log: {log_path}")
        if args_cli.compare_with and os.path.isfile(args_cli.compare_with):
            print(compare_debug_logs(args_cli.compare_with, log_path))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
