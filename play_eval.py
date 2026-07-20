# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained RL agent with multimodal_rl.

Author: Elle Miller 
"""


import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
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
    "--dressing-eval",
    action="store_true",
    default=False,
    help="Run dressing-task evaluation (strict success, fingers, wrist distance) and exit.",
)
parser.add_argument(
    "--object-name",
    type=str,
    default=None,
    help="Label for evaluation tables (default: experiment name or object_type).",
)
parser.add_argument(
    "--object-type",
    type=str,
    default=None,
    help="Object category for tables (default: env cfg object_type).",
)
parser.add_argument(
    "--max-episodes",
    type=int,
    default=50,
    help="Stop dressing evaluation after this many completed episodes.",
)
parser.add_argument(
    "--num-fingers",
    type=int,
    default=5,
    help="Number of fingertips for insertion reporting.",
)
parser.add_argument(
    "--eval-save-dir",
    type=str,
    default=None,
    help="With --dressing-eval: save evaluation JSON (includes per-episode insertion gates).",
)
parser.add_argument(
    "--insertion-gate-analysis-dir",
    type=str,
    default=None,
    help="With --dressing-eval: run insertion gate sensitivity analysis into this directory (single method).",
)
parser.add_argument(
    "--record-finger-insertion-gate",
    nargs="?",
    const="finger_insertion_gate_records",
    default=None,
    help=(
        "Record per_finger_soft_inside for --finger-gate-env-id each step. "
        "Optional DIR (default: finger_insertion_gate_records). Writes JSON + CSV per episode."
    ),
)
parser.add_argument(
    "--plot-finger-insertion-gate",
    type=str,
    default=None,
    help="Alias for --record-finger-insertion-gate DIR (also saves PNG plots unless --finger-gate-no-plots).",
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
    "--debug-joints",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Print policy action vs joint_pos_cmd vs measured joint_pos (overrides env cfg). "
        "Reach bracelet / deformable bracelet tasks support this via _debug_print_joint_cmd_vs_actual."
    ),
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
# Rendering options (useful for RTX5090 and similar GPUs)
parser.add_argument(
    "--renderer", type=str, default="PathTracing", choices=["RayTracedLighting", "PathTracing"], help="Renderer to use."
)
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1, help="Number of samples per pixel per frame.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
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

from dressing_eval import evaluate_dressing_rollouts, resolve_num_eval_envs, save_dressing_eval_results
from finger_insertion_gate_viz import (
    FingerInsertionEpisodeTrace,
    FingerInsertionStep,
    finalize_finger_gate_episode,
    read_finger_insertion_from_env,
    save_all_traces_csv,
)
from insertion_gate_analysis import analyze_insertion_gate_sensitivity
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.tools.writer import Writer


def main():
    """Play a trained RL agent from a checkpoint.

    Loads a checkpoint and runs the agent in the environment, optionally recording videos.
    """
    # Parse configuration
    env_cfg, agent_cfg = register_task_to_hydra(args_cli.task, "skrl_cfg_entry_point")

    if args_cli.agent_cfg is not None:
        specialised_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent_cfg)
        agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    # Set seed (important for seed-deterministic runs)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    if args_cli.video_dir is None:
        args_cli.video_dir = agent_cfg["experiment"].get("video_dir") or os.path.join(LOG_PATH, "videos")
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    # Update the environment config
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    if getattr(env_cfg, "debug_joint_cmd_vs_actual", False):
        print(
            "[play_eval] joint debug ON: "
            f"env_id={getattr(env_cfg, 'debug_joint_print_env_id', 0)}, "
            f"interval={getattr(env_cfg, 'debug_joint_print_interval', 10)}"
        )

    # Setup logging
    writer = Writer(agent_cfg, play=True)

    # Make environment (order: gymnasium Env -> FrameStack -> IsaacLab)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    # Setup models
    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    # Configure and instantiate PPO agent
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

    # Load checkpoint
    if not args_cli.checkpoint:
        parser.error("Missing checkpoint path. Use --checkpoint /path/to/best_agent.pt")
    try:
        resume_path = resolve_checkpoint_path(args_cli.checkpoint)
    except (ValueError, FileNotFoundError) as err:
        parser.error(str(err))
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    load_play_checkpoint(agent, resume_path)
    modules = torch.load(resume_path, map_location=env.device)
    if isinstance(modules, dict):
        for name in modules.keys():
            print(f"  - {name}")

    # Reset environment
    timestep = 0
    ep_length = env.env.unwrapped.max_episode_length - 1

    returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
    mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)

    num_eval_envs = resolve_num_eval_envs(env, int(agent_cfg["trainer"]["num_eval_envs"]))
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
        finger_gate_current = FingerInsertionEpisodeTrace(
            episode_index=0, env_id=finger_gate_env_id
        )
        print(
            f"[play_eval] recording finger insertion soft gate: env_id={finger_gate_env_id} -> {finger_gate_dir}"
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
            f"[play_eval] finger gate episode {ep}: {len(finger_gate_current.steps)} steps -> "
            f"{paths.get('csv', paths.get('json'))}"
        )
        finger_gate_traces.append(finger_gate_current)
        finger_gate_current = FingerInsertionEpisodeTrace(
            episode_index=len(finger_gate_traces), env_id=finger_gate_env_id
        )

    if args_cli.dressing_eval:
        exp_name = (agent_cfg.get("experiment") or {}).get("experiment_name")
        result = evaluate_dressing_rollouts(
            env,
            agent,
            encoder,
            simulation_app=simulation_app,
            checkpoint_path=resume_path,
            environment_config=env_cfg,
            object_name=args_cli.object_name or exp_name,
            object_type=args_cli.object_type or getattr(env_cfg, "object_type", None),
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
        env.close()
        return

    # Simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # Agent stepping
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)

            # Environment stepping
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
                if bool(terminated[eid].item()) or bool(truncated[eid].item()):
                    _finalize_finger_gate_episode(
                        terminated=bool(terminated[eid].item()),
                        truncated=bool(truncated[eid].item()),
                    )

            # Compute evaluation rewards
            mask_update = 1 - torch.logical_or(terminated, truncated).float()

            # Update evaluation metrics
            returns += rewards * mask
            mask *= mask_update

            # Per-episode returns for the eval env slice (same indices as training)
            r_ev = rewards[:num_eval_envs]
            done_ev = torch.logical_or(terminated[:num_eval_envs], truncated[:num_eval_envs])
            episode_return_sum += r_ev
            done_flat = done_ev.squeeze(-1)
            if print_eval_episode_returns and done_flat.any():
                idx = done_flat.nonzero(as_tuple=False).view(-1)
                mean_episode_return = episode_return_sum[idx].mean().item()
                print(
                    f"[play eval episode end] timestep={timestep} "
                    f"mean_returns (over {idx.numel()} completed eval env(s))={mean_episode_return:.4f}"
                )
                for j in idx.tolist():
                    print(f"  eval_env_id={j} return={episode_return_sum[j, 0].item():.4f}")
            episode_return_sum *= (1.0 - done_ev.float())

            # Manually reset eval episodes every ep_length
            if timestep > 0 and timestep % ep_length == 0:
                if finger_gate_current is not None and len(finger_gate_traces) < finger_gate_episode_cap:
                    _finalize_finger_gate_episode(terminated=False, truncated=True)
                mean_eval_return = returns.mean().item()
                states, infos = env.reset(hard=True)

                returns = torch.zeros(size=(env.num_envs, 1), device=env.device)
                mask = torch.Tensor([[1] for _ in range(env.num_envs)]).to(env.device)
                episode_return_sum.zero_()

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if finger_gate_dir and len(finger_gate_traces) >= finger_gate_episode_cap:
            break

        timestep += 1

    if finger_gate_current is not None and finger_gate_current.steps and len(finger_gate_traces) < finger_gate_episode_cap:
        _finalize_finger_gate_episode(terminated=False, truncated=False)

    if finger_gate_dir and finger_gate_traces:
        combined = os.path.join(finger_gate_dir, "all_episodes_finger_insertion_gate.csv")
        save_all_traces_csv(finger_gate_traces, combined, num_fingers=args_cli.num_fingers)
        print(f"[play_eval] combined finger gate CSV ({len(finger_gate_traces)} episode(s)): {combined}")

    # Close the simulator
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        raise
    finally:
        print("CLOSING")
        simulation_app.close()
