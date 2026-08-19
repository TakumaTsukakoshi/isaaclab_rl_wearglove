"""Utility helpers shared between the RoTO training / inference scripts."""

import gymnasium as gym
import numpy as np
import os
import random
import torch

from multimodal_rl.models.encoder import Encoder
from multimodal_rl.models.running_standard_scaler import RunningStandardScaler
from multimodal_rl.rl.memories import Memory
from multimodal_rl.rl.policy_value import DeterministicValue, GaussianPolicy
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.rl.trainer import Trainer
from multimodal_rl.ssl.dynamics import ForwardDynamics
from multimodal_rl.ssl.reconstruction import Reconstruction
from multimodal_rl.wrappers.frame_stack import FrameStack
from multimodal_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper

# Import task modules to register environments
from tasks import airec  # noqa: F401

# Logging directory (change this to a custom path if desired)
LOG_PATH = os.getcwd()


def make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer):
    """Instantiate the optional self-supervised auxiliary task.

    Args:
        env: The gymnasium environment.
        rl_memory: Rollout memory buffer for RL.
        encoder: Encoder network.
        value: Value network.
        value_preprocessor: Value preprocessor.
        env_cfg: Environment configuration.
        agent_cfg: Agent configuration dictionary.
        writer: Writer for logging.

    Returns:
        SSL task instance or None if no SSL task is configured.
    """
    ssl_cfg = agent_cfg.get("ssl_task")
    if not ssl_cfg:
        return None

    rl_rollout = agent_cfg["agent"]["rollouts"]
    task_type = ssl_cfg.get("type")
    task_map = {
        "reconstruction": Reconstruction,
        "forward_dynamics": ForwardDynamics,
    }

    task_cls = task_map.get(task_type)
    if task_cls is None:
        return None

    return task_cls(
        ssl_cfg,
        rl_rollout,
        rl_memory,
        encoder,
        value,
        value_preprocessor,
        env,
        env_cfg,
        writer,
    )


def make_env(agent_cfg, env_cfg, writer, args_cli, *, video_name_prefix: str | None = None):
    """Create and wrap the Isaac Lab environment with gym + writer utilities.

    Args:
        agent_cfg: Agent configuration dictionary.
        env_cfg: Environment configuration.
        writer: Writer for logging.
        args_cli: Command-line arguments.

    Returns:
        Wrapped gymnasium environment.
    """
    # Update env_cfg with observation settings from agent_cfg
    # Note: configclass instances don't have .update() method, so we assign attributes directly
    if "observations" in agent_cfg:
        obs_cfg = agent_cfg["observations"]
        env_cfg.obs_list = obs_cfg.get("obs_list", getattr(env_cfg, "obs_list", []))
        env_cfg.obs_stack = obs_cfg.get("obs_stack", getattr(env_cfg, "obs_stack", 1))
        if "pixel_cfg" in obs_cfg:
            env_cfg.pixel_cfg = obs_cfg["pixel_cfg"]
        if "tactile_cfg" in obs_cfg:
            env_cfg.tactile_cfg = obs_cfg["tactile_cfg"]

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    obs, _ = env.reset()

    # Build observation space dictionary accounting for frame stacking
    gym_dict = {}
    for k, v in obs["policy"].items():
        obs_shape = list(v.shape)
        # Multiply the last dimension (channels) by the stack size
        obs_shape[-1] = obs_shape[-1] * env.unwrapped.obs_stack
        if k == "rgb":
            gym_dict[k] = gym.spaces.Box(
                low=0,
                high=255,
                shape=obs_shape[1:],
                dtype=np.uint8,
            )
        elif k == "depth":
            gym_dict[k] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=obs_shape[1:],
                dtype=np.float32,
            )
        else:
            gym_dict[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape[1:], dtype=np.float32)

    single_obs_space = gym.spaces.Dict()
    single_obs_space["policy"] = gym.spaces.Dict(gym_dict)
    obs_space = gym.vector.utils.batch_space(single_obs_space, env_cfg.scene.num_envs)
    single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env_cfg.num_actions,))
    action_space = gym.vector.utils.batch_space(single_action_space, env_cfg.scene.num_envs)
    env.unwrapped.set_spaces(single_obs_space, obs_space, single_action_space, action_space)

    # Wrap for video recording
    if args_cli.video:
        prefix = video_name_prefix if video_name_prefix is not None else "rl-video"
        video_kwargs = {
            "video_folder": writer.video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": prefix,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback to", writer.video_dir)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Apply frame stacking if needed
    if env.unwrapped.obs_stack > 1:
        env = FrameStack(env, obs_stack=env.unwrapped.obs_stack)

    # Apply Isaac Lab wrapper
    env = IsaacLabWrapper(env, env_cfg.num_eval_envs, obs_stack=env.unwrapped.obs_stack, debug=env_cfg.debug)
    return env


def make_models(env, env_cfg, agent_cfg, dtype):
    """Build encoder, policy, and value networks.

    Args:
        env: The gymnasium environment.
        env_cfg: Environment configuration.
        agent_cfg: Agent configuration dictionary.
        dtype: Data type for tensors.

    Returns:
        Tuple of (policy, value, encoder, value_preprocessor) networks.
    """
    observation_space = env.observation_space["policy"]
    action_space = env.action_space

    enc_type = agent_cfg.get("encoder", {}).get("type", "mlp")
    if enc_type == "wear_hepi":
        from tasks.airec.encoder_wear_hepi import WearHepiFusionEncoder

        encoder = WearHepiFusionEncoder(observation_space, action_space, env_cfg, agent_cfg, device=env.device)
    else:
        encoder = Encoder(observation_space, action_space, env_cfg, agent_cfg, device=env.device)
    z_dim = encoder.num_outputs

    policy = GaussianPolicy(
        z_dim=z_dim,
        observation_space=observation_space,
        action_space=env.action_space,
        device=env.device,
        **agent_cfg["policy"],
    )

    value = DeterministicValue(
        z_dim=z_dim,
        observation_space=observation_space,
        action_space=env.action_space,
        device=env.device,
        **agent_cfg["value"],
    )

    value_preprocessor = RunningStandardScaler(size=1, device=env.device, dtype=dtype, debug=env_cfg.debug)
    value.value_preprocessor = value_preprocessor

    print("*****Encoder*****")
    print(encoder)
    print("*****RL models*****")
    print(policy)
    print(value)
    print(value_preprocessor)

    return policy, value, encoder, value_preprocessor


def make_memory(env, env_cfg, size, num_envs):
    """Allocate rollout storage for PPO.

    Args:
        env: The gymnasium environment.
        env_cfg: Environment configuration.
        size: Size of the memory buffer (number of rollout steps).
        num_envs: Number of parallel environments.

    Returns:
        Memory buffer instance.
    """
    memory = Memory(
        memory_size=size,
        num_envs=num_envs,
        device=env.device,
        env_cfg=env_cfg,
    )
    return memory


def make_trainer(env, agent, agent_cfg, ssl_task=None, writer=None, print_eval_episode_returns: bool = False):
    """Create the high-level Trainer wrapper.

    Args:
        env: The gymnasium environment.
        agent: The RL agent (PPO).
        agent_cfg: Agent configuration dictionary.
        ssl_task: Optional self-supervised learning task.
        writer: Optional writer for logging.

    Returns:
        Trainer instance.
    """
    num_timesteps_M = agent_cfg["trainer"]["max_global_timesteps_M"]
    num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
    trainer = Trainer(
        env=env,
        agents=agent,
        agent_cfg=agent_cfg,
        num_timesteps_M=num_timesteps_M,
        num_eval_envs=num_eval_envs,
        ssl_task=ssl_task,
        writer=writer,
        print_eval_episode_returns=print_eval_episode_returns,
    )
    return trainer


def update_env_cfg(args_cli, env_cfg, agent_cfg):
    """Sync Isaac Lab config with CLI + agent overrides.

    Args:
        args_cli: Command-line arguments.
        env_cfg: Environment configuration to update.
        agent_cfg: Agent configuration dictionary.

    Returns:
        Updated environment configuration.
    """
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.debug = agent_cfg["experiment"]["debug"]

    # Override configurations with either config file or CLI args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.obs_list = agent_cfg["observations"]["obs_list"]
    env_cfg.num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
    env_cfg.obs_stack = agent_cfg["observations"]["obs_stack"]
    env_cfg.scene_mode = str(getattr(args_cli, "scene_mode", "full"))
    if env_cfg.scene_mode not in ("full", "free_space"):
        raise ValueError(
            f"scene_mode must be 'full' or 'free_space', got {env_cfg.scene_mode!r}"
        )

    if bool(getattr(args_cli, "disable_self_collision", False)):
        articulation_props = getattr(
            getattr(env_cfg.robot_cfg, "spawn", None), "articulation_props", None
        )
        if articulation_props is None:
            raise AttributeError(
                "--disable-self-collision requested, but robot_cfg.spawn.articulation_props is unavailable"
            )
        articulation_props.enabled_self_collisions = False

    if getattr(args_cli, "debug_joints", None) is not None:
        env_cfg.debug_joint_cmd_vs_actual = bool(args_cli.debug_joints)
    if getattr(args_cli, "debug_joint_env_id", None) is not None:
        env_cfg.debug_joint_print_env_id = int(args_cli.debug_joint_env_id)
    if getattr(args_cli, "debug_joint_interval", None) is not None:
        env_cfg.debug_joint_print_interval = int(args_cli.debug_joint_interval)

    if bool(getattr(args_cli, "show_task_markers", False)):
        if not hasattr(env_cfg, "show_task_markers"):
            raise AttributeError(
                "--show-task-markers requested, but env_cfg has no show_task_markers field"
            )
        env_cfg.show_task_markers = True
    if bool(getattr(args_cli, "show_com_marker", False)):
        if not hasattr(env_cfg, "show_com_marker"):
            raise AttributeError(
                "--show-com-marker requested, but env_cfg has no show_com_marker field"
            )
        env_cfg.show_com_marker = True
    if bool(getattr(args_cli, "debug_com", False)):
        if not hasattr(env_cfg, "debug_com_print"):
            raise AttributeError(
                "--debug-com requested, but env_cfg has no debug_com_print field"
            )
        env_cfg.debug_com_print = True
    if bool(getattr(args_cli, "debug_opening_pca", False)):
        if not hasattr(env_cfg, "debug_opening_pca_on_reset"):
            raise AttributeError(
                "--debug-opening-pca requested, but env_cfg has no debug_opening_pca_on_reset field"
            )
        env_cfg.debug_opening_pca_on_reset = True
        # Match production NSEW (opening_ring + pca_frozen) when inspecting PCA.
        if hasattr(env_cfg, "debug_opening_pca_use_opening_ring"):
            env_cfg.debug_opening_pca_use_opening_ring = True

    return env_cfg


def resolve_checkpoint_path(checkpoint: str) -> str:
    """Normalize a checkpoint path from the CLI (handles newlines and duplicated prefixes).

    Common mistake: run from ``isaaclab_rl_wearglove/`` with
    ``--checkpoint home/tamon/code/isaaclab_rl_wearglove/logs/.../best_agent.pt``,
    which ``abspath`` turns into ``.../isaaclab_rl_wearglove/home/tamon/...``.
    """
    raw = " ".join(str(checkpoint).split())
    if not raw:
        raise ValueError("empty checkpoint path")

    candidates: list[str] = []
    candidates.append(raw)
    candidates.append(os.path.expanduser(raw))
    candidates.append(os.path.abspath(raw))

    # Fix duplicated ``.../isaaclab_rl_wearglove/home/...`` segment.
    dup_marker = "/isaaclab_rl_wearglove/home/"
    abs_raw = os.path.abspath(raw)
    if dup_marker in abs_raw:
        fixed = "/home/" + abs_raw.split(dup_marker, 1)[-1]
        candidates.append(fixed)
        candidates.append(os.path.realpath(fixed))

    seen: set[str] = set()
    for path in candidates:
        path = os.path.realpath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Checkpoint file not found. Tried:\n  "
        + "\n  ".join(seen)
        + "\nUse an absolute path, e.g. "
        "--checkpoint /home/tamon/code/isaaclab_rl_wearglove/logs/.../best_agent.pt"
    )


def load_play_checkpoint(agent, path: str) -> None:
    """Load a training checkpoint for play / eval.

    Older checkpoints store ``value_preprocessor`` separately, while
    ``DeterministicValue`` also holds ``value.value_preprocessor`` as a submodule.
    Remove the nested submodule before loading ``value`` weights, then re-attach
    the loaded preprocessor (or the one built at init if absent from the file).
    """
    if agent.writer is None:
        raise ValueError("Cannot load checkpoint: writer is None")

    value = agent.value
    if hasattr(value, "_modules") and "value_preprocessor" in value._modules:
        del value._modules["value_preprocessor"]

    modules = torch.load(path, map_location=agent.device)
    if not isinstance(modules, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(modules)}")

    # Inference-only: skip optimizers (and avoid Optimizer.load_state_dict(strict=...) issues).
    skip_names = {"policy_optimiser", "value_optimiser", "encoder_optimiser"}

    for name, data in modules.items():
        if name in skip_names:
            continue
        module = agent.writer.checkpoint_modules.get(name)
        if module is None:
            print(f"Warning: Cannot load '{name}' module (not registered)")
            continue
        if not hasattr(module, "load_state_dict"):
            continue
        state = data
        if name == "value" and isinstance(data, dict):
            state = {k: v for k, v in data.items() if not k.startswith("value_preprocessor.")}
        incompatible = module.load_state_dict(state, strict=False)
        missing = getattr(incompatible, "missing_keys", None) or []
        unexpected = getattr(incompatible, "unexpected_keys", None) or []
        if missing:
            print(f"Warning: checkpoint '{name}' missing keys: {missing}")
        if unexpected:
            print(f"Warning: checkpoint '{name}' unexpected keys: {unexpected}")
        if hasattr(module, "eval"):
            module.eval()

    loaded_vp = None
    if agent.writer is not None:
        loaded_vp = agent.writer.checkpoint_modules.get("value_preprocessor")
    if loaded_vp is not None:
        value.value_preprocessor = loaded_vp
    elif getattr(agent, "_value_preprocessor", None) is not None:
        value.value_preprocessor = agent._value_preprocessor


def set_seed(seed: int = 42) -> None:
    """Apply the same seed across numpy/torch/random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_seed(
    args_cli,
    env,
    agent_cfg=None,
    env_cfg=None,
    writer=None,
    seed=None,
    print_eval_episode_returns: bool = False,
):
    """Train the PPO agent for a single seed configuration.

    Args:
        args_cli: Command-line arguments.
        env: The gymnasium environment.
        agent_cfg: Agent configuration dictionary.
        env_cfg: Environment configuration.
        writer: Writer for logging.
        seed: Random seed for training.
    """
    dtype = torch.float32

    agent_cfg["seed"] = seed
    set_seed(agent_cfg["seed"])

    # Setup models
    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)

    # Create tensors in memory for RL (only for the training envs, not eval envs)
    env.num_train_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
    if env.num_train_envs < 1:
        raise ValueError(
            f"num_train_envs must be >= 1 (got {env.num_train_envs}): scene.num_envs={env_cfg.scene.num_envs} "
            f"and trainer.num_eval_envs={agent_cfg['trainer']['num_eval_envs']}. "
            "Set num_eval_envs < scene.num_envs or increase --num_envs."
        )
    rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=env.num_train_envs)
    ssl_task = make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer)

    # Configure and instantiate PPO agent
    ppo_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_agent_cfg.update(agent_cfg["agent"])
    agent = PPO(
        encoder,
        policy,
        value,
        value_preprocessor,
        memory=rl_memory,
        cfg=ppo_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer,
        ssl_task=ssl_task,
        dtype=dtype,
        debug=agent_cfg["experiment"]["debug"],
    )

    # Start training
    trainer = make_trainer(env, agent, agent_cfg, ssl_task, writer, print_eval_episode_returns=print_eval_episode_returns)
    trainer.train()
    print("Training complete!")
