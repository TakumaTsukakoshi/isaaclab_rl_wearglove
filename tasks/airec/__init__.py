# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, airec2_finger_taskspace, wear_finger, wear_finger_taskspace, wear_finger_hepi, reach_bracelet, reach_deformable_bracelet

print("Registering AIREC environments")

agents_dir = os.path.dirname(agents.__file__)

agent_config = os.path.join(agents_dir, "wear.yaml")
wear_taskspace_agent = os.path.join(agents_dir, "wear_taskspace.yaml")
wear_hepi_agent = os.path.join(agents_dir, "wear_hepi.yaml")
airec2_taskspace_agent = os.path.join(agents_dir, "airec2_taskspace.yaml")
reach_bracelet_agent = os.path.join(agents_dir, "reach_bracelet.yaml")
reach_deformable_bracelet_agent = os.path.join(agents_dir, "reach_deformable_bracelet.yaml")
gym.register(
    id="AIREC_Wear",
    entry_point="tasks.airec.wear_finger:WearEnv",
    kwargs={
        "env_cfg_entry_point": wear_finger.WearEnvCfg,
        "skrl_cfg_entry_point": agent_config,
    },
    disable_env_checker=True,
)

gym.register(
    id="AIREC_Wear_TaskSpace",
    entry_point="tasks.airec.wear_finger_taskspace:WearFingerTaskSpaceEnv",
    kwargs={
        "env_cfg_entry_point": wear_finger_taskspace.WearFingerTaskSpaceEnvCfg,
        "skrl_cfg_entry_point": wear_taskspace_agent,
    },
    disable_env_checker=True,
)

gym.register(
    id="AIREC_Wear_TaskSpace_HEPi",
    entry_point="tasks.airec.wear_finger_hepi:WearFingerHepiTaskSpaceEnv",
    kwargs={
        "env_cfg_entry_point": wear_finger_hepi.WearFingerHepiTaskSpaceEnvCfg,
        "skrl_cfg_entry_point": wear_hepi_agent,
    },
    disable_env_checker=True,
)

gym.register(
    id="AIREC2_Finger_TaskSpace",
    entry_point="tasks.airec.airec2_finger_taskspace:AIREC2FingerTaskSpaceEnv",
    kwargs={
        "env_cfg_entry_point": airec2_finger_taskspace.AIREC2FingerTaskSpaceEnvCfg,
        "skrl_cfg_entry_point": airec2_taskspace_agent,
    },
    disable_env_checker=True,
)

gym.register(
    id="AIREC_Reach_Bracelet",
    entry_point="tasks.airec.reach_bracelet:ReachBraceletEnv",
    kwargs={
        "env_cfg_entry_point": reach_bracelet.ReachBraceletEnvCfg,
        "skrl_cfg_entry_point": reach_bracelet_agent,
    },
    disable_env_checker=True,
)

gym.register(
    id="AIREC_Reach_Deformable_Bracelet",
    entry_point="tasks.airec.reach_deformable_bracelet:ReachDeformableBraceletEnv",
    kwargs={
        "env_cfg_entry_point": reach_deformable_bracelet.ReachDeformableBraceletEnvCfg,
        "skrl_cfg_entry_point": reach_deformable_bracelet_agent,
    },
    disable_env_checker=True,
)
# from . import agents, reach

# print("Registering nextage environments")

# agents_dir = os.path.dirname(agents.__file__)

# agent_config = os.path.join(agents_dir, "reach.yaml")

# gym.register(
#     id="AIREC_Reach",
#     entry_point="tasks.airec.reach:ReachEnv",
#     kwargs={
#         "env_cfg_entry_point": reach.ReachEnvCfg,
#         "skrl_cfg_entry_point": agent_config,
#     },
#     disable_env_checker=True,
# )

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

# """
# Cartpole balancing environment.
# """

# import gymnasium as gym

# from . import agents

# import os


# ##
# # Register Gym environments.
# ##

# print("Registering airec environments")

# _VARIANT_FILES = {
#     "default_cfg": "default.yaml",
#     # proprioception only
#     "human": "human.yaml",
#     "ptd": "ptd.yaml",
#     "rl_only": "rl_only.yaml",
#     "imitation": "imitation.yaml",
# }

# # from . import agents, block, chain, head, imitation
# from . import agents, imitation

# _AGENTS_DIR = os.path.dirname(agents.__file__)


# def _variant_paths(task_name: str) -> dict[str, str]:
#     base = os.path.join(_AGENTS_DIR, task_name)
#     return {key: os.path.join(base, filename) for key, filename in _VARIANT_FILES.items()}


# def _register_task(task_id: str, env_cls, cfg_cls) -> None:
#     kwargs = {"env_cfg_entry_point": cfg_cls}
#     kwargs.update(_variant_paths(task_id.lower()))

#     print("Registering airec task:", task_id, "with kwargs:", kwargs)
#     gym.register(
#         id=task_id,
#         entry_point=f"tasks.airec.{task_id.lower()}:{env_cls.__name__}",
#         disable_env_checker=True,
#         kwargs=kwargs,
#     )


# # _register_task("Head", head.HeadEnv, head.HeadEnvCfg)

# # _register_task("Block", block.BlockEnv, block.BlockEnvCfg)
# # _register_task("Chain", chain.ChainEnv, chain.ChainEnvCfg)
# _register_task("Imitation", imitation.ImitationEnv, imitation.ImitationEnvCfg)

# # ``scripts/train.py`` / ``gym.make`` sometimes use the class name as ``--task``.
# gym.register(
#     id="ImitationEnv",
#     entry_point="tasks.airec.imitation:ImitationEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": imitation.ImitationEnvCfg,
#         **_variant_paths("imitation"),
#     },
# )
# print("Registering airec task alias: ImitationEnv")