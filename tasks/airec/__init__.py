# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, wear

print("Registering nextage environments")

agents_dir = os.path.dirname(agents.__file__)

agent_config = os.path.join(agents_dir, "wear.yaml")

gym.register(
    id="AIREC_Wear",
    entry_point="tasks.airec.wear:WearEnv",
    kwargs={
        "env_cfg_entry_point": wear.WearEnvCfg,
        "skrl_cfg_entry_point": agent_config,
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

