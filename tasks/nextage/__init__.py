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
    id="Nextage_Wear",
    entry_point="tasks.nextage.wear:WearEnv",
    kwargs={
        "env_cfg_entry_point": wear.WearEnvCfg,
        "skrl_cfg_entry_point": agent_config,
    },
    disable_env_checker=True,
)
