# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wear glove + AIREC + Shadow Hand with **task-space** policy commands (14D dual-arm poses → diff IK).

Joint-space training uses :mod:`tasks.airec.wear_finger` (``AIREC_Wear``). This module keeps all Cartesian /
IK configuration and control overrides here so it does not overlap ``wear_finger.py`` or ``airec2_finger.py``.

Train::

    python train.py --task AIREC_Wear_TaskSpace
"""

from __future__ import annotations

import torch
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from tasks.airec.airec2_finger_taskspace import apply_task_space_action_on_env, init_task_space_ik_on_env
from tasks.airec.wear_finger import WearEnv, WearEnvCfg


@configclass
class WearFingerTaskSpaceEnvCfg(WearEnvCfg):
    """Same scene and rewards as :class:`~tasks.airec.wear_finger.WearEnvCfg`; policy outputs 14D EE poses (root frame)."""

    num_actions: int = 14
    action_space: int = 14
    task_space_pos_min: tuple[float, float, float] = (-0.55, -0.55, 0.35)
    task_space_pos_max: tuple[float, float, float] = (0.55, 0.55, 1.45)
    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )


class WearFingerTaskSpaceEnv(WearEnv):
    """Task-space variant of :class:`~tasks.airec.wear_finger.WearEnv`."""

    cfg: WearFingerTaskSpaceEnvCfg

    @property
    def policy_action_dim(self) -> int:
        return int(self.cfg.num_actions)

    def __init__(self, cfg: WearFingerTaskSpaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Parent allocates ``actions`` with width ``len(actuated_dof_indices)``; policy is 14D. Keep a fixed
        # (N, 14) buffer so the first ``reset`` observation matches later steps (avoids prop 121 vs 109 / encoder mismatch).
        n = int(cfg.num_actions)
        self.actions = torch.zeros((self.num_envs, n), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        init_task_space_ik_on_env(self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.last_action = self.actions.clone()
        self.actions.copy_(actions)

    def _apply_action(self) -> None:
        apply_task_space_action_on_env(self)
