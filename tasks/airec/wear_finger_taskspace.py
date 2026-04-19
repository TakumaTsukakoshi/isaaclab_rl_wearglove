# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wear glove + AIREC + Shadow Hand with **task-space** policy commands (14D dual-arm poses + 12D hands → diff IK).

Joint-space training uses :mod:`tasks.airec.wear_finger` (``AIREC_Wear``). Policy actions are **26D**: the same
14D root-frame EE targets as before, plus thumb + first-finger joints (see
:attr:`tasks.airec.airec2_finger_taskspace.TASKSPACE_HAND_JOINT_NAMES`). Hand channels use
:attr:`~tasks.airec.airec2_finger.AIRECEnvCfg.taskspace_hand_span_fraction` around :attr:`robot.data.default_joint_pos`
(same nominal as ``assets/airec_finger`` ``joint_pos``).

Train::

    python train.py --task AIREC_Wear_TaskSpace
"""

from __future__ import annotations

import torch
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from tasks.airec.airec2_finger_taskspace import (
    NUM_TASKSPACE_HAND_ACTIONS,
    NUM_TASKSPACE_POSE_ACTIONS,
    TASKSPACE_HAND_JOINT_NAMES,
    apply_task_space_action_on_env,
    init_task_space_ik_on_env,
    sync_taskspace_hand_joint_cmd_from_default,
)
from tasks.airec.wear_finger import WearEnv, WearEnvCfg


@configclass
class WearFingerTaskSpaceEnvCfg(WearEnvCfg):
    """Same scene and rewards as :class:`~tasks.airec.wear_finger.WearEnvCfg`; policy outputs 26D (14D poses + 12D hands)."""

    num_actions: int = NUM_TASKSPACE_POSE_ACTIONS + NUM_TASKSPACE_HAND_ACTIONS
    action_space: int = NUM_TASKSPACE_POSE_ACTIONS + NUM_TASKSPACE_HAND_ACTIONS
    taskspace_hand_joint_names: tuple[str, ...] = TASKSPACE_HAND_JOINT_NAMES
    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )


class WearFingerTaskSpaceEnv(WearEnv):
    """Task-space variant of :class:`~tasks.airec.wear_finger.WearEnv`."""

    cfg: WearFingerTaskSpaceEnvCfg

    def __init__(self, cfg: WearFingerTaskSpaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Parent allocates ``actions`` for joint-space width; policy is 26D. Keep a fixed (N, num_actions) buffer.
        n = int(cfg.num_actions)
        self.actions = torch.zeros((self.num_envs, n), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        init_task_space_ik_on_env(self, taskspace_hand_joint_names=cfg.taskspace_hand_joint_names)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            sync_taskspace_hand_joint_cmd_from_default(self, None)
        else:
            sync_taskspace_hand_joint_cmd_from_default(self, self._normalize_env_ids(env_ids))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.last_action = self.actions.clone()
        self.actions.copy_(actions)

    def _apply_action(self) -> None:
        apply_task_space_action_on_env(self)
