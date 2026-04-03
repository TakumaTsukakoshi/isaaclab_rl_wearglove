# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Block manipulation scene + :class:`TeleopJointSpaceMixin` for joint teleop / demos."""

from __future__ import annotations

from typing import Literal

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from tasks.airec.block import BlockEnv, BlockEnvCfg
from tasks.airec.teleop_joint_space import TeleopJointSpaceMixin
from tasks.airec.teleop_viewer import teleop_viewer_cfg


@configclass
class TeleopBlockEnvCfg(BlockEnvCfg):
    """Block task defaults for interactive teleop (single env; inherits ``episode_length_s`` from :class:`BlockEnvCfg`)."""

    #: Long horizon so teleop is not cut off every 10s; episode resets can snap the viewport (esp. external) to a far default camera.
    episode_length_s = 10.0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

    viewer_mode: Literal["head", "external", "none"] = "head"

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except AttributeError:
            pass
        object.__setattr__(self, "viewer", teleop_viewer_cfg(self.viewer_mode))


class TeleopBlockEnv(TeleopJointSpaceMixin, BlockEnv):
    """Block task with joint-space teleop (see :class:`TeleopJointSpaceMixin`)."""

    cfg: TeleopBlockEnvCfg