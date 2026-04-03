# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation (empty) scene + :class:`TeleopJointSpaceMixin` for joint teleop / demo recording.

This uses :class:`tasks.airec.imitation.ImitationEnv` with ``object_type="none"`` (no glove). For the
**wear glove** task — **deformable glove** + **Shadow Hand** reference + AIREC — use
:class:`tasks.airec.teleop_wearglove.TeleopWearGloveEnv` and run e.g.
``teleop_joints_wearglove.py`` (defaults to ``--task wearglove``) or ``teleop_joints.py --task wearglove``.
"""

from __future__ import annotations

from typing import Literal

from tasks.airec.imitation import ImitationEnv, ImitationEnvCfg
from tasks.airec.teleop_joint_space import TeleopJointSpaceMixin
from tasks.airec.teleop_viewer import teleop_viewer_cfg
from isaaclab.utils import configclass


@configclass
class TeleopImitationEnvCfg(ImitationEnvCfg):
    """Same layout as :class:`ImitationEnvCfg` with teleop viewport presets (inherits ``episode_length_s``)."""

    #: Long default when no demo path (see :class:`ImitationEnv`); avoids frequent timeouts that confuse the viewport.
    episode_length_s = 3600.0

    #: ``head`` / ``external`` / ``none``
    viewer_mode: Literal["head", "external", "none"] = "head"

    def __post_init__(self) -> None:
        object.__setattr__(self, "viewer", teleop_viewer_cfg(self.viewer_mode))


class TeleopImitationEnv(TeleopJointSpaceMixin, ImitationEnv):
    """Empty-scene imitation task with joint-space teleop (see :class:`TeleopJointSpaceMixin`)."""

    cfg: TeleopImitationEnvCfg