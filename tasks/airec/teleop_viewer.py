# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Static viewport presets for teleop scripts (e.g. :class:`tasks.airec.teleop_imitation.TeleopImitationEnvCfg`)."""

from __future__ import annotations

from typing import Literal

from isaaclab.envs import ViewerCfg

_TELEOP_VIEWER_RES = (1920, 1080)


def teleop_viewer_cfg(mode: Literal["head", "external", "none"] = "head") -> ViewerCfg:
    """Viewport presets for teleop envs.

    All modes use ``origin_type="world"`` and a **static** eye/lookat so the Omniverse viewport
    is not overwritten every frame (you can orbit / pan freely). Tuned for env 0 at the origin;
    the robot does not move the camera once placed.

    ``head``: over-shoulder / upper-torso height looking toward the bed (+x workspace).

    ``external``: in front of the robot (+x), looking back at the upper body.

    ``none``: same world-frame distance as ``external`` (not Isaac's default 7.5 m diagonal, which feels "lost in space").
    """

    if mode == "none":
        # Do not rely on ViewerCfg defaults (eye 7.5,7.5,7.5 is extremely far); use a neutral near-scene pose.
        return ViewerCfg(
            resolution=_TELEOP_VIEWER_RES,
            origin_type="world",
            eye=(3.0, 0.3, 1.4),
            lookat=(0.2, 0.0, 1.05),
        )

    if mode == "head":
        return ViewerCfg(
            resolution=_TELEOP_VIEWER_RES,
            origin_type="world",
            eye=(0.35, 0.0, 1.52),
            lookat=(1.05, 0.0, 0.88),
        )
    return ViewerCfg(
        resolution=_TELEOP_VIEWER_RES,
        origin_type="world",
        eye=(3.0, 0.3, 1.4),
        lookat=(0.2, 0.0, 1.05),
    )