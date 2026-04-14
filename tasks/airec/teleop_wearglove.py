# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Wear glove task + :class:`TeleopJointSpaceMixin` for joint teleop / demos.

Scene (see :class:`tasks.airec.wear_finger.WearEnv` / :mod:`tasks.airec.airec2_finger`):

- **AIREC** (`self.robot`): policy / teleop actuation — torso, arms, and AIREC finger joints (same
  articulation as the humanoid). This is what you train or drive to pull the garment on.
- **Shadow Hand** (`self.hand`): a separate dexterous hand asset (``SHADOW_HAND_CFG``). It is **not**
  driven by the RL action vector in the default loop; it is reset in ``_reset_target_pose`` and serves
  as a **spatial reference**: :class:`~isaaclab.sensors.FrameTransformerCfg` targets under
  ``/World/envs/env_.*/ShadowHand/...`` define thumb / pinky / wrist goals so rewards measure whether
  the **glove opening** and **AIREC fingertips** align **toward** those Shadow Hand frames — i.e.
  whether AIREC can achieve a “wearing” motion **relative to** the Shadow Hand pose.
- **Deformable glove** (`self.object`): the garment; ``object_type="deformable"`` on :class:`WearEnvCfg`.

So “wearing toward Shadow Hand” in this codebase means: **optimize / teleop AIREC** so the glove and
AIREC hands track the **goal geometry** attached to Shadow Hand links. Teleop via
:class:`TeleopJointSpaceMixin` primarily adjusts **arms + torso** (and locks some body DOFs); **full
finger control** for AIREC usually comes from **training** (action includes hand joints) or from
recorded ``joint_commands`` replay — not from the same keyboard as the arms.

Run::

    ./isaaclab.sh -p teleop_joints_wearglove.py
    ./isaaclab.sh -p teleop_joints.py --task wearglove
"""

from __future__ import annotations

from typing import Literal

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from tasks.airec.wear_finger import WearEnv, WearEnvCfg
from tasks.airec.teleop_joint_space import TeleopJointSpaceMixin
from tasks.airec.teleop_viewer import teleop_viewer_cfg


@configclass
class TeleopWearGloveEnvCfg(WearEnvCfg):
    """Wear glove task defaults for interactive teleop (single env; inherits from :class:`WearEnvCfg`)."""

    # Long horizon so teleop is not cut off; episode resets can snap the viewport (esp. external) to a far default camera.
    episode_length_s = 3600.0

    viewer_mode: Literal["head", "external", "none"] = "head"

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except AttributeError:
            pass
        # Deformable gloves require replicate_physics=False (Isaac Lab InteractiveSceneCfg note: optimized PhysX
        # parsing does not support deformables with replication). True + soft body has been observed to leave
        # articulation tensor views stale → weakref.ReferenceError in articulation_data.joint_vel on scene.update.
        object.__setattr__(self, "scene", InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False))
        object.__setattr__(self, "viewer", teleop_viewer_cfg(self.viewer_mode))


class TeleopWearGloveEnv(TeleopJointSpaceMixin, WearEnv):
    """Wear glove task with joint-space teleop (see :class:`TeleopJointSpaceMixin`)."""

    cfg: TeleopWearGloveEnvCfg