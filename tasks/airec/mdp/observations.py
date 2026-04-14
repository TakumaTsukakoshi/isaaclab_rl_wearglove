# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Geometry observations for Wear glove (cloth_tasks-style: env-local tensors).

All positions are expressed with ``env_origins`` subtracted (episode-local), matching
``cloth_tasks/mdp/observations.py`` conventions.
"""

from __future__ import annotations

import torch


def glove_nodal_positions_w_minus_origin(env) -> torch.Tensor:
    """Glove deformable nodal positions: ``(num_envs, V, 3)``."""
    pos = env.object.data.nodal_pos_w.clone()
    pos -= env.scene.env_origins.unsqueeze(1)
    return pos


def glove_nodal_velocities_w(env) -> torch.Tensor:
    """Glove nodal linear velocities: ``(num_envs, V, 3)`` (world frame, no origin shift)."""
    return env.object.data.nodal_vel_w.clone()


def ee_arm_link_positions_minus_origin(env, body_names: tuple[str, str]) -> torch.Tensor:
    """End-effector body positions (arm link 7) ``(num_envs, 2, 3)`` in world minus env origin."""
    robot = env.robot
    bnames = list(robot.data.body_names)
    i_r = bnames.index(body_names[0])
    i_l = bnames.index(body_names[1])
    ee = torch.stack(
        (
            robot.data.body_pos_w[:, i_r],
            robot.data.body_pos_w[:, i_l],
        ),
        dim=1,
    )
    return ee - env.scene.env_origins.unsqueeze(1)


def shadow_hand_goal_positions_minus_origin(env) -> torch.Tensor:
    """Stacked goal frames (thumb, pinky, fore, middle, ring, wrist): ``(num_envs, 6, 3)``.

    Order matches :class:`tasks.airec.wear_finger.WearEnv` frame sensors.
    """
    return torch.stack(
        (
            env.thumb_goal_frame.data.target_pos_source[..., 0, :].clone(),
            env.pinky_goal_frame.data.target_pos_source[..., 0, :].clone(),
            env.fore_goal_frame.data.target_pos_source[..., 0, :].clone(),
            env.middle_goal_frame.data.target_pos_source[..., 0, :].clone(),
            env.ring_goal_frame.data.target_pos_source[..., 0, :].clone(),
            env.wrist_goal_frame.data.target_pos_source[..., 0, :].clone(),
        ),
        dim=1,
    ) - env.scene.env_origins.unsqueeze(1)


def subsample_glove_nodes(x: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """Reduce ``(num_envs, V, D)`` along ``V`` via striding (VRAM for HEPi / rollout buffers).

    Node order is mesh-dependent; striding gives a cheap spatial subsample vs full ``V``.
    """
    if max_nodes <= 0 or x.ndim < 3 or x.shape[1] <= max_nodes:
        return x
    _, v, _ = x.shape
    step = max(1, v // max_nodes)
    idx = torch.arange(0, v, step, device=x.device, dtype=torch.long)[:max_nodes]
    return x[:, idx, :]


def flatten_spatial(x: torch.Tensor) -> torch.Tensor:
    """Flatten trailing (..., D) to (num_envs, -1)."""
    return x.reshape(x.shape[0], -1)
