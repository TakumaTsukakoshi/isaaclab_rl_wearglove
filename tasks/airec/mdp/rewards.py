# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Wear-glove reward terms following GeometryRL cloth-hanging formulation (ICLR 2025, arXiv:2502.07005, App. B.7).

**Paper mapping**

- **Hole centroid** ``c_hole`` → glove opening reference: ``goal_cent_pos`` (midpoint of north/south anchors), env-local.
- **Hanger centroid** ``c_hanger`` → Shadow wrist goal, env-local (``wrist`` target minus ``env_origins``).
- **Alignment** ``θ_align`` → angle between the glove opening-plane normal (cross of N–S and E–W rim vectors) and the
  wrist frame +Y axis in world (``quat_apply(wrist_quat, e_y)``), matching the paper’s ``|cos(θ_align) − 1|`` term.
- **Point velocities** → mean nodal speed (same role as cloth particles).
- **Distortion** → mean relative rim-edge stretch vs rest mesh on the six undirected edges of the N/S/E/W anchor quad
  (analog of ``|l_i − l^0_i| / l^0_i`` on cloth edges).
- **Action rate** → ``‖a_t − a_{t−1}‖`` (paper ``A_actions``).

Time-dependent weighting on the hole–hanger term uses ``max_episode_length`` as ``T`` (paper uses ``T = 100``).

See also GeometryRL Orbit cloth tasks: ``geometry_rl/geometry_rl/orbit/tasks/manipulation/cloth_tasks/mdp/rewards.py``.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

# Paper B.7 (Cloth-Hanging) coefficients
W_RHOLE_HANGER_EARLY = 0.8
W_RHOLE_HANGER_LATE = 4.0
W_V_POINTS = 0.2
W_D_DISTORTION = 1.0
W_A_ACTIONS = 0.002
ALIGN_SCALE = 0.1


def _goal_wrist_env_local(env) -> torch.Tensor:
    return env.wrist_goal_frame.data.target_pos_source[..., 0, :] - env.scene.env_origins


def r_hole_hanger_alignment_cost(env) -> torch.Tensor:
    """``R_hole-hanger = ‖c_hole − c_hanger‖ + 0.1 · |cos(θ_align) − 1|`` per env ``(num_envs,)``."""
    c_hole = env.goal_cent_pos
    c_hanger = _goal_wrist_env_local(env)
    dist = torch.norm(c_hole - c_hanger, dim=-1)

    ns = env.goal_north_pos - env.goal_south_pos
    ew = env.goal_east_pos - env.goal_west_pos
    n = torch.cross(ns, ew, dim=-1)
    n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-8)

    wrist_quat = env.wrist_goal_frame.data.target_quat_source[..., 0, :]
    y_axis = torch.zeros(env.num_envs, 3, device=env.device, dtype=wrist_quat.dtype)
    y_axis[:, 1] = 1.0
    h_dir = quat_apply(wrist_quat, y_axis)
    h_dir = h_dir / (torch.norm(h_dir, dim=-1, keepdim=True) + 1e-8)

    cos_align = (n * h_dir).sum(dim=-1).clamp(-1.0, 1.0)
    align = torch.abs(cos_align - 1.0)
    return dist + ALIGN_SCALE * align


def v_points_mean_nodal_speed(env) -> torch.Tensor:
    """``V_points = (1/N) Σ_i ‖v_i‖`` — mean L2 speed over glove nodes ``(num_envs,)``."""
    v = env.object.data.nodal_vel_w
    return v.norm(dim=-1).mean(dim=-1)


def d_distortion_rim_edges(env) -> torch.Tensor:
    """``D_distortion = (1/M) Σ_e |l_e − l^0_e| / l^0_e`` on the six rim edges between N/S/E/W anchors."""
    if getattr(env, "anchor_idx", None) is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    pairs = (
        ("north", "south"),
        ("east", "west"),
        ("north", "east"),
        ("north", "west"),
        ("south", "east"),
        ("south", "west"),
    )
    rest = env.object.data.default_nodal_state_w[0, :, :3].to(device=env.device, dtype=torch.float32)
    stretches: list[torch.Tensor] = []
    for a, b in pairs:
        ia = env.anchor_idx[a]
        ib = env.anchor_idx[b]
        pa = env.object.data.nodal_pos_w[:, ia, :]
        pb = env.object.data.nodal_pos_w[:, ib, :]
        lc = torch.norm(pa - pb, dim=-1)
        l0 = torch.norm(rest[ia] - rest[ib]).clamp_min(1e-8)
        stretches.append((lc - l0).abs() / l0)
    return torch.stack(stretches, dim=0).mean(dim=0)


def a_actions_l2(env) -> torch.Tensor:
    """``A_actions = ‖a_t − a_{t−1}‖`` (L2), per env ``(num_envs,)``."""
    return torch.norm(env.actions - env.prev_actions, dim=-1)


def _hole_hanger_weight(env) -> torch.Tensor:
    """Paper: ``0.8`` when ``t < T−2``, else ``4.0`` (episode step in ``episode_length_buf``)."""
    T = int(env.max_episode_length)
    cutoff = max(T - 2, 0)
    late = env.episode_length_buf >= cutoff
    return torch.where(
        late,
        torch.full_like(env.episode_length_buf, W_RHOLE_HANGER_LATE, dtype=torch.float32),
        torch.full_like(env.episode_length_buf, W_RHOLE_HANGER_EARLY, dtype=torch.float32),
    )


def geometryrl_b7_cloth_hanging_reward(env) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total reward from App. B.7 (maximize — costs enter with a minus sign).

    ``R_tot = − w_hh · R_hole-hanger − 0.2·V − 1.0·D − 0.002·A`` with time-varying ``w_hh``.
    """
    r_hh = r_hole_hanger_alignment_cost(env)
    v = v_points_mean_nodal_speed(env)
    d = d_distortion_rim_edges(env)
    a = a_actions_l2(env)
    w_hh = _hole_hanger_weight(env)

    rewards = -(w_hh * r_hh + W_V_POINTS * v + W_D_DISTORTION * d + W_A_ACTIONS * a)

    log = {
        "r_b7_total": rewards,
        "b7_r_hole_hanger": r_hh,
        "b7_v_points": v,
        "b7_d_distortion": d,
        "b7_a_actions": a,
        "b7_w_hole_hanger": w_hh,
    }
    return rewards, log


# Backwards-compatible names (older call sites / notebooks)
glove_nodal_velocity_l2_mean = v_points_mean_nodal_speed


def glove_centroid_to_wrist_distance(env) -> torch.Tensor:
    """Euclidean distance between glove nodal centroid and wrist goal (env-local) ``(num_envs,)``."""
    c = env.object.data.nodal_pos_w.mean(dim=1) - env.scene.env_origins
    w = _goal_wrist_env_local(env)
    return torch.norm(c - w, dim=-1)


def mean_target_alignment_error(env) -> torch.Tensor:
    """Mean distance from each of six Shadow goal points to glove centroid (world frame) ``(num_envs,)``."""
    c = env.object.data.nodal_pos_w.mean(dim=1)
    goals = torch.stack(
        (
            env.thumb_goal_frame.data.target_pos_source[..., 0, :],
            env.pinky_goal_frame.data.target_pos_source[..., 0, :],
            env.fore_goal_frame.data.target_pos_source[..., 0, :],
            env.middle_goal_frame.data.target_pos_source[..., 0, :],
            env.ring_goal_frame.data.target_pos_source[..., 0, :],
            env.wrist_goal_frame.data.target_pos_source[..., 0, :],
        ),
        dim=1,
    )
    d = torch.norm(goals - c.unsqueeze(1), dim=-1)
    return d.mean(dim=-1)
