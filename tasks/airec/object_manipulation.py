# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ObjectManipulationEnv: Base environment for robot2object and object2goal tasks.

Shared by Block (rigid object) and Head (articulated human) environments.
Subclasses define object/goal assets, frame configurations, and reset logic.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_apply,
    quat_from_matrix,
    matrix_from_quat,
    create_rotation_matrix_from_view,
    quat_rotate_inverse,
    quat_mul,
    quat_from_euler_xyz,
)
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.envs import ViewerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab_rl_wearglove.tasks.airec.airec2_finger import (
    AIRECEnv,
    AIRECEnvCfg,
    distance_reward,
)
from isaaclab.utils import configclass

def _reward_dist_std(target_dist: float, target_reward: float) -> float:
    """sigma s.t. 1-tanh(target_dist/sigma)=target_reward"""
    return (2.0 * target_dist) / math.log((2.0 - target_reward) / target_reward)


# Shared visualization colours
GOAL_COLOUR = (1.0, 0.2, 1.0)
OBJECT_COLOUR = (0.8, 1.0, 0.6)
ROBOT_COLOUR = (0.2, 1.0, 0.6)

# Object→goal connecting lines (magenta)
O2G_LINE_VIS_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/o2g_lines",
    markers={
        "connecting_line": sim_utils.CylinderCfg(
            radius=0.004,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_COLOUR, roughness=1.0),
        ),
    },
)

# Robot→object connecting lines (green) + frame markers for robot pose + target quat direction
_r2o_frame_cfg = FRAME_MARKER_CFG.copy()
_r2o_frame_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
R2O_LINE_VIS_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/r2o_lines",
    markers={
        "connecting_line": sim_utils.CylinderCfg(
            radius=0.004,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ROBOT_COLOUR, roughness=1.0),
        ),
        "frame": _r2o_frame_cfg.markers["frame"],
        "frame_target": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.08, 0.08, 0.08),
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ROBOT_COLOUR, roughness=1.0),
        ),
    },
)


@configclass
class ObjectManipulationEnvCfg(AIRECEnvCfg):
    """Configuration for object manipulation tasks (robot2object, object2goal).

    Subclasses add object_cfg, goal_cfg, object_frame_cfg, goal_frame_cfg, etc.
    """

    bed_random_x: float = 0.0  # max ± in x for bed position at episode start
    bed_random_y: float = 0.0  # max ± in y for bed position at episode start
    object_noise: float = 0.0  # max ± in x,y from bed center for object at episode start
    goal_random_x: float = 0.0  # max ± in x for goal position at episode start
    goal_random_y: float = 0.0  # max ± in y for goal position at episode start

    # Random start along the loaded imitation demo (requires ``imitation_demo_path`` on parent cfg).
    # Indexes ``episode_length_buf`` so :meth:`AIRECEnv._get_imitation_reference_joint_pos` and
    # related helpers align with a sampled timestep; optionally initializes the robot from
    # ``measured_joint_pos`` at that timestep.
    imitation_random_demo_start: bool = True
    imitation_reset_joint_state_from_demo: bool = True
    imitation_demo_start_min_index: int = 0
    imitation_demo_start_max_index: int | None = None  # inclusive; None = last index allowed by margin
    imitation_demo_start_min_remaining_steps: int = 0  # sample t <= T - 1 - this (room for horizon / tail)
    #: Extra uniform ± rad on **body** actuated joints when resetting from demo. ``0`` uses
    #: ``reset_robot_joint_noise_scale``; base joints get ``reset_robot_pos_*`` / ``reset_robot_quat_noise_scale``.
    imitation_demo_start_joint_noise_scale: float = 0.0
    #: If the demo includes object pose, place the rigid task object from demo (local pos + world quat).
    imitation_reset_object_state_from_demo: bool = True

    # Probability of sampling initial state from the demo (joint + object when enabled) at each reset.
    # When ``imitation_demo_reset_decay_timesteps_M`` > 0, linearly interpolate from *start* to *end* over that
    # many **global training timesteps** (same units as :meth:`ObjectManipulationEnv.set_imitation_global_training_timestep`).
    # When decay is 0, the schedule is disabled and *start* is used for all time.
    imitation_demo_reset_fraction_start: float = 1.0
    imitation_demo_reset_fraction_end: float = 0.0
    imitation_demo_reset_decay_timesteps_M: float = 0.0

    # Layout for object/goal mats (Block task)
    base_link_to_bed_edge: float = 0.1
    bed_width: float = 0.6
    bed_height: float = 0.7
    bed_depth: float = 0.05
    goal_pos: tuple = (-1.0, 0.0, 0.3)

    # Reward hyperparameters (subclasses override as needed)
    reward_r2o_scale: float = 1.0
    reward_r2o_b: float = 0.2
    reward_lift_scale: float = 5.0
    reward_o2g_scale: float = 10.0

    # Distance reward std (sigma for tanh funnel): sigma = 2*target_dist / ln((2-target_reward)/target_reward)
    tanh_target_reward: float = 0.01
    reward_r2o_target_dist: float = 0.5
    reward_lift_target_dist: float = 0.3
    reward_o2g_target_dist: float = 2.0

    reward_r2o_dist_std: float = 0.0
    reward_lift_dist_std: float = 0.0
    reward_o2g_dist_std: float = 0.0

    #: Terminate with a success bonus when the object is near the goal and nearly stationary.
    terminate_on_goal_settle: bool = True
    lift_goal_settle_radius_m: float = 0.1
    lift_goal_settle_max_speed_mps: float = 0.01
    lift_goal_settle_success_bonus: float = 10000.0
    goal_settle_radius_m: float = 0.2
    goal_settle_max_speed_mps: float = 0.01
    goal_settle_success_bonus: float = 10000.0

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except AttributeError:
            pass
        if self.reward_r2o_dist_std == 0.0:
            object.__setattr__(self, "reward_r2o_dist_std", _reward_dist_std(self.reward_r2o_target_dist, self.tanh_target_reward))
        if self.reward_lift_dist_std == 0.0:
            object.__setattr__(self, "reward_lift_dist_std", _reward_dist_std(self.reward_lift_target_dist, self.tanh_target_reward))
        if self.reward_o2g_dist_std == 0.0:
            object.__setattr__(self, "reward_o2g_dist_std", _reward_dist_std(self.reward_o2g_target_dist, self.tanh_target_reward))

    env_spacing = 5

    left_eye = (env_spacing, -env_spacing, 1.5)
    right_eye = (env_spacing, 0, 1.5)
    lookat = (env_spacing/2, -env_spacing/2, 1)
    viewer: ViewerCfg = ViewerCfg(eye=right_eye, lookat=lookat, resolution=(1920, 1080))

     # Visual mats showing object and goal randomization regions
    object_mat_size_x = 2 * (bed_random_x)
    object_mat_size_y = 2 * (bed_random_y)
    object_mat_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_mat",
        spawn=sim_utils.CuboidCfg(
            size=(object_mat_size_x, object_mat_size_y, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 1.0, 0.6),
                opacity=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(base_link_to_bed_edge + bed_width / 2, 0, 0.0025),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    goal_mat_size_x = 2 * goal_random_x
    goal_mat_size_y = 2 * goal_random_y
    goal_mat_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_mat",
        spawn=sim_utils.CuboidCfg(
            size=(goal_mat_size_x, goal_mat_size_y, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 1.0),
                opacity=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(goal_pos[0], goal_pos[1], 0.0025),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


class ObjectManipulationEnv(AIRECEnv):
    """Base environment for object manipulation: robot→object and object→goal.

    Subclasses must:
    - In _setup_scene: add self.object, self.goal, self.object_frames, self.goal_frames
    - In __init__ (after super): call _init_object_manipulation(r2o_robot_idx, r2o_object_idx, o2g_object_idx, o2g_goal_idx)
    - Implement _reset_object(env_ids)
    - Implement _get_dones() or override _get_dones_termination() for task-specific termination
    """

    cfg: ObjectManipulationEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        self._o2g_line_vis = VisualizationMarkers(O2G_LINE_VIS_CFG)
        self._r2o_line_vis = VisualizationMarkers(R2O_LINE_VIS_CFG)

        # Object and goal randomization mats
        if self.cfg.object_mat_size_x > 0 and self.cfg.object_mat_size_y > 0:
            self.object_mat = RigidObject(self.cfg.object_mat_cfg)
            self.scene.rigid_objects["object_mat"] = self.object_mat
        if self.cfg.goal_mat_size_x > 0 and self.cfg.goal_mat_size_y > 0:
            self.goal_mat = RigidObject(self.cfg.goal_mat_cfg)
            self.scene.rigid_objects["goal_mat"] = self.goal_mat

    def _init_object_manipulation(
        self,
        r2o_robot_idx: torch.Tensor,
        r2o_object_idx: torch.Tensor,
        o2g_object_idx: torch.Tensor,
        o2g_goal_idx: torch.Tensor,
        object_center_vel_idx: int = 0,
    ) -> None:
        """Initialize shared object-manipulation tensors and indices.

        Call this at the end of subclass __init__ after super().__init__ has run
        (so object_frames, goal_frames exist from _setup_scene).

        Args:
            r2o_robot_idx: Indices into airec_frames for robot body parts (robot→object pairs).
            r2o_object_idx: Indices into object_frames for corresponding object frames.
            o2g_object_idx: Indices into object_frames for object→goal pairs.
            o2g_goal_idx: Indices into goal_frames for corresponding goal frames.
            object_center_vel_idx: Index into object_frames for object center velocity (reward penalty).
        """
        self.visualize = False
        self._object_center_vel_idx = object_center_vel_idx
        self._r2o_robot_idx = r2o_robot_idx.to(self.device)
        self._r2o_object_idx = r2o_object_idx.to(self.device)
        self._o2g_object_idx = o2g_object_idx.to(self.device)
        self._o2g_goal_idx = o2g_goal_idx.to(self.device)

        self.num_object_frames = len(self.object_frames.data.target_frame_names)
        self.num_robot2object_frames = len(r2o_robot_idx)
        self.num_object2goal_frames = len(o2g_object_idx)

        self.object_frames_pos = torch.zeros(
            (self.num_envs, self.num_object_frames, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.object_frames_rot = torch.zeros(
            (self.num_envs, self.num_object_frames, 3, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.object_frames_vel = torch.zeros(
            (self.num_envs, self.num_object_frames, 3),
            dtype=self.dtype,
            device=self.device,
        )

        self.robot2object_frames_pos = torch.zeros(
            (self.num_envs, self.num_robot2object_frames, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.robot2object_frames_rot = torch.zeros(
            (self.num_envs, self.num_robot2object_frames, 3, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.robot2object_frames_relative_rot = torch.zeros(
            (self.num_envs, self.num_robot2object_frames, 3, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.robot2object_frames_angular_dist = torch.zeros(
            (self.num_envs, self.num_robot2object_frames),
            dtype=self.dtype,
            device=self.device,
        )
        self.robot2object_frames_vel = torch.zeros(
            (self.num_envs, self.num_robot2object_frames, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.robot2object_frames_euclidean_distance = torch.zeros(
            (self.num_envs, self.num_robot2object_frames),
            dtype=self.dtype,
            device=self.device,
        )


        self.object2goal_frames_pos = torch.zeros(
            (self.num_envs, self.num_object2goal_frames, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.object2goal_frames_vel = torch.zeros(
            (self.num_envs, self.num_object2goal_frames, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.object2goal_frames_euclidean_distance = torch.zeros(
            (self.num_envs, self.num_object2goal_frames),
            dtype=self.dtype,
            device=self.device,
        )

        self.object2liftgoal_distance = torch.zeros(
            (self.num_envs, 3),
            dtype=self.dtype,
            device=self.device,
        )

        self.object2liftgoal_euclidean_distance = torch.zeros(
            (self.num_envs, ),
            dtype=self.dtype,
            device=self.device,
        )

        self.success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # Episode task metrics (wandb; not reward-scaled). Reset in _reset_idx.
        self._lift_success_episode = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._goal_success_episode = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._lift_success_timesteps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # Min over frames & time: closest object→goal distance (m) so far this episode.
        self._episode_best_object2goal_min_m = torch.full(
            (self.num_envs,), float("inf"), dtype=self.dtype, device=self.device
        )

        # Updated by the RL trainer each step; drives ``imitation_demo_reset_*`` schedule.
        self._imitation_global_training_timestep: int = 0

    def set_imitation_global_training_timestep(self, t: int) -> None:
        """Set global training timestep counter (sum of training-env steps); used for demo-reset decay."""
        self._imitation_global_training_timestep = int(t)

    def _imitation_demo_reset_probability(self) -> float:
        """Current probability [0, 1] of using a demo-based reset for an env (Bernoulli per reset)."""
        s = float(self.cfg.imitation_demo_reset_fraction_start)
        e = float(self.cfg.imitation_demo_reset_fraction_end)
        decay_m = float(self.cfg.imitation_demo_reset_decay_timesteps_M or 0.0)
        s = max(0.0, min(1.0, s))
        e = max(0.0, min(1.0, e))
        if decay_m <= 0.0:
            return s
        T = decay_m * 1e6
        if T <= 0.0:
            return s
        t = float(max(0, int(self._imitation_global_training_timestep)))
        alpha = min(1.0, t / T)
        return max(0.0, min(1.0, s + alpha * (e - s)))

    def _update_episode_task_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        """Update per-episode success flags and lift timestep counts (reward-independent)."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        lift_ok = self.object2liftgoal_euclidean_distance[env_ids] < self.cfg.lift_goal_settle_radius_m
        self._lift_success_episode[env_ids] = self._lift_success_episode[env_ids] | lift_ok
        self._lift_success_timesteps[env_ids] = self._lift_success_timesteps[env_ids] + lift_ok.long()
        o2g_min = self.object2goal_frames_euclidean_distance[env_ids].min(dim=-1).values
        self._episode_best_object2goal_min_m[env_ids] = torch.minimum(
            self._episode_best_object2goal_min_m[env_ids], o2g_min
        )
        self._goal_success_episode[env_ids] = o2g_min < self.cfg.goal_settle_radius_m

    def _goal_settle_satisfied(self) -> torch.Tensor | None:
        """True when min object→goal distance and root linear speed satisfy settle thresholds.

        Uses :attr:`object2goal_frames_euclidean_distance` (already updated) and
        ``object.data.root_lin_vel_w`` (rigid body or articulation root). Returns ``None`` if
        goal-settle termination/bonus is disabled or the task has no suitable ``object`` asset.
        """
        if not self.cfg.terminate_on_goal_settle:
            return None
        obj = getattr(self, "object", None)
        if obj is None or not hasattr(obj.data, "root_lin_vel_w"):
            return None
        o2g_min = self.object2goal_frames_euclidean_distance.min(dim=-1).values
        obj_speed = torch.norm(obj.data.root_lin_vel_w[:, :3], dim=-1)
        return (o2g_min <= self.cfg.goal_settle_radius_m) & (obj_speed < self.cfg.goal_settle_max_speed_mps)

    def _compute_object_manipulation_values(self, env_ids: torch.Tensor | None = None) -> None:
        """Compute robot2object and object2goal positions, velocities, distances.

        Uses object_frames, goal_frames, airec_frames and the configured index pairs.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Object frames
        self.object_frames_vel[env_ids] = (
            self.object_frames.data.target_pos_source[env_ids] - self.object_frames_pos[env_ids]
        ) / self.cfg.physics_dt
        self.object_frames_pos[env_ids] = self.object_frames.data.target_pos_source[env_ids]
        self.object_frames_rot[env_ids] = matrix_from_quat(self.object_frames.data.target_quat_source[env_ids])

        # just for ball right now
        # self.object_height_above_surface[env_ids] = self.object_frames_pos[env_ids, self.center_id, 2] - self.cfg.bed_height - self.cfg.bed_depth/2 - self.cfg.object_radius
        lift_goal_pos = self.lift_goal_frames.data.target_pos_source[env_ids,0,:]
        self.object2liftgoal_distance[env_ids] = self.object_frames_pos[env_ids, self.center_id] - lift_goal_pos
        self.object2liftgoal_euclidean_distance[env_ids] = torch.norm(self.object2liftgoal_distance[env_ids], dim=-1)

        # Robot→object: robot_pos - object_pos for each pair
        robot_frames_pos = self.airec_frames.data.target_pos_source[env_ids][:, self._r2o_robot_idx]
        object_frames_pos = self.object_frames.data.target_pos_source[env_ids][:, self._r2o_object_idx]
        r2o = object_frames_pos - robot_frames_pos
        self.robot2object_frames_vel[env_ids] = (r2o - self.robot2object_frames_pos[env_ids]) / self.cfg.physics_dt
        self.robot2object_frames_pos[env_ids] = r2o
        self.robot2object_frames_euclidean_distance[env_ids] = torch.norm(r2o, dim=-1)  # [env_ids, num_pairs]
        # for the base, we only use the x and y distance
        self.robot2object_frames_euclidean_distance[env_ids, self.base_id] = torch.norm(
            r2o[:, self.base_id, :2], dim=-1
        )

      
        # 1. Directions in World Space
        robot_pos_w = self.airec_frames.data.target_pos_w[env_ids][:, self._r2o_robot_idx]
        object_pos_w = self.object_frames.data.target_pos_w[env_ids][:, self._r2o_object_idx]
        r2o_vec_w = object_pos_w - robot_pos_w
        r2o_dir_w = F.normalize(r2o_vec_w, dim=-1, eps=1e-8)

        # 2. Get the current World Forward vector of each robot frame
        # We take the World Quaternion and apply it to the local X-axis [1, 0, 0]
        robot_quat_w = self.airec_frames.data.target_quat_w[env_ids][:, self._r2o_robot_idx]
        n_env, n_frames, _ = robot_quat_w.shape

        # Instead of expanding/reshaping manually, quat_apply handles batching 
        # if the local_forward matches the trailing dimension
        local_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        robot_forward_w = quat_apply(robot_quat_w.view(-1, 4), local_forward.expand(n_env * n_frames, 3))
        robot_forward_w = robot_forward_w.view(n_env, n_frames, 3)

        # 3. Compute the True Angular Distance (Radians)
        # Dot product: 1.0 (aligned), 0.0 (90 deg), -1.0 (opposite)
        dot = torch.sum(r2o_dir_w * robot_forward_w, dim=-1).clamp(-1.0, 1.0)

        # acos gives you the actual angle in radians [0 to PI]
        # This is much more reliable for debugging and reward shaping
        self.robot2object_frames_angular_dist[env_ids] = 1.5707 - torch.acos(dot)

        # print("base angular dist: ", self.robot2object_frames_angular_dist[0, self.base_id])
        # print("lhand angular dist: ", self.robot2object_frames_angular_dist[0, self.left_hand_id])
        # print("rhand angular dist: ", self.robot2object_frames_angular_dist[0, self.right_hand_id])
        # print("base: ", self.robot2object_frames_angular_dist[0, self.base_id])
        # print("lhand: ", self.robot2object_frames_angular_dist[0, self.left_hand_id])
        # print("rhand: ", self.robot2object_frames_angular_dist[0, self.right_hand_id])
        # print("left_elbow: ", self.robot2object_frames_angular_dist[0, self.left_elbow_id])
        # print("right_elbow: ", self.robot2object_frames_angular_dist[0, self.right_elbow_id])
        # print("torso: ", self.robot2object_frames_angular_dist[0, self.torso_id])
        # print("********************************************************")

        # Object→goal: object_pos - goal_pos for each pair
        o2g_object_pos = self.object_frames.data.target_pos_source[env_ids][
            :, self._o2g_object_idx
        ]
        o2g_goal_pos = self.goal_frames.data.target_pos_source[env_ids][:, self._o2g_goal_idx]
        o2g = o2g_object_pos - o2g_goal_pos
        self.object2goal_frames_vel[env_ids] = (o2g - self.object2goal_frames_pos[env_ids]) / self.cfg.physics_dt
        self.object2goal_frames_pos[env_ids] = o2g
        self.object2goal_frames_euclidean_distance[env_ids] = torch.norm(o2g, dim=-1)

        if self.visualize:
            self._draw_o2g_lines()
            self._draw_r2o_lines()

        if self._imitation_ref_joint_cmd is not None:
            n = max(int(getattr(self.cfg, "imitation_gt_future_steps", 1)), 1)
            self.imitation_cmd_horizon = self._get_imitation_reference_joint_cmd_horizon(n)
            self.imitation_pos_horizon = self._get_imitation_reference_joint_pos_horizon(n)
            self.imitation_cmd = self._get_imitation_reference_joint_cmd()
            self.imitation_pos = self._get_imitation_reference_joint_pos()
            self.imitation_cmd_diff = self.joint_pos_cmd - self.imitation_cmd
            self.imitation_pos_diff = self.joint_pos - self.imitation_pos
            self.imitation_pos_sq = self.imitation_pos_diff * self.imitation_pos_diff
            self.imitation_mse = (self.imitation_pos_sq).mean(dim=-1)

        self._update_episode_task_metrics(env_ids)

    def _draw_o2g_lines(self) -> None:
        """Draw connecting cylinders between paired object and goal frames."""
        obj_w = self.object_frames.data.target_pos_w[:, self._o2g_object_idx]
        goal_w = self.goal_frames.data.target_pos_w[:, self._o2g_goal_idx]
        start = obj_w.reshape(-1, 3)
        end = goal_w.reshape(-1, 3)
        n = start.shape[0]

        midpoints = (start + end) * 0.5
        direction = end - start
        lengths = torch.norm(direction, dim=-1)

        default_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, -1)
        dir_norm = torch.nn.functional.normalize(direction, dim=-1, eps=1e-8)
        cross = torch.linalg.cross(default_dir, dir_norm)
        cross_mag = torch.norm(cross, dim=-1, keepdim=True)
        cross_safe = torch.where(
            cross_mag > 1e-6, cross / cross_mag,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(n, -1)
        )
        cos_angle = torch.clamp((default_dir * dir_norm).sum(-1), -1.0, 1.0)
        half_angle = torch.acos(cos_angle) * 0.5
        w = torch.cos(half_angle)
        xyz = cross_safe * torch.sin(half_angle).unsqueeze(-1)
        quats = torch.cat([w.unsqueeze(-1), xyz], dim=-1)

        scales = torch.ones(n, 3, device=self.device)
        scales[:, 2] = lengths

        self._o2g_line_vis.visualize(
            translations=midpoints,
            orientations=quats,
            scales=scales,
        )

    def _draw_r2o_lines(self) -> None:
        """Draw connecting cylinders, frame markers at robot poses, and target quat frames (lower opacity)."""
        robot_w = self.airec_frames.data.target_pos_w[:, self._r2o_robot_idx]
        object_w = self.object_frames.data.target_pos_w[:, self._r2o_object_idx]
        robot_quat_w = self.airec_frames.data.target_quat_w[:, self._r2o_robot_idx]
        start = robot_w.reshape(-1, 3)
        end = object_w.reshape(-1, 3)
        n = start.shape[0]

        midpoints = (start + end) * 0.5
        direction = end - start
        lengths = torch.norm(direction, dim=-1)

        default_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, -1)
        dir_norm = torch.nn.functional.normalize(direction, dim=-1, eps=1e-8)
        cross = torch.linalg.cross(default_dir, dir_norm)
        cross_mag = torch.norm(cross, dim=-1, keepdim=True)
        cross_safe = torch.where(
            cross_mag > 1e-6, cross / cross_mag,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(n, -1)
        )
        cos_angle = torch.clamp((default_dir * dir_norm).sum(-1), -1.0, 1.0)
        half_angle = torch.acos(cos_angle) * 0.5
        w = torch.cos(half_angle)
        xyz = cross_safe * torch.sin(half_angle).unsqueeze(-1)
        line_quats = torch.cat([w.unsqueeze(-1), xyz], dim=-1)

        line_scales = torch.ones(n, 3, device=self.device)
        line_scales[:, 2] = lengths

        frame_pos = robot_w.reshape(-1, 3)
        frame_quat = robot_quat_w.reshape(-1, 4)
        frame_scales = torch.ones(n, 3, device=self.device)

        r2o_dir = F.normalize(direction, dim=-1, eps=1e-8)
        robot_z = quat_apply(frame_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, -1))
        z_proj = robot_z - (robot_z * r2o_dir).sum(dim=-1, keepdim=True) * r2o_dir
        z_proj_norm = torch.norm(z_proj, dim=-1, keepdim=True)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, -1)
        z_fallback = world_up - (world_up * r2o_dir).sum(dim=-1, keepdim=True) * r2o_dir
        z_fallback_norm = torch.norm(z_fallback, dim=-1, keepdim=True).clamp(min=1e-8)
        z_target = torch.where(z_proj_norm > 0.1, z_proj / z_proj_norm.clamp(min=1e-8), z_fallback / z_fallback_norm)
        y_target = torch.linalg.cross(z_target, r2o_dir, dim=-1)
        y_target_norm = torch.norm(y_target, dim=-1, keepdim=True).clamp(min=1e-8)
        y_target = y_target / y_target_norm
        rot_mat = torch.stack([r2o_dir, y_target, z_target], dim=-1)
        target_quats = quat_from_matrix(rot_mat)
        target_scales = torch.ones(n, 3, device=self.device)

        translations = torch.cat([midpoints, frame_pos, frame_pos], dim=0)
        orientations = torch.cat([line_quats, frame_quat, target_quats], dim=0)
        scales = torch.cat([line_scales, frame_scales, target_scales], dim=0)
        marker_indices = torch.cat(
            [
                torch.zeros(n, dtype=torch.long, device=self.device),
                torch.ones(n, dtype=torch.long, device=self.device),
                torch.full((n,), 2, dtype=torch.long, device=self.device),
            ],
            dim=0,
        )

        self._r2o_line_vis.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )

    def _get_gt_object_manipulation(self) -> torch.Tensor:
        """Return concatenated object-manipulation observation components.

        Subclasses can override or extend. Default order:
        object_frames (pos, rot, vel), robot2object (pos, dist, rot, vel),
        object2goal (pos, dist), action_diff.
        With a loaded teleop demo (``joint_commands``), also appends a flattened window of
        reference commands (length ``imitation_gt_future_steps * num_joints``), matching
        :class:`tasks.airec.imitation.ImitationEnv` ``gt`` layout.
        """
        out = torch.cat(
            (
                # self.object_frames_pos.flatten(1),
                # self.object_frames_rot.flatten(1),
                # self.object_frames_vel.flatten(1),
                self.robot2object_frames_pos.flatten(1),
                self.robot2object_frames_euclidean_distance,
                self.robot2object_frames_angular_dist,
                # Use 6D representation to save space - slice the 3x3 to get 3x2
                self.robot2object_frames_relative_rot[..., :2].flatten(1),
                self.robot2object_frames_vel.flatten(1),
                # lifting
                self.object2liftgoal_distance.flatten(1),
                self.object2liftgoal_euclidean_distance.unsqueeze(-1),
                # goal stuff
                self.object2goal_frames_pos.flatten(1),
                self.object2goal_frames_euclidean_distance,
                self.action_diff,
            ),
            dim=-1,
        )
        # if self._imitation_ref_joint_cmd is not None:
            
        #     out = torch.cat(
        #         (
        #             out,
        #             self.imitation_cmd_horizon.flatten(1),
        #             self.imitation_pos_horizon.flatten(1),
        #             self.imitation_cmd_diff,
        #             self.imitation_pos_diff,
        #             self.imitation_mse.unsqueeze(-1),
        #         ),
        #         dim=-1,
        #     )
        return out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environments. Zeros object-manipulation velocities."""
        # Clear episode task metrics before super(): parent _reset_idx refreshes geometry and
        # calls _compute_object_manipulation_values, which would otherwise count the new pose.
        reset_ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        self._lift_success_episode[reset_ids] = False
        self._goal_success_episode[reset_ids] = False
        self._lift_success_timesteps[reset_ids] = 0
        self._episode_best_object2goal_min_m[reset_ids] = float("inf")
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.object_frames_vel[env_ids] = 0.0
        self.object2goal_frames_vel[env_ids] = 0.0
        self.robot2object_frames_vel[env_ids] = 0.0
        self.robot2object_frames_angular_dist[env_ids] = 0.0
        self.object2liftgoal_distance[env_ids] = 0.0
        self.object2liftgoal_euclidean_distance[env_ids] = 0.0

        applied, wrote_phys = self._apply_random_demo_episode_start(env_ids)
        if applied:
            self._compute_intermediate_values(env_ids)
            if wrote_phys:
                self.airec_frames_vel[env_ids] = 0.0
                self.object_frames_vel[env_ids] = 0.0
                self.object2goal_frames_vel[env_ids] = 0.0
                self.robot2object_frames_vel[env_ids] = 0.0
                self.robot2object_frames_angular_dist[env_ids] = 0.0
                self.object2liftgoal_distance[env_ids] = 0.0
                self.object2liftgoal_euclidean_distance[env_ids] = 0.0

    def _apply_random_demo_episode_start(
        self, env_ids: Sequence[int] | torch.Tensor | None
    ) -> tuple[bool, bool]:
        """Sample a demo timestep per env, set ``episode_length_buf``, optionally robot and object from demo.

        Requires ``cfg.imitation_demo_path`` (parent) so ``_imitation_ref_joint_pos`` is loaded.
        Object root pose uses ``_imitation_ref_object_pos_local`` / ``_imitation_ref_object_quat_w`` when
        present in the ``.npz``. Local positions are added to each env's ``scene.env_origins`` to
        obtain world placement.

        Environments with index ``< cfg.num_eval_envs`` (eval slice; see ``IsaacLabWrapper``) never
        receive this special start: they keep the standard randomized reset from the parent.

        Joint noise matches :meth:`AIRECEnv._reset_robot` when ``imitation_demo_start_joint_noise_scale``
        is 0 (body joints, then base xy/yaw deltas). Object pose adds ``object_noise`` in xy and yaw
        from ``reset_robot_quat_noise_scale`` when those are non-zero.

        When ``imitation_demo_reset_decay_timesteps_M`` > 0 (or start < 1), each env is included
        independently with probability :meth:`_imitation_demo_reset_probability` (Bernoulli); others
        keep the default reset from :meth:`AIRECEnv._reset_idx` / ``_reset_object``.

        Returns:
            ``(applied, wrote_physics)``: ``applied`` if the demo time index was set; ``wrote_physics``
            if robot and/or object state was written (caller should zero finite-diff velocities).
        """
        if not self.cfg.imitation_random_demo_start:
            return False, False
        ref = self._imitation_ref_joint_pos
        if ref is None:
            return False, False

        p = self._imitation_demo_reset_probability()
        if p <= 0.0:
            return False, False

        env_ids_t = self.robot._ALL_INDICES if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        n_eval = int(getattr(self.cfg, "num_eval_envs", 0) or 0)
        if n_eval > 0:
            env_ids_t = env_ids_t[env_ids_t >= n_eval]
            if env_ids_t.numel() == 0:
                return False, False

        n_all = int(env_ids_t.shape[0])
        if p < 1.0:
            demo_mask = torch.rand(n_all, device=self.device) < p
            if not demo_mask.any():
                return False, False
            env_ids_t = env_ids_t[demo_mask]

        T = int(ref.shape[0])
        if T < 1:
            return False, False

        t_min = max(0, int(self.cfg.imitation_demo_start_min_index))
        margin = max(0, int(self.cfg.imitation_demo_start_min_remaining_steps))
        upper = T - 1 - margin
        if self.cfg.imitation_demo_start_max_index is not None:
            upper = min(upper, int(self.cfg.imitation_demo_start_max_index))
        upper = max(t_min, min(upper, T - 1))
        if upper < t_min:
            raise ValueError(
                f"Empty imitation demo start range: T={T}, t_min={t_min}, upper={upper}. "
                "Lower imitation_demo_start_min_remaining_steps or adjust min/max index."
            )

        n = int(env_ids_t.shape[0])
        starts = torch.randint(
            low=t_min,
            high=upper + 1,
            size=(n,),
            device=self.device,
            dtype=torch.long,
        )
        self.episode_length_buf[env_ids_t] = starts

        wrote_physics = False
        if self.cfg.imitation_reset_joint_state_from_demo:
            joint_rows = ref[starts].clone()
            demo_ns = float(self.cfg.imitation_demo_start_joint_noise_scale)
            body_scale = demo_ns if demo_ns > 0.0 else float(self.cfg.reset_robot_joint_noise_scale)
            nab = len(self.actuated_body_dof_indices)
            joint_rows[:, self.actuated_body_dof_indices] += sample_uniform(
                -1.0, 1.0, (n, nab), device=self.device
            ) * body_scale
            if self.cfg.enable_base_control and self.actuated_base_dof_indices:
                ab = self.actuated_base_dof_indices
                joint_rows[:, ab[0]] += sample_uniform(-1.0, 1.0, (n,), device=self.device) * float(
                    self.cfg.reset_robot_pos_noise_x
                )
                joint_rows[:, ab[1]] += sample_uniform(-1.0, 1.0, (n,), device=self.device) * float(
                    self.cfg.reset_robot_pos_noise_y
                )
                joint_rows[:, ab[2]] += sample_uniform(-1.0, 1.0, (n,), device=self.device) * float(
                    self.cfg.reset_robot_quat_noise_scale
                )
            joint_rows = torch.clamp(
                joint_rows,
                self.robot_joint_pos_lower_limits,
                self.robot_joint_pos_upper_limits,
            )
            joint_vel = torch.zeros_like(joint_rows)
            self.robot.set_joint_position_target(joint_rows, env_ids=env_ids_t)
            self.robot.write_joint_state_to_sim(joint_rows, joint_vel, env_ids=env_ids_t)
            self.joint_pos_cmd[env_ids_t] = joint_rows
            self.prev_joint_pos_cmd[env_ids_t] = joint_rows
            self.actions[env_ids_t] = 0.0
            self.prev_actions[env_ids_t] = 0.0
            self.robot.write_root_velocity_to_sim(
                torch.zeros((n, 6), dtype=self.dtype, device=self.device),
                env_ids=env_ids_t,
            )
            wrote_physics = True

        obj_pos = getattr(self, "_imitation_ref_object_pos_local", None)
        obj_quat = getattr(self, "_imitation_ref_object_quat_w", None)
        if (
            self.cfg.imitation_reset_object_state_from_demo
            and obj_pos is not None
            and obj_quat is not None
            and hasattr(self, "object")
            and isinstance(self.object, RigidObject)
        ):
            pos_rows = obj_pos[starts]
            quat_rows = obj_quat[starts]
            env_origins = self.scene.env_origins[env_ids_t]
            pos_w = pos_rows + env_origins
            obj_n = float(self.cfg.object_noise)
            if obj_n > 0.0:
                obj_offset = sample_uniform(-obj_n, obj_n, (n, 3), device=self.device)
                obj_offset[:, 2] = 0.0
                pos_w = pos_w + obj_offset
            yaw_n = float(self.cfg.reset_robot_quat_noise_scale)
            if yaw_n > 0.0:
                yaw = sample_uniform(-1.0, 1.0, (n,), device=self.device) * yaw_n
                dq = quat_from_euler_xyz(
                    torch.zeros(n, device=self.device),
                    torch.zeros(n, device=self.device),
                    yaw,
                )
                quat_rows = quat_mul(dq, quat_rows)

            root = self.object.data.default_root_state.clone()[env_ids_t]
            root[:, 0:3] = pos_w
            root[:, 3:7] = quat_rows
            root[:, 7:13] = 0.0
            self.object.write_root_state_to_sim(root, env_ids_t)
            wrote_physics = True

        return True, wrote_physics

    def _reset_env(self, env_ids):
        """Reset task-specific state. Delegates to _reset_object."""
        self._reset_object(env_ids)

    def _reset_object(self, env_ids):
        """Reset object and goal for given environments. Implement in subclass."""
        raise NotImplementedError("Subclass must implement _reset_object")

    def _reset_bed_object_goal(self, env_ids: torch.Tensor) -> None:
        """Reset bed, object, and goal with randomization. Call from subclasses that have bed/object/goal."""
        env_origins = self.scene.env_origins[env_ids]
        n = len(env_ids)

        # Reset bed: default + env_origins + bed_random_x/y (x,y only)
        bed_default_state = self.bed.data.default_root_state.clone()[env_ids]
        bed_offset = torch.zeros(n, 3, device=self.device)
        bed_offset[:, 0] = sample_uniform(
            -self.cfg.bed_random_x, self.cfg.bed_random_x, (n,), device=self.device
        )
        bed_offset[:, 1] = sample_uniform(
            -self.cfg.bed_random_y, self.cfg.bed_random_y, (n,), device=self.device
        )
        bed_default_state[:, 0:3] = bed_default_state[:, 0:3] + env_origins + bed_offset
        self.bed.write_root_pose_to_sim(bed_default_state[:, :7], env_ids)

        # Bed center (top surface) for object placement
        bed_center = bed_default_state[:, 0:3].clone()
        bed_center[:, 2] += self.cfg.bed_depth / 2

        # Reset object: bed center + object_noise (deviation from bed center in x,y)
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_offset = sample_uniform(
            -self.cfg.object_noise, self.cfg.object_noise, (n, 3), device=self.device
        )
        object_offset[:, 2] = 0
        object_default_state[:, 0:3] = (
            bed_center
            + object_offset
            + torch.tensor([0, 0, self.cfg.object_radius], device=self.device).expand(n, 3)
        )
        self.object.write_root_state_to_sim(object_default_state, env_ids)

        # Reset goal: default + env_origins + goal_random_x/y (x,y only)
        goal_default_state = self.chair.data.default_root_state.clone()[env_ids]
        goal_offset = torch.zeros(n, 3, device=self.device)
        goal_offset[:, 0] = sample_uniform(
            -self.cfg.goal_random_x, self.cfg.goal_random_x, (n,), device=self.device
        )
        goal_offset[:, 1] = sample_uniform(
            -self.cfg.goal_random_y, self.cfg.goal_random_y, (n,), device=self.device
        )
        goal_default_state[:, 0:3] = goal_default_state[:, 0:3] + env_origins + goal_offset
        self.chair.write_root_pose_to_sim(goal_default_state[:, :7], env_ids)

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards using shared compute_rewards. Subclasses override for extras or stacking."""
        (
            _,
            robot2object_rewards,
            lift_reward,
            object_goal_reward,
            robot2object_rot_rewards,
            joint_vel_penalty,
            action_diff_penalty,
            object_center_vel_penalty,
        ) = compute_rewards(
            self.robot2object_frames_euclidean_distance,
            self.object2goal_frames_euclidean_distance,
            self.object2liftgoal_euclidean_distance,
            self.robot2object_frames_angular_dist,
            self.normalised_joint_vel,
            self.action_diff,
            self.object_frames_vel[:, self._object_center_vel_idx],
            self.cfg.reward_r2o_scale,
            self.cfg.reward_r2o_dist_std,
            self.cfg.reward_r2o_b,
            self.cfg.reward_lift_scale,
            self.cfg.reward_lift_dist_std,
            self.cfg.reward_o2g_scale,
            self.cfg.reward_o2g_dist_std,
            self._lift_success_episode.to(dtype=self.dtype),
        )

        imitation_reward = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        imitation_joint_mse = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        if self._imitation_ref_joint_cmd is not None:
            self.imitation_cmd = self._get_imitation_reference_joint_cmd()
            self.imitation_pos = self._get_imitation_reference_joint_pos()
            self.imitation_cmd_diff = self.joint_pos_cmd - self.imitation_cmd
            self.imitation_pos_diff = self.joint_pos - self.imitation_pos
            self.imitation_pos_sq = self.imitation_pos_diff * self.imitation_pos_diff
            imitation_joint_mse = (self.imitation_pos_sq).mean(dim=-1)
            self.imitation_mse = imitation_joint_mse
            imitation_reward = distance_reward(self.imitation_mse, std=0.3) * 5 


        r2o_rewards = robot2object_rewards 
        penalties = action_diff_penalty
        goal_rewards = object_goal_reward.mean(dim=-1)

        settle = self._goal_settle_satisfied()
        if settle is not None:
            goal_rewards = goal_rewards + settle.to(self.dtype) * self.cfg.goal_settle_success_bonus

        mean_r2o = torch.mean(self.robot2object_frames_euclidean_distance, dim=-1)
        p_demo = float(self._imitation_demo_reset_probability())
        p_demo_t = torch.full((self.num_envs,), p_demo, device=self.device, dtype=self.dtype)

        imitation_reward = imitation_reward * p_demo_t
        self.extras["log"] = {
            "airec_frames2goal_rewards": (robot2object_rewards),
            "lift_reward": (lift_reward),
            "object_goal_reward": (object_goal_reward),
            "robot2object_rot_rewards": (robot2object_rot_rewards),
            "success": (self.success.float()),
            "joint_vel_penalty": (joint_vel_penalty),
            "action_diff_penalty": (action_diff_penalty),
            "object_center_vel_penalty": (object_center_vel_penalty),
            "base_velocity": (torch.norm(self.joint_vel[:, self.actuated_base_dof_indices], dim=-1)),
            "imitation_reward": (imitation_reward),
            "imitation_joint_mse": (imitation_joint_mse),
            # Task metrics (absolute thresholds; not reward-scaled)
            "mean_robot_object_dist": mean_r2o,
            "imitation_demo_reset_p": p_demo_t,
        }
        if settle is not None:
            self.extras["log"]["goal_settle"] = settle.to(self.dtype)
        self.extras.setdefault("counters", {})
        self.extras["counters"].update(
            {
                "lift_success": self._lift_success_episode.to(dtype=self.dtype),
                "goal_success": self._goal_success_episode.to(dtype=self.dtype),
                "lift_success_timesteps": self._lift_success_timesteps.to(dtype=self.dtype),
                # Closest object→goal (m) this episode per env; trainer logs **min** over eval envs (not mean).
                "episode_best_object2goal_min_m": self._episode_best_object2goal_min_m,
                # Trainer → wandb: ``Eval episode counters / imitation_demo_reset_fraction``
                "imitation_demo_reset_fraction": p_demo_t,
            }
        )
        return torch.stack([r2o_rewards, lift_reward, goal_rewards, imitation_reward], dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation conditions.
        
        Returns:
            Tuple of (termination, truncation) tensors.
        """
        self._compute_intermediate_values()

        # object must not fall below bed height
        fall = self.object_frames_pos[:, self.center_id, 2] < self.cfg.bed_height

        # CoM behind base (X < 0 in base frame) = tipping
        com_tip_termination = self.com_pos_b[:, 0] < 0

        termination = fall | com_tip_termination
        settle = self._goal_settle_satisfied()
        if settle is not None:
            termination = termination | settle

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out


@torch.jit.script
def compute_rewards(
    robot2object_euclidean_distance: torch.Tensor,
    object2goal_euclidean_distance: torch.Tensor,
    object2lift_goal_distance: torch.Tensor,
    robot2object_angular_dist: torch.Tensor,
    normalised_joint_vel: torch.Tensor,
    action_diff: torch.Tensor,
    object_center_vel: torch.Tensor,
    r2o_scale: float,
    r2o_dist_std: float,
    r2o_b: float,
    lift_scale: float,
    lift_dist_std: float,
    o2g_scale: float,
    o2g_dist_std: float,
    lift_success_episode: torch.Tensor,
):
    r2o_coarse = distance_reward(robot2object_euclidean_distance, std=r2o_dist_std) * r2o_scale * r2o_b
    robot2object_rewards = torch.mean(r2o_coarse, dim=-1) * 1
    # prevent strange lifting behaviour by enforcing distance to arms
    max_dist = torch.max(robot2object_euclidean_distance, dim=-1).values
    can_lift = (max_dist < 0.5).float()

    lift_reward = distance_reward(object2lift_goal_distance, std=lift_dist_std) * 5
    lift_reward *= can_lift

    o2g_reward = distance_reward(object2goal_euclidean_distance, std=o2g_dist_std) * 25 #o2g_scale
    object2goal_rewards = o2g_reward * lift_success_episode * can_lift

    r2o_rot_coarse = distance_reward(robot2object_angular_dist, std=r2o_dist_std)
    robot2object_rot_rewards = torch.mean(r2o_rot_coarse, dim=-1) * 0

    joint_vel_penalty = -torch.sum(torch.square(normalised_joint_vel), dim=-1) * 0.00
    action_diff_penalty = -torch.sum(torch.square(action_diff), dim=-1) * 0.00
    # robot2object_vel_penalty = -torch.mean(torch.norm(robot2object_vel, dim=-1), dim=-1) * r2o_vel_scale
    object_center_vel_penalty = -torch.mean(torch.square(object_center_vel), dim=-1) * 1

    rewards = (
        robot2object_rewards + object2goal_rewards
        + lift_reward
    )

    return (
        rewards,
        robot2object_rewards,
        lift_reward,
        object2goal_rewards,
        robot2object_rot_rewards,
        joint_vel_penalty,
        action_diff_penalty,
        object_center_vel_penalty,
    )