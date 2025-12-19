# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, DeformableObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialCfg,   
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, DeformableBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
    sample_uniform,
)
import isaaclab.utils.math as math_utils
from collections.abc import Sequence

from tasks.airec.airec import AIRECEnv, AIRECEnvCfg, insert_success_reward, randomize_rotation, rotation_distance
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    # TiledCamera,
    # TiledCameraCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
import sys
sys.path.append("tasks/airec")
from insert_rew import InsertReward

torch.autograd.set_detect_anomaly(True)

@configclass
class ReachEnvCfg(AIRECEnvCfg):

    # reset config
    reset_object_position_noise = 0.05
    reset_goal_position_noise = 0.01  # scale factor for -1 to 1 m
    default_goal_pos = [0.5, 0.5, 0.4]
    default_right_goal_pos = [0.70, -0.050, 1.07]
    default_left_goal_pos = [0.70, 0.050, 1.07]
    default_object_pos = [0.27, 0.00, 1.07] # 0.13 # 1.07

    object_goal_tracking_scale = 16.0
    object_goal_tracking_finegrained_scale = 5.0

    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"

    # goal frame transformers
    goal_marker_cfg = FRAME_MARKER_CFG.copy()
    goal_marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    goal_marker_cfg.prim_path = "/World/Visuals/GoalMarker"
    # goal_marker_cfg.prim_path = "/World/envs/env_.*/Visuals/GoalMarker"


    right_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # prim_path="/World/envs/env_.*/Visuals/RightGoal/Geom",
                prim_path="/World/envs/env_.*/ShadowHand/robot0_thdistal",
                name="right_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    # rot=[0.7071, 0.0, 0.0, -0.7071]
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    left_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # prim_path="/World/envs/env_.*/Visuals/LeftGoal/Geom",
                prim_path="/World/envs/env_.*/ShadowHand/robot0_lfdistal",
                name="left_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    wrist_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/ShadowHand/robot0_wrist",
                name="wrist_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    # glove edge point(N, S, E, W) 
    glove_north: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    glove_south: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    glove_east: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )

    glove_west: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            ),
        }
    )

    glove_cent: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            ),
        }
    )


class ReachEnv(AIRECEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: ReachEnvCfg

    def __init__(self, cfg: ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # define glove opening edges

        # right and left goal positions/rotations
        self.right_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.right_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.left_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.left_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_wrist_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_wrist_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        # goal related tensors
        # self.right_ee_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_goal_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.right_ee_goal_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.right_ee_goal_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        # self.left_ee_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ee_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ee_goal_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.left_ee_goal_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.left_ee_goal_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        
        # save reward weights so they can be adjusted online
        self.object_goal_tracking_scale = cfg.object_goal_tracking_scale
        self.object_goal_tracking_finegrained_scale = cfg.object_goal_tracking_finegrained_scale

        # default goal positions
        self.default_right_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.default_right_goal_pos[:, :] = torch.tensor(self.cfg.default_right_goal_pos, device=self.device)
        self.default_left_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.default_left_goal_pos[:, :] = torch.tensor(self.cfg.default_left_goal_pos, device=self.device)

        # over/under distance reward
        self.wrist_ee_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.wrist_ee_euclidean_distance = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        self.top_wrist_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.top_wrist_euclidean_distance = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        self.under_wrist_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.under_wrist_euclidean_distance = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)

        # insert controller
        self.insert_reward = InsertReward(self.num_envs, device=self.device, inward_assume="+x")
        self.right_insert_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.right_insert_dwell = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.left_insert_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.left_insert_dwell = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        super()._setup_scene()
        self.goal_north_markers = VisualizationMarkers(self.cfg.glove_north)
        self.goal_south_markers = VisualizationMarkers(self.cfg.glove_south)
        self.goal_east_markers  = VisualizationMarkers(self.cfg.glove_east)
        self.goal_west_markers  = VisualizationMarkers(self.cfg.glove_west)
        self.goal_cent_markers  = VisualizationMarkers(self.cfg.glove_cent)

        self.right_goal_frame = FrameTransformer(self.cfg.right_goal_config)
        self.right_goal_frame.set_debug_vis(True)
        self.left_goal_frame = FrameTransformer(self.cfg.left_goal_config)
        self.left_goal_frame.set_debug_vis(True)
        self.wrist_goal_frame = FrameTransformer(self.cfg.wrist_goal_config)
        self.wrist_goal_frame.set_debug_vis(True)

        self.scene.sensors["left_goal_frame"] = self.left_goal_frame
        self.scene.sensors["right_goal_frame"] = self.right_goal_frame
        self.scene.sensors["wrist_goal_frame"] = self.wrist_goal_frame

        right_goal_path_env0 = "/World/envs/env_0/Visuals/RightGoal/Geom"
        left_goal_path_env0  = "/World/envs/env_0/Visuals/LeftGoal/Geom"

        goal_rb_cfg = sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),  
        )
        goal_rb_cfg.func(right_goal_path_env0, goal_rb_cfg)
        goal_rb_cfg.func(left_goal_path_env0,  goal_rb_cfg)

        self._right_goal_rb_path = "/World/envs/env_.*/Visuals/RightGoal/Geom"
        self._left_goal_rb_path  = "/World/envs/env_.*/Visuals/LeftGoal/Geom"

        self.right_goal_rb = RigidObject(RigidObjectCfg(prim_path=self._right_goal_rb_path))
        self.left_goal_rb  = RigidObject(RigidObjectCfg(prim_path=self._left_goal_rb_path))

        self.scene.rigid_objects["right_goal"] = self.right_goal_rb
        self.scene.rigid_objects["left_goal"]  = self.left_goal_rb

    def _get_gt(self):
        gt = torch.cat(
            (   
                # xyz diffs (3,)
                self.right_ee_goal_distance,
                # rotation quaternion (4,)
                self.right_ee_goal_rotation,
                # xyz diffs (3,)
                self.left_ee_goal_distance,
                # rotation quaternion (4,)
                self.left_ee_goal_rotation,
                # xyz diffs (3,)
                self.right_ee_left_ee_distance,
                # euclidean distance (1,)
                self.ee_euclidean_distance.unsqueeze(1),
                # goal eucledean distance 
                self.goal_euclidean_distance.unsqueeze(1),
                # goal ee euclidean distance
                self.ee_goal_euclidean_distance.unsqueeze(1),

                # angular distances (1,)
                self.right_ee_goal_angular_distance.unsqueeze(1),
                # euclidean distances (1,) [transform from (num_envs,) to (num_envs,1)]
                self.right_ee_goal_euclidean_distance.unsqueeze(1),
                # angular distances (1,)
                self.left_ee_goal_angular_distance.unsqueeze(1),
                # euclidean distances (1,) [transform from (num_envs,) to (num_envs,1)]
                self.left_ee_goal_euclidean_distance.unsqueeze(1),
            ),
            dim=-1,
        )
        return gt
    
    def _get_rewards(self) -> torch.Tensor:
        (
            rewards,
            r_stretch,
            r_wrist_goal,
            r_right_ee_goal,
            r_left_ee_goal,
            r_object_goal,
            r_joint_vel,
            r_angular_right_ee_goal,
            r_angular_left_ee_goal,
            r_right_insert,
            r_left_insert,
            r_success_reward
        ) = compute_rewards(
            self.reaching_object_goal_scale,
            self.reaching_ee_object_scale,
            self.stretch_object_scale,
            self.episode_length_buf,
            self.object_goal_tracking_scale,
            self.joint_vel_penalty_scale,
            self.object_pos,
            self.right_ee_goal_euclidean_distance,
            self.left_ee_goal_euclidean_distance,
            self.right_ee_goal_angular_distance,
            self.left_ee_goal_angular_distance,
            self.joint_vel,
            self.cfg.minimal_width,
            self.ee_euclidean_distance,
            self.ee_goal_euclidean_distance,
            self.right_ee_euclidean_distance,
            self.left_ee_euclidean_distance,
            self.right_ee_object_euclidean_distance,
            self.left_ee_object_euclidean_distance,
            self.right_ee_object_angular_distance,
            self.left_ee_object_angular_distance,
            self.rotation_ee_object_scale,
            self.rotation_object_goal_scale,
            self.right_insert_success,
            self.left_insert_success,
            self.wrist_ee_euclidean_distance,
            self.goal_wrist_pos,
            self.north_edge_pos,
            self.south_edge_pos,
        )
        def chk(name, x):
            if not torch.isfinite(x).all():
                bad = ~torch.isfinite(x)
                i = bad.nonzero()[0].tolist()
                print(f"[BAD] {name} first bad index={i}, value={x.view(-1)[bad.view(-1)][0]}")
                # env単位の報告（rewardは (N,) or (N,1) なので）
                env_id = i[0]
                print("env_id =", env_id)
                raise RuntimeError(f"{name} contains NaN/Inf")

        chk("r_stretch", r_stretch)
        chk("r_right_ee_goal", r_right_ee_goal)
        chk("r_left_ee_goal", r_left_ee_goal)
        chk("r_angular_right_ee_goal", r_angular_right_ee_goal)
        chk("r_angular_left_ee_goal", r_angular_left_ee_goal)
        chk("r_success_reward", r_success_reward)
        chk("rewards(total)", rewards)


        # Keep logs aligned with what's returned/computed
        self.extras["log"] = {
            "r_stretch": r_stretch,
            "r_wrist_goal": r_wrist_goal,
            "object_goal_tracking": r_object_goal,
            "joint_vel_penalty": r_joint_vel,
            "reach_object_goal_right" : r_right_ee_goal,
            "reach_object_goal_left" : r_left_ee_goal,
            "r_angular_right_ee_goal": r_angular_right_ee_goal,
            "r_angular_left_ee_goal": r_angular_left_ee_goal,
            "r_right_insert": r_right_insert,
            "r_left_insert": r_left_insert,
            "r_success_reward": r_success_reward
        }

        if "tactile" in self.cfg.obs_list:
            self.extras["log"].update(
                {
                    "normalised_forces_left_x": self.normalised_forces[:, 0],
                    "normalised_forces_right_x": self.normalised_forces[:, 1],
                }
            )

        self.extras["counters"] = {}
        return rewards
    
    def _normalize_env_ids(self, env_ids):
        if isinstance(env_ids, int):
            return torch.tensor([env_ids], dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)

    #     self.hand.write_root_state_to_sim(default_state, env_ids)
    def _reset_target_pose(self, env_ids):
        # Make sure this is already on the right device once
        default_state = self.hand.data.default_root_state.clone()[env_ids]

        pos_noise = sample_uniform(-0.02, 0.02, (len(env_ids), 3), device=self.device)

        init_pos = default_state[0, 0:3].unsqueeze(0).repeat(len(env_ids), 1)
      
        default_state[:, 0:3] = (
            init_pos
            + pos_noise * self.cfg.reset_goal_position_noise
            + self.scene.env_origins[env_ids]
        )

        init_rot = default_state[0, 3:7].unsqueeze(0).repeat(len(env_ids), 1)

        default_state[:, 3:7] = init_rot

        default_state[:, 7:] = 0.0

        joint_pos = self.hand.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)

        self.hand.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.hand.write_root_state_to_sim(default_state, env_ids=env_ids)


    def _compute_intermediate_values(self, reset=False, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values()
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # self.goal_cent_pos = (self.goal_north_pos+self.goal_south_pos+self.goal_east_pos+self.goal_west_pos)/4.0
        
        self.goal_wrist_pos[env_ids] = self.wrist_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.right_goal_pos[env_ids] = self.right_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.left_goal_pos[env_ids] = self.left_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.right_goal_rot[env_ids] = self.right_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        self.left_goal_rot[env_ids] = self.left_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        
        B = len(env_ids)
        dt_b = torch.full((B,), float(self.cfg.physics_dt), device=self.device)

        right_insert_out = self.insert_reward.step(
            pos_ee_s=self.right_ee_pos[env_ids],
            quat_ee_s=self.right_ee_rot[env_ids],
            pos_goal_s=self.right_goal_pos[env_ids],
            quat_goal_s=self.right_goal_rot[env_ids],
            dt=dt_b,
            idx=env_ids.to(self.device)
        )
        left_insert_out = self.insert_reward.step(
            pos_ee_s=self.left_ee_pos[env_ids],
            quat_ee_s=self.left_ee_rot[env_ids],
            pos_goal_s=self.left_goal_pos[env_ids],
            quat_goal_s=self.left_goal_rot[env_ids],
            dt=dt_b,
            idx=env_ids.to(self.device)
        )
        self.right_insert_success[env_ids] = right_insert_out["success"]       # (B,) bool
        self.right_insert_dwell[env_ids]   = right_insert_out["dwell"]         # (B,) float
        self.left_insert_success[env_ids] = left_insert_out["success"]       # (B,) bool
        self.left_insert_dwell[env_ids]   = left_insert_out["dwell"]         # (B,) float
        # print(f"dwell_right:{right_insert_out['dwell'][0]} dwell_left:{left_insert_out['dwell'][0]}")
        # print(f"state_right:{right_insert_out['state'][0]} state_left:{left_insert_out['state'][0]}")
        # print(f"d_right:{right_insert_out['d'][0:4]} r_right:{right_insert_out['r'][0:4]} c_right:{right_insert_out['c'][0:4]}")
        # print(f"d_left:{left_insert_out['d'][0:4]} r_left:{left_insert_out['r'][0:4]} c_left:{left_insert_out['c'][0:4]}")

        # upper/under distance
        self.wrist_ee_distance[env_ids] = self.ee_pos[env_ids] - self.goal_wrist_pos[env_ids]
        self.wrist_ee_euclidean_distance[env_ids] = torch.norm(self.wrist_ee_distance[env_ids], dim=1)

        # self.top_wrist_distance[env_ids] = self.north_edge_pos[env_ids] - self.goal_wrist_pos[env_ids]
        # self.under_wrist_distance[env_ids] = self.goal_wrist_pos[env_ids] - self.south_edge_pos[env_ids]
        # self.top_wrist_euclidean_distance[env_ids] = torch.norm(self.top_wrist_distance[env_ids], dim=1)
        # self.under_wrist_euclidean_distance[env_ids] = torch.norm(self.under_wrist_distance[env_ids], dim=1)
        # print(f"east: {self.west_edge_pos[0]} right_goal_pos:{self.right_goal_pos[0]}")
        self.right_ee_goal_distance[env_ids] = self.right_first_finger_pos[env_ids] - self.right_goal_pos[env_ids]
        self.right_ee_goal_euclidean_distance[env_ids] = torch.norm(self.right_ee_goal_distance[env_ids], dim=1)
        self.right_ee_goal_rotation[env_ids] = quat_mul(self.right_first_finger_rot[env_ids], quat_conjugate(self.right_goal_rot[env_ids]))
        self.right_ee_goal_angular_distance[env_ids] = rotation_distance(self.right_first_finger_rot[env_ids], self.right_goal_rot[env_ids])
        # self.left_ee_goal_distance[env_ids] = self.left_l_ee_pos[env_ids] - self.left_goal_pos[env_ids]
        # print(f"east: {self.east_edge_pos[0]} left_goal_pos:{self.left_goal_pos[0]}")
        self.left_ee_goal_distance[env_ids] = self.left_first_finger_pos[env_ids] - self.left_goal_pos[env_ids]
        self.left_ee_goal_euclidean_distance[env_ids] = torch.norm(self.left_ee_goal_distance[env_ids], dim=1)
        self.left_ee_goal_rotation[env_ids] = quat_mul(self.left_first_finger_rot[env_ids], quat_conjugate(self.left_goal_rot[env_ids]))
        self.left_ee_goal_angular_distance[env_ids] = rotation_distance(self.left_first_finger_rot[env_ids], self.left_goal_rot[env_ids])
        # print(self.right_ee_goal_euclidean_distance[0], self.left_ee_goal_euclidean_distance[0])

from tasks.airec.airec import distance_reward, joint_vel_penalty, object_goal_reward, angular_distance_reward, insert_success_reward, success_reward

@torch.jit.script
def compute_rewards(
    reaching_object_goal_scale: float,
    reaching_ee_object_scale: float,
    stretch_object_scale: float,
    episode_timestep_counter: torch.Tensor,
    object_goal_tracking_scale: float,
    joint_vel_penalty_scale: float,
    object_pos: torch.Tensor,
    right_ee_goal_euclidean_distance: torch.Tensor,
    left_ee_goal_euclidean_distance: torch.Tensor,
    right_ee_goal_angular_distance: torch.Tensor,
    left_ee_goal_angular_distance: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    minimal_distance: float,
    ee_euclidean_distance: torch.Tensor,
    ee_goal_euclidean_distance: torch.Tensor,
    right_ee_euclidean_distance: torch.Tensor,
    left_ee_euclidean_distance: torch.Tensor,
    right_ee_object_euclidean_distance: torch.Tensor,
    left_ee_object_euclidean_distance: torch.Tensor,
    right_ee_object_angular_distance: torch.Tensor,
    left_ee_object_angular_distance: torch.Tensor,
    rotation_ee_object_scale: float,
    rotation_object_goal_scale: float,
    right_insert_success: torch.Tensor,
    left_insert_success: torch.Tensor,
    wrist_ee_distance: torch.Tensor,
    wrist_pos: torch.Tensor,
    top_pos: torch.Tensor,
    under_pos: torch.Tensor,
):
    joint_vel_penalty_scale = 0
    rotation_ee_goal_scale = 1.5 # 10.0
    reaching_ee_goal_scale = 2.0
    stretch_object_scale = 1.0
    object_goal_tracking_scale = 0.0
    insert_scale = 0.0
    

    # reaching reward
    r_wrist_goal = distance_reward(wrist_ee_distance, std=0.2) * reaching_ee_goal_scale * 0.0
    # print(f"right_ee_goal_euclidean_distance {right_ee_goal_euclidean_distance[0]}, left_ee_goal_euclidean_distance {left_ee_goal_euclidean_distance[0]}")
    r_right_object_goal = distance_reward(right_ee_goal_euclidean_distance, std=0.09) * reaching_ee_goal_scale 
    r_left_object_goal = distance_reward(left_ee_goal_euclidean_distance, std=0.09) * reaching_ee_goal_scale 
    # angular distance rewards
    r_angular_right_ee_goal = angular_distance_reward(right_ee_goal_angular_distance, std=0.6) * rotation_ee_goal_scale
    r_angular_left_ee_goal = angular_distance_reward(left_ee_goal_angular_distance, std=0.6) * rotation_ee_goal_scale

    # joint velocity penalty
    r_joint_vel = joint_vel_penalty(robot_joint_vel) * joint_vel_penalty_scale
    r_stretch = distance_reward(ee_goal_euclidean_distance, std=0.01) * stretch_object_scale
    # insertion success reward
    r_right_insert = insert_success_reward(right_insert_success) * insert_scale
    r_left_insert = insert_success_reward(left_insert_success) * insert_scale

    # minillion bonus reward
    r_object_goal = object_goal_reward(right_ee_goal_euclidean_distance, r_right_insert, std=0.3) * object_goal_tracking_scale
    r_successed = success_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, minimal_distance)
    rewards = r_stretch + r_wrist_goal + r_right_object_goal + r_left_object_goal + r_object_goal + r_joint_vel + r_angular_right_ee_goal + r_angular_left_ee_goal + r_right_insert + r_left_insert + r_successed

    return (rewards, r_stretch, r_wrist_goal, r_right_object_goal, r_left_object_goal, r_object_goal, r_joint_vel, r_angular_right_ee_goal, r_angular_left_ee_goal,  r_right_insert, r_left_insert, r_successed)
