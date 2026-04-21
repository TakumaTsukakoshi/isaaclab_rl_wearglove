# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, DeformableObjectCfg, DeformableObject
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
from collections.abc import Sequence

from tasks.airec.airec2_finger import (
    AIRECEnv,
    AIRECEnvCfg,
    angular_distance_reward,
    distance_cond_reward,
    distance_reward,
    joint_vel_penalty,
    rotation_distance,
)
from tasks.airec.mdp.rewards import geometryrl_b7_cloth_hanging_reward
from isaaclab.sensors import (
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@configclass
class WearEnvCfg(AIRECEnvCfg):
    """Glove + AIREC + Shadow Hand.

    - **AIREC** end-effectors (first fingers, thumbs, etc.) interact with the **deformable glove**.
    - **Shadow Hand** is a second articulation (:attr:`~tasks.airec.airec2_finger.AIRECEnvCfg.hand_cfg`).
      Goal frames (:attr:`thumb_goal_config`, :attr:`pinky_goal_config`, :attr:`wrist_goal_config`, …)
      point at Shadow Hand links so rewards and insertion logic test alignment of the garment / AIREC
      hands **toward** that hand — the standard setup for “can AIREC wear the glove toward Shadow Hand?”.
    - **Task-space RL** (optional): :mod:`tasks.airec.wear_finger_taskspace` uses dual-arm diff IK plus the same
      thumb / first-finger joints as ``assets/airec_finger`` defaults; see :class:`~tasks.airec.wear_finger_taskspace.WearFingerTaskSpaceEnvCfg`.
    """

    object_type = "deformable"

    # reset config
    reset_object_position_noise = 0.05
    reset_goal_position_noise = 0.01  # scale factor for -1 to 1 m
    default_goal_pos = [0.5, 0.5, 0.4]
    default_thumb_goal_pos = [0.70, -0.050, 1.07]
    default_pinky_goal_pos = [0.70, 0.050, 1.07]
    # default_object_pos = [0.27, 0.00, 1.07] # 0.13 # 1.07　default maybe for airec1
    # default_object_pos = [0.27, 0.00, 1.07] # airec1
    # default_object_pos = [0.26, 0.00, 0.85] # airec2 default
    # default_object_pos = [0.18, 0.00, 0.83] # airec2
    default_object_pos = [0.14, 0.00, 0.83] # airec2

    #: If True, use arXiv:2502.07005 App. B.7 (cloth-hanging) style reward via :mod:`tasks.airec.mdp.rewards`.
    use_geometryrl_b7_reward: bool = False

    object_goal_tracking_scale = 16.0
    object_goal_tracking_finegrained_scale = 5.0

    object_usd = os.path.join(
        _REPO_ROOT, "assets", "Glove", "GL_Gloves068", "GL_Gloves068_obj_revise.usd"
    )

    object_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1.0, 0.0, 0.0, 0.0]),#rot=[0.7071, 0.0, 0.7071, 0.0]
        spawn=UsdFileCfg(
            usd_path=object_usd,
            copy_from_source=True,
            visible=True,
            scale=(1.0, 1.2, 1.1),
            # scale=(1.0, 1.5, 1.4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.01, # default 0.005
                # rest_offset=0.006, # default 0.003
            ),


            deformable_props=DeformableBodyPropertiesCfg(
                deformable_enabled=True,
                kinematic_enabled=False,
                self_collision=False,
                simulation_hexahedral_resolution=20,  # default 10 
                # simulation_hexahedral_resolution=20,  # default 10 
                collision_simplification=True,
                collision_simplification_remeshing=True,
                collision_simplification_remeshing_resolution=10, # 40
                collision_simplification_target_triangle_count=0,
                collision_simplification_force_conforming=True,
                solver_position_iteration_count=16, # default 8
                # solver_position_iteration_count=32, # default 8
                contact_offset=0.010, # default
                rest_offset=0.006, # default
                # contact_offset=0.010,
                # rest_offset=0.006,
                # contact_offset=0.025,
                # rest_offset=0.020,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.2, 0.2),
            opacity=1.0,             
        ),   
        ),
        debug_vis=False,
    )
    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"

    # goal frame transformers
    goal_marker_cfg = FRAME_MARKER_CFG.copy()
    goal_marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    goal_marker_cfg.prim_path = "/World/Visuals/GoalMarker"
    # goal_marker_cfg.prim_path = "/World/envs/env_.*/Visuals/GoalMarker"

    # finger goal frame trandformers
    # Source frame must match a rigid body prim in the spawned AIREC USD (AIREC2 uses ``world``; ``base_link`` may be absent).
    thumb_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/world",
        debug_vis=False,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # prim_path="/World/envs/env_.*/Visuals/RightGoal/Geom",
                prim_path="/World/envs/env_.*/ShadowHand/robot0_thdistal",
                name="thumb_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    # rot=[0.7071, 0.0, 0.0, -0.7071]
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    pinky_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/world",
        debug_vis=False,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # prim_path="/World/envs/env_.*/Visuals/LeftGoal/Geom",
                prim_path="/World/envs/env_.*/ShadowHand/robot0_lfdistal",
                name="pinky_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    fore_goal_config: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/world",
            debug_vis=False,
            visualizer_cfg=goal_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # prim_path="/World/envs/env_.*/Visuals/LeftGoal/Geom",
                    prim_path="/World/envs/env_.*/ShadowHand/robot0_ffdistal",
                    name="fore_goal",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                        rot = [0.7071, -0.7071, 0.0, 0.0]
                    ),
                    )
                ],
            )
    
    middle_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/world",
        debug_vis=False,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # prim_path="/World/envs/env_.*/Visuals/LeftGoal/Geom",
                prim_path="/World/envs/env_.*/ShadowHand/robot0_mfdistal",
                name="middle_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
            )
        ],
    )

    ring_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/world",
        debug_vis=False,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # prim_path="/World/envs/env_.*/Visuals/LeftGoal/Geom",
                prim_path="/World/envs/env_.*/ShadowHand/robot0_rfdistal",
                name="ring_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot = [0.7071, -0.7071, 0.0, 0.0]
                ),
                )
            ],
        )


    wrist_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/world",
        debug_vis=False,
        visualizer_cfg=goal_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/ShadowHand/robot0_wrist",
                name="wrist_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    # rot = [0.7071, -0.7071, 0.0, 0.0]
                    rot = [0.0, 0.0, 0.0, 0.0]
                ),
            )
        ],
    )

    # glove edge point(N, S, E, W) 
    glove_north: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_north_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    glove_south: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_south_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    glove_east: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_east_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )

    glove_west: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_west_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            ),
        }
    )

    glove_cent: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_cent_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            ),
        }
    )

    # Visualization for thumb and pinky targets (in local robot coordinates)
    thumb_target_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/thumb_target_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),  # Orange
            ),
        }
    )

    pinky_target_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pinky_target_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),  # Cyan
            ),
        }
    )


class WearEnv(AIRECEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: WearEnvCfg

    def __init__(self, cfg: WearEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # define glove opening edges
        self.goal_north_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_north_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_south_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_south_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_east_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_east_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_west_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_west_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        self.goal_cent_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_cent_pos[:, :] = torch.tensor(self.cfg.default_goal_pos, device=self.device)
        self.goal_cent_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.depth_distance = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        self.depth_thumb_distance = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        self.depth_pinky_distance = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        # stretch
        self.garment_stretch_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.human_stretch_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.human_stretch_euclidean_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.goal_stretch_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_stretch_euclidean_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.direction = torch.zeros((self.num_envs, 3), dtype =torch.float, device=self.device)
        self.norm = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.wrist_origin = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.wrist_lateral_axis = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.unit_dir = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.thumb_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.pinky_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        
        # Identity rotations for thumb and pinky target visualization
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, -1)


        # right and left goal positions/rotations
        self.thumb_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.thumb_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.pinky_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.pinky_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.fore_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.fore_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.middle_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.middle_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.ring_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ring_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_wrist_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_wrist_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        self.garment_right_ee_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.garment_right_ee_euclidean_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.garment_left_ee_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.garment_left_ee_euclidean_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        # goal related tensors
        # self.right_ee_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_thumb_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_thumb_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.right_ee_thumb_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.right_ee_thumb_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        # self.left_ee_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ee_pinky_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ee_pinky_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.left_ee_pinky_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.left_ee_pinky_angular_distance = torch.zeros((self.num_envs,), device=self.device)

        # save reward weights so they can be adjusted online
        self.object_goal_tracking_scale = cfg.object_goal_tracking_scale
        self.object_goal_tracking_finegrained_scale = cfg.object_goal_tracking_finegrained_scale

        # default goal positions
        self.default_thumb_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.default_thumb_goal_pos[:, :] = torch.tensor(self.cfg.default_thumb_goal_pos, device=self.device)
        self.default_pinky_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.default_pinky_goal_pos[:, :] = torch.tensor(self.cfg.default_pinky_goal_pos, device=self.device)

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

        # debugging
        self.right_left_goal_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)


    def _setup_scene(self):
        super()._setup_scene()
        self._add_object_to_scene()
        self.goal_north_markers = VisualizationMarkers(self.cfg.glove_north)
        self.goal_south_markers = VisualizationMarkers(self.cfg.glove_south)
        self.goal_east_markers  = VisualizationMarkers(self.cfg.glove_east)
        self.goal_west_markers  = VisualizationMarkers(self.cfg.glove_west)
        self.goal_cent_markers  = VisualizationMarkers(self.cfg.glove_cent)
        # Initialize visualization markers for thumb and pinky targets
        self.thumb_target_markers = VisualizationMarkers(self.cfg.thumb_target_marker)
        self.pinky_target_markers = VisualizationMarkers(self.cfg.pinky_target_marker)
        

        self.thumb_goal_frame = FrameTransformer(self.cfg.thumb_goal_config)
        self.thumb_goal_frame.set_debug_vis(False)
        self.pinky_goal_frame = FrameTransformer(self.cfg.pinky_goal_config)
        self.pinky_goal_frame.set_debug_vis(False)
        self.fore_goal_frame = FrameTransformer(self.cfg.fore_goal_config)
        self.fore_goal_frame.set_debug_vis(False)
        self.middle_goal_frame = FrameTransformer(self.cfg.middle_goal_config)
        self.middle_goal_frame.set_debug_vis(False)
        self.ring_goal_frame = FrameTransformer(self.cfg.ring_goal_config)
        self.ring_goal_frame.set_debug_vis(False)
        self.wrist_goal_frame = FrameTransformer(self.cfg.wrist_goal_config)
        self.wrist_goal_frame.set_debug_vis(False)

        self.scene.sensors["pinky_goal_frame"] = self.pinky_goal_frame
        self.scene.sensors["thumb_goal_frame"] = self.thumb_goal_frame
        self.scene.sensors["fore_goal_frame"] = self.fore_goal_frame
        self.scene.sensors["middle_goal_frame"] = self.middle_goal_frame
        self.scene.sensors["ring_goal_frame"] = self.ring_goal_frame
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
                self.ee_distance,
                # euclidean distance (1,)
                self.ee_euclidean_distance.unsqueeze(1),
                # xyz diffs (3,)
                self.garment_right_ee_distance,
                # euclidean distance (1,)
                self.garment_right_ee_euclidean_distance.unsqueeze(1),
                # xyz diffs (3,)
                self.garment_left_ee_distance,
                # euclidean distance (1,)
                self.garment_left_ee_euclidean_distance.unsqueeze(1),
                # xyz diffs (3,)
                self.right_ee_thumb_distance,
                # euclidean distance (1,)
                self.right_ee_thumb_euclidean_distance.unsqueeze(1),
                ## xyz diffs (3,)
                self.left_ee_pinky_distance,
                # euclidean distances (1,)
                self.left_ee_pinky_euclidean_distance.unsqueeze(1),
                # angular distances (1,)
                self.right_ee_thumb_angular_distance.unsqueeze(1),
                # angular distances (1,)
                self.left_ee_pinky_angular_distance.unsqueeze(1),
                # xyz diffs (3,)
                # self.depth_distance.unsqueeze(1),
                # # xyz diffs (3,)
                # self.depth_thumb_distance.unsqueeze(1),
                # # xyz diffs (3,)
                # self.depth_pinky_distance.unsqueeze(1),

                # # xyz diffs (3,)
                # self.goal_wrist_pos,
                # # xyz diffs (3,)
                # self.goal_cent_pos,
                # # xyz diffs (3,)
                # self.goal_north_pos,
                # # xyz diffs (3,)
                # self.goal_south_pos,
            ),
            dim=-1,
        )
        return gt

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            e = self.robot._ALL_INDICES
        else:
            e = self._normalize_env_ids(env_ids)
        self.prev_actions[e] = 0.0

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.use_geometryrl_b7_reward:
            rewards, b7_log = geometryrl_b7_cloth_hanging_reward(self)
            self.extras["log"] = dict(b7_log)
            term_log = getattr(self, "_term_log", None)
            if term_log is not None:
                self.extras["log"].update(term_log)
        else:
            (
                rewards,
                r_stretch,
                r_right_ee_thumb_distance,
                r_left_ee_pinky_distance,
                r_depth_distance,
                r_depth_thumb_distance,
                r_depth_pinky_distance,
                r_angular_right_ee_thumb,
                r_angular_left_ee_pinky,
                r_right_ee_touch_distance,
                r_left_ee_touch_distance,
            ) = compute_rewards(
                self.reaching_object_goal_scale,
                self.reaching_ee_object_scale,
                self.stretch_object_scale,
                self.episode_length_buf,
                self.object_goal_tracking_scale,
                self.joint_vel_penalty_scale,
                self.goal_stretch_euclidean_distance,
                self.right_ee_thumb_euclidean_distance,
                self.left_ee_pinky_euclidean_distance,
                self.right_ee_thumb_angular_distance,
                self.left_ee_pinky_angular_distance,
                self.garment_right_ee_euclidean_distance,
                self.garment_left_ee_euclidean_distance,
                self.joint_vel,
                self.right_insert_success,
                self.left_insert_success,
                self.depth_distance,
                self.depth_thumb_distance,
                self.depth_pinky_distance,
                self.goal_wrist_pos[:, 2],
                self.north_edge_pos[:, 2],
                self.south_edge_pos[:, 2],
                self.thumb_target[:, 2],
                self.pinky_target[:, 2],
                self.cfg.minimal_width,
            )

            self.extras["log"] = {
                "r_stretch": r_stretch,
                "reach_reward_right": r_right_ee_thumb_distance,
                "reach_reward_left": r_left_ee_pinky_distance,
                "depth_reward": r_depth_distance,
                "depth_thumb_reward": r_depth_thumb_distance,
                "depth_pinky_reward": r_depth_pinky_distance,
                "angular_reward_right": r_angular_right_ee_thumb,
                "angular_reward_left": r_angular_left_ee_pinky,
                "touch_reward_right": r_right_ee_touch_distance,
                "touch_reward_left": r_left_ee_touch_distance,
            }

        if "tactile" in self.cfg.obs_list:
            self.extras["log"].update(
                {
                    "normalised_forces_left_x": self.normalised_forces[:, 0],
                    "normalised_forces_right_x": self.normalised_forces[:, 1],
                }
            )
        
        # Termination flags from ``AIRECEnv._get_dones`` (same control step; merged here because
        # ``_get_rewards`` overwrites ``extras["log"]`` after ``_get_dones`` runs).
        term_log = getattr(self, "_term_log", None)
        if term_log is not None:
            self.extras["log"].update(term_log)

        self.extras["counters"] = {}
        return rewards
    
    def _normalize_env_ids(self, env_ids):
        if isinstance(env_ids, int):
            return torch.tensor([env_ids], dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
    
    def _reset_goal_aperture(self, env_ids, thumb_offset=0.02, pinky_offset=0.02):
        # raw thumb / pinky positions
        thumb = self.pinky_goal_pos[env_ids]      # shape: (N, 3)
        pinky = self.thumb_goal_pos[env_ids]     # shape: (N, 3)

        # wrist origin
        wrist_origin = self.goal_wrist_pos[env_ids]   # shape: (N, 3)

        # wrist lateral axis in world frame
        # here you are assuming local/world lateral is +y
        wrist_lateral_axis = torch.tensor(
            [0.0, 1.0, 0.0],
            dtype=thumb.dtype,
            device=thumb.device
        ).unsqueeze(0).expand(len(env_ids), 3)

        # normalize lateral axis per env
        axis_norm = torch.norm(wrist_lateral_axis, dim=-1, keepdim=True).clamp_min(1e-8)
        wrist_lateral_axis = wrist_lateral_axis / axis_norm

        # project thumb and pinky onto wrist lateral axis
        thumb_vec = thumb - wrist_origin
        pinky_vec = pinky - wrist_origin

        thumb_t = torch.sum(thumb_vec * wrist_lateral_axis, dim=-1)   # shape: (N,)
        pinky_t = torch.sum(pinky_vec * wrist_lateral_axis, dim=-1)   # shape: (N,)

        # projected width along wrist lateral direction
        # Convert scalar distance to 3D vector by repeating across dimensions
        stretch_distance_scalar = torch.abs(thumb_t - pinky_t)  # shape: [N]
        self.human_stretch_distance[env_ids] = stretch_distance_scalar.unsqueeze(-1).expand(-1, 3)

        # if you also want actual outward offset targets
        direction = pinky - thumb
        norm = torch.norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
        # print(f"norm: {norm}")
        unit_dir = direction / norm

        # self.thumb_target[env_ids] = thumb - thumb_offset * unit_dir + self.scene.env_origins[env_ids]
        # self.pinky_target[env_ids] = pinky + pinky_offset * unit_dir + self.scene.env_origins[env_ids]
        self.thumb_target[env_ids] = thumb - thumb_offset * unit_dir 
        self.pinky_target[env_ids] = pinky + pinky_offset * unit_dir 

        # Euclidean distance between offset targets
        target_delta = self.thumb_target[env_ids] - self.pinky_target[env_ids]
        # print("target_delta", target_delta)
        self.human_stretch_euclidean_distance[env_ids] = torch.norm(target_delta, dim=-1)

        # optionally store these too
        self.wrist_origin[env_ids] = wrist_origin
        self.wrist_lateral_axis[env_ids] = wrist_lateral_axis

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

        # print("number of nodes", self.object.data.nodal_pos_w.size())
        # print(f"anchor_idx: {self.anchor_idx}")
        self.goal_north_pos[env_ids] = self.object.data.nodal_pos_w[env_ids, self.anchor_idx["north"], :] - self.scene.env_origins[env_ids]  
        self.goal_south_pos[env_ids] = self.object.data.nodal_pos_w[env_ids, self.anchor_idx["south"], :] - self.scene.env_origins[env_ids]  
        self.goal_east_pos[env_ids] = self.object.data.nodal_pos_w[env_ids, self.anchor_idx["east"], :] - self.scene.env_origins[env_ids] 
        self.goal_west_pos[env_ids] = self.object.data.nodal_pos_w[env_ids, self.anchor_idx["west"], :] - self.scene.env_origins[env_ids]
        self.goal_cent_pos[env_ids] = (self.goal_north_pos[env_ids]+self.goal_south_pos[env_ids])/2.0
        
        self.goal_wrist_pos[env_ids] = self.wrist_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.thumb_goal_pos[env_ids] = self.thumb_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.pinky_goal_pos[env_ids] = self.pinky_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        
        # Visualize goal markers
        # self.goal_north_markers.visualize(self.goal_north_pos, self.goal_north_rot)
        # self.goal_south_markers.visualize(self.goal_south_pos, self.goal_south_rot)
        # East/West use identity quaternion; must be indexed by env_ids on partial resets.
        # self.goal_east_markers.visualize(self.goal_east_pos[env_ids], self.identity_quat[env_ids])
        # self.goal_west_markers.visualize(self.goal_west_pos[env_ids], self.identity_quat[env_ids])
        # self.goal_cent_markers.visualize(self.goal_cent_pos, self.goal_cent_rot)
        
        # Visualize thumb and pinky targets (must index by env_ids: subset reset has |env_ids| < num_envs)
        self.thumb_target_markers.visualize(
            self.thumb_target[env_ids] + self.scene.env_origins[env_ids],
            self.identity_quat[env_ids],
        )
        self.pinky_target_markers.visualize(
            self.pinky_target[env_ids] + self.scene.env_origins[env_ids],
            self.identity_quat[env_ids],
        )
        self.goal_east_markers.visualize(self.goal_east_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids])
        self.goal_west_markers.visualize(self.goal_west_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids])
        # self.goal_north_markers.visualize(self.goal_north_pos[env_ids] , self.identity_quat[env_ids])
        # self.goal_south_markers.visualize(self.goal_south_pos[env_ids] , self.identity_quat[env_ids])
        # self.goal_cent_markers.visualize(self.goal_cent_pos[env_ids] , self.identity_quat[env_ids])

        self.garment_right_ee_distance[env_ids] = self.right_ee_pos[env_ids] - self.goal_west_pos[env_ids]  
        self.garment_right_ee_euclidean_distance[env_ids] = torch.norm(self.garment_right_ee_distance[env_ids], dim=1)
        self.garment_left_ee_distance[env_ids] = self.left_ee_pos[env_ids] - self.goal_east_pos[env_ids]
        self.garment_left_ee_euclidean_distance[env_ids] = torch.norm(self.garment_left_ee_distance[env_ids], dim=1)

        B = len(env_ids)
        dt_b = torch.full((B,), float(self.cfg.physics_dt), device=self.device)

        right_insert_out = self.insert_reward.step(
            pos_ee_s=self.right_ee_pos[env_ids],
            quat_ee_s=self.right_ee_rot[env_ids],
            pos_goal_s=self.thumb_goal_pos[env_ids],
            quat_goal_s=self.thumb_goal_rot[env_ids],
            dt=dt_b,
            idx=env_ids.to(self.device)
        )
        left_insert_out = self.insert_reward.step(
            pos_ee_s=self.left_ee_pos[env_ids],
            quat_ee_s=self.left_ee_rot[env_ids],
            pos_goal_s=self.pinky_goal_pos[env_ids],
            quat_goal_s=self.pinky_goal_rot[env_ids],
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

        self.top_wrist_distance[env_ids] = self.north_edge_pos[env_ids] - self.goal_wrist_pos[env_ids]
        self.under_wrist_distance[env_ids] = self.goal_wrist_pos[env_ids] - self.south_edge_pos[env_ids]
        self.top_wrist_euclidean_distance[env_ids] = torch.norm(self.top_wrist_distance[env_ids], dim=1)
        self.under_wrist_euclidean_distance[env_ids] = torch.norm(self.under_wrist_distance[env_ids], dim=1)
        # print(f"east: {self.west_edge_pos[0]} thumb_goal_pos:{self.thumb_goal_pos[0]}")
        self.right_ee_thumb_distance[env_ids] = self.right_ee_pos[env_ids] - self.thumb_target[env_ids]
        self.right_ee_thumb_euclidean_distance[env_ids] = torch.norm(self.right_ee_thumb_distance[env_ids], dim=1)
        self.right_ee_thumb_rotation[env_ids] = quat_mul(self.right_ee_rot[env_ids], quat_conjugate(self.thumb_goal_rot[env_ids]))
        self.right_ee_thumb_angular_distance[env_ids] = rotation_distance(self.right_ee_rot[env_ids], self.thumb_goal_rot[env_ids])
        # self.left_ee_goal_distance[env_ids] = self.left_l_ee_pos[env_ids] - self.pinky_goal_pos[env_ids]
        self.left_ee_pinky_distance[env_ids] = self.left_ee_pos[env_ids] - self.pinky_target[env_ids]
        self.left_ee_pinky_euclidean_distance[env_ids] = torch.norm(self.left_ee_pinky_distance[env_ids], dim=1)
        self.left_ee_pinky_rotation[env_ids] = quat_mul(self.left_ee_rot[env_ids], quat_conjugate(self.pinky_goal_rot[env_ids]))
        self.left_ee_pinky_angular_distance[env_ids] = rotation_distance(self.left_ee_rot[env_ids], self.pinky_goal_rot[env_ids])
        # print(f"left_ee_pinky_euclidean_distance: {self.left_ee_pinky_euclidean_distance[0]} right_ee_thumb_euclidean_distance: {self.right_ee_thumb_euclidean_distance[0]}")
        # shadow hand aperature
        self.goal_stretch_euclidean_distance[env_ids] = torch.abs(self.ee_euclidean_distance[env_ids] - self.human_stretch_euclidean_distance[env_ids])
        # print(f"goal_stretch_euclidean_distance: {self.goal_stretch_euclidean_distance[0]}, human_stretch_euclidean_distance: {self.human_stretch_euclidean_distance[0]}, ee_euclidean_distance: {self.ee_euclidean_distance[0]}")
        # print(f"garment_right_ee_euclidean_distance: {self.garment_right_ee_euclidean_distance[0]}, garment_left_ee_euclidean_distance: {self.garment_left_ee_euclidean_distance[0]}, right_ee_thumb_euclidean_distance: {self.right_ee_thumb_euclidean_distance[0]}, left_ee_pinky_euclidean_distance: {self.left_ee_pinky_euclidean_distance[0]}")
        # print(f"Goal stretch Euclidean distance: {self.goal_stretch_euclidean_distance[env_ids]}")
        self.depth_distance[env_ids] = torch.abs(self.goal_cent_pos[env_ids, 0] - self.goal_wrist_pos[env_ids, 0])
        # print(f"depth_distance: {self.depth_distance[0]}") # ~0.35
        self.depth_thumb_distance[env_ids] = torch.abs(self.goal_west_pos[env_ids, 0] - self.thumb_target[env_ids, 0])
        self.depth_pinky_distance[env_ids] = torch.abs(self.goal_east_pos[env_ids, 0] - self.pinky_target[env_ids, 0])
        # print(f"goal_west_pos: {self.goal_west_pos[0]} thumb_target: {self.thumb_target[0]}")
        # print(f"goal_east_pos: {self.goal_east_pos[0]} pinky_target: {self.pinky_target[0]}")

def compute_rewards(
    reaching_object_goal_scale: float,
    reaching_ee_object_scale: float,
    stretch_object_scale: float,
    episode_timestep_counter: torch.Tensor,
    object_goal_tracking_scale: float,
    joint_vel_penalty_scale: float,
    goal_stretch_euclidean_distance: torch.Tensor,
    right_ee_thumb_euclidean_distance: torch.Tensor,
    left_ee_pinky_euclidean_distance: torch.Tensor,
    right_ee_thumb_angular_distance: torch.Tensor,
    left_ee_pinky_angular_distance: torch.Tensor,
    garment_right_ee_euclidean_distance: torch.Tensor,
    garment_left_ee_euclidean_distance: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    right_insert_success: torch.Tensor,
    left_insert_success: torch.Tensor,
    depth_distance: torch.Tensor,
    depth_thumb_distance: torch.Tensor,
    depth_pinky_distance: torch.Tensor,
    wrist_height: torch.Tensor,
    top_height: torch.Tensor,
    bottom_height: torch.Tensor,
    thumb_height: torch.Tensor,
    pinky_height: torch.Tensor,
    minimal_width: float,
):
    rotation_object_goal_scale = 0.0 # 10.0
    reaching_object_goal_scale = 1.0    
    stretch_object_scale = 0.0
    touching_object_goal_scale = 0.0
    depth_reward_scale = 0.0
    depth_thumb_reward_scale = 0.0
    depth_pinky_reward_scale = 0.0

    # FOR REACHING (include condition))
    r_stretch = distance_reward(goal_stretch_euclidean_distance, std=0.05) * stretch_object_scale # 0.03
    # r_right_ee_thumb_distance = distance_cond_reward(garment_right_ee_euclidean_distance, right_ee_thumb_euclidean_distance, minimal_width, std=0.4) * reaching_object_goal_scale # default 0.4
    # r_left_ee_pinky_distance = distance_cond_reward(garment_left_ee_euclidean_distance, left_ee_pinky_euclidean_distance, minimal_width, std=0.2) * reaching_object_goal_scale * 0.0 # default 0.3
    r_right_ee_thumb_distance = distance_reward(right_ee_thumb_euclidean_distance, std=0.4) * reaching_object_goal_scale # default 0.4
    r_left_ee_pinky_distance = distance_reward(left_ee_pinky_euclidean_distance, std=0.2) * reaching_object_goal_scale * 0.0 # default 0.3
    r_right_ee_touch_distance = distance_reward(garment_right_ee_euclidean_distance, std=0.01) * touching_object_goal_scale 
    r_left_ee_touch_distance = distance_reward(garment_left_ee_euclidean_distance, std=0.01) * touching_object_goal_scale 
    # print(garment_right_ee_euclidean_distance[0], garment_left_ee_euclidean_distance[0])
    # FOR REACHING+INSERTING
    # r_garment_thumb_distance = distance_reward(goal_distance_thumb_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_pinky_distance = distance_reward(garment_pinky_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_fore_distance = distance_reward(garment_fore_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_middle_distance = distance_reward(garment_middle_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_ring_distance = distance_reward(garment_ring_euclidean_distance, std=0.09) * reaching_object_goal_scale
    r_depth_distance = distance_reward(depth_distance, std=0.1) * (top_height > wrist_height) * (wrist_height < bottom_height) * depth_reward_scale
    r_depth_thumb_distance = distance_reward(depth_thumb_distance, std=0.03) * (top_height > thumb_height) * (thumb_height < bottom_height) * depth_thumb_reward_scale
    r_depth_pinky_distance = distance_reward(depth_pinky_distance, std=0.06) * (top_height > pinky_height) * (pinky_height < bottom_height) * depth_pinky_reward_scale

    # FOR REACHING+INSERTING+TERMINATE
    # r_wrist_goal = wrist_distance_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, std=0.2) * reaching_object_goal_scale * 2.5  
    r_angular_right_ee_thumb = angular_distance_reward(right_ee_thumb_angular_distance, std=0.3) * rotation_object_goal_scale
    r_angular_left_ee_pinky = angular_distance_reward(left_ee_pinky_angular_distance, std=0.2) * rotation_object_goal_scale
    r_joint_vel = joint_vel_penalty(robot_joint_vel) * joint_vel_penalty_scale

    # minillion bonus reward
    # r_object_goal = object_goal_reward(right_ee_thumb_euclidean_distance, r_right_insert, std=0.3) * object_goal_tracking_scale
    # r_successed = success_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, minimal_distance)
    rewards = r_stretch  + r_right_ee_thumb_distance + r_left_ee_pinky_distance + r_depth_distance + r_depth_thumb_distance + r_depth_pinky_distance + r_angular_right_ee_thumb + r_angular_left_ee_pinky + r_right_ee_touch_distance + r_left_ee_touch_distance

    return (rewards, r_stretch,  r_right_ee_thumb_distance, r_left_ee_pinky_distance, r_depth_distance, r_depth_thumb_distance, r_depth_pinky_distance, r_angular_right_ee_thumb, r_angular_left_ee_pinky, r_right_ee_touch_distance, r_left_ee_touch_distance)
