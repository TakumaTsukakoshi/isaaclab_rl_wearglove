# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

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
from collections.abc import Sequence

from tasks.airec.airec2_finger import AIRECEnv, AIRECEnvCfg, insert_success_reward, randomize_rotation, rotation_distance
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

@configclass
class WearEnvCfg(AIRECEnvCfg):

    # reset config
    reset_object_position_noise = 0.05
    reset_goal_position_noise = 0.01  # scale factor for -1 to 1 m
    default_goal_pos = [0.5, 0.5, 0.4]
    default_thumb_goal_pos = [0.70, -0.050, 1.07]
    default_pinky_goal_pos = [0.70, 0.050, 1.07]
    # default_object_pos = [0.27, 0.00, 1.07] # 0.13 # 1.07　default maybe for airec1
    default_object_pos = [0.27, 0.00, 1.07] # airec2

    object_goal_tracking_scale = 16.0
    object_goal_tracking_finegrained_scale = 5.0

    object_usd = '/home/tamon/code/isaaclab_rl_wearglove/assets/Glove/GL_Gloves068/GL_Gloves068_obj_revise.usd'

    object_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[1.0, 0.0, 0.0, 0.0]),#rot=[0.7071, 0.0, 0.7071, 0.0]
        spawn=UsdFileCfg(
            usd_path=object_usd,
            copy_from_source=True,
            visible=True,
            scale=(1.0, 1.3, 1.2),
            # scale=(1.0, 1.5, 1.4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.01, # default 0.005
                # rest_offset=0.006, # default 0.003
            ),


            deformable_props=DeformableBodyPropertiesCfg(
                deformable_enabled=True,
                kinematic_enabled=False,
                self_collision=True,
                simulation_hexahedral_resolution=40,  # default 10 
                collision_simplification=True,
                collision_simplification_remeshing=True,
                collision_simplification_remeshing_resolution=30, # 40
                collision_simplification_target_triangle_count=0,
                collision_simplification_force_conforming=True,
                contact_offset=0.015, # default
                rest_offset=0.010, # default
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
    thumb_goal_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
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
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
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
            prim_path="/World/envs/env_.*/Robot/base_link",
            debug_vis=True,
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
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
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
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
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
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
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
        self.right_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.pinky_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.left_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
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
        self.goal_north_markers = VisualizationMarkers(self.cfg.glove_north)
        self.goal_south_markers = VisualizationMarkers(self.cfg.glove_south)
        self.goal_east_markers  = VisualizationMarkers(self.cfg.glove_east)
        self.goal_west_markers  = VisualizationMarkers(self.cfg.glove_west)
        self.goal_cent_markers  = VisualizationMarkers(self.cfg.glove_cent)
        
        # Initialize visualization markers for thumb and pinky targets
        self.thumb_target_markers = VisualizationMarkers(self.cfg.thumb_target_marker)
        self.pinky_target_markers = VisualizationMarkers(self.cfg.pinky_target_marker)

        self.thumb_goal_frame = FrameTransformer(self.cfg.thumb_goal_config)
        self.thumb_goal_frame.set_debug_vis(True)
        self.pinky_goal_frame = FrameTransformer(self.cfg.pinky_goal_config)
        self.pinky_goal_frame.set_debug_vis(True)
        self.fore_goal_frame = FrameTransformer(self.cfg.fore_goal_config)
        self.fore_goal_frame.set_debug_vis(True)
        self.middle_goal_frame = FrameTransformer(self.cfg.middle_goal_config)
        self.middle_goal_frame.set_debug_vis(True)
        self.ring_goal_frame = FrameTransformer(self.cfg.ring_goal_config)
        self.ring_goal_frame.set_debug_vis(True)
        self.wrist_goal_frame = FrameTransformer(self.cfg.wrist_goal_config)
        self.wrist_goal_frame.set_debug_vis(True)

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
                self.right_ee_thumb_distance,
                # rotation quaternion (4,)
                # self.right_ee_goal_rotation,
                # xyz diffs (3,)
                self.left_ee_pinky_distance,
                # rotation quaternion (4,)
                # self.left_ee_goal_rotation,
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
                # xyz diffs (3,)
                # self.top_wrist_distance,
                # xyz diffs (3,)
                # self.under_wrist_distance,
                # euclidean distances (1,) 
                # self.top_wrist_euclidean_distance.unsqueeze(1),
                # euclidean distances (1,) 
                # self.under_wrist_euclidean_distance.unsqueeze(1),
                # xyz diffs (3,)
                # self.wrist_ee_distance,
                # # # euclidean distances (1,) 
                # self.wrist_ee_euclidean_distance.unsqueeze(1),

            ),
            dim=-1,
        )
        return gt
    
    def _get_rewards(self) -> torch.Tensor:
        (
            rewards,
            r_stretch,
            r_right_ee_thumb_distance,
            r_left_ee_pinky_distance,
            r_garment_right_ee_distance, 
            r_garment_left_ee_distance
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
            self.goal_wrist_pos,
            self.north_edge_pos,
            self.south_edge_pos,
            self.cfg.minimal_width
        )

        # Keep logs aligned with what's returned/computed
        self.extras["log"] = {
            "r_stretch": r_stretch,
            "reach_reward_right": r_right_ee_thumb_distance,
            "reach_reward_left": r_left_ee_pinky_distance,
            "garment_right_ee_distance_reward": r_garment_right_ee_distance,
            "garment_left_ee_distance_reward": r_garment_left_ee_distance
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
    
    def _reset_goal_aperture(self, env_ids, thumb_offset=0.03, pinky_offset=0.03):
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
        unit_dir = direction / norm

        self.thumb_target[env_ids] = thumb - thumb_offset * unit_dir + self.scene.env_origins[env_ids]
        self.pinky_target[env_ids] = pinky + pinky_offset * unit_dir + self.scene.env_origins[env_ids]

        # Euclidean distance between offset targets
        target_delta = self.thumb_target[env_ids] - self.pinky_target[env_ids]
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
        self.goal_north_pos = self.object.data.nodal_pos_w[:, self.anchor_idx["north"], :]
        self.goal_south_pos = self.object.data.nodal_pos_w[:, self.anchor_idx["south"], :]
        self.goal_east_pos = self.object.data.nodal_pos_w[:, self.anchor_idx["east"], :]
        self.goal_west_pos = self.object.data.nodal_pos_w[:, self.anchor_idx["west"], :]

        self.goal_cent_pos = (self.goal_north_pos+self.goal_south_pos+self.goal_east_pos+self.goal_west_pos)/4.0
        
        self.goal_wrist_pos[env_ids] = self.wrist_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.thumb_goal_pos[env_ids] = self.thumb_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.pinky_goal_pos[env_ids] = self.pinky_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.right_goal_rot[env_ids] = self.thumb_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        self.left_goal_rot[env_ids] = self.pinky_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        
        # Visualize goal markers
        # self.goal_north_markers.visualize(self.goal_north_pos, self.goal_north_rot)
        # self.goal_south_markers.visualize(self.goal_south_pos, self.goal_south_rot)
        # self.goal_east_markers.visualize(self.goal_east_pos, self.goal_east_rot)
        # self.goal_west_markers.visualize(self.goal_west_pos, self.goal_west_rot)
        # self.goal_cent_markers.visualize(self.goal_cent_pos, self.goal_cent_rot)
        
        # Visualize thumb and pinky targets
        self.thumb_target_markers.visualize(self.thumb_target, self.identity_quat)
        self.pinky_target_markers.visualize(self.pinky_target, self.identity_quat)
        
        self.garment_right_ee_distance[env_ids] = self.right_first_finger_pos[env_ids] - self.goal_east_pos[env_ids]
        self.garment_right_ee_euclidean_distance[env_ids] = torch.norm(self.garment_right_ee_distance[env_ids], dim=1)
        self.garment_left_ee_distance[env_ids] = self.left_first_finger_pos[env_ids] - self.goal_west_pos[env_ids]
        self.garment_left_ee_euclidean_distance[env_ids] = torch.norm(self.garment_left_ee_distance[env_ids], dim=1)

        B = len(env_ids)
        dt_b = torch.full((B,), float(self.cfg.physics_dt), device=self.device)

        right_insert_out = self.insert_reward.step(
            pos_ee_s=self.right_ee_pos[env_ids],
            quat_ee_s=self.right_ee_rot[env_ids],
            pos_goal_s=self.thumb_goal_pos[env_ids],
            quat_goal_s=self.right_goal_rot[env_ids],
            dt=dt_b,
            idx=env_ids.to(self.device)
        )
        left_insert_out = self.insert_reward.step(
            pos_ee_s=self.left_ee_pos[env_ids],
            quat_ee_s=self.left_ee_rot[env_ids],
            pos_goal_s=self.pinky_goal_pos[env_ids],
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

        self.top_wrist_distance[env_ids] = self.north_edge_pos[env_ids] - self.goal_wrist_pos[env_ids]
        self.under_wrist_distance[env_ids] = self.goal_wrist_pos[env_ids] - self.south_edge_pos[env_ids]
        self.top_wrist_euclidean_distance[env_ids] = torch.norm(self.top_wrist_distance[env_ids], dim=1)
        self.under_wrist_euclidean_distance[env_ids] = torch.norm(self.under_wrist_distance[env_ids], dim=1)
        # print(f"east: {self.west_edge_pos[0]} thumb_goal_pos:{self.thumb_goal_pos[0]}")
        self.right_ee_thumb_distance[env_ids] = self.right_first_finger_pos[env_ids] - self.thumb_goal_pos[env_ids]
        self.right_ee_thumb_euclidean_distance[env_ids] = torch.norm(self.right_ee_thumb_distance[env_ids], dim=1)
        self.right_ee_thumb_rotation[env_ids] = quat_mul(self.right_first_finger_rot[env_ids], quat_conjugate(self.right_goal_rot[env_ids]))
        self.right_ee_thumb_angular_distance[env_ids] = rotation_distance(self.right_first_finger_rot[env_ids], self.right_goal_rot[env_ids])
        # self.left_ee_goal_distance[env_ids] = self.left_l_ee_pos[env_ids] - self.pinky_goal_pos[env_ids]
        self.left_ee_pinky_distance[env_ids] = self.left_first_finger_pos[env_ids] - self.pinky_goal_pos[env_ids]
        self.left_ee_pinky_euclidean_distance[env_ids] = torch.norm(self.left_ee_pinky_distance[env_ids], dim=1)
        self.left_ee_pinky_rotation[env_ids] = quat_mul(self.left_first_finger_rot[env_ids], quat_conjugate(self.left_goal_rot[env_ids]))
        self.left_ee_pinky_angular_distance[env_ids] = rotation_distance(self.left_first_finger_rot[env_ids], self.left_goal_rot[env_ids])
        # print(self.left_ee_pinky_euclidean_distance[0], self.right_ee_thumb_euclidean_distance[0])
        
        # shadow hand aperature
        self.goal_stretch_euclidean_distance[env_ids] = torch.abs(self.ee_euclidean_distance[env_ids] - self.human_stretch_euclidean_distance[env_ids])
        # print(f"Goal stretch Euclidean distance: {self.goal_stretch_euclidean_distance[env_ids]}")

from tasks.airec.airec import distance_reward, distance_cond_reward, joint_vel_penalty, object_goal_reward, angular_distance_reward, insert_success_reward, success_reward, wrist_distance_reward

@torch.jit.script
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
    wrist_pos: torch.Tensor,
    top_pos: torch.Tensor,
    under_pos: torch.Tensor,
    minimal_width: float,
):
    joint_vel_penalty_scale = 0
    rotation_ee_object_scale = 0.0 # 1.0
    rotation_object_goal_scale = 0.0 # 10.0
    reaching_ee_object_scale = 0.0
    object_goal_tracking_scale = 0.0
    insert_scale = 0.0
    reaching_object_goal_scale = 2.0
    stretch_object_scale = 1.5
    garment_tracking_scale = 1.0

    # FOR REACHING (include condition))
    r_stretch = distance_reward(goal_stretch_euclidean_distance, std=0.01) * stretch_object_scale
    r_right_ee_thumb_distance = distance_cond_reward(garment_right_ee_euclidean_distance, right_ee_thumb_euclidean_distance, minimal_width, std=0.05) * reaching_object_goal_scale
    r_left_ee_pinky_distance = distance_cond_reward(garment_left_ee_euclidean_distance, left_ee_pinky_euclidean_distance, minimal_width, std=0.09) * reaching_object_goal_scale
    r_garment_right_ee_distance = distance_reward(garment_right_ee_euclidean_distance, std=0.01) * garment_tracking_scale
    r_garment_left_ee_distance = distance_reward(garment_left_ee_euclidean_distance, std=0.01) * garment_tracking_scale
    
    # FOR REACHING+INSERTING
    # r_garment_thumb_distance = distance_reward(goal_distance_thumb_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_pinky_distance = distance_reward(garment_pinky_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_fore_distance = distance_reward(garment_fore_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_middle_distance = distance_reward(garment_middle_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_ring_distance = distance_reward(garment_ring_euclidean_distance, std=0.09) * reaching_object_goal_scale

    # FOR REACHING+INSERTING+TERMINA
    # r_wrist_goal = wrist_distance_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, std=0.2) * reaching_object_goal_scale * 2.5  
    r_angular_right_ee_goal = angular_distance_reward(right_ee_thumb_angular_distance, std=0.4) * rotation_object_goal_scale
    r_angular_left_ee_goal = angular_distance_reward(left_ee_pinky_angular_distance, std=0.4) * rotation_object_goal_scale
    r_joint_vel = joint_vel_penalty(robot_joint_vel) * joint_vel_penalty_scale

    # minillion bonus reward
    # r_object_goal = object_goal_reward(right_ee_thumb_euclidean_distance, r_right_insert, std=0.3) * object_goal_tracking_scale
    # r_successed = success_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, minimal_distance)
    rewards = r_stretch  + r_right_ee_thumb_distance + r_left_ee_pinky_distance + r_garment_right_ee_distance + r_garment_left_ee_distance

    return (rewards, r_stretch,  r_right_ee_thumb_distance, r_left_ee_pinky_distance, r_garment_right_ee_distance, r_garment_left_ee_distance)
