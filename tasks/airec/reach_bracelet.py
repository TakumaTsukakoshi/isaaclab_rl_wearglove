# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialCfg,   
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, DeformableBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
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
    smooth_gate,
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
class ReachBraceletEnvCfg(AIRECEnvCfg):
    """Rigid bracelet + AIREC + Shadow Hand (same layout as :class:`~tasks.airec.wear_finger.WearEnvCfg`).

    - **Object** is a **rigid** USD (:attr:`object_usd` / :attr:`object_cfg`); not a deformable glove.
    - **AIREC** fingertips reach toward **Shadow Hand** goals (thumb / pinky / wrist frame transformers), same
      reward structure as the wear / reach-finger tasks.
    - **use_glove** is a legacy flag name: when ``True``, N/S/E/W “garment opening” features use **deformable**
      nodal anchors (glove-style). When ``False`` (default for this task), those features are off, but
      ``object_type="rigid"`` is kept so the **bracelet** still spawns.
    """

    # If True: deformable-style N/S/E/W markers and nodal goal features (glove code path).
    # If False (default): no cloth anchors; rigid bracelet remains in the scene when ``object_type=="rigid"``.
    use_glove: bool = False
    #: If True, use arXiv:2502.07005 App. B.7 (cloth-hanging) style reward via :mod:`tasks.airec.mdp.rewards`.
    use_geometryrl_b7_reward: bool = False

    object_type = "rigid"
    #: Hide parent ``AIRECEnv`` red kinematic anchor cuboids on the rim (still used for ``north_edge_pos`` etc.).
    show_anchor_rim_cuboids: bool = False

    # reset config
    reset_object_position_noise = 0.00
    #: Bracelet keeps ``object_cfg.init_state.rot`` on every reset (only position noise applies).
    randomize_object_rotation: bool = False
    reset_goal_position_noise = 0.01  # scale factor for -1 to 1 m
    default_goal_pos = [0.5, 0.5, 0.4]
    default_thumb_goal_pos = [0.70, -0.050, 1.07]
    default_pinky_goal_pos = [0.70, 0.050, 1.07]
    # default_object_pos = [0.27, 0.00, 1.07] # 0.13 # 1.07　default maybe for airec1
    # default_object_pos = [0.27, 0.00, 1.07] # airec1
    # default_object_pos = [0.26, 0.00, 0.85] # airec2 default
    # default_object_pos = [0.18, 0.00, 0.83] # airec2
    default_object_pos = [0.24, 0.00, 0.85] # airec2

    object_goal_tracking_scale = 16.0
    object_goal_tracking_finegrained_scale = 5.0

    #: Rim sample points on the rigid bracelet in the object's **root frame** (m). Each reset step:
    #: ``p_env = root_pos_w + quat_apply(root_quat_w, offset) - env_origin`` (same env-local convention as the glove's nodal rim).
    #: **Center** follows the glove task: ``goal_cent_pos = (goal_north_pos + goal_south_pos) / 2`` (midpoint of N–S, not the geometric centroid of four points).
    #: Tune offsets to match ``bracelet.usd`` (opening normal / lateral axes).
    bracelet_rim_offset_north: tuple[float, float, float] = (0.0, 0.03, 0.0)
    bracelet_rim_offset_south: tuple[float, float, float] = (0.0, -0.03, 0.0)
    bracelet_rim_offset_east: tuple[float, float, float] = (0.10, 0.0, 0.0)
    bracelet_rim_offset_west: tuple[float, float, float] = (-0.10, 0.0, 0.0)
    #: Opening-frame Z target for the wrist (m) in depth reward ``abs(z - desired)``.
    bracelet_desired_insert_depth: float = 0.0
    #: Soft in-opening gate ``exp(-max(0, radial^2-1)/std)``; larger = more lenient outside the ellipse.
    bracelet_inside_opening_std: float = 0.15

    object_usd = os.path.join(
        _REPO_ROOT, "assets", "Bracelet", "bracelet_b.usd"
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[0.7071, 0.0, 0.0, 0.7071]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[0.5, 0.5, -0.5, -0.5]),
        spawn=UsdFileCfg(
            usd_path=object_usd,
            copy_from_source=True,
            visible=True,
            scale=(1.0, 1.0, 1.0),
            # scale=(1.0, 1.5, 1.4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02, # default 0.005
                rest_offset=0.005, # default 0.003
            ),


            # Rigid body only (deformable hexa / remesh / soft-contact fields belong on DeformableBodyPropertiesCfg).
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=32,
                max_depenetration_velocity=0.5,
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

    # N/S/E/W/C rim markers (``VisualizationMarkers`` spheres); used when glove *or* rigid bracelet rim is active.
    bracelet_north: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_north_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    bracelet_south: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_south_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    bracelet_east: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_east_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )

    bracelet_west: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_west_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            ),
        }
    )

    bracelet_cent: VisualizationMarkersCfg = VisualizationMarkersCfg(
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

    fore_target_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/fore_target_marker",
        markers={
            "sphere":
            sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green
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

    finger_joint_names = [
            "robot0_FFJ3",
            "robot0_FFJ2",
            "robot0_FFJ1",
            "robot0_MFJ3",
            "robot0_MFJ2",
            "robot0_MFJ1",
            "robot0_RFJ3",
            "robot0_RFJ2",
            "robot0_RFJ1",
            "robot0_LFJ4",
            "robot0_LFJ3",
            "robot0_LFJ2",
            "robot0_LFJ1",
            "robot0_THJ4",
            "robot0_THJ3",
            "robot0_THJ2",
            "robot0_THJ1",
            "robot0_THJ0",
        ]


class ReachBraceletEnv(AIRECEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: ReachBraceletEnvCfg

    def __init__(self, cfg: ReachBraceletEnvCfg, render_mode: str | None = None, **kwargs):
        # ``WearEnv`` forces ``object_type="none"`` when ``use_glove`` is False so the deformable glove
        # disappears. For this rigid-bracelet task, keep ``object_type="rigid"`` unless we explicitly
        # drop the scene object (no rigid/deformable object requested).
        self._use_glove = bool(getattr(cfg, "use_glove", True))
        if self._use_glove and cfg.object_type != "deformable":
            raise ValueError(
                "ReachBraceletEnv: use_glove=True only supports object_type='deformable' (nodal rim). "
                "For the rigid bracelet USD, use use_glove=False (default) with object_type='rigid'."
            )
        if not self._use_glove and cfg.object_type != "rigid":
            cfg.object_type = "none"
        super().__init__(cfg, render_mode, **kwargs)

        # Opening-edge buffers (populated from deformable nodal anchors only when ``_use_glove``).
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
        # Last aperture reset: wrist reference and lateral axis in **env-local** / **world direction** respectively.
        self.wrist_origin = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.wrist_lateral_axis = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.unit_dir = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # Dynamic outward-biased reach targets (**env-local**), updated each step from live ShadowHand fingertips.
        self.thumb_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.pinky_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.fore_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        
        # Identity rotations for thumb and pinky target visualization
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, -1)


        # ShadowHand thdistal / lfdistal poses from goal FrameTransformers (names are legacy ``*_goal_*``).
        # ``*_pos`` are **live env-local** fingertip positions refreshed every step; ``*_rot`` are current tips.
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

        # Last ShadowHand root pose written in ``_reset_target_pose`` (same convention as ``write_root_state_to_sim``):
        # position is **sim world** (includes ``env_origins``); quaternion **world** wxyz.
        self.goal_hand_root_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_hand_root_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_hand_root_quat[:, 0] = 1.0

        # Opening-frame kinematics buffers (world rotation, env-local positions).
        # These MUST be persistent (num_envs, 3) tensors because _compute_intermediate_values may run on a subset of env_ids.
        self.wrist_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.north_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.south_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # Opening-frame offsets of dynamic ``thumb_target`` / ``pinky_target`` w.r.t. ``goal_cent_pos`` (bracelet frame).
        self.thumb_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.pinky_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.east_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.west_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.wrist_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.inside_opening_soft = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.insert_depth = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        # ShadowHand fore/middle/ring tips in opening frame (live ``*_goal_pos`` w.r.t. ``goal_cent_pos``).
        self.fore_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.middle_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ring_in_open = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # Ellipse ``(x/rx)^2 + (y/ry)^2`` in opening frame for each digit tip (actual goals, not outward targets).
        self.thumb_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.fore_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.middle_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ring_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.pinky_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        # Soft gate from summed outside-ellipse mass over all five tips (opening frame).
        self.fingers_inside_opening_soft = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
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
        self.object_goal_tracking_scale = self.cfg.object_goal_tracking_scale
        self.object_goal_tracking_finegrained_scale = self.cfg.object_goal_tracking_finegrained_scale

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
        self.finger_joint_ids, _ = self.hand.find_joints(self.cfg.finger_joint_names)
        self._shadow_hand_finger_hold = torch.zeros(
            (self.num_envs, len(self.finger_joint_ids)),
            device=self.device,
            dtype=self.hand.data.joint_pos.dtype,
        )
        self.wrist_xy_center_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.wrist_center_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
    
    def _apply_action(self) -> None:
        super()._apply_action()
        # hold finger targets
        finger_target_pos = self._shadow_hand_finger_hold
        zv = torch.zeros_like(finger_target_pos)
        self.hand.set_joint_position_target(
            finger_target_pos,
            joint_ids=self.finger_joint_ids,
        )
        self.hand.set_joint_velocity_target(zv, joint_ids=self.finger_joint_ids)

    def _setup_scene(self):
        super()._setup_scene()
        # Parent enables upper-fingertip frame debug by default; it draws extra axes near the workspace.
        self.left_upper_ee_frame.set_debug_vis(False)
        self.right_upper_ee_frame.set_debug_vis(False)
        # Rigid / deformable task object (bracelet) is added whenever ``object_type != "none"``.
        if self.cfg.object_type != "none":
            self._add_object_to_scene()
        # N/S/E/W/C markers: deformable glove uses nodal rim; rigid bracelet uses root pose + cfg offsets.
        if self._use_glove or self.cfg.object_type == "rigid":
            self.goal_north_markers = VisualizationMarkers(self.cfg.bracelet_north)
            self.goal_south_markers = VisualizationMarkers(self.cfg.bracelet_south)
            self.goal_east_markers = VisualizationMarkers(self.cfg.bracelet_east)
            self.goal_west_markers = VisualizationMarkers(self.cfg.bracelet_west)
            self.goal_cent_markers = VisualizationMarkers(self.cfg.bracelet_cent)
        else:
            self.goal_north_markers = None
            self.goal_south_markers = None
            self.goal_east_markers = None
            self.goal_west_markers = None
            self.goal_cent_markers = None
        # Initialize visualization markers for thumb and pinky targets
        self.thumb_target_markers = VisualizationMarkers(self.cfg.thumb_target_marker)
        self.pinky_target_markers = VisualizationMarkers(self.cfg.pinky_target_marker)
        

        self.thumb_goal_frame = FrameTransformer(self.cfg.thumb_goal_config)
        self.thumb_goal_frame.set_debug_vis(True)
        self.pinky_goal_frame = FrameTransformer(self.cfg.pinky_goal_config)
        self.pinky_goal_frame.set_debug_vis(True)
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
                self.depth_distance.unsqueeze(1),
                # # xyz diffs (3,)
                # self.depth_thumb_distance.unsqueeze(1),
                # # xyz diffs (3,)
                # self.depth_pinky_distance.unsqueeze(1),

                # # xyz diffs (3,)
                self.goal_wrist_pos,
                # # xyz diffs (3,)
                self.goal_cent_pos,
                # # xyz diffs (3,)
                self.goal_north_pos,
                # # xyz diffs (3,)
                self.goal_south_pos,
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

        # When ``_use_glove`` is False the base reset can skip goal-hand setup (``object_type=="none"`` path).
        # For rigid bracelet we still refresh ShadowHand pose and thumb/pinky aperture after reset.
        if not self._use_glove:
            self._reset_target_pose(e)
            # Refresh transforms before aperture logic (thumb/pinky goal frames depend on ShadowHand pose).
            self._compute_intermediate_values(env_ids=e)
            self._reset_goal_aperture(e)

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.use_geometryrl_b7_reward:
            rewards, b7_log = geometryrl_b7_cloth_hanging_reward(self)
            self.extras["log"] = dict(b7_log)
            term_log = getattr(self, "_term_log", None)
            if term_log is not None:
                self.extras["log"].update(term_log)
        else:
            # Without deformable rim features, keep the same reward API with neutral garment / depth terms.
            if self._use_glove or self.cfg.object_type == "rigid":
                garment_r = self.garment_right_ee_euclidean_distance
                garment_l = self.garment_left_ee_euclidean_distance
                depth = self.depth_distance
                depth_t = self.depth_thumb_distance
                depth_p = self.depth_pinky_distance
            else:
                garment_r = torch.full((self.num_envs,), 1e3, device=self.device, dtype=torch.float32)
                garment_l = torch.full((self.num_envs,), 1e3, device=self.device, dtype=torch.float32)
                depth = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
                depth_t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
                depth_p = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

            (
                rewards,
                r_right_ee_thumb_distance,
                r_left_ee_pinky_distance,
                r_depth_distance,
                r_depth_thumb_distance,
                r_depth_pinky_distance,
                r_angular_right_ee_thumb,
                r_angular_left_ee_pinky,
                r_right_ee_touch_distance,
                r_left_ee_touch_distance,
                r_joint_vel,
                r_wrist_center_alignment,
                r_wrist_center_3d_alignment,
                r_right_angle_penalty,
                r_left_angle_penalty,
            ) = compute_rewards(
                self.reaching_object_goal_scale,
                self.reaching_ee_object_scale,
                self.stretch_object_scale,
                self.episode_length_buf,
                self.object_goal_tracking_scale,
                self.joint_vel_penalty_scale,
                self.ee_euclidean_distance,
                self.goal_stretch_euclidean_distance,
                self.right_ee_thumb_euclidean_distance,
                self.left_ee_pinky_euclidean_distance,
                self.right_ee_thumb_angular_distance,
                self.left_ee_pinky_angular_distance,
                garment_r,
                garment_l,
                self.joint_vel,
                self.right_insert_success,
                self.left_insert_success,
                depth,
                depth_t,
                depth_p,
                self.inside_opening_soft,
                self.fingers_inside_opening_soft,
                self.wrist_radial_normalized,
                # Use opening-frame Y (north/south axis) instead of world/env Z so wrist randomization doesn't break gating.
                self.wrist_in_open[:, 1],
                self.north_in_open[:, 1],
                self.south_in_open[:, 1],
                self.thumb_in_open[:, 1],
                self.pinky_in_open[:, 1],
                self.cfg.minimal_width,
                self.wrist_xy_center_distance,
                self.wrist_center_distance,
                )

            self.extras["log"] = {
                "reach_reward_right": r_right_ee_thumb_distance,
                "reach_reward_left": r_left_ee_pinky_distance,
                "depth_reward": r_depth_distance,
                "depth_thumb_reward": r_depth_thumb_distance,
                "depth_pinky_reward": r_depth_pinky_distance,
                "wrist_center_alignment_reward": r_wrist_center_alignment,
                "wrist_center_3d_alignment_reward": r_wrist_center_3d_alignment,
                "right_angle_penalty": r_right_angle_penalty,
                "left_angle_penalty": r_left_angle_penalty,
                "angular_reward_right": r_angular_right_ee_thumb,
                "angular_reward_left": r_angular_left_ee_pinky,
                "touch_reward_right": r_right_ee_touch_distance,
                "touch_reward_left": r_left_ee_touch_distance,
                "joint_vel_reward": r_joint_vel,
                "inside_opening_soft": self.inside_opening_soft,
                "wrist_radial_normalized": self.wrist_radial_normalized,
                "insert_depth": self.insert_depth,
                "depth_distance": self.depth_distance,
                "fingers_inside_opening_soft": self.fingers_inside_opening_soft,
                "thumb_radial_normalized": self.thumb_radial_normalized,
                "fore_radial_normalized": self.fore_radial_normalized,
                "middle_radial_normalized": self.middle_radial_normalized,
                "ring_radial_normalized": self.ring_radial_normalized,
                "pinky_radial_normalized": self.pinky_radial_normalized,
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

        # Randomize pitch (Y-axis rotation) by ±5° in world frame, applied on top of the default root orientation.
        B = int(len(env_ids))
        pitch_rad = sample_uniform(
            torch.deg2rad(torch.tensor(-5.0, device=self.device, dtype=torch.float32)),
            torch.deg2rad(torch.tensor(5.0, device=self.device, dtype=torch.float32)),
            (B,),
            device=self.device,
        )
        yaw_rad = sample_uniform(
            torch.deg2rad(torch.tensor(-5.0, device=self.device, dtype=torch.float32)),
            torch.deg2rad(torch.tensor(5.0, device=self.device, dtype=torch.float32)),
            (B,),
            device=self.device,
        )
        zero = torch.zeros_like(pitch_rad)
        q_yaw = quat_from_euler_xyz(zero, zero, yaw_rad)  # (B, 4) wxyz
        q_pitch = quat_from_euler_xyz(zero, pitch_rad, zero)  # (B, 4) wxyz
        q_yaw_pitch = quat_mul(q_yaw, q_pitch)
        default_state[:, 3:7] = quat_mul(q_yaw_pitch, init_rot)

        default_state[:, 7:] = 0.0

        joint_pos = self.hand.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)

        self.hand.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.hand.write_root_state_to_sim(default_state, env_ids=env_ids)
        # Cache root pose actually written (world pos + world quat) for aperture / frame logic.
        self.goal_hand_root_pos[env_ids] = default_state[:, 0:3].to(dtype=self.goal_hand_root_pos.dtype)
        self.goal_hand_root_quat[env_ids] = default_state[:, 3:7].to(dtype=self.goal_hand_root_quat.dtype)
        self._shadow_hand_finger_hold[env_ids] = joint_pos[:, self.finger_joint_ids].clone()

    def _update_goal_aperture_targets(self, env_ids, thumb_offset=0.03, pinky_offset=0.02) -> None:
        """Recompute outward reach targets and stretch scalars from **current** ShadowHand geometry.

        Live fingertip **env-local** positions are ``thumb_goal_pos`` / ``pinky_goal_pos`` (FrameTransformer to
        ``robot0_thdistal`` / ``robot0_lfdistal``, updated each physics step). ``thumb_target`` / ``pinky_target``
        are **env-local** points offset outward along the thumb→pinky line. Wrist lateral width still uses
        hand-local ``+Y`` mapped by ``goal_hand_root_quat`` (cached at reset from the written root pose).
        """
        B = int(len(env_ids))
        thumb_current = self.thumb_goal_pos[env_ids]
        pinky_current = self.pinky_goal_pos[env_ids]
        wrist_origin = self.goal_wrist_pos[env_ids]

        dt = self.thumb_goal_pos.dtype
        local_lateral = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=dt).unsqueeze(0).expand(B, 3)
        q_hand = self.goal_hand_root_quat[env_ids].to(dtype=dt)
        wrist_lateral_axis = quat_apply(q_hand, local_lateral)

        axis_norm = torch.norm(wrist_lateral_axis, dim=-1, keepdim=True).clamp_min(1e-6)
        wrist_lateral_axis = wrist_lateral_axis / axis_norm

        thumb_vec = thumb_current - wrist_origin
        pinky_vec = pinky_current - wrist_origin
        thumb_t = torch.sum(thumb_vec * wrist_lateral_axis, dim=-1)
        pinky_t = torch.sum(pinky_vec * wrist_lateral_axis, dim=-1)
        stretch_distance_scalar = torch.abs(thumb_t - pinky_t)
        self.human_stretch_distance[env_ids] = stretch_distance_scalar.unsqueeze(-1).expand(-1, 3)

        direction = pinky_current - thumb_current
        norm = torch.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
        unit_dir = direction / norm

        self.thumb_target[env_ids] = thumb_current - thumb_offset * unit_dir
        self.pinky_target[env_ids] = pinky_current + pinky_offset * unit_dir

        target_delta = self.thumb_target[env_ids] - self.pinky_target[env_ids]
        self.human_stretch_euclidean_distance[env_ids] = torch.norm(target_delta, dim=-1)

        self.wrist_origin[env_ids] = wrist_origin
        self.wrist_lateral_axis[env_ids] = wrist_lateral_axis

    def _reset_goal_aperture(self, env_ids, thumb_offset=0.03, pinky_offset=0.02):
        """Reset-time hook: same as per-step aperture update (expects fresh fingertip frames if called after compute)."""
        self._update_goal_aperture_targets(env_ids, thumb_offset=thumb_offset, pinky_offset=pinky_offset)

    def _bracelet_rim_goals_env_local(self, env_ids: torch.Tensor) -> None:
        """Set ``goal_{north,south,east,west}_pos`` and ``goal_cent_pos`` (env-local) for a rigid bracelet.

        Glove uses mesh nodes; here ``p_env = root_pos_w + quat_apply(root_quat_w, offset_b) - env_origin`` with
        body-frame offsets from :attr:`ReachBraceletEnvCfg.bracelet_rim_offset_*`. Center matches the glove task:
        ``(north + south) / 2``.
        """
        if self.cfg.object_type != "rigid" or not hasattr(self, "object") or self.object is None:
            return
        B = int(env_ids.shape[0])
        root_p = self.object.data.root_pos_w[env_ids]
        root_q = self.object.data.root_quat_w[env_ids]
        origins = self.scene.env_origins[env_ids]

        def _expand_off(t: tuple[float, float, float]) -> torch.Tensor:
            return torch.tensor(t, device=self.device, dtype=torch.float32).unsqueeze(0).expand(B, 3)

        self.goal_north_pos[env_ids] = (
            root_p + quat_apply(root_q, _expand_off(self.cfg.bracelet_rim_offset_north)) - origins
        )
        self.goal_south_pos[env_ids] = (
            root_p + quat_apply(root_q, _expand_off(self.cfg.bracelet_rim_offset_south)) - origins
        )
        self.goal_east_pos[env_ids] = (
            root_p + quat_apply(root_q, _expand_off(self.cfg.bracelet_rim_offset_east)) - origins
        )
        self.goal_west_pos[env_ids] = (
            root_p + quat_apply(root_q, _expand_off(self.cfg.bracelet_rim_offset_west)) - origins
        )
        self.goal_cent_pos[env_ids] = (self.goal_north_pos[env_ids] + self.goal_south_pos[env_ids]) / 2.0

    def _compute_intermediate_values(self, reset=False, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values()
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if self._use_glove:
            # Deformable-object rim points (nodal anchors).
            self.goal_north_pos[env_ids] = (
                self.object.data.nodal_pos_w[env_ids, self.anchor_idx["north"], :] - self.scene.env_origins[env_ids]
            )
            self.goal_south_pos[env_ids] = (
                self.object.data.nodal_pos_w[env_ids, self.anchor_idx["south"], :] - self.scene.env_origins[env_ids]
            )
            self.goal_east_pos[env_ids] = (
                self.object.data.nodal_pos_w[env_ids, self.anchor_idx["east"], :] - self.scene.env_origins[env_ids]
            )
            self.goal_west_pos[env_ids] = (
                self.object.data.nodal_pos_w[env_ids, self.anchor_idx["west"], :] - self.scene.env_origins[env_ids]
            )
            self.goal_cent_pos[env_ids] = (self.goal_north_pos[env_ids] + self.goal_south_pos[env_ids]) / 2.0
        elif self.cfg.object_type == "rigid":
            self._bracelet_rim_goals_env_local(env_ids)
        else:
            # No garment rim (e.g. ``object_type=="none"``).
            self.goal_north_pos[env_ids] = 0.0
            self.goal_south_pos[env_ids] = 0.0
            self.goal_east_pos[env_ids] = 0.0
            self.goal_west_pos[env_ids] = 0.0
            self.goal_cent_pos[env_ids] = 0.0

        # ShadowHand / wrist goals from FrameTransformer (source ``Robot/world``): **env-local** xyz, same frame as EE buffers.
        self.goal_wrist_pos[env_ids] = self.wrist_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.thumb_goal_pos[env_ids] = self.thumb_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.pinky_goal_pos[env_ids] = self.pinky_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.thumb_goal_rot[env_ids] = self.thumb_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        self.pinky_goal_rot[env_ids] = self.pinky_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        self.fore_goal_pos[env_ids] = self.fore_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.fore_goal_rot[env_ids] = self.fore_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        self.middle_goal_pos[env_ids] = self.middle_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.middle_goal_rot[env_ids] = self.middle_goal_frame.data.target_quat_source[..., 0, :][env_ids]
        self.ring_goal_pos[env_ids] = self.ring_goal_frame.data.target_pos_source[..., 0, :][env_ids]
        self.ring_goal_rot[env_ids] = self.ring_goal_frame.data.target_quat_source[..., 0, :][env_ids]

        # Dynamic outward targets track live fingertips; opening-frame terms below use the updated ``thumb_target`` / ``pinky_target``.
        self._update_goal_aperture_targets(env_ids)

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
        if self.goal_east_markers is not None:
            self.goal_east_markers.visualize(
                self.goal_east_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
            )
            self.goal_west_markers.visualize(
                self.goal_west_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
            )
            self.goal_north_markers.visualize(
                self.goal_north_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
            )
            self.goal_south_markers.visualize(
                self.goal_south_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
            )
            self.goal_cent_markers.visualize(
                self.goal_cent_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
            )
            # print(f"goal_north_pos: {self.goal_north_pos[0]}, goal_south_pos: {self.goal_south_pos[0]}, goal_east_pos: {self.goal_east_pos[0]}, goal_west_pos: {self.goal_west_pos[0]}, goal_cent_pos: {self.goal_cent_pos[0]}")

        if self._use_glove or self.cfg.object_type == "rigid":
            self.garment_right_ee_distance[env_ids] = self.right_ee_pos[env_ids] - self.goal_west_pos[env_ids]
            self.garment_right_ee_euclidean_distance[env_ids] = torch.norm(
                self.garment_right_ee_distance[env_ids], dim=1
            )
            self.garment_left_ee_distance[env_ids] = self.left_ee_pos[env_ids] - self.goal_east_pos[env_ids]
            self.garment_left_ee_euclidean_distance[env_ids] = torch.norm(
                self.garment_left_ee_distance[env_ids], dim=1
            )
        else:
            self.garment_right_ee_distance[env_ids] = 0.0
            self.garment_right_ee_euclidean_distance[env_ids] = 1e3
            self.garment_left_ee_distance[env_ids] = 0.0
            self.garment_left_ee_euclidean_distance[env_ids] = 1e3

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
        # print(f"right_ee_pos: {self.right_ee_pos[0]} thumb_target: {self.thumb_target[0]}")
        self.right_ee_thumb_euclidean_distance[env_ids] = torch.norm(self.right_ee_thumb_distance[env_ids], dim=1)
        # print(f"right_ee_thumb_euclidean_distance: {self.right_ee_thumb_euclidean_distance[0]}")
        self.right_ee_thumb_rotation[env_ids] = quat_mul(self.right_ee_rot[env_ids], quat_conjugate(self.thumb_goal_rot[env_ids]))
        self.right_ee_thumb_angular_distance[env_ids] = rotation_distance(self.right_ee_rot[env_ids], self.thumb_goal_rot[env_ids])
        # print(f"right_ee_thumb_angular_distance: {self.right_ee_thumb_angular_distance[0]}")
        # self.left_ee_goal_distance[env_ids] = self.left_l_ee_pos[env_ids] - self.pinky_goal_pos[env_ids]
        self.left_ee_pinky_distance[env_ids] = self.left_ee_pos[env_ids] - self.pinky_target[env_ids]
        self.left_ee_pinky_euclidean_distance[env_ids] = torch.norm(self.left_ee_pinky_distance[env_ids], dim=1)
        self.left_ee_pinky_rotation[env_ids] = quat_mul(self.left_ee_rot[env_ids], quat_conjugate(self.pinky_goal_rot[env_ids]))
        self.left_ee_pinky_angular_distance[env_ids] = rotation_distance(self.left_ee_rot[env_ids], self.pinky_goal_rot[env_ids])
        # print(f"left_ee_pinky_angular_distance: {self.left_ee_pinky_angular_distance[0]}")
        # print(f"left_ee_pinky_euclidean_distance: {self.left_ee_pinky_euclidean_distance[0]} right_ee_thumb_euclidean_distance: {self.right_ee_thumb_euclidean_distance[0]}")
        # shadow hand aperature
        self.goal_stretch_euclidean_distance[env_ids] = torch.abs(self.ee_euclidean_distance[env_ids] - self.human_stretch_euclidean_distance[env_ids])
        # print(f"ee_euclidean_distance: {self.ee_euclidean_distance[0]}")
        # print(f"goal_stretch_euclidean_distance: {self.goal_stretch_euclidean_distance[0]}, human_stretch_euclidean_distance: {self.human_stretch_euclidean_distance[0]}, ee_euclidean_distance: {self.ee_euclidean_distance[0]}")
        # print(f"garment_right_ee_euclidean_distance: {self.garment_right_ee_euclidean_distance[0]}, garment_left_ee_euclidean_distance: {self.garment_left_ee_euclidean_distance[0]}, right_ee_thumb_euclidean_distance: {self.right_ee_thumb_euclidean_distance[0]}, left_ee_pinky_euclidean_distance: {self.left_ee_pinky_euclidean_distance[0]}")
        # print(f"Goal stretch Euclidean distance: {self.goal_stretch_euclidean_distance[env_ids]}")

        # ------------------------------------------------------------------
        # Opening / bracelet frame: ``goal_cent_pos``, ``goal_wrist_pos``, rim goals are **env-local**.
        # ``wrist_in_open`` etc. are **opening-local** offsets from ``goal_cent_pos`` (inverse world quat of
        # the rigid bracelet root). Opening-local x/y span the opening cross-section; local z is the depth
        # axis candidate for insertion. Depth reward uses these (not raw world X) so policies cannot hack
        # world-X alignment without sitting inside the rim ellipse.
        # ``thumb_target`` / ``pinky_target`` were refreshed earlier via ``_update_goal_aperture_targets``.
        # ------------------------------------------------------------------
        if self.cfg.object_type == "rigid" and hasattr(self, "object") and self.object is not None:
            open_quat_w = self.object.data.root_quat_w[env_ids]
        else:
            open_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0).expand(
                len(env_ids), 4
            )

        p_open = self.goal_cent_pos[env_ids]
        p_rel_wrist = self.goal_wrist_pos[env_ids] - p_open
        self.wrist_in_open[env_ids] = quat_apply_inverse(open_quat_w, p_rel_wrist)

        p_rel_north = self.goal_north_pos[env_ids] - p_open
        p_rel_south = self.goal_south_pos[env_ids] - p_open
        self.north_in_open[env_ids] = quat_apply_inverse(open_quat_w, p_rel_north)
        self.south_in_open[env_ids] = quat_apply_inverse(open_quat_w, p_rel_south)

        p_rel_east = self.goal_east_pos[env_ids] - p_open
        p_rel_west = self.goal_west_pos[env_ids] - p_open
        self.east_in_open[env_ids] = quat_apply_inverse(open_quat_w, p_rel_east)
        self.west_in_open[env_ids] = quat_apply_inverse(open_quat_w, p_rel_west)

        p_rel_thumb = self.thumb_target[env_ids] - p_open
        p_rel_pinky = self.pinky_target[env_ids] - p_open
        self.thumb_in_open[env_ids] = quat_apply_inverse(open_quat_w, p_rel_thumb)
        self.pinky_in_open[env_ids] = quat_apply_inverse(open_quat_w, p_rel_pinky)

        # Depth / in-opening diagnostics (rigid bracelet): after rim + wrist + digit targets in opening frame.
        if self.cfg.object_type == "rigid" and hasattr(self, "object") and self.object is not None:
            x = self.wrist_in_open[env_ids, 0]
            y = self.wrist_in_open[env_ids, 1]
            z = self.wrist_in_open[env_ids, 2]
            rad_eps = torch.as_tensor(1e-4, device=self.device, dtype=x.dtype)
            radius_x = 0.5 * torch.abs(self.east_in_open[env_ids, 0] - self.west_in_open[env_ids, 0]).clamp_min(rad_eps)
            radius_y = 0.5 * torch.abs(self.north_in_open[env_ids, 1] - self.south_in_open[env_ids, 1]).clamp_min(rad_eps)
            radial_normalized = (x / radius_x).pow(2) + (y / radius_y).pow(2)
            outside_error = torch.clamp(radial_normalized - 1.0, min=0.0)
            std = torch.as_tensor(self.cfg.bracelet_inside_opening_std, device=self.device, dtype=x.dtype).clamp_min(
                torch.as_tensor(1e-6, device=self.device, dtype=x.dtype)
            )
            inside_opening_soft = torch.exp(-outside_error / std)
            self.wrist_xy_center_distance[env_ids] = torch.norm(
                    self.wrist_in_open[env_ids, 0:2],
                    dim=-1,
                )
            desired = torch.as_tensor(self.cfg.bracelet_desired_insert_depth, device=self.device, dtype=x.dtype)
            self.depth_distance[env_ids] = torch.abs(z - desired)
            self.wrist_radial_normalized[env_ids] = radial_normalized
            self.inside_opening_soft[env_ids] = inside_opening_soft
            self.insert_depth[env_ids] = z
            self.wrist_center_distance[env_ids] = torch.norm(torch.stack([x,y,z - desired],dim=-1,),dim=-1,)

            # Live digit tips (``*_goal_pos``) in opening frame: same ellipse radii as wrist; depth along local z.
            thumb_tip_o = quat_apply_inverse(open_quat_w, self.thumb_goal_pos[env_ids] - p_open)
            fore_tip_o = quat_apply_inverse(open_quat_w, self.fore_goal_pos[env_ids] - p_open)
            middle_tip_o = quat_apply_inverse(open_quat_w, self.middle_goal_pos[env_ids] - p_open)
            ring_tip_o = quat_apply_inverse(open_quat_w, self.ring_goal_pos[env_ids] - p_open)
            pinky_tip_o = quat_apply_inverse(open_quat_w, self.pinky_goal_pos[env_ids] - p_open)
            self.fore_in_open[env_ids] = fore_tip_o
            self.middle_in_open[env_ids] = middle_tip_o
            self.ring_in_open[env_ids] = ring_tip_o

            def _radial2(pt: torch.Tensor) -> torch.Tensor:
                return (pt[:, 0] / radius_x).pow(2) + (pt[:, 1] / radius_y).pow(2)

            tr = _radial2(thumb_tip_o)
            fr = _radial2(fore_tip_o)
            mr = _radial2(middle_tip_o)
            rr = _radial2(ring_tip_o)
            pr = _radial2(pinky_tip_o)
            def _inside_soft(radial: torch.Tensor) -> torch.Tensor:
                outside = torch.clamp(radial - 1.0, min=0.0)
                return torch.exp(-outside / std)

            thumb_inside = _inside_soft(tr)
            fore_inside = _inside_soft(fr)
            middle_inside = _inside_soft(mr)
            ring_inside = _inside_soft(rr)
            pinky_inside = _inside_soft(pr)

            self.fingers_inside_opening_soft[env_ids] = (
                0.2 * thumb_inside
                + 0.25 * fore_inside
                + 0.30 * middle_inside
                + 0.25 * ring_inside
                + 0.10 * pinky_inside
            )

            self.depth_thumb_distance[env_ids] = torch.abs(thumb_tip_o[:, 2] - desired)
            self.depth_pinky_distance[env_ids] = torch.abs(pinky_tip_o[:, 2] - desired)
        elif self._use_glove:
            self.depth_distance[env_ids] = torch.abs(self.goal_cent_pos[env_ids, 0] - self.goal_wrist_pos[env_ids, 0])
            self.depth_thumb_distance[env_ids] = torch.abs(
                self.goal_west_pos[env_ids, 0] - self.thumb_target[env_ids, 0]
            )
            self.depth_pinky_distance[env_ids] = torch.abs(
                self.goal_east_pos[env_ids, 0] - self.pinky_target[env_ids, 0]
            )
            self.inside_opening_soft[env_ids] = 1.0
            self.wrist_radial_normalized[env_ids] = 0.0
            self.insert_depth[env_ids] = 0.0
            self.fingers_inside_opening_soft[env_ids] = 1.0
            self.thumb_radial_normalized[env_ids] = 0.0
            self.fore_radial_normalized[env_ids] = 0.0
            self.middle_radial_normalized[env_ids] = 0.0
            self.ring_radial_normalized[env_ids] = 0.0
            self.pinky_radial_normalized[env_ids] = 0.0
        else:
            self.depth_distance[env_ids] = 0.0
            self.depth_thumb_distance[env_ids] = 0.0
            self.depth_pinky_distance[env_ids] = 0.0
            self.inside_opening_soft[env_ids] = 0.0
            self.wrist_radial_normalized[env_ids] = 0.0
            self.insert_depth[env_ids] = 0.0
            self.fingers_inside_opening_soft[env_ids] = 0.0
            self.thumb_radial_normalized[env_ids] = 0.0
            self.fore_radial_normalized[env_ids] = 0.0
            self.middle_radial_normalized[env_ids] = 0.0
            self.ring_radial_normalized[env_ids] = 0.0
            self.pinky_radial_normalized[env_ids] = 0.0

def compute_rewards(
    reaching_object_goal_scale: float,
    reaching_ee_object_scale: float,
    stretch_object_scale: float,
    episode_timestep_counter: torch.Tensor,
    object_goal_tracking_scale: float,
    joint_vel_penalty_scale: float,
    ee_euclidean_distance: torch.Tensor,
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
    inside_opening_soft: torch.Tensor,
    fingers_inside_opening_soft: torch.Tensor,
    wrist_radial_normalized: torch.Tensor,
    wrist_height: torch.Tensor,
    top_height: torch.Tensor,
    bottom_height: torch.Tensor,
    thumb_height: torch.Tensor,
    pinky_height: torch.Tensor,
    minimal_width: float,
    wrist_xy_center_distance: torch.Tensor,
    wrist_center_distance: torch.Tensor,
):
    rotation_object_goal_scale = 0.4 # 10.0
    reaching_object_goal_scale = 1.0    
    stretch_object_scale = 0.0
    touching_object_goal_scale = 0.0
    depth_reward_scale = 5.0
    depth_thumb_reward_scale = 0.5
    depth_pinky_reward_scale = 0.5
    # joint_vel_penalty_scale = -0.01
    joint_vel_penalty_scale = 0.0
    max_angle_penalty = 0.8 # 45 degree is the maximum angle for the thumb and pinky
    angle_penalty_scale = -0.1
    wrist_center_alignment_scale = 0.5
    wrist_center_3d_alignment_scale = 10.0

    # FOR REACHING (include condition))
    r_stretch = distance_reward(goal_stretch_euclidean_distance, std=0.05) * stretch_object_scale # 0.03
    # r_right_ee_thumb_distance = distance_cond_reward(garment_right_ee_euclidean_distance, right_ee_thumb_euclidean_distance, minimal_width, std=0.4) * reaching_object_goal_scale # default 0.4
    # r_left_ee_pinky_distance = distance_cond_reward(garment_left_ee_euclidean_distance, left_ee_pinky_euclidean_distance, minimal_width, std=0.2) * reaching_object_goal_scale * 0.0 # default 0.3
    # r_right_ee_thumb_distance = distance_reward(right_ee_thumb_euclidean_distance, std=0.4) * 1.5 * reaching_object_goal_scale * (top_height > wrist_height) * (wrist_height > bottom_height) *(ee_euclidean_distance < 0.3) # default 0.4
    # r_left_ee_pinky_distance = distance_reward(left_ee_pinky_euclidean_distance, std=0.3) * reaching_object_goal_scale * (top_height > wrist_height) * (wrist_height > bottom_height) *(ee_euclidean_distance < 0.3) # default 0.3
    # print(f"top_height: {top_height[0]}, wrist_height: {wrist_height[0]}, bottom_height: {bottom_height[0]}")
    r_right_ee_thumb_distance = distance_reward(right_ee_thumb_euclidean_distance, std=0.4) * 1.5 * reaching_object_goal_scale *(ee_euclidean_distance < 0.3) # default 0.4
    r_left_ee_pinky_distance = distance_reward(left_ee_pinky_euclidean_distance, std=0.3) * reaching_object_goal_scale *(ee_euclidean_distance < 0.3) # default 0.3
    r_right_ee_touch_distance = distance_reward(garment_right_ee_euclidean_distance, std=0.01) * touching_object_goal_scale 
    r_left_ee_touch_distance = distance_reward(garment_left_ee_euclidean_distance, std=0.01) * touching_object_goal_scale 
    # print(garment_right_ee_euclidean_distance[0], garment_left_ee_euclidean_distance[0])
    # FOR REACHING+INSERTING
    # r_garment_thumb_distance = distance_reward(goal_distance_thumb_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_pinky_distance = distance_reward(garment_pinky_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_fore_distance = distance_reward(garment_fore_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_middle_distance = distance_reward(garment_middle_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # r_garment_ring_distance = distance_reward(garment_ring_euclidean_distance, std=0.09) * reaching_object_goal_scale
    # print(f"top_height: {top_height[0]}, wrist_height: {wrist_height[0]}, bottom_height: {bottom_height[0]}")
    # Wrist ellipse (``inside_opening_soft``) and all five digit tips (``fingers_inside_opening_soft``) in opening frame.
    r_depth_distance = (
        distance_reward(depth_distance, std=0.1)
        * inside_opening_soft
        * fingers_inside_opening_soft
        * depth_reward_scale
        * (ee_euclidean_distance < 0.3)
    )
    r_depth_thumb_distance = (
        distance_reward(depth_thumb_distance, std=0.03)
        * fingers_inside_opening_soft
        * (top_height > thumb_height)
        * (thumb_height > bottom_height)
        * depth_thumb_reward_scale
        * (ee_euclidean_distance < 0.3)
    )
    r_depth_pinky_distance = (
        distance_reward(depth_pinky_distance, std=0.06)
        * fingers_inside_opening_soft
        * (top_height > pinky_height)
        * (pinky_height > bottom_height)
        * depth_pinky_reward_scale
        * (ee_euclidean_distance < 0.3)
    )
    r_wrist_center_alignment = (
        distance_reward(wrist_xy_center_distance, std=0.04)
        * wrist_center_alignment_scale
        * fingers_inside_opening_soft
        * (ee_euclidean_distance < 0.3)
    )
    r_wrist_center_3d_alignment = (
        distance_reward(wrist_center_distance, std=0.16)
        * wrist_center_3d_alignment_scale
        * fingers_inside_opening_soft
        * (ee_euclidean_distance < 0.3)
    )
    # print(f"wrist_xy_center_distance: {wrist_xy_center_distance[0]}, wrist_center_distance: {wrist_center_distance[0]}")

    # FOR REACHING+INSERTING+TERMINATE
    # r_wrist_goal = wrist_distance_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, std=0.2) * reaching_object_goal_scale * 2.5  

    right_reach_phase_weight = smooth_gate(right_ee_thumb_euclidean_distance, threshold=0.08, sharpness=30.0)
    left_reach_phase_weight = smooth_gate(left_ee_pinky_euclidean_distance, threshold=0.08, sharpness=30.0)
    r_angular_right_ee_thumb = angular_distance_reward(right_ee_thumb_angular_distance, std=0.2) * rotation_object_goal_scale * right_reach_phase_weight
    r_angular_left_ee_pinky = angular_distance_reward(left_ee_pinky_angular_distance, std=0.2) * rotation_object_goal_scale * left_reach_phase_weight
    # right_insert_phase_weight = fingers_inside_opening_soft * inside_opening_soft * (ee_euclidean_distance < 0.3)
    # left_insert_phase_weight = fingers_inside_opening_soft * inside_opening_soft * (ee_euclidean_distance < 0.3)
    right_insert_phase_weight = 1.0 - right_reach_phase_weight
    left_insert_phase_weight = 1.0 - left_reach_phase_weight
    r_right_angle_penalty = torch.relu(right_ee_thumb_angular_distance-max_angle_penalty) * angle_penalty_scale * right_insert_phase_weight
    r_left_angle_penalty = torch.relu(left_ee_pinky_angular_distance-max_angle_penalty) * angle_penalty_scale * left_insert_phase_weight
    r_joint_vel = joint_vel_penalty(robot_joint_vel) * joint_vel_penalty_scale

    # minillion bonus reward
    # r_object_goal = object_goal_reward(right_ee_thumb_euclidean_distance, r_right_insert, std=0.3) * object_goal_tracking_scale
    # r_successed = success_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, minimal_distance)
    rewards = r_right_ee_thumb_distance * right_reach_phase_weight + r_left_ee_pinky_distance * left_reach_phase_weight + r_depth_distance + r_depth_thumb_distance + r_depth_pinky_distance + r_angular_right_ee_thumb + r_angular_left_ee_pinky + r_right_ee_touch_distance + r_left_ee_touch_distance + r_joint_vel + r_wrist_center_alignment + r_wrist_center_3d_alignment + r_right_angle_penalty + r_left_angle_penalty

    return (rewards, r_right_ee_thumb_distance * right_reach_phase_weight, r_left_ee_pinky_distance * left_reach_phase_weight, r_depth_distance, r_depth_thumb_distance, r_depth_pinky_distance, r_angular_right_ee_thumb, r_angular_left_ee_pinky, r_right_ee_touch_distance, r_left_ee_touch_distance, r_joint_vel, r_wrist_center_alignment, r_wrist_center_3d_alignment, r_right_angle_penalty, r_left_angle_penalty)

