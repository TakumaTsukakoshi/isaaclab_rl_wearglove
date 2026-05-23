# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialCfg,   
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    sample_uniform,
)
from collections.abc import Sequence

from tasks.airec.airec2_finger_deformable import (
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
class ReachDeformableBraceletEnvCfg(AIRECEnvCfg):
    """Deformable bracelet + AIREC + Shadow Hand (same layout as :class:`~tasks.airec.wear_finger.WearEnvCfg`).

    - **Object** is a **deformable** USD (:attr:`object_usd` / :attr:`object_cfg`).
    - **AIREC** fingertips reach toward **Shadow Hand** goals (thumb / pinky / wrist frame transformers), same
      reward structure as the wear / reach-finger tasks.
    - We set ``use_glove=True`` so rim-driven garment / depth terms match the rigid reach-bracelet task.
    - With :attr:`deformable_bracelet_geom_rim_goals`, N/S/E/W follow **deformed** geometry instead of four fixed nodes.
    """
    # Reuse glove-style rim buffers (N/S/E/W goals + garment distances) for the deformable bracelet.
    use_glove: bool = True
    #: If True, use arXiv:2502.07005 App. B.7 (cloth-hanging) style reward via :mod:`tasks.airec.mdp.rewards`.
    use_geometryrl_b7_reward: bool = False

    #: If True, ``goal_{north,south,east,west}`` are recomputed each step from **deformed** nodal geometry: a static
    #: outer-rim vertex set (from the rest pose) is projected into each env's PCA plane; E/W are max/min along the
    #: dominant in-plane axis and N/S along its in-plane perpendicular (sign stabilized against the rest pose).
    #: If False, use the glove-style **fixed** nodal indices from :meth:`AIRECEnv._choose_mouth_nodes_4dirs` (same as before).
    deformable_bracelet_geom_rim_goals: bool = True
    #: Vertices whose distance to the bracelet centroid **axis** (first PC of the rest mesh) exceeds this quantile
    #: of ``rho`` are kept as the rim subset for geometry goals.
    deformable_bracelet_rim_band_quantile: float = 0.55
    #: If the rim mask has fewer vertices than this, fall back to **all** mesh vertices for stability.
    deformable_bracelet_rim_min_vertices: int = 32
    #: If True (with :attr:`deformable_bracelet_geom_rim_goals`), N/S/E/W use **fixed nodal indices** chosen once
    #: from the rest mesh. Each step only the **positions** of those vertices update with deformation.
    #: If False, recompute extrema and opening PCA every step (previous behaviour).
    deformable_bracelet_geom_freeze_nsew_at_init: bool = True
    #: If True (with freeze), N/S/E/W labels use **env-local world axes** (+Z north / −Z south, min Y east / max Y west).
    #: If False, labels use the legacy **rim PCA** extrema (same as old ``_freeze_geom_rim_nsew_labels_from_rest``).
    #: Insert / ellipse gates use **base env-local** offsets from ``goal_cent`` (no rim PCA rotation).
    deformable_bracelet_freeze_nsew_use_world_axes: bool = True
    #: For world-axis rim labels: take the top this many extreme rim vertices (per axis), centroid them, then pick the
    #: rim vertex **closest to that centroid** (stable vs single argmax noise). Capped by rim vertex count.
    deformable_bracelet_rim_world_axis_extreme_k: int = 8

    #: Target wrist **env-local X** offset from ``goal_cent_pos`` for depth reward ``abs(Δx - desired)``.
    bracelet_desired_insert_depth: float = 0.0
    #: Soft gate on wrist radial error in **base Y–Z** (ellipse radii from N/S and E/W rim goals in env-local).
    bracelet_inside_opening_std: float = 0.15

    # Sparse task-success: wrist goal within ``bracelet_success_threshold`` (env-local, m).
    terminate_on_task_success: bool = False
    bracelet_success_threshold: float = 0.01  # 1 cm
    task_success_bonus: float = 10000.0

    #: When cumulative episode success rate exceeds :attr:`adaptive_physics_success_threshold`, switch from
    #: coarse (:attr:`~tasks.airec.airec2_finger_deformable.AIRECEnvCfg.physics_dt` / decimation) to fine PhysX.
    #: RL control step stays ``1/10`` s in both cases.
    adaptive_physics_on_success: bool = True
    adaptive_physics_success_threshold: float = 0.5
    adaptive_physics_min_episodes: int = 20
    fine_physics_dt: float = 1 / 2000
    fine_decimation: int = 200

    object_type = "deformable"
    #: Hide parent ``AIRECEnv`` red kinematic anchor cuboids on the rim (used for ``north_edge_pos`` when
    #: :attr:`deformable_bracelet_geom_rim_goals` is False; geometric mode uses ``goal_north/south`` for top/under wrist).
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
    default_object_pos = [0.24, 0.00, 0.85] # airec2 bracelet
    default_object_pos_glove = [0.14, 0.00, 0.84] # airec2 glove

    object_goal_tracking_scale = 16.0
    object_goal_tracking_finegrained_scale = 5.0

    #: Rim sample points on the rigid bracelet in the object's **root frame** (m). Each reset step:
    #: ``p_env = root_pos_w + quat_apply(root_quat_w, offset) - env_origin`` (same env-local convention as the glove's nodal rim).
    #: **Center** follows the glove task: ``goal_cent_pos = (goal_north_pos + goal_south_pos) / 2`` (midpoint of N–S, not the geometric centroid of four points).
    #: Tune offsets to match ``bracelet.usd`` (opening normal / lateral axes).
    bracelet_rim_offset_north: tuple[float, float, float] = (0.0, 0.03, 0.0)
    bracelet_rim_offset_south: tuple[float, float, float] = (0.0, -0.03, 0.0)
    bracelet_rim_offset_east: tuple[float, float, float] = (-0.10, 0.0, 0.0)
    bracelet_rim_offset_west: tuple[float, float, float] = (0.10, 0.0, 0.0)

    object_usd = os.path.join(
        _REPO_ROOT, "assets", "Bracelet", "deformable_bracelet_new.usd"
    )
    object_usd_glove = os.path.join(
        _REPO_ROOT, "assets", "Glove", "GL_Gloves068", "GL_Gloves068_obj_revise.usd"
    )


    object_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=default_object_pos_glove, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=object_usd_glove,
            copy_from_source=True,
            visible=True,
            scale=(1.0, 1.3, 1.2),
            # scale=(1.0, 1.5, 1.4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset/rest_offset are mainly controlled by deformable_props below for deformables.
            ),
            deformable_props=DeformableBodyPropertiesCfg(
                deformable_enabled=True,
                kinematic_enabled=False,
                self_collision=False,
                # Keep resolutions modest; increase if you need finer deformation.
                # simulation_hexahedral_resolution=16,
                # collision_simplification=True,
                # collision_simplification_remeshing=True,
                # collision_simplification_remeshing_resolution=8,
                # collision_simplification_target_triangle_count=0,
                # collision_simplification_force_conforming=True,
                # solver_position_iteration_count=16,
                simulation_hexahedral_resolution=16,
                collision_simplification=True,
                collision_simplification_remeshing=True,
                collision_simplification_remeshing_resolution=4,
                collision_simplification_target_triangle_count=0,
                collision_simplification_force_conforming=True,
                solver_position_iteration_count=4,
                contact_offset=0.006,
                rest_offset=0.003,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.5, 0.3),
            opacity=1.0,             
        ),   
        ),
        debug_vis=False,
    )

    # object_cfg: DeformableObjectCfg = DeformableObjectCfg(
    #     prim_path="/World/envs/env_.*/Object",
    #     init_state=DeformableObjectCfg.InitialStateCfg(pos=default_object_pos, rot=[0.5, 0.5, -0.5, -0.5]),
    #     spawn=UsdFileCfg(
    #         usd_path=object_usd,
    #         copy_from_source=True,
    #         visible=True,
    #         scale=(1.0, 1.0, 1.0),
    #         # scale=(1.0, 1.5, 1.4),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=True,
    #             # contact_offset/rest_offset are mainly controlled by deformable_props below for deformables.
    #         ),
    #         deformable_props=DeformableBodyPropertiesCfg(
    #             deformable_enabled=True,
    #             kinematic_enabled=False,
    #             self_collision=False,
    #             # Keep resolutions modest; increase if you need finer deformation.
    #             # simulation_hexahedral_resolution=16,
    #             # collision_simplification=True,
    #             # collision_simplification_remeshing=True,
    #             # collision_simplification_remeshing_resolution=8,
    #             # collision_simplification_target_triangle_count=0,
    #             # collision_simplification_force_conforming=True,
    #             # solver_position_iteration_count=16,
    #             simulation_hexahedral_resolution=4,
    #             collision_simplification=True,
    #             collision_simplification_remeshing=True,
    #             collision_simplification_remeshing_resolution=4,
    #             collision_simplification_target_triangle_count=0,
    #             collision_simplification_force_conforming=True,
    #             solver_position_iteration_count=4,
    #             contact_offset=0.006,
    #             rest_offset=0.003,
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #         diffuse_color=(0.3, 0.3, 0.0),
    #         opacity=1.0,             
    #     ),   
    #     ),
    #     debug_vis=False,
    # )
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
                    rot = [0.7071, -0.7071, 0.0, 0.0]
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


class ReachDeformableBraceletEnv(AIRECEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: ReachDeformableBraceletEnvCfg

    def __init__(self, cfg: ReachDeformableBraceletEnvCfg, render_mode: str | None = None, **kwargs):
        # ``WearEnv`` forces ``object_type="none"`` when ``use_glove`` is False so the deformable glove
        # disappears. For this rigid-bracelet task, keep ``object_type="rigid"`` unless we explicitly
        # drop the scene object (no rigid/deformable object requested).
        self._use_glove = bool(getattr(cfg, "use_glove", True))
        if not self._use_glove:
            raise ValueError("ReachDeformableBraceletEnv expects use_glove=True (nodal rim features).")
        cfg.sim.dt = cfg.physics_dt
        cfg.sim.render_interval = cfg.decimation

        super().__init__(cfg, render_mode, **kwargs)

        self._physics_timestep_upgraded = False
        self._curriculum_episode_count = 0
        self._curriculum_success_count = 0

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
        self.wrist_origin = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.wrist_lateral_axis = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.unit_dir = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.thumb_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.pinky_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.fore_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        
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

          # Last ShadowHand root pose written in ``_reset_target_pose`` (same convention as ``write_root_state_to_sim``):
        # position is **sim world** (includes ``env_origins``); quaternion **world** wxyz.
        self.goal_hand_root_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_hand_root_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_hand_root_quat[:, 0] = 1.0

        # Bracelet metrics in env-local / robot base (no rim-PCA opening rotation).
        self.wrist_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.inside_opening_soft = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.insert_depth = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.thumb_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.fore_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.middle_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ring_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.pinky_radial_normalized = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
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
        self.task_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.right_left_goal_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.finger_joint_ids, _ = self.hand.find_joints(self.cfg.finger_joint_names)
        self._shadow_hand_finger_hold = torch.zeros(
            (self.num_envs, len(self.finger_joint_ids)),
            device=self.device,
            dtype=self.hand.data.joint_pos.dtype,
        )

        self._bracelet_rim_idx: torch.Tensor | None = None
        self._bracelet_geom_e1_ref: torch.Tensor | None = None
        self._frozen_geom_north_idx: int | None = None
        self._frozen_geom_south_idx: int | None = None
        self._frozen_geom_east_idx: int | None = None
        self._frozen_geom_west_idx: int | None = None
        if getattr(self.cfg, "deformable_bracelet_geom_rim_goals", False) and self.cfg.object_type == "deformable":
            self._bracelet_rim_idx = self._build_bracelet_rim_node_indices()
            self._bracelet_geom_e1_ref = self._reference_bracelet_rim_e1()
            if getattr(self.cfg, "deformable_bracelet_geom_freeze_nsew_at_init", True):
                if getattr(self.cfg, "deformable_bracelet_freeze_nsew_use_world_axes", True):
                    self._freeze_geom_rim_nsew_labels_by_world_axes()
                else:
                    self._freeze_geom_rim_nsew_labels_from_rest()
        self.wrist_xy_center_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        # Env-local vector wrist goal − rim center (``torch.norm(..., dim=1)`` → ``wrist_center_euclidean_distance``).
        self.wrist_center_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # Backward-compatible alias (same scalar).
        self.wrist_center_euclidean_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.thumb_inside_ellipse = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.pinky_inside_ellipse = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.wrist_inside_ellipse = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.fingers_inside_soft_gate = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.per_finger_soft_inside = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)
    

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
                self.right_ee_thumb_distance,
                # euclidean distance (1,)
                self.right_ee_thumb_euclidean_distance.unsqueeze(1),
                ## xyz diffs (3,)
                self.left_ee_pinky_distance,
                # euclidean distances (1,)
                self.left_ee_pinky_euclidean_distance.unsqueeze(1),
                # xyz diffs (3,)
                self.wrist_center_distance,
                # euclidean distance (1,)
                self.wrist_center_euclidean_distance.unsqueeze(1),
                # per finger soft inside (5,)
                self.per_finger_soft_inside,

            ),
            dim=-1,
        )
        return gt

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        super()._pre_physics_step(actions)

    def _maybe_upgrade_physics_timestep(self) -> None:
        if not self.cfg.adaptive_physics_on_success or self._physics_timestep_upgraded:
            return
        rate = self._curriculum_success_count / max(self._curriculum_episode_count, 1)
        if (
            self._curriculum_episode_count >= self.cfg.adaptive_physics_min_episodes
            and rate > self.cfg.adaptive_physics_success_threshold
        ):
            self._apply_sim_timestep(self.cfg.fine_physics_dt, self.cfg.fine_decimation)
            self._physics_timestep_upgraded = True
            print(
                "[ReachDeformableBraceletEnv] Upgraded physics: "
                f"physics_dt={self.cfg.fine_physics_dt:.6g}, decimation={self.cfg.fine_decimation} "
                f"(success_rate={rate:.3f} over {self._curriculum_episode_count} episodes)"
            )

    def _update_physics_curriculum_on_reset(self, env_ids) -> None:
        if not self.cfg.adaptive_physics_on_success or self._physics_timestep_upgraded:
            return
        n = int(env_ids.numel()) if hasattr(env_ids, "numel") else len(env_ids)
        if n == 0:
            return
        successes = int(self.task_success[env_ids].sum().item())
        self._curriculum_episode_count += n
        self._curriculum_success_count += successes
        self._maybe_upgrade_physics_timestep()

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            reset_ids = self.robot._ALL_INDICES
        else:
            reset_ids = self._normalize_env_ids(env_ids)
        if self.cfg.adaptive_physics_on_success and not self._physics_timestep_upgraded:
            self._update_physics_curriculum_on_reset(reset_ids)

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
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Parent calls ``_compute_intermediate_values()`` and writes ``self._term_log`` for wandb.
        termination, time_out = super()._get_dones()

        # Bracelet root vs. ShadowHand wrist goal, both env-local (see ``_compute_intermediate_values``).
        self.task_success[:] = (
            self.wrist_center_euclidean_distance < self.cfg.bracelet_success_threshold
        )

        # Surface to wandb via the same ``_term_log`` dict that ``AIRECEnv._get_dones`` populates.
        if not hasattr(self, "_term_log") or self._term_log is None:
            self._term_log = {}
        self._term_log["task_success"] = self.task_success.float()
        self._term_log["wrist_center_distance"] = self.wrist_center_euclidean_distance

        if self.cfg.terminate_on_task_success:
            termination = termination | self.task_success
            self._term_log["term_any"] = termination.float()

        return termination, time_out

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
                r_wrist_center_distance,
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
                depth,
                depth_t,
                depth_p,
                # Heights are in base frame (env-local); we keep the API as "height scalars".
                self.goal_wrist_pos[:, 2],
                self.goal_north_pos[:, 2],
                self.goal_south_pos[:, 2],
                self.thumb_target[:, 2],
                self.pinky_target[:, 2],
                self.wrist_center_euclidean_distance,
                self.thumb_inside_ellipse,
                self.pinky_inside_ellipse,
                self.wrist_inside_ellipse,
                self.fingers_inside_soft_gate,
            )
  
            self.extras["log"] = {
                "reach_reward_right": r_right_ee_thumb_distance,
                "reach_reward_left": r_left_ee_pinky_distance,
                "wrist_center_distance_reward": r_wrist_center_distance,
                "angular_reward_right": r_angular_right_ee_thumb,
                "angular_reward_left": r_angular_left_ee_pinky,
                "fingers_inside_soft_gate": self.fingers_inside_soft_gate,
            }

        if "tactile" in self.cfg.obs_list:
            self.extras["log"].update(
                {
                    "normalised_forces_left_x": self.normalised_forces[:, 0],
                    "normalised_forces_right_x": self.normalised_forces[:, 1],
                }
            )
        
        # Sparse task-success bonus: bracelet within ``bracelet_success_threshold`` (m) of wrist goal.
        # ``_get_dones`` runs before ``_get_rewards`` each control step, so ``self.task_success`` is current.
        success_bonus = self.task_success.float() * float(self.cfg.task_success_bonus)
        rewards = rewards + success_bonus
        self.extras["log"]["task_success_bonus"] = success_bonus
        self.extras["log"]["task_success"] = self.task_success.float()
        self.extras["log"]["wrist_center_distance"] = self.wrist_center_euclidean_distance
        if self.cfg.adaptive_physics_on_success:
            rate = self._curriculum_success_count / max(self._curriculum_episode_count, 1)
            self.extras["log"]["curriculum_success_rate"] = torch.full(
                (self.num_envs,), rate, device=self.device, dtype=torch.float32
            )
            self.extras["log"]["physics_timestep_upgraded"] = torch.full(
                (self.num_envs,),
                float(self._physics_timestep_upgraded),
                device=self.device,
                dtype=torch.float32,
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

        # def _reset_target_pose(self, env_ids):
    #     # Make sure this is already on the right device once
    #     default_state = self.hand.data.default_root_state.clone()[env_ids]

    #     pos_noise = sample_uniform(-0.02, 0.02, (len(env_ids), 3), device=self.device)

    #     init_pos = default_state[0, 0:3].unsqueeze(0).repeat(len(env_ids), 1)
      
        # default_state[:, 0:3] = (
        #     init_pos
        #     + pos_noise * self.cfg.reset_goal_position_noise
        #     + self.scene.env_origins[env_ids]
        # )

        # init_rot = default_state[0, 3:7].unsqueeze(0).repeat(len(env_ids), 1)

        # # Randomize pitch (Y-axis rotation) by ±5° in world frame, applied on top of the default root orientation.
        # B = int(len(env_ids))
        # pitch_rad = sample_uniform(
        #     torch.deg2rad(torch.tensor(-5.0, device=self.device, dtype=torch.float32)),
        #     torch.deg2rad(torch.tensor(5.0, device=self.device, dtype=torch.float32)),
        #     (B,),
        #     device=self.device,
        # )
        # yaw_rad = sample_uniform(
        #     torch.deg2rad(torch.tensor(-5.0, device=self.device, dtype=torch.float32)),
        #     torch.deg2rad(torch.tensor(5.0, device=self.device, dtype=torch.float32)),
        #     (B,),
        #     device=self.device,
        # )
        # zero = torch.zeros_like(pitch_rad)
        # q_yaw = quat_from_euler_xyz(zero, zero, yaw_rad)  # (B, 4) wxyz
        # q_pitch = quat_from_euler_xyz(zero, pitch_rad, zero)  # (B, 4) wxyz
        # q_yaw_pitch = quat_mul(q_yaw, q_pitch)
        # default_state[:, 3:7] = quat_mul(q_yaw_pitch, init_rot)

        # default_state[:, 7:] = 0.0

        # joint_pos = self.hand.data.default_joint_pos[env_ids]
        # joint_vel = torch.zeros_like(joint_pos)

        # self.hand.set_joint_position_target(joint_pos, env_ids=env_ids)
        # self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # self.hand.write_root_state_to_sim(default_state, env_ids=env_ids)
        # # Cache root pose actually written (world pos + world quat) for aperture / frame logic.
        # self.goal_hand_root_pos[env_ids] = default_state[:, 0:3].to(dtype=self.goal_hand_root_pos.dtype)
        # self.goal_hand_root_quat[env_ids] = default_state[:, 3:7].to(dtype=self.goal_hand_root_quat.dtype)
        # self._shadow_hand_finger_hold[env_ids] = joint_pos[:, self.finger_joint_ids].clone()
    
    def _reset_target_pose(self, env_ids):
        # Default root state for selected envs
        default_state = self.hand.data.default_root_state.clone()[env_ids]

        # No randomization for position
        default_state[:, 0:3] = (
            self.hand.data.default_root_state[env_ids, 0:3]
            + self.scene.env_origins[env_ids]
        )

        # No randomization for rotation
        default_state[:, 3:7] = self.hand.data.default_root_state[env_ids, 3:7]

        # Reset root velocity
        default_state[:, 7:] = 0.0

        # Reset joints
        joint_pos = self.hand.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)

        self.hand.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.hand.write_root_state_to_sim(default_state, env_ids=env_ids)

        # Cache root pose actually written
        self.goal_hand_root_pos[env_ids] = default_state[:, 0:3].to(
            dtype=self.goal_hand_root_pos.dtype
        )
        self.goal_hand_root_quat[env_ids] = default_state[:, 3:7].to(
            dtype=self.goal_hand_root_quat.dtype
        )

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

    def _build_bracelet_rim_node_indices(self) -> torch.Tensor:
        """Static rim vertex indices from the **rest** mesh (world-space default state, env 0)."""
        P0 = self.object.data.default_nodal_state_w[0, :, :3].to(device=self.device, dtype=torch.float32)
        v = int(P0.shape[0])
        mean = P0.mean(dim=0)
        x0 = P0 - mean
        _, _, vt = torch.pca_lowrank(x0, q=3, center=False)
        axis = vt[:, 0]
        axis = axis / (torch.norm(axis) + 1e-8)
        x_perp = x0 - (x0 @ axis).unsqueeze(1) * axis.unsqueeze(0)
        rho = torch.norm(x_perp, dim=1)
        q = float(getattr(self.cfg, "deformable_bracelet_rim_band_quantile", 0.55))
        thr = torch.quantile(rho, q)
        idx = torch.nonzero(rho >= thr, as_tuple=False).squeeze(1)
        min_v = int(getattr(self.cfg, "deformable_bracelet_rim_min_vertices", 32))
        if int(idx.numel()) < min_v:
            idx = torch.arange(v, device=self.device, dtype=torch.long)
        return idx.long()

    def _reference_bracelet_rim_e1(self) -> torch.Tensor:
        """Dominant in-plane direction on the rest rim (for consistent ± on ``e1`` each step)."""
        assert self._bracelet_rim_idx is not None
        p = self.object.data.default_nodal_state_w[0, self._bracelet_rim_idx, :3].to(
            device=self.device, dtype=torch.float32
        )
        xm = p - p.mean(dim=0, keepdim=True)
        k = int(xm.shape[0])
        cov = (xm.T @ xm) / max(k, 1)
        _, evecs = torch.linalg.eigh(cov)
        e_big = evecs[:, 2]
        return e_big / (torch.norm(e_big) + 1e-8)

    def _compute_rest_rim_pca_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Rim PCA on env-0 rest positions (env-local): returns ``(q_wxyz, e1, e2, x_centered)`` for opening frame / legacy labels."""
        if self._bracelet_rim_idx is None or self._bracelet_geom_e1_ref is None:
            return None
        idx = self._bracelet_rim_idx
        k = int(idx.numel())
        p_w = self.object.data.default_nodal_state_w[0, idx, :3].to(device=self.device, dtype=torch.float32)
        origin0 = self.scene.env_origins[0].to(dtype=p_w.dtype)
        p_row = p_w - origin0.unsqueeze(0)
        mean = p_row.mean(dim=0, keepdim=True)
        x = p_row - mean
        cov = (x.T @ x) / max(k, 1)
        _, evecs = torch.linalg.eigh(cov)
        n = evecs[:, 0]
        n = n / (torch.norm(n) + 1e-8)
        e_big = evecs[:, 2]
        e_big = e_big / (torch.norm(e_big) + 1e-8)
        ref = self._bracelet_geom_e1_ref.to(device=p_row.device, dtype=p_row.dtype)
        flip = -1.0 if float((e_big * ref).sum().item()) < 0.0 else 1.0
        e1 = e_big * flip
        e2 = torch.linalg.cross(n, e1)
        e2 = e2 / (torch.norm(e2) + 1e-8)
        up_ref = torch.tensor([0.0, 0.0, 1.0], device=p_row.device, dtype=p_row.dtype)
        if torch.dot(e2, up_ref) < 0:
            n = -n
            e2 = -e2
        r_mat = torch.stack([e1, e2, n], dim=-1)
        q = quat_from_matrix(r_mat.unsqueeze(0))[0].clone()
        return q, e1, e2, x

    @staticmethod
    def _pick_representative_rim_vertex_from_extreme_band(
        p: torch.Tensor,
        values: torch.Tensor,
        idx: torch.Tensor,
        largest: bool,
        k_extreme: int,
    ) -> tuple[int, int]:
        """Pick rim-local and global mesh index: top-``k`` by ``values``, centroid in 3D, then closest rim vertex to centroid."""
        n = int(values.numel())
        k = min(int(k_extreme), n)
        if k <= 0:
            return 0, int(idx[0].item())
        if largest:
            extreme_local = torch.topk(values, k=k, largest=True).indices
        else:
            extreme_local = torch.topk(values, k=k, largest=False).indices
        centroid = p[extreme_local].mean(dim=0)
        dist = torch.norm(p - centroid.unsqueeze(0), dim=-1)
        rep_local = int(torch.argmin(dist).item())
        rep_global = int(idx[rep_local].item())
        return rep_local, rep_global

    def _freeze_geom_rim_nsew_labels_by_world_axes(self) -> None:
        """Pick N/S/E/W **global** vertex indices from env-0 rest rim using env-local world axes (no PCA for labels).

        Convention (env-local):
            North = +Z band (top K max Z) → centroid → nearest rim vertex
            South = −Z band (top K min Z) → centroid → nearest rim vertex
            East  = −Y band (top K min Y) → centroid → nearest rim vertex
            West  = +Y band (top K max Y) → centroid → nearest rim vertex
        """
        if self._bracelet_rim_idx is None:
            return
        idx = self._bracelet_rim_idx
        if int(idx.numel()) == 0:
            return

        p_w = self.object.data.default_nodal_state_w[0, idx, :3].to(device=self.device, dtype=torch.float32)
        origin0 = self.scene.env_origins[0].to(device=p_w.device, dtype=p_w.dtype)
        p = p_w - origin0.unsqueeze(0)

        y = p[:, 1]
        z = p[:, 2]
        k_ext = int(getattr(self.cfg, "deformable_bracelet_rim_world_axis_extreme_k", 8))

        north_local, self._frozen_geom_north_idx = self._pick_representative_rim_vertex_from_extreme_band(
            p, z, idx, largest=True, k_extreme=k_ext
        )
        south_local, self._frozen_geom_south_idx = self._pick_representative_rim_vertex_from_extreme_band(
            p, z, idx, largest=False, k_extreme=k_ext
        )
        east_local, self._frozen_geom_east_idx = self._pick_representative_rim_vertex_from_extreme_band(
            p, y, idx, largest=False, k_extreme=k_ext
        )
        west_local, self._frozen_geom_west_idx = self._pick_representative_rim_vertex_from_extreme_band(
            p, y, idx, largest=True, k_extreme=k_ext
        )

        print("[World-axis NSEW rim labels (extreme-band → centroid → nearest vertex)]")
        labels = (
            ("N", self._frozen_geom_north_idx, north_local),
            ("S", self._frozen_geom_south_idx, south_local),
            ("E", self._frozen_geom_east_idx, east_local),
            ("W", self._frozen_geom_west_idx, west_local),
        )
        for name, gidx, li in labels:
            pos_w = p_w[li].detach().cpu().tolist()
            pos_env = p[li].detach().cpu().tolist()
            print(
                f"  {name}: global_idx={gidx}, rim_local={li}, env_local={pos_env}, world={pos_w} (K={min(k_ext, int(idx.numel()))})"
            )

    def _freeze_geom_rim_nsew_labels_from_rest(self) -> None:
        """Legacy: N/S/E/W from rim PCA extrema on the rest mesh (evaluated once at init)."""
        r = self._compute_rest_rim_pca_frame()
        if r is None:
            return
        _, e1, e2, x = r
        idx = self._bracelet_rim_idx
        assert idx is not None
        a = (x * e1.unsqueeze(0)).sum(-1)
        b = (x * e2.unsqueeze(0)).sum(-1)
        ie = int(torch.argmax(a).item())
        iw = int(torch.argmin(a).item())
        inorth = int(torch.argmax(b).item())
        isouth = int(torch.argmin(b).item())
        self._frozen_geom_east_idx = int(idx[ie].item())
        self._frozen_geom_west_idx = int(idx[iw].item())
        self._frozen_geom_north_idx = int(idx[inorth].item())
        self._frozen_geom_south_idx = int(idx[isouth].item())

        print("[PCA-axis NSEW rim labels]")
        p_w = self.object.data.default_nodal_state_w[0, idx, :3].to(device=self.device, dtype=torch.float32)
        for name, li in (("N", inorth), ("S", isouth), ("E", ie), ("W", iw)):
            gidx = int(idx[li].item())
            print(f"  {name}: global_idx={gidx}, rim_local={li}, world={p_w[li].detach().cpu().tolist()}")

    def _bracelet_geom_rim_goals_env_local(self, env_ids: torch.Tensor) -> None:
        """Env-local rim goals from current nodal positions (geometry), matching rigid reward conventions."""
        if self._bracelet_rim_idx is None or self._bracelet_geom_e1_ref is None:
            return
        origins = self.scene.env_origins[env_ids]
        if getattr(self.cfg, "deformable_bracelet_geom_freeze_nsew_at_init", True):
            if (
                self._frozen_geom_north_idx is None
                or self._frozen_geom_south_idx is None
                or self._frozen_geom_east_idx is None
                or self._frozen_geom_west_idx is None
            ):
                return
            in_n = self._frozen_geom_north_idx
            in_s = self._frozen_geom_south_idx
            in_e = self._frozen_geom_east_idx
            in_w = self._frozen_geom_west_idx
            self.goal_north_pos[env_ids] = self.object.data.nodal_pos_w[env_ids, in_n, :] - origins
            self.goal_south_pos[env_ids] = self.object.data.nodal_pos_w[env_ids, in_s, :] - origins
            self.goal_east_pos[env_ids] = self.object.data.nodal_pos_w[env_ids, in_e, :] - origins
            self.goal_west_pos[env_ids] = self.object.data.nodal_pos_w[env_ids, in_w, :] - origins
            self.goal_cent_pos[env_ids] = (self.goal_north_pos[env_ids] + self.goal_south_pos[env_ids]) * 0.5
            return

        idx = self._bracelet_rim_idx
        k = int(idx.numel())
        p = self.object.data.nodal_pos_w[env_ids][:, idx, :] - origins.unsqueeze(1)
        mean = p.mean(dim=1, keepdim=True)
        x = p - mean
        cov = torch.bmm(x.transpose(1, 2), x) / max(k, 1)
        _, evecs = torch.linalg.eigh(cov)
        n = evecs[:, :, 0]
        n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-8)
        e_big = evecs[:, :, 2]
        e_big = e_big / (torch.norm(e_big, dim=-1, keepdim=True) + 1e-8)
        ref = self._bracelet_geom_e1_ref.view(1, 3).to(device=p.device, dtype=p.dtype)
        flip = torch.where((e_big * ref).sum(dim=-1) < 0.0, -1.0, 1.0).view(-1, 1)
        e1 = e_big * flip
        e2 = torch.linalg.cross(n, e1, dim=-1)
        e2 = e2 / (torch.norm(e2, dim=-1, keepdim=True) + 1e-8)
        a = (x * e1.unsqueeze(1)).sum(-1)
        b = (x * e2.unsqueeze(1)).sum(-1)
        ie = torch.argmax(a, dim=1)
        iw = torch.argmin(a, dim=1)
        inorth = torch.argmax(b, dim=1)
        isouth = torch.argmin(b, dim=1)
        b_range = torch.arange(p.shape[0], device=p.device, dtype=torch.long)
        self.goal_east_pos[env_ids] = p[b_range, ie]
        self.goal_west_pos[env_ids] = p[b_range, iw]
        self.goal_north_pos[env_ids] = p[b_range, inorth]
        self.goal_south_pos[env_ids] = p[b_range, isouth]
        self.goal_cent_pos[env_ids] = (self.goal_north_pos[env_ids] + self.goal_south_pos[env_ids]) * 0.5

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

    def _ellipse_inner_ring_gate_zy(
        self,
        env_ids: torch.Tensor,
        pos: torch.Tensor,
        target_scale: float = 0.85,
        sharpness: float = 10.0,
    ):
        eps = torch.as_tensor(1e-4, device=self.device, dtype=pos.dtype)
        radius_y = 0.5 * torch.abs(self.goal_east_pos[env_ids, 1] - self.goal_west_pos[env_ids, 1]).clamp_min(eps)
        radius_z = 0.5 * torch.abs(self.goal_north_pos[env_ids, 2] - self.goal_south_pos[env_ids, 2]).clamp_min(eps)
       
        ellipse_value = ((pos[env_ids, 2] - self.goal_cent_pos[env_ids, 2])/ (radius_z + eps)) ** 2 + ((pos[env_ids, 1] - self.goal_cent_pos[env_ids, 1])/ (radius_y + eps)) ** 2
        target_value = target_scale ** 2

        return torch.exp(-sharpness * (ellipse_value - target_value) ** 2)
    
    def _soft_side_gate_y(
        self,
        env_ids: torch.Tensor,
        pos: torch.Tensor,
        direction: float,
        sharpness: float = 20.0,
    ):
        # direction = +1.0: y+ 側を好む
        # direction = -1.0: y- 側を好む
        return torch.sigmoid(
            sharpness * direction * (pos[env_ids, 1] - self.goal_cent_pos[env_ids, 1])
        )
    
    def _ellipse_soft_gate_zy(
        self,
        env_ids: torch.Tensor,
        pos: torch.Tensor,
        sharpness: float = 10.0,
    ):
        eps = torch.as_tensor(1e-6, device=self.device, dtype=pos.dtype)
        radius_z = 0.5 * torch.abs(self.goal_north_pos[env_ids, 2] - self.goal_south_pos[env_ids, 2]).clamp_min(eps)
        radius_y = 0.5 * torch.abs(self.goal_east_pos[env_ids, 1] - self.goal_west_pos[env_ids, 1]).clamp_min(eps)
        ellipse_value = ((pos[env_ids, 2] - self.goal_cent_pos[env_ids, 2])/ (radius_z + eps)) ** 2 + ((pos[env_ids, 1] - self.goal_cent_pos[env_ids, 1])/ (radius_y + eps)) ** 2


        return torch.sigmoid(sharpness * (1.0 - ellipse_value))
    def _compute_intermediate_values(self, reset=False, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values()
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if self._use_glove:
            if getattr(self.cfg, "deformable_bracelet_geom_rim_goals", False) and self.cfg.object_type == "deformable":
                self._bracelet_geom_rim_goals_env_local(env_ids)
            else:
                # Deformable-object rim points (fixed nodal anchors from reset).
                self.goal_north_pos[env_ids] = (
                    self.object.data.nodal_pos_w[env_ids, self.anchor_idx["north"], :]
                    - self.scene.env_origins[env_ids]
                )
                self.goal_south_pos[env_ids] = (
                    self.object.data.nodal_pos_w[env_ids, self.anchor_idx["south"], :]
                    - self.scene.env_origins[env_ids]
                )
                self.goal_east_pos[env_ids] = (
                    self.object.data.nodal_pos_w[env_ids, self.anchor_idx["east"], :]
                    - self.scene.env_origins[env_ids]
                )
                self.goal_west_pos[env_ids] = (
                    self.object.data.nodal_pos_w[env_ids, self.anchor_idx["west"], :]
                    - self.scene.env_origins[env_ids]
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

        # Dynamic outward targets track live fingertips; depth / ellipse below use **env-local base** offsets from ``goal_cent``.
        self._update_goal_aperture_targets(env_ids)

        # Visualize thumb and pinky targets (must index by env_ids: subset reset has |env_ids| < num_envs)
        # self.thumb_target_markers.visualize(
        #     self.thumb_target[env_ids] + self.scene.env_origins[env_ids],
        #     self.identity_quat[env_ids],
        # )
        # self.pinky_target_markers.visualize(
        #     self.pinky_target[env_ids] + self.scene.env_origins[env_ids],
        #     self.identity_quat[env_ids],
        # )
        # if self.goal_east_markers is not None:
        #     self.goal_east_markers.visualize(
        #         self.goal_east_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
        #     )
        #     self.goal_west_markers.visualize(
        #         self.goal_west_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
        #     )
        #     self.goal_north_markers.visualize(
        #         self.goal_north_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
        #     )
        #     self.goal_south_markers.visualize(
        #         self.goal_south_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
        #     )
        #     self.goal_cent_markers.visualize(
        #         self.goal_cent_pos[env_ids] + self.scene.env_origins[env_ids], self.identity_quat[env_ids]
        #     )
        #     # print(f"goal_north_pos: {self.goal_north_pos[0]}, goal_south_pos: {self.goal_south_pos[0]}, goal_east_pos: {self.goal_east_pos[0]}, goal_west_pos: {self.goal_west_pos[0]}, goal_cent_pos: {self.goal_cent_pos[0]}")

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

        # upper/under distance
        self.wrist_ee_distance[env_ids] = self.ee_pos[env_ids] - self.goal_wrist_pos[env_ids]
        self.wrist_ee_euclidean_distance[env_ids] = torch.norm(self.wrist_ee_distance[env_ids], dim=1)

        if getattr(self.cfg, "deformable_bracelet_geom_rim_goals", False) and self.cfg.object_type == "deformable":
            self.top_wrist_distance[env_ids] = self.goal_north_pos[env_ids] - self.goal_wrist_pos[env_ids]
            self.under_wrist_distance[env_ids] = self.goal_wrist_pos[env_ids] - self.goal_south_pos[env_ids]
        else:
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
        # Env-local (robot base) bracelet metrics — same as :class:`~tasks.airec.reach_bracelet.ReachBraceletEnv`.
        # All positions are already env-local; no rim-PCA / ``quat_apply_inverse`` opening frame.
        # ------------------------------------------------------------------
        self.wrist_center_distance[env_ids] = self.goal_wrist_pos[env_ids] - self.goal_cent_pos[env_ids]
        self.wrist_center_euclidean_distance[env_ids] = torch.norm(self.wrist_center_distance[env_ids], dim=1)

        if self._use_glove or self.cfg.object_type == "rigid":
            desired = torch.as_tensor(
                self.cfg.bracelet_desired_insert_depth, device=self.device, dtype=torch.float32
            )
            wrist_dx = self.goal_wrist_pos[env_ids, 0] - self.goal_cent_pos[env_ids, 0]
            self.insert_depth[env_ids] = wrist_dx
            self.depth_distance[env_ids] = torch.abs(wrist_dx - desired)

            self.depth_thumb_distance[env_ids] = torch.abs(
                self.goal_west_pos[env_ids, 0] - self.thumb_target[env_ids, 0]
            )
            self.depth_pinky_distance[env_ids] = torch.abs(
                self.goal_east_pos[env_ids, 0] - self.pinky_target[env_ids, 0]
            )

            _thumb_inside_ellipse = self._ellipse_inner_ring_gate_zy(env_ids, self.thumb_target)
            _thumb_side_gate = self._soft_side_gate_y(env_ids, self.thumb_target, 1.0)
            self.thumb_inside_ellipse[env_ids] = _thumb_inside_ellipse * _thumb_side_gate

            _pinky_inside_ellipse = self._ellipse_inner_ring_gate_zy(env_ids, self.pinky_target)
            _pinky_side_gate = self._soft_side_gate_y(env_ids, self.pinky_target, -1.0)
            self.pinky_inside_ellipse[env_ids] = _pinky_inside_ellipse * _pinky_side_gate

            self.wrist_inside_ellipse[env_ids] = self._ellipse_soft_gate_zy(env_ids, self.goal_wrist_pos)

            rad_eps = torch.as_tensor(1e-4, device=self.device, dtype=self.goal_wrist_pos.dtype)
            radius_y = 0.5 * torch.abs(
                self.goal_east_pos[env_ids, 1] - self.goal_west_pos[env_ids, 1]
            ).clamp_min(rad_eps)
            radius_z = 0.5 * torch.abs(
                self.goal_north_pos[env_ids, 2] - self.goal_south_pos[env_ids, 2]
            ).clamp_min(rad_eps)
            dy = self.wrist_center_distance[env_ids, 1]
            dz = self.wrist_center_distance[env_ids, 2]
            radial_normalized = (dy / radius_y).pow(2) + (dz / radius_z).pow(2)
            outside_error = torch.clamp(radial_normalized - 1.0, min=0.0)
            std = torch.as_tensor(self.cfg.bracelet_inside_opening_std, device=self.device).clamp_min(1e-6)
            self.inside_opening_soft[env_ids] = torch.exp(-outside_error / std)
            self.wrist_radial_normalized[env_ids] = radial_normalized
            self.wrist_xy_center_distance[env_ids] = torch.norm(self.wrist_center_distance[env_ids, 1:3], dim=-1)

            finger_heights = torch.stack(
                [
                    self.thumb_target[env_ids, 2],
                    self.fore_goal_pos[env_ids, 2],
                    self.middle_goal_pos[env_ids, 2],
                    self.ring_goal_pos[env_ids, 2],
                    self.pinky_target[env_ids, 2],
                ],
                dim=-1,
            )
            dist_from_south = finger_heights - self.goal_south_pos[env_ids, 2].unsqueeze(-1)
            dist_from_north = self.goal_north_pos[env_ids, 2].unsqueeze(-1) - finger_heights
            margin = torch.minimum(dist_from_south, dist_from_north)
            self.per_finger_soft_inside[env_ids] = torch.sigmoid(margin / 0.02)
            self.fingers_inside_soft_gate[env_ids] = self.per_finger_soft_inside[env_ids].mean(dim=-1)
        else:
            self.depth_distance[env_ids] = 0.0
            self.depth_thumb_distance[env_ids] = 0.0
            self.depth_pinky_distance[env_ids] = 0.0
            self.inside_opening_soft[env_ids] = 0.0
            self.wrist_radial_normalized[env_ids] = 0.0
            self.insert_depth[env_ids] = 0.0
            self.thumb_inside_ellipse[env_ids] = 0.0
            self.pinky_inside_ellipse[env_ids] = 0.0
            self.wrist_inside_ellipse[env_ids] = 0.0
            self.fingers_inside_soft_gate[env_ids] = 0.0
            self.wrist_xy_center_distance[env_ids] = 0.0
            self.per_finger_soft_inside[env_ids] = 0.0
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
    depth_distance: torch.Tensor,
    depth_thumb_distance: torch.Tensor,
    depth_pinky_distance: torch.Tensor,
    wrist_height: torch.Tensor,
    top_height: torch.Tensor,
    bottom_height: torch.Tensor,
    thumb_height: torch.Tensor,
    pinky_height: torch.Tensor,
    wrist_center_euclidean_distance: torch.Tensor,
    thumb_inside_ellipse: torch.Tensor,
    pinky_inside_ellipse: torch.Tensor,
    wrist_inside_ellipse: torch.Tensor,
    fingers_inside_soft_gate: torch.Tensor,
):
    # reward weights
    depth_reward_scale = 0.0
    depth_thumb_reward_scale = 0.0
    depth_pinky_reward_scale = 0.0
    # rewards thresholds
    ee_distance_threshold = 0.3 # default 0.3
    right_ee_thumb_angular_threshold = 1.4
    left_ee_pinky_angular_threshold = 0.8
    ######## conditions for rewards ########
    ee_near_condition = (ee_euclidean_distance < ee_distance_threshold) #& (right_ee_thumb_angular_distance < ee_angular_thretholds["right_ee_thumb"])
    right_ee_thumb_angular_condition = (right_ee_thumb_angular_distance < right_ee_thumb_angular_threshold)
    left_ee_pinky_angular_condition = (left_ee_pinky_angular_distance < left_ee_pinky_angular_threshold)
    wrist_between_height_condition = (top_height > wrist_height) & (wrist_height > bottom_height)
    thumb_between_height_condition = (top_height > thumb_height) & (thumb_height > bottom_height)
    pinky_between_height_condition = (top_height > pinky_height) & (pinky_height > bottom_height)
  
    ######## rewards for reaching ########
    reaching_right_ee_thumb_scale = 2.0
    reaching_left_ee_pinky_scale = 1.0
    right_ee_thumb_condition = (ee_near_condition) & thumb_between_height_condition
    left_ee_pinky_condition = (ee_near_condition) & pinky_between_height_condition 
    # print(f"thumb_inside_ellipse: {thumb_inside_ellipse[0]}, pinky_inside_ellipse: {pinky_inside_ellipse[0]}, wrist_inside_ellipse: {wrist_inside_ellipse[0]}")
    r_right_ee_thumb_distance = (
        distance_reward(right_ee_thumb_euclidean_distance, std=0.10) 
        * reaching_right_ee_thumb_scale 
        * (right_ee_thumb_condition) 
        # * thumb_inside_ellipse # default 0.15
    )
    r_left_ee_pinky_distance = (
        distance_reward(left_ee_pinky_euclidean_distance, std=0.05) 
        * reaching_left_ee_pinky_scale 
        * (left_ee_pinky_condition) 
        # * pinky_inside_ellipse # default 0.10
    )

    ######## rewards for insert ########
    reaching_wrist_center_scale = 20.0
    wrist_center_condition = ee_near_condition 
    
    r_wrist_center_distance = (
        distance_reward(wrist_center_euclidean_distance, std=0.14)
        * reaching_wrist_center_scale 
        * wrist_center_condition
        * fingers_inside_soft_gate
    )
    ######### rewards for angular #########
    rotation_right_ee_thumb_scale = 0.0
    rotation_left_ee_pinky_scale = 0.0
    right_ee_thumb_rotation_condition = (ee_near_condition) & thumb_between_height_condition
    left_ee_pinky_rotation_condition = (ee_near_condition) & pinky_between_height_condition
    # print(f"right_ee_thumb_angular_distance: {right_ee_thumb_angular_distance[0]}, left_ee_pinky_angular_distance: {left_ee_pinky_angular_distance[0]}")

    r_angular_right_ee_thumb = (
        angular_distance_reward(right_ee_thumb_angular_distance, std=0.2) 
        * rotation_right_ee_thumb_scale 
        * (right_ee_thumb_rotation_condition)
    )
    r_angular_left_ee_pinky = (
        angular_distance_reward(left_ee_pinky_angular_distance, std=0.15) 
        * rotation_left_ee_pinky_scale
        * (left_ee_pinky_rotation_condition)
    )

    r_depth_distance = (
        distance_reward(depth_distance, std=0.15)
        * depth_reward_scale
        * ee_near_condition
    )
    r_depth_thumb_distance = (
        distance_reward(depth_thumb_distance, std=0.1)
        * thumb_between_height_condition
        * depth_thumb_reward_scale
        * ee_near_condition
    )
    r_depth_pinky_distance = (
        distance_reward(depth_pinky_distance, std=0.08)
        * pinky_between_height_condition
        * depth_pinky_reward_scale
        * ee_near_condition
    )
    # r_successed = success_reward(wrist_ee_distance, wrist_pos, top_pos, under_pos, minimal_distance)
    rewards = r_right_ee_thumb_distance + r_left_ee_pinky_distance + r_depth_distance + r_depth_thumb_distance + r_depth_pinky_distance + r_angular_right_ee_thumb + r_angular_left_ee_pinky + r_wrist_center_distance

    return (rewards, r_right_ee_thumb_distance, r_left_ee_pinky_distance, r_depth_distance, r_depth_thumb_distance, r_depth_pinky_distance, r_angular_right_ee_thumb, r_angular_left_ee_pinky, r_wrist_center_distance)

