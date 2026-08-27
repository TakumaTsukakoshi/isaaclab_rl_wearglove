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
from tasks.airec.physics import tune_physx_gpu_buffers_for_vec_env

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
    reset_goal_position_noise = 1.0  # scale factor for -1 to 1 m
    default_goal_pos = [0.5, 0.5, 0.4]
    default_thumb_goal_pos = [0.70, -0.050, 1.07]
    default_pinky_goal_pos = [0.70, 0.050, 1.07]
    # default_object_pos = [0.27, 0.00, 1.07] # 0.13 # 1.07　default maybe for airec1
    # default_object_pos = [0.27, 0.00, 1.07] # airec1
    # default_object_pos = [0.26, 0.00, 0.85] # airec2 default
    default_object_pos = [0.24, 0.00, 0.85] # airec2 rigid bracelet
    # default_object_pos = [0.24, 0.00, 0.90] # airec2


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
    #: Opening-frame Z target for the wrist (m) in depth reward ``abs(z - desired)``.
    bracelet_desired_insert_depth: float = 0.0
    #: Soft in-opening gate ``exp(-max(0, radial^2-1)/std)``; larger = more lenient outside the ellipse.
    bracelet_inside_opening_std: float = 0.15
    #: ``soft`` = mean(sigmoid(m_i / k)); ``hard`` = mean(1[m_i > 0]). Reward uses ``fingers_inside_soft_gate``.
    insertion_gate_mode: str = "soft"
    #: Temperature ``k`` in ``g_i = sigmoid(m_i / k)`` for soft per-finger insertion gates (m = margin to opening rims).
    insertion_gate_temperature: float = 0.01
    #: Normalized opening ellipse in Y-Z: inside when ``ellipse_value <= eval_opening_ellipse_threshold``.
    eval_opening_ellipse_threshold: float = 1.0

    #: Thumb deep-inside gate (Y–Z ``ev`` from ``thumb_target``): peaks when ``ev << 1`` (center), zero at rim/outside.
    #: ``sigmoid(sharpness * (margin - ev))`` with ``margin < 1`` avoids rewarding ``ev ≈ 1`` (on edge).
    thumb_ellipse_inside_margin: float = 0.85
    thumb_ellipse_inside_sharpness: float = 8.0
    #: Upper-arm EE must be near ``thumb_target`` (3D, m) for the reach gate to turn on.
    thumb_upper_ee_proximity_std: float = 0.10
    #: After wrist reaches the opening (``wrist_center_euclidean_distance < bracelet_success_threshold``),
    #: thumb reach gate is zero (post-insert / success phase does not need a high gate).
    thumb_gate_active_before_wrist_success: bool = False

    # Sparse task-success: wrist goal within ``bracelet_success_threshold`` (env-local, m).
    # When False (default), success does not end the episode; only failure terms / time-out do.
    # ``task_success_bonus`` is awarded once per episode on the first success step.
    terminate_on_task_success: bool = False
    bracelet_success_threshold: float = 0.01  # 1 cm
    task_success_bonus: float = 1000.0
    #: After the one-shot success bonus, ignore further policy actions and hold actuated joint
    #: targets at the measured success pose until reset. Episode still continues (no terminate).
    lock_motion_after_task_success: bool = True
    #: Train and play_eval: ``task_success`` / bonus / motion lock require
    #: ``wrist_within_goal AND all_5_confirmed_insertions``. Episode still runs to timeout.
    #: ``play_eval.py --no-complete-dressing-success`` restores wrist-only lock for eval.
    eval_success_requires_all_fingers: bool = True
    eval_insertion_delta_m: float = 0.003
    eval_insertion_confirm_frames: int = 4

    #: Print policy action vs ``joint_pos_cmd`` vs sim ``joint_pos`` every ``debug_joint_print_interval`` steps.
    debug_joint_cmd_vs_actual: bool = False
    debug_joint_print_env_id: int = 0
    debug_joint_print_interval: int = 1

    #: When cumulative episode success rate exceeds :attr:`adaptive_physics_success_threshold`, switch from
    #: coarse (:attr:`~tasks.airec.airec2_finger.AIRECEnvCfg.physics_dt` / :attr:`~tasks.airec.airec2_finger.AIRECEnvCfg.decimation`)
    #: to fine PhysX (``fine_physics_dt`` / ``fine_decimation``). RL control step stays ``1/10`` s in both cases.
    adaptive_physics_on_success: bool = False
    adaptive_physics_success_threshold: float = 0.5
    adaptive_physics_min_episodes: int = 20
    fine_physics_dt: float = 1 / 2000
    fine_decimation: int = 200

    #: Rim / fingertip ``VisualizationMarkers`` spheres. Requires GUI (do not use ``--headless``).
    show_task_markers: bool = False
    #: Disable markers when ``scene.num_envs`` exceeds this (Fabric GPU OOM with 1k+ envs).
    show_task_markers_max_num_envs: int = 256
    #: Geometric opening-rim PCA debug (``bracelet_opening_pca_viz``). Requires GUI for debug_draw.
    debug_opening_pca_on_reset: bool = False
    debug_opening_pca_refresh_each_step: bool = False
    debug_opening_pca_arrow_scale: float = 0.08
    debug_opening_pca_use_rest_pose: bool = False
    debug_opening_pca_use_opening_ring: bool = False

    object_usd = os.path.join(
        # _REPO_ROOT, "assets", "Bracelet", "bracelet_b.usd"
        _REPO_ROOT, "assets", "Bracelet", "bracelet_b_new.usd"
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
                # contact_offset=0.006, # default 0.005
                # rest_offset=0.003, # default 0.003
                contact_offset=0.01, # default 0.005
                rest_offset=0.005, # default 0.003
            ),


            # Rigid body only (deformable hexa / remesh / soft-contact fields belong on DeformableBodyPropertiesCfg).
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=32,
                max_depenetration_velocity=1.0,
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

    # Finger / wrist reach-target debug spheres (env-local). Saturated, distinct; no pure red or green.
    thumb_target_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/thumb_target_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.0)),
            ),
        },
    )
    fore_target_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/fore_target_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.55, 1.0)),
            ),
        },
    )
    middle_target_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/middle_target_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        },
    )
    ring_target_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ring_target_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            ),
        },
    )
    pinky_target_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pinky_target_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.5, 1.0)),
            ),
        },
    )
    wrist_target_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/wrist_target_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            ),
        },
    )
    right_ee_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/right_ee_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.0)),
            ),
        },
    )
    left_ee_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/left_ee_marker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.25, 1.0)),
            ),
        },
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
        # Keep ``cfg.sim`` aligned with coarse timestep (configclass ``sim`` is bound at class definition time).
        cfg.sim.dt = cfg.physics_dt
        cfg.sim.render_interval = cfg.decimation
        tune_physx_gpu_buffers_for_vec_env(
            cfg.sim.physx,
            int(cfg.scene.num_envs),
            deformable=False,
        )
        # Rigid AIREC + Shadow Hand + bracelet generates far more contact patches than
        # Isaac's default ``5 * 2**15`` (163840). PhysX asked ~320k at 4096 envs; the
        # tuner above clamps patches to that default and overflows. Restore a floor.
        n_env = int(cfg.scene.num_envs)
        patch_floor = 2**20 if n_env > 1024 else 2**19  # 1_048_576 / 524_288
        contact_floor = 2**22 if n_env > 1024 else 2**21
        physx = cfg.sim.physx
        if hasattr(physx, "gpu_max_rigid_patch_count"):
            physx.gpu_max_rigid_patch_count = max(
                int(physx.gpu_max_rigid_patch_count or 0), patch_floor
            )
        if hasattr(physx, "gpu_max_rigid_contact_count"):
            physx.gpu_max_rigid_contact_count = max(
                int(physx.gpu_max_rigid_contact_count or 0), contact_floor
            )
        max_m = int(getattr(cfg, "show_task_markers_max_num_envs", 256))
        use_markers = bool(cfg.show_task_markers) and cfg.scene.num_envs <= max_m
        if bool(cfg.show_task_markers) and not use_markers:
            print(
                f"[ReachBraceletEnv] show_task_markers disabled "
                f"(num_envs={cfg.scene.num_envs} > {max_m})"
            )
        cfg.show_task_markers = use_markers

        super().__init__(cfg, render_mode, **kwargs)
        if bool(getattr(cfg, "show_task_markers", False)) and not self.sim.has_gui():
            print(
                "[ReachBraceletEnv] show_task_markers=True but no GUI "
                "(omit --headless). Markers disabled."
            )
            cfg.show_task_markers = False
            self.cfg.show_task_markers = False

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
        # Last aperture reset: wrist reference and lateral axis in **env-local** / **world direction** respectively.
        self.wrist_origin = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.wrist_lateral_axis = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.unit_dir = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # Dynamic outward-biased reach targets (**env-local**), updated each step from live ShadowHand fingertips.
        self.thumb_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.pinky_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.wrist_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
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
        # self.right_ee_thumb_distance = torch.zeros((self.num_envs, 3), device=self.device)
        # self.right_ee_thumb_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.right_upper_ee_thumb_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_upper_ee_thumb_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.right_ee_thumb_rotation = torch.zeros((self.num_envs, 4), device=self.device)
        self.right_ee_thumb_angular_distance = torch.zeros((self.num_envs,), device=self.device)
        # self.left_ee_goal_distance = torch.zeros((self.num_envs, 3), device=self.device)
        # self.left_ee_pinky_distance = torch.zeros((self.num_envs, 3), device=self.device)
        # self.left_ee_pinky_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
        self.left_upper_ee_pinky_distance = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_upper_ee_pinky_euclidean_distance = torch.zeros((self.num_envs,), device=self.device)
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
        self.insert_thumb_reward = InsertReward(self.num_envs, device=self.device, inward_assume="+x")
        self.insert_pinky_reward = InsertReward(self.num_envs, device=self.device, inward_assume="+x")
        self.insert_fore_reward = InsertReward(self.num_envs, device=self.device, inward_assume="+x")
        self.insert_middle_reward = InsertReward(self.num_envs, device=self.device, inward_assume="+x")
        self.insert_ring_reward = InsertReward(self.num_envs, device=self.device, inward_assume="+x")

        self.thumb_insert_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.thumb_insert_dwell = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.pinky_insert_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.pinky_insert_dwell = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.fore_insert_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.fore_insert_dwell = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.middle_insert_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.middle_insert_dwell = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.ring_insert_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.ring_insert_dwell = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # Sparse task-success buffer (bracelet within ``cfg.bracelet_success_threshold`` of wrist goal).
        # Populated each step in ``_get_dones`` and consumed in ``_get_rewards`` for the +bonus term.
        self.task_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # One-shot gate so success bonus is not re-awarded every step while success holds.
        self._task_success_bonus_awarded = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        # Actuated-joint pose snapshot used when ``lock_motion_after_task_success`` is True.
        self._success_hold_joint_pos = torch.zeros(
            (self.num_envs, len(self.actuated_dof_indices)),
            device=self.device,
            dtype=torch.float32,
        )
        # Complete-dressing buffers (wrist + 5-finger crossing tracker for bonus / lock).
        self.wrist_within_goal = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.eval_all_5_inserted = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._episode_end_eval_inserted = torch.zeros(
            (self.num_envs, 5), dtype=torch.bool, device=self.device
        )
        self._eval_finger_crossing_tracker = None
        self._eval_base_body_ids: dict[str, int | None] | None = None

        # debugging
        if self._is_free_space_mode():
            self.finger_joint_ids = []
            self._shadow_hand_finger_hold = torch.zeros(
                (self.num_envs, 0), device=self.device, dtype=torch.float32
            )
        else:
            self.finger_joint_ids, _ = self.hand.find_joints(self.cfg.finger_joint_names)
            self._shadow_hand_finger_hold = torch.zeros(
                (self.num_envs, len(self.finger_joint_ids)),
                device=self.device,
                dtype=self.hand.data.joint_pos.dtype,
            )
        self.wrist_xy_center_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        # Env-local vector from opening center to wrist (``torch.norm(..., dim=1)`` → ``wrist_center_euclidean_distance``).
        self.wrist_center_distance = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # Backward-compatible alias (same scalar).
        self.wrist_center_euclidean_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.thumb_inside_ellipse = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.thumb_target_upper_ee_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.thumb_upper_ee_proximity = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        # Normalized Y–Z ellipse value for ``thumb_target`` (or blended gate position); ``< 1`` = inside opening.
        self.thumb_ellipse_value = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.thumb_reach_gate_active = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.pinky_inside_ellipse = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.wrist_inside_ellipse = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.fingers_inside_soft_gate = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.fingers_inside_hard_gate = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.per_finger_soft_inside = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)
        self.per_finger_hard_inside = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)
        self.per_finger_insert_margin = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)
        self.per_finger_height_z = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)
        self.per_finger_ellipse_value = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)
        self.per_finger_inside_ellipse = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)
    

    def _apply_action(self) -> None:
        super()._apply_action()
        if self._is_free_space_mode():
            return
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
        if self._is_free_space_mode():
            self.goal_north_markers = None
            self.goal_south_markers = None
            self.goal_east_markers = None
            self.goal_west_markers = None
            self.goal_cent_markers = None
            self.thumb_target_markers = None
            self.pinky_target_markers = None
            return
        # Rigid / deformable task object (bracelet) is added whenever ``object_type != "none"``.
        if self.cfg.object_type != "none":
            self._add_object_to_scene()
        # N/S/E/W/C markers: deformable glove uses nodal rim; rigid bracelet uses root pose + cfg offsets.
        if self.cfg.show_task_markers and (self._use_glove or self.cfg.object_type == "rigid"):
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
        if self.cfg.show_task_markers:
            self.thumb_target_markers = VisualizationMarkers(self.cfg.thumb_target_marker)
            self.pinky_target_markers = VisualizationMarkers(self.cfg.pinky_target_marker)
            self.fore_target_markers = VisualizationMarkers(self.cfg.fore_target_marker)
            self.middle_target_markers = VisualizationMarkers(self.cfg.middle_target_marker)
            self.ring_target_markers = VisualizationMarkers(self.cfg.ring_target_marker)
            self.wrist_target_markers = VisualizationMarkers(self.cfg.wrist_target_marker)
            self.right_ee_markers = VisualizationMarkers(self.cfg.right_ee_marker)
            self.left_ee_markers = VisualizationMarkers(self.cfg.left_ee_marker)
        else:
            self.thumb_target_markers = None
            self.pinky_target_markers = None
            self.fore_target_markers = None
            self.middle_target_markers = None
            self.ring_target_markers = None
            self.wrist_target_markers = None
            self.right_ee_markers = None
            self.left_ee_markers = None
        

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
                self.right_upper_ee_thumb_distance,
                # euclidean distance (1,)
                self.right_upper_ee_thumb_euclidean_distance.unsqueeze(1),
                ## xyz diffs (3,)
                self.left_upper_ee_pinky_distance,
                # euclidean distances (1,)
                self.left_upper_ee_pinky_euclidean_distance.unsqueeze(1),
    
                # xyz diffs (3,)
                self.wrist_center_distance,
                # euclidean distance (1,)
                self.wrist_center_euclidean_distance.unsqueeze(1),
                # per finger soft inside (5,)
                self.per_finger_soft_inside,
                # xyz com_b (3,)
                self.com_pos_b,
            ),
            dim=-1,
        )
        return gt

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        super()._pre_physics_step(actions)
        if not bool(getattr(self.cfg, "lock_motion_after_task_success", False)):
            return
        locked = self._task_success_bonus_awarded
        if not bool(locked.any()):
            return
        ids = locked.nonzero(as_tuple=False).flatten()
        # Overwrite policy-mapped targets with the pose frozen at success-bonus time.
        self.joint_pos_cmd[ids.unsqueeze(1), self.actuated_idx.unsqueeze(0)] = (
            self._success_hold_joint_pos[ids]
        )
        self.prev_joint_pos_cmd[ids.unsqueeze(1), self.actuated_idx.unsqueeze(0)] = (
            self._success_hold_joint_pos[ids]
        )

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
                "[ReachBraceletEnv] Upgraded physics: "
                f"physics_dt={self.cfg.fine_physics_dt:.6g}, decimation={self.cfg.fine_decimation} "
                f"(success_rate={rate:.3f} over {self._curriculum_episode_count} episodes)"
            )

    def _episode_success_rate(self) -> float:
        """Fraction of completed episodes that reached task success (bonus awarded)."""
        return self._curriculum_success_count / max(self._curriculum_episode_count, 1)

    def _update_success_stats_on_reset(self, env_ids) -> None:
        """Count finished episodes that earned the one-shot success bonus (before clearing gates)."""
        if env_ids is None:
            return
        env_ids = self._normalize_env_ids(env_ids)
        # Skip cold-start ``__init__`` reset (episode length still 0).
        alive = self.episode_length_buf[env_ids] > 0
        if not bool(alive.any()):
            return
        env_ids = env_ids[alive]
        n = int(env_ids.numel())
        # Bonus gate = success at any point this episode (not only the final step).
        successes = int(self._task_success_bonus_awarded[env_ids].sum().item())
        self._curriculum_episode_count += n
        self._curriculum_success_count += successes
        if self.cfg.adaptive_physics_on_success and not self._physics_timestep_upgraded:
            self._maybe_upgrade_physics_timestep()

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            reset_ids = self.robot._ALL_INDICES
        else:
            reset_ids = self._normalize_env_ids(env_ids)
        # Record episode success *before* clearing one-shot gates / hold pose.
        self._update_success_stats_on_reset(reset_ids)

        super()._reset_idx(env_ids)
        if env_ids is None:
            e = self.robot._ALL_INDICES
        else:
            e = self._normalize_env_ids(env_ids)
        self.prev_actions[e] = 0.0
        # One-shot success bonus / motion lock are per-episode; must clear on every reset.
        self.task_success[e] = False
        self._task_success_bonus_awarded[e] = False
        self._success_hold_joint_pos[e] = 0.0
        self.wrist_within_goal[e] = False
        self.eval_all_5_inserted[e] = False
        # Keep ``_episode_end_eval_*`` snapshots for play_eval to read after Isaac Lab reset.
        tracker = getattr(self, "_eval_finger_crossing_tracker", None)
        if tracker is not None:
            tracker.reset_envs(e)
        if self._is_free_space_mode():
            return

        # When ``_use_glove`` is False the base reset can skip goal-hand setup (``object_type=="none"`` path).
        # For rigid bracelet we still refresh ShadowHand pose and thumb/pinky aperture after reset.
        if not self._use_glove:
            self._reset_target_pose(e)
            # Refresh transforms before aperture logic (thumb/pinky goal frames depend on ShadowHand pose).
            self._compute_intermediate_values(env_ids=e)
            self._reset_goal_aperture(e)

    def _eval_complete_success_enabled(self) -> bool:
        """True when bonus / motion lock need wrist + all five confirmed insertions."""
        return bool(getattr(self.cfg, "eval_success_requires_all_fingers", True)) and (
            not self._is_free_space_mode()
        )

    def _ensure_eval_insertion_tracker(self):
        """Reuse ``bracelet_eval.FingerCrossingTracker`` (same confirmed-insertion detector)."""
        if self._eval_finger_crossing_tracker is not None:
            return self._eval_finger_crossing_tracker
        from bracelet_eval import FingerCrossingTracker

        self._eval_finger_crossing_tracker = FingerCrossingTracker(
            int(self.num_envs),
            self.device,
            torch.float32,
            delta=float(getattr(self.cfg, "eval_insertion_delta_m", 0.003)),
            confirm_frames=int(getattr(self.cfg, "eval_insertion_confirm_frames", 4)),
            ellipse_threshold=float(getattr(self.cfg, "eval_opening_ellipse_threshold", 1.0)),
        )
        return self._eval_finger_crossing_tracker

    def _stack_eval_finger_base_env_local(self) -> torch.Tensor | None:
        """``(num_envs, 5, 3)`` finger-base COMs in env-local frame (same bodies as eval)."""
        from bracelet_eval import BASE_BODY_CANDIDATES, FINGER_ORDER, _resolve_body_index

        hand = getattr(self, "hand", None)
        if hand is None:
            return None
        body_pos_w = getattr(hand.data, "body_pos_w", None)
        if body_pos_w is None:
            return None
        if self._eval_base_body_ids is None:
            body_names = list(
                getattr(hand, "body_names", None) or getattr(hand.data, "body_names", None) or []
            )
            self._eval_base_body_ids = {
                name: _resolve_body_index(body_names, candidates)
                for name, candidates in BASE_BODY_CANDIDATES.items()
            }
        if any(self._eval_base_body_ids[name] is None for name in FINGER_ORDER):
            return None
        origins = getattr(self, "env_origins", None)
        if origins is None:
            scene = getattr(self, "scene", None)
            origins = getattr(scene, "env_origins", None) if scene is not None else None
        if origins is None:
            origins = body_pos_w.new_zeros((body_pos_w.shape[0], 3))
        origins = origins.to(device=body_pos_w.device, dtype=body_pos_w.dtype)
        cols = [body_pos_w[:, self._eval_base_body_ids[name]] - origins for name in FINGER_ORDER]
        return torch.stack(cols, dim=1)

    def _update_eval_insertion_tracker(self) -> torch.Tensor | None:
        """Advance the shared crossing tracker after live opening / finger poses are current."""
        from bracelet_eval import opening_radii

        distal = self._stack_eval_finger_base_env_local()
        cent = getattr(self, "goal_cent_pos", None)
        east = getattr(self, "goal_east_pos", None)
        west = getattr(self, "goal_west_pos", None)
        north = getattr(self, "goal_north_pos", None)
        south = getattr(self, "goal_south_pos", None)
        if distal is None or cent is None or east is None or west is None or north is None or south is None:
            self._episode_end_eval_inserted.zero_()
            return None
        radius_y, radius_z = opening_radii(east, west, north, south)
        tracker = self._ensure_eval_insertion_tracker()
        active = torch.ones((int(self.num_envs),), dtype=torch.bool, device=distal.device)
        inserted = tracker.update(distal, cent, radius_y, radius_z, active)
        self._episode_end_eval_inserted.copy_(inserted)
        self._episode_end_eval_max_inserted = tracker.max_inserted.clone()
        self._episode_end_eval_ever_all = tracker.ever_all.clone()
        self._episode_end_eval_first_insert_step = tracker.first_insert_step.clone()
        self._episode_end_eval_insert_steps = tracker.inserted_steps.clone()
        return inserted

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._is_free_space_mode():
            self._compute_intermediate_values()
            com_tip = (self.com_pos_b[:, 0] < float(self.cfg.com_tip_x_min)) | (
                self.com_pos_b[:, 0] > float(self.cfg.com_tip_x_max)
            )
            termination = com_tip
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            self.task_success.zero_()
            self._term_log = {
                "term_com_tip": com_tip.float(),
                "com_pos_b_x": self.com_pos_b[:, 0].clone(),
                "term_any": termination.float(),
            }
            self._episode_end_per_finger_soft_inside = self.per_finger_soft_inside.clone()
            self._episode_end_per_finger_insert_margin = self.per_finger_insert_margin.clone()
            self._episode_end_per_finger_height_z = self.per_finger_height_z.clone()
            self._episode_end_per_finger_ellipse_value = self.per_finger_ellipse_value.clone()
            self._episode_end_per_finger_inside_ellipse = self.per_finger_inside_ellipse.clone()
            self._episode_end_task_success = self.task_success.clone()
            return termination, time_out

        # Parent calls ``_compute_intermediate_values()`` and writes ``self._term_log`` for wandb.
        termination, time_out = super()._get_dones()

        # Bracelet root vs. ShadowHand wrist goal, both env-local (see ``_compute_intermediate_values``).
        wrist_ok = self.wrist_center_euclidean_distance < self.cfg.bracelet_success_threshold
        self.wrist_within_goal[:] = wrist_ok

        if self._eval_complete_success_enabled():
            # Insertion first (this step's live geometry), then complete success, then latch.
            inserted = self._update_eval_insertion_tracker()
            all_5 = (
                inserted.all(dim=1)
                if inserted is not None
                else torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            )
            self.eval_all_5_inserted[:] = all_5
            complete_success = wrist_ok & all_5
            self.task_success |= complete_success
        else:
            self.eval_all_5_inserted.zero_()
            self.task_success[:] = wrist_ok

        # Surface to wandb via the same ``_term_log`` dict that ``AIRECEnv._get_dones`` populates.
        if not hasattr(self, "_term_log") or self._term_log is None:
            self._term_log = {}
        self._term_log["task_success"] = self.task_success.float()
        self._term_log["wrist_center_distance"] = self.wrist_center_euclidean_distance
        if self._eval_complete_success_enabled():
            self._term_log["wrist_within_goal"] = self.wrist_within_goal.float()
            self._term_log["eval_all_5_inserted"] = self.eval_all_5_inserted.float()

        if self.cfg.terminate_on_task_success:
            termination = termination | self.task_success
            self._term_log["term_any"] = termination.float()

        self._episode_end_per_finger_soft_inside = self.per_finger_soft_inside.clone()
        self._episode_end_per_finger_insert_margin = self.per_finger_insert_margin.clone()
        self._episode_end_per_finger_height_z = self.per_finger_height_z.clone()
        self._episode_end_per_finger_ellipse_value = self.per_finger_ellipse_value.clone()
        self._episode_end_per_finger_inside_ellipse = self.per_finger_inside_ellipse.clone()
        self._episode_end_opening_south_z = self.goal_south_pos[:, 2].clone()
        self._episode_end_opening_north_z = self.goal_north_pos[:, 2].clone()
        self._episode_end_task_success = self.task_success.clone()
        self._episode_end_wrist_center_euclidean_distance = self.wrist_center_euclidean_distance.clone()


        return termination, time_out

    def _get_rewards(self) -> torch.Tensor:
        if self._is_free_space_mode():
            rewards = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            self.extras["log"] = {"free_space_reward": rewards}
            term_log = getattr(self, "_term_log", None)
            if term_log is not None:
                self.extras["log"].update(term_log)
            self.extras["counters"] = {}
            self._debug_print_joint_cmd_vs_actual()
            return rewards

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
                self.right_upper_ee_thumb_euclidean_distance,
                self.left_upper_ee_pinky_euclidean_distance,
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
                "fingers_inside_soft_gate": self.fingers_inside_soft_gate,
                # "thumb_upper_ee_proximity": self.thumb_upper_ee_proximity,
                # "thumb_ellipse_value": self.thumb_ellipse_value,
            }

        if "tactile" in self.cfg.obs_list:
            self.extras["log"].update(
                {
                    "normalised_forces_left_x": self.normalised_forces[:, 0],
                    "normalised_forces_right_x": self.normalised_forces[:, 1],
                }
            )

        # Sparse task-success bonus (one-shot): bracelet within ``bracelet_success_threshold`` (m).
        # ``_get_dones`` runs before ``_get_rewards`` each control step, so ``self.task_success`` is current.
        newly_successful = self.task_success & ~self._task_success_bonus_awarded
        success_bonus = newly_successful.float() * float(self.cfg.task_success_bonus)
        if bool(getattr(self.cfg, "lock_motion_after_task_success", False)) and newly_successful.any():
            ids = newly_successful.nonzero(as_tuple=False).flatten()
            sl = self.actuated_dof_indices
            self._success_hold_joint_pos[ids] = self.robot.data.joint_pos[ids][:, sl].clone()
        self._task_success_bonus_awarded |= newly_successful
        rewards = rewards + success_bonus
        self.extras["log"]["task_success_bonus"] = success_bonus
        self.extras["log"]["task_success"] = self.task_success.float()
        self.extras["log"]["motion_locked"] = self._task_success_bonus_awarded.float()
        self.extras["log"]["wrist_center_distance"] = self.wrist_center_euclidean_distance
        if self._eval_complete_success_enabled():
            self.extras["log"]["wrist_within_goal"] = self.wrist_within_goal.float()
            self.extras["log"]["eval_all_5_inserted"] = self.eval_all_5_inserted.float()
            self.extras["log"]["eval_fingers_inserted"] = self._episode_end_eval_inserted.float()
        self.extras["log"]["episode_success"] = newly_successful.float()
        if self.cfg.adaptive_physics_on_success:
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

        rate = self._episode_success_rate()
        self.extras["counters"] = {
            "success_rate": torch.full(
                (self.num_envs,), rate, device=self.device, dtype=torch.float32
            ),
            "success_episodes": torch.full(
                (self.num_envs,),
                float(self._curriculum_success_count),
                device=self.device,
                dtype=torch.float32,
            ),
            "total_episodes": torch.full(
                (self.num_envs,),
                float(self._curriculum_episode_count),
                device=self.device,
                dtype=torch.float32,
            ),
        }
        self._debug_print_joint_cmd_vs_actual()
        return rewards
    
    def _normalize_env_ids(self, env_ids):
        if isinstance(env_ids, int):
            return torch.tensor([env_ids], dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
    
    # def _reset_target_pose(self, env_ids):
    #     default_state = self.hand.data.default_root_state.clone()[env_ids]

    #     num_envs = len(env_ids)

    #     # x, y: ±0.02 m, z: ±0.01 m
    #     pos_noise = torch.empty((num_envs, 3), device=self.device)
    #     pos_noise[:, 0] = sample_uniform(-0.02, 0.02, (num_envs,), device=self.device)  # x
    #     pos_noise[:, 1] = sample_uniform(-0.02, 0.02, (num_envs,), device=self.device)  # y
    #     pos_noise[:, 2] = sample_uniform(-0.01, 0.01, (num_envs,), device=self.device)  # z

    #     init_pos = default_state[0, 0:3].unsqueeze(0).repeat(num_envs, 1)

    #     default_state[:, 0:3] = (
    #         init_pos
    #         + pos_noise
    #         + self.scene.env_origins[env_ids]
    #     )

    #     init_rot = default_state[0, 3:7].unsqueeze(0).repeat(len(env_ids), 1)

    #     # Randomize pitch (Y-axis rotation) by ±5° in world frame, applied on top of the default root orientation.
    #     B = int(len(env_ids))
    #     # pitch_rad = sample_uniform(
    #     #     torch.deg2rad(torch.tensor(-5.0, device=self.device, dtype=torch.float32)),
    #     #     torch.deg2rad(torch.tensor(5.0, device=self.device, dtype=torch.float32)),
    #     #     (B,),
    #     #     device=self.device,
    #     # )
    #     yaw_rad = sample_uniform(
    #         torch.deg2rad(torch.tensor(0.0, device=self.device, dtype=torch.float32)),
    #         torch.deg2rad(torch.tensor(0.0, device=self.device, dtype=torch.float32)),
    #         (B,),
    #         device=self.device,
    #     )
    #     zero = torch.zeros_like(yaw_rad)
    #     q_yaw = quat_from_euler_xyz(zero, zero, yaw_rad)  # (B, 4) wxyz
    #     # q_pitch = quat_from_euler_xyz(zero, pitch_rad, zero)  # (B, 4) wxyz
    #     # q_yaw_pitch = quat_mul(q_yaw, q_pitch)
    #     default_state[:, 3:7] = quat_mul(q_yaw, init_rot)

    #     default_state[:, 7:] = 0.0

    #     joint_pos = self.hand.data.default_joint_pos[env_ids]
    #     joint_vel = torch.zeros_like(joint_pos)

    #     self.hand.set_joint_position_target(joint_pos, env_ids=env_ids)
    #     self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    #     self.hand.write_root_state_to_sim(default_state, env_ids=env_ids)
    #     # Cache root pose actually written (world pos + world quat) for aperture / frame logic.
    #     self.goal_hand_root_pos[env_ids] = default_state[:, 0:3].to(dtype=self.goal_hand_root_pos.dtype)
    #     self.goal_hand_root_quat[env_ids] = default_state[:, 3:7].to(dtype=self.goal_hand_root_quat.dtype)
    #     self._shadow_hand_finger_hold[env_ids] = joint_pos[:, self.finger_joint_ids].clone()
    
    def _reset_target_pose(self, env_ids):
        default_state = self.hand.data.default_root_state.clone()[env_ids]
        num_envs = len(env_ids)

        # =========================================================
        # 1. Position randomization
        # =========================================================

        # x, y: ±5 cm, z: ±1 cm
        pos_noise = torch.empty(
            (num_envs, 3),
            device=self.device,
            dtype=default_state.dtype,
        )
        pos_scale = 0.05
        pos_noise[:, 0] = sample_uniform(
            -pos_scale, pos_scale, (num_envs,), device=self.device
        )   
        pos_noise[:, 1] = sample_uniform(
            -pos_scale, pos_scale, (num_envs,), device=self.device
        )
        pos_noise[:, 2] = sample_uniform(
            -pos_scale, pos_scale, (num_envs,), device=self.device
        )

        # 各環境のdefault位置を使用
        init_pos = default_state[:, 0:3].clone()

        default_state[:, 0:3] = (
            init_pos
            + pos_noise 
            + self.scene.env_origins[env_ids]
        )

        # =========================================================
        # 2. Orientation randomization
        # =========================================================

        # 各環境のdefault orientation
        init_rot = default_state[:, 3:7].clone()

        # yaw: ±angle_range degrees
        angle_range = 10.0
        yaw_rad = sample_uniform(
            torch.deg2rad(
                torch.tensor(
                    -angle_range,
                    device=self.device,
                    dtype=default_state.dtype,
                )
            ),
            torch.deg2rad(
                torch.tensor(
                    angle_range,
                    device=self.device,
                    dtype=default_state.dtype,
                )
            ),
            (num_envs,),
            device=self.device,
        )

        # pitch: ±angle_range degrees
        pitch_rad = sample_uniform(
            torch.deg2rad(
                torch.tensor(
                    -angle_range,
                    device=self.device,
                    dtype=default_state.dtype,
                )
            ),
            torch.deg2rad(
                torch.tensor(
                    angle_range,
                    device=self.device,
                    dtype=default_state.dtype,
                )
            ),
            (num_envs,),
            device=self.device,
        )

        roll_rad = sample_uniform(
            torch.deg2rad(
                torch.tensor(
                    -angle_range,
                    device=self.device,
                    dtype=default_state.dtype,
                )
            ),
            torch.deg2rad(
                torch.tensor(
                    angle_range,
                    device=self.device,
                    dtype=default_state.dtype,
                )
            ),
            (num_envs,),
            device=self.device,
        )

        roll_rad = torch.zeros_like(yaw_rad)
        # pitch_rad = torch.zeros_like(yaw_rad)
        # yaw_rad = torch.zeros_like(yaw_rad)

        # ランダムなyaw + pitch姿勢
        q_random = quat_from_euler_xyz(
            roll_rad,
            pitch_rad,
            yaw_rad,
        )

        # World frameでランダム回転をdefault姿勢に追加
        default_state[:, 3:7] = quat_mul(
            q_random,
            init_rot,
        )

        # =========================================================
        # 3. Reset root velocity
        # =========================================================

        default_state[:, 7:] = 0.0

        # =========================================================
        # 4. Reset joints
        # =========================================================

        joint_pos = self.hand.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        self.hand.set_joint_position_target(
            joint_pos,
            env_ids=env_ids,
        )

        self.hand.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
        )

        self.hand.write_root_state_to_sim(
            default_state,
            env_ids=env_ids,
        )

        # Cache root pose actually written
        self.goal_hand_root_pos[env_ids] = default_state[:, 0:3].to(
            dtype=self.goal_hand_root_pos.dtype
        )

        self.goal_hand_root_quat[env_ids] = default_state[:, 3:7].to(
            dtype=self.goal_hand_root_quat.dtype
        )

        self._shadow_hand_finger_hold[env_ids] = (
            joint_pos[:, self.finger_joint_ids].clone()
        )
    
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
        # print(f"thumb_target: {self.thumb_target[0]} pinky_target: {self.pinky_target[0]}")

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

    def _thumb_upper_ee_proximity(
        self,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian proximity in [0, 1] between ``thumb_target`` and ``right_upper_ee_pos`` (env-local)."""
        dist = torch.norm(self.thumb_target[env_ids] - self.right_upper_ee_pos[env_ids], dim=-1)
        std = float(self.cfg.thumb_upper_ee_proximity_std)
        prox = torch.exp(-0.5 * (dist / std) ** 2)
        return prox, dist

    def _thumb_deep_inside_gate(
        self,
        ellipse_value: torch.Tensor,
    ) -> torch.Tensor:
        """Inside opening (``ev < 1``), peaked toward the center (``ev`` small), not on the rim (``ev ≈ 1``)."""
        margin = float(self.cfg.thumb_ellipse_inside_margin)
        sharpness = float(self.cfg.thumb_ellipse_inside_sharpness)
        inside = torch.sigmoid(30.0 * (1.0 - ellipse_value))
        core = torch.sigmoid(sharpness * (margin - ellipse_value))
        return inside * core

    def _thumb_reach_phase_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """1 during approach; 0 after wrist success (post-through), when configured."""
        if not self.cfg.thumb_gate_active_before_wrist_success:
            return torch.ones(env_ids.shape[0], device=self.device, dtype=torch.float32)
        return (
            self.wrist_center_euclidean_distance[env_ids] > self.cfg.bracelet_success_threshold
        ).float()

    def _ellipse_value_zy(
        self,
        env_ids: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized squared radius in opening Y–Z: ``1`` on rim, ``< 1`` inside."""
        eps = torch.as_tensor(1e-6, device=self.device, dtype=pos.dtype)
        radius_z = 0.5 * torch.abs(self.goal_north_pos[env_ids, 2] - self.goal_south_pos[env_ids, 2]).clamp_min(eps)
        radius_y = 0.5 * torch.abs(self.goal_east_pos[env_ids, 1] - self.goal_west_pos[env_ids, 1]).clamp_min(eps)
        return ((pos[:, 2] - self.goal_cent_pos[env_ids, 2]) / (radius_z + eps)) ** 2 + (
            (pos[:, 1] - self.goal_cent_pos[env_ids, 1]) / (radius_y + eps)
        ) ** 2

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
        sharpness: float = 5.0,
    ):
        pos_ev = pos[env_ids] if pos.shape[0] == self.num_envs else pos
        ellipse_value = self._ellipse_value_zy(env_ids, pos_ev)
        return torch.sigmoid(sharpness * (1.0 - ellipse_value))
    
    def _compute_intermediate_values(self, reset=False, env_ids: torch.Tensor | None = None):
        if self._is_free_space_mode():
            AIRECEnv._compute_intermediate_values(self, env_ids=env_ids)
            return

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

        # Visualize thumb and pinky targets (must index by env_ids: subset reset has |env_ids| < num_envs)
        # self.thumb_target_markers.visualize(
        #     self.thumb_target[env_ids] + self.scene.env_origins[env_ids],
        #     self.identity_quat[env_ids],
        # )
        # self.pinky_target_markers.visualize(
        #     self.pinky_target[env_ids] + self.scene.env_origins[env_ids],
        #     self.identity_quat[env_ids],
        # )
        # self.fore_target_markers.visualize(
        #     self.fore_goal_pos[env_ids] + self.scene.env_origins[env_ids],
        #     self.identity_quat[env_ids],
        # )
        # self.middle_target_markers.visualize(
        #     self.middle_goal_pos[env_ids] + self.scene.env_origins[env_ids],
        #     self.identity_quat[env_ids],
        # )
        # self.ring_target_markers.visualize(
        #     self.ring_goal_pos[env_ids] + self.scene.env_origins[env_ids],
        #     self.identity_quat[env_ids],
        # )
        # self.wrist_target_markers.visualize(
        #     self.goal_wrist_pos[env_ids] + self.scene.env_origins[env_ids],
        #     self.identity_quat[env_ids],
        # )
        # self.right_ee_markers.visualize(
        #     self.right_upper_ee_pos[env_ids] + self.scene.env_origins[env_ids],
        #     self.identity_quat[env_ids],
        # )
        # self.left_ee_markers.visualize(
        #     self.left_upper_ee_pos[env_ids] + self.scene.env_origins[env_ids],
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
            # print(f"goal_north_pos: {self.goal_north_pos[0]}, goal_south_pos: {self.goal_south_pos[0]}, goal_east_pos: {self.goal_east_pos[0]}, goal_west_pos: {self.goal_west_pos[0]}, goal_cent_pos: {self.goal_cent_pos[0]}")

        if self._use_glove or self.cfg.object_type == "rigid":
            self.garment_right_ee_distance[env_ids] = self.right_upper_ee_pos[env_ids] - self.goal_west_pos[env_ids]
            self.garment_right_ee_euclidean_distance[env_ids] = torch.norm(
                self.garment_right_ee_distance[env_ids], dim=1
            )
            self.garment_left_ee_distance[env_ids] = self.left_upper_ee_pos[env_ids] - self.goal_east_pos[env_ids]
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

        self.top_wrist_distance[env_ids] = self.north_edge_pos[env_ids] - self.goal_wrist_pos[env_ids]
        self.under_wrist_distance[env_ids] = self.goal_wrist_pos[env_ids] - self.south_edge_pos[env_ids]
        self.top_wrist_euclidean_distance[env_ids] = torch.norm(self.top_wrist_distance[env_ids], dim=1)
        self.under_wrist_euclidean_distance[env_ids] = torch.norm(self.under_wrist_distance[env_ids], dim=1)
        # print(f"east: {self.west_edge_pos[0]} thumb_goal_pos:{self.thumb_goal_pos[0]}")
        self.right_upper_ee_thumb_distance[env_ids] = self.right_upper_ee_pos[env_ids] - self.thumb_target[env_ids]
        self.right_upper_ee_thumb_euclidean_distance[env_ids] = torch.norm(self.right_upper_ee_thumb_distance[env_ids], dim=1)
        # print(f"right_ee_thumb_euclidean_distance: {self.right_ee_thumb_euclidean_distance[0]}")
        self.right_ee_thumb_rotation[env_ids] = quat_mul(self.right_ee_rot[env_ids], quat_conjugate(self.thumb_goal_rot[env_ids]))
        self.right_ee_thumb_angular_distance[env_ids] = rotation_distance(self.right_ee_rot[env_ids], self.thumb_goal_rot[env_ids])
        # print(f"right_ee_thumb_angular_distance: {self.right_ee_thumb_angular_distance[0]}")
        # self.left_ee_goal_distance[env_ids] = self.left_l_ee_pos[env_ids] - self.pinky_goal_pos[env_ids]
        self.left_upper_ee_pinky_distance[env_ids] = self.left_upper_ee_pos[env_ids] - self.pinky_target[env_ids]
        self.left_upper_ee_pinky_euclidean_distance[env_ids] = torch.norm(self.left_upper_ee_pinky_distance[env_ids], dim=1)
        self.left_ee_pinky_rotation[env_ids] = quat_mul(self.left_ee_rot[env_ids], quat_conjugate(self.pinky_goal_rot[env_ids]))
        self.left_ee_pinky_angular_distance[env_ids] = rotation_distance(self.left_ee_rot[env_ids], self.pinky_goal_rot[env_ids])

        # Scalar 3D distance from opening center to wrist (env-local base frame).
        self.wrist_center_distance[env_ids] = self.goal_wrist_pos[env_ids] - self.goal_cent_pos[env_ids]  # (B, 3)
        self.wrist_center_euclidean_distance[env_ids] = torch.norm(self.wrist_center_distance[env_ids], dim=1)  # (B,)

        # Reach-phase thumb gate: max when upper EE is near ``thumb_target`` AND thumb is deep inside the opening.
        thumb_prox, thumb_upper_dist = self._thumb_upper_ee_proximity(env_ids)
        self.thumb_target_upper_ee_distance[env_ids] = thumb_upper_dist
        self.thumb_upper_ee_proximity[env_ids] = thumb_prox
        self.thumb_ellipse_value[env_ids] = self._ellipse_value_zy(env_ids, self.thumb_target[env_ids])
        reach_active = self._thumb_reach_phase_mask(env_ids)
        self.thumb_reach_gate_active[env_ids] = reach_active
        deep_inside = self._thumb_deep_inside_gate(self.thumb_ellipse_value[env_ids])
        _thumb_side_gate = self._soft_side_gate_y(env_ids, self.thumb_target, -1.0)
        self.thumb_inside_ellipse[env_ids] = (
            deep_inside * _thumb_side_gate * thumb_prox * reach_active
        )
        # print(f"thumb_upper_distance: {thumb_upper_dist[0]}")
        # print(f"thumb_upper_ee_proximity: {thumb_prox[0]}")
        # print(f"thumb_ellipse_value: {self.thumb_ellipse_value[0]}")
        # print(f"deep_inside: {deep_inside[0]}")
        # print(f"thumb_side_gate: {_thumb_side_gate[0]}")
        # print(f"thumb_inside_ellipse: {self.thumb_inside_ellipse[0]}")

        _pinky_inside_ellipse = self._ellipse_inner_ring_gate_zy(env_ids, self.pinky_target)
        _pinky_side_gate = self._soft_side_gate_y(env_ids, self.pinky_target, 1.0)
        self.pinky_inside_ellipse[env_ids] = (_pinky_inside_ellipse * _pinky_side_gate)

        _wrist_inside_ellipse = self._ellipse_soft_gate_zy(env_ids, self.goal_wrist_pos)
        self.wrist_inside_ellipse[env_ids] = _wrist_inside_ellipse

        # print(f"left_ee_pinky_angular_distance: {self.left_ee_pinky_angular_distance[0]}")
        # print(f"left_ee_pinky_euclidean_distance: {self.left_ee_pinky_euclidean_distance[0]} right_ee_thumb_euclidean_distance: {self.right_ee_thumb_euclidean_distance[0]} wrist_center_euclidean_distance: {self.wrist_center_euclidean_distance[0]}")
        # shadow hand aperature
        self.goal_stretch_euclidean_distance[env_ids] = torch.abs(self.ee_euclidean_distance[env_ids] - self.human_stretch_euclidean_distance[env_ids])
        # Must index by env_ids: reset calls this on a subset (e.g. 31), not all num_envs.
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
        dist_from_south_pos = finger_heights - self.goal_south_pos[env_ids, 2].unsqueeze(-1)
        dist_from_north_pos = self.goal_north_pos[env_ids, 2].unsqueeze(-1) - finger_heights
        margin = torch.minimum(dist_from_south_pos, dist_from_north_pos)
        gate_k = float(getattr(self.cfg, "insertion_gate_temperature", 0.01))
        self.per_finger_height_z[env_ids] = finger_heights
        self.per_finger_insert_margin[env_ids] = margin
        # soft: g_i = sigmoid(m_i / k); hard: 1[m_i > 0] (same decision boundary at 0.5 / m_i = 0)
        soft_inside = torch.sigmoid(margin / max(gate_k, 1e-8))
        hard_inside = (margin > 0.0).to(dtype=soft_inside.dtype)
        self.per_finger_soft_inside[env_ids] = soft_inside
        self.per_finger_hard_inside[env_ids] = hard_inside
        self.fingers_inside_hard_gate[env_ids] = hard_inside.mean(dim=-1)
        gate_mode = str(getattr(self.cfg, "insertion_gate_mode", "soft")).lower()
        if gate_mode == "hard":
            self.fingers_inside_soft_gate[env_ids] = self.fingers_inside_hard_gate[env_ids]
        elif gate_mode == "soft":
            self.fingers_inside_soft_gate[env_ids] = soft_inside.mean(dim=-1)
        else:
            raise ValueError(
                f"Unknown insertion_gate_mode={gate_mode!r}; expected 'soft' or 'hard'."
            )
        # print(f"per_finger_soft_inside: {self.per_finger_soft_inside[0]}")
        # print(f"fingers_inside_soft_gate: {self.fingers_inside_soft_gate[0]}")
        # print(f"wrist_center_euclidean_distance: {self.wrist_center_euclidean_distance[0]}")

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
    reaching_right_ee_thumb_scale = 20.0
    reaching_left_ee_pinky_scale = 10.0
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
    reaching_wrist_center_scale = 200.0
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

