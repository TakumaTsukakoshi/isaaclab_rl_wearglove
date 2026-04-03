# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from assets.chain import CHAIN_CFG
from assets.airec import AIREC_CFG
from tasks.airec.airec import AIRECEnvCfg
from tasks.airec.object_manipulation import ObjectManipulationEnvCfg, ObjectManipulationEnv, GOAL_COLOUR
from assets.airec import (
    AIREC_CFG,
    ACTUATED_BASE_JOINTS,
    ACTUATED_TORSO_JOINTS,
    ACTUATED_HEAD_JOINTS,
    ACTUATED_LARM_JOINTS,
    ACTUATED_RARM_JOINTS,
    ACTUATED_LHAND_JOINTS,
    ACTUATED_RHAND_JOINTS,
    BASE_WHEELS,
)
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.math import sample_uniform, sample_gaussian
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import (
    ContactSensorCfg,
    ContactSensor,
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    TiledCameraCfg,
)
from isaaclab.envs import ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg

from tasks.airec.physics import bed_material, bed_rigid_props, contact_props, object_material

@configclass
class ChainEnvCfg(ObjectManipulationEnvCfg):
    """Configuration for the Chain environment.

    Defines simulation parameters, robot/chain configurations, frame transformers,
    and task-specific settings for the chain manipulation task.
    """

    episode_length_s = 5.0

    # get full reward here
    object_radius = 0.12 

    reward_r2o_scale: float = 1
    reward_r2o_std: float = 0.1
    reward_r2o_b: float = object_radius
    reward_o2g_scale: float = 10.0
    reward_o2g_std: float = 0.1

    # no offset here... I want as close as possible!
    reward_o2g_b: float = 0.0

    reward_action_scale: float = 0.0
    reward_r2o_vel_scale: float = 0.0
    reward_o2g_vel_scale: float = 0.0

    default_object_pos = (1.0, -0.4, 1.1)
    reset_object_pos_scale = 0.1
    reset_object_joint_deg = 10
    object_cfg: ArticulationCfg = CHAIN_CFG.replace(
        prim_path="/World/envs/env_.*/Object",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=default_object_pos,
            rot=(0.5, -0.5, -0.5, -0.5),
            joint_pos={
                "joint_1": 0.4, # limit is 0.3
                "joint_2": 0.4,
            },
        )
    )

    # N goal positions — kinematic, collision-disabled spheres are created automatically
    # Order: top_link, mid_link, bottom_link
    x_offset = 0.1
    z_offset = 1.0
    goal_positions: tuple[tuple[float, float, float], ...] = (
        (x_offset, 0.2, z_offset+0.1),
        (x_offset, 0.0, z_offset),
        (x_offset, -0.2, z_offset+0.1),
    )
    goal_sphere_radius: float = 0.01
    goal_sphere_colour: tuple[float, float, float] = (1.0, 0.2, 1.0)

    # Bed configuration
    bed_colour = (0.2, 0.2, 1.0)
    bed_height = 0.7
    bed_depth = 0.1
    bed_length = 2.0
    bed_width = 1
    base_link_to_bed_edge = 0.7
    bed_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bed",
        spawn=sim_utils.CuboidCfg(
            size=(bed_width, bed_length, bed_depth),
            physics_material=bed_material,
            rigid_props=bed_rigid_props,
            collision_props=contact_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=bed_colour),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(base_link_to_bed_edge + bed_width / 2, -0.2, bed_height + bed_depth / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # pillow
    pillow_radius = 0.15
    pillow_length = bed_width
    pillow_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/pillow",
        spawn=sim_utils.CylinderCfg(
            radius=pillow_radius,
            height=pillow_length,
            physics_material=bed_material,
            rigid_props=bed_rigid_props,
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=bed_colour),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(base_link_to_bed_edge + bed_width / 2, -0.4 + bed_length / 2, bed_height+bed_depth+pillow_radius),
            rot=(0.0, 0.707, 0.0, 0.707),
        ),
    )

    ## Visualisation markers
    object_frames_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/object_frames",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_COLOUR, opacity=1.0),
            ),
        },
    )
    goal_frames_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_frames",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_COLOUR, opacity=1.0),
            ),
        },
    )

    ## Frame transformers for object, goal
    radius = 0.18
    top_offset = [0, 0, radius]
    mid_offset = [0, 0, radius]
    base_offset = [0, 0, radius]

    object_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=object_frames_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Object/top_link",
                name="top_link",
                offset=OffsetCfg(pos=top_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Object/mid_link",
                name="mid_link",
                offset=OffsetCfg(pos=mid_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Object/bottom_link",
                name="bottom_link",
                offset=OffsetCfg(pos=base_offset),
            ),
        ],
    )

    goal_frames_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=goal_frames_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"/World/envs/env_.*/goal_{i}",
                name=f"goal_{i}",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            )
            for i in range(len(goal_positions))
        ],
    )

    # Contact sensor configuration for object contact detection
    object_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Object",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
    )


class ChainEnv(ObjectManipulationEnv):
    """Chain manipulation environment: robot reaches chain links, chain moves to goal pose."""

    cfg: ChainEnvCfg

    def __init__(self, cfg: ChainEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        o2g_pairs = [
            ("top_link", "goal_0"),
            ("mid_link", "goal_1"),
            ("bottom_link", "goal_2"),
        ]
        r2o_pairs = [
            ("left_hand", "mid_link"),
            ("right_hand", "mid_link"),
            ("left_elbow", "top_link"),
            ("right_elbow", "bottom_link"),
            ("torso", "mid_link"),
        ]
        object_names = self.object_frames.data.target_frame_names
        goal_names = self.goal_frames.data.target_frame_names
        robot_names = self.airec_frames.data.target_frame_names

        r2o_robot_idx = torch.tensor([robot_names.index(r) for r, _ in r2o_pairs], device=self.device)
        r2o_object_idx = torch.tensor([object_names.index(o) for _, o in r2o_pairs], device=self.device)
        o2g_object_idx = torch.tensor([object_names.index(o) for o, _ in o2g_pairs], device=self.device)
        o2g_goal_idx = torch.tensor([goal_names.index(g) for _, g in o2g_pairs], device=self.device)
        mid_link_id = object_names.index("mid_link")

        # for termination
        self.center_id = mid_link_id

        self._init_object_manipulation(r2o_robot_idx, r2o_object_idx, o2g_object_idx, o2g_goal_idx, object_center_vel_idx=mid_link_id)

        self.o2g_top_link_id = 0
        self.o2g_mid_link_id = 1
        self.o2g_bottom_link_id = 2
        self.r2o_left_hand_id = 0
        self.r2o_right_hand_id = 1
        self.r2o_left_elbow_id = 2
        self.r2o_right_elbow_id = 3
        self.r2o_torso_id = 4

        self.object_joint_pos = torch.zeros((self.num_envs, self.object.num_joints), dtype=self.dtype, device=self.device)
        self.object_joint_vel = torch.zeros((self.num_envs, self.object.num_joints), dtype=self.dtype, device=self.device)
        self.object_forces = torch.zeros((self.num_envs, self.object.num_bodies, 3), dtype=self.dtype, device=self.device)

        randomisable_positions_joints = ["joint_1", "joint_2"]
        self.randomisable_positions_joints_ids = [
            self.object.joint_names.index(name) for name in randomisable_positions_joints
        ]

        self.extras["log"] = {
            "lhand_goal_dist_reward": None,
            "rhand_goal_dist_reward": None,
            "object_goal_dist_reward": None,
            "lhand_goal_dist": None,
            "rhand_goal_distance": None,
            "object_goal_distance": None,
            "object_height_reward": None,
            "object_chest_reward": None,
            "joint_vel": None,
        }


    def _setup_scene(self):
        """Set up the simulation scene with chain, bed, and frame transformers."""
        super()._setup_scene()

       
        self.bed = RigidObject(self.cfg.bed_cfg)
        self.scene.rigid_objects["bed"] = self.bed

        self.pillow = RigidObject(self.cfg.pillow_cfg)
        self.scene.rigid_objects["pillow"] = self.pillow

        self.object = Articulation(self.cfg.object_cfg)
        self.scene.articulations["object"] = self.object

        self.object_frames = FrameTransformer(self.cfg.object_frame_cfg)
        self.object_frames.set_debug_vis(True)
        self.scene.sensors["object_frames"] = self.object_frames

        for i, pos in enumerate(self.cfg.goal_positions):
            goal_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/goal_{i}",
                spawn=sim_utils.SphereCfg(
                    radius=self.cfg.goal_sphere_radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=self.cfg.goal_sphere_colour, opacity=1.0
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            goal_obj = RigidObject(goal_cfg)
            self.scene.rigid_objects[f"goal_{i}"] = goal_obj
        self.goal_frames = FrameTransformer(self.cfg.goal_frames_config)
        self.goal_frames.set_debug_vis(True)
        self.scene.sensors["goal_frames"] = self.goal_frames

        light = sim_utils.DomeLightCfg(
                color=(1.0, 1.0, 1.0),
                intensity=1000.0, 
                texture_file="/home/elle/code/debug/airec_rl/assets/stierberg_sunrise_4k.hdr",
                texture_format="latlong"
            )
        light.func("/World/bglight", light)

    def _get_gt(self):
        return self._get_gt_object_manipulation()
    
    def _get_rewards(self) -> torch.Tensor:
        base = super()._get_rewards()
        self.extras["log"].update({
            "lhand_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.r2o_left_hand_id]),
            "rhand_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.r2o_right_hand_id]),
            "lelbow_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.r2o_left_elbow_id]),
            "relbow_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.r2o_right_elbow_id]),
            "torso_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.r2o_torso_id]),
            "o2g_top_link_dist": (self.object2goal_frames_euclidean_distance[:, self.o2g_top_link_id]),
            "o2g_mid_link_dist": (self.object2goal_frames_euclidean_distance[:, self.o2g_mid_link_id]),
            "o2g_bottom_link_dist": (self.object2goal_frames_euclidean_distance[:, self.o2g_bottom_link_id]),
            "joint_vel": (torch.abs(self.normalised_joint_vel)),
            "action_diff": (self.action_diff),
            "robot2object_vel": (torch.mean(torch.norm(self.robot2object_frames_vel, dim=-1), dim=-1)),
        })
        stacked_rewards = torch.stack([base[:, 0], base[:, 1], base[:, 3]], dim=-1)
        return stacked_rewards

    def _reset_object(self, env_ids):
        """Reset chain pose and joint properties for given environments.

        Args:
            env_ids: Environment indices to reset.
        """
        joint_pos_reset_noise = np.deg2rad(self.cfg.reset_object_joint_deg)
        default_joint_pos = self.object.data.default_joint_pos[env_ids][:, self.randomisable_positions_joints_ids]

        joint_pos = default_joint_pos + sample_uniform(
            -joint_pos_reset_noise,
            joint_pos_reset_noise,
            (len(env_ids), len(self.randomisable_positions_joints_ids)),
            self.device,
        )
        joint_vel = torch.zeros_like(joint_pos)
        self.object.set_joint_position_target(joint_pos, joint_ids=self.randomisable_positions_joints_ids, env_ids=env_ids)
        self.object.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self.randomisable_positions_joints_ids, env_ids=env_ids)

        # Reset object root position with noise
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device) * self.cfg.reset_object_pos_scale
        pos_noise[:, 2] = 0
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3]
            + self.scene.env_origins[env_ids] + pos_noise
        )
        self.object.write_root_state_to_sim(object_default_state, env_ids)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._compute_object_manipulation_values(env_ids)

        self.object_joint_pos[env_ids] = self.object.data.joint_pos[env_ids]
        self.object_joint_vel[env_ids] = self.object.data.joint_vel[env_ids]
