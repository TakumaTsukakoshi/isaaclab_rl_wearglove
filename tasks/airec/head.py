# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from assets.human import HUMAN_CFG
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

from tasks.airec.physics import contact_props, bed_material, object_material, bed_rigid_props

@configclass
class HeadEnvCfg(ObjectManipulationEnvCfg):
    """Configuration for the Head environment.
    
    Defines simulation parameters, robot/human configurations, frame transformers,
    and task-specific settings for the human assistance task.
    """

    episode_length_s = 5.0

    reward_r2o_scale: float = 1
    reward_r2o_std: float = 0.1
    reward_r2o_b: float = 0.02
    reward_o2g_scale: float = 10.0
    reward_o2g_std: float = 0.1
    reward_action_scale: float = 0.0
    reward_r2o_vel_scale: float = 0.0
    reward_o2g_vel_scale: float = 0.0

    default_human_pos = (0.5, -0.27, 1.2)
    reset_human_pos_scale = 0.1
    reset_human_joint_deg = 10
    human_cfg: ArticulationCfg = HUMAN_CFG.replace(
        prim_path="/World/envs/env_.*/Human",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=default_human_pos,
            rot=(0.5, -0.5, -0.5, -0.5),
            joint_pos={
                "okiagari_joint": 0.0, # limit is 0.3
                ".*hip.*": 0.5,
                ".*knee.*": -1.2,
                "right_shoulder_pitch_joint": 1.57,
                "right_shoulder_roll_joint": -1.3,
                "right_elbow_joint": 0,
                "left_shoulder_pitch_joint": -1.57,
                "left_shoulder_roll_joint": 1.3,
                "left_elbow_joint": 0.0,
            },
        )
    )

    # N goal positions — kinematic, collision-disabled spheres are created automatically
    # Order: 
    x_offset = -0.05
    z_offset = 1.1
    goal_positions: tuple[tuple[float, float, float], ...] = (
        (0.4801 + x_offset, -0.4685, z_offset),
        (0.2904 + x_offset, -0.4685, z_offset),
        (0.3363 + x_offset, 0., z_offset),
        (0.4863 + x_offset, 0.2993, z_offset),
        (0.1863 + x_offset, 0.2993, z_offset),
    )
    goal_sphere_radius: float = 0.01
    goal_sphere_colour: tuple[float, float, float] = (1.0, 0.2, 1.0)

    # Bed configuration
    bed_colour = (0.2, 0.2, 1.0)
    bed_height = 0.45
    bed_depth = 0.25
    bed_length = 2.0
    bed_width = 1
    base_link_to_bed_edge = 0.1
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=bed_colour),
            collision_props=contact_props,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(base_link_to_bed_edge + bed_width / 2, -0.4 + bed_length / 2, bed_height+bed_depth+pillow_radius),rot=(0.0, 0.707, 0.0, 0.707),
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
    human_left_shoulder_offset = [0, -0.15, -0.2]
    human_right_shoulder_offset = [0, 0.15, -0.2]
    human_left_thigh_offset = [-0.05, -0.05, -0.25]
    human_right_thigh_offset = [-0.05, -0.05, -0.25]
    human_base_offset = [0.03, 0, 0]

    object_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=object_frames_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Human/left_thigh",
                name="human_left_thigh",
                offset=OffsetCfg(pos=human_left_thigh_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Human/right_thigh",
                name="human_right_thigh",
                offset=OffsetCfg(pos=human_right_thigh_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Human/chest_joint",
                name="human_base",
                offset=OffsetCfg(pos=human_base_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Human/chest",
                name="human_left_shoulder",
                offset=OffsetCfg(pos=human_left_shoulder_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Human/chest",
                name="human_right_shoulder",
                offset=OffsetCfg(pos=human_right_shoulder_offset),
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

    # Contact sensor configuration for hand contact detection
    human_contact_sensor_cfg = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Human",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
        # visualizer_cfg=marker_cfg,
    )


class HeadEnv(ObjectManipulationEnv):
    """Human assistance environment: robot reaches human body parts, human moves to goal pose."""

    cfg: HeadEnvCfg

    def __init__(self, cfg: HeadEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        o2g_pairs = [
            ("human_left_thigh", "goal_0"),
            ("human_right_thigh", "goal_1"),
            ("human_base", "goal_2"),
            ("human_left_shoulder", "goal_3"),
            ("human_right_shoulder", "goal_4"),
        ]
        r2o_pairs = [
            ("left_hand", "human_left_shoulder"),
            ("right_hand", "human_left_thigh"),
            ("left_elbow", "human_right_shoulder"),
            ("right_elbow", "human_right_thigh"),
            ("torso", "human_base"),
        ]
        object_names = self.object_frames.data.target_frame_names
        goal_names = self.goal_frames.data.target_frame_names
        robot_names = self.airec_frames.data.target_frame_names

        r2o_robot_idx = torch.tensor([robot_names.index(r) for r, _ in r2o_pairs], device=self.device)
        r2o_object_idx = torch.tensor([object_names.index(o) for _, o in r2o_pairs], device=self.device)
        o2g_object_idx = torch.tensor([object_names.index(o) for o, _ in o2g_pairs], device=self.device)
        o2g_goal_idx = torch.tensor([goal_names.index(g) for _, g in o2g_pairs], device=self.device)
        human_base_id = object_names.index("human_base")

        self._init_object_manipulation(r2o_robot_idx, r2o_object_idx, o2g_object_idx, o2g_goal_idx, object_center_vel_idx=human_base_id)

        self.o2g_left_thigh_id = 0
        self.o2g_right_thigh_id = 1
        self.o2g_human_base_id = 2
        self.o2g_left_shoulder_id = 3
        self.o2g_right_shoulder_id = 4
        self.r2o_left_hand_id = 0
        self.r2o_right_hand_id = 1
        self.r2o_left_elbow_id = 2
        self.r2o_right_elbow_id = 3
        self.r2o_torso_id = 4

        self.object_joint_pos = torch.zeros((self.num_envs, self.object.num_joints), dtype=self.dtype, device=self.device)
        self.object_joint_vel = torch.zeros((self.num_envs, self.object.num_joints), dtype=self.dtype, device=self.device)
        self.object_forces = torch.zeros((self.num_envs, self.object.num_bodies, 3), dtype=self.dtype, device=self.device)

        self.human_base_id = object_names.index("human_base")
        self.default_chest_quat = torch.zeros((self.num_envs, 4), dtype=self.dtype, device=self.device)
        self.human_left_thigh_id = object_names.index("human_left_thigh")
        self.human_right_thigh_id = object_names.index("human_right_thigh")

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

        # Joints for randomization
        randomisable_dynamics_joints = ['okiagari_joint']
        randomisable_positions_joints = [
            'okiagari_joint',
            'left_hip_joint',
            'right_hip_joint',
            'left_knee_joint',
            'right_knee_joint',
            'left_elbow_joint',
            'right_elbow_joint',
            'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'right_shoulder_roll_joint',
        ]

        self.randomisable_dynamics_joints_ids = [
            self.object.joint_names.index(name) for name in randomisable_dynamics_joints
        ]
        self.randomisable_positions_joints_ids = [
            self.object.joint_names.index(name) for name in randomisable_positions_joints
        ]


    def _setup_scene(self):
        """Set up the simulation scene with human, bed, and frame transformers."""
        super()._setup_scene()

       
        self.bed = RigidObject(self.cfg.bed_cfg)
        self.scene.rigid_objects["bed"] = self.bed

        self.pillow = RigidObject(self.cfg.pillow_cfg)
        self.scene.rigid_objects["pillow"] = self.pillow

        self.object = Articulation(self.cfg.human_cfg)
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
                intensity=400.0, 
                texture_file="/home/elle/code/debug/airec_rl/assets/hospital_room_2_4k.hdr",
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
            "o2g_left_thigh_dist": (self.object2goal_frames_euclidean_distance[:, self.o2g_left_thigh_id]),
            "o2g_right_thigh_dist": (self.object2goal_frames_euclidean_distance[:, self.o2g_right_thigh_id]),
            "o2g_base_dist": (self.object2goal_frames_euclidean_distance[:, self.o2g_human_base_id]),
            "o2g_left_shoulder_dist": (self.object2goal_frames_euclidean_distance[:, self.o2g_left_shoulder_id]),
            "o2g_right_shoulder_dist": (self.object2goal_frames_euclidean_distance[:, self.o2g_right_shoulder_id]),
            "joint_vel": (torch.abs(self.normalised_joint_vel)),
            "action_diff": (self.action_diff),
            "robot2object_vel": (torch.mean(torch.norm(self.robot2object_frames_vel, dim=-1), dim=-1)),
        })
        stacked_rewards = torch.stack([base[:, 0], base[:, 1], base[:, 3]], dim=-1)
        return stacked_rewards

    def _reset_object(self, env_ids):
        """Reset human pose and joint properties for given environments.
        
        Args:
            env_ids: Environment indices to reset.
        """

        joint_pos_reset_noise = np.deg2rad(self.cfg.reset_human_joint_deg)
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

        min_stiffness=0.1
        max_stiffness=10
        min_damping=0.1
        max_damping=10
        min_joint_friction=0.001
        max_joint_friction=0.01

        randomised_stiffness = sample_uniform(
            min_stiffness, max_stiffness, (len(env_ids), len(self.randomisable_dynamics_joints_ids)), self.device
        )
        randomised_damping = sample_uniform(
            min_damping, max_damping, (len(env_ids), len(self.randomisable_dynamics_joints_ids)), self.device
        )
        randomised_joint_friction = sample_uniform(
            min_joint_friction, max_joint_friction, (len(env_ids), len(self.randomisable_dynamics_joints_ids)), self.device
        )

        # Reset human root position with noise
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device) * self.cfg.reset_human_pos_scale
        pos_noise[:, 2] = 0
        human_default_state = self.object.data.default_root_state.clone()[env_ids]
        human_default_state[:, 0:3] = (
            human_default_state[:, 0:3]
            + self.scene.env_origins[env_ids] + pos_noise
        )
        self.object.write_root_state_to_sim(human_default_state, env_ids)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._compute_object_manipulation_values(env_ids)

        self.object_joint_pos[env_ids] = self.object.data.joint_pos[env_ids]
        self.object_joint_vel[env_ids] = self.object.data.joint_vel[env_ids]

        # Snapshot chest quaternion once sensor data is fresh after reset.
        just_reset = (self.episode_length_buf[env_ids] == 2)
        if just_reset.any():
            snap_ids = env_ids[just_reset] if env_ids.shape[0] != self.num_envs else just_reset.nonzero(as_tuple=False).squeeze(-1)
            self.default_chest_quat[snap_ids] = self.object_frames_rot[snap_ids, self.human_base_id]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation conditions.
        
        Returns:
            Tuple of (termination, truncation) tensors.
        """
        self._compute_intermediate_values()

        min_timesteps = 25
        max_xy_vel = 0.5

        # human chest must stay close to the robot
        max_norm = 2
        out_of_reach = torch.norm(self.object_frames_pos[:, self.human_base_id, :], dim=1) >= max_norm

        x_vel = abs(self.object_frames_vel[:,self.human_base_id,0]) * (self.episode_length_buf > min_timesteps)
        y_vel = abs(self.object_frames_vel[:,self.human_base_id,1]) * (self.episode_length_buf > min_timesteps)
        too_violent = (x_vel > max_xy_vel) | (y_vel > max_xy_vel)

        # Terminate if chest quaternion deviates too far from its post-reset orientation.
        # |q · q_default| < cos(max_angle/2)  means the rotation exceeds max_angle.
        max_chest_angle_deg = 80.0
        cos_half = math.cos(math.radians(max_chest_angle_deg) / 2.0)
        chest_quat = self.object_frames_rot[:, self.human_base_id]          # (num_envs, 4)
        quat_dot = torch.abs(torch.sum(chest_quat * self.default_chest_quat, dim=-1))  # (num_envs,)
        chest_deviated = (quat_dot < cos_half) & (self.episode_length_buf > min_timesteps)

        termination = out_of_reach

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out