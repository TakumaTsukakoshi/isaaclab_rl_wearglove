# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab_rl_wearglove.tasks.airec.airec import AIRECEnvCfg
from tasks.airec.object_manipulation import ObjectManipulationEnvCfg, ObjectManipulationEnv
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.envs import ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg

from tasks.airec.physics import contact_props, object_material, bed_material, object_rigid_props, bed_rigid_props, sphere_mass_props


@configclass
class BlockEnvCfg(ObjectManipulationEnvCfg):
    """Configuration for the Block environment (rigid sphere to goal).

    Imitation demo episode starts (parent :class:`ObjectManipulationEnv`) apply only to **training**
    env indices ``>= num_eval_envs`` (``trainer.num_eval_envs`` → :attr:`AIRECEnvCfg.num_eval_envs`).
    Eval envs ``0 … num_eval_envs-1`` always use the normal bed/object randomized reset. Demo-start
    joint/object noise uses the same scales as :meth:`AIRECEnv._reset_robot` / ``object_noise`` when
    ``imitation_demo_start_joint_noise_scale`` is 0.
    """

    episode_length_s = 10.0

    # Bed configuration
    bed_colour = (0.2, 0.2, 1.0)
    bed_height = 0.7
    bed_depth = 0.05
    bed_length = 1.0
    bed_width = 0.6
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
            pos=(base_link_to_bed_edge + bed_width / 2, 0, bed_height),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Instantiate object and goal as RigidObjects
    object_init_pos = (base_link_to_bed_edge + bed_width / 2, 0, 0.75)
    object_radius = 0.2
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=object_radius,
            physics_material=object_material,
            rigid_props=object_rigid_props, 
            collision_props=contact_props,
            visual_material=sim_utils.PreviewSurfaceCfg(opacity=1.0, diffuse_color=(1.0, 0.6, 1.0)),
            mass_props=sphere_mass_props,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=object_init_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    chair_height = 0.6
    chair_pos = (-1, 0.0, chair_height/2)
    chair_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/chair",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, chair_height),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=3.0, dynamic_friction=3.0, restitution=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 1.0), opacity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=chair_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ## Visualisation markers
    object_frames_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/object_frames",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 1.0), opacity=1.0),
            ),
        },
    )
    lift_goal_frames_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/lift_goal_frames",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=object_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0, 1.0, 0), opacity=0.5),
            ),
        },
    )
    goal_frames_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_frames",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=object_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0, 1.0, 0), opacity=0.5),
            ),
        },
    )

    ## Frame transformers for object, goal
    object_frames_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=object_frames_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/object",
                name="center",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ],
    )
    lift_goal_frames_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=lift_goal_frames_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/base_link",
                name="lift_goal",
                offset=OffsetCfg(pos=[0.2, 0.0, 1.05]),
            ),
        ],
    )
    goal_frames_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=goal_frames_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/chair",
                name="goal",
                offset=OffsetCfg(pos=[0.0, 0.0, chair_height/2+object_radius]),
            ),
        ],
    )



class BlockEnv(ObjectManipulationEnv):
    """Block manipulation: robot reaches object (sphere), object moves to goal."""

    cfg: BlockEnvCfg

    def __init__(self, cfg: BlockEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # r2o: 5 robot parts → 1 object (center). o2g: 1 object → 1 goal.
        robot_names = self.airec_frames.data.target_frame_names
        object_names = self.object_frames.data.target_frame_names
        goal_names = self.goal_frames.data.target_frame_names

        r2o_pairs = [
            ("left_upperarm", "center"),
            ("right_upperarm", "center"),
            ("left_hand", "center"),
            ("right_hand", "center"),
            ("left_elbow", "center"),
            ("right_elbow", "center"),
            ("torso", "center"),
            ("base", "center"),
        ]
        r2o_robot_idx = torch.tensor([robot_names.index(r) for r, _ in r2o_pairs], device=self.device)
        r2o_object_idx = torch.tensor([object_names.index(o) for _, o in r2o_pairs], device=self.device)
        o2g_object_idx = torch.tensor([object_names.index("center")], device=self.device)
        o2g_goal_idx = torch.tensor([goal_names.index("goal")], device=self.device)

        self._init_object_manipulation(r2o_robot_idx, r2o_object_idx, o2g_object_idx, o2g_goal_idx, object_center_vel_idx=0)

        self.center_id = 0
        self.left_hand_id = 0
        self.right_hand_id = 1
        self.left_elbow_id = 2
        self.right_elbow_id = 3
        self.torso_id = 4
        self.base_id = 5
        self.left_upperarm_id = 6
        self.right_upperarm_id = 7
        
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
        super()._setup_scene()

        self.bed = RigidObject(self.cfg.bed_cfg)
        self.scene.rigid_objects["bed"] = self.bed

        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

        self.chair = RigidObject(self.cfg.chair_cfg)
        self.scene.rigid_objects["chair"] = self.chair

        self.object_frames = FrameTransformer(self.cfg.object_frames_config)
        self.object_frames.set_debug_vis(True)
        self.scene.sensors["object_frames"] = self.object_frames

        self.lift_goal_frames = FrameTransformer(self.cfg.lift_goal_frames_config)
        self.lift_goal_frames.set_debug_vis(True)
        self.scene.sensors["lift_goal_frames"] = self.lift_goal_frames

        self.goal_frames = FrameTransformer(self.cfg.goal_frames_config)
        self.goal_frames.set_debug_vis(True)
        self.scene.sensors["goal_frames"] = self.goal_frames

        light = sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=250.0,
            texture_format="latlong"
        )
        light.func("/World/bglight", light)

    def _get_gt(self):
        return self._get_gt_object_manipulation()

    def _get_rewards(self) -> torch.Tensor:
        base = super()._get_rewards()
        self.extras["log"].update({
            "lhand_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.left_hand_id]),
            "rhand_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.right_hand_id]),
            "lelbow_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.left_elbow_id]),
            "relbow_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.right_elbow_id]),
            "torso_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.torso_id]),
            "base_goal_dist": (self.robot2object_frames_euclidean_distance[:, self.base_id]),
            "object_goal_dist": (self.object2goal_frames_euclidean_distance[:, 0]),
            "joint_vel": (torch.abs(self.normalised_joint_vel)),
            "action_diff": (self.action_diff),
            "robot2object_vel": (torch.mean(torch.norm(self.robot2object_frames_vel, dim=-1), dim=-1)),
        })
        summed_rewards = torch.sum(base, dim=-1)
        return summed_rewards
        return torch.stack([base[:, 0], base[:, 1], base[:, 2]], dim=-1)

    def _reset_object(self, env_ids):
        self._reset_bed_object_goal(env_ids)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._compute_object_manipulation_values(env_ids)
        self.success[env_ids] = self.object2goal_frames_euclidean_distance[env_ids, 0] < 0.03