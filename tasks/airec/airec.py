# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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
    IMITATION_DEFAULT_JOINT_NAMES,
)
# JOINT VELOCITY OVERRIDE SCALE
from assets.airec import OVERRIDE_SCALE

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    TiledCameraCfg,
)
from isaaclab.utils.math import sample_uniform
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    matrix_from_quat
)
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from roto.tasks.roto_env import RotoEnvCfg, RotoEnv

from tasks.airec.physics import PHYSICS_DT, airec_sim_cfg


@configclass
class AIRECEnvCfg(RotoEnvCfg):
    physics_dt = PHYSICS_DT

    # simulation
    sim: airec_sim_cfg

    robot_cfg = AIREC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    reset_robot_joint_noise_scale = 0.05 # 0.2 radians is 11 degrees
    reset_robot_pos_noise_x: float = 0.01  # max ± in x for robot base position at reset
    reset_robot_pos_noise_y: float = 0.01  # max ± in y for robot base position at reset
    reset_robot_quat_noise_scale = 0 # np.pi/40  # max yaw range (radians); 2*pi = full 360°

    # Visual mat showing robot reset randomization region (±pos_noise in x,y)
    mat_size_x = 2 * reset_robot_pos_noise_x
    mat_size_y = 2 * reset_robot_pos_noise_y
    mat_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/reset_mat",
        spawn=sim_utils.CuboidCfg(
            size=(mat_size_x, mat_size_y, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.8, 0.2),
                opacity=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0025),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    enable_base_control = False

    ACTUATED_ARM_JOINTS_INTERLEAVED = [
        j for pair in zip(ACTUATED_LARM_JOINTS, ACTUATED_RARM_JOINTS) for j in pair
    ]
    ACTUATED_HAND_JOINTS_INTERLEAVED = [
        j for pair in zip(ACTUATED_LHAND_JOINTS, ACTUATED_RHAND_JOINTS) for j in pair
    ]
    actuated_body_joint_names = ACTUATED_TORSO_JOINTS + ACTUATED_ARM_JOINTS_INTERLEAVED + ACTUATED_HAND_JOINTS_INTERLEAVED

    # All available joints: base (trans_x, trans_y, yaw) + torso + arms + hands. Head excluded; held at default.
    actuated_body_joint_names = ACTUATED_TORSO_JOINTS + ACTUATED_LARM_JOINTS + ACTUATED_RARM_JOINTS + ACTUATED_LHAND_JOINTS + ACTUATED_RHAND_JOINTS
    if enable_base_control:
        actuated_joint_names = actuated_body_joint_names + ACTUATED_BASE_JOINTS
    else:   
        actuated_joint_names = actuated_body_joint_names

    num_actions = len(actuated_joint_names)
    # isaac 4.5 stuff
    action_space = 0
    observation_space = 0
    state_space = 0
    
    # temp
    replicate_physics = False
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5, replicate_physics=replicate_physics)
    
    eye = (2,2, 2)
    lookat = (0.0, 0, 1)

    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    # Marker for Center of Mass (a small red sphere)
    com_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/com_marker",
        markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0,0,1.0), opacity=1.0),
                ),
            },
    )

    gravity_marker_cfg = FRAME_MARKER_CFG.copy()
    gravity_marker_cfg.markers["frame"].scale = (0.3, 0.3, 0.3)
    gravity_marker_cfg.prim_path = "/Visuals/GravityMarker"

    # End-effector frame transformer configuration
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"
    airec_left_hand_offset = [0.0, 0.0, 0.0]  
    airec_right_hand_offset = [0.0, 0.0, 0.02]
    airec_left_elbow_offset = [0.03, -0.2, 0.02]
    airec_right_elbow_offset = [0.03, -0.2, 0.02]
    airec_torso_offset = [0.12, 0.0, 0.35]
    airec_base_offset = [0, 0, 0.95]
    airec_left_upperarm_offset = [0.05, 0, 0.13]
    airec_right_upperarm_offset = [0.05, 0, -0.13]

    # airec_left_hand_offset = [0.0, 0.0, 0.00]  
    # airec_right_hand_offset = [0.0, 0.0, 0.0]
    # airec_left_elbow_offset = [0.0, 0.0, 0.0]
    # airec_right_elbow_offset = [0.0, 0.0, 0.0]
    # airec_torso_offset = [0.0, 0.0, 0.0]
    airec_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_arm_link_3",
                name="right_upperarm",
                offset=OffsetCfg(pos=airec_right_upperarm_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_arm_link_3",
                name="left_upperarm",
                offset=OffsetCfg(pos=airec_left_upperarm_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_arm_link_4",
                name="right_elbow",
                offset=OffsetCfg(pos=airec_right_elbow_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_arm_link_4",
                name="left_elbow",
                offset=OffsetCfg(pos=airec_left_elbow_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/torso_link_3",
                name="torso",
                offset=OffsetCfg(pos=airec_torso_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/base_link",
                name="base",
                offset=OffsetCfg(pos=airec_base_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_hand_second_finger_link_1",
                name="left_hand",
                offset=OffsetCfg(pos=airec_left_hand_offset),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_hand_second_finger_link_1",
                name="right_hand",
                offset=OffsetCfg(pos=airec_right_hand_offset),
            ),
        ],
    )

    # Contact sensor marker configuration
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/ContactCfg"

    # Contact sensor configuration for hand contact detection
    robot_contact_sensor_cfg = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/(.*_hand_.*first_finger_link_2|.*_hand_.*third_finger_link_2|.*_hand_.*second_finger_link_2|.*_hand_base_link|.*_hand_thumb_link_4)",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
    )

    # camera
    img_dim = 80
    eye = (0.0, -0.6, 0.65)
    target = (0.0, 0, 0.5)
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/head_sr300_camera_link/tiled_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=img_dim,
        height=img_dim,
    )

    render_cfg = sim_utils.RenderCfg(rendering_mode="quality")

    #: First ``num_eval_envs`` parallel envs are eval (see ``IsaacLabWrapper.eval_env_ids`` = ``arange``).
    #: Imitation demo pose resets skip those indices so evaluation always uses standard randomization.
    num_eval_envs: int = 0

    # Optional joint imitation from teleop demo (see ``teleop_demonstrations`` for ``.npz`` layout).
    imitation_demo_path: str | None = None
    #: Width (in joint-space MSE units) for ``exp(-mse / std²)`` shaping.
    imitation_reward_std: float = 0.5
    imitation_reward_scale: float = 1.0
    #: Subset of joint indices for imitation MSE. ``None`` = base, torso, and arms only (see
    #: :data:`assets.airec.IMITATION_DEFAULT_JOINT_NAMES`). Set a tuple of DOF indices to override
    #: (e.g. to include head/hands, match all columns with ``tuple(range(num_joints))`` in code).
    imitation_joint_indices: tuple[int, ...] | None = None
    #: If ``gt`` is in ``obs_list``, stack this many consecutive demo ``joint_commands`` rows
    #: ``t, t+1, …`` (clamped) into one vector of size ``imitation_gt_future_steps * num_joints``.
    imitation_gt_future_steps: int = 1


def imitation_joint_position_mse(
    joint_pos: torch.Tensor,
    reference_joint_pos: torch.Tensor,
    joint_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error between joint poses per environment.

    Args:
        joint_pos: (num_envs, num_joints)
        reference_joint_pos: (num_envs, num_joints)
        joint_indices: If set, MSE is over these columns only (same indices applied to both tensors).
    """
    if joint_indices is not None:
        if joint_indices.numel() == 0:
            return torch.zeros(joint_pos.shape[0], device=joint_pos.device, dtype=joint_pos.dtype)
        diff = joint_pos[:, joint_indices] - reference_joint_pos[:, joint_indices]
    else:
        diff = joint_pos - reference_joint_pos
    return (diff * diff).mean(dim=-1)


def imitation_joint_tracking_reward_exp(
    joint_pos: torch.Tensor,
    reference_joint_pos: torch.Tensor,
    joint_indices: torch.Tensor | None,
    mse_std: float,
    scale: float,
) -> torch.Tensor:
    """Dense imitation reward ``scale * exp(-mse / (mse_std² + eps))`` per environment."""
    err = imitation_joint_position_mse(joint_pos, reference_joint_pos, joint_indices)
    denom = mse_std * mse_std + 1e-8
    return scale * torch.exp(-err / denom)


class AIRECEnv(RotoEnv):
    """Base AIREC robot environment for RL tasks.
    
    Inherits from RotoEnv and provides AIREC-specific functionality including
    end-effector tracking for both hands and contact sensing.
    """
    cfg: AIRECEnvCfg

    def __init__(self, cfg: AIRECEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # Joint limits and targets
        # this is bad - only actuated joints 
        self.num_joints = self.robot.num_joints
        self.torso_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in ACTUATED_TORSO_JOINTS
            if joint_name in self.robot.joint_names
        ]
        self.torso_dof_indices.sort()
        self.larm_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in ACTUATED_LARM_JOINTS
            if joint_name in self.robot.joint_names
        ]
        self.larm_dof_indices.sort()
        self.rarm_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in ACTUATED_RARM_JOINTS
            if joint_name in self.robot.joint_names
        ]
        self.rarm_dof_indices.sort()
        self.hand_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in ACTUATED_LHAND_JOINTS + ACTUATED_RHAND_JOINTS
            if joint_name in self.robot.joint_names
        ]
        self.hand_dof_indices.sort()
        self.base_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in ACTUATED_BASE_JOINTS
            if joint_name in self.robot.joint_names
        ]
        self.base_dof_indices.sort()

        # Indices of actuated joints (body only: torso + arms)
        self.actuated_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in cfg.actuated_joint_names
        ]
        self.actuated_dof_indices.sort()

        self.actuated_body_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in cfg.actuated_body_joint_names
        ]
        self.actuated_body_dof_indices.sort()


        # base
        self.actuated_base_dof_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in ACTUATED_BASE_JOINTS
            if joint_name in self.robot.joint_names
        ]
        self.actuated_base_dof_indices.sort()
        # torque sensing only on the arms and torso
        self.torque_sensing_indices = [
            self.robot.joint_names.index(joint_name) for joint_name in ACTUATED_TORSO_JOINTS + ACTUATED_LARM_JOINTS + ACTUATED_RARM_JOINTS
        ]
        self.torque_sensing_indices.sort()
        num_torque_sensing_indices = len(self.torque_sensing_indices)

        self.joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_acc = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_pos_error = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.normalised_joint_applied_torque = torch.zeros((self.num_envs, num_torque_sensing_indices), device=self.device)
        self.normalised_joint_computed_torque = torch.zeros((self.num_envs, num_torque_sensing_indices), device=self.device)
        self.torque_error = torch.zeros((self.num_envs, num_torque_sensing_indices), device=self.device)

        self.normalised_joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.normalised_joint_vel = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.base_cmd = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)

        self.robot_joint_pos_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_joint_pos_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_hard_vel_limits = self.robot.data.joint_vel_limits[0, :].to(device=self.device)
        self.robot_effort_limits = self.robot.data.joint_effort_limits[0, self.torque_sensing_indices].to(device=self.device)

        print("****************************")
        print("Robot has ", self.robot.num_joints, " joints")
        # Joint pos command contains the full shebang of 47 joints, even though a subset are actuated
        # This is better for physics
        self.joint_pos_cmd = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.prev_joint_pos_cmd = torch.zeros((self.num_envs, self.num_joints), device=self.device)

        if self.cfg.enable_base_control:
            workspace = 3.0 # meters
            new_limits = torch.tensor([workspace, workspace, workspace], device=self.device)
            self.robot_joint_pos_lower_limits[self.actuated_base_dof_indices] = -new_limits
            self.robot_joint_pos_upper_limits[self.actuated_base_dof_indices] = new_limits
        
        print("Number of actuated DOF indices: ", len(self.actuated_dof_indices))
        print("Actuated DOF indices: ", self.actuated_dof_indices)
        print("Torso DOF indices: ", self.torso_dof_indices)
        print("Larm DOF indices: ", self.larm_dof_indices)
        print("Rarm DOF indices: ", self.rarm_dof_indices)
        print("Hand DOF indices: ", self.hand_dof_indices)
        print("Body DOF indices: ", self.actuated_body_dof_indices)
        print("Base DOF indices: ", self.actuated_base_dof_indices)

        # Actions are the (-1, 1) normalized joint position commands
        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)

        print("Number of actions: ", self.cfg.num_actions)

        # Joint position commands are what are sent
        default_joint_pos = self.robot.data.default_joint_pos
        self.joint_pos_cmd[:, :] = default_joint_pos[:, :]
        self.prev_joint_pos_cmd[:] = self.joint_pos_cmd
        print("Default joint positions: ", default_joint_pos[0, :])
        print("Upper limits: ", self.robot_joint_pos_upper_limits)
        print("Lower limits: ", self.robot_joint_pos_lower_limits)
        print("Hard vel limits: ", self.robot_hard_vel_limits)

        # We take the first environment's default positions
        joint_ids = [self.robot.joint_names.index(joint_name) for joint_name in self.robot.joint_names]
        default_pos = self.robot.data.default_joint_pos[0].cpu().numpy()
        print(f"{'Joint Name':<25} | {'ID':<5} | {'Default Pos':<12}")
        print("-" * 50)
        for name, j_id, pos in zip(self.robot.data.joint_names, joint_ids, default_pos):
            actuated = True if name in self.cfg.actuated_joint_names else False
            print(f"{name:<25} | {j_id:<5} | {pos:<12.4f} | {'Actuated' if actuated else 'Not Actuated'}")
        print("********************************************************")

        # Initialize a mask of zeros for all 40 actuated joints
        self.action_diff = torch.zeros((self.num_envs, self.cfg.num_actions), dtype=self.dtype, device=self.device)
        self.weighted_action_diff = torch.zeros((self.num_envs, self.cfg.num_actions), dtype=self.dtype, device=self.device)
        self.action_diff_mask = torch.ones((self.cfg.num_actions,), dtype=self.dtype, device=self.device)
        hand_joints = [
                "left_hand_first_finger_joint_1", "left_hand_second_finger_joint_1", "left_hand_third_finger_joint_1",
                "left_hand_thumb_joint_1", "left_hand_thumb_joint_2", "left_hand_thumb_joint_3",
                "left_hand_first_finger_joint_2", "left_hand_second_finger_joint_2", "left_hand_third_finger_joint_2",
                "left_hand_thumb_joint_4",
                "right_hand_first_finger_joint_1", "right_hand_second_finger_joint_1", "right_hand_third_finger_joint_1",
                "right_hand_thumb_joint_1", "right_hand_thumb_joint_2", "right_hand_thumb_joint_3",
                "right_hand_first_finger_joint_2", "right_hand_second_finger_joint_2", "right_hand_third_finger_joint_2",
                "right_hand_thumb_joint_4",
        ]
        # make a list of human frame transformers
        self.num_frames = self.airec_frames.num_bodies
        self.airec_frames_pos = torch.zeros((self.num_envs, self.num_frames, 3), dtype=self.dtype, device=self.device)
        self.airec_frames_rot = torch.zeros((self.num_envs, self.num_frames, 3, 3), dtype=self.dtype, device=self.device)
        self.airec_frames_vel = torch.zeros((self.num_envs, self.num_frames, 3), dtype=self.dtype, device=self.device)
        self.torso_id = self.airec_frames.data.target_frame_names.index("torso") 
        self.left_hand_id = self.airec_frames.data.target_frame_names.index("left_hand") 
        self.right_hand_id = self.airec_frames.data.target_frame_names.index("right_hand") 
        self.left_elbow_id = self.airec_frames.data.target_frame_names.index("left_elbow") 
        self.right_elbow_id = self.airec_frames.data.target_frame_names.index("right_elbow")
        self.base_id = self.airec_frames.data.target_frame_names.index("base")
        self.base_link_body_idx = self.robot.data.body_names.index("base_link")

        # CoM: world position (for markers) and base-relative (for obs + termination)
        self.com_pos_w = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)
        self.com_pos_b = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=self.device)

        self.extras["log"] = {}

        self.extras["counters"] = {}

        self._imitation_ref_joint_pos: torch.Tensor | None = None
        self._imitation_ref_joint_cmd: torch.Tensor | None = None
        self._imitation_ref_base_vel: torch.Tensor | None = None
        self._imitation_joint_idx_t: torch.Tensor | None = None
        #: Optional (T, 3) env-local positions + (T, 4) world quaternions from teleop ``.npz``.
        self._imitation_ref_object_pos_local: torch.Tensor | None = None
        self._imitation_ref_object_quat_w: torch.Tensor | None = None

        if self.cfg.imitation_demo_path:
            from tasks.airec.teleop_demonstrations import load_teleop_demo

            demo = load_teleop_demo(self.cfg.imitation_demo_path)
            ref = torch.as_tensor(demo.measured_joint_pos, dtype=self.dtype, device=self.device)
            if ref.shape[1] != self.num_joints:
                raise ValueError(
                    f"imitation_demo_path: measured_joint_pos has {ref.shape[1]} joints, robot has {self.num_joints}"
                )
            self._imitation_ref_joint_pos = ref
            ref_cmd = torch.as_tensor(demo.joint_commands, dtype=self.dtype, device=self.device)
            if ref_cmd.shape != ref.shape:
                raise ValueError(
                    f"imitation_demo_path: joint_commands shape {ref_cmd.shape} must match measured_joint_pos {ref.shape}"
                )
            self._imitation_ref_joint_cmd = ref_cmd
            if self.actuated_base_dof_indices:
                ref_bv_np = demo.joint_commands[:, self.actuated_base_dof_indices]
                self._imitation_ref_base_vel = torch.as_tensor(
                    ref_bv_np, dtype=self.dtype, device=self.device
                )
            else:
                self._imitation_ref_base_vel = None
            if demo.object_pos_local is not None and demo.object_quat_w is not None:
                pos_t = torch.as_tensor(demo.object_pos_local, dtype=self.dtype, device=self.device)
                quat_t = torch.as_tensor(demo.object_quat_w, dtype=self.dtype, device=self.device)
                self._imitation_ref_object_pos_local = pos_t
                self._imitation_ref_object_quat_w = quat_t
                if pos_t.shape[0] != ref.shape[0]:
                    raise ValueError(
                        "imitation_demo_path: object_pos_local length must match measured_joint_pos time dimension"
                    )
                if quat_t.shape[0] != ref.shape[0]:
                    raise ValueError(
                        "imitation_demo_path: object_quat_w length must match measured_joint_pos time dimension"
                    )
            print(
                f"Loaded imitation demo: {self.cfg.imitation_demo_path} "
                f"({ref.shape[0]} steps; measured_joint_pos + joint_commands references"
                f"{'; object local pos + quat' if self._imitation_ref_object_pos_local is not None else ''})"
            )
            if self.cfg.imitation_joint_indices is not None:
                self._imitation_joint_idx_t = torch.tensor(
                    self.cfg.imitation_joint_indices, device=self.device, dtype=torch.long
                )

            else:
                jn = self.robot.joint_names
                missing = [n for n in IMITATION_DEFAULT_JOINT_NAMES if n not in jn]
                if missing:
                    raise ValueError(f"imitation default joint names not found on robot: {missing}")
                idxs = [jn.index(n) for n in IMITATION_DEFAULT_JOINT_NAMES]
                self._imitation_joint_idx_t = torch.tensor(idxs, device=self.device, dtype=torch.long)
                print(
                    f"Imitation MSE: {len(idxs)} joints (base + torso + arms; head/hands excluded)"
                )


    def _get_imitation_reference_joint_pos(self) -> torch.Tensor:
        """Per-env reference pose from the demo, indexed by episode step (clamped to demo length)."""
        ref = self._imitation_ref_joint_pos
        if ref is None:
            raise RuntimeError("Imitation reference requested but imitation_demo_path is not set")
        T = ref.shape[0]
        idx = torch.clamp(self.episode_length_buf, max=T - 1)
        return ref[idx.long()]

    def _get_imitation_reference_joint_pos_horizon(self, horizon: int) -> torch.Tensor:
        """Demo joint_pos for steps ``t .. t+horizon-1`` per env (clamped), shape ``(num_envs, horizon, num_joints)``."""
        ref = self._imitation_ref_joint_pos
        if ref is None:
            raise RuntimeError("Imitation reference requested but imitation_demo_path is not set")
        T = ref.shape[0]
        h = max(int(horizon), 1)
        t0 = torch.clamp(self.episode_length_buf.long(), max=T - 1)
        offsets = torch.arange(h, device=self.device, dtype=torch.long)
        time_idx = torch.clamp(t0.unsqueeze(-1) + offsets, max=T - 1)
        return ref[time_idx]

    def _get_imitation_reference_joint_cmd(self) -> torch.Tensor:
        """Per-env reference ``joint_pos_cmd`` from the demo, indexed by episode step (clamped to demo length)."""
        ref = self._imitation_ref_joint_cmd
        if ref is None:
            raise RuntimeError("Imitation joint_commands reference requested but imitation_demo_path is not set")
        T = ref.shape[0]
        idx = torch.clamp(self.episode_length_buf, max=T - 1)
        return ref[idx.long()]

    def _get_imitation_reference_joint_cmd_horizon(self, horizon: int) -> torch.Tensor:
        """Demo ``joint_commands`` for steps ``t .. t+horizon-1`` per env (clamped), shape ``(num_envs, horizon, num_joints)``."""
        ref = self._imitation_ref_joint_cmd
        if ref is None:
            raise RuntimeError("Imitation joint_commands reference requested but imitation_demo_path is not set")
        T = ref.shape[0]
        h = max(int(horizon), 1)
        t0 = torch.clamp(self.episode_length_buf.long(), max=T - 1)
        offsets = torch.arange(h, device=self.device, dtype=torch.long)
        time_idx = torch.clamp(t0.unsqueeze(-1) + offsets, max=T - 1)
        return ref[time_idx]

    def _setup_scene(self):
        """Set up the simulation scene with robot, sensors, ground plane, and lighting."""
        super()._setup_scene()

        self.robot = Articulation(self.cfg.robot_cfg)
        self.airec_frames = FrameTransformer(self.cfg.airec_frame_cfg)
        self.com_markers = VisualizationMarkers(self.cfg.com_marker_cfg)
        self.gravity_markers = VisualizationMarkers(self.cfg.gravity_marker_cfg)

        # Add ground plane
        # Replace your ground plane with a smaller, circular floor

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(1000, 1000), visible=False))

        # Visual mat showing robot reset randomization region
        self.reset_mat = RigidObject(self.cfg.mat_cfg)
        self.scene.rigid_objects["reset_mat"] = self.reset_mat

        # Clone, filter, and replicate environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        
        # Add sensors to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["airec_frames"] = self.airec_frames    

        # light bulb
        warm = (1.3, 1.0, 0.3)
        light_bulb = sim_utils.SphereLightCfg(intensity=1500.0, color=warm)
        light_bulb.func("/World/bulb", light_bulb, translation=(1.2, 0, 1.6))    

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions from policy before physics step.

        Args:
            actions: Actions from the policy.
        """
        self.prev_joint_pos_cmd[:] = self.joint_pos_cmd
        self.prev_actions[:] = self.actions
        self.actions = actions.clone()

        if self.cfg.enable_base_control:

            self.joint_pos_cmd[:, self.actuated_body_dof_indices] = scale(
                self.actions[:, 3:],
                self.robot_joint_pos_lower_limits[self.actuated_body_dof_indices],
                self.robot_joint_pos_upper_limits[self.actuated_body_dof_indices],
            )

            self.joint_pos_cmd[:, self.actuated_base_dof_indices] = scale(
                self.actions[:, :3],  # first 3 actions are base (vx, vy, yaw_rate)
                -self.robot_hard_vel_limits[self.actuated_base_dof_indices],
                self.robot_hard_vel_limits[self.actuated_base_dof_indices],
            )

        else:

            self.joint_pos_cmd[:, self.actuated_body_dof_indices] = scale(
                self.actions,
                self.robot_joint_pos_lower_limits[self.actuated_body_dof_indices],
                self.robot_joint_pos_upper_limits[self.actuated_body_dof_indices],
            )


    def _apply_action(self) -> None:
        """Apply actions to the robot.

        Called multiple times per RL step for decimation.
        - Body joints: position control via joint_pos_cmd.
        - Base (when enabled): velocity control via base_cmd.
        """
        self.robot.set_joint_position_target(self.joint_pos_cmd[:, self.actuated_body_dof_indices], joint_ids=self.actuated_body_dof_indices)
        if self.cfg.enable_base_control and self.actuated_base_dof_indices:
            # Base: velocity control via joint velocity targets. AIREC has fixed-base with prismatic
            # joints (trans_x, trans_y, yaw); drive those joints directly. write_root_velocity_to_sim
            # conflicts with joint actuators and can cause CoM/root desync from the visual.
            base_vel = self.joint_pos_cmd[:, self.actuated_base_dof_indices]  # (vx, vy, yaw_rate)
            self.robot.set_joint_velocity_target(base_vel, joint_ids=self.actuated_base_dof_indices)

    def _get_proprioception(self):
        """Return proprioceptive feature vector.
        
        Returns:
            Concatenated tensor containing normalized joint positions, normalized joint
            velocities, joint position commands, previous joint position commands,
            and end-effector positions for both hands.
        """

        prop = torch.cat(
            (
                self.normalised_joint_pos[:, self.actuated_dof_indices],
                self.normalised_joint_vel[:, self.actuated_dof_indices],
                self.joint_pos_error[:, self.actuated_dof_indices],
                self.normalised_joint_applied_torque,
                self.torque_error,
                # node stuffs
                self.airec_frames_pos.flatten(1),
                # Use 6D representation to save space - slice the 3x3 to get 3x2
                self.airec_frames_rot[..., :2].flatten(1),
                self.airec_frames_vel.flatten(1),
                self.action_diff.flatten(1),
                self.com_pos_b,
            ),
            dim=-1,
        )
        return prop
    

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environments at the specified indices.
        
        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot to default state
        self._reset_robot(env_ids)

        # Reset task-specific environment state (implemented in subclasses)
        self._reset_env(env_ids)

        # Refresh intermediate values for observations
        self._compute_intermediate_values(env_ids)

        # Zero velocities — finite-diff from robot teleport is garbage
        self.airec_frames_vel[env_ids] = 0.0

    def _reset_robot(self, env_ids,):
        """Reset the robot joint positions and velocities.

        Args:
            env_ids: Environment indices to reset.
        """
        joint_pos = self.robot.data.default_joint_pos[env_ids]

        # add noise only to the actuated body joints (not base)
        joint_pos[:, self.actuated_body_dof_indices] += sample_uniform(
            -1, 1,
            (len(env_ids), len(self.actuated_body_dof_indices)),
            self.device,
        ) * self.cfg.reset_robot_joint_noise_scale
        joint_pos = torch.clamp(joint_pos, self.robot_joint_pos_lower_limits, self.robot_joint_pos_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        # Base pose: position (trans_x, trans_y) and yaw. The pose is driven by base joints,
        # not root state, so we must set joint_pos for the base DOFs.
        pos_noise = torch.zeros(len(env_ids), 2, device=self.device)
        pos_noise[:, 0] = sample_uniform(-1, 1, (len(env_ids),), self.device) * self.cfg.reset_robot_pos_noise_x
        pos_noise[:, 1] = sample_uniform(-1, 1, (len(env_ids),), self.device) * self.cfg.reset_robot_pos_noise_y
        yaw = sample_uniform(-1, 1, (len(env_ids),), self.device) * self.cfg.reset_robot_quat_noise_scale

        if self.cfg.enable_base_control and self.actuated_base_dof_indices:
            # actuated_base_dof_indices: [trans_x, trans_y, yaw]. Joint values are displacement from spawn.
            joint_pos[:, self.actuated_base_dof_indices[0]] = pos_noise[:, 0]
            joint_pos[:, self.actuated_base_dof_indices[1]] = pos_noise[:, 1]
            joint_pos[:, self.actuated_base_dof_indices[2]] = yaw

        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Zero root velocities. Pose is set via joint_pos; avoid overwriting with write_root_state.
        zero_vel = torch.zeros((len(env_ids), 6), dtype=self.dtype, device=self.device)
        self.robot.write_root_velocity_to_sim(zero_vel, env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.joint_pos_cmd[env_ids] = joint_pos
        self.prev_joint_pos_cmd[env_ids] = self.joint_pos_cmd[env_ids]
    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Compute intermediate values for observations and rewards.
        
        Updates joint velocities, end-effector positions, and rotations for both hands.
        
        Args:
            env_ids: Environment indices to update. If None, updates all environments.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Get robot data
        self.joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        self.joint_acc[env_ids] = self.robot.data.joint_acc[env_ids]
        self.joint_pos_error[env_ids] = self.joint_pos_cmd[env_ids] - self.joint_pos[env_ids]

        # Joint axis is dim 1; [env_ids][indices] would wrongly index the batch (dim 0).
        self.normalised_joint_computed_torque[env_ids] = (
            self.robot.data.computed_torque[env_ids][:, self.torque_sensing_indices] / self.robot_effort_limits
        )
        self.normalised_joint_applied_torque[env_ids] = (
            self.robot.data.applied_torque[env_ids][:, self.torque_sensing_indices] / self.robot_effort_limits
        )
        self.torque_error[env_ids] = self.normalised_joint_computed_torque[env_ids] - self.normalised_joint_applied_torque[env_ids]

        self.normalised_joint_pos[env_ids] = unscale(
            self.joint_pos[env_ids], self.robot_joint_pos_lower_limits, self.robot_joint_pos_upper_limits
        )
        self.normalised_joint_vel[env_ids] = unscale(
            self.joint_vel[env_ids], -self.robot_hard_vel_limits * OVERRIDE_SCALE, self.robot_hard_vel_limits * OVERRIDE_SCALE
        )
        self.action_diff = torch.abs(self.actions - self.prev_actions)
        self.weighted_action_diff = self.action_diff * self.action_diff_mask

        self.airec_frames_vel[env_ids] = (self.airec_frames.data.target_pos_source[env_ids] - self.airec_frames_pos[env_ids]) / self.cfg.physics_dt
        self.airec_frames_pos[env_ids] = self.airec_frames.data.target_pos_source[env_ids]
        self.airec_frames_rot[env_ids] = matrix_from_quat(self.airec_frames.data.target_quat_source[env_ids])

        # Full-body CoM in world, then relative to base_link
        body_com_pos = self.robot.data.body_com_pose_w[..., :3]
        mass = self.robot.data.default_mass.to(device=self.device)
        total_mass = mass.sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.com_pos_w[env_ids] = (body_com_pos[env_ids] * mass[env_ids].unsqueeze(-1)).sum(dim=1) / total_mass[env_ids]

        # base_link from robot body data (same source as body_com_pose_w)
        base_link_pos_w = self.robot.data.body_link_pose_w[:, self.base_link_body_idx, :3]
        base_quat = self.robot.data.body_link_pose_w[:, self.base_link_body_idx, 3:7]

        # com_pos_b = vector from base_link to CoM, expressed in base_link frame
        com_to_base_w = self.com_pos_w[env_ids] - base_link_pos_w[env_ids]
        self.com_pos_b[env_ids] = quat_apply_inverse(base_quat[env_ids], com_to_base_w)

        self.com_markers.visualize(translations=self.com_pos_w)


@torch.jit.script
def scale(x, lower, upper):
    """Scale input `x` from [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """Unscale input `x` from [lower, upper] to [-1, 1].
    For fixed joints (upper == lower), returns 0 to avoid division by zero.
    """
    denom = upper - lower
    # Avoid div-by-zero for fixed joints: use 1.0 so result is 0 when range is zero
    denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
    return (2.0 * x - upper - lower) / denom


# Basic reward components
@torch.jit.script
def distance_reward(object_ee_distance, std: float = 0.1, b: float = 0.0):
    """Compute smooth distance-based reward using tanh shaping.
    
    Args:
        object_ee_distance: Distance between object and end-effector.
        std: Standard deviation for scaling the distance.
        
    Returns:
        Distance reward value.
    """
    r_reach = 1 - torch.tanh((object_ee_distance - b)/ std)
    return r_reach

@torch.jit.script
def exp_distance_reward(object_ee_distance, std: float = 0.1, b: float = 0.0):
    """Compute smooth distance-based reward using tanh shaping.
    
    Args:
        object_ee_distance: Distance between object and end-effector.
        std: Standard deviation for scaling the distance.
        
    Returns:
        Distance reward value.
    """
    r_reach = torch.exp(-(object_ee_distance - b)**2 / std)
    return r_reach
