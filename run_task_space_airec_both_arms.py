# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AIREC dual-arm **task-space** teleop demo using differential IK (same backend as the RL task-space env).

This complements ``run_diff_ik_airec_both_arms.py``: both ultimately map **Cartesian pose goals** (task space)
to **joint position targets** via :class:`~isaaclab.controllers.DifferentialIKController`. Here the layout
and naming follow ``tasks/airec/airec2_finger_taskspace.py`` (``_init_task_space_controller``): shared
``ik_controller_cfg``, :class:`~isaaclab.managers.SceneEntityCfg` per arm, Jacobian link indices, and
``joint_pos_des_{left,right}`` buffers. End-effector link names match ``wear_finger`` / ``airec2_finger``
(``right_arm_link_7``, ``left_arm_link_7``). Joint lists match ``assets.airec_finger`` (``ACTUATED_*_JOINTS``).

Run::

    ./isaaclab.sh -p run_task_space_airec_both_arms.py --robot airec --arm both
    ./isaaclab.sh -p run_task_space_airec_both_arms.py --robot franka_panda

"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AIREC task-space (pose) goals → diff IK → joint targets.")
parser.add_argument("--robot", type=str, default="airec", help="Robot: franka_panda, ur10, airec")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments.")
parser.add_argument(
    "--arm",
    type=str,
    default="both",
    choices=["left", "right", "both"],
    help="For AIREC: which arms receive IK (requires left DOFs for left/both).",
)
parser.add_argument(
    "--smooth-alpha",
    type=float,
    default=1.0,
    help="0<alpha<=1: blend task-space goals toward targets each step (1=no smoothing).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from assets.airec_finger import (
    ACTUATED_LARM_JOINTS,
    ACTUATED_RARM_JOINTS,
    ACTUATED_TORSO_JOINTS,
    AIREC_CFG,
)

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort: skip

# Same defaults as ``tasks/airec/airec2_finger_taskspace.py`` / ``AIRECEnvCfg.ik_controller_cfg``.
IK_CONTROLLER_CFG = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=False,
    ik_method="dls",
)

# IK-only robot cfg: actuator joint lists must match whatever PhysX exposes (see run_diff_ik discussion).
AIREC_IK_ROBOT_CFG = AIREC_CFG.replace(
    articulation_root_prim_path=None,
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "all_dof": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={".*": 400.0},
            damping={".*": 40.0},
        ),
    },
)


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "airec":
        robot = AIREC_IK_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported.")


def init_task_space_ik(
    scene: InteractiveScene,
    robot,
    device: str,
    num_envs: int,
):
    """Mirror ``AIRECEnv._init_task_space_controller`` in ``airec2_finger_taskspace.py``."""
    ik_right = DifferentialIKController(IK_CONTROLLER_CFG, num_envs=num_envs, device=device)
    ik_left = DifferentialIKController(IK_CONTROLLER_CFG, num_envs=num_envs, device=device)

    right_arm_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["right_arm_joint_[1-7]"],
        body_names=["right_arm_link_7"],
    )
    left_arm_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["left_arm_joint_[1-7]"],
        body_names=["left_arm_link_7"],
    )
    right_arm_entity_cfg.resolve(scene)
    left_arm_entity_cfg.resolve(scene)

    r_body = int(right_arm_entity_cfg.body_ids[0])
    l_body = int(left_arm_entity_cfg.body_ids[0])
    if robot.is_fixed_base:
        right_ee_jacobi_idx = r_body - 1
        left_ee_jacobi_idx = l_body - 1
    else:
        right_ee_jacobi_idx = r_body
        left_ee_jacobi_idx = l_body

    return {
        "ik_right": ik_right,
        "ik_left": ik_left,
        "right_arm_entity_cfg": right_arm_entity_cfg,
        "left_arm_entity_cfg": left_arm_entity_cfg,
        "right_ee_jacobi_idx": right_ee_jacobi_idx,
        "left_ee_jacobi_idx": left_ee_jacobi_idx,
    }


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    device = robot.device
    n = scene.num_envs
    alpha = float(args_cli.smooth_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("--smooth-alpha must be in (0, 1].")

    if args_cli.robot == "airec":
        jn = set(robot.joint_names)
        has_left_arm = all(f"left_arm_joint_{i}" in jn for i in range(1, 8))
        has_torso = all(t in jn for t in ACTUATED_TORSO_JOINTS)
        if args_cli.arm in ("left", "both") and not has_left_arm:
            raise ValueError(
                "No left arm DOFs on this articulation. Use --arm right, or fix USD / Isaac articulation root."
            )
    else:
        has_left_arm = True
        has_torso = True

    ts = None
    if args_cli.robot == "airec":
        ts = init_task_space_ik(scene, robot, device, n)
        ik_right, ik_left = ts["ik_right"], ts["ik_left"]
        rac, lac = ts["right_arm_entity_cfg"], ts["left_arm_entity_cfg"]
        rji, lji = ts["right_ee_jacobi_idx"], ts["left_ee_jacobi_idx"]
        joint_pos_des_right = torch.zeros((n, len(ACTUATED_RARM_JOINTS)), device=device)
        joint_pos_des_left = torch.zeros((n, len(ACTUATED_LARM_JOINTS)), device=device)

    diff_ik_cfg = IK_CONTROLLER_CFG
    if args_cli.robot != "airec":
        diff_ik_single = DifferentialIKController(diff_ik_cfg, num_envs=n, device=device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    if args_cli.robot == "airec":
        if args_cli.arm in ("right", "both"):
            right_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_current"))
            right_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_goal"))
        if has_left_arm and args_cli.arm in ("left", "both"):
            left_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_current"))
            left_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_goal"))
        torso_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/torso_current"))
    else:
        ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Task-space pose targets (pos + quat wxyz), same convention as ``run_diff_ik_airec_both_arms.py``:
    # one row per waypoint; IK command does not add env_origins (markers add origins for display).
    if args_cli.robot == "airec":
        right_waypoints = torch.tensor(
            [[0.329539, -0.162015, 0.9546824, -0.7045705, -0.41164598, -0.39074743, -0.42596304]],
            device=device,
        )
        left_waypoints = torch.tensor(
            [[0.359539, 0.162015, 0.9546824, -0.37044662, 0.44323373, 0.7046307, -0.41207874]],
            device=device,
        )
        num_goals = int(right_waypoints.shape[0])

        cmd_right = torch.zeros(n, 7, device=device)
        cmd_left = torch.zeros(n, 7, device=device)
        cmd_right[:] = right_waypoints[0]
        cmd_left[:] = left_waypoints[0]

        if args_cli.arm in ("right", "both"):
            ik_right.reset()
            ik_right.set_command(cmd_right)
        if has_left_arm and args_cli.arm in ("left", "both"):
            ik_left.reset()
            ik_left.set_command(cmd_left)

        torso_entity_cfg = None
        initial_torso_pos = None
        if has_torso:
            torso_entity_cfg = SceneEntityCfg("robot", joint_names=list(ACTUATED_TORSO_JOINTS))
            torso_entity_cfg.resolve(scene)
            initial_torso_pos = robot.data.default_joint_pos[:, torso_entity_cfg.joint_ids].clone()

        goal_idx = 0
    else:
        if args_cli.robot == "franka_panda":
            ee_goals = torch.tensor([[0.5, 0.5, 0.7, 0.707, 0, 0.707, 0]], device=device)
        else:
            ee_goals = torch.tensor([[0.5, 0.5, 0.7, 0.707, 0, 0.707, 0]], device=device)
        ik_commands = torch.zeros(n, diff_ik_single.action_dim, device=device)
        ik_commands[:] = ee_goals[0]
        robot_entity_cfg = (
            SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
            if args_cli.robot == "franka_panda"
            else SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
        )
        robot_entity_cfg.resolve(scene)
        ee_body_ids = [int(robot_entity_cfg.body_ids[0])]
        ee_jacobi_idx = ee_body_ids[0] - 1 if robot.is_fixed_base else ee_body_ids[0]
        diff_ik_single.reset()
        diff_ik_single.set_command(ik_commands)
        goal_idx = 0

    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % 200 == 0:
            count = 0
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            if args_cli.robot == "airec":
                goal_idx = (goal_idx + 1) % max(num_goals, 1)
                if args_cli.arm in ("right", "both"):
                    cmd_right[:] = right_waypoints[goal_idx]
                    ik_right.reset()
                    ik_right.set_command(cmd_right)
                    joint_pos_des_right[:] = joint_pos[:, rac.joint_ids]
                if has_left_arm and args_cli.arm in ("left", "both"):
                    cmd_left[:] = left_waypoints[goal_idx]
                    ik_left.reset()
                    ik_left.set_command(cmd_left)
                    joint_pos_des_left[:] = joint_pos[:, lac.joint_ids]
                elif has_left_arm:
                    joint_pos_des_left[:] = joint_pos[:, lac.joint_ids]
            else:
                goal_idx = (goal_idx + 1) % ee_goals.shape[0]
                ik_commands[:] = ee_goals[goal_idx]
                diff_ik_single.reset()
                diff_ik_single.set_command(ik_commands)
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
        else:
            if args_cli.robot == "airec":
                root_pose_w = robot.data.root_pose_w

                if args_cli.arm in ("right", "both"):
                    tgt = right_waypoints[goal_idx].unsqueeze(0).expand(n, -1)
                    if alpha < 1.0:
                        cmd_right[:, :3] += alpha * (tgt[:, :3] - cmd_right[:, :3])
                        cmd_right[:, 3:7] = tgt[:, 3:7]
                    else:
                        cmd_right[:] = tgt
                    ik_right.set_command(cmd_right)
                    jac_r = robot.root_physx_view.get_jacobians()[:, rji, :, rac.joint_ids]
                    ee_r = robot.data.body_pose_w[:, rac.body_ids[0]]
                    q_r = robot.data.joint_pos[:, rac.joint_ids]
                    pos_b, quat_b = subtract_frame_transforms(
                        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_r[:, 0:3], ee_r[:, 3:7]
                    )
                    joint_pos_des_right = ik_right.compute(pos_b, quat_b, jac_r, q_r)
                else:
                    joint_pos_des_right = robot.data.joint_pos[:, rac.joint_ids].clone()

                if has_left_arm and args_cli.arm in ("left", "both"):
                    tgt = left_waypoints[goal_idx].unsqueeze(0).expand(n, -1)
                    if alpha < 1.0:
                        cmd_left[:, :3] += alpha * (tgt[:, :3] - cmd_left[:, :3])
                        cmd_left[:, 3:7] = tgt[:, 3:7]
                    else:
                        cmd_left[:] = tgt
                    ik_left.set_command(cmd_left)
                    jac_l = robot.root_physx_view.get_jacobians()[:, lji, :, lac.joint_ids]
                    ee_l = robot.data.body_pose_w[:, lac.body_ids[0]]
                    q_l = robot.data.joint_pos[:, lac.joint_ids]
                    pos_b, quat_b = subtract_frame_transforms(
                        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_l[:, 0:3], ee_l[:, 3:7]
                    )
                    joint_pos_des_left = ik_left.compute(pos_b, quat_b, jac_l, q_l)
                elif has_left_arm:
                    joint_pos_des_left = robot.data.joint_pos[:, lac.joint_ids].clone()
            else:
                jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
                ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
                root_pose_w = robot.data.root_pose_w
                jp = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                pos_b, quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                joint_pos_des = diff_ik_single.compute(pos_b, quat_b, jacobian, jp)

        if args_cli.robot == "airec":
            if args_cli.arm == "both":
                robot.set_joint_position_target(joint_pos_des_right, joint_ids=rac.joint_ids)
                if has_left_arm:
                    robot.set_joint_position_target(joint_pos_des_left, joint_ids=lac.joint_ids)
            elif args_cli.arm == "right":
                robot.set_joint_position_target(joint_pos_des_right, joint_ids=rac.joint_ids)
                if has_left_arm:
                    robot.set_joint_position_target(joint_pos_des_left, joint_ids=lac.joint_ids)
            else:
                if has_left_arm:
                    robot.set_joint_position_target(joint_pos_des_left, joint_ids=lac.joint_ids)
                robot.set_joint_position_target(joint_pos_des_right, joint_ids=rac.joint_ids)

            if has_torso and torso_entity_cfg is not None:
                robot.set_joint_position_target(initial_torso_pos, joint_ids=torso_entity_cfg.joint_ids)
                robot.set_joint_velocity_target(
                    torch.zeros_like(initial_torso_pos), joint_ids=torso_entity_cfg.joint_ids
                )
        else:
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        scene.write_data_to_sim()

        if args_cli.robot == "airec" and has_torso and torso_entity_cfg is not None:
            robot.write_joint_state_to_sim(
                initial_torso_pos,
                torch.zeros_like(initial_torso_pos),
                joint_ids=torso_entity_cfg.joint_ids,
            )
            if has_left_arm:
                if args_cli.arm in ("right", "both"):
                    robot.write_joint_state_to_sim(
                        joint_pos_des_left,
                        torch.zeros_like(joint_pos_des_left),
                        joint_ids=lac.joint_ids,
                    )
                elif args_cli.arm in ("left", "both"):
                    robot.write_joint_state_to_sim(
                        joint_pos_des_right,
                        torch.zeros_like(joint_pos_des_right),
                        joint_ids=rac.joint_ids,
                    )

        sim.step()
        count += 1
        scene.update(sim_dt)

        if args_cli.robot == "airec":
            torso_marker.visualize(robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7])
            if args_cli.arm in ("right", "both"):
                ee_w = robot.data.body_state_w[:, rac.body_ids[0], 0:7]
                right_ee_marker.visualize(ee_w[:, 0:3], ee_w[:, 3:7])
                right_goal_marker.visualize(
                    cmd_right[:, 0:3] + scene.env_origins, cmd_right[:, 3:7]
                )
            if has_left_arm and args_cli.arm in ("left", "both"):
                ee_w = robot.data.body_state_w[:, lac.body_ids[0], 0:7]
                left_ee_marker.visualize(ee_w[:, 0:3], ee_w[:, 3:7])
                left_goal_marker.visualize(
                    cmd_left[:, 0:3] + scene.env_origins, cmd_left[:, 3:7]
                )
        else:
            ee_w = robot.data.body_state_w[:, ee_body_ids[0], 0:7]
            ee_marker.visualize(ee_w[:, 0:3], ee_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Task-space IK demo: setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
