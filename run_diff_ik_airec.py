# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import AssetBaseCfg
from assets.airec_gripper_temp import AIREC_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "airec":
        robot = AIREC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, airec")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    # Lower lambda_val for better accuracy (trades stability for precision)
    # Default is 0.01, lower values like 0.001-0.005 give better tracking
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", 
        use_relative_mode=False, 
        ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    # AIREC-specific goals adjusted for humanoid robot workspace
    # Start with SIMPLE goal: minimal movement, mostly rotating joint 1
    if args_cli.robot == "airec":
        ee_goals = [
            # Goal 1: Just stay near default position (minimal movement)
            # This should verify the IK can track a close goal
           [0.329539, -0.142015,  0.9546824, -0.7045705, -0.41164598, -0.39074743, -0.42596304],
        ]
    else:
        ee_goals = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
        ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    elif args_cli.robot == "airec":
         # Get torso joint IDs to keep torso fixed during motion
        torso_joint_names = ["torso_joint_1", "torso_joint_2", "torso_joint_3"]
        torso_entity_cfg = SceneEntityCfg("robot", joint_names=torso_joint_names)
        torso_entity_cfg.resolve(scene)
        
        # Store initial torso joint positions
        initial_torso_pos = robot.data.default_joint_pos[:, torso_entity_cfg.joint_ids].clone()

        # Get left arm joint IDs to keep left arm fixed during motion
        left_arm_entity_cfg = SceneEntityCfg("robot", joint_names=["left_arm_joint_[1-7]"], body_names=["left_arm_link_7"])
        left_arm_entity_cfg.resolve(scene)
        # Store initial left arm joint positions
        initial_left_arm_pos = robot.data.default_joint_pos[:, left_arm_entity_cfg.joint_ids].clone()
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["right_arm_joint_[1-7]"], body_names=["right_arm_link_7"])

    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, airec")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    ee_body_ids = [int(robot_entity_cfg.body_ids[0])]

    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = ee_body_ids[0] - 1
    else:
        ee_jacobi_idx = ee_body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            # start from current pose
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        
        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        if args_cli.robot == "airec":
            robot.set_joint_position_target(initial_torso_pos, joint_ids=torso_entity_cfg.joint_ids)
            robot.set_joint_position_target(initial_left_arm_pos, joint_ids=left_arm_entity_cfg.joint_ids)
            robot.write_joint_state_to_sim(initial_left_arm_pos, torch.zeros_like(initial_left_arm_pos), joint_ids=left_arm_entity_cfg.joint_ids)
            robot.write_joint_state_to_sim(initial_torso_pos, torch.zeros_like(initial_torso_pos), joint_ids=torso_entity_cfg.joint_ids)
        
        # perform step
        sim.step()
        # if count == 0:
        #     print(f"Initial EE position: {ee_pose_w[:, 0:3].cpu().numpy()}")
        #     print(f"Initial EE orientation (quat): {ee_pose_w[:, 3:7].cpu().numpy()}")
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)
        # import ipdb; ipdb.set_trace()

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, ee_body_ids[0], 0:7]

        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

        pos_error = torch.norm(ee_pose_w[:, 0:3] - (ik_commands[:, 0:3] + scene.env_origins), dim=1)
        # print(f"Step {count}: Avg position error = {pos_error.mean().item():.4f}m")
        print(f"  Per-env errors (m): {pos_error.cpu().numpy()}")       


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()