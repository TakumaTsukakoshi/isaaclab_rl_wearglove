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
parser.add_argument("--arm", type=str, default="both", choices=["left", "right", "both"], help="Which arm to control (left/right/both). Default: both")
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

    # Create controller(s)
    # Lower lambda_val for better accuracy (trades stability for precision)
    # Default is 0.01, lower values like 0.001-0.005 give better tracking
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", 
        use_relative_mode=False, 
        ik_method="dls"
    )
    
    if args_cli.robot == "airec":
        # For dual-arm robots, create controllers based on selected arm
        if args_cli.arm in ["right", "both"]:
            diff_ik_controller_right = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        if args_cli.arm in ["left", "both"]:
            diff_ik_controller_left = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    else:
        # For single-arm robots, create one controller
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    
    if args_cli.robot == "airec":
        # Create markers based on selected arm
        if args_cli.arm in ["right", "both"]:
            right_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_current"))
            right_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_goal"))
        if args_cli.arm in ["left", "both"]:
            left_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_current"))
            left_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_goal"))
        # Torso marker
        torso_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/torso_current"))
    else:
        # Single-arm markers
        ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals and buffers
    if args_cli.robot == "airec":
        # Right arm goals
        right_arm_goals = [
            # Goal 1: Just stay near default position (minimal movement)
            # This should verify the IK can track a close goal
           [0.329539, -0.162015,  0.9546824, -0.7045705, -0.41164598, -0.39074743, -0.42596304],
        ]
        # Left arm goals - using symmetric but reachable position for left arm
        left_arm_goals = [
            # Goal 1: Left arm target position (symmetric workspace)
        #    [0.329539, 0.162015,  0.9546824, -0.7045705, -0.41164598, 0.39074743, -0.42596304],
           [0.359539, 0.162015,  0.9546824, -0.37044662, 0.44323373, 0.7046307, -0.41207874]
        ]
        
        # Track the given commands
        current_goal_idx = 0
        
        # Create goal tensors
        right_arm_goals = torch.tensor(right_arm_goals, device=sim.device)
        left_arm_goals = torch.tensor(left_arm_goals, device=sim.device)
        
        # Create buffers to store actions for selected arms
        if args_cli.arm in ["right", "both"]:
            ik_commands_right = torch.zeros(scene.num_envs, diff_ik_controller_right.action_dim, device=robot.device)
            ik_commands_right[:] = right_arm_goals[current_goal_idx]
        
        if args_cli.arm in ["left", "both"]:
            ik_commands_left = torch.zeros(scene.num_envs, diff_ik_controller_left.action_dim, device=robot.device)
            ik_commands_left[:] = left_arm_goals[current_goal_idx]
    else:
        # Single-arm robot goals
        if args_cli.robot == "franka_panda":
            ee_goals = [
                [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            ]
        else:  # ur10
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
        # Right arm configuration
        right_arm_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["right_arm_joint_[1-7]"], body_names=["right_arm_link_7"]
        )
        
        # Left arm configuration
        left_arm_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["left_arm_joint_[1-7]"], body_names=["left_arm_link_7"]
        )
        
        # Torso configuration (to keep fixed)
        torso_joint_names = ["torso_joint_1", "torso_joint_2", "torso_joint_3"]
        torso_entity_cfg = SceneEntityCfg("robot", joint_names=torso_joint_names)
        torso_entity_cfg.resolve(scene)
        
        # Store initial torso joint positions
        initial_torso_pos = robot.data.default_joint_pos[:, torso_entity_cfg.joint_ids].clone()

    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, airec")
    
    # Resolving the scene entities
    if args_cli.robot != "airec":
        robot_entity_cfg.resolve(scene)
    else:
        right_arm_entity_cfg.resolve(scene)
        left_arm_entity_cfg.resolve(scene)

    # Set up end-effector body IDs and Jacobian indices
    if args_cli.robot != "airec":
        ee_body_ids = [int(robot_entity_cfg.body_ids[0])]
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index.
        if robot.is_fixed_base:
            ee_jacobi_idx = ee_body_ids[0] - 1
        else:
            ee_jacobi_idx = ee_body_ids[0]
    else:
        # Right arm end-effector
        right_ee_body_ids = [int(right_arm_entity_cfg.body_ids[0])]
        if robot.is_fixed_base:
            right_ee_jacobi_idx = right_ee_body_ids[0] - 1
        else:
            right_ee_jacobi_idx = right_ee_body_ids[0]
        
        # Left arm end-effector
        left_ee_body_ids = [int(left_arm_entity_cfg.body_ids[0])]
        if robot.is_fixed_base:
            left_ee_jacobi_idx = left_ee_body_ids[0] - 1
        else:
            left_ee_jacobi_idx = left_ee_body_ids[0]


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
            if args_cli.robot == "airec":
                # Reset controllers and goals based on selected arm
                if args_cli.arm in ["right", "both"]:
                    ik_commands_right[:] = right_arm_goals[current_goal_idx]
                    diff_ik_controller_right.reset()
                    diff_ik_controller_right.set_command(ik_commands_right)
                    joint_pos_des_right = joint_pos[:, right_arm_entity_cfg.joint_ids].clone()
                else:
                    # Keep right arm at default position if not controlled
                    joint_pos_des_right = joint_pos[:, right_arm_entity_cfg.joint_ids].clone()
                
                if args_cli.arm in ["left", "both"]:
                    ik_commands_left[:] = left_arm_goals[current_goal_idx]
                    diff_ik_controller_left.reset()
                    diff_ik_controller_left.set_command(ik_commands_left)
                    joint_pos_des_left = joint_pos[:, left_arm_entity_cfg.joint_ids].clone()
                else:
                    # Keep left arm at default position if not controlled
                    joint_pos_des_left = joint_pos[:, left_arm_entity_cfg.joint_ids].clone()
                # change goal
                current_goal_idx = (current_goal_idx + 1) % len(right_arm_goals)
            else:
                ik_commands[:] = ee_goals[current_goal_idx]
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)
                # change goal
                current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            if args_cli.robot == "airec":
                root_pose_w = robot.data.root_pose_w
                
                # Right arm control (only if selected)
                if args_cli.arm in ["right", "both"]:
                    jacobian_right = robot.root_physx_view.get_jacobians()[:, right_ee_jacobi_idx, :, right_arm_entity_cfg.joint_ids]
                    ee_pose_w_right = robot.data.body_pose_w[:, right_arm_entity_cfg.body_ids[0]]
                    joint_pos_right = robot.data.joint_pos[:, right_arm_entity_cfg.joint_ids]
                    # compute frame in root frame
                    ee_pos_b_right, ee_quat_b_right = subtract_frame_transforms(
                        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7]
                    )
                    # compute the joint commands for right arm
                    joint_pos_des_right = diff_ik_controller_right.compute(ee_pos_b_right, ee_quat_b_right, jacobian_right, joint_pos_right)
                else:
                    # Keep right arm fixed
                    joint_pos_des_right = robot.data.joint_pos[:, right_arm_entity_cfg.joint_ids].clone()
                
                # Left arm control (only if selected)
                if args_cli.arm in ["left", "both"]:
                    jacobian_left = robot.root_physx_view.get_jacobians()[:, left_ee_jacobi_idx, :, left_arm_entity_cfg.joint_ids]
                    ee_pose_w_left = robot.data.body_pose_w[:, left_arm_entity_cfg.body_ids[0]]
                    joint_pos_left = robot.data.joint_pos[:, left_arm_entity_cfg.joint_ids]
                    # compute frame in root frame
                    ee_pos_b_left, ee_quat_b_left = subtract_frame_transforms(
                        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7]
                    )
                    # print(f"Left EE Pose (world frame): {ee_quat_b_left[0].cpu().numpy()}")
                    # compute the joint commands for left arm
                    joint_pos_des_left = diff_ik_controller_left.compute(ee_pos_b_left, ee_quat_b_left, jacobian_left, joint_pos_left)
                else:
                    # Keep left arm fixed
                    joint_pos_des_left = robot.data.joint_pos[:, left_arm_entity_cfg.joint_ids].clone()
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
        if args_cli.robot == "airec":
            # Apply commands for both arms
            if args_cli.arm in ["both"]:
                robot.set_joint_position_target(joint_pos_des_right, joint_ids=right_arm_entity_cfg.joint_ids)
                robot.set_joint_position_target(joint_pos_des_left, joint_ids=left_arm_entity_cfg.joint_ids)
            elif args_cli.arm == "right":
                robot.set_joint_position_target(joint_pos_des_right, joint_ids=right_arm_entity_cfg.joint_ids)
                robot.set_joint_position_target(joint_pos_des_left, joint_ids=left_arm_entity_cfg.joint_ids)
                # Keep left arm fixed with zero velocity - explicitly write state to ensure no motion
            elif args_cli.arm == "left":
                robot.set_joint_position_target(joint_pos_des_left, joint_ids=left_arm_entity_cfg.joint_ids)
                robot.set_joint_position_target(joint_pos_des_right, joint_ids=right_arm_entity_cfg.joint_ids)
                # Keep right arm fixed with zero velocity - explicitly write state to ensure no motion

            # Keep torso fixed with zero velocity - explicitly write state to ensure no motion
            robot.set_joint_position_target(initial_torso_pos, joint_ids=torso_entity_cfg.joint_ids)
            robot.set_joint_velocity_target(torch.zeros_like(initial_torso_pos), joint_ids=torso_entity_cfg.joint_ids)
        else:
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        scene.write_data_to_sim()
        
        # For AIREC, explicitly write torso state after physics step to prevent drift
        if args_cli.robot == "airec":
            robot.write_joint_state_to_sim(
                initial_torso_pos, 
                torch.zeros_like(initial_torso_pos), 
                joint_ids=torso_entity_cfg.joint_ids
            )
            if args_cli.arm in ["right", "both"]:
                robot.write_joint_state_to_sim(
                    joint_pos_des_left, 
                    torch.zeros_like(joint_pos_des_left), 
                    joint_ids=left_arm_entity_cfg.joint_ids
                )
            elif args_cli.arm in ["left", "both"]:
                robot.write_joint_state_to_sim(
                    joint_pos_des_right, 
                    torch.zeros_like(joint_pos_des_right), 
                    joint_ids=right_arm_entity_cfg.joint_ids
                )
        
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation and visualize
        if args_cli.robot == "airec":
            # Torso visualization
            torso_pose_w = robot.data.root_pose_w
            torso_marker.visualize(torso_pose_w[:, 0:3], torso_pose_w[:, 3:7])
            
            # Visualize based on selected arm(s)
            if args_cli.arm in ["right", "both"]:
                ee_pose_w_right = robot.data.body_state_w[:, right_arm_entity_cfg.body_ids[0], 0:7]
                right_ee_marker.visualize(ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7])
                right_goal_marker.visualize(ik_commands_right[:, 0:3] + scene.env_origins, ik_commands_right[:, 3:7])
                pos_error_right = torch.norm(ee_pose_w_right[:, 0:3] - (ik_commands_right[:, 0:3] + scene.env_origins), dim=1)
            
            if args_cli.arm in ["left", "both"]:
                ee_pose_w_left = robot.data.body_state_w[:, left_arm_entity_cfg.body_ids[0], 0:7]
                left_ee_marker.visualize(ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7])
                left_goal_marker.visualize(ik_commands_left[:, 0:3] + scene.env_origins, ik_commands_left[:, 3:7])
                pos_error_left = torch.norm(ee_pose_w_left[:, 0:3] - (ik_commands_left[:, 0:3] + scene.env_origins), dim=1)
            
            # Print error based on selected arm(s)
            if args_cli.arm == "right":
                print(f"Right arm error (m): {pos_error_right.mean().item():.4f}")
            elif args_cli.arm == "left":
                print(f"Left arm error (m): {pos_error_left.mean().item():.4f}")
            else:  # both
                print(f"Right arm error (m): {pos_error_right.mean().item():.4f}, Left arm error (m): {pos_error_left.mean().item():.4f}")
        else:
            ee_pose_w = robot.data.body_state_w[:, ee_body_ids[0], 0:7]
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
            pos_error = torch.norm(ee_pose_w[:, 0:3] - (ik_commands[:, 0:3] + scene.env_origins), dim=1)
            # print(f"  Per-env errors (m): {pos_error.cpu().numpy()}")       


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