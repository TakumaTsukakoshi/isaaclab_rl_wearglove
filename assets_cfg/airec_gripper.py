"""Configuration for the AIREC Humanoid Robot.

This configuration loads the AIREC robot from a local USD file and explicitly
defines actuator groups for all 47 joints. The USD file’s default joint drive
properties (e.g. stiffness, damping, and other drive parameters) are maintained.

Reference: /home/simon/IsaacLab/Models/AIREC/dry-airec.usd
"""
import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os
from math import radians

"""
['base_link', 'base_front_left_wheel_link', 'base_front_right_wheel_link', 'base_link_tip', 
'base_rear_left_wheel_link', 'base_rear_right_wheel_link', 'base_front_left_wheel_link_inner_barrel_center', 
'base_front_left_wheel_link_outer_barrel_center', 'base_front_right_wheel_link_inner_barrel_center', 
'base_front_right_wheel_link_outer_barrel_center', 'torso_link_0', 'base_rear_left_wheel_link_inner_barrel_center', 
'base_rear_left_wheel_link_outer_barrel_center', 'base_rear_right_wheel_link_inner_barrel_center', 
'base_rear_right_wheel_link_outer_barrel_center', 'torso_link_1', 'torso_link_2', 'torso_link_3', 
'torso_link_tip', 'torso_left_shoulder_link', 'torso_right_shoulder_link', 'torso_waist_top_link', 
'head_link_0', 'left_arm_link_0', 'right_arm_link_0', 'head_link_1', 'left_arm_link_1', 'right_arm_link_1',
'head_link_2', 'left_arm_link_2', 'right_arm_link_2', 'head_link_3', 'left_arm_link_3', 'right_arm_link_3',
'head_link_tip', 'left_arm_link_4', 'right_arm_link_4', 'head_insta360_camera_link', 'head_link_face_center',
'head_see3cam_left_camera_link', 'head_see3cam_right_camera_link', 'head_sr300_camera_link', 'left_arm_link_5',
'right_arm_link_5', 'head_insta360_camera_color_frame', 'head_see3cam_left_camera_color_frame', 
'head_see3cam_right_camera_color_frame', 'head_sr300_camera_color_frame', 'left_arm_link_6',
'right_arm_link_6', 'head_insta360_camera_color_optical_frame', 'head_see3cam_left_camera_color_optical_frame',
'head_see3cam_right_camera_color_optical_frame', 'head_sr300_camera_color_optical_frame', 'left_arm_link_7', 
'right_arm_link_7', 'left_arm_link_tip', 'right_arm_link_tip', 'left_arm_link_load', 'left_hand_base_link', 
'right_arm_link_load', 'right_hand_base_link', 
'right_gripper_gripper_base', 'right_gripper_right_finger_base_link', 'right_gripper_left_finger_base_link','right_gripper_left_finger_link',
'right_gripper_right_finger_link',
'left_gripper_gripper_base', 'left_gripper_right_finger_base_link', 'left_gripper_left_finger_base_link','left_gripper_left_finger_link',
'left_gripper_right_finger_link']

"""

OVERRIDE_SCALE = 0.1
PARENT_DIR = os.getcwd()


# tensor([0.0873, 0.1571, 0.1571, 0.5585, 0.2618, 0.2618, 0.4712, 0.2618, 0.2618,
#         0.3840, 0.3316, 0.3316, 0.3316, 0.3316, 0.4014, 0.4014, 0.4014, 0.4014,
#         0.4014, 0.4014], device='cuda:0')

# tensor([[0.8727, 1.5708, 1.5708, 5.5851, 2.6180, 2.6180, 4.7124, 2.6180, 2.6180,
#          3.8397, 3.3161, 3.3161, 3.3161, 3.3161, 4.0143, 4.0143, 4.0143, 4.0143,
#          4.0143, 4.0143, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090,
#          1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090,
#          1.3090, 1.3090, 1.3090, 1.3090]], device='cuda:0')

# tensor([1.0000, 1.0000, 1.0000, 5.5851, 1.0000, 1.0000, 4.7124, 1.0000, 1.0000,
#         3.8397, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
#         1.0000, 1.0000, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090,
#         1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090, 1.3090,
#         1.3090, 1.3090, 1.3090, 1.3090], device='cuda:0')

AIREC_CFG = ArticulationCfg(
    ###########################################################################
    # Where and how to load the AIREC USD
    ###########################################################################

    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(PARENT_DIR, "assets/airec_gripperv6/airec3_gripper_v6.usd"),
        activate_contact_sensors=False,
        #to fix self collision, look up collision filtering in isaac sim docs
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16, # default 8
            solver_velocity_iteration_count=8,
            sleep_threshold=0.00001,
            stabilization_threshold=0.00001,
            fix_root_link=True,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.1, # default 1.0
        ),
    ),

    ###########################################################################
    # Initial state: Place the robot in the world.
    # Do not override the USD-specified joint positions so that the original
    # drive (and other joint) properties remain intact.
    ###########################################################################
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Robot base position in world (x, y, z). Adjust if needed.
        joint_pos={
        ##########initial joint positions##########
        # "torso_joint_1": radians(-45),
        # "torso_joint_2": radians(100),
        # "torso_joint_3": radians(0),
        # "head_joint_1": radians(0),
        # "head_joint_2": radians(40),
        # "head_joint_3": radians(0),
        # "left_arm_joint_1": radians(80),
        # "left_arm_joint_2": radians(-32),
        # "left_arm_joint_3": radians(-89),
        # "left_arm_joint_4": radians(110),
        # "left_arm_joint_5": radians(-120),
        # "left_arm_joint_6": radians(12),
        # "left_arm_joint_7": radians(89),
        # "right_arm_joint_1": radians(80),
        # "right_arm_joint_2": radians(-32),
        # "right_arm_joint_3": radians(-89),
        # "right_arm_joint_4": radians(110),
        # "right_arm_joint_5": radians(120),
        # "right_arm_joint_6": radians(12),
        # "right_arm_joint_7": radians(89),
        # "left_gripper_mimic_joint": radians(0),
        # "left_gripper_finger_joint": radians(2.5),
        # "right_gripper_mimic_joint": radians(0),
        # "right_gripper_finger_joint":radians(2.5),
        ##########final joint positions##########
        # "torso_joint_1": radians(-10),
        "torso_joint_1": radians(-15),
        "torso_joint_2": radians(20),
        "torso_joint_3": radians(0),
        "head_joint_1": radians(0),
        "head_joint_2": radians(0),
        "head_joint_3": radians(0),
        "left_arm_joint_1": radians(-3),
        "left_arm_joint_2": radians(-22),
        "left_arm_joint_3": radians(-41),
        # "left_arm_joint_4": radians(100),
        # "left_arm_joint_4": radians(107),
        "left_arm_joint_4": radians(110),
        "left_arm_joint_5": radians(-169),
        "left_arm_joint_6": radians(-30),
        "left_arm_joint_7": radians(89),
        "right_arm_joint_1": radians(-3),
        "right_arm_joint_2": radians(-22),
        "right_arm_joint_3": radians(-41),
        # "right_arm_joint_4": radians(100),
        # "right_arm_joint_4": radians(107),
        "right_arm_joint_4": radians(110),
        "right_arm_joint_5": radians(169),
        "right_arm_joint_6": radians(-30),
        "right_arm_joint_7": radians(89),
        # "left_gripper_mimic_joint": radians(0),
        # "left_gripper_finger_joint": radians(0),
        # "right_gripper_mimic_joint": radians(0),
        # "right_gripper_finger_joint":radians(0),
        "left_gripper_right_finger_joint": radians(2.5),
        "left_gripper_left_finger_joint": radians(2.5),
        "right_gripper_right_finger_joint": radians(2.5),
        "right_gripper_left_finger_joint": radians(2.5),
        },
    ),

    ###########################################################################
    # Actuators: Each group explicitly lists certain joint names.
    # Empty stiffness and damping dicts are passed so that the USD default
    # drive properties are not overridden.
    ###########################################################################

    actuators={

        # -------
        # Torso
        # -------
        "torso_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "torso_joint_1",
                "torso_joint_2",
                "torso_joint_3",
            ],
            stiffness={"torso_joint_1": 3000.0,
                       "torso_joint_2": 1000.0,
                       "torso_joint_3": 500.0,
                    },
            damping={"torso_joint_1": 150.0,
                    "torso_joint_2": 100.0,
                    "torso_joint_3": 50.0
                    },
            velocity_limit_sim={
                "torso_joint_1": 0.872 * OVERRIDE_SCALE,
                "torso_joint_2": 1.571 * OVERRIDE_SCALE,
                "torso_joint_3": 1.571 * OVERRIDE_SCALE
            },
        ),

        # ----
        # Head
        # ----
        "head_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_joint_1",
                "head_joint_2",
                "head_joint_3",
            ],
            stiffness={".*": 1000.0},
            damping={".*": 100.0},
            effort_limit={
                "head_joint_1": 8.0,
                "head_joint_2": 6.0,
                "head_joint_3": 4.0,
            },
            velocity_limit_sim={
                "head_joint_1": 5.585 * OVERRIDE_SCALE,
                "head_joint_2": 4.712 * OVERRIDE_SCALE,
                "head_joint_3": 3.839 * OVERRIDE_SCALE
            },
        ),

        # --------
        # Left arm
        # --------
        "left_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_arm_joint_1",
                "left_arm_joint_2",
                "left_arm_joint_3",
                "left_arm_joint_4",
                "left_arm_joint_5",
                "left_arm_joint_6",
                "left_arm_joint_7",
            ],
            effort_limit={
                "left_arm_joint_1": 70.0,
                "left_arm_joint_2": 150.0,
                "left_arm_joint_3": 100.0,
                "left_arm_joint_4": 190.0,
                "left_arm_joint_5": 80.0,
                "left_arm_joint_6": 60.0,
                "left_arm_joint_7": 50.0,
            },
            stiffness={"left_arm_joint_1": 300,
                    "left_arm_joint_2": 300,
                    "left_arm_joint_3": 200,
                    "left_arm_joint_4": 200,
                    "left_arm_joint_5": 20,
                    "left_arm_joint_6": 20,
                    "left_arm_joint_7": 20,
                    },
            damping={"left_arm_joint_1": 30,
                    "left_arm_joint_2": 30,
                    "left_arm_joint_3": 20,
                    "left_arm_joint_4": 20,
                    "left_arm_joint_5": 1,
                    "left_arm_joint_6": 1,
                    "left_arm_joint_7": 1
            },
            velocity_limit_sim={
                "left_arm_joint_1": 2.617 * OVERRIDE_SCALE,
                "left_arm_joint_2": 2.617 * OVERRIDE_SCALE,
                "left_arm_joint_3": 3.316 * OVERRIDE_SCALE,
                "left_arm_joint_4": 3.316 * OVERRIDE_SCALE,
                "left_arm_joint_5": 4.014 * OVERRIDE_SCALE,
                "left_arm_joint_6": 4.014 * OVERRIDE_SCALE,
                "left_arm_joint_7": 4.014 * OVERRIDE_SCALE
            },
        ),

        # ---------
        # Right arm
        # ---------
        "right_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_arm_joint_1",
                "right_arm_joint_2",
                "right_arm_joint_3",
                "right_arm_joint_4",
                "right_arm_joint_5",
                "right_arm_joint_6",
                "right_arm_joint_7",
            ],
            effort_limit={
                "right_arm_joint_1": 70.0,
                "right_arm_joint_2": 150.0,
                "right_arm_joint_3": 100.0,
                "right_arm_joint_4": 190.0,
                "right_arm_joint_5": 80.0,
                "right_arm_joint_6": 60.0,
                "right_arm_joint_7": 50.0,
            },
            stiffness={"right_arm_joint_1": 300,
                    "right_arm_joint_2": 300,
                    "right_arm_joint_3": 200,
                    "right_arm_joint_4": 200,
                    "right_arm_joint_5": 20,
                    "right_arm_joint_6": 20,
                    "right_arm_joint_7": 20,
                    },
            damping={"right_arm_joint_1": 30,
                    "right_arm_joint_2": 30,
                    "right_arm_joint_3": 20,
                    "right_arm_joint_4": 20,
                    "right_arm_joint_5": 1,
                    "right_arm_joint_6": 1,
                    "right_arm_joint_7": 1
            },
            velocity_limit_sim={
                "right_arm_joint_1": 2.617 * OVERRIDE_SCALE,
                "right_arm_joint_2": 2.617 * OVERRIDE_SCALE,
                "right_arm_joint_3": 3.316 * OVERRIDE_SCALE,
                "right_arm_joint_4": 3.316 * OVERRIDE_SCALE,
                "right_arm_joint_5": 4.014 * OVERRIDE_SCALE,
                "right_arm_joint_6": 4.014 * OVERRIDE_SCALE,
                "right_arm_joint_7": 4.014 * OVERRIDE_SCALE
            },
        ),

        # -----------
        # Left hand
        # -----------
        "left_hand_joints": ImplicitActuatorCfg(
            joint_names_expr=[
               "left_gripper_right_finger_joint",
               "left_gripper_left_finger_joint",
            ],
            stiffness={".*": 3.0},
            damping={".*": 0.3},
        ),

        # ------------
        # Right hand
        # ------------
        "right_hand_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_gripper_right_finger_joint",
                "right_gripper_left_finger_joint",
            ],
            stiffness={".*": 3.0},
            damping={".*": 0.3},
        ),
    },
)