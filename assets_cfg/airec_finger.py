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
'right_arm_link_load', 'right_hand_base_link', 'left_hand_first_finger_link_0', 'left_hand_palm_link', 
'left_hand_second_finger_link_0', 'left_hand_third_finger_link_0', 'left_hand_thumb_link_0', 'right_hand_first_finger_link_0', 
'right_hand_palm_link', 'right_hand_second_finger_link_0', 'right_hand_third_finger_link_0', 'right_hand_thumb_link_0', 
'left_hand_first_finger_link_1', 'left_hand_second_finger_link_1', 'left_hand_third_finger_link_1', 'left_hand_thumb_link_1',
'right_hand_first_finger_link_1', 'right_hand_second_finger_link_1', 'right_hand_third_finger_link_1',
'right_hand_thumb_link_1', 'left_hand_first_finger_link_2', 'left_hand_second_finger_link_2', 'left_hand_third_finger_link_2', 
'left_hand_thumb_link_2', 'right_hand_first_finger_link_2', 'right_hand_second_finger_link_2', 'right_hand_third_finger_link_2', 
'right_hand_thumb_link_2', 'left_hand_first_finger_link_tip', 'left_hand_second_finger_link_tip', 'left_hand_third_finger_link_tip',
 'left_hand_thumb_link_2_3', 'right_hand_first_finger_link_tip', 'right_hand_second_finger_link_tip', 'right_hand_third_finger_link_tip',
   'right_hand_thumb_link_2_3', 'left_hand_thumb_link_3', 'right_hand_thumb_link_3', 'left_hand_thumb_link_4', 'right_hand_thumb_link_4',
     'left_hand_thumb_link_tip', 'right_hand_thumb_link_tip']
"""
"""
AIREC2
base_link
base_front_left_wheel_link
base_front_left_wheel_link_inner_barrel_center
base_front_left_wheel_link_outer_barrel_center
base_front_right_wheel_link
base_front_right_wheel_link_inner_barrel_center
base_front_right_wheel_link_outer_barrel_center
torso_link_0
torso_link_1
torso_link_2
torso_link_3
torso_link_tip

head_link_0
head_link_1
head_link_2
head_link_3
head_link_tip

head_insta360_camera_link
head_insta360_camera_color_frame
head_insta360_camera_color_optical_frame

head_link_face_center

head_see3cam_left_camera_link
head_see3cam_left_camera_color_frame
head_see3cam_left_camera_color_optical_frame

head_see3cam_right_camera_link
head_see3cam_right_camera_color_frame
head_see3cam_right_camera_color_optical_frame

head_sr300_camera_link
head_sr300_camera_color_frame
head_sr300_camera_color_optical_frame

torso_left_shoulder_link
torso_right_shoulder_link
torso_waistup_link

left_arm_link_0
left_arm_link_1
left_arm_link_2
left_arm_link_3
left_arm_link_4
left_arm_link_5
left_arm_link_6
left_arm_link_7
left_arm_link_tip
left_arm_link_load

left_hand_base_link
left_hand_palm_link

left_hand_first_finger_link_0
left_hand_first_finger_link_1
left_hand_first_finger_link_2
left_hand_first_finger_link_tip

left_hand_second_finger_link_0
left_hand_second_finger_link_1
left_hand_second_finger_link_2
left_hand_second_finger_link_tip

left_hand_third_finger_link_0
left_hand_third_finger_link_1
left_hand_third_finger_link_2
left_hand_third_finger_link_tip

left_hand_thumb_link_0
left_hand_thumb_link_1
left_hand_thumb_link_2
left_hand_thumb_link_3
left_hand_thumb_link_4
left_hand_thumb_link_tip

right_arm_link_0
right_arm_link_1
right_arm_link_2
right_arm_link_3
right_arm_link_4
right_arm_link_5
right_arm_link_6
right_arm_link_7
right_arm_link_tip
right_arm_link_load

right_hand_base_link
right_hand_palm_link

right_hand_first_finger_link_0
right_hand_first_finger_link_1
right_hand_first_finger_link_2
right_hand_first_finger_link_tip

right_hand_second_finger_link_0
right_hand_second_finger_link_1
right_hand_second_finger_link_2
right_hand_second_finger_link_tip

right_hand_third_finger_link_0
right_hand_third_finger_link_1
right_hand_third_finger_link_2
right_hand_third_finger_link_tip

right_hand_thumb_link_0
right_hand_thumb_link_1
right_hand_thumb_link_2
right_hand_thumb_link_3
right_hand_thumb_link_4
right_hand_thumb_link_tip

base_rear_left_wheel_link
base_rear_left_wheel_link_inner_barrel_center
base_rear_left_wheel_link_outer_barrel_center

base_rear_right_wheel_link
base_rear_right_wheel_link_inner_barrel_center
base_rear_right_wheel_link_outer_barrel_center
"""


OVERRIDE_SCALE = 0.10
PARENT_DIR = os.getcwd()

ACTUATED_TORSO_JOINTS = ["torso_joint_1", "torso_joint_2", "torso_joint_3"]
ACTUATED_HEAD_JOINTS = ["head_joint_1", "head_joint_2", "head_joint_3"]
ACTUATED_RARM_JOINTS = [f"right_arm_joint_{i}" for i in range(1, 8)]
ACTUATED_LARM_JOINTS = [f"left_arm_joint_{i}" for i in range(1, 8)]
ACTUATED_LHAND_JOINTS = ["left_hand_first_finger_joint_1", "left_hand_first_finger_joint_2","left_hand_thumb_joint_1", "left_hand_thumb_joint_2", "left_hand_thumb_joint_3", "left_hand_thumb_joint_4"]
ACTUATED_RHAND_JOINTS = ["right_hand_first_finger_joint_1", "right_hand_first_finger_joint_2", "right_hand_thumb_joint_1", "right_hand_thumb_joint_2", "right_hand_thumb_joint_3", "right_hand_thumb_joint_4"]
ACTUATED_BASE_JOINTS = [
    "base_joint_trans_x",
    "base_joint_trans_y",
    "base_joint_rot_yaw",]  # No base joints for AIREC2

IMITATION_DEFAULT_JOINT_NAMES: tuple[str, ...] = (
    tuple(ACTUATED_TORSO_JOINTS)
    + tuple(ACTUATED_LARM_JOINTS)
    + tuple(ACTUATED_RARM_JOINTS)
    + tuple(ACTUATED_LHAND_JOINTS)
    + tuple(ACTUATED_RHAND_JOINTS)
)

robot_articulation_settings = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=24, # default 8
            solver_velocity_iteration_count=12,
            sleep_threshold=0.00001,
            stabilization_threshold=0.00001,
            fix_root_link=True,
        )

rigid_body_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.5
        )

default_left_arm_stiffness = {"left_arm_joint_1": 300, 
                    "left_arm_joint_2": 300,
                    "left_arm_joint_3": 200,
                    "left_arm_joint_4": 200,
                    "left_arm_joint_5": 20,
                    "left_arm_joint_6": 20,
                    "left_arm_joint_7": 20,
                    }

default_left_arm_damping = {"left_arm_joint_1": 30, 
                    "left_arm_joint_2": 30,
                    "left_arm_joint_3": 20,
                    "left_arm_joint_4": 20,
                    "left_arm_joint_5": 1,
                    "left_arm_joint_6": 1,
                    "left_arm_joint_7": 1
                    }
default_right_arm_stiffness = {"right_arm_joint_1": 300, 
                    "right_arm_joint_2": 300,
                    "right_arm_joint_3": 200,
                    "right_arm_joint_4": 200,
                    "right_arm_joint_5": 20,
                    "right_arm_joint_6": 20,
                    "right_arm_joint_7": 20,
                    }
default_right_arm_damping = {"right_arm_joint_1": 30, 
                    "right_arm_joint_2": 30,
                    "right_arm_joint_3": 20,
                    "right_arm_joint_4": 20,
                    "right_arm_joint_5": 1,
                    "right_arm_joint_6": 1,
                    "right_arm_joint_7": 1
                    }

default_torso_stiffness={"torso_joint_1": 3000.0,
                       "torso_joint_2": 1000.0,
                       "torso_joint_3": 500.0,
                    }
default_torso_damping={"torso_joint_1": 150.0,
                    "torso_joint_2": 100.0,
                    "torso_joint_3": 50.0
                    }

high_torso_stiffness={"torso_joint_1": 10000,
                       "torso_joint_2": 8000,
                       "torso_joint_3": 3000, 
                    }
high_torso_damping={"torso_joint_1": 1000,
                    "torso_joint_2": 800,
                    "torso_joint_3": 300,
                    }

high_left_arm_stiffness = {"left_arm_joint_1": 5000, 
                    "left_arm_joint_2": 5000,
                    "left_arm_joint_3": 3000,
                    "left_arm_joint_4": 1000,
                    "left_arm_joint_5": 1000,
                    "left_arm_joint_6": 1000,
                    "left_arm_joint_7": 1000,
                    }

high_left_arm_damping = {"left_arm_joint_1": 500, 
                    "left_arm_joint_2": 500,
                    "left_arm_joint_3": 100,
                    "left_arm_joint_4": 100,
                    "left_arm_joint_5": 100,
                    "left_arm_joint_6": 100,
                    "left_arm_joint_7": 100
                    }

high_right_arm_stiffness = {"right_arm_joint_1": 5000, 

                    "right_arm_joint_2": 5000,
                    "right_arm_joint_3": 3000,
                    "right_arm_joint_4": 3000,
                    "right_arm_joint_5": 1000,
                    "right_arm_joint_6": 1000,
                    "right_arm_joint_7": 1000,
                    }

high_right_arm_damping = {"right_arm_joint_1": 500, 
                    "right_arm_joint_2": 500,
                    "right_arm_joint_3": 300,
                    "right_arm_joint_4": 300,
                    "right_arm_joint_5": 100,
                    "right_arm_joint_6": 100,
                    "right_arm_joint_7": 100
                    }

AIREC_CFG = ArticulationCfg(
    ###########################################################################
    # Where and how to load the AIREC USD
    ###########################################################################
    # Pin the articulation root so PhysX does not bind a nested subtree (e.g. right arm only), which
    # would yield joint_names like right_arm_joint_* only and break actuators / IK that expect both arms.
    # Must start with ``/`` so Isaac joins ``…/Robot`` + ``/base_link`` → ``…/Robot/base_link`` (not ``Robotbase_link``).
    articulation_root_prim_path="/world",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=os.path.join(PARENT_DIR, "assets/airec/dry-airec_collision_filtered-new.usd"),
        usd_path=os.path.join(PARENT_DIR, "assets/airec2_finger_v3/airec2_finger.usd"),
        activate_contact_sensors=False,
        #to fix self collision, look up collision filtering in isaac sim docs
        articulation_props=robot_articulation_settings,
        rigid_props=rigid_body_props,
    ),

    ###########################################################################
    # Initial state: Place the robot in the world.
    # Do not override the USD-specified joint positions so that the original
    # drive (and other joint) properties remain intact.
    ###########################################################################
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.0),  # Robot base position in world (x, y, z). Adjust if needed.
    #     joint_pos={
     
    #     "torso_joint_1": radians(-30),
    #     "torso_joint_2": radians(60),
    #     "torso_joint_3": radians(0),
    #     "head_joint_1": radians(0),
    #     "head_joint_2": radians(0),
    #     "head_joint_3": radians(0),
    #     "left_arm_joint_1": radians(20),
    #     # "left_arm_joint_2": radians(-14),
    #     # "left_arm_joint_2": radians(-20), default
    #     "left_arm_joint_2": radians(-10),
    #     "left_arm_joint_3": radians(-14),
    #     # "left_arm_joint_4": radians(100),
    #     "left_arm_joint_4": radians(105),
    #     "left_arm_joint_5": radians(26),
    #     # "left_arm_joint_6": radians(45),
    #     "left_arm_joint_6": radians(65),
    #     "left_arm_joint_7": radians(-17),
    #     # "right_arm_joint_1": radians(33),
    #     "right_arm_joint_1": radians(20),
    #     # "right_arm_joint_2": radians(-14),
    #     # "right_arm_joint_2": radians(-20),default
    #     "right_arm_joint_2": radians(-10),
    #     "right_arm_joint_3": radians(-14),
    #     # "right_arm_joint_4": radians(100),
    #     "right_arm_joint_4": radians(105),
    #     "right_arm_joint_5": radians(26),
    #     # "right_arm_joint_6": radians(45),
    #     "right_arm_joint_6": radians(65),
    #     "right_arm_joint_7": radians(17),
    #     # "left_hand_thumb_joint_1": radians(75),
    #     # "left_hand_thumb_joint_2": radians(13),
    #     # "left_hand_thumb_joint_3": radians(4),
    #     # "left_hand_thumb_joint_4": radians(0),
    #     "left_hand_thumb_joint_1": radians(75),
    #     "left_hand_thumb_joint_2": radians(5),
    #     "left_hand_thumb_joint_3": radians(5),
    #     "left_hand_thumb_joint_4": radians(0),
    #     "left_hand_first_finger_joint_1": radians(89),
    #     "left_hand_first_finger_joint_2": radians(0),
    #     "left_hand_second_finger_joint_1": radians(89),
    #     "left_hand_second_finger_joint_2": radians(89),
    #     # "left_hand_second_finger_joint_1": radians(89),
    #     # "left_hand_second_finger_joint_2": radians(89),
    #     "left_hand_third_finger_joint_1": radians(89),
    #     "left_hand_third_finger_joint_2": radians(89),
    #     # "right_hand_thumb_joint_1": radians(75),
    #     # "right_hand_thumb_joint_1": radians(75),
    #     # "right_hand_thumb_joint_2": radians(13),
    #     # "right_hand_thumb_joint_3": radians(4),
    #     # "right_hand_thumb_joint_4": radians(0),
    #     "right_hand_thumb_joint_1": radians(75),
    #     "right_hand_thumb_joint_2": radians(5),
    #     "right_hand_thumb_joint_3": radians(5),
    #     "right_hand_thumb_joint_4": radians(0),
    #     "right_hand_first_finger_joint_1": radians(89),
    #     "right_hand_first_finger_joint_2": radians(0),
    #     "right_hand_second_finger_joint_1": radians(89),
    #     "right_hand_second_finger_joint_2": radians(89),
    #     # "right_hand_second_finger_joint_1": radians(89),
    #     # "right_hand_second_finger_joint_2": radians(89),
    #     "right_hand_third_finger_joint_1": radians(89),
    #     "right_hand_third_finger_joint_2": radians(89),
    #     },
    # ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Robot base position in world (x, y, z). Adjust if needed.
        joint_pos={
     
        "torso_joint_1": radians(-30),
        "torso_joint_2": radians(60),
        "torso_joint_3": radians(0),
        "head_joint_1": radians(0),
        "head_joint_2": radians(0),
        "head_joint_3": radians(0),
        "left_arm_joint_1": radians(20),
        # "left_arm_joint_2": radians(-14),
        # "left_arm_joint_2": radians(-20), default
        "left_arm_joint_2": radians(-18),
        "left_arm_joint_3": radians(-14),
        # "left_arm_joint_4": radians(100),
        "left_arm_joint_4": radians(105),
        "left_arm_joint_5": radians(26),
        # "left_arm_joint_6": radians(45),
        "left_arm_joint_6": radians(65),
        "left_arm_joint_7": radians(-13),
        # "right_arm_joint_1": radians(33),
        "right_arm_joint_1": radians(20),
        # "right_arm_joint_2": radians(-14),
        # "right_arm_joint_2": radians(-20),default
        "right_arm_joint_2": radians(-18),
        "right_arm_joint_3": radians(-14),
        # "right_arm_joint_4": radians(100),
        "right_arm_joint_4": radians(105),
        "right_arm_joint_5": radians(26),
        # "right_arm_joint_6": radians(45),
        "right_arm_joint_6": radians(65),
        "right_arm_joint_7": radians(13),
        # "left_hand_thumb_joint_1": radians(75),
        # "left_hand_thumb_joint_2": radians(13),
        # "left_hand_thumb_joint_3": radians(4),
        # "left_hand_thumb_joint_4": radians(0),
        "left_hand_thumb_joint_1": radians(75),
        "left_hand_thumb_joint_2": radians(5),
        "left_hand_thumb_joint_3": radians(5),
        "left_hand_thumb_joint_4": radians(0),
        "left_hand_first_finger_joint_1": radians(89),
        "left_hand_first_finger_joint_2": radians(0),
        "left_hand_second_finger_joint_1": radians(89),
        "left_hand_second_finger_joint_2": radians(89),
        # "left_hand_second_finger_joint_1": radians(89),
        # "left_hand_second_finger_joint_2": radians(89),
        "left_hand_third_finger_joint_1": radians(89),
        "left_hand_third_finger_joint_2": radians(89),
        # "right_hand_thumb_joint_1": radians(75),
        # "right_hand_thumb_joint_1": radians(75),
        # "right_hand_thumb_joint_2": radians(13),
        # "right_hand_thumb_joint_3": radians(4),
        # "right_hand_thumb_joint_4": radians(0),
        "right_hand_thumb_joint_1": radians(75),
        "right_hand_thumb_joint_2": radians(5),
        "right_hand_thumb_joint_3": radians(5),
        "right_hand_thumb_joint_4": radians(0),
        "right_hand_first_finger_joint_1": radians(89),
        "right_hand_first_finger_joint_2": radians(0),
        "right_hand_second_finger_joint_1": radians(89),
        "right_hand_second_finger_joint_2": radians(89),
        # "right_hand_second_finger_joint_1": radians(89),
        # "right_hand_second_finger_joint_2": radians(89),
        "right_hand_third_finger_joint_1": radians(89),
        "right_hand_third_finger_joint_2": radians(89),
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
            stiffness=high_torso_stiffness,
            damping=high_torso_damping,
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
            stiffness=default_left_arm_stiffness,
            damping=default_left_arm_damping,
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
            stiffness=default_right_arm_stiffness,
            damping=default_right_arm_damping,
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
                "left_hand_first_finger_joint_1",
                "left_hand_second_finger_joint_1",
                "left_hand_third_finger_joint_1",
                "left_hand_thumb_joint_1",
                "left_hand_thumb_joint_2",
                "left_hand_thumb_joint_3",
                "left_hand_first_finger_joint_2",
                "left_hand_second_finger_joint_2",
                "left_hand_third_finger_joint_2",
                "left_hand_thumb_joint_4",
            ],
            stiffness={".*": 350.0},
            damping={".*": 170.0},
        ),

        # ------------
        # Right hand
        # ------------
        "right_hand_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_hand_first_finger_joint_1",
                "right_hand_second_finger_joint_1",
                "right_hand_third_finger_joint_1",
                "right_hand_thumb_joint_1",
                "right_hand_thumb_joint_2",
                "right_hand_thumb_joint_3",
                "right_hand_first_finger_joint_2",
                "right_hand_second_finger_joint_2",
                "right_hand_third_finger_joint_2",
                "right_hand_thumb_joint_4",
            ],
            stiffness={".*": 350.0},
            damping={".*": 170.0},
        ),
    },
)