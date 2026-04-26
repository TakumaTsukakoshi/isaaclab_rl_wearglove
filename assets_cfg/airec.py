"""Configuration for the AIREC Humanoid Robot.

This configuration loads the AIREC robot from a local USD file and explicitly
defines actuator groups for all 47 joints. The USD file’s default joint drive
properties (e.g. stiffness, damping, and other drive parameters) are maintained.

Reference: /home/simon/IsaacLab/Models/AIREC/dry-airec.usd
"""
import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from math import radians


OVERRIDE_SCALE = 1.0

# Joint name lists for AIREC robot
# dry-airec2_standard.usd has no planar base DOFs (no base_joint_trans_* / base_joint_rot_yaw).
ACTUATED_BASE_JOINTS: tuple[str, ...] = ()

ACTUATED_TORSO_JOINTS = [
    "torso_joint_1",
    "torso_joint_2",
    "torso_joint_3",
]

ACTUATED_HEAD_JOINTS = [
    "head_joint_1",
    "head_joint_2",
    "head_joint_3",
]

ACTUATED_LARM_JOINTS = [
    "left_arm_joint_1",
    "left_arm_joint_2",
    "left_arm_joint_3",
    "left_arm_joint_4",
    "left_arm_joint_5",
    "left_arm_joint_6",
    "left_arm_joint_7",
]

ACTUATED_RARM_JOINTS = [
    "right_arm_joint_1",
    "right_arm_joint_2",
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
    "right_arm_joint_6",
    "right_arm_joint_7",
]

ACTUATED_LHAND_JOINTS = [
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
]

ACTUATED_RHAND_JOINTS = [
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
]

BASE_WHEELS = [
    "base_front_left_wheel_joint",
    "base_front_right_wheel_joint",
    "base_rear_left_wheel_joint",
    "base_rear_right_wheel_joint",
]

# Imitation reward MSE default subset: torso + arms (excludes head, hands, wheel joints).
IMITATION_DEFAULT_JOINT_NAMES: tuple[str, ...] = (
    tuple(ACTUATED_TORSO_JOINTS)
    + tuple(ACTUATED_LARM_JOINTS)
    + tuple(ACTUATED_RARM_JOINTS)
)

robot_articulation_settings = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=16,
    sleep_threshold=0.0,
    stabilization_threshold=0.002,
)

default_arm_stiffness = {"left_arm_joint_1": 300, 
                    "left_arm_joint_2": 300,
                    "left_arm_joint_3": 200,
                    "left_arm_joint_4": 200,
                    "left_arm_joint_5": 20,
                    "left_arm_joint_6": 20,
                    "left_arm_joint_7": 20,
                    }

default_arm_damping = {"left_arm_joint_1": 30, 
                    "left_arm_joint_2": 30,
                    "left_arm_joint_3": 20,
                    "left_arm_joint_4": 20,
                    "left_arm_joint_5": 1,
                    "left_arm_joint_6": 1,
                    "left_arm_joint_7": 1
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

high_left_arm_stiffness = {"left_arm_joint_1": 2000, 
                    "left_arm_joint_2": 2000,
                    "left_arm_joint_3": 1000,
                    "left_arm_joint_4": 1000,
                    "left_arm_joint_5": 300,
                    "left_arm_joint_6": 300,
                    "left_arm_joint_7": 300,
                    }

high_left_arm_damping = {"left_arm_joint_1": 200, 
                    "left_arm_joint_2": 200,
                    "left_arm_joint_3": 100,
                    "left_arm_joint_4": 100,
                    "left_arm_joint_5": 30,
                    "left_arm_joint_6": 30,
                    "left_arm_joint_7": 30
                    }

high_right_arm_stiffness = {"right_arm_joint_1": 2000, 

                    "right_arm_joint_2": 2000,
                    "right_arm_joint_3": 1000,
                    "right_arm_joint_4": 1000,
                    "right_arm_joint_5": 300,
                    "right_arm_joint_6": 300,
                    "right_arm_joint_7": 300,
                    }

high_right_arm_damping = {"right_arm_joint_1": 200, 
                    "right_arm_joint_2": 200,
                    "right_arm_joint_3": 100,
                    "right_arm_joint_4": 100,
                    "right_arm_joint_5": 30,
                    "right_arm_joint_6": 30,
                    "right_arm_joint_7": 30
                    }
AIREC_CFG = ArticulationCfg(
    ###########################################################################
    # Where and how to load the AIREC USD
    ###########################################################################

    spawn=sim_utils.UsdFileCfg(
            # usd_path=f"/home/tamon/code/isaaclab_rl_wearglove/assets/airec_finger/dry-airec2_standard.usd",
            usd_path=f"/home/tamon/code/isaaclab_rl_wearglove/assets/airec/dry-airec_collision_filtered-new.usd",
            copy_from_source=True,
            articulation_props=robot_articulation_settings,
            activate_contact_sensors=True,
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.3)),
        ),

    # spawn=sim_utils.UrdfFileCfg(
    #         asset_path=f"/home/elle/code/debug/airec_rl/assets/airec/dry-airec-elle.urdf",
    #         usd_dir=f"/home/elle/code/debug/airec_rl/assets/airec",
    #         usd_file_name="dry-airec-elle-collision.usd",
    #         fix_base=True,
    #         joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
    #             gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
    #                 stiffness=300.0,
    #                 damping=10.0,
    #             ),
    #         ),
    #         articulation_props=robot_articulation_settings,
    #         activate_contact_sensors=True,
    #         visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.3)),
    #     ),


    ###########################################################################
    # Initial state: Place the robot in the world.
    # Do not override the USD-specified joint positions so that the original
    # drive (and other joint) properties remain intact.
    ###########################################################################
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Robot base position in world (x, y, z). Adjust if needed.
        joint_pos={
        # "torso_joint_1": radians(0),
        # "torso_joint_2": radians(0),
        # "torso_joint_3": radians(0),
        # "head_joint_1": radians(0),
        # "head_joint_2": radians(0),
        # "head_joint_3": radians(0),
        # "left_arm_joint_1": radians(0),
        # "left_arm_joint_2": radians(0),
        # "left_arm_joint_3": radians(0),
        # "left_arm_joint_4": radians(0),
        # "left_arm_joint_5": radians(0),
        # "left_arm_joint_6": radians(0),
        # "left_arm_joint_7": radians(0),
        # "right_arm_joint_1": radians(0),
        # "right_arm_joint_2": radians(0),
        # "right_arm_joint_3": radians(0),
        # "right_arm_joint_4": radians(0),
        # "right_arm_joint_5": radians(0),
        # "right_arm_joint_6": radians(0),
        # "right_arm_joint_7": radians(0),
        "torso_joint_1": radians(-34),
        "torso_joint_2": radians(60),
        "torso_joint_3": radians(0),
        # old block
        "left_arm_joint_1": radians(105),
        "left_arm_joint_2": radians(-44),
        "left_arm_joint_3": radians(-92),
        "left_arm_joint_4": radians(55),
        "left_arm_joint_5": radians(90),
        "left_arm_joint_6": radians(25),
        "left_arm_joint_7": radians(0),
        "right_arm_joint_1": radians(105),
        "right_arm_joint_2": radians(-44),
        "right_arm_joint_3": radians(-92),
        "right_arm_joint_4": radians(55),
        "right_arm_joint_5": radians(90),
        "right_arm_joint_6": radians(25),
        "right_arm_joint_7": radians(0),
        # new block
        # "left_arm_joint_1": radians(112),
        # "left_arm_joint_2": radians(-60),
        # "left_arm_joint_3": radians(-90),
        # "left_arm_joint_4": radians(42),
        # "left_arm_joint_5": radians(94),
        # "left_arm_joint_6": radians(20),
        # "left_arm_joint_7": radians(0),
        # "right_arm_joint_1": radians(112),
        # "right_arm_joint_2": radians(-60),
        # "right_arm_joint_3": radians(-90),
        # "right_arm_joint_4": radians(42),
        # "right_arm_joint_5": radians(94),
        # "right_arm_joint_6": radians(20),
        # "right_arm_joint_7": radians(0),
        },
        joint_vel={".*": 0.0},
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
            effort_limit_sim={
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
            effort_limit_sim={
                "left_arm_joint_1": 70.0,
                "left_arm_joint_2": 150.0,
                "left_arm_joint_3": 100.0,
                "left_arm_joint_4": 190.0,
                "left_arm_joint_5": 80.0,
                "left_arm_joint_6": 60.0,
                "left_arm_joint_7": 50.0,
            },
            stiffness=high_left_arm_stiffness,
            damping=high_left_arm_damping,
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
            effort_limit_sim={
                "right_arm_joint_1": 70.0,
                "right_arm_joint_2": 150.0,
                "right_arm_joint_3": 100.0,
                "right_arm_joint_4": 190.0,
                "right_arm_joint_5": 80.0,
                "right_arm_joint_6": 60.0,
                "right_arm_joint_7": 50.0,
            },
            stiffness=high_right_arm_stiffness,
            damping=high_right_arm_damping,
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
            stiffness={".*": 3.0},
            damping={".*": 0.3},
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
            stiffness={".*": 3.0},
            damping={".*": 0.3},
        ),
    },
)