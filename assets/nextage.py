# nextage.py
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

PARENT_DIR = os.getcwd()


OVERRIDE_SCALE = 0.15

ROBOTIQ_PATH="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/Robotiq/2F-85/configuration/Robotiq_2F_85_robot.usd"


ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOTIQ_PATH,
        activate_contact_sensors=False,
        copy_from_source=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    actuators={
        "torso_head": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*",
            ],
            effort_limit=100.0,
            velocity_limit=6.684611,
            stiffness=80.0,
            damping=4.0,
        ),}
)

NEXTAGE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(PARENT_DIR, "assets/nextage_description/urdf/nextage_edit_fixed.usd"),
        activate_contact_sensors=True,
        copy_from_source=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={ 
            # CHEST & HEAD 
            "CHEST_JOINT0":0.0,
            "HEAD_JOINT0":0.0,
            "HEAD_JOINT1":0.0,
            # LARM & RARM
            # "LARM_JOINT0":  -0.90,  # Y
            # "LARM_JOINT1":  0.93, # Y 
            # "LARM_JOINT2":  -0.80, # X
            # "LARM_JOINT3":  -3.14,  # Y
            # "LARM_JOINT4":  0.15,# Z  
            # "LARM_JOINT5":  -0.97,  # Z
            # "RARM_JOINT0":  0.90,
            # "RARM_JOINT1":  0.93,
            # "RARM_JOINT2":  -0.80,
            # "RARM_JOINT3":  3.14,
            # "RARM_JOINT4":  0.15, 
            # "RARM_JOINT5":  0.97,
            "LARM_JOINT0":  -1.0,  # Y
            "LARM_JOINT1":  0.90, # Y 
            "LARM_JOINT2":  -0.70, # X
            "LARM_JOINT3":  -3.14,  # Y
            "LARM_JOINT4":  0.18,# Z  
            "LARM_JOINT5":  -0.97,  # Z
            "RARM_JOINT0":  1.0,
            "RARM_JOINT1":  0.90,
            "RARM_JOINT2":  -0.70,
            "RARM_JOINT3":  3.14,
            "RARM_JOINT4":  0.18, 
            "RARM_JOINT5":  0.97,
            # gripper
            "left_inner_finger_joint_L" : 0.0,
            "right_inner_finger_joint_L" : 0.0,
            "right_outer_knuckle_joint_L" : 0.0,
            "finger_joint_L" : 0.0,
            
            "left_inner_finger_joint" : 0.0,
            "right_inner_finger_joint" : 0.0,
            "right_outer_knuckle_joint" : 0.0,
            "finger_joint" : 0.0
        },
    ),

    actuators={
        "torso_head": ImplicitActuatorCfg(
            joint_names_expr=[
                "CHEST_JOINT0",
                "HEAD_JOINT0",
                "HEAD_JOINT1",
            ],
            effort_limit=100.0,
            # velocity_limit=6.684611,
            stiffness=80.0,
            damping=4.0,
            velocity_limit_sim={
                "CHEST_JOINT0": 6.684611 * OVERRIDE_SCALE,
                "HEAD_JOINT0":  6.684611 * OVERRIDE_SCALE,
                "HEAD_JOINT1":  6.684611 * OVERRIDE_SCALE
            }
        ),

        "larm_proximal": ImplicitActuatorCfg(
            joint_names_expr=["LARM_JOINT[0-2]"],
            effort_limit=150.0,
            # velocity_limit=2.426008,
            stiffness=80.0,
            damping=4.0,
            velocity_limit_sim={
                "LARM_JOINT0":  2.426008 * OVERRIDE_SCALE, 
                "LARM_JOINT1":  2.426008 * OVERRIDE_SCALE, 
                "LARM_JOINT2":  2.426008 * OVERRIDE_SCALE
            }
        ),
    
        "larm_distal": ImplicitActuatorCfg(
            joint_names_expr=["LARM_JOINT[3-5]"],
            effort_limit=150.0,
            # velocity_limit=6.352998,
            stiffness=80.0,
            damping=4.0,
            velocity_limit_sim={
                "LARM_JOINT3":  6.352998 * OVERRIDE_SCALE, 
                "LARM_JOINT4":  6.352998 * OVERRIDE_SCALE, 
                "LARM_JOINT5":  6.352998 * OVERRIDE_SCALE
            }
        ),

        "rarm_proximal": ImplicitActuatorCfg(
            joint_names_expr=["RARM_JOINT[0-2]"],
            effort_limit=150.0,
            # velocity_limit=2.426008, 
            stiffness=80.0,
            damping=4.0,
            velocity_limit_sim={
                "RARM_JOINT0":  2.426008 * OVERRIDE_SCALE, 
                "RARM_JOINT1":  2.426008 * OVERRIDE_SCALE, 
                "RARM_JOINT2":  2.426008 * OVERRIDE_SCALE
            }
        ),
  
        "rarm_distal": ImplicitActuatorCfg(
            joint_names_expr=["RARM_JOINT[3-5]"],
            effort_limit=150.0,
            # velocity_limit=6.352998,
            stiffness=80.0,
            damping=4.0,
            velocity_limit_sim={
                "RARM_JOINT3":  6.352998 * OVERRIDE_SCALE, 
                "RARM_JOINT4":  6.352998 * OVERRIDE_SCALE, 
                "RARM_JOINT5":  6.352998 * OVERRIDE_SCALE
            }
        ),

        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_outer_knuckle_joint_L", "finger_joint_L"],
            # joint_names_expr=["finger_joint_L"],
            velocity_limit=0.2,        
            effort_limit=200.0,
            stiffness=2000.0,
            damping=2000.0,
            # velocity_limit_sim={
            #     "right_outer_knuckle_joint_L": 0.2 * OVERRIDE_SCALE, 
            #     "finger_joint_L":  0.2 * OVERRIDE_SCALE
            # }
        ),

        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_outer_knuckle_joint", "finger_joint"],
            # joint_names_expr=["finger_joint"],
            velocity_limit=0.2,        
            effort_limit=200.0,
            stiffness=2000.0,
            damping=2000.0,
            # velocity_limit_sim={
            #     "right_outer_knuckle_joint": 0.2 * OVERRIDE_SCALE, 
            #     "finger_joint":  0.2 * OVERRIDE_SCALE
            # }
        ),


        # "left_gripper": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "finger_joint_L",
        #         "right_inner_knuckle_joint_L", 
        #         "right_inner_finger_joint_L",
        #         "left_inner_knuckle_joint_L",
        #         "left_inner_finger_joint_L",
        #         "right_outer_knuckle_joint_L",
        #         # "left_outer_knuckle_joint_L",
        #     ],
        #     effort_limit=100.0,
        #     velocity_limit=1.0,
        #     stiffness=6000.0,
        #     damping=300.0,
        # ),
        # "right_gripper": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "finger_joint", 
        #         "right_inner_knuckle_joint",
        #         "right_inner_finger_joint",
        #         "left_inner_knuckle_joint",
        #         "left_inner_finger_joint",
        #         "right_outer_knuckle_joint",
        #         # "left_outer_knuckle_joint",
        #     ],
        #     effort_limit=100.0,
        #     velocity_limit=1.0,
        #     stiffness=6000.0,
        #     damping=300.0,
        # ),
    },

    soft_joint_pos_limit_factor=1.0,
)

# NEXTAGE_HIGH_PD_CFG = NEXTAGE_CFG.copy()
# for k in ["torso_head","larm_proximal", "larm_distal", "rarm_proximal", "rarm_distal"]:
#     NEXTAGE_HIGH_PD_CFG.actuators[k].stiffness = 400.0
#     NEXTAGE_HIGH_PD_CFG.actuators[k].damping   = 80.0

