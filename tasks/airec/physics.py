from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg

PHYSICS_DT = 1 / 120
DECIMATION = 4
SLEEP_THRESHOLD = 0.0
STABILIZATION_THRESHOLD = 0.002
MAX_DEPENETRATION_VELOCITY = 5.0
NUM_POSITION_ITERATIONS = 8
MAX_CONTACT_IMPULSE = 1e8
MAX_LINEAR_VELOCITY = 1000.0
MAX_ANGULAR_VELOCITY = 3666.0
########################## glove physx ##########################
GLOVE_PHYSICS_DT = 1 / 120
GLOVE_DECIMATION = 4
########################## bracelet physx ##########################
# Smaller dt reduces tunneling for thin bracelet vs palm (cost ↑). If stable at 1/120, revert.
BRACELET_PHYSICS_DT = 1 / 360
BRACELET_DECIMATION = 6
BRACELET_NUM_POSITION_ITERATIONS = 32
BRACELET_NUM_VELOCITY_ITERATIONS = 16

physx_fast = PhysxCfg(
    solver_type=1,
    min_position_iteration_count=8,
    max_position_iteration_count=BRACELET_NUM_POSITION_ITERATIONS,
    max_velocity_iteration_count=BRACELET_NUM_VELOCITY_ITERATIONS,
    enable_enhanced_determinism=True, 

    friction_offset_threshold=0.01,
    friction_correlation_distance=0.005,
    bounce_threshold_velocity=0.2,

    enable_ccd=True,

    gpu_total_aggregate_pairs_capacity=2**25,
    gpu_found_lost_aggregate_pairs_capacity=2**25,
    gpu_found_lost_pairs_capacity=2**27,
    gpu_max_rigid_patch_count=2**23,
    gpu_max_rigid_contact_count=2**23, # Increase this too!
    gpu_heap_capacity=2**26,
    gpu_max_num_partitions=1,
)

robot_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=NUM_POSITION_ITERATIONS,
    solver_velocity_iteration_count=1,
    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
)

robot_articulation_settings = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=NUM_POSITION_ITERATIONS,
    solver_velocity_iteration_count=1,
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
    # fix_root_link=True,
)

# High-precision contact for dexterous manipulation
contact_props = CollisionPropertiesCfg(
    collision_enabled=True,
    
    # 1. Eliminate the Air Gap
    # rest_offset = 0.0 ensures the ball actually touches the bed/fingers.
    # contact_offset = 0.002 (2mm) gives the solver enough "warning" to 
    # prepare for the collision without being too computationally expensive.
    contact_offset=0.005,
    rest_offset=0.0,
    
    # 2. Add Rotational Resistance (The "Grip")
    # Torsional friction prevents the ball from spinning like a frictionless bearing.
    # A 5mm patch radius is a realistic approximation for a finger-tip or ball compression.
    torsional_patch_radius=0.005, 
    min_torsional_patch_radius=0.001,
)


physx_optimized = PhysxCfg(
    # 1. Solver Type: TGS (1) is significantly more stable for articulations than PGS (0)
    solver_type=1,
    min_position_iteration_count=4,
    max_position_iteration_count=NUM_POSITION_ITERATIONS, # isaac team set 192 for contact-rich
    max_velocity_iteration_count=1,
    
    # 3. Determinism & Stability
    enable_enhanced_determinism=True, # Critical if you are training RL policies
    
    # 4. Friction & Contact: Fine-tuning how friction is applied across surfaces.
    # friction_offset_threshold: Lowering this makes friction kick in sooner.
    # friction_correlation_distance: Merges contacts; 0.005 (5mm) is a good balance for hands.
    friction_offset_threshold=0.01,
    friction_correlation_distance=0.005,
    bounce_threshold_velocity=0.2, # Prevents tiny micro-bounces during sliding
    
    # 5. CCD (Continuous Collision Detection): 
    # Prevents fingers from "tunneling" through thin objects during fast movements.
    enable_ccd=True,
    
    # 6. GPU Buffer Management: 
    # Dexterous hands + complex objects generate many contact points. 
    # We increase these to prevent "Out of Buffer" simulation crashes.
    gpu_total_aggregate_pairs_capacity=2**25,
    gpu_found_lost_aggregate_pairs_capacity=2**25,
    gpu_found_lost_pairs_capacity=2**27,
    gpu_max_rigid_patch_count=2**21,
    gpu_max_rigid_contact_count=2**22,
    gpu_heap_capacity=2**26,
    gpu_max_num_partitions=1,  # Important for stable simulation.

)

STATIC_FRICTION = 1.0
DYNAMIC_FRICTION = 1.0
RESTITUTION = 0.0

bed_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=STATIC_FRICTION,
    dynamic_friction=DYNAMIC_FRICTION,
    restitution=RESTITUTION,
)

chair_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=STATIC_FRICTION,
    dynamic_friction=DYNAMIC_FRICTION,
    restitution=RESTITUTION,
)

object_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=STATIC_FRICTION, 
    dynamic_friction=DYNAMIC_FRICTION, 
    restitution=RESTITUTION,
)

robot_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=1.0, 
    restitution=0.0,    
)

object_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=NUM_POSITION_ITERATIONS, 
    solver_velocity_iteration_count=1,
    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
    max_contact_impulse=MAX_CONTACT_IMPULSE,
    max_linear_velocity=MAX_LINEAR_VELOCITY,
    max_angular_velocity=MAX_ANGULAR_VELOCITY,
)

robot_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=NUM_POSITION_ITERATIONS, 
    solver_velocity_iteration_count=1,
    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
    max_contact_impulse=MAX_CONTACT_IMPULSE,
    max_linear_velocity=MAX_LINEAR_VELOCITY,
    max_angular_velocity=MAX_ANGULAR_VELOCITY,
)

robot_articulation_settings = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=NUM_POSITION_ITERATIONS,
    solver_velocity_iteration_count=1,
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
    fix_root_link=True,
)

sphere_mass_props = sim_utils.MassPropertiesCfg(mass=2)
caterpillar_mass_props = sim_utils.MassPropertiesCfg(mass=10)
human_mass_props = sim_utils.MassPropertiesCfg(mass=5)

bed_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    solver_position_iteration_count=NUM_POSITION_ITERATIONS, 
    solver_velocity_iteration_count=1,
    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
)

object_articulation_settings = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=NUM_POSITION_ITERATIONS, # Higher for stability
    solver_velocity_iteration_count=1, # Higher for stability
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
)

chair_articulation_settings = sim_utils.ArticulationRootPropertiesCfg(
    articulation_enabled=False,
)
chair_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=True,
)

# this will apply to the robot since I can't specify it in the AIREC_CFG
airec_sim_cfg = SimulationCfg(
    dt=PHYSICS_DT,
    render_interval=DECIMATION,
    physics_material=robot_material,
    physx=physx_optimized
)

glove_sim_cfg = SimulationCfg(
    dt=GLOVE_PHYSICS_DT,
    render_interval=GLOVE_DECIMATION,
    physics_material=robot_material,
    physx=physx_optimized
)

bracelet_sim_cfg = SimulationCfg(
    dt=BRACELET_PHYSICS_DT,
    render_interval=BRACELET_DECIMATION,
    physics_material=robot_material,
    physx=physx_fast
)