# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg

PHYSICS_DT = 1 / 240
DECIMATION = 8
SLEEP_THRESHOLD = 0.0
STABILIZATION_THRESHOLD = 0.002
MAX_DEPENETRATION_VELOCITY = 5.0

# High-precision contact for dexterous manipulation
contact_props = CollisionPropertiesCfg(
    collision_enabled=True,
    
    # 1. Eliminate the Air Gap
    # rest_offset = 0.0 ensures the ball actually touches the bed/fingers.
    # contact_offset = 0.002 (2mm) gives the solver enough "warning" to 
    # prepare for the collision without being too computationally expensive.
    contact_offset=0.002,
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
    
    # 2. Iterations: Higher counts prevent fingers from 'sinking' into objects 
    # and improve the stability of grasps.
    min_position_iteration_count=8,
    max_position_iteration_count=255,
    min_velocity_iteration_count=1,
    max_velocity_iteration_count=255,
    
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
    # gpu_total_aggregate_pairs_capacity=2**25,
    # gpu_found_lost_aggregate_pairs_capacity=2**25,
    # gpu_found_lost_pairs_capacity=2**27,
    # gpu_max_rigid_patch_count=2**19,
    # gpu_max_rigid_contact_count=2**20, # Increase this too!
    # gpu_heap_capacity=2**26,

    # Refined GPU Buffer Management
    gpu_max_rigid_patch_count=2**19,       # Covers your ~233k requirement with 100% margin
    gpu_max_rigid_contact_count=2**22,     # Increased to 4M to support high-contact density
    gpu_heap_capacity=2**27,               # Bumped slightly to support the contact increase
    gpu_total_aggregate_pairs_capacity=2**21, # Still massive, but more reasonable
    gpu_found_lost_pairs_capacity=2**21,
)

bed_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.5, 
    dynamic_friction=0.4, 
    restitution=0.0
)

object_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0, 
    dynamic_friction=0.8, 
    restitution=0.0,
)

robot_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.5,
    dynamic_friction=1.2, 
    restitution=0.0      
)


object_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=16, 
    solver_velocity_iteration_count=8,
    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
)

sphere_mass_props = sim_utils.MassPropertiesCfg(mass=2)
catepillar_mass_props = sim_utils.MassPropertiesCfg(mass=100)
human_mass_props = sim_utils.MassPropertiesCfg(mass=10)

bed_rigid_props = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    solver_position_iteration_count=8, 
    solver_velocity_iteration_count=4,
    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
)

object_articulation_settings = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=16, # Higher for stability
    solver_velocity_iteration_count=8, # Higher for stability
    sleep_threshold=SLEEP_THRESHOLD,
    stabilization_threshold=STABILIZATION_THRESHOLD,
)


# this will apply to the robot since I can't specify it in the AIREC_CFG
airec_sim_cfg = SimulationCfg(
    dt=PHYSICS_DT,
    render_interval=DECIMATION,
    physics_material=robot_material,
    physx=physx_optimized
)