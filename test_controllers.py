#!/usr/bin/env python3
"""
Script to Compare Joint-Space vs Task-Space Controllers

This script runs the AIREC environment with both controller types
and visualizes the differences.

Usage:
    python test_controllers.py --task AIREC_Wear_TaskSpace --render
"""

import torch
import argparse
from pathlib import Path

import gymnasium as gym
from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationContext


def main(args):
    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Make environment
    env = gym.make(
        args.task,
        cfg=None,  # Uses default config from registered environment
        render_mode="rgb_array"
    )
    
    print(f"\n{'='*60}")
    print(f"Environment: {args.task}")
    print(f"{'='*60}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Number of Envs: {env.unwrapped.num_envs}")
    print(f"\nController Type: ", end="")
    
    # Identify controller type
    if hasattr(env.unwrapped, 'cfg') and hasattr(env.unwrapped.cfg, 'controller_type'):
        print(env.unwrapped.cfg.controller_type)
    elif hasattr(env.unwrapped, 'ik_controller_right'):
        print("Task-Space (IK-based)")
    else:
        print("Joint-Space (Direct)")
    print(f"{'='*60}\n")
    
    # Test actions
    obs, _ = env.reset()
    
    num_steps = 100
    total_reward = 0.0
    
    print("Testing for 100 steps...")
    print(f"Step | Reward | Action Norm | Max Action")
    print("-" * 50)
    
    for step in range(num_steps):
        # Send random actions
        actions = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward.sum()
        
        action_norm = torch.norm(torch.tensor(actions), dim=-1).mean()
        action_max = torch.abs(torch.tensor(actions)).max()
        
        if step % 20 == 0:
            print(f"{step:4d} | {reward.mean():6.3f} | {action_norm:10.4f} | {action_max:9.4f}")
        
        if terminated.any() or truncated.any():
            print(f"\nEpisode completed at step {step}")
            break
    
    print(f"\n{'='*60}")
    print(f"Average Reward: {total_reward / num_steps:.4f}")
    print(f"{'='*60}\n")
    
    env.close()
    print("Test completed successfully!")


def compare_controllers():
    """Compare both controller types side-by-side."""
    
    print("\n" + "="*70)
    print("CONTROLLER COMPARISON")
    print("="*70)
    
    comparison = {
        "Feature": ["Control Input", "Dimensionality", "Computation", "Intuitiveness", "Stability"],
        "Joint-Space": [
            "Joint angles",
            "14D (AIREC)",
            "~1ms per step",
            "Low (raw angles)",
            "High"
        ],
        "Task-Space": [
            "EE velocity/pose",
            "12D (6D × 2 arms)",
            "~3-5ms per step (IK)",
            "High (end-effector)",
            "Medium (tuning dependent)"
        ]
    }
    
    # Print comparison table
    max_width = max(len(key) for key in comparison["Feature"])
    
    print(f"\n{'':<{max_width}} | {'Joint-Space':<30} | {'Task-Space':<30}")
    print("-" * (max_width + 65))
    
    for i, feature in enumerate(comparison["Feature"]):
        joint = comparison["Joint-Space"][i]
        task = comparison["Task-Space"][i]
        print(f"{feature:<{max_width}} | {joint:<30} | {task:<30}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test and compare different controller types"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="AIREC_Wear_TaskSpace",
        choices=["AIREC_Wear_Joint", "AIREC_Wear_Task", "AIREC_Wear_TaskSpace"],
        help="Task to test"
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    
    # Append AppLauncher args
    AppLauncher.add_app_launcher_args(parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show comparison
    compare_controllers()
    
    # Run test
    main(args)
