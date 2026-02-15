#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run the Spot robot in the long corridor environment with velocity tracking.

This script demonstrates how to use the Spot robot with the long corridor USD environment
and control it using Isaac Sim's standard velocity commands and controllers.
"""

import argparse

from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run Spot in long corridor environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--play", action="store_true", help="Run in play mode (deterministic, single env)")
parser.add_argument("--video", action="store_true", help="Record video of the simulation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

# Load the environment configuration
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.spot_long_corridor_env_cfg import (
    SpotLongCorridorEnvCfg,
    SpotLongCorridorEnvCfg_PLAY,
)


def main():
    """Main function to run the Spot robot in the long corridor."""

    # Get the configuration
    if args.play:
        cfg = SpotLongCorridorEnvCfg_PLAY()
        cfg.scene.num_envs = 1
    else:
        cfg = SpotLongCorridorEnvCfg()
        cfg.scene.num_envs = args.num_envs

    # Set viewer settings
    if args.headless:
        cfg.viewer.eye = None
    else:
        cfg.viewer.eye = (15.0, 15.0, 5.0)

    # Create simulation context (already initialized by AppLauncher)
    # Note: ManagerBasedRLEnv will create its own SimulationContext, so we don't create one here
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=cfg)

    # Print environment info
    print(f"Environment created with {env.num_envs} parallel environments")
    print(f"Observation shape: {env.observation_manager.group_obs_dim}")
    print(f"Action shape: {env.action_manager.total_action_dim}")

    # Simulation loop
    sim_time = 0.0
    count = 0

    # Reset the environment
    obs, info = env.reset()

    print("\nStarting simulation loop...")
    print("Running for 20 seconds of simulation time per episode")

    while simulation_app.is_running():
        # Reset if episode is done or on initial reset
        with torch.no_grad():
            # Generate random actions (for testing)
            # In a real scenario, you would use a trained policy or manual controller here
            actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device) * 0.5

            # Step the environment
            obs, rewards, dones, truncated, info = env.step(actions)

        # Update time
        sim_time += env.step_dt
        count += 1

        # Print status every 100 steps
        if count % 100 == 0:
            avg_reward = rewards.mean().item()
            print(
                f"Step: {count} | Sim Time: {sim_time:.2f}s | Avg Reward: {avg_reward:.4f} | "
                f"Dones: {dones.sum().item()}/{env.num_envs}"
            )

        # Reset environments that are done
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            # For ManagerBasedRLEnv, call the internal _reset_idx method
            env._reset_idx(done_ids)
            print(f"Reset {done_ids.numel()} environments")

        # Break if we want to end the simulation
        if sim_time > 100.0:  # Run for 100 seconds
            print("Simulation time limit reached")
            break

    print("\nSimulation complete!")

    # Close the environment
    env.close()

    # Shutdown
    simulation_app.close()


if __name__ == "__main__":
    main()
