#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play the Spot robot in the long corridor using trained policies.

This script demonstrates:
1. Loading a trained policy for the Spot robot in the long corridor
2. Running inference with the policy
3. Visualizing the robot's movement
"""

import argparse
import os

from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Play Spot in long corridor with trained policy.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments")
parser.add_argument(
    "--checkpoint", 
    type=str, 
    default=None, 
    help="Path to the trained policy checkpoint (.pt file)"
)
parser.add_argument(
    "--use_play_config", 
    action="store_true", 
    help="Use play configuration (deterministic, fewer envs)"
)
parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of simulation steps to run")
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


def load_policy(policy_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load a trained TorchScript policy.
    
    Args:
        policy_path: Path to the trained policy (.pt file)
        device: Device to load the policy on ('cuda' or 'cpu')
        
    Returns:
        The loaded policy model
    """
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    print(f"Loading policy from: {policy_path}")
    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    return policy


def main():
    """Main function to run Spot in long corridor with a policy."""

    # Get the configuration
    if args.use_play_config:
        cfg = SpotLongCorridorEnvCfg_PLAY()
        cfg.scene.num_envs = min(args.num_envs, 2)
    else:
        cfg = SpotLongCorridorEnvCfg()
        cfg.scene.num_envs = args.num_envs

    # Set device
    cfg.sim.device = args.device

    # Set viewer settings
    if args.headless:
        cfg.viewer.eye = None
    else:
        cfg.viewer.eye = (15.0, 15.0, 5.0)

    # Create simulation context (already initialized by AppLauncher)
    # Note: ManagerBasedRLEnv will create its own SimulationContext, so we don't create one here

    # Create environment
    print("Creating environment...")
    env = ManagerBasedRLEnv(cfg=cfg)

    # Print environment info
    print(f"\nEnvironment Info:")
    print(f"  Number of environments: {env.num_envs}")
    print(f"  Observation dimension: {env.observation_manager.group_obs_dim}")
    print(f"  Action dimension: {env.action_manager.total_action_dim}")
    print(f"  Available commands: {list(env.command_manager.available_commands.keys())}")

    # Load policy if provided
    policy = None
    if args.checkpoint:
        policy = load_policy(args.checkpoint, device=args.device)
        print(f"\nPolicy loaded successfully!")
    else:
        print("\nNo policy checkpoint provided. Using random actions for demonstration.")

    # Reset the environment
    obs, info = env.reset()
    print(f"Environment reset. Initial observation shape: {obs.shape}")

    # Simulation loop
    step_count = 0
    episode_count = 0
    episode_reward = torch.zeros(env.num_envs, device=env.device)

    print(f"\nStarting simulation loop for {args.num_steps} steps...")
    print("Use Ctrl+C to exit early.\n")

    try:
        while simulation_app.is_running() and step_count < args.num_steps:
            with torch.no_grad():
                # Generate actions
                if policy is not None:
                    # Use the trained policy
                    # Extract the observation features expected by the policy
                    # Typically policies expect normalized observations
                    actions = policy(obs)
                else:
                    # Use random actions for demonstration
                    actions = torch.randn(
                        env.num_envs, 
                        env.action_manager.total_action_dim, 
                        device=env.device
                    ) * 0.3

                # Step the environment
                obs, rewards, dones, truncated, info = env.step(actions)

                # Accumulate rewards
                episode_reward += rewards

            step_count += 1

            # Print status every 100 steps
            if step_count % 100 == 0:
                avg_reward = episode_reward.mean().item()
                print(
                    f"Step: {step_count:4d} | Avg Episode Reward: {avg_reward:7.4f} | "
                    f"Episodes Done: {episode_count:3d} | Active Envs: {(~dones).sum().item()}/{env.num_envs}"
                )

            # Reset environments that are done
            if dones.any():
                done_ids = torch.where(dones)[0]
                episode_count += dones.sum().item()
                
                # Log final rewards for completed episodes
                for done_id in done_ids:
                    final_reward = episode_reward[done_id].item()
                    print(f"  → Env {done_id}: Episode complete. Final reward: {final_reward:.4f}")
                
                # Reset episode reward for done environments
                episode_reward[done_ids] = 0.0
                
                # Reset the environments
                obs[done_ids] = env.reset_idx(done_ids)[0][done_ids]

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    print(f"\n{'='*60}")
    print(f"Simulation Summary:")
    print(f"  Total steps executed: {step_count}")
    print(f"  Total episodes completed: {episode_count}")
    if step_count > 0:
        print(f"  Average steps per episode: {step_count / max(episode_count, 1):.2f}")
    print(f"{'='*60}")

    # Close the environment
    env.close()

    # Shutdown
    simulation_app.close()


if __name__ == "__main__":
    main()
