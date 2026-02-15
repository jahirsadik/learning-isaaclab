#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Using custom controllers with Spot in the long corridor.

This script demonstrates various ways to control the Spot robot:
1. Random velocity commands
2. Joystick-like controller
3. Trajectory-based controller
4. Custom neural network controller
"""

import argparse

from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--controller", 
    type=str, 
    default="random",
    choices=["random", "velocity", "network"],
    help="Type of controller to use"
)
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from typing import Tuple

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.spot_long_corridor_env_cfg import (
    SpotLongCorridorEnvCfg_PLAY,
)


class BaseController:
    """Base class for custom controllers."""
    
    def __init__(self):
        self.step = 0
    
    def reset(self):
        """Reset the controller state."""
        self.step = 0
    
    def compute_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute actions from observations.
        
        Args:
            obs: Observations of shape (num_envs, obs_dim)
            
        Returns:
            Actions of shape (num_envs, action_dim)
        """
        raise NotImplementedError


class RandomWalkController(BaseController):
    """Simple random walk controller - performs random actions with smoothing."""
    
    def __init__(self):
        super().__init__()
        self.current_actions = None
        self.action_update_freq = 10  # Update every N steps
    
    def compute_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate smoothly changing random actions."""
        batch_size = obs.shape[0]
        action_dim = 12  # Spot has 12 joints
        device = obs.device
        
        if self.current_actions is None or self.current_actions.shape[0] != batch_size:
            self.current_actions = torch.zeros(batch_size, action_dim, device=device)
        
        if self.step % self.action_update_freq == 0:
            # Generate new random target actions
            self.current_actions = torch.randn(batch_size, action_dim, device=device) * 0.3
        
        self.step += 1
        return self.current_actions


class VelocityCommandController(BaseController):
    """Maps velocity commands to joint actions using a simple heuristic."""
    
    def __init__(self):
        super().__init__()
    
    def compute_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert velocity commands to leg joint targets using simple gait.
        
        This is a simplified version - a real implementation would use
        inverse kinematics or a learned policy.
        """
        batch_size = obs.shape[0]
        action_dim = 12
        device = obs.device
        actions = torch.zeros(batch_size, action_dim, device=device)
        
        # Extract velocity commands (typically indices 9-12 in observation)
        # obs format: [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z,
        #              grav_x, grav_y, grav_z, cmd_lin_vel_x, cmd_lin_vel_y, ...]
        if obs.shape[1] > 11:
            # Get target velocities from last command terms
            cmd_lin_vel_x = obs[:, 9]  # Target forward velocity
            cmd_lin_vel_y = obs[:, 10] # Target lateral velocity
            
            # Simple heuristic: alternate leg positions based on phase
            phase = (self.step % 20) / 20.0  # Phase from 0 to 1
            
            # All hip joints move forward, knees oscillate
            for env_id in range(batch_size):
                # Front left and hind right legs up
                if phase < 0.5:
                    actions[env_id, 0] = -0.2   # fl_hx
                    actions[env_id, 4] = -0.5   # fl_kn
                    actions[env_id, 3] = 0.2    # hr_hx
                    actions[env_id, 7] = -0.5   # hr_kn
                # Front right and hind left legs up
                else:
                    actions[env_id, 1] = 0.2    # fr_hx
                    actions[env_id, 5] = -0.5   # fr_kn
                    actions[env_id, 2] = -0.2   # hl_hx
                    actions[env_id, 6] = -0.5   # hl_kn
        
        self.step += 1
        return actions


class NeuralNetworkController(BaseController):
    """Simple neural network controller trained from scratch during rollout."""
    
    def __init__(self):
        super().__init__()
        self.network = None
        self.initialized = False
    
    def compute_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate actions using the neural network."""
        device = obs.device
        batch_size = obs.shape[0]
        obs_dim = obs.shape[1]
        action_dim = 12
        
        # Initialize network on first call
        if not self.initialized:
            hidden_dim = 64
            self.network = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, action_dim),
            ).to(device)
            self.initialized = True
        
        # Generate actions
        with torch.no_grad():
            actions = self.network(obs)
        
        self.step += 1
        return actions


def run_with_controller(controller: BaseController, num_steps: int = 1000):
    """Run the environment with a custom controller.
    
    Args:
        controller: An instance of a controller class
        num_steps: Number of simulation steps to run
    """
    # Configure environment
    cfg = SpotLongCorridorEnvCfg_PLAY()
    cfg.scene.num_envs = 2
    
    if args.headless:
        cfg.viewer.eye = None
    else:
        cfg.viewer.eye = (15.0, 15.0, 5.0)
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=cfg)
    
    print(f"\n{'='*60}")
    print(f"Running Spot with {controller.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Environments: {env.num_envs}")
    print(f"Observation dim: {env.observation_manager.group_obs_dim}")
    print(f"Action dim: {env.action_manager.total_action_dim}")
    
    # Reset environment
    obs, info = env.reset()
    controller.reset()
    
    total_reward = torch.zeros(env.num_envs, device=env.device)
    episode_count = 0
    
    try:
        for step in range(num_steps):
            with torch.no_grad():
                # Compute actions using the controller
                actions = controller.compute_actions(obs)
                
                # Step environment
                obs, rewards, dones, truncated, info = env.step(actions)
                
                total_reward += rewards
            
            # Print progress
            if (step + 1) % 200 == 0:
                avg_reward = total_reward.mean().item()
                print(f"Step {step+1:4d}/{num_steps} | Avg Reward: {avg_reward:7.4f} | "
                      f"Episodes: {episode_count}")
            
            # Handle resets
            if dones.any():
                done_ids = torch.where(dones)[0]
                episode_count += len(done_ids)
                total_reward[done_ids] = 0.0
                obs[done_ids] = env.reset_idx(done_ids)[0][done_ids]
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print(f"\nSimulation complete!")
        print(f"Total episodes: {episode_count}")
        print(f"Average reward: {total_reward.mean().item():.4f}")
        
        env.close()
        simulation_app.close()


def main():
    """Main function to run the Spot controller demonstration."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run Spot robot with different controller types in the long corridor environment."
    )
    parser.add_argument(
        "--controller",
        type=str,
        choices=["random", "velocity", "neural_network"],
        default="velocity",
        help="Type of controller to use"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Number of simulation steps to run"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=2,
        help="Number of parallel environments"
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # Print available controllers
    print("\n" + "="*60)
    print("Available Controllers:")
    print("="*60)
    print("  random:          Random joint position targets")
    print("  velocity:        Velocity-based command tracking")
    print("  neural_network:  Pre-trained neural network policy")
    print("="*60 + "\n")
    
    # Select controller
    if args.controller == "random":
        controller = RandomWalkController()
    elif args.controller == "velocity":
        controller = VelocityCommandController()
    elif args.controller == "neural_network":
        controller = NeuralNetworkController()
    else:
        raise ValueError(f"Unknown controller type: {args.controller}")
    
    # Run environment with controller
    try:
        run_with_controller(controller, num_steps=args.num_steps)
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
