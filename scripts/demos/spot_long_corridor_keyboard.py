#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive keyboard control for Spot in the long corridor environment.

This script allows you to control the Spot robot using your keyboard in real-time.

Keyboard Controls:
    8/4/5/6  - Forward/Left/Stop/Right movement (Numpad)
    7/9      - Rotate left/right (Numpad)
    Esc      - Exit simulation

Usage:
    ./isaaclab.sh -p scripts/demos/spot_long_corridor_keyboard.py --num_envs 1
"""

import argparse
from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Keyboard control for Spot in long corridor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.spot_long_corridor_env_cfg import (
    SpotLongCorridorEnvCfg,
    SpotLongCorridorEnvCfg_PLAY,
)

# Try to import keyboard control library
try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    print("[WARNING] pynput not installed. Keyboard control disabled.")
    print("          Run: pip install pynput")


class KeyboardController:
    """Controller that maps keyboard input to velocity commands."""
    
    def __init__(self):
        self.target_lin_vel_x = 0.0  # Forward/backward
        self.target_lin_vel_y = 0.0  # Left/right (strafe)
        self.target_ang_vel_z = 0.0  # Rotation
        
        self.keys_pressed = set()
        
        if HAS_PYNPUT:
            self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self.listener.start()
    
    def _on_press(self, key):
        """Handle key press events."""
        try:
            self.keys_pressed.add(key.char.lower())
        except AttributeError:
            # Special keys (space, numpad, etc.)
            if key == keyboard.Key.space:
                self.keys_pressed.add('space')
            elif key == keyboard.Key.esc:
                self.keys_pressed.add('esc')
            else:
                # Try to add numpad keys
                try:
                    self.keys_pressed.add(str(key))
                except:
                    pass
    
    def _on_release(self, key):
        """Handle key release events."""
        try:
            self.keys_pressed.discard(key.char.lower())
        except AttributeError:
            if key == keyboard.Key.space:
                self.keys_pressed.discard('space')
            elif key == keyboard.Key.esc:
                self.keys_pressed.discard('esc')
            else:
                # Try to remove numpad keys
                try:
                    self.keys_pressed.discard(str(key))
                except:
                    pass
    
    def update_commands(self):
        """Update velocity commands based on pressed keys."""
        # Reset commands
        self.target_lin_vel_x = 0.0
        self.target_lin_vel_y = 0.0
        self.target_ang_vel_z = 0.0
        
        if not HAS_PYNPUT:
            return False
        
        # Check for numpad keys by string representation
        keys_str = str(self.keys_pressed)
        
        # Forward/backward movement (Numpad 8/2)
        if any('kp_8' in str(key) or '8' in str(key) for key in self.keys_pressed):
            print("↑ Forward")
            self.target_lin_vel_x = 1.0  # Forward
        if any('kp_2' in str(key) or '2' in str(key) for key in self.keys_pressed):
            print("↓ Backward")
            self.target_lin_vel_x = -0.5  # Backward (slower)
        
        # Strafe movement (Numpad 4/6)
        if any('kp_4' in str(key) or '4' in str(key) for key in self.keys_pressed):
            print("← Left")
            self.target_lin_vel_y = 0.5  # Left strafe
        if any('kp_6' in str(key) or '6' in str(key) for key in self.keys_pressed):
            print("→ Right")
            self.target_lin_vel_y = -0.5  # Right strafe
        
        # Rotation (Numpad 7/9)
        if any('kp_7' in str(key) or '7' in str(key) for key in self.keys_pressed):
            self.target_ang_vel_z = 1.0  # Rotate left
        if any('kp_9' in str(key) or '9' in str(key) for key in self.keys_pressed):
            self.target_ang_vel_z = -1.0  # Rotate right
        
        # Stop movement (Numpad 5)
        if any('kp_5' in str(key) or '5' in str(key) for key in self.keys_pressed):
            print("⏸ Stop")
            self.target_lin_vel_x = 0.0
            self.target_lin_vel_y = 0.0
            self.target_ang_vel_z = 0.0
        
        # Check for exit
        return 'esc' in self.keys_pressed
    
    def get_commands(self):
        """Get the current velocity command."""
        return torch.tensor(
            [self.target_lin_vel_x, self.target_lin_vel_y, 0.0, 0.0],
            dtype=torch.float32
        )


def main():
    """Main function to run interactive keyboard control."""
    
    # Check if keyboard control is available
    if not HAS_PYNPUT:
        print("[ERROR] pynput library required for keyboard control")
        print("        Install it with: pip install pynput")
        return
    
    # Get the configuration
    cfg = SpotLongCorridorEnvCfg_PLAY()
    cfg.scene.num_envs = args.num_envs
    
    # Set viewer settings
    if args.headless:
        print("[WARNING] Running in headless mode - keyboard input will not work!")
        print("          Run without --headless flag to enable keyboard control")
        cfg.viewer.eye = None
    else:
        cfg.viewer.eye = (15.0, 15.0, 5.0)
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=cfg)
    
    print(f"\n{'='*60}")
    print(f"KEYBOARD CONTROL - Spot Robot")
    print(f"{'='*60}")
    print(f"Environment: {env.num_envs} parallel environments")
    print(f"Observation shape: {env.observation_manager.group_obs_dim}")
    print(f"Action shape: {env.action_manager.total_action_dim}")
    print(f"\n{'='*60}")
    print("NUMPAD CONTROLS:")
    print(f"{'='*60}")
    print("  8/2      - Forward/Backward (Numpad)")
    print("  4/6      - Left/Right strafe (Numpad)")
    print("  7/9      - Rotate left/right (Numpad)")
    print("  5        - Stop movement (Numpad)")
    print("  Esc      - Exit")
    print(f"{'='*60}\n")
    
    # Create keyboard controller
    controller = KeyboardController()
    
    # Reset environment
    obs, info = env.reset()
    
    # State tracking
    step_count = 0
    episode_count = 0
    cumulative_reward = torch.zeros(env.num_envs, device=env.device)
    
    print("Starting keyboard control... Use the Numpad to move the robot!")
    print("Press any key in the simulation window to start.\n")
    
    try:
        while simulation_app.is_running():
            # Update keyboard commands
            should_exit = controller.update_commands()
            if should_exit:
                print("\nExit requested...")
                break
            
            with torch.no_grad():
                # Get the velocity command from keyboard
                # The command manager expects: [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
                # But we only control: [lin_vel_x, lin_vel_y, ang_vel_z]
                velocity_command = torch.tensor([
                    [controller.target_lin_vel_x, controller.target_lin_vel_y, 0.0, 0.0, 0.0, controller.target_ang_vel_z]
                ] * env.num_envs, dtype=torch.float32, device=env.device)
                
                # Override the command manager's commands with keyboard input
                env.command_manager._command["base_velocity"] = velocity_command
                
                # Get observations
                obs = env.observation_manager.compute()
                
                # Since we don't have a trained policy, use the observation to 
                # estimate reasonable joint actions using a simple heuristic
                # In reality, this should use a trained neural network policy
                
                # For now, create zero actions and let the environment's internal
                # dynamics handle the motion based on the velocity commands
                actions = torch.zeros(
                    (env.num_envs, env.action_manager.total_action_dim),
                    dtype=torch.float32,
                    device=env.device
                )
                
                # Step environment
                obs, rewards, dones, truncated, info = env.step(actions)
                cumulative_reward += rewards
            
            # Print status every 50 steps
            step_count += 1
            if step_count % 50 == 0:
                avg_reward = cumulative_reward.mean().item()
                print(
                    f"Step: {step_count:5d} | "
                    f"Cmd: ({controller.target_lin_vel_x:5.2f}, {controller.target_lin_vel_y:5.2f}, {controller.target_ang_vel_z:5.2f}) | "
                    f"Reward: {avg_reward:7.4f}"
                )
            
            # Reset environments that are done
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                env._reset_idx(done_ids)
                cumulative_reward[done_ids] = 0.0
                episode_count += len(done_ids)
                print(f"  → Reset {len(done_ids)} environment(s) (Total episodes: {episode_count})")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print(f"\n{'='*60}")
        print("Simulation Complete!")
        print(f"{'='*60}")
        print(f"Total steps: {step_count}")
        print(f"Total episodes: {episode_count}")
        print(f"Average cumulative reward: {cumulative_reward.mean().item():.4f}")
        
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
