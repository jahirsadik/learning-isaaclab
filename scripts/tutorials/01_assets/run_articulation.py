# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.sim.utils.prims as prim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG  # isort:skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    
    # # Origin 1
    # prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # # Origin 2
    # prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    for i in range(len(origins)):
        # Import cartpole USD into each origin group
        prim_utils.create_prim(
            f"/World/Origin{i+1}",
            "Xform",
            translation=origins[i],
        )

    # Articulation
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    origins.append([4.0, 2.0, 0.0])
    prim_utils.create_prim(
            f"/World/Origin3",
            "Xform",
            translation=origins[2],
        )
    
    jetbot_cfg = JETBOT_CONFIG.copy()
    jetbot_cfg.prim_path = "/World/Origin3/Jetbot"
    jetbot = Articulation(cfg=jetbot_cfg)

    # return the scene information
    scene_entities = {"cartpole": cartpole, "jetbot": jetbot}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["cartpole"]
    jetbot = entities["jetbot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    sim_time = 0.0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if sim_time > 30:
            # reset counter
            print(f"Simulation time: {sim_time:.2f}s. Resetting robot states...")
            sim_time = 0.0
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins[:2]
            root_state[:, 4:7] += torch.rand_like(root_state[:, 4:7]) * 0.2 - 0.1  # small orientation noise
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # jetbot 
            jb_root_state = jetbot.data.default_root_state.clone()
            jb_root_state[:, :3] += origins[2]
            jetbot.write_root_pose_to_sim(jb_root_state[:, :7])
            jetbot.write_root_velocity_to_sim(jb_root_state[:, 7:])
            jetbot.write_joint_state_to_sim(jetbot.data.default_joint_pos, jetbot.data.default_joint_vel)
            jetbot.reset()
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # Initialize velocity tensor
        jetbot_joint_vel = torch.zeros_like(jetbot.data.joint_pos)

        # Set different speeds for the two wheels to create an arc
        # Index 0 is usually Left Wheel, Index 1 is Right Wheel
        # Example: Left=10 rad/s, Right=15 rad/s -> Robot curves to the Left
        jetbot_joint_vel[:, 0] = 10.0 
        jetbot_joint_vel[:, 1] = 15.0 

        # Apply the velocity target (Make sure to pass jetbot_joint_vel, not jetbot_efforts)
        jetbot.set_joint_velocity_target(jetbot_joint_vel)
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        robot.write_data_to_sim()
        jetbot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        sim_time += sim_dt
        print(f"Sim Time: {sim_time:.2f}s", end="\r")
        # Update buffers
        robot.update(sim_dt)
        jetbot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
