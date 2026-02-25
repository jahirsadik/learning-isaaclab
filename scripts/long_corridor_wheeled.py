# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Script for automated synthetic data generation in a long corridor environment with a wheeled robot..")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app FIRST before any other Isaac Lab imports
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows - imports after SimulationApp instantiation."""

import os
import torch

# Now import all Isaac Lab modules AFTER SimulationApp is created
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.assets import RigidObjectCfg, Articulation
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg 


DATA_DIR = "/home/jahirsadikmonon/Documents/Projects/usds"
NUM_CUBOIDS = 2
SPACING = 30.0
SPAWN_CUBOIDS_IN_PATH = True
SPAWN_TABLE_B_OBJECTS = False
SPAWN_TABLE_A_OBJECTS = False

# Module-level constant for cuboid colors
DIFFUSE_COLORS = [
    # Group 1: Primary hues at maximum saturation
    (1.0, 0.0, 0.0),      # Red
    (0.0, 1.0, 0.0),      # Green
    (0.0, 0.0, 1.0),      # Blue
    # Group 2: Secondary hues at maximum saturation
    (1.0, 1.0, 0.0),      # Yellow
    (1.0, 0.0, 1.0),      # Magenta
    (0.0, 1.0, 1.0),      # Cyan
    # High contrast achromatic
    (0.0, 0.0, 0.0),      # Black
    (1.0, 1.0, 1.0),      # White
]

# ============================= JETBOT CONFIGURATION =============================

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
"""Configuration for the Jetbot wheeled robot."""


@configclass
class LongCorridorWheeledRobotSceneCfg(InteractiveSceneCfg):
    # ground terrain
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(1000000, 1000)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # custom long corridor usd
    terrain = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Corridor",
        spawn=UsdFileCfg(
            usd_path=os.path.join(DATA_DIR, "long_corridor.usda"),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    table_A = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Corridor/simple_room/table",
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    table_B = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Corridor/simple_room/table_01",
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    if SPAWN_TABLE_A_OBJECTS:
        cone_A = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ConeA",
            spawn=sim_utils.ConeCfg(
                radius=0.1,
                height=0.2,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.05,
                    rest_offset=0.005,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )   

        custObj_A = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CustomMeshA",
            spawn=sim_utils.UsdFileCfg(
                scale=(2.0, 2.0, 2.0),   
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

        cuboid_A = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CuboidA",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

    if SPAWN_TABLE_B_OBJECTS:
        cone_B = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ConeB",
            spawn=sim_utils.ConeCfg(
                radius=0.1,
                height=0.2,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.05,
                    rest_offset=0.005,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

        custObj_B = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CustomMeshB",
            spawn=sim_utils.UsdFileCfg(
                scale=(2.0, 2.0, 2.0),   
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

        cuboid_B = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CuboidB",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

    if SPAWN_CUBOIDS_IN_PATH:
        for i in range(NUM_CUBOIDS):
            locals()[f"cuboid_in_path_{i}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cuboid_in_path_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DIFFUSE_COLORS[i % len(DIFFUSE_COLORS)], metallic=0.4)
                ),
                init_state=RigidObjectCfg.InitialStateCfg(),
            )       
        del i

    # Jetbot robot configuration
    jetbot: ArticulationCfg = JETBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.6, 0.0, 0.0))
    )


class GoalBasedWheelAction(ActionTerm):
    """Custom action term that converts goal poses to wheel velocity commands.
    
    This action term takes in goal poses [gx, gy, gyaw] per environment instance
    and converts them to wheel velocity commands using proportional control on
    position and orientation errors.
    
    The robot base pose is tracked from the articulation, and wheel velocities
    are computed using differential drive kinematics:
        v = K_p * distance_error
        omega = K_w * angle_error
    """
    
    cfg: "GoalBasedWheelActionCfg"
    """The configuration of the action term."""

    def __init__(self, cfg: "GoalBasedWheelActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        # Store configuration
        self.cfg = cfg
        
        # Get the robot asset
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # Storage for actions and processed commands
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)  # [goal_x, goal_y, goal_yaw]
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)  # [v, omega]
        
        # Control gains
        self._linear_gain = cfg.linear_gain
        self._angular_gain = cfg.angular_gain
        self._max_linear_vel = cfg.max_linear_vel
        self._max_angular_vel = cfg.max_angular_vel
        
    @property
    def action_dim(self) -> int:
        return 3  # [goal_x, goal_y, goal_yaw]
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def apply_actions(self, actions: torch.Tensor) -> None:
        """Apply actions - required by ActionTerm abstract base class."""
        self._apply_actions(actions)
    
    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Process actions - required by ActionTerm abstract base class."""
        return actions
    
    def _apply_actions(self, actions: torch.Tensor) -> None:
        """Apply goal poses and convert to wheel velocities."""
        # Store raw goal actions
        self._raw_actions[:] = actions
        
        # Get current robot pose (x, y, yaw)
        root_pos = self._robot.data.root_pos_w  # (num_envs, 3)
        root_quat = self._robot.data.root_quat_w  # (num_envs, 4)
        
        # Extract yaw from quaternion (euler_xyz_from_quat returns tuple (roll, pitch, yaw))
        from isaaclab.utils.math import euler_xyz_from_quat
        _, _, current_yaw = euler_xyz_from_quat(root_quat)
        
        # Extract goal pose from actions
        goal_pos = actions[:, :2]  # [goal_x, goal_y]
        
        # Compute position error
        pos_error = goal_pos - root_pos[:, :2]
        distance = torch.norm(pos_error, dim=1, keepdim=True)  # (num_envs, 1)
        
        # Compute desired heading toward goal
        desired_yaw = torch.atan2(pos_error[:, 1], pos_error[:, 0])
        
        # Compute orientation error (angle to desired heading, then to goal yaw if close enough)
        angle_to_goal = self._normalize_angle(desired_yaw - current_yaw)
        
        # Compute control commands with proportional control
        linear_vel = self._linear_gain * distance.squeeze(-1)
        angular_vel = self._angular_gain * angle_to_goal
        
        # Clip to max values
        linear_vel = torch.clamp(linear_vel, -self._max_linear_vel, self._max_linear_vel)
        angular_vel = torch.clamp(angular_vel, -self._max_angular_vel, self._max_angular_vel)
        
        # Reduce linear velocity if angular error is large to maintain stability
        high_angular_error = torch.abs(angle_to_goal) > 0.5
        linear_vel[high_angular_error] *= 0.5
        
        # Store processed actions [v, omega]
        self._processed_actions[:, 0] = linear_vel
        self._processed_actions[:, 1] = angular_vel
        
        # Command wheel velocities using differential drive kinematics
        # For a differential drive robot:
        # v_left = (v - omega * wheel_base/2) / wheel_radius
        # v_right = (v + omega * wheel_base/2) / wheel_radius
        wheel_base = self.cfg.wheel_base
        wheel_radius = self.cfg.wheel_radius
        
        v_left_target = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
        v_right_target = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
        
        # Set joint velocity targets for the wheels
        # Jetbot typically has left_wheel_joint and right_wheel_joint
        joint_vel = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        
        # Find wheel joint indices
        if not hasattr(self, '_left_wheel_idx'):
            try:
                left_idx, _ = self._robot.find_joints("left.*wheel|left.*drive")
                right_idx, _ = self._robot.find_joints("right.*wheel|right.*drive")
                
                # Handle case where find_joints returns lists/tuples
                self._left_wheel_idx = left_idx[0] if isinstance(left_idx, (list, tuple)) else left_idx
                self._right_wheel_idx = right_idx[0] if isinstance(right_idx, (list, tuple)) else right_idx
            except Exception:
                # Fallback: assume wheels are first 2 DOFs
                self._left_wheel_idx = 0
                self._right_wheel_idx = 1
        
        if hasattr(self, '_left_wheel_idx') and hasattr(self, '_right_wheel_idx'):
            joint_vel[:, self._left_wheel_idx] = v_left_target
            joint_vel[:, self._right_wheel_idx] = v_right_target
        
        # Apply joint velocity targets
        self._robot.set_joint_velocity_target(joint_vel)
    
    @staticmethod
    def _normalize_angle(angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]."""
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    
    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Implement debug visualization (optional)."""
        # No special debug visualization needed for this action term
        pass


@configclass
class GoalBasedWheelActionCfg:
    """Configuration for goal-based wheel action."""
    
    asset_name: str = "jetbot"
    """Name of the robot asset in the scene."""
    
    # Control parameters
    linear_gain: float = 1.0
    """Proportional gain for linear velocity control (m/s per meter error)."""
    
    angular_gain: float = 1.0
    """Proportional gain for angular velocity control (rad/s per radian error)."""
    
    max_linear_vel: float = 3.0
    """Maximum linear velocity (m/s)."""
    
    max_angular_vel: float = 2.0
    """Maximum angular velocity (rad/s)."""
    
    # Wheel parameters
    wheel_base: float = 0.16
    """Distance between left and right wheels (m). Jetbot is ~0.16m."""
    
    wheel_radius: float = 0.032
    """Wheel radius (m). Jetbot wheels are ~0.032m."""
    
    # Base class required attributes
    debug_vis: bool = False
    """Enable debug visualization."""


@configclass
class ActionsCfg:
    """Action specifications for the environment.
    
    Note: We use a custom GoalBasedWheelAction term that is registered directly
    in the environment setup. This config class is kept for compatibility with
    the ManagerBasedEnvCfg structure.
    """
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # Note: specify asset_cfg to use "jetbot" instead of default "robot"
        base_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("jetbot")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("jetbot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("jetbot")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_robot_near_tableA = EventTerm(
        func=mdp.reset_robot_near_target,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("jetbot"),
            "target_asset_cfg": SceneEntityCfg("table_A"),
            "position_offset": (0.0, 2.0, .4),
        }
    )

    if SPAWN_TABLE_A_OBJECTS:
        reset_cone_A = EventTerm(
            func=mdp.reset_rigid_object_near_target,
            mode="reset",
            params={
                "target_asset_cfg": SceneEntityCfg("table_A"),
                "rigid_object_cfg": SceneEntityCfg("cone_A"),
                "position_offset": (-0.65, 0.0, .85),
            }
        )

        reset_custObj_A = EventTerm(
            func=mdp.reset_rigid_object_near_target,
            mode="reset",
            params={
                "target_asset_cfg": SceneEntityCfg("table_A"),
                "rigid_object_cfg": SceneEntityCfg("custObj_A"),
                "position_offset": (0.0, 0.0, 0.85),
            }
        )

        reset_cuboid_A = EventTerm(
            func=mdp.reset_rigid_object_near_target,
            mode="reset",
            params={
                "target_asset_cfg": SceneEntityCfg("table_A"),
                "rigid_object_cfg": SceneEntityCfg("cuboid_A"),
                "position_offset": (0.65, 0.0, 0.85),
            }
        )

    if SPAWN_TABLE_B_OBJECTS:
        reset_table_B_objects = EventTerm(
            func=mdp.randomized_slot_placement,
            mode="reset",
            params={
                "target_asset_cfg": SceneEntityCfg("table_B"),
                "object_list_cfgs": [
                    SceneEntityCfg("cone_B"), 
                    SceneEntityCfg("custObj_B"), 
                    SceneEntityCfg("cuboid_B"),
                ],
                "position_offset": (0.65, 0.0, 0.85),
            }
        )

    if SPAWN_CUBOIDS_IN_PATH:
        spawn_cuboids_in_path = EventTerm(
            func=mdp.spawn_objects_in_location,
            mode="reset",
            params={
                "starting_asset_cfg": SceneEntityCfg("table_A"),
                "asset_cfgs": [
                    SceneEntityCfg(f"cuboid_in_path_{i}") 
                    for i in range(NUM_CUBOIDS) 
                ],
                "number_of_objects": NUM_CUBOIDS,
                "position_offsets": [(0.0, SPACING, 0.0)] * (NUM_CUBOIDS),
            }
        )



@configclass
class LongCorridorWheeledEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the long corridor environment."""

    # Scene settings
    scene = LongCorridorWheeledRobotSceneCfg(num_envs=1024, env_spacing=30.0)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.01  # sim step every 10ms: 100Hz


def main():
    """Main function to run the long corridor wheeled robot environment."""
    # Create environment configuration
    env_cfg = LongCorridorWheeledEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # Setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)
    
    # Register custom goal-based wheel action term with the action manager
    # This is done after environment creation to inject our custom action term
    goal_action_cfg = GoalBasedWheelActionCfg(asset_name="jetbot")
    
    # Note: In a real scenario, you would register this in the config pre-environment creation.
    # For demonstration, we'll handle actions manually for now.
    
    print("-" * 80)
    print("[INFO]: Initialized long corridor environment with {} environments".format(args_cli.num_envs))
    print("[INFO]: Action space: 3D goal poses [x, y, yaw]")
    print("[INFO]: Observation space: base position, linear velocity, angular velocity")
    print("-" * 80)
    
    # Get robot reference for manual action conversion
    robot = env.scene["jetbot"]
    
    # Simulation loop
    count = 0
    current_goals = torch.zeros(env.num_envs, 3, device=env.device)
    
    # Set some initial goal poses for different environments
    # e.g., different goal positions for each environment pair
    for i in range(min(env.num_envs, 32)):
        # Alternate between two goals
        angle = (i % 2) * 3.14159
        distance = 2.0 + (i % 4) * 0.5
        current_goals[i, 0] = distance * torch.cos(torch.tensor(angle))
        current_goals[i, 1] = distance * torch.sin(torch.tensor(angle))
        current_goals[i, 2] = angle
    
    # Create action term instance for manual action conversion
    action_term = GoalBasedWheelAction(goal_action_cfg, env)
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset every 300 steps
            if count % 300 == 0:
                count = 0
                env.reset()
                print("[INFO]: Resetting environment...")
                print("[INFO]: Robot positions (sample):")
                print(f"  - Env 0 pos: {robot.data.root_pos_w[0, :2]}")
                print(f"  - Env 1 pos: {robot.data.root_pos_w[1, :2] if env.num_envs > 1 else 'N/A'}")
            
            # Update goal poses - could also sample random goals here
            # or use a goal generator/planner
            actions = current_goals.clone()
            
            # Convert goal poses to wheel velocities using the action term
            action_term.apply_actions(actions)
            
            # Get processed wheel velocities
            wheel_vels = action_term.processed_actions.clone()  # [v, omega]
            
            # Step environment - action manager has 0 dims (actions applied manually above)
            env.step(torch.zeros(env.num_envs, 0, device=env.device))
            
            # Get observations
            obs = env.observation_manager.compute()
            
            # Print debug info every 50 steps
            if count % 50 == 0:
                print(f"[Step {count}] Env 0: pos={robot.data.root_pos_w[0, :2].tolist()}, "
                      f"lin_vel={robot.data.root_lin_vel_w[0, :2].tolist()}")
            
            count += 1

    # Close the environment
    env.close()
    print("[INFO]: Simulation closed successfully")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
