# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation, RigidObject
import torch


##
# Scene definition
##
import os
from isaaclab.sim import UsdFileCfg

# Commands
from dataclasses import dataclass
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.math import wrap_to_pi, quat_rotate_inverse, yaw_quat

DATA_DIR = "/home/jahirsadikmonon/Documents/Projects/usds"
NUM_CUBOIDS = 13
SPACING = 7.0
SPAWN_CUBOIDS_IN_PATH = True
SPAWN_TABLE_B_OBJECTS = False
SPAWN_TABLE_A_OBJECTS = False

# Module-level constant for cuboid colors
DIFFUSE_COLORS = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 1.0, 0.0),  # Yellow
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (0.5, 0.5, 0.5),  # Gray
    (1.0, 0.5, 0.0),  # Orange
    (0.5, 0.0, 0.5),  # Purple
    (0.0, 0.5, 0.5),  # Teal
    (0.5, 0.5, 0.0),  # Olive
    (0.5, 0.0, 0.0),  # Maroon
    (0.0, 0.5, 0.0),  # Dark Green
]

@configclass
class LongCorridorCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(1000000, 1000)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    print("Current directory:", DATA_DIR)
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
                    kinematic_enabled=False,  # Make sure it's NOT kinematic
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.05, # Increase this
                    rest_offset=0.005,    # Add a tiny gap
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
                kinematic_enabled=False,  # Make sure it's NOT kinematic
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.05, # Increase this
                rest_offset=0.005,    # Add a tiny gap
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
        for i in range(NUM_CUBOIDS):  # or range(NUM_CUBOIDS)
            locals()[f"cuboid_in_path_{i}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cuboid_in_path_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DIFFUSE_COLORS[i % len(DIFFUSE_COLORS)], metallic=0.1)
                ),
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
        
        # !!! CRITICAL FIX: Delete the loop variable so it doesn't become a class attribute !!!
        del i

    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=750.0,
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )


##
# MDP settings
##

# my_project/mdp/patrol_logic.py

# @dataclass
# class PatrolCommandCfg(CommandTermCfg):
#     """Configuration for the patrol command."""
#     class_type = None 
#     resampling_time_range = (0.0, 0.0) 
#     debug_vis = True
    
#     # CHANGED: We now ask for the keys in the scene, not the coordinates
#     asset_name_a: str = "table_A"
#     asset_name_b: str = "table_B"
    
#     waypoint_threshold: float = 1.0
#     walk_speed: float = 0.6
#     turn_speed: float = 1.0


# class PatrolCommand(CommandTerm):
#     """
#     Generates velocity commands to drive the robot between Table A and B.
#     """
#     cfg: PatrolCommandCfg

#     def __init__(self, cfg: PatrolCommandCfg, env):
#         super().__init__(cfg, env)
        
#         # 0 = To B, 1 = To A, 2 = Done
#         self.patrol_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
#         self._command = torch.zeros(self.num_envs, 3, device=self.device)

#         # Access scene elements
#         self.table_a = env.scene[cfg.asset_name_a]
#         self.table_b = env.scene[cfg.asset_name_b]
        
#         self.metrics = {}

#     @property
#     def command(self):
#         return self._command

#     def _resample_command(self, env_ids):
#         self.patrol_state[env_ids] = 0
#         self._command[env_ids] = 0.0

#     def _update_command(self):
#         # Use self._env (with underscore)
#         robot_pos_w = self._env.scene["robot"].data.root_pos_w
#         pos_a_w = self.table_a.data.root_pos_w
#         pos_b_w = self.table_b.data.root_pos_w

#         # Target selection
#         target_pos = torch.where(
#             self.patrol_state.unsqueeze(1) == 0, 
#             pos_b_w, 
#             pos_a_w
#         )

#         # Distance logic
#         to_target = target_pos[:, :2] - robot_pos_w[:, :2]
#         dist = torch.norm(to_target, dim=1)
        
#         # State transitions
#         at_b = (self.patrol_state == 0) & (dist < self.cfg.waypoint_threshold)
#         self.patrol_state[at_b] = 1
        
#         at_a = (self.patrol_state == 1) & (dist < self.cfg.waypoint_threshold)
#         self.patrol_state[at_a] = 2

#         # Command generation
#         desired_yaw = torch.atan2(to_target[:, 1], to_target[:, 0])
#         _, _, current_yaw = get_euler_xyz(self._env.scene["robot"].data.root_quat_w)
#         yaw_error = wrap_to_pi(desired_yaw - current_yaw)
        
#         lin_vel_x = torch.where(torch.abs(yaw_error) < 1.57, self.cfg.walk_speed, 0.0)
#         lin_vel_x[self.patrol_state == 2] = 0.0
        
#         ang_vel_z = torch.clamp(yaw_error * 2.0, -self.cfg.turn_speed, self.cfg.turn_speed)
#         ang_vel_z[self.patrol_state == 2] = 0.0

#         self._command[:, 0] = lin_vel_x
#         self._command[:, 1] = 0.0
#         self._command[:, 2] = ang_vel_z

#     def _update_metrics(self):
#         # FIX IS HERE: Cast to float() so .mean() works during logging
#         self.metrics["patrol_state"] = self.patrol_state.float()


# Helper needed if not imported
# def get_euler_xyz(quat):
#     # Simplified placeholder for quaternion to euler conversion
#     # In real code use: isaaclab.utils.math.euler_xyz_from_quat
#     from isaaclab.utils.math import euler_xyz_from_quat
#     return euler_xyz_from_quat(quat)


# @configclass
# class CommandsCfg:
#     """Command specifications for the MDP."""
#     base_velocity = PatrolCommandCfg(
#         class_type=PatrolCommand, # Added this to match your original structure
#         resampling_time_range=(0.0, 0.0), # Important: 0.0 means "update every step"
#         debug_vis=True,
#         asset_name_a="table_A",
#         asset_name_b="table_B",
#         waypoint_threshold=1.0,
#         walk_speed=0.8,
#     )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # @configclass
    # class DebugCfg(ObsGroup):
    #     """Observations NOT seen by policy, used for rewards or logging."""
    #     def __post_init__(self):
    #         # This makes env.unwrapped.obs_buf["debug"] a DICTIONARY of tensors
    #         # instead of one big tensor.
    #         self.concatenate_terms = False

    #     cur_pos = ObsTerm(func=mdp.root_pos_w)  # Global translation
    #     tableA_pos = ObsTerm(func=mdp.root_pos_w,params={"asset_cfg": SceneEntityCfg("table_A")})
    #     tableB_pos = ObsTerm(func=mdp.root_pos_w,params={"asset_cfg": SceneEntityCfg("table_B")})

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # debug: DebugCfg = DebugCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_near_tableA = EventTerm(
        func=mdp.reset_robot_near_target,
        mode="reset",
        params={
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

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"),
            "threshold": 1.0,
        },
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.05)


# def patrol_finished(env, command_name: str = "base_velocity"):
#     """Terminate if patrol state is 2 (Done)."""
#     # Access the command term by name
#     command_term = env.command_manager.get_term(command_name)
#     # Return boolean tensor (True = Reset)
#     return command_term.patrol_state == 2


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )

    # mission_complete = DoneTerm(
    #     func=patrol_finished,
    #     params={"command_name": "base_velocity"},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LongCorridorEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: LongCorridorCfg = LongCorridorCfg(num_envs=2, env_spacing=60.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 60.0
        # viewer settings
        self.viewer.eye = [0.0, -25.0, 3.0]
        self.viewer.lookat = [0.0, -30.0, 3.0]
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.gravity = (0.0, 0.0, -1000)
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15       
        # table_pos = self.scene.table_A.init_state.pos
        # self.scene.robot.init_state.pos = (table_pos[0] + 0.0, table_pos[1] + 0.1, table_pos[2] + 0.0)

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
        # else:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = False
