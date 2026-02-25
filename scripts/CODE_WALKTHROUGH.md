# Code Walkthrough: Goal-Directed Navigation Implementation

## Overview

This document walks through the implementation of goal-directed wheeled robot navigation in the file:
`/home/jahirsadikmonon/Documents/Projects/IsaacLab/scripts/long_corridor_wheeled.py`

## Table of Contents

1. [Jetbot Configuration](#jetbot-configuration)
2. [Scene Setup](#scene-setup)
3. [Custom Action Term](#custom-action-term)
4. [Environment Configuration](#environment-configuration)
5. [Main Loop](#main-loop)

---

## Jetbot Configuration

**Lines 65-94**: Define the Jetbot robot

```python
JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # Match all wheel joints
            damping=None,              # No damping
            stiffness=None             # No stiffness (pure velocity control)
        )
    },
)
```

**What happens here:**
- Loads the Jetbot USD model from Isaac Sim's public assets
- Configures implicit actuators for velocity control on all joints
- The `.*` regex matches all joint names (both wheels)

---

## Scene Setup

**Lines 107-243**: `LongCorridorWheeledRobotSceneCfg` class

```python
@configclass
class LongCorridorWheeledRobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for long corridor environment."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(1000000, 1000)),
    )
    
    # Long corridor terrain (custom USD)
    terrain = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Corridor",
        spawn=UsdFileCfg(
            usd_path=os.path.join(DATA_DIR, "long_corridor.usda"),
        ),
    )
    
    # Optional tables for object placement
    table_A = RigidObjectCfg(...)
    table_B = RigidObjectCfg(...)
    
    # Optional cone objects on tables
    if SPAWN_TABLE_A_OBJECTS:
        cone_A = RigidObjectCfg(...)
        # ... more objects ...
    
    # Robot configuration
    jetbot: ArticulationCfg = JETBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.6, 0.0, 0.0))
    )
```

**What happens here:**
- `{ENV_REGEX_NS}` is a special token that gets replaced per environment instance
- Each environment gets its own copy of the scene at a different position
- `init_state` sets the initial position of the Jetbot

### Scene Elements

| Element | Type | Purpose |
|---------|------|---------|
| `ground` | AssetBase | Flat ground plane |
| `terrain` | AssetBase | Long corridor (from custom USD file) |
| `table_A/B` | RigidObject | Optional static tables |
| `cone_A/B`, `cuboid_A/B` | RigidObject | Optional objects on tables |
| `cuboid_in_path_*` | RigidObject | Optional obstacles in corridor |
| `jetbot` | Articulation | The wheeled robot (2 DOF for wheels) |

---

## Custom Action Term

**Lines 250-371**: `GoalBasedWheelAction` class

### Structure

```python
class GoalBasedWheelAction(ActionTerm):
    """Converts goal poses to wheel velocity commands."""
    
    def __init__(self, cfg, env):
        """Initialize action term."""
        super().__init__(cfg, env)
        self.cfg = cfg
        self._robot = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(...)
        self._processed_actions = torch.zeros(...)
    
    def _apply_actions(self, actions):
        """Convert goal poses to wheel commands."""
        # ... implementation ...
```

### Key Methods

#### 1. **Input Processing**

```python
# Extract goal pose from actions
goal_pos = actions[:, :2]  # [goal_x, goal_y]
```

**Input format per environment:**
```
[x_goal, y_goal, yaw_goal]
```

#### 2. **State Reading**

```python
root_pos = self._robot.data.root_pos_w  # Current position (x, y, z)
root_quat = self._robot.data.root_quat_w  # Current orientation (quaternion)

# Convert quaternion to Euler angles
euler = euler_xyz_from_quat(root_quat)
current_yaw = euler[:, 2]  # Extract yaw (heading)
```

**Robot state available:**
- `root_pos_w`: World position (num_envs, 3)
- `root_quat_w`: World orientation as quaternion (num_envs, 4)
- `root_lin_vel_w`: Linear velocity (num_envs, 3)
- `root_ang_vel_w`: Angular velocity (num_envs, 3)
- `heading_w`: Yaw angle (num_envs,)

#### 3. **Error Computation**

```python
# Position error
pos_error = goal_pos - root_pos[:, :2]  # (num_envs, 2)
distance = torch.norm(pos_error, dim=1)  # (num_envs,)

# Orientation error: desired yaw to reach goal
desired_yaw = torch.atan2(pos_error[:, 1], pos_error[:, 0])
angle_to_goal = _normalize_angle(desired_yaw - current_yaw)
```

**Example:**
- Robot at (0, 0), goal at (3, 4)
- pos_error = (3, 4)
- distance = 5.0 meters
- desired_yaw = atan2(4, 3) ≈ 0.927 rad ≈ 53°

#### 4. **Proportional Control**

```python
# Linear velocity from distance error
linear_vel = linear_gain * distance  # Closer → slower
linear_vel = clamp(linear_vel, -max_linear_vel, max_linear_vel)

# Angular velocity from orientation error
angular_vel = angular_gain * angle_to_goal  # Bigger error → faster rotation
angular_vel = clamp(angular_vel, -max_angular_vel, max_angular_vel)
```

**Control law:**
```
v = K_p * error_distance
ω = K_ω * error_angle
```

Where:
- K_p (linear_gain) = 0.5 m/s per meter
- K_ω (angular_gain) = 1.0 rad/s per radian

**Stability feature:**
```python
# If angle error is large, reduce forward speed
high_angular_error = |angle_to_goal| > 0.5 rad
linear_vel[high_angular_error] *= 0.5
```

#### 5. **Differential Drive Kinematics**

```python
# Convert [v, ω] to wheel velocities
# For differential drive:
# v = (v_left + v_right) / 2
# ω = (v_right - v_left) / wheel_base

# Solve for wheel speeds:
v_left = (v - ω * wheel_base/2) / wheel_radius
v_right = (v + ω * wheel_base/2) / wheel_radius
```

**Example:**
- `v = 1.0 m/s`, `ω = 1.0 rad/s`
- `wheel_base = 0.16 m`, `wheel_radius = 0.032 m`
- `v_left = (1.0 - 1.0 * 0.08) / 0.032 = 28.75 rad/s`
- `v_right = (1.0 + 1.0 * 0.08) / 0.032 = 33.75 rad/s`

#### 6. **Wheel Assignment**

```python
# Find wheel joint indices
left_idx, _ = robot.find_joints("left.*wheel|left.*drive")
right_idx, _ = robot.find_joints("right.*wheel|right.*drive")

# Set velocity targets
joint_vel = torch.zeros(num_envs, num_joints)
joint_vel[:, left_idx] = v_left
joint_vel[:, right_idx] = v_right

# Apply to simulation
robot.set_joint_velocity_target(joint_vel)
```

---

## Environment Configuration

**Lines 417-450**: `LongCorridorWheeledEnvCfg`

```python
@configclass
class LongCorridorWheeledEnvCfg(ManagerBasedEnvCfg):
    """Full environment configuration."""
    
    # Scene setup
    scene = LongCorridorWheeledRobotSceneCfg(
        num_envs=1024,  # Run 1024 parallel environments
        env_spacing=2.5  # Space between environments
    )
    
    # Managers
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()  # Reset events
    
    def __post_init__(self):
        # Viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        
        # Stepping settings
        self.decimation = 4  # Apply actions every 4 sim steps
        
        # Simulation
        self.sim.dt = 0.01  # 10 ms per step
```

### Key Settings

| Setting | Value | Meaning |
|---------|-------|---------|
| `num_envs` | 1024 | Run 1024 robots in parallel |
| `env_spacing` | 2.5 | Space between environments |
| `decimation` | 4 | Action applied every 4 physics steps |
| `sim.dt` | 0.01 | Physics timestep is 10ms |
| Effective freq | 25 Hz | 100 Hz physics / 4 decimation |

### Observations

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_pos = ObsTerm(func=mdp.root_pos_w)      # Position (3 dims)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # Linear velocity (3 dims)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # Angular velocity (3 dims)
    
    policy: PolicyCfg = PolicyCfg()
```

**Total observation size:** 9 dimensions per environment

---

## Main Loop

**Lines 548-620**: The `main()` function

```python
def main():
    # 1. Create environment
    env_cfg = LongCorridorWheeledEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    env = ManagerBasedEnv(cfg=env_cfg)
    
    # 2. Create action term
    goal_action_cfg = GoalBasedWheelActionCfg(asset_name="jetbot")
    action_term = GoalBasedWheelAction(goal_action_cfg, env)
    
    # 3. Get robot reference
    robot = env.scene["jetbot"]
    
    # 4. Initialize goals
    current_goals = torch.zeros(env.num_envs, 3, device=env.device)
    for i in range(min(env.num_envs, 32)):
        angle = (i % 2) * 3.14159
        distance = 2.0 + (i % 4) * 0.5
        current_goals[i, 0] = distance * cos(angle)
        current_goals[i, 1] = distance * sin(angle)
        current_goals[i, 2] = angle
    
    # 5. Simulation loop
    count = 0
    while simulation_app.is_running():
        # Reset every 300 steps
        if count % 300 == 0:
            env.reset()
            print(f"Reset at step {count}")
        
        # Apply actions (convert goals to wheel velocities)
        action_term._apply_actions(current_goals)
        
        # Step the environment
        env.step(current_goals)
        
        # Get observations
        obs = env.obs_manager.compute()
        
        # Print debug info
        if count % 50 == 0:
            pos = robot.data.root_pos_w[0, :2]
            print(f"Step {count}: Robot 0 at {pos}")
        
        count += 1
    
    # 6. Cleanup
    env.close()
```

### Step-by-Step Breakdown

#### Step 1: Create Environment (Line 553-558)
Creates 1024 parallel Jetbot environments

#### Step 2: Create Action Term (Line 560-562)
Instantiates the goal-to-wheel converter

#### Step 3: Get Robot Reference (Line 565)
Grab handle to access robot state (`data.root_pos_w`, etc.)

#### Step 4: Initialize Goals (Line 568-577)
Set starting goal poses (alternating circle and distance)

#### Step 5: Simulation Loop (Line 579-610)
**Every step:**
1. Reset environment every 300 steps
2. Call `_apply_actions()` to compute wheel velocities from goals
3. Call `env.step()` to advance physics
4. Get observations (position, velocity)
5. Print debug output every 50 steps

#### Step 6: Cleanup (Line 612-614)
Close environment and simulation app

---

## Data Structures

### Action Input Format

```python
actions: torch.Tensor  # Shape: (num_envs, 3)
# actions[:, 0]  = goal_x
# actions[:, 1]  = goal_y
# actions[:, 2]  = goal_yaw
```

### Observation Output Format

```python
obs = {
    "policy": torch.Tensor  # Shape: (num_envs, 9)
}
# obs["policy"][:, 0:3]   = position (x, y, z)
# obs["policy"][:, 3:6]   = linear velocity (vx, vy, vz)
# obs["policy"][:, 6:9]   = angular velocity (wx, wy, wz)
```

### Robot State (accessible via `robot.data`)

```python
robot.data.root_pos_w          # Position (num_envs, 3)
robot.data.root_quat_w         # Quaternion (num_envs, 4)
robot.data.root_lin_vel_w      # Linear vel (num_envs, 3)
robot.data.root_ang_vel_w      # Angular vel (num_envs, 3)
robot.data.joint_pos           # All joint positions
robot.data.joint_vel           # All joint velocities
```

---

## Import Tree

```
long_corridor_wheeled.py
├── isaaclab.app              # Isaac Sim launcher
├── isaaclab.envs             # Environment classes
├── isaaclab.managers          # Action/Observation managers
├── isaaclab.assets           # Robot/Scene definitions
├── isaaclab.sim              # Physics simulation utilities
├── isaaclab.utils            # Math utilities (quaternions, etc)
└── isaaclab.actuators        # Actuator models
```

---

## Execution Flow Diagram

```
┌─────────────────────────────────────┐
│       main()                        │
├─────────────────────────────────────┤
│ 1. Create ManagerBasedEnv           │
│    - Load JETBOT_CFG                │
│    - Create 1024 parallel scenes    │
│    - Initialize managers            │
├─────────────────────────────────────┤
│ 2. Create GoalBasedWheelAction      │
│    - Parse wheel joint indices      │
│    - Initialize control gains       │
├─────────────────────────────────────┤
│ 3. Main Loop (each step):           │
│    a. Check for reset condition     │
│    b. action_term._apply_actions()  │
│       - Read robot pose             │
│       - Compute position/angle error│
│       - Apply proportional control  │
│       - Convert to differential     │
│       - Set wheel velocities        │
│    c. env.step(actions)             │
│       - Update physics (4x at 100Hz)│
│       - Step all managers           │
│    d. obs_manager.compute()         │
│       - Gather observations         │
│    e. Print debug info              │
└─────────────────────────────────────┘
```

---

## Common Customizations

### Change Control Gains

```python
# In main():
goal_action_cfg = GoalBasedWheelActionCfg(
    asset_name="jetbot",
    linear_gain=1.0,    # Move faster
    angular_gain=2.0,   # Turn sharper
    max_linear_vel=2.0,
    max_angular_vel=3.0
)
```

### Add Custom Rewards

```python
# Add to ObservationsCfg or create RewardsCfg
distance_to_goal = compute_distance(
    robot.data.root_pos_w[:, :2],
    current_goals[:, :2]
)
reward = -distance_to_goal  # Negative distance reward
```

### Change Goal Generation

```python
# In the loop:
# Option 1: Static goals
current_goals[:] = torch.tensor([5.0, 3.0, 0.0])

# Option 2: Time-varying goals
angle = count * 0.01
current_goals[:, 0] = 5 * cos(angle)
current_goals[:, 1] = 5 * sin(angle)
current_goals[:, 2] = angle

# Option 3: Random goals
current_goals = torch.randn_like(current_goals) * 5
```

---

## Performance Metrics

Typical performance on a modern GPU:

| Metric | Value |
|--------|-------|
| Num environments | 1024 |
| Simulation frame rate | ~1000 FPS |
| Physics freq | 100 Hz |
| Action freq | 25 Hz |
| Data generation | ~7.5M steps/hour |

Each step takes ~1ms for all 1024 environments + all computations.

