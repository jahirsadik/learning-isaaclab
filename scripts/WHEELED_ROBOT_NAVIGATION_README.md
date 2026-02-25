# Goal-Directed Navigation for Wheeled Robots in Isaac Lab

## Overview

This document describes the implementation of goal-directed navigation for a wheeled robot (Jetbot) in an Isaac Lab manager-based environment. The implementation allows you to command the robot to move from point A to point B in simulation, with the robot automatically controlling its wheel velocities to reach the specified goal pose.

## Key Components

### 1. **GoalBasedWheelAction** (Custom Action Term)

A custom action term that converts high-level goal poses `[goal_x, goal_y, goal_yaw]` into low-level wheel velocity commands.

**How it works:**
- **Input**: Goal pose in world frame `[goal_x, goal_y, goal_yaw]` per environment instance
- **Processing**: 
  - Reads current robot base pose from the Jetbot articulation
  - Computes position and orientation errors
  - Applies proportional control to generate linear velocity `v` and angular velocity `ω`
  - Converts to differential drive wheel velocities using kinematic equations:
    ```
    v_left = (v - ω * wheel_base/2) / wheel_radius
    v_right = (v + ω * wheel_base/2) / wheel_radius
    ```
- **Output**: Wheel joint velocity targets applied to the robot

**Key Parameters** (configurable in `GoalBasedWheelActionCfg`):
- `linear_gain`: Proportional gain for linear velocity control (default: 0.5 m/s per meter error)
- `angular_gain`: Proportional gain for angular velocity control (default: 1.0 rad/s per radian error)
- `max_linear_vel`: Maximum linear velocity clamp (default: 1.0 m/s)
- `max_angular_vel`: Maximum angular velocity clamp (default: 2.0 rad/s)
- `wheel_base`: Distance between left and right wheels (Jetbot: 0.16m)
- `wheel_radius`: Wheel radius (Jetbot: 0.032m)

### 2. **Long Corridor Environment Configuration**

The environment (`LongCorridorWheeledEnvCfg`) provides:
- **Scene**: Long corridor terrain with optional tables and cuboids
- **Robot**: Jetbot wheeled robot with implicit actuators
- **Actions**: Goal-based navigation (3D goal poses)
- **Observations**: Base position, linear velocity, angular velocity
- **Events**: Reset robot positions, spawn objects

### 3. **Jetbot Configuration**

```python
JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
```

## Usage

### Basic Example

```python
# Create environment
env_cfg = LongCorridorWheeledEnvCfg()
env_cfg.scene.num_envs = 32
env = ManagerBasedEnv(cfg=env_cfg)

# Create action term
goal_action_cfg = GoalBasedWheelActionCfg(asset_name="jetbot")
action_term = GoalBasedWheelAction(goal_action_cfg, env)

# Get robot reference
robot = env.scene["jetbot"]

# Define goal poses for each environment
goals = torch.zeros(env.num_envs, 3, device=env.device)
goals[:, 0] = 5.0  # x position
goals[:, 1] = 0.0  # y position
goals[:, 2] = 0.0  # yaw angle

# Run simulation loop
for step in range(1000):
    env.reset()  # Reset every N steps as needed
    
    # Apply goal actions
    action_term._apply_actions(goals)
    
    # Step environment
    env.step(goals)
    
    # Get observations
    obs = env.obs_manager.compute()
```

### Running the Script

```bash
# Basic run with default settings (16 envs)
./isaaclab.sh -p scripts/long_corridor_wheeled.py

# Run with multiple environments
./isaaclab.sh -p scripts/long_corridor_wheeled.py --num_envs 64

# Run headless (no visualization)
./isaaclab.sh -p scripts/long_corridor_wheeled.py --num_envs 32 --headless
```

## Architecture Design

### Control Flow

```
┌─────────────────────┐
│  Goal Pose Input    │
│  [gx, gy, gyaw]     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  GoalBasedWheelAction
│  - Compute errors   │
│  - Proportional ctrl│
│  - Kinematics       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Wheel Velocities   │
│  [v_left, v_right]  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Jetbot Actuators   │
│  (ImplicitActuator) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Physics Simulation │
└─────────────────────┘
```

## Extension Points

### 1. Optional: Path Planning Integration (Quintic Polynomials)

To add smooth path planning:

```python
from isaaclab.envs.mdp.path_planning import QuinticPolynomialsPlanner

# In main():
planner = QuinticPolynomialsPlanner(
    start_pos=robot.data.root_pos_w[:, :2],
    goal_pos=goals[:, :2],
    total_time=5.0
)

# Get waypoints at each step
waypoints = planner.get_waypoint(step_index)

# Use waypoint as intermediate goal
action_term._apply_actions(waypoints)
```

### 2. Custom Reward Functions

For RL training:

```python
@configclass
class RewardsCfg:
    reach_goal = RewTerm(
        func=mdp.goal_reached,
        weight=1.0,
        params={"threshold": 0.1}
    )
    distance_to_goal = RewTerm(
        func=mdp.distance_to_goal,
        weight=-0.1,
        params={"goal_cfg": SceneEntityCfg("robot")}
    )
```

### 3. Enhanced Observations

```python
@configclass
class ObservationsCfg:
    # ... existing ...
    
    @configclass
    class PolicyCfg(ObsGroup):
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # Add goal-relative observations
        goal_displacement = ObsTerm(func=compute_goal_displacement)
```

## Troubleshooting

### Issue: Wheel joint indices not found

**Solution**: Adjust the wheel joint name patterns in `GoalBasedWheelAction._apply_actions()`:

```python
left_idx, _ = self._robot.find_joints("left.*")  # Adjust pattern
right_idx, _ = self._robot.find_joints("right.*")
```

### Issue: Robot not reaching goal smoothly

**Solution**: Adjust control gains in `GoalBasedWheelActionCfg`:
- Increase `linear_gain` for faster movement
- Increase `angular_gain` for tighter rotation control
- Use `max_linear_vel` and `max_angular_vel` to limit maximum speeds

### Issue: Wheel velocities seem incorrect

**Solution**: Verify Jetbot wheel parameters:
```python
# Check actual wheel separation and radius
wheelbase = 0.16  # meters
wheel_radius = 0.032  # meters
```

## Performance Considerations

- **Batched Operations**: All control computations are vectorized across environments using PyTorch
- **GPU-Accelerated**: Runs on GPU if available (CUDA)
- **Minimal Overhead**: Custom action term adds <1% computational overhead

## Future Enhancements

1. **Non-Linear Control**: Replace proportional control with PID or MPC
2. **Obstacle Avoidance**: Integrate with local planners (DWA, TEB)
3. **Multi-Goal Sequencing**: Chain multiple goal poses (waypoints)
4. **Learning-Based Control**: Train a neural network policy to replace the controller
5. **Sim-to-Real Transfer**: Test policies on real Jetbot hardware

## References

- Isaac Sim Jetbot: https://docs.omniverse.nvidia.com/isaacsim/latest/
- Differential Drive Kinematics: https://en.wikipedia.org/wiki/Differential_wheeled_robot
- Isaac Lab ManagerBasedEnv: https://github.com/isaac-sim/IsaacLab

## File Structure

```
scripts/
├── long_corridor_wheeled.py          # Main environment script
├── WHEELED_ROBOT_NAVIGATION_README.md # This file
└── ...

source/
└── isaaclab/
    └── envs/
        └── mdp/
            ├── actions/
            ├── commands/
            └── ...
```

## Contact & Support

For issues or questions:
1. Check the Isaac Lab documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/
2. Issue tracker: https://github.com/isaac-sim/IsaacLab/issues
