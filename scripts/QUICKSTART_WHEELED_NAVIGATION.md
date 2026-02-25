# Quick Start Guide: Goal-Directed Navigation

## TL;DR - Get Started in 5 Minutes

### 1. Run the Example

```bash
cd /home/jahirsadikmonon/Documents/Projects/IsaacLab
./isaaclab.sh -p scripts/long_corridor_wheeled.py --num_envs 16
```

The script will:
- Spawn 16 parallel environments with Jetbot robots
- Set different goal poses for each robot
- Print robot positions every 50 steps
- Run for 300 steps then reset

### 2. Key Classes

**GoalBasedWheelActionCfg** - Configure your controller
```python
cfg = GoalBasedWheelActionCfg(
    asset_name="jetbot",
    linear_gain=0.5,      # Speed up: increase to 1.0
    angular_gain=1.0,     # Tighter turns: increase to 2.0
    max_linear_vel=1.0,   # Max forward speed
    max_angular_vel=2.0   # Max rotation speed
)
```

**GoalBasedWheelAction** - The controller
```python
action_term = GoalBasedWheelAction(cfg, env)
action_term._apply_actions(goal_poses)  # Apply actions
```

### 3. Basic Loop Pattern

```python
# Create environment
env = ManagerBasedEnv(cfg=LongCorridorWheeledEnvCfg())

# Create action term
cfg = GoalBasedWheelActionCfg()
action_term = GoalBasedWheelAction(cfg, env)
robot = env.scene["jetbot"]

# Set goals
goals = torch.zeros(env.num_envs, 3)  # [x, y, yaw]
goals[:, 0] = 5.0   # Move to x=5
goals[:, 1] = 3.0   # Move to y=3

# Run loop
for step in range(100):
    if step % 50 == 0:
        env.reset()
    
    action_term._apply_actions(goals)
    env.step(goals)
    
    # Read robot poses
    pos = robot.data.root_pos_w[:, :2]  # x, y positions
    print(f"Step {step}: Robot 0 at {pos[0]}")
```

### 4. Modify Robot Goals Dynamically

```python
# Update goals each step
for step in range(100):
    # Move goal in a circle
    angle = step * 0.1
    radius = 3.0
    goals[:, 0] = radius * torch.cos(torch.tensor(angle))
    goals[:, 1] = radius * torch.sin(torch.tensor(angle))
    goals[:, 2] = angle
    
    action_term._apply_actions(goals)
    env.step(goals)
```

### 5. Tune for Your Needs

**Problem: Robot moves too slow**
```python
cfg.linear_gain = 1.0  # Increase from 0.5
cfg.max_linear_vel = 2.0  # Increase from 1.0
```

**Problem: Robot overshoots goals**
```python
cfg.linear_gain = 0.2  # Decrease from 0.5
cfg.max_linear_vel = 0.5  # Decrease from 1.0
```

**Problem: Robot doesn't rotate properly**
```python
cfg.angular_gain = 2.0  # Increase from 1.0
cfg.max_angular_vel = 3.0  # Increase from 2.0
```

## File Structure

The main implementation is in: `/home/jahirsadikmonon/Documents/Projects/IsaacLab/scripts/long_corridor_wheeled.py`

Key sections:
- **Lines 65-94**: JETBOT configuration
- **Lines 250-371**: `GoalBasedWheelAction` class
- **Lines 374-415**: `GoalBasedWheelActionCfg` configuration
- **Lines 560-610**: `main()` function with example usage

## What's Implemented

✅ Goal-based action term (converts goal poses to wheel commands)  
✅ Differential drive kinematics  
✅ Proportional control for position and orientation  
✅ Multiple parallel environments  
✅ Robot pose tracking  
✅ Configurable control gains  

## What's Optional (Extensions)

Below are features you can add:

### Path Planning
```python
# Use quintic polynomial planning for smooth trajectories
from isaaclab.envs.mdp import QuinticPolynomialsPlanner
```

### Obstacle Avoidance
```python
# Integrate with local planners like DWA or TEB
```

### Learning
```python
# Train a neural network policy to learn where to go
# using RL algorithms (PPO, SAC, etc.)
```

## Common Parameters Explained

| Parameter | Units | Typical Range | Effect |
|-----------|-------|---------------|--------|
| `linear_gain` | m/s per m | 0.1-2.0 | Speed response to distance error |
| `angular_gain` | rad/s per rad | 0.5-5.0 | Rotation response to angle error |
| `max_linear_vel` | m/s | 0.5-3.0 | Maximum forward speed |
| `max_angular_vel` | rad/s | 1.0-5.0 | Maximum rotation speed |
| `wheel_base` | m | ~0.16 for Jetbot | Distance between wheels |
| `wheel_radius` | m | ~0.032 for Jetbot | Wheel radius |

## Tips & Tricks

1. **Want smooth movement?** Set `linear_gain = 0.2` and `angular_gain = 0.8`
2. **Want agile movement?** Set `linear_gain = 1.0` and `angular_gain = 2.0`
3. **Want realistic robot?** Match physical parameters: `wheel_base` and `wheel_radius`
4. **Want multiple robots?** Just change `env_cfg.scene.num_envs = 1000`
5. **Want faster simulation?** Increase `env_cfg.decimation` to 8 or 16

## Debugging

Print robot state:
```python
print("Position:", robot.data.root_pos_w)  # x, y, z in world frame
print("Velocity:", robot.data.root_lin_vel_w)  # linear velocity
print("Yaw:", robot.data.heading_w)  # heading angle
```

Check wheel joint indices:
```python
left_idx, left_name = robot.find_joints("left.*")
print(f"Left wheel: {left_name} (index {left_idx})")
```

## Next Steps

1. **Run the script**: `./isaaclab.sh -p scripts/long_corridor_wheeled.py`
2. **Modify goal patterns**: Edit the goal setting in `main()`
3. **Tune gains**: Adjust `GoalBasedWheelActionCfg` parameters
4. **Add observations**: Implement custom reward functions
5. **Train an agent**: Use the environment with RL algorithms

See `WHEELED_ROBOT_NAVIGATION_README.md` for detailed documentation.
