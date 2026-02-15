# Spot Robot in Long Corridor Environment

This guide explains how to set up and run the Boston Dynamics Spot robot in your long corridor USD environment using Isaac Sim and Isaac Lab.

## Overview

The configuration provides:
- **Spot Robot Control**: Uses the Boston Dynamics Spot robot model with realistic actuators
- **Long Corridor Environment**: Your custom USD environment with obstacles
- **Velocity Tracking**: The robot learns to track velocity commands
- **Controllers**: Multiple control options (random, manual, trained policies)

## File Structure

```
source/isaaclab_tasks/manager_based/locomotion/velocity/
├── config/spot/
│   ├── spot_long_corridor_env_cfg.py     # Environment configuration
│   └── __init__.py                        # Gymnasium registration
│
scripts/demos/
├── run_spot_long_corridor.py              # Run with random actions
├── play_spot_long_corridor.py             # Run with trained policy
```

## Quick Start

### 1. Basic Simulation (Random Actions)

Run the environment with random actions to verify everything is working:

```bash
cd /home/jahirsadikmonon/Documents/Projects/IsaacLab

# Run with 4 parallel environments
./isaaclab.sh -p scripts/demos/run_spot_long_corridor.py --num_envs 4

# Run in headless mode (no viewer)
./isaaclab.sh -p scripts/demos/run_spot_long_corridor.py --num_envs 4 --headless

# Run play mode (deterministic, single environment)
./isaaclab.sh -p scripts/demos/run_spot_long_corridor.py --play
```

### 2. Run with Trained Policy

If you have a trained policy checkpoint:

```bash
./isaaclab.sh -p scripts/demos/play_spot_long_corridor.py \
    --checkpoint /path/to/policy.pt \
    --num_envs 2 \
    --num_steps 5000
```

### 3. Train a Policy

To train your own policy from scratch:

```bash
# Using RSL-RL (recommended)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Long-Corridor-Spot-v0 \
    --num_envs 4096 \
    --headless

# Using Stable Baselines3
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task Isaac-Velocity-Long-Corridor-Spot-v0 \
    --num_envs 512 \
    --headless
```

The trained checkpoint will be saved in `logs/rsl_rl/` or `logs/sb3/`.

## Environment Configuration

### Key Parameters

The long corridor environment is configured with:

- **Simulation**: 500 Hz physics, 50 Hz control
- **Robots**: Spot quadruped with DelayedPD and RemotizedPD actuators
- **Scene**: Your custom long corridor USD with tables and obstacles
- **Observations**: 
  - Base linear/angular velocity
  - Projected gravity
  - Velocity commands
  - Joint positions/velocities
  - Last actions
  
- **Actions**: Joint position targets (scale: 0.2)
- **Rewards**:
  - Velocity tracking (1.5x linear, 0.75x angular)
  - Penalties for unwanted motions
  - Air-time rewards
  - Energy efficiency

- **Terminations**:
  - Episode timeout (20 seconds)
  - Contact with base

### Customization

Edit `source/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/spot_long_corridor_env_cfg.py` to:

#### Change robot parameters:
```python
# In SpotLongCorridorEnvCfg.__post_init__
self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.actions.joint_pos.scale = 0.2  # Action scale
```

#### Adjust reward weights:
```python
# In RewardsCfg class
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=1.5,  # Adjust this value
    ...
)
```

#### Modify command ranges:
```python
# In CommandsCfg class
ranges=mdp.UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-1.0, 1.5),     # Min/max x velocity
    lin_vel_y=(-0.5, 0.5),     # Min/max y velocity
    ang_vel_z=(-1.0, 1.0),     # Min/max angular velocity
)
```

#### Enable/disable corridor objects:
```python
SPAWN_TABLE_A_OBJECTS = True   # Spawn objects in table A area
SPAWN_TABLE_B_OBJECTS = True   # Spawn objects in table B area
SPAWN_CUBOIDS_IN_PATH = True   # Spawn cuboids along the path
NUM_CUBOIDS = 5                # Number of cuboids
```

## Understanding the Controllers

### 1. Velocity Commands
The robot receives target velocity commands (lin_vel_x, lin_vel_y, ang_vel_z, heading) that change every 10 seconds. The policy learns to execute these commands by generating joint position targets.

### 2. Action Processing
- **Input**: Commands + Observations
- **Policy**: Neural network that processes observations
- **Output**: Joint position targets
- **Execution**: PD controller tracks the targets

### 3. Actuator Models
Spot uses specialized actuators:
- **Hip joints** (hx, hy): DelayedPD with 0-8ms delay
- **Knee joints** (kn): RemotizedPD with realistic torque curves

## Training Tips

1. **Curriculum Learning**: Start with simple tasks, gradually increase difficulty
   ```python
   # Modify in config to enable curriculum
   curriculum: CurriculumCfg = CurriculumCfg()
   ```

2. **Hyperparameter Tuning**:
   - Increase learning rate if training is too slow
   - Decrease entropy coefficient for more deterministic policies
   - Adjust reward weights for different behaviors

3. **Checkpointing**:
   ```bash
   # Resume from checkpoint
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
       --task Isaac-Velocity-Long-Corridor-Spot-v0 \
       --checkpoint logs/rsl_rl/spot_long_corridor/model_5000.pt
   ```

## Troubleshooting

### Issue: USD not found
- Make sure `DATA_DIR` in the config points to the correct directory
- Verify `long_corridor.usda` exists at that location

### Issue: SPOT_CFG not found
- Ensure you're running from the Isaac Lab root directory
- Check that `isaaclab_assets` is properly installed

### Issue: Observation/Action dimension mismatch
- Don't modify the observation/action configurations without updating the policy
- Train a new policy if you change these dimensions

### Issue: Out of memory
- Reduce `num_envs` (try 512 instead of 4096)
- Set `sim.use_fabric = False` to reduce memory usage
- Use `--headless` flag when training

## Advanced Features

### Recording Videos
```bash
./isaaclab.sh -p scripts/demos/play_spot_long_corridor.py \
    --checkpoint policy.pt \
    --video \
    --num_steps 1000
```

### Custom Terrain
Modify the terrain in the USD file or customize the scene configuration.

### Multi-Agent Training
The environment supports parallel training of multiple policies. See the RL framework documentation.

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Boston Dynamics Spot](https://www.bostondynamics.com/sites/default/files/2023-07/spot-datasheet.pdf)
- [RL Training Guides](https://isaac-sim.github.io/IsaacLab/main/source/features/reinforcement-learning.html)

## Contact & Support

For issues specific to this configuration, refer to the Isaac Lab documentation or community forums.
