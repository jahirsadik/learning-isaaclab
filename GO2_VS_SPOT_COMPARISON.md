# GO2 vs Spot Configuration Comparison

This guide explains the key differences between the Unitree GO2 and Boston Dynamics Spot configurations in the long corridor environment.

## Side-by-Side Comparison

| Feature | GO2 | Spot |
|---------|-----|------|
| **Robot Size** | Smaller (~0.6m) | Larger (~0.9m) |
| **Joints** | 12 (3 per leg) | 12 (3 per leg) |
| **Actuator Types** | Simple PD | DelayedPD + RemotizedPD |
| **Max Linear Speed** | ~2.0 m/s | ~1.5 m/s |
| **Torque Limits** | Uniform | Joint-specific (lookup table) |
| **Control Freq** | 50 Hz | 50 Hz |
| **Physics Timing** | 500 Hz | 500 Hz |

## Configuration File Structure

### GO2 Configuration
```python
# source/isaaclab_tasks/.../config/go2/go2_long_corridor_env_cfg.py
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.scene.robot.init_state.pos = (0.0, 0.0, 2.5)  # Higher initial position
self.actions.joint_pos.scale = 0.25                # Smaller action scale
```

### Spot Configuration
```python
# source/isaaclab_tasks/.../config/spot/spot_long_corridor_env_cfg.py
from isaaclab_assets.robots.spot import SPOT_CFG

self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)  # Lower initial position
self.actions.joint_pos.scale = 0.2                 # Slightly smaller action scale
```

## Key Differences

### 1. Actuator Models

**GO2** - Simple Actuators:
- Uses basic PD (Proportional-Derivative) control
- Uniform torque limits across all joints
- Fast response time
- Easy to control

```python
# GO2 uses simple PD actuators
actuators={
    "default": PDActuatorCfg(...)
}
```

**Spot** - Realistic Actuators:
- **Hip Joints**: DelayedPD with 0-8ms delay (simulates communication latency)
- **Knee Joints**: RemotizedPD with realistic torque curves (non-linear mapping)
- Complex joint parameter lookup table
- More challenging to control

```python
# Spot uses specialized actuators
actuators={
    "spot_hip": DelayedPDActuatorCfg(
        min_delay=0,  # 0ms
        max_delay=4,  # 8ms
    ),
    "spot_knee": RemotizedPDActuatorCfg(
        joint_parameter_lookup=joint_parameter_lookup,
    ),
}
```

### 2. Reward Functions

**GO2**:
```python
# More aggressive movement rewards
track_lin_vel_xy_exp = RewTerm(..., weight=1.5, ...)
track_ang_vel_z_exp = RewTerm(..., weight=0.75, ...)

# Moderate penalties
dof_torques_l2 = RewTerm(..., weight=-1.0e-5, ...)
action_rate_l2 = RewTerm(..., weight=-0.01, ...)
```

**Spot**:
```python
# Same reward structure but may need different tuning
track_lin_vel_xy_exp = RewTerm(..., weight=1.5, ...)
track_ang_vel_z_exp = RewTerm(..., weight=0.75, ...)

# Penalties adjusted for Spot's slower dynamics
dof_torques_l2 = RewTerm(..., weight=-1.0e-5, ...)
action_rate_l2 = RewTerm(..., weight=-0.01, ...)
```

### 3. Initial State

**GO2**:
- Starts at z=2.5m (higher)
- Ready for running immediately
- Quick convergence to walking gait

**Spot**:
- Starts at z=0.5m (lower)
- More stable initial configuration
- Needs more training for dynamic gaits

### 4. Action Scaling

**GO2**: `scale=0.25`
- Larger actions possible
- More aggressive movements
- Quicker response

**Spot**: `scale=0.2`
- More conservative actions
- Smoother movements
- Better for stability

## Performance Comparison

### Training Time
| Robot | RSL-RL Training | Notes |
|-------|-----------------|-------|
| GO2 | 2-3 hours | Converges faster, simpler dynamics |
| Spot | 4-6 hours | More complex actuators, slower convergence |

### Walking Stability
- **GO2**: Naturally stable, difficult to make fall
- **Spot**: Requires careful control, more realistic failure modes

### Speed Capabilities
- **GO2**: Faster typical speeds (1.5-2.0 m/s)
- **Spot**: Steadier, more controlled movements (0.8-1.5 m/s)

## Migration Guide: GO2 → Spot

If you're switching from GO2 to Spot:

### 1. Update Imports
```python
# Old (GO2)
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# New (Spot)
from isaaclab_assets.robots.spot import SPOT_CFG
```

### 2. Adjust Initialization
```python
# Old (GO2)
self.scene.robot.init_state.pos = (0.0, 0.0, 2.5)

# New (Spot)
self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
```

### 3. Check Joint Names
```python
# GO2 joint pattern: motor_[fl/fr/hl/hr]_hip[x/y], motor_[fl/fr/hl/hr]_knee
# Spot joint pattern: [fl/fr/hl/hr]_h[xy], [fl/fr/hl/hr]_kn

# Update any reward/event functions that reference joint names:
".*_foot"      # Spot format
".*_hip"       # GO2 format
```

### 4. Retrain Policies
```bash
# Old command (GO2)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Long-Corridor-Go2-v0

# New command (Spot)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Long-Corridor-Spot-v0
```

### 5. Adjust Hyperparameters
```python
# If Spot training is slow:
learning_rate=2.0e-3        # Increase from 1.0e-3
max_learning_rate=2.0e-2    # Increase from 1.0e-2
entropy_coef=0.01           # Increase from 0.0025

# If Spot is unstable:
clip_param=0.15             # Decrease from 0.2
value_loss_coef=0.7         # Increase from 0.5
```

## Control Interface Differences

Both robots use the same environment interface:

```python
# Same for both GO2 and Spot
obs, info = env.reset()
actions = policy(obs)                    # or custom controller
obs, rewards, dones, truncated, info = env.step(actions)
```

**But internally:**
- GO2: Simpler actuator response, faster convergence
- Spot: Realistic delays, non-linear torque response

## Choosing Which Robot to Use

### Use GO2 If:
✅ You want quick training and deployment
✅ You need high-speed locomotion
✅ You want simple control mechanics
✅ You're prototyping/testing

### Use Spot If:
✅ You want realistic Boston Dynamics behavior
✅ You're planning real robot deployment
✅ You need stable, controlled movements
✅ You want to match real hardware dynamics

## Reference: Joint Mappings

### GO2 Joints (12 total)
```
Front Left:   fl_hip_x, fl_hip_y, fl_knee
Front Right:  fr_hip_x, fr_hip_y, fr_knee
Hind Left:    hl_hip_x, hl_hip_y, hl_knee
Hind Right:   hr_hip_x, hr_hip_y, hr_knee
```

### Spot Joints (12 total)
```
Front Left:   fl_hx, fl_hy, fl_kn
Front Right:  fr_hx, fr_hy, fr_kn
Hind Left:    hl_hx, hl_hy, hl_kn
Hind Right:   hr_hx, hr_hy, hr_kn
```

**Note**: Spot's knee joint uses a non-linear transmission ratio defined by the lookup table.

## Troubleshooting

### "GO2 policy doesn't work with Spot"
- Different observation spaces and actuator dynamics
- Retrain from scratch for Spot
- Transfer learning may help as a starting point

### "Spot walks slower than GO2"
- Expected behavior due to more conservative control
- Adjust reward weights to encourage speed
- May also indicate need for more training

### "Spot is unstable"
- Due to joint delays and realistic dynamics
- Reduce learning rate during training
- Increase curriculum difficulty gradually

## Advanced: Using Pre-Trained Spot Policies

If you have a pre-trained Spot policy from another source:

```bash
# Test with existing checkpoint
./isaaclab.sh -p scripts/demos/play_spot_long_corridor.py \
    --checkpoint spot_flat_policy.pt \
    --num_envs 50

# Transfer learning from flat terrain to corridor
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Long-Corridor-Spot-v0 \
    --resume spot_flat_policy.pt \
    --learning_rate 1.0e-4  # Lower LR for fine-tuning
```

---

**Summary**: Both configurations use the same environment framework, but Spot's realistic actuator models make it more challenging to control and slower to train. Choose based on your specific needs!
