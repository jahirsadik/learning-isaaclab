#!/usr/bin/env python3
"""
Long Corridor environment — with optional camera recording.
"""

# ============================================================================
# STEP 1: AppLauncher MUST come before all isaaclab/carb imports.
# ============================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Long corridor env without a robot.")
parser.add_argument("--num-envs",      type=int,  default=1,    help="Number of parallel environments.")
parser.add_argument("--camera",        action="store_true",      help="Enable viewport recording.")
parser.add_argument("--frames",        type=int,  default=100,  help="Number of frames to record (--camera only).")
parser.add_argument("--debug-camera",  action="store_true",      help="Animate camera without saving frames.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# --- FIX 1: Always enable cameras if the camera argument is passed ---
# The Camera sensor class requires this flag to be True even in GUI mode.
if args.camera:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ============================================================================
# STEP 2: All other imports
# ============================================================================
import os
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image

import omni.usd
from pxr import UsdGeom, Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import UsdFileCfg

# Native IsaacLab Sensors
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# ============================================================================
# Scene / env configuration flags
# ============================================================================
DATA_DIR               = "/home/jahirsadikmonon/Documents/Projects/usds"
NUM_CUBOIDS            = 5
SPACING                = 30.0
SPAWN_CUBOIDS_IN_PATH  = True
SPAWN_TABLE_B_OBJECTS  = False
SPAWN_TABLE_A_OBJECTS  = False

DIFFUSE_COLORS = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
    (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
]

# ============================================================================
# Camera / recording configuration
# ============================================================================
def get_camera_prim_path(env_id: int = 0) -> str:
    """Resolve the camera prim path for a specific env index."""
    return f"/World/envs/env_{env_id}/RecordingCamera"

CAMERA_FOCAL_LENGTH = 15.0

CAMERA_START = np.array([1.0,  -50.0,  2.5])
CAMERA_END   = np.array([1.0,   55.0,  2.5])

CAMERA_LOOK_START = CAMERA_START + np.array([0.0,  5.0, 0.0])
CAMERA_LOOK_END   = CAMERA_END   + np.array([0.0,  5.0, 0.0])

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "synthetic_data"

# ============================================================================
# Scene definition
# ============================================================================
@configclass
class NoRobotSceneCfg(InteractiveSceneCfg):
    """Corridor scene without a robot."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(1000000, 1000)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

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

    # --- FIX 2: Increased Resolution for Better Quality ---
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RecordingCamera",
        update_period=0.0,
        height=1080,          # Increased from 720
        width=1920,           # Increased from 1280
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )

    if SPAWN_TABLE_A_OBJECTS:
        cuboid_A = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CuboidA",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

    if SPAWN_TABLE_B_OBJECTS:
        cuboid_B = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CuboidB",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
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
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=DIFFUSE_COLORS[i % len(DIFFUSE_COLORS)], metallic=0.4
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
        del i

@configclass
class NoRobotObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()

@configclass
class NoRobotEventCfg:
    if SPAWN_CUBOIDS_IN_PATH:
        spawn_cuboids_in_path = EventTerm(
            func=mdp.spawn_objects_in_location,
            mode="reset",
            params={
                "starting_asset_cfg": SceneEntityCfg("table_A"),
                "asset_cfgs": [SceneEntityCfg(f"cuboid_in_path_{i}") for i in range(NUM_CUBOIDS)],
                "number_of_objects": NUM_CUBOIDS,
                "position_offsets": [(0.0, SPACING, 0.0)] * NUM_CUBOIDS,
            },
        )

@configclass
class EmptyActionsCfg:
    pass

@configclass
class LongCorridorNoRobotEnvCfg(ManagerBasedEnvCfg):
    scene:        NoRobotSceneCfg        = NoRobotSceneCfg(num_envs=args.num_envs, env_spacing=60.0)
    observations: NoRobotObservationsCfg = NoRobotObservationsCfg()
    actions:      EmptyActionsCfg        = EmptyActionsCfg()
    events:       NoRobotEventCfg        = NoRobotEventCfg()

    def __post_init__(self):
        self.decimation          = 4
        self.episode_length_s    = 60.0
        self.viewer.eye          = [0.0, -25.0, 3.0]
        self.viewer.lookat       = [0.0, -30.0, 3.0]
        self.sim.dt              = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

# ============================================================================
# Camera helpers
# ============================================================================
def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)

def _set_camera_pose(stage, camera_path: str, position: np.ndarray, look_at: np.ndarray) -> bool:
    """Set camera world-space pose via a single xformOp:transform."""
    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        return False

    pos_gf  = Gf.Vec3d(*position.tolist())
    look_gf = Gf.Vec3d(*look_at.tolist())
    up_gf   = Gf.Vec3d(0.0, 0.0, 1.0) 

    view_mat      = Gf.Matrix4d().SetLookAt(pos_gf, look_gf, up_gf)
    cam_transform = view_mat.GetInverse()

    xformable = UsdGeom.Xformable(prim)
    xform_op  = prim.GetAttribute("xformOp:transform")
    if not xform_op.IsValid():
        xformable.ClearXformOpOrder()
        xform_op = xformable.AddTransformOp()

    xform_op.Set(cam_transform)
    return True

def _pump(env: ManagerBasedEnv, zero_actions: torch.Tensor, n: int = 1):
    """Step the environment (keeps physics live) n times."""
    for _ in range(n):
        env.step(zero_actions)

def record_camera_sweep(env: ManagerBasedEnv, zero_actions: torch.Tensor):
    """Animate the camera from CAMERA_START to CAMERA_END while recording frames."""
    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / session_id
    frames_dir  = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Recording to: {frames_dir}")

    stage = omni.usd.get_context().get_stage()

    # Set camera to the start pose BEFORE warming up
    for env_id in range(env.num_envs):
        _set_camera_pose(stage, get_camera_prim_path(env_id), CAMERA_START, CAMERA_LOOK_START)

    print("[INFO] Warming up renderer (30 ticks)...")
    _pump(env, zero_actions, 30)

    n_frames = args.frames
    print(f"[INFO] Recording {n_frames} frames  |  start={CAMERA_START}  end={CAMERA_END}")

    for i in range(n_frames):
        t    = i / max(n_frames - 1, 1)
        pos  = _lerp(CAMERA_START, CAMERA_END, t)
        look = _lerp(CAMERA_LOOK_START, CAMERA_LOOK_END, t)

        # Move ALL env cameras in sync
        for env_id in range(env.num_envs):
            _set_camera_pose(stage, get_camera_prim_path(env_id), pos, look)

        # Step physics. IsaacLab's camera sensor automatically updates and captures here!
        env.step(zero_actions)

        if not args.debug_camera:
            frame_path = frames_dir / f"rgb_{i:04d}.png"
            
            # Read the RGB tensor directly from the camera sensor (from env_0)
            camera_data = env.scene.sensors["camera"].data.output["rgb"]
            
            if camera_data is not None:
                # Get the image for environment 0, move to CPU, convert to NumPy
                rgb_tensor = camera_data[0]
                rgb_data = rgb_tensor.cpu().numpy()
                
                # Strip the alpha channel if the sensor returned RGBA
                if rgb_data.shape[-1] == 4:
                    rgb_data = rgb_data[:, :, :3]
                
                # Convert to standard 8-bit image format for saving
                if rgb_data.dtype != np.uint8:
                    if rgb_data.max() <= 1.0: # Sometimes tensors are normalized floats
                        rgb_data = (rgb_data * 255).astype(np.uint8)
                    else:
                        rgb_data = rgb_data.astype(np.uint8)
                        
                img = Image.fromarray(rgb_data)
                img.save(str(frame_path))
            else:
                print(f"[WARN] Frame {i}: Camera sensor returned no data.")

        if (i + 1) % 25 == 0:
            print(f"[INFO]   {i + 1}/{n_frames} frames recorded")

    # Save metadata
    meta = {
        "timestamp":    datetime.now().isoformat(),
        "n_frames":     n_frames,
        "camera_start": CAMERA_START.tolist(),
        "camera_end":   CAMERA_END.tolist(),
        "focal_length": CAMERA_FOCAL_LENGTH,
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[INFO] Recording complete — {n_frames} frames saved to {frames_dir}")
    print(f"[INFO] To create MP4:")
    print(f"         ffmpeg -framerate 30 -i {frames_dir}/rgb_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p {session_dir}/video.mp4")

# ============================================================================
# Main
# ============================================================================
def main():
    env_cfg = LongCorridorNoRobotEnvCfg()
    env     = ManagerBasedEnv(cfg=env_cfg)

    print(f"[INFO] Environment created — {env.num_envs} env(s).")
    print(f"[INFO] Camera recording: {'ON' if args.camera else 'OFF'}")

    env.sim.play()
    env.reset()

    zero_actions = torch.zeros(env.num_envs, 0, device=env.device)

    if args.camera:
        try:
            record_camera_sweep(env, zero_actions)
        except Exception as e:
            import traceback
            print(f"[ERROR] Recording failed: {e}")
            traceback.print_exc()
        finally:
            print("\n[INFO] Window open for inspection. Ctrl+C to exit.")
            try:
                while True:
                    env.step(zero_actions)
            except KeyboardInterrupt:
                pass
    else:
        step = 0
        try:
            while True:
                with torch.inference_mode():
                    env.step(zero_actions)
                step += 1
                if step % 100 == 0:
                    print(f"[INFO] Step {step}")
        except KeyboardInterrupt:
            print("[INFO] Ctrl+C received, shutting down.")
        except Exception as e:
            import traceback
            print(f"[ERROR] Loop crashed: {e}")
            traceback.print_exc()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()