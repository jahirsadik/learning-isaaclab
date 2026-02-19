#!/usr/bin/env python3
"""
Synthetic Data Generation using OpenGL Direct Capture

Generates randomized scenes from long_corridor.usda and captures frames
using direct OpenGL screenshot mechanism.

Usage:
    conda activate env_isaaclab
    python scripts/generate_synthetic_video.py
"""

import os
import sys
import argparse
import random
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List
import time

# Initialize Isaac Sim application context first
try:
    from isaacsim import SimulationApp
    
    sim_app = SimulationApp({
        "headless": False,
        "width": 1280,
        "height": 720,
        "anti_aliasing": "smaa",
    })
    
    import omni.usd
    import omni.kit.commands
    from pxr import Usd, UsdGeom, Gf, Sdf
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

    print("[INFO] IsaacSim context initialized")
    
except Exception as e:
    print(f"[ERROR] Failed to initialize IsaacSim: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = Path("/home/jahirsadikmonon/Documents/Projects/usds")
USD_FILE = DATA_DIR / "long_corridor.usda"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "synthetic_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_HEIGHT = 2.5   # World-space Y (height). Increase if camera is still below corridor floor.
CAMERA_OFFSET_X = 0.0

TABLE_A_Z = 50.0
TABLE_B_Z = -55.0

FRAMES_PER_PHASE = 100
TOTAL_FRAMES = FRAMES_PER_PHASE  # A -> B only, no return

# How many sim_app.update() ticks to wait after each capture call before
# moving on. Tune this up if you still get black frames (try 10 or 15).
CAPTURE_SETTLE_FRAMES = 8

COLORS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
]

# ============================================================================
# Utility Functions
# ============================================================================

def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def get_session_dir() -> Path:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = session_dir / "frames"
    frames_dir.mkdir()
    log(f"Session directory: {session_dir}")
    return session_dir

def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)

def save_metadata(session_dir: Path, config: dict):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "usd_file": str(USD_FILE),
        "total_frames": TOTAL_FRAMES,
        "frames_per_phase": FRAMES_PER_PHASE,
        **config
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log("Metadata saved")

# ============================================================================
# Scene Setup
# ============================================================================

def create_stage(usd_file: Path) -> Usd.Stage:
    log("Creating new USD stage...")
    stage = Usd.Stage.CreateInMemory()

    stage.DefinePrim("/World", "Scope")
    stage.DefinePrim("/World/defaultGroundPlane", "Scope")
    stage.DefinePrim("/World/envs", "Scope")
    stage.DefinePrim("/World/envs/env_0", "Scope")

    if usd_file.exists():
        log(f"Loading USD: {usd_file}")
        corridor_prim = stage.DefinePrim("/World/envs/env_0/Corridor", "Xform")
        ref = Sdf.Reference(str(usd_file))
        corridor_prim.GetReferences().AddReference(ref)

        xformable = UsdGeom.Xformable(corridor_prim)
        xformable.AddRotateXOp().Set(-90.0)
        log("USD file referenced and rotated successfully")
    else:
        log(f"WARNING: USD file not found: {usd_file}", "WARN")

    return stage


def create_camera(stage: Usd.Stage, camera_path: str = "/World/envs/env_0/camera") -> str:
    """
    Create a camera prim.
    
    IMPORTANT: Do NOT set an initial position/translate op here.
    update_camera_pose() owns all transform state and will set a single
    xformOp:transform matrix.  Having both an xformOp:translate (set here)
    and an xformOp:transform (added by update_camera_pose) causes USD to
    compose both ops, placing the camera in a garbage location.
    """
    log(f"Creating camera at {camera_path}...")

    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.GetFocalLengthAttr().Set(15.0)

    # Leave the prim with NO xform ops — update_camera_pose will add the
    # single xformOp:transform it needs on the first call.

    return camera_path


def update_camera_pose(stage: Usd.Stage, camera_path: str,
                       position: np.ndarray, look_at: np.ndarray) -> bool:
    """
    Update camera position and orientation.

    Uses a single xformOp:transform so there are no competing ops.
    Clears any pre-existing ops on first call to avoid stale state.
    """
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        log(f"Camera prim not found: {camera_path}", "ERROR")
        return False

    try:
        pos_gf  = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        look_gf = Gf.Vec3d(float(look_at[0]),  float(look_at[1]),  float(look_at[2]))
        up_gf   = Gf.Vec3d(0, 1, 0)

        view_mat      = Gf.Matrix4d().SetLookAt(pos_gf, look_gf, up_gf)
        cam_transform = view_mat.GetInverse()

        xformable = UsdGeom.Xformable(camera_prim)

        # Check whether we already have our transform op
        xform_op = camera_prim.GetAttribute("xformOp:transform")
        if not xform_op.IsValid():
            # First call: clear any ops that might have been set elsewhere
            # (e.g., a stale xformOp:translate from create_camera), then add
            # the single authoritative op.
            xformable.ClearXformOpOrder()
            xform_op = xformable.AddTransformOp()

        xform_op.Set(cam_transform)
        return True

    except Exception as e:
        log(f"update_camera_pose error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def spawn_random_objects(stage: Usd.Stage, session_config: dict) -> dict:
    log("Spawning objects with randomized colors...")

    objects_info = {"table_a": [], "table_b": [], "corridor": []}

    for i in range(3):
        color = random.choice(COLORS)
        name = f"obj_table_a_{i}"
        sphere = UsdGeom.Sphere.Define(stage, f"/World/envs/env_0/{name}")
        sphere.GetRadiusAttr().Set(0.1)
        xformable = UsdGeom.Xformable(sphere.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3f(-0.5 + i * 0.5, 0.0, 0.85))
        objects_info["table_a"].append({"name": name, "color": color})
        log(f"  Table A: {name} -> {color}")

    for i in range(3):
        color = random.choice(COLORS)
        name = f"obj_table_b_{i}"
        sphere = UsdGeom.Sphere.Define(stage, f"/World/envs/env_0/{name}")
        sphere.GetRadiusAttr().Set(0.1)
        xformable = UsdGeom.Xformable(sphere.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3f(-0.5 + i * 0.5, 25.0, 0.85))
        objects_info["table_b"].append({"name": name, "color": color})
        log(f"  Table B: {name} -> {color}")

    session_config["objects"] = objects_info
    return objects_info

# ============================================================================
# Recording
# ============================================================================

def _pump(n: int = 1):
    """Tick the simulator n times."""
    for _ in range(n):
        sim_app.update()


def _bind_viewport_camera(camera_path: str) -> object:
    """
    Bind the active viewport to camera_path and return the viewport handle.
    Must be called AFTER the stage containing that camera is open in the
    usd_context, otherwise the binding is silently lost on the next stage load.
    """
    viewport = get_active_viewport()
    if not viewport:
        log("Could not get active viewport!", "ERROR")
        return None
    viewport.camera_path = camera_path
    log(f"Viewport bound to: {camera_path}")
    return viewport


def animate_and_record(stage: Usd.Stage, camera_path: str,
                       session_dir: Path, session_config: dict,
                       debug: bool = False) -> bool:
    """Animate camera and record frames from the viewport."""

    log("Starting camera animation and frame recording...")

    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # --- FIX: Bind viewport AFTER the correct stage is already open ---
    viewport = _bind_viewport_camera(camera_path)
    if viewport is None:
        return False

    # Warm-up: let textures stream in before recording starts
    log("Warming up renderer (30 frames)...")
    _pump(30)

    # Animation waypoints
    pos_start  = np.array([CAMERA_OFFSET_X, CAMERA_HEIGHT, TABLE_A_Z])
    pos_end    = np.array([CAMERA_OFFSET_X, CAMERA_HEIGHT, TABLE_B_Z])
    look_start = np.array([CAMERA_OFFSET_X, CAMERA_HEIGHT, TABLE_A_Z - 5])
    look_end   = np.array([CAMERA_OFFSET_X, CAMERA_HEIGHT, TABLE_B_Z - 5])

    print("Starting pos: {}, Looking at: {}".format(pos_start, look_start)
            + "\nEnding pos: {}, Looking at: {}".format(pos_end, look_end))

    log(f"Animation path: {pos_start} -> {pos_end}")
    log(f"Camera height (Y): {CAMERA_HEIGHT}")

    frame_count = 0

    def record_phase(phase_name, pos_a, pos_b, look_a, look_b):
        nonlocal frame_count
        log(f"{phase_name} ({FRAMES_PER_PHASE} frames)...")
        for i in range(FRAMES_PER_PHASE):
            t = i / max(FRAMES_PER_PHASE - 1, 1)
            pos  = lerp(pos_a, pos_b, t)
            look = lerp(look_a, look_b, t)
            print("Current pos: {}, Looking at: {}".format(pos, look))

            if not update_camera_pose(stage, camera_path, pos, look):
                log("Failed to update camera pose", "ERROR")
                return False

            # Tick once to push the new camera matrix to the renderer
            _pump(1)

            if not debug:
                frame_path = frames_dir / f"rgb_{frame_count:04d}.png"
                capture_viewport_to_file(viewport, str(frame_path))
                # --- FIX: Wait multiple ticks so async write actually completes ---
                _pump(CAPTURE_SETTLE_FRAMES)

            if (i + 1) % 25 == 0:
                log(f"  {i + 1}/{FRAMES_PER_PHASE} frames")

            frame_count += 1

        return True

    try:
        # A → B only, single linear pass
        if not record_phase("A → B", pos_start, pos_end, look_start, look_end):
            return False

        log(f"Recording complete: {frame_count} frames saved")
        return True

    except Exception as e:
        log(f"Error during recording: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic video data from IsaacSim")
    parser.add_argument("--debug", action="store_true",
                        help="Run animation without saving frames")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Synthetic Video Data Generation")
    if args.debug:
        print("[DEBUG MODE - FRAMES WILL NOT BE SAVED]")
    print("=" * 80 + "\n")

    session_dir = get_session_dir()
    session_config = {}

    try:
        import tempfile
        temp_stage_path = str(Path(tempfile.gettempdir()) / "isaac_synthetic_stage.usda")

        stage = create_stage(USD_FILE)
        spawn_random_objects(stage, session_config)
        camera_path = create_camera(stage)

        stage.Export(temp_stage_path)
        log(f"Stage exported to: {temp_stage_path}")

        # Open the stage in IsaacSim
        usd_context = omni.usd.get_context()
        usd_context.open_stage(temp_stage_path)
        log("Stage opened in IsaacSim viewport")

        # Reload stage reference from the now-live context
        stage = usd_context.get_stage()

        # --- FIX: Viewport binding and recording happen here, AFTER open_stage ---
        if not animate_and_record(stage, camera_path, session_dir, session_config, debug=args.debug):
            log("Animation recording failed", "ERROR")
            return 1

        save_metadata(session_dir, session_config)

        print("\n" + "=" * 80)
        log("Synthetic data generation complete!", "SUCCESS")
        print("=" * 80)
        print(f"\nOutput directory: {session_dir}")
        print(f"Frames:           {session_dir / 'frames'}")
        print(f"\nTo create MP4:")
        print(f"  ffmpeg -framerate 30 -i {session_dir / 'frames'}/rgb_%04d.png "
              f"-c:v libx264 -pix_fmt yuv420p {session_dir / 'video.mp4'}")
        print("=" * 80 + "\n")
        return 0

    except Exception as e:
        log(f"Fatal error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        try:
            print("\n[INFO] Window open for inspection. Close manually or Ctrl+C to exit.")
            while True:
                sim_app.update()
        except KeyboardInterrupt:
            print("\n[INFO] Closing IsaacSim...")
            try:
                sim_app.close()
            except:
                pass


if __name__ == "__main__":
    sys.exit(main())
