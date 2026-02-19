# Quick Start Guide: Synthetic Video Generation

## TL;DR - Get Started in 2 Minutes

```bash
cd ~/Documents/Projects/IsaacLab
./isaaclab.sh -p scripts/generate_synthetic_video.py
```

Outputs appear in: `outputs/synthetic_data/TIMESTAMP/`

---

## Which Script Should I Use?

| Script | Use When | Status |
|--------|----------|--------|
| **generate_synthetic_video.py** | Starting fresh, want simplest working version | ✅ Recommended |
| **generate_synthetic_data_v2.py** | Need advanced Replicator features (depth, normals) | 🔧 Advanced |
| **generate_synthetic_data.py** | Learning the architecture, quick prototyping | 📚 Educational |

---

## What Each Script Does

### generate_synthetic_video.py (Recommended)

```
1. Creates USD stage
2. Loads long_corridor.usda scene
3. Spawns 6 colored spheres (random colors on each table)
4. Creates camera
5. Animates camera: Table A → Table B → Table A (200 frames)
6. Records each frame as PNG
7. Saves metadata (config, object colors, etc.)
```

**Output:**
- `frames/rgb_0000.png` through `rgb_0199.png` (200 images)
- `metadata.json` (scene config)
- `stage.usda` (exported USD)

### Key Features

✅ Works headless (no GUI needed)  
✅ Randomized colors each run  
✅ Camera path interpolation  
✅ Metadata tracking  
✅ Error handling & logging  
✅ Clean, documented code  

---

## Omniverse Extensions Used

| Extension | Purpose | Notes |
|-----------|---------|-------|
| **Replicator** | Frame capture & data export | Core for recording |
| **ImageWriter** | Export frames to disk | Part of Replicator |
| **USD Composer** | USD stage manipulation | For scene management |
| **PhysX** | Physics simulation (optional) | For dynamics later |

**Not needed yet but useful later:**
- **SceneCapture** - Record multiple render passes
- **Write** - High-performance frame writing
- **Randomizer** - Procedural scene variation

---

## Project Structure

```
IsaacLab/
├── scripts/
│   ├── generate_synthetic_video.py          ← START HERE
│   ├── generate_synthetic_data_v2.py
│   ├── generate_synthetic_data.py
│   └── SYNTHETIC_DATA_README.md             ← Full docs
│
├── outputs/
│   └── synthetic_data/
│       └── 20250217_143022/                 ← Each run gets timestamp
│           ├── frames/
│           │   ├── rgb_0000.png
│           │   ├── rgb_0001.png
│           │   └── ...
│           ├── stage.usda
│           └── metadata.json
│
└── source/
    └── isaaclab_tasks/
        └── manager_based/locomotion/velocity/
            └── long_corridor_env_cfg.py     ← Reference config
```

---

## Common Tasks

### Generate a Single Dataset
```bash
./isaaclab.sh -p scripts/generate_synthetic_video.py
```

### Batch Generate 10 Datasets
```bash
for i in {1..10}; do
  echo "Dataset $i/10..."
  ./isaaclab.sh -p scripts/generate_synthetic_video.py
  sleep 2  # Brief pause between runs
done
```

### Convert Frames to MP4
```bash
FRAMES="/path/to/frames"
ffmpeg -framerate 30 -i "$FRAMES/rgb_%04d.png" -c:v libx264 -pix_fmt yuv420p output.mp4
```

### Inspect Metadata
```bash
#!/usr/bin/env python3
import json
from pathlib import Path

session_dir = Path("outputs/synthetic_data/20250217_143022")
metadata = json.loads((session_dir / "metadata.json").read_text())

print(f"Total frames: {metadata['total_frames']}")
print(f"Objects on Table A: {metadata['objects']['table_a']}")
print(f"Objects on Table B: {metadata['objects']['table_b']}")
```

---

## Customization Examples

### Change Number of Objects

Edit `generate_synthetic_video.py`, in `spawn_random_objects()`:

```python
# Change from 3 to 5 objects per table
for i in range(5):  # ← Changed from 3
    color = random.choice(COLORS)
    # ... rest of code
```

### Change Camera Path

Edit the position/look-at vectors:

```python
# Slower, higher path
pos_start = np.array([CAMERA_OFFSET_X, TABLE_A_Y, CAMERA_HEIGHT + 1.0])  # Added 1m height
FRAMES_PER_PHASE = 200  # ← Double the frames for slower movement
```

### Add New Colors

```python
COLORS = [
    (1.0, 0.0, 0.0),      # Red
    (0.0, 1.0, 0.0),      # Green
    (0.0, 0.0, 1.0),      # Blue
    (1.0, 1.0, 0.0),      # Yellow
    (1.0, 0.0, 1.0),      # Magenta
    (0.0, 1.0, 1.0),      # Cyan
    (0.5, 0.5, 0.5),      # Gray        ← NEW
    (1.0, 0.5, 0.0),      # Orange     ← NEW
]
```

### Record Higher Resolution

```python
# Change from 1280×720 to 1920×1080
render_product = rep.create.render_product(
    camera_path=camera_path,
    resolution=(1920, 1080),  # ← 1080p
)
```

---

## Troubleshooting Checklist

### Script won't start

- [ ] Using `./isaaclab.sh -p` (not `./isaaclab.sh --headless`)
- [ ] Paths are correct: `/home/jahirsadikmonon/Documents/Projects/usds/long_corridor.usda`
- [ ] Disk space available: `df -h ~`
- [ ] Python 3.10+ installed: `python --version`

### No frames generated

- [ ] Check output directory created: `ls outputs/synthetic_data/`
- [ ] Check frames subdirectory: `ls outputs/synthetic_data/TIMESTAMP/frames/`
- [ ] Verify permissions: `touch outputs/synthetic_data/test.txt`
- [ ] Check logs for errors (printed during execution)

### Frames are black/blank

- [ ] Camera might not be properly positioned
- [ ] Try non-headless: `./isaaclab.sh --show-gui scripts/generate_synthetic_video.py`
- [ ] Verify USD scene loaded (check stage.usda file size)

### Out of memory

- [ ] Reduce frames: `FRAMES_PER_PHASE = 50` (instead of 100)
- [ ] Lower resolution: `resolution=(640, 480)`
- [ ] Process on GPU with less VRAM

---

## File Locations Reference

| Item | Path |
|------|------|
| Input USD Scene | `/home/jahirsadikmonon/Documents/Projects/usds/long_corridor.usda` |
| Scripts Dir | `/home/jahirsadikmonon/Documents/Projects/IsaacLab/scripts/` |
| Main Script | `/home/jahirsadikmonon/Documents/Projects/IsaacLab/scripts/generate_synthetic_video.py` |
| Output Base | `/home/jahirsadikmonon/Documents/Projects/IsaacLab/outputs/synthetic_data/` |
| Reference Config | `/home/jahirsadikmonon/Documents/Projects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/long_corridor_env_cfg.py` |

---

## Next Steps

1. ✅ **Run the script** (you are here)
   ```bash
   ./isaaclab.sh --headless python scripts/generate_synthetic_video.py
   ```

2. 📊 **Inspect outputs**
   ```bash
   ls -la outputs/synthetic_data/*/frames/ | head -20
   ```

3. 🎬 **Create MP4 (optional)**
   ```bash
   ffmpeg -framerate 30 -i outputs/synthetic_data/TIMESTAMP/frames/rgb_%04d.png video.mp4
   ```

4. 🔧 **Customize for your needs**
   - Edit colors, objects, camera path
   - See SYNTHETIC_DATA_README.md for full docs

5. 📈 **Scale up**
   - Batch generation script
   - Multiple camera paths
   - Integrate with training pipeline

---

## Performance Expectations

| Operation | Time | Hardware |
|-----------|------|----------|
| Setup | 5-10s | RTX 4090 |
| Record 200 frames | 30-60s | RTX 4090 |
| Total per run | ~1-2 min | RTX 4090 |
| 10 datasets | 10-20 min | RTX 4090 |
| 100 datasets | 100-200 min | RTX 4090 |

---

## Key Concepts

### Headless Mode
- `--headless`: Run without GUI (faster, uses less RAM)
- Perfect for batch generation on servers/clusters
- Use `--show-gui` to debug visually

### Frame Format
- **PNG sequences**: Individual images, good for streaming/parallelization
- **MP4 video**: Single file, better for storage efficiency
- Can convert to either as needed

### Metadata Tracking
- `metadata.json` saves scene configuration
- Enables reproducibility and data tracking
- Include labels/annotations for ML training

---

## API Quick Reference

### Essential Imports
```python
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, Gf
```

### Create Camera
```python
camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera.GetFocalLengthAttr().Set(35.0)
```

### Create Sphere
```python
sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
sphere.GetRadiusAttr().Set(0.5)
```

### Set Position
```python
sphere.GetPrim().GetAttribute("xformOp:translate").Set((x, y, z))
```

### Record Frames
```python
render_product = rep.create.render_product(
    camera_path="/World/Camera",
    resolution=(1280, 720),
)
writer = rep.writers.get("BasicWriter")
writer.initialize(output_dir="/path")
writer.attach(render_product)
```

---

## Resources

- **IsaacSim Replicator Docs**: https://docs.isaacsim.omniverse.nvidia.com (linked in original message)
- **USD Documentation**: https://graphics.pixar.com/usd/
- **IsaacLab Docs**: Internal project documentation
- **Replicator Tutorials**: Within IsaacSim distribution

---

## Questions?

If scripts don't work:
1. Check this checklist
2. Review SYNTHETIC_DATA_README.md (full documentation)
3. Inspect script error logs
4. Try with `--show-gui` to debug visually

Happy generating! 🎉
