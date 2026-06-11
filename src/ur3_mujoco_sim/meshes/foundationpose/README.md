# FoundationPose CAD assets (textured OBJ)

Blender exports (OBJ+MTL+JPG) are converted with UVs preserved:

```bash
cd src/ur3_mujoco_sim
python3 scripts/blender_obj_to_foundationpose.py --input-dir ~/Downloads \\
  square_1 cylinder_1 cylinder_2
```

This scales mm → m (`--scale 0.001`), centers the mesh at the origin, and copies textures as `*_texture.jpg`.

Legacy untextured OBJs from STL can still be built with `scripts/stl_to_centered_obj.py`.

## YOLO class → mesh + texture

See `mesh_texture_map.yaml`. Before Isaac launch:

```bash
export FOUNDATION_POSE_OBJECT=Box_1        # or Cylinder_1 / Cylinder_2
source ~/isaac_ros_assets/setup_foundationpose_env.sh
```

| YOLO class   | Mesh             | Texture                 |
|--------------|------------------|-------------------------|
| `Box_1`      | `square_1.obj`   | `square_1_texture.jpg`  |
| `Cylinder_1` | `cylinder_1.obj` | `cylinder_1_texture.jpg`|
| `Cylinder_2` | `cylinder_2.obj` | `cylinder_2_texture.jpg`|

## Isaac ROS `symmetry_axes` (cylinders)

Cylinders have continuous symmetry about mesh **+Y** — use `y_full` in `symmetry_axes_by_class` (see `ur3_action_sequence.yaml`). The sequencer exports `FOUNDATION_POSE_SYMMETRY_AXES` when launching Isaac.

See: [Isaac FoundationPose mesh center](https://nvidia-isaac-ros.github.io/concepts/pose_estimation/foundationpose/tutorial_shift_mesh_center.html).
