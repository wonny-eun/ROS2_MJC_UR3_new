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

Cylinders have a continuous symmetry about their **geometric axis**. After OBJ export, verify which local axis (`x`, `y`, or `z`) aligns with the cylinder axis in your mesh (e.g. open in MeshLab). Then set `symmetry_axes` in `isaac_ros_foundationpose` accordingly (e.g. `z_full` for full rotation steps about local +Z). Boxes typically omit symmetry.

See: [Isaac FoundationPose mesh center](https://nvidia-isaac-ros.github.io/concepts/pose_estimation/foundationpose/tutorial_shift_mesh_center.html).
