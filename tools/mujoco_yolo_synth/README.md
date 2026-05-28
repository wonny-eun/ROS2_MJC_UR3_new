### Goal
Generate **synthetic RGB + segmentation labels** from **MuJoCo** and train **YOLOv8-seg**.

This uses `mujoco.Renderer(...).render(segmentation=True)` to get per-pixel instance IDs and writes labels in **YOLOv8-seg TXT** polygon format.

---

### 1) Install dependencies

#### System (Ubuntu / ROS 2 Humble)

You already have MuJoCo via `mujoco_vendor` (e.g. `/opt/ros/humble/opt/mujoco_vendor`).

Install Python deps:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install mujoco opencv-python numpy ultralytics pyyaml
```

If you want CUDA training, install the appropriate PyTorch build first (see PyTorch instructions), then:

```bash
python3 -m pip install ultralytics
```

---

### 2) Prepare your MuJoCo model for randomization

The generator needs:
- **One movable object** controlled by a **FREE joint** (so we can randomize pose)
- One or more **geom names** that belong to that object (used for segmentation filtering)

In your MJCF, define something like:
- a body with `freejoint` (or a joint of type `free`)
- geoms inside it with stable `name="..."` attributes

Optionally (for texture/material randomization):
- define multiple `material` entries in `<asset>`
- pass `--material-ids ...` to randomly swap `geom_matid` among them

---

### 3) Generate the dataset

From your project root (`/home/wonny/ur3_control`), run:

```bash
python3 /home/wonny/ur3_control/tools/mujoco_yolo_synth/generate_yolo_seg_dataset.py \
  --model ~/ur3_control/src/ROS2_MuJoCo_UR3/ur3_mujoco_sim/models/ur3_scene_table.xml \
  --out ~/datasets/ur3_yolo_data \
  --n 2000 \
  --val-split 0.1 \
  --width 640 --height 480 \
  --class-name object \
  --object-geom-names sphere_geom cylinder_geom box_geom \
  --freejoint-joint-name target_free_joint \
  --randomize-color \
  --randomize-lighting \
  --rgb-noise-sigma 3.0 \
  --depth-noise-sigma-m 0.002 \
  --depth-quant-step-m 0.001
```

#### UR3 table multi-object dataset (center **and** corners in the image)

Default `--cam-frame-placement mixed` offsets the camera look-at so objects are **not always centered** (corners/margins included). Use `--cam-frame-placement center` only if you want the old always-centered bias.

```bash
python3 /home/wonny/ur3_control/tools/mujoco_yolo_synth/generate_yolo_seg_dataset.py \
  --model /home/wonny/ur3_control/src/ROS2_MuJoCo_UR3/src/ur3_mujoco_sim/mjcf/ur3_scene_table.xml \
  --out /home/wonny/ur3_control/data/yolo_multi_drop \
  --n 1500 --val-split 0.1 --seed 0 \
  --width 640 --height 480 \
  --object-geom-names cylinder_1_vis cylinder_2_vis square_1_vis \
  --class-map cylinder_1_vis:Cylinder_1 cylinder_2_vis:Cylinder_2 square_1_vis:Box_1 \
  --freejoint-body-names cylinder_1_obj cylinder_2_obj square_1_obj \
  --randomize-color --randomize-lighting --randomize-surface \
  --cam-frame-placement mixed --cam-frame-offcenter-frac 0.55 \
  --cam-lookat-offset-xy-m 0.12 --cam-min-dist-m 0.1
```

Output structure:

```text
<out>/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  dataset.yaml
```

---

### 4) Train YOLOv8-seg with Ultralytics

```bash
cd /tmp/yolo_synth_dataset

# Train a segmentation model
yolo task=segment mode=train model=yolov8n-seg.pt data=dataset.yaml imgsz=640 epochs=50 batch=16
```

The best weights are typically saved under:

```text
runs/segment/train/weights/best.pt
```

---

### 5) Quick inference check

```bash
yolo task=segment mode=predict model=runs/segment/train/weights/best.pt source=images/val save=True
```

---

### Notes / troubleshooting

- **Segmentation API mismatch**: If you get an error about `segmentation=True`, upgrade `mujoco` Python:

```bash
python3 -m pip install --upgrade mujoco
```

- **Label quality**: Polygon approximation is controlled by:
  - `--poly-eps-px` (bigger = fewer points)
  - `--min-mask-area-px`
  - `--max-polys` for disconnected components

- **Depth is optional**: YOLOv8-seg trains on RGB; depth is saved only if you pass `--save-depth`.

