# Real gripper (Dynamixel XM540)

| File | Purpose |
|------|---------|
| `dynamixel_adaptive_grasp.py` | Current-based position adaptive close (U2D2 + XM540-W270-R) |

## Standalone test

```bash
ls-serial   # from ~/.bash_aliases
python3 ~/ur3_control/gripper/dynamixel_adaptive_grasp.py --no-prompt
```

Keep torque on after close (for lift):

```bash
python3 ~/ur3_control/gripper/dynamixel_adaptive_grasp.py --no-prompt --keep-torque
```

## Action sequencer

`gripper_hybrid_close` uses this script when `robot_backend: real` (see `ur3_action_sequence_real.yaml` → `gripper:`).
