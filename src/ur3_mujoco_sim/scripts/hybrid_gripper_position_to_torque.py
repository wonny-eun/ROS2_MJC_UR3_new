#!/usr/bin/env python3
"""
Hybrid position-to-current (torque) gripper controller for MuJoCo.

Demonstrates:
  State 0 — position-controlled closing (0.0 → 1.396 rad)
  State 1 — contact detected via actuator feedback force + low joint velocity
  State 2 — direct torque (current) hold without external touch/force sensors

Actuator name matches UR3_RG2.xml: ``gripper_motor_joint_pos``.

MuJoCo 3.x note:
  There is no ``mjGAIN_DIRECT`` enum. Direct torque/current drive is implemented as
  ``mjGAIN_FIXED`` with gain=1 and ``mjBIAS_NONE`` (equivalent to a ``<motor>`` actuator).
"""

from __future__ import annotations

import argparse
import enum
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

try:
    import mujoco.viewer as mj_viewer
except ImportError:  # pragma: no cover - older mujoco builds
    mj_viewer = None

SCRIPT_DIR = Path(__file__).resolve().parent
MJCF_DIR = SCRIPT_DIR.parent / "mjcf"
DEFAULT_UR3_SCENE = MJCF_DIR / "ur3_scene_table.xml"
UR3_ROBOT_ONLY = MJCF_DIR / "UR3_RG2.xml"

# ---------------------------------------------------------------------------
# Robot / actuator identifiers (match UR3_RG2.xml)
# ---------------------------------------------------------------------------
ACTUATOR_NAME = "gripper_motor_joint_pos"
JOINT_NAME = "gripper_motor_joint"

GRIP_OPEN_RAD = 0.0
GRIP_CLOSED_RAD = 1.396
ACTUATOR_CTRL_MIN = 0.0
ACTUATOR_CTRL_MAX = 1.396
ACTUATOR_FORCE_LIMIT_NM = 10.0  # forcerange in XML


class GripperState(enum.IntEnum):
    POSITION_APPROACH = 0
    SWITCH_TO_TORQUE = 1
    TORQUE_HOLD = 2


@dataclass(frozen=True)
class ContactDetectConfig:
    """Thresholds for inferring contact from motor feedback only."""

    min_feedback_torque_nm: float = 2.0
    max_joint_speed_rad_s: float = 0.15
    confirm_steps: int = 5
    # Position stall: closing commanded but joint barely moves while torque is high.
    min_position_error_rad: float = 0.02
    stall_window_steps: int = 10
    stall_max_delta_rad: float = 0.002


@dataclass(frozen=True)
class GripperControlConfig:
    position_kp: float = 500.0
    position_kv: float = 30.0
    close_position_rad: float = GRIP_CLOSED_RAD
    hold_torque_nm: float = 2.5
    # Ramp the position setpoint during approach to avoid slamming into force limits.
    approach_ramp_rad_s: float = 0.35
    contact: ContactDetectConfig = ContactDetectConfig()
    motor_torque_constant_nm_per_a: float = 0.05


class MotorCurrentMapper:
    """
    Map MuJoCo actuator torque (N·m) to a motor-driver current register (mA).

    Real servos:  τ = Kt * I  →  I_A = τ / Kt  →  I_mA = 1000 * τ / Kt

    Example: Kt = 0.05 N·m/A, τ = 2.5 N·m  →  I = 50 A  →  register = 50000 mA
    (Use your datasheet Kt; many small gripper motors are 0.02–0.10 N·m/A.)
    """

    def __init__(self, kt_nm_per_a: float) -> None:
        if kt_nm_per_a <= 0.0:
            raise ValueError("kt_nm_per_a must be positive")
        self._kt = float(kt_nm_per_a)

    def torque_nm_to_current_ma(self, torque_nm: float) -> float:
        return 1000.0 * float(torque_nm) / self._kt

    def current_ma_to_torque_nm(self, current_ma: float) -> float:
        return (float(current_ma) / 1000.0) * self._kt


class HybridGripperController:
    """Runtime position ↔ direct-torque actuator reconfiguration."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: GripperControlConfig) -> None:
        self.model = model
        self.data = data
        self.cfg = cfg
        self.mapper = MotorCurrentMapper(cfg.motor_torque_constant_nm_per_a)

        self.actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR_NAME)
        if self.actuator_id < 0:
            raise RuntimeError(f"Actuator {ACTUATOR_NAME!r} not found in model")

        self.joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)
        if self.joint_id < 0:
            raise RuntimeError(f"Joint {JOINT_NAME!r} not found in model")

        self.dof_adr = int(model.jnt_dofadr[self.joint_id])
        self.qpos_adr = int(model.jnt_qposadr[self.joint_id])

        self.state = GripperState.POSITION_APPROACH
        self._contact_counter = 0
        self._approach_setpoint_rad = GRIP_OPEN_RAD
        self._stall_q_history: list[float] = []

        # Cache original position-actuator PD parameters for restore/debug.
        self._pos_kp = float(model.actuator_gainprm[self.actuator_id, 0])
        self._pos_kv = float(-model.actuator_biasprm[self.actuator_id, 2])

        self.set_position_mode()

    # --- Actuator mode switching -------------------------------------------------

    def set_position_mode(self) -> None:
        """
        Position servo (XML ``<position kp=... kv=...>``).

        force = gain * ctrl + bias
        with gain = kp, bias = -kp*q - kv*qd  (affine bias on joint pos/vel)
        """
        aid = self.actuator_id
        m = self.model
        kp = self.cfg.position_kp
        kv = self.cfg.position_kv

        m.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[aid, :] = 0.0
        m.actuator_gainprm[aid, 0] = kp
        m.actuator_biasprm[aid, :] = 0.0
        m.actuator_biasprm[aid, 1] = -kp
        m.actuator_biasprm[aid, 2] = -kv

        self.data.ctrl[self.actuator_id] = np.clip(
            float(self.data.qpos[self.qpos_adr]),
            ACTUATOR_CTRL_MIN,
            ACTUATOR_CTRL_MAX,
        )

    def set_direct_torque_mode(self) -> None:
        """
        Direct torque/current mode.

        MuJoCo 3.x equivalent of legacy "direct gain":
          gaintype = mjGAIN_FIXED, gainprm[0] = 1
          biastype = mjBIAS_NONE
          ctrl     = commanded torque (N·m)
        """
        aid = self.actuator_id
        m = self.model

        m.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_gainprm[aid, :] = 0.0
        m.actuator_gainprm[aid, 0] = 1.0
        m.actuator_biasprm[aid, :] = 0.0

    # --- Feedback ----------------------------------------------------------------

    def feedback_torque_nm(self) -> float:
        """
        Actuator force/torque (N·m) — primary motor feedback proxy.

        ``data.actuator_force[actuator_id]`` is the scalar force/torque produced
        by this actuator after the force limit (``forcerange`` in XML).
        """
        return float(self.data.actuator_force[self.actuator_id])

    def joint_feedback_torque_nm(self) -> float:
        """Same torque expressed on the joint DOF via ``data.qfrc_actuator``."""
        return float(self.data.qfrc_actuator[self.dof_adr])

    def joint_velocity_rad_s(self) -> float:
        return float(self.data.qvel[self.dof_adr])

    def joint_position_rad(self) -> float:
        return float(self.data.qpos[self.qpos_adr])

    # --- State machine -----------------------------------------------------------

    def _position_stalled_while_closing(self) -> bool:
        """True when the joint stops advancing despite a large close error."""
        cfg = self.cfg.contact
        q = self.joint_position_rad()
        self._stall_q_history.append(q)
        if len(self._stall_q_history) > cfg.stall_window_steps:
            self._stall_q_history.pop(0)

        if len(self._stall_q_history) < cfg.stall_window_steps:
            return False

        pos_error = self._approach_setpoint_rad - q
        delta_q = max(self._stall_q_history) - min(self._stall_q_history)
        return pos_error >= cfg.min_position_error_rad and delta_q <= cfg.stall_max_delta_rad

    def contact_detected(self) -> bool:
        """
        Contact inferred when closing under position control stalls AND
        feedback torque rises (motor working against object / force limit).

        Primary condition (ideal contact):
          |actuator_force| >= threshold  AND  |qvel| <= threshold

        Secondary condition (stall under load, robust to brief limit-cycle bounce):
          closing position error + high torque + joint barely moving over a window.

        ``data.actuator_force`` is in N·m (same units as real motor shaft torque).
        Map to driver current via I_mA = 1000 * tau / Kt.
        """
        tau = self.feedback_torque_nm()
        qd = abs(self.joint_velocity_rad_s())
        cfg = self.cfg.contact
        closing = self._approach_setpoint_rad > self.joint_position_rad() + 1e-4
        torque_spike = tau >= cfg.min_feedback_torque_nm if closing else abs(tau) >= cfg.min_feedback_torque_nm

        low_speed = qd <= cfg.max_joint_speed_rad_s
        stalled = self._position_stalled_while_closing()

        if torque_spike and (low_speed or stalled):
            self._contact_counter += 1
        else:
            self._contact_counter = 0

        return self._contact_counter >= cfg.confirm_steps

    def step_control(self) -> None:
        """Call once per simulation step *before* ``mujoco.mj_step``."""
        dt = float(self.model.opt.timestep)

        if self.state == GripperState.POSITION_APPROACH:
            self._approach_setpoint_rad = min(
                self._approach_setpoint_rad + self.cfg.approach_ramp_rad_s * dt,
                self.cfg.close_position_rad,
            )
            self.data.ctrl[self.actuator_id] = self._approach_setpoint_rad
            if self.contact_detected():
                self.state = GripperState.SWITCH_TO_TORQUE

        elif self.state == GripperState.SWITCH_TO_TORQUE:
            hold_nm = float(np.clip(self.cfg.hold_torque_nm, -ACTUATOR_FORCE_LIMIT_NM, ACTUATOR_FORCE_LIMIT_NM))
            self.set_direct_torque_mode()
            self.data.ctrl[self.actuator_id] = hold_nm
            self.state = GripperState.TORQUE_HOLD

        elif self.state == GripperState.TORQUE_HOLD:
            hold_nm = float(np.clip(self.cfg.hold_torque_nm, -ACTUATOR_FORCE_LIMIT_NM, ACTUATOR_FORCE_LIMIT_NM))
            self.data.ctrl[self.actuator_id] = hold_nm


# Minimal self-contained scene (object in gripper closing path).
DEMO_SCENE_XML = """
<mujoco model="hybrid_gripper_demo">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8"/>
  </visual>
  <default>
    <joint damping="0.3"/>
    <geom friction="1.0 0.05 0.001" solref="0.005 1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <body name="gripper_base" pos="0 0 0.15">
      <geom name="palm" type="box" size="0.03 0.04 0.015" rgba="0.3 0.3 0.35 1"/>
      <body name="finger" pos="0 0.05 0">
        <joint name="gripper_motor_joint" type="hinge" axis="0 0 1"
               range="0 1.396" damping="0.15"/>
        <geom name="finger_geom" type="box" size="0.015 0.06 0.015" pos="0 0.06 0"
              rgba="0.8 0.2 0.2 1"/>
      </body>
    </body>
    <!-- Fixed grasp target in the finger sweep (contact ~0.75 rad with ramped close). -->
    <geom name="grasp_target" type="box" pos="0.04 0.17 0.15" size="0.025 0.025 0.04"
          rgba="0.2 0.5 0.9 1" contype="1" conaffinity="1"/>
  </worldbody>
  <actuator>
    <position name="gripper_motor_joint_pos" joint="gripper_motor_joint"
              ctrlrange="0 1.396" kp="300" kv="25" forcerange="-10 10"/>
  </actuator>
</mujoco>
"""


def resolve_model_path(user_path: Optional[str], *, use_ur3_scene: bool) -> mujoco.MjModel:
    if use_ur3_scene:
        path = DEFAULT_UR3_SCENE
    elif user_path:
        path = Path(user_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Use the full path, e.g.\n  {SCRIPT_DIR / 'hybrid_gripper_position_to_torque.py'}\n"
                f"Do not paste literal '.../' from docs."
            )
        path = path.resolve()
    else:
        print("Using built-in demo scene (standalone gripper + grasp target).")
        return mujoco.MjModel.from_xml_string(DEMO_SCENE_XML)

    if path.resolve() == UR3_ROBOT_ONLY.resolve():
        print(
            "WARNING: UR3_RG2.xml is robot-only (no table/objects). "
            "Contact detection will not trigger in free air.\n"
            f"Prefer: python3 {Path(__file__).name} --ur3-scene --viewer"
        )

    print(f"Loading MJCF: {path}")
    return mujoco.MjModel.from_xml_path(str(path))


def run_sim(
    model: mujoco.MjModel,
    *,
    use_viewer: bool,
    max_steps: int,
    log_hz: float,
    real_time: bool,
    hold_sec: float,
) -> None:
    data = mujoco.MjData(model)
    ctrl = HybridGripperController(model, data, GripperControlConfig())

    # Start fully open.
    data.qpos[ctrl.qpos_adr] = GRIP_OPEN_RAD
    data.ctrl[ctrl.actuator_id] = GRIP_OPEN_RAD
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    log_stride = max(1, int(round((1.0 / max(log_hz, 1e-6)) / dt)))
    hold_steps = max(0, int(round(hold_sec / dt)))
    t0 = time.monotonic()
    contact_step: Optional[int] = None
    step = 0

    def _log_status(step_idx: int) -> None:
        tau_act = ctrl.feedback_torque_nm()
        tau_joint = ctrl.joint_feedback_torque_nm()
        i_ma = ctrl.mapper.torque_nm_to_current_ma(tau_act)
        print(
            f"[step {step_idx:6d}] state={ctrl.state.name} "
            f"q={ctrl.joint_position_rad():.4f} rad  qd={ctrl.joint_velocity_rad_s():+.4f} rad/s  "
            f"tau_act={tau_act:+.3f} N·m  tau_joint={tau_joint:+.3f} N·m  "
            f"~I={i_ma:+.0f} mA  ctrl={float(data.ctrl[ctrl.actuator_id]):+.3f}",
            flush=True,
        )

    def _sim_step() -> bool:
        """Advance one step. Return False when the headless run should stop."""
        nonlocal step, contact_step

        step_start = time.monotonic()
        ctrl.step_control()
        mujoco.mj_step(model, data)
        if step % log_stride == 0:
            _log_status(step)

        if ctrl.state == GripperState.TORQUE_HOLD and contact_step is None:
            contact_step = step
            print(
                f"\n>>> Contact confirmed — switched to torque hold at step {step} "
                f"(q={ctrl.joint_position_rad():.4f} rad)\n",
                flush=True,
            )

        step += 1

        if not use_viewer and contact_step is not None and hold_steps > 0:
            if step - contact_step >= hold_steps:
                return False

        if not use_viewer and step >= max_steps:
            return False

        if real_time:
            elapsed_step = time.monotonic() - step_start
            time.sleep(max(0.0, dt - elapsed_step))

        return True

    if use_viewer:
        if mj_viewer is None:
            raise SystemExit(
                "MuJoCo viewer is not installed in this environment.\n"
                "Run headless instead:\n"
                f"  python3 {SCRIPT_DIR / 'hybrid_gripper_position_to_torque.py'} --real-time"
            )
        print("Viewer open — close the window to exit.", flush=True)
        try:
            viewer_ctx = mj_viewer.launch_passive(model, data)
        except Exception as exc:  # noqa: BLE001 — GLFW / display errors vary by platform
            raise SystemExit(
                "Could not open MuJoCo viewer.\n"
                "Ensure DISPLAY is set (echo $DISPLAY) and you are not on a headless SSH session.\n"
                "Headless alternative:\n"
                f"  python3 {SCRIPT_DIR / 'hybrid_gripper_position_to_torque.py'} --real-time\n"
                f"Details: {exc}"
            ) from exc
        with viewer_ctx as viewer:
            while viewer.is_running():
                if max_steps > 0 and step >= max_steps:
                    break
                if not _sim_step():
                    break
                viewer.sync()
    else:
        print(
            f"Headless sim (real_time={real_time}, hold_sec={hold_sec}, max_steps={max_steps}).",
            flush=True,
        )
        while _sim_step():
            pass

    elapsed = time.monotonic() - t0
    print(
        f"\nFinished in {elapsed:.2f}s — final state={ctrl.state.name}, "
        f"q={ctrl.joint_position_rad():.4f} rad, "
        f"hold torque cmd={ctrl.cfg.hold_torque_nm:.3f} N·m "
        f"(~{ctrl.mapper.torque_nm_to_current_ma(ctrl.cfg.hold_torque_nm):.0f} mA)",
        flush=True,
    )
    if ctrl.state != GripperState.TORQUE_HOLD:
        print(
            "Note: contact was never detected. For the UR3 scene use --ur3-scene; "
            "for robot-only XML the gripper closes in free air (no object resistance).",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid gripper position → torque hold (MuJoCo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python3 "{SCRIPT_DIR / 'hybrid_gripper_position_to_torque.py'}"
  python3 "{SCRIPT_DIR / 'hybrid_gripper_position_to_torque.py'}" --viewer
  python3 "{SCRIPT_DIR / 'hybrid_gripper_position_to_torque.py'}" --ur3-scene --viewer
  python3 "{SCRIPT_DIR / 'hybrid_gripper_position_to_torque.py'}" --real-time --hold-sec 5
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help=f"MJCF path (must contain {ACTUATOR_NAME}). Default: built-in demo scene.",
    )
    parser.add_argument(
        "--ur3-scene",
        action="store_true",
        help=f"Load full pick scene: {DEFAULT_UR3_SCENE}",
    )
    parser.add_argument("--viewer", action="store_true", help="Open passive MuJoCo viewer")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Step limit (0 = no limit in viewer; headless uses 50000 when unset)",
    )
    parser.add_argument("--log-hz", type=float, default=20.0)
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Pace simulation to wall clock (on by default with --viewer)",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=5.0,
        help="Headless: keep running this many seconds after torque-hold contact (default 5)",
    )
    args = parser.parse_args()

    use_viewer = args.viewer
    real_time = args.real_time or use_viewer
    max_steps = args.max_steps
    if max_steps <= 0:
        max_steps = 0 if use_viewer else 50_000

    if args.ur3_scene and args.model:
        parser.error("Use either --ur3-scene or --model, not both.")

    try:
        model = resolve_model_path(args.model or None, use_ur3_scene=args.ur3_scene)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    run_sim(
        model,
        use_viewer=use_viewer,
        max_steps=max_steps,
        log_hz=args.log_hz,
        real_time=real_time,
        hold_sec=max(0.0, args.hold_sec),
    )


if __name__ == "__main__":
    main()
