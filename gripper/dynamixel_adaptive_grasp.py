#!/usr/bin/env python3
"""
XM540-W270-R adaptive grasp (current-based position + contact detection).

Used standalone or from action_sequencer ``gripper_hybrid_close`` on real robot.

  python3 ~/ur3_control/gripper/dynamixel_adaptive_grasp.py
  python3 ~/ur3_control/gripper/dynamixel_adaptive_grasp.py --no-prompt
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dynamixel_sdk import PacketHandler, PortHandler

# XM540-W270-R control table (Protocol 2.0)
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1
MODE_POSITION = 3
MODE_CURRENT_BASED_POSITION = 5

DEFAULT_DEVICE = (
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT891L5D-if00-port0"
)

# XM540 single-turn position control: encoder 0..4095 (4096 ticks per revolution).
SINGLE_TURN_RESOLUTION = 4096
WRAP_OPEN_GOAL = 0
WRAP_BOUNDARY_TICKS = 4090


@dataclass
class GraspConfig:
    device: str = DEFAULT_DEVICE
    baudrate: int = 1_000_000
    dxl_id: int = 1
    protocol_version: float = 2.0
    goal_current_raw: int = 150
    profile_acceleration: int = 10
    profile_velocity: int = 30
    open_profile_acceleration: int = 10
    open_profile_velocity: int = 30
    open_position: int = 0
    step_pulse: int = 40
    max_total_move: int = 910
    contact_threshold: int = 80
    contact_count_required: int = 3
    sleep_time: float = 0.15
    hold_time: float = 5.0
    interactive_prompt: bool = False
    torque_off_after_hold: bool = True
    require_contact: bool = False
    move_timeout_sec: float = 10.0
    position_tolerance: int = 20
    torque_off_after_move: bool = False

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "GraspConfig":
        def _s(key: str, default: str) -> str:
            v = raw.get(key)
            return str(v).strip() if v is not None and str(v).strip() else default

        def _i(key: str, default: int) -> int:
            if key not in raw:
                return default
            return int(raw[key])

        def _f(key: str, default: float) -> float:
            if key not in raw:
                return default
            return float(raw[key])

        def _b(key: str, default: bool) -> bool:
            if key not in raw:
                return default
            return bool(raw[key])

        return cls(
            device=_s("dynamixel_device", DEFAULT_DEVICE),
            baudrate=_i("dynamixel_baudrate", 1_000_000),
            dxl_id=_i("dynamixel_id", 1),
            protocol_version=_f("dynamixel_protocol_version", 2.0),
            goal_current_raw=_i("dynamixel_goal_current_raw", 150),
            profile_acceleration=_i("dynamixel_profile_acceleration", 10),
            profile_velocity=_i("dynamixel_profile_velocity", 30),
            open_profile_acceleration=_i("dynamixel_open_profile_acceleration", 5),
            open_profile_velocity=_i("dynamixel_open_profile_velocity", 20),
            open_position=_i("dynamixel_open_position", 0),
            step_pulse=_i("dynamixel_step_pulse", 20),
            max_total_move=_i("dynamixel_max_total_move", 1365),
            contact_threshold=_i("dynamixel_contact_threshold", 80),
            contact_count_required=_i("dynamixel_contact_count_required", 3),
            sleep_time=_f("dynamixel_sleep_time", 0.05),
            hold_time=_f("dynamixel_hold_time", 5.0),
            interactive_prompt=_b("dynamixel_interactive_prompt", False),
            torque_off_after_hold=_b("dynamixel_torque_off_after_hold", True),
            require_contact=_b("dynamixel_require_contact", False),
            move_timeout_sec=_f("dynamixel_move_timeout_sec", 10.0),
            position_tolerance=_i("dynamixel_position_tolerance", 20),
            torque_off_after_move=_b("dynamixel_torque_off_after_move", False),
        )


def _check_result(packet_handler: PacketHandler, comm_result: int, error: int, context: str) -> None:
    if comm_result != 0:
        raise RuntimeError(
            f"[{context}] communication failed: {packet_handler.getTxRxResult(comm_result)}"
        )
    if error != 0:
        raise RuntimeError(
            f"[{context}] device error: {packet_handler.getRxPacketError(error)}"
        )


def _to_signed_16(val: int) -> int:
    if val >= 32768:
        val -= 65536
    return val


def _to_signed_32(val: int) -> int:
    if val >= 2147483648:
        val -= 4294967296
    return val


def _position_unsigned(pos: int) -> int:
    """Map signed/unsigned encoder reading to single-turn [0, 4095]."""
    tick = int(pos) % SINGLE_TURN_RESOLUTION
    return tick + SINGLE_TURN_RESOLUTION if tick < 0 else tick


def _position_error_ticks(goal: int, present: int) -> int:
    """Shortest-path distance on the single-turn encoder circle."""
    goal_u = _position_unsigned(goal)
    present_u = _position_unsigned(present)
    direct = abs(goal_u - present_u)
    return min(direct, SINGLE_TURN_RESOLUTION - direct)


def _resolve_goal_avoiding_wrap(
    goal: int,
    current: int,
    *,
    log_print=print,
) -> tuple[int, bool]:
    """
    Avoid commanding goal=0 from the 4095/0 wrap boundary.

    Returns (effective_goal, skip_move). When skip_move is True the motor is
    already at the physical open position and no goal write should be issued.
    """
    goal_i = int(goal)
    current_i = int(current)
    if goal_i == WRAP_OPEN_GOAL and _position_unsigned(current_i) >= WRAP_BOUNDARY_TICKS:
        log_print(
            f"dynamixel: skip move to {WRAP_OPEN_GOAL} — already at wrap/open "
            f"(present={current_i}, unsigned={_position_unsigned(current_i)})"
        )
        return current_i, True
    return goal_i, False


def _open_dynamixel_port(cfg: GraspConfig, *, log_print=print):
    """Open serial port and return (port_handler, packet_handler). Caller must closePort()."""
    port_handler = PortHandler(cfg.device)
    packet_handler = PacketHandler(cfg.protocol_version)
    if not port_handler.openPort():
        raise RuntimeError(f"failed to open port: {cfg.device}")
    log_print(f"dynamixel: port open OK ({cfg.device})")
    if not port_handler.setBaudRate(cfg.baudrate):
        raise RuntimeError(f"failed to set baudrate: {cfg.baudrate}")
    log_print(f"dynamixel: baudrate {cfg.baudrate}")
    return port_handler, packet_handler


def run_move_to_position(
    config: GraspConfig,
    goal_position: int,
    *,
    profile_velocity: Optional[int] = None,
    profile_acceleration: Optional[int] = None,
    log_print=print,
    leave_torque_on: Optional[bool] = None,
) -> bool:
    """Move XM540 to an absolute encoder goal (position control mode)."""
    cfg = config
    goal = int(goal_position)
    vel = int(cfg.profile_velocity if profile_velocity is None else profile_velocity)
    acc = int(cfg.profile_acceleration if profile_acceleration is None else profile_acceleration)
    keep_torque = cfg.torque_off_after_move is False if leave_torque_on is None else bool(leave_torque_on)
    port_handler, packet_handler = _open_dynamixel_port(cfg, log_print=log_print)

    def read_position() -> int:
        pos, comm_result, error = packet_handler.read4ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_PRESENT_POSITION
        )
        _check_result(packet_handler, comm_result, error, "read position")
        return _to_signed_32(pos)

    def write_goal_position(pos: int, *, current: int) -> None:
        effective, skip = _resolve_goal_avoiding_wrap(pos, current, log_print=log_print)
        if skip:
            return
        comm_result, error = packet_handler.write4ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_GOAL_POSITION, int(effective)
        )
        _check_result(packet_handler, comm_result, error, "write goal position")

    try:
        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )
        _check_result(packet_handler, comm_result, error, "torque off")

        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_OPERATING_MODE, MODE_POSITION
        )
        _check_result(packet_handler, comm_result, error, "operating mode")

        comm_result, error = packet_handler.write4ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_PROFILE_ACCELERATION, acc
        )
        _check_result(packet_handler, comm_result, error, "profile acceleration")

        comm_result, error = packet_handler.write4ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_PROFILE_VELOCITY, vel
        )
        _check_result(packet_handler, comm_result, error, "profile velocity")

        start_position = read_position()
        goal, skip_move = _resolve_goal_avoiding_wrap(goal, start_position, log_print=log_print)
        tol = int(cfg.position_tolerance)
        log_print(
            f"dynamixel: move_to_position start={start_position} goal={goal} "
            f"profile_vel={vel} profile_acc={acc} "
            f"tol={tol} timeout={cfg.move_timeout_sec:.1f}s"
        )

        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
        )
        _check_result(packet_handler, comm_result, error, "torque on")

        if skip_move or _position_error_ticks(goal, start_position) <= tol:
            if not skip_move:
                log_print(
                    f"dynamixel: already at goal present={start_position} "
                    f"(target={goal}, err={_position_error_ticks(goal, start_position)} ticks)"
                )
        else:
            write_goal_position(goal, current=start_position)
            deadline = time.monotonic() + max(0.5, float(cfg.move_timeout_sec))
            last_pos = start_position
            while time.monotonic() < deadline:
                present = read_position()
                err = _position_error_ticks(goal, present)
                if err <= tol:
                    log_print(f"dynamixel: reached goal position {present} (target={goal})")
                    break
                if present == last_pos and err <= tol * 3:
                    log_print(
                        f"dynamixel: stopped near goal present={present} target={goal} "
                        f"(within {tol * 3})"
                    )
                    break
                last_pos = present
                time.sleep(cfg.sleep_time)
            else:
                present = read_position()
                raise RuntimeError(
                    f"move_to_position timed out: present={present} goal={goal} "
                    f"err={_position_error_ticks(goal, present)} tolerance={tol}"
                )

        if not keep_torque:
            comm_result, error = packet_handler.write1ByteTxRx(
                port_handler, cfg.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
            )
            _check_result(packet_handler, comm_result, error, "torque off after move")
            log_print("dynamixel: torque off after move")

        return True
    finally:
        port_handler.closePort()


def run_adaptive_grasp(
    config: Optional[GraspConfig] = None,
    *,
    log_print=print,
) -> bool:
    """Run adaptive close. Returns True on success (contact detected if require_contact)."""
    cfg = config or GraspConfig()
    port_handler, packet_handler = _open_dynamixel_port(cfg, log_print=log_print)
    contact_detected = False

    def read_position() -> int:
        pos, comm_result, error = packet_handler.read4ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_PRESENT_POSITION
        )
        _check_result(packet_handler, comm_result, error, "read position")
        return _to_signed_32(pos)

    def read_current() -> int:
        cur, comm_result, error = packet_handler.read2ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_PRESENT_CURRENT
        )
        _check_result(packet_handler, comm_result, error, "read current")
        return _to_signed_16(cur)

    def write_goal_position(goal_position: int) -> None:
        comm_result, error = packet_handler.write4ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_GOAL_POSITION, goal_position
        )
        _check_result(packet_handler, comm_result, error, "write goal position")

    try:
        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )
        _check_result(packet_handler, comm_result, error, "torque off")

        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_OPERATING_MODE, MODE_CURRENT_BASED_POSITION
        )
        _check_result(packet_handler, comm_result, error, "operating mode")

        comm_result, error = packet_handler.write2ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_GOAL_CURRENT, cfg.goal_current_raw
        )
        _check_result(packet_handler, comm_result, error, "goal current")

        comm_result, error = packet_handler.write4ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_PROFILE_ACCELERATION, cfg.profile_acceleration
        )
        _check_result(packet_handler, comm_result, error, "profile acceleration")

        comm_result, error = packet_handler.write4ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_PROFILE_VELOCITY, cfg.profile_velocity
        )
        _check_result(packet_handler, comm_result, error, "profile velocity")

        start_position = read_position()
        log_print(
            f"dynamixel: start_pos={start_position} step={cfg.step_pulse} "
            f"max_move={cfg.max_total_move} contact_thr={cfg.contact_threshold}"
        )

        if cfg.interactive_prompt:
            input("Press Enter to start adaptive grasp…")

        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, cfg.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
        )
        _check_result(packet_handler, comm_result, error, "torque on")

        moved = 0
        contact_count = 0
        while moved < cfg.max_total_move:
            current_position = read_position()
            write_goal_position(current_position + cfg.step_pulse)
            time.sleep(cfg.sleep_time)
            current_raw = read_current()
            moved += cfg.step_pulse

            if abs(current_raw) > cfg.contact_threshold:
                contact_count += 1
            else:
                contact_count = 0

            log_print(
                f"dynamixel: moved={moved} pos={current_position} "
                f"i={current_raw} contact_count={contact_count}"
            )

            if contact_count >= cfg.contact_count_required:
                contact_detected = True
                hold_position = read_position()
                write_goal_position(hold_position)
                log_print("dynamixel: contact detected — holding position")
                break
        else:
            log_print("dynamixel: max travel reached without contact")

        if cfg.require_contact and not contact_detected:
            raise RuntimeError("adaptive grasp finished without contact detection")

        log_print(f"dynamixel: holding {cfg.hold_time:.1f}s")
        time.sleep(cfg.hold_time)

        if cfg.torque_off_after_hold:
            comm_result, error = packet_handler.write1ByteTxRx(
                port_handler, cfg.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
            )
            _check_result(packet_handler, comm_result, error, "torque off after hold")
            log_print("dynamixel: torque off")

        return True

    finally:
        port_handler.closePort()


def main() -> int:
    parser = argparse.ArgumentParser(description="XM540 adaptive grasp via Dynamixel SDK")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="serial device (default: U2D2 by-id path)",
    )
    parser.add_argument("--no-prompt", action="store_true", help="skip Enter prompt")
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="leave torque enabled after hold (for lift after grasp)",
    )
    parser.add_argument(
        "--require-contact",
        action="store_true",
        help="fail if max travel reached without contact",
    )
    args = parser.parse_args()
    cfg = GraspConfig(
        device=args.device,
        interactive_prompt=not args.no_prompt,
        torque_off_after_hold=not args.keep_torque,
        require_contact=args.require_contact,
    )
    try:
        run_adaptive_grasp(cfg)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
