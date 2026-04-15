"""UDP bridge for S2R_ICD protocol.

Sends rover simulation data (position, rotation, wheel angles) as
S2R_ICD-formatted packets via UDP Unicast.

Packet structure (59 bytes total for ISSA-ROVER-DATA):
    STX (1B) | LEN (2B) | SEQ (1B) | MSG_ID (3B) |
    TIMESTAMP (8B) | PAYLOAD (40B) | CRC32 (4B)

Reference: S2R_ICD.xlsx — ES-ICD-ISSA-ES-0010 (ISSA-ROVER-DATA, 0x030101)
"""

import logging
import math
import socket
import struct
import threading
import time
import zlib
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.spatial.transform import Rotation as R
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Protocol constants ──────────────────────────────────────────────────────

STX = 0xFE
MSG_ID_ISSA_ROVER_DATA = b'\x03\x01\x01'  # Src=ISSA(3), Des=ES(1), Msg#=1

# 2015-01-01 00:00:00 UTC in seconds since Unix epoch
_EPOCH_2015_UTC = 1420070400


# ── Quaternion → RPY conversion ─────────────────────────────────────────────

def isaac_quat_to_rpy_deg(orientation) -> Tuple[float, float, float]:
    """Convert Isaac quaternion (w,x,y,z) to Roll, Pitch, Yaw in degrees [0, 360).

    Args:
        orientation: Quaternion as (w, x, y, z) — Isaac Sim format.

    Returns:
        (roll, pitch, yaw) in degrees, each in [0, 360).
    """
    if not _HAS_SCIPY:
        # Fallback: simple Euler extraction (less robust at gimbal lock)
        w, x, y, z = float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])
        # Roll (x-axis)
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr, cosr))
        # Pitch (y-axis)
        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.degrees(math.asin(sinp))
        # Yaw (z-axis)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny, cosy))
        return roll % 360, pitch % 360, yaw % 360

    w, x, y, z = float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])
    rot = R.from_quat([x, y, z, w])  # scipy uses (x,y,z,w)
    rpy = rot.as_euler('xyz', degrees=True)
    return float(rpy[0]) % 360, float(rpy[1]) % 360, float(rpy[2]) % 360


# ── Packet builder ──────────────────────────────────────────────────────────

class S2RPacketBuilder:
    """Builds S2R_ICD protocol packets.

    Thread-safe: the sequence counter uses a lock.
    """

    def __init__(self) -> None:
        self._seq = 0
        self._seq_lock = threading.Lock()

    def _next_seq(self) -> int:
        with self._seq_lock:
            seq = self._seq
            self._seq = (self._seq + 1) % 256
            return seq

    @staticmethod
    def _timestamp_bytes() -> bytes:
        """Current time as ms since 2015-01-01 00:00:00 UTC, 8 bytes LE."""
        now_s = time.time()
        ms_since_2015 = int((now_s - _EPOCH_2015_UTC) * 1000)
        return struct.pack('<Q', ms_since_2015)

    def build_rover_data(
        self,
        pos_x: float, pos_y: float, pos_z: float,
        roll: float, pitch: float, yaw: float,
        wheel_fl: float, wheel_fr: float,
        wheel_rl: float, wheel_rr: float,
    ) -> bytes:
        """Build ISSA-ROVER-DATA (0x030101) packet.

        Args:
            pos_x, pos_y, pos_z: Rover position (meters).
            roll, pitch, yaw: Rover rotation (degrees, 0-360).
            wheel_fl/fr/rl/rr: Wheel joint angles (radians).

        Returns:
            Complete S2R_ICD packet (59 bytes).
        """
        # Payload: 10 x float32 little-endian = 40 bytes
        payload = struct.pack('<10f',
            pos_x, pos_y, pos_z,
            roll, pitch, yaw,
            wheel_fl, wheel_fr, wheel_rl, wheel_rr,
        )

        # Header
        stx = struct.pack('B', STX)
        length = struct.pack('<H', len(payload))      # 2B LE
        seq = struct.pack('B', self._next_seq())       # 1B
        msg_id = MSG_ID_ISSA_ROVER_DATA                # 3B
        timestamp = self._timestamp_bytes()            # 8B

        packet_body = stx + length + seq + msg_id + timestamp + payload

        # CRC-32
        crc = zlib.crc32(packet_body) & 0xFFFFFFFF
        checksum = struct.pack('<I', crc)

        return packet_body + checksum


# ── UDP sender ──────────────────────────────────────────────────────────────

class UDPSender:
    """Manages a UDP socket for unicast transmission."""

    def __init__(self, target_ip: str, target_port: int) -> None:
        self._addr = (target_ip, target_port)
        self._sock: Optional[socket.socket] = None

    def open(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info("UDP socket opened → %s:%d", *self._addr)

    def send(self, data: bytes) -> None:
        if self._sock is None:
            self.open()
        try:
            self._sock.sendto(data, self._addr)
        except OSError as e:
            logger.error("UDP send failed: %s", e)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
            logger.info("UDP socket closed")


# ── High-level bridge ───────────────────────────────────────────────────────

class UDPBridge:
    """Sends rover data from the simulation as S2R_ICD UDP packets.

    Called each frame from the main simulation loop.

    Usage:
        bridge = UDPBridge("172.18.0.8", 7777, robot_manager)
        bridge.setup()
        # In main loop:
        bridge.send_rover_data()
        # On shutdown:
        bridge.shutdown()
    """

    def __init__(
        self,
        target_ip: str,
        target_port: int,
        robot_manager=None,
    ) -> None:
        self._target_ip = target_ip
        self._target_port = target_port
        self._rm = robot_manager
        self._packet_builder = S2RPacketBuilder()
        self._sender: Optional[UDPSender] = None

    def setup(self) -> None:
        """Open UDP socket."""
        self._sender = UDPSender(self._target_ip, self._target_port)
        self._sender.open()
        logger.info(
            "UDPBridge ready → %s:%d", self._target_ip, self._target_port
        )

    def send_rover_data(self) -> None:
        """Extract rover data from robot_manager and send as ICD packet.

        Extracts the first robot's pose and wheel angles, builds an
        ISSA-ROVER-DATA packet, and sends it via UDP unicast.
        """
        if self._sender is None or self._rm is None:
            return

        # Get first robot's data
        rigid_groups = getattr(self._rm, "rigid_groups", {})
        robots = getattr(self._rm, "robots", {})

        if not rigid_groups:
            return

        # Use the first robot
        robot_name = next(iter(rigid_groups))
        rrg = rigid_groups[robot_name]

        # ── Position & Orientation ──
        try:
            position, orientation = rrg.get_pose_of_base_link()
            pos_x = float(position[0])
            pos_y = float(position[1])
            pos_z = float(position[2])
            roll, pitch, yaw = isaac_quat_to_rpy_deg(orientation)
        except Exception as e:
            logger.debug("Failed to get robot pose: %s", e)
            return

        # ── Wheel joint angles ──
        wheel_fl = wheel_fr = wheel_rl = wheel_rr = 0.0
        robot = robots.get(robot_name)
        if robot is not None:
            try:
                angles = robot.get_wheels_joint_angles()
                # Expected order: [left_wheels..., right_wheels...]
                # For 4-wheel: [FL, RL, FR, RR] (left=[FL,RL], right=[FR,RR])
                if len(angles) >= 4:
                    wheel_fl = float(angles[0])   # left[0] = front left
                    wheel_rl = float(angles[1])   # left[1] = rear left
                    wheel_fr = float(angles[2])   # right[0] = front right
                    wheel_rr = float(angles[3])   # right[1] = rear right
                elif len(angles) == 2:
                    # 2-wheel differential: left, right
                    wheel_fl = wheel_rl = float(angles[0])
                    wheel_fr = wheel_rr = float(angles[1])
            except Exception as e:
                logger.debug("Failed to get wheel angles: %s", e)

        # ── Build & send ──
        packet = self._packet_builder.build_rover_data(
            pos_x, pos_y, pos_z,
            roll, pitch, yaw,
            wheel_fl, wheel_fr, wheel_rl, wheel_rr,
        )
        self._sender.send(packet)

    def shutdown(self) -> None:
        """Close UDP socket."""
        if self._sender is not None:
            self._sender.close()
            self._sender = None
        logger.info("UDPBridge shutdown complete")
