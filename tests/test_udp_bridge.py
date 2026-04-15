"""Tests for UDP bridge: packet building, quaternion conversion, and loopback send/receive."""

import math
import socket
import struct
import threading
import time
import zlib

import numpy as np
import pytest

from bridges.udp_bridge import (
    S2RPacketBuilder,
    STX,
    MSG_ID_ISSA_ROVER_DATA,
    UDPSender,
    UDPBridge,
    isaac_quat_to_rpy_deg,
    _EPOCH_2015_UTC,
)
from tools.udp_receiver import decode_packet


# ── Quaternion → RPY conversion ───────────────────────────────────────────────

class TestQuaternionConversion:
    """Verify Isaac (w,x,y,z) → Roll/Pitch/Yaw (degrees, 0-360)."""

    def test_identity_quaternion(self):
        """Identity quaternion (w=1) should give (0, 0, 0)."""
        roll, pitch, yaw = isaac_quat_to_rpy_deg([1.0, 0.0, 0.0, 0.0])
        assert abs(roll) < 1e-6 or abs(roll - 360) < 1e-6
        assert abs(pitch) < 1e-6 or abs(pitch - 360) < 1e-6
        assert abs(yaw) < 1e-6 or abs(yaw - 360) < 1e-6

    def test_yaw_90(self):
        """90° yaw only: quat = (cos(45°), 0, 0, sin(45°))."""
        angle = math.radians(90)
        w = math.cos(angle / 2)
        z = math.sin(angle / 2)
        roll, pitch, yaw = isaac_quat_to_rpy_deg([w, 0.0, 0.0, z])
        assert abs(roll) < 1.0 or abs(roll - 360) < 1.0
        assert abs(pitch) < 1.0 or abs(pitch - 360) < 1.0
        assert abs(yaw - 90.0) < 1.0

    def test_roll_45(self):
        """45° roll only."""
        angle = math.radians(45)
        w = math.cos(angle / 2)
        x = math.sin(angle / 2)
        roll, pitch, yaw = isaac_quat_to_rpy_deg([w, x, 0.0, 0.0])
        assert abs(roll - 45.0) < 1.0

    def test_output_range_0_360(self):
        """All outputs should be in [0, 360)."""
        # Negative rotation: -30° yaw → should map to 330°
        angle = math.radians(-30)
        w = math.cos(angle / 2)
        z = math.sin(angle / 2)
        roll, pitch, yaw = isaac_quat_to_rpy_deg([w, 0.0, 0.0, z])
        assert 0 <= roll < 360
        assert 0 <= pitch < 360
        assert 0 <= yaw < 360
        assert abs(yaw - 330.0) < 1.0


# ── Packet building ──────────────────────────────────────────────────────────

class TestS2RPacketBuilder:
    """Verify packet structure, field encoding, and CRC integrity."""

    def setup_method(self):
        self.builder = S2RPacketBuilder()

    def test_packet_length(self):
        """ISSA-ROVER-DATA packet should be exactly 59 bytes."""
        pkt = self.builder.build_rover_data(
            1.0, 2.0, 3.0,
            10.0, 20.0, 30.0,
            0.1, 0.2, 0.3, 0.4,
        )
        assert len(pkt) == 59

    def test_stx_marker(self):
        """First byte must be 0xFE."""
        pkt = self.builder.build_rover_data(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert pkt[0] == STX

    def test_msg_id(self):
        """MSG_ID bytes at offset 4-6 must be 0x030101."""
        pkt = self.builder.build_rover_data(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert pkt[4:7] == MSG_ID_ISSA_ROVER_DATA

    def test_payload_length_field(self):
        """LEN field (bytes 1-2) should be 40 (payload size)."""
        pkt = self.builder.build_rover_data(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        length = struct.unpack_from('<H', pkt, 1)[0]
        assert length == 40

    def test_payload_values(self):
        """Payload float32 values should match inputs."""
        vals = (1.5, -2.5, 3.5, 10.0, 20.0, 30.0, 0.1, 0.2, 0.3, 0.4)
        pkt = self.builder.build_rover_data(*vals)
        # Payload starts at offset 15, 10 x float32
        decoded = struct.unpack_from('<10f', pkt, 15)
        for i, (expected, actual) in enumerate(zip(vals, decoded)):
            assert abs(expected - actual) < 1e-6, f"Field {i}: {expected} != {actual}"

    def test_crc32_valid(self):
        """CRC-32 at end of packet should validate against body."""
        pkt = self.builder.build_rover_data(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        body = pkt[:-4]
        stored_crc = struct.unpack_from('<I', pkt, len(pkt) - 4)[0]
        computed_crc = zlib.crc32(body) & 0xFFFFFFFF
        assert stored_crc == computed_crc

    def test_sequence_increments(self):
        """Sequence counter should increment with each packet."""
        pkt1 = self.builder.build_rover_data(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        pkt2 = self.builder.build_rover_data(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        seq1 = pkt1[3]
        seq2 = pkt2[3]
        assert seq2 == (seq1 + 1) % 256

    def test_sequence_wraps_at_256(self):
        """Sequence counter should wrap around at 256."""
        builder = S2RPacketBuilder()
        builder._seq = 255
        pkt1 = builder.build_rover_data(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        pkt2 = builder.build_rover_data(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert pkt1[3] == 255
        assert pkt2[3] == 0

    def test_timestamp_reasonable(self):
        """Timestamp should be recent (within last minute)."""
        pkt = self.builder.build_rover_data(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        ts_ms = struct.unpack_from('<Q', pkt, 7)[0]
        now_ms = int((time.time() - _EPOCH_2015_UTC) * 1000)
        # Should be within 5 seconds
        assert abs(now_ms - ts_ms) < 5000


# ── Receiver compatibility ────────────────────────────────────────────────────

class TestReceiverCompatibility:
    """Verify that packets built by S2RPacketBuilder decode correctly
    with the tools/udp_receiver.py decode_packet function."""

    def test_roundtrip_decode(self):
        """Build a packet and decode it — all fields should match."""
        builder = S2RPacketBuilder()
        vals = (10.5, -20.3, 5.0, 45.0, 90.0, 180.0, 0.1, -0.2, 0.3, -0.4)
        pkt = builder.build_rover_data(*vals)

        result = decode_packet(pkt)

        assert "error" not in result
        assert result["crc_ok"] is True
        assert result["msg_id"] == "0x030101"
        assert result["total_bytes"] == 59
        assert result["length"] == 40

        rover = result["rover_data"]
        assert abs(rover["pos_x"] - 10.5) < 1e-5
        assert abs(rover["pos_y"] - (-20.3)) < 1e-5
        assert abs(rover["pos_z"] - 5.0) < 1e-5
        assert abs(rover["roll_deg"] - 45.0) < 1e-5
        assert abs(rover["pitch_deg"] - 90.0) < 1e-5
        assert abs(rover["yaw_deg"] - 180.0) < 1e-5
        assert abs(rover["wheel_fl"] - 0.1) < 1e-5
        assert abs(rover["wheel_fr"] - (-0.2)) < 1e-5
        assert abs(rover["wheel_rl"] - 0.3) < 1e-5
        assert abs(rover["wheel_rr"] - (-0.4)) < 1e-5


# ── UDP loopback send/receive test ────────────────────────────────────────────

class TestUDPLoopback:
    """End-to-end: build packet, send via UDP, receive and decode on loopback."""

    def test_loopback_send_receive(self):
        """Send a packet via UDPSender to localhost, receive and verify."""
        # Bind a receiver on a random port
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", 0))
        recv_sock.settimeout(3.0)
        port = recv_sock.getsockname()[1]

        received = {}

        def receiver():
            try:
                data, _ = recv_sock.recvfrom(65536)
                received["data"] = data
            except socket.timeout:
                received["error"] = "timeout"
            finally:
                recv_sock.close()

        # Start receiver thread
        t = threading.Thread(target=receiver, daemon=True)
        t.start()

        # Build and send
        builder = S2RPacketBuilder()
        pkt = builder.build_rover_data(
            100.0, 200.0, 50.0,   # position
            15.0, 25.0, 350.0,    # RPY
            1.0, 2.0, 3.0, 4.0,  # wheels
        )

        sender = UDPSender("127.0.0.1", port)
        sender.send(pkt)
        sender.close()

        t.join(timeout=5.0)

        assert "data" in received, f"Did not receive packet: {received.get('error')}"
        assert received["data"] == pkt

        # Decode and verify
        result = decode_packet(received["data"])
        assert result["crc_ok"] is True
        rover = result["rover_data"]
        assert abs(rover["pos_x"] - 100.0) < 1e-5
        assert abs(rover["pos_y"] - 200.0) < 1e-5
        assert abs(rover["pos_z"] - 50.0) < 1e-5
        assert abs(rover["yaw_deg"] - 350.0) < 1e-5


# ── UDPBridge with mock robot_manager ─────────────────────────────────────────

class MockBasePrim:
    def __init__(self, position, orientation):
        self._pos = np.array(position, dtype=np.float64)
        self._ori = np.array(orientation, dtype=np.float64)

    def get_world_pose(self):
        return self._pos, self._ori


class MockRigidGroup:
    def __init__(self, position, orientation):
        self._base_prim = MockBasePrim(position, orientation)

    def get_pose_of_base_link(self):
        return self._base_prim.get_world_pose()


class MockRobot:
    def __init__(self, angles):
        self._angles = angles

    def get_wheels_joint_angles(self):
        return self._angles


class MockRobotManager:
    def __init__(self, position, orientation, wheel_angles):
        self.rigid_groups = {"rover1": MockRigidGroup(position, orientation)}
        self.robots = {"rover1": MockRobot(wheel_angles)}


class TestUDPBridgeIntegration:
    """Test UDPBridge.send_rover_data() with mocked robot_manager
    and real UDP loopback."""

    def test_send_rover_data_loopback(self):
        """Full pipeline: mock robot → UDPBridge → UDP → decode."""
        # Identity quaternion: (w,x,y,z) = (1,0,0,0) → RPY = (0,0,0)
        position = [5.0, 10.0, 1.5]
        orientation = [1.0, 0.0, 0.0, 0.0]  # Isaac format (w,x,y,z)
        wheel_angles = [0.1, 0.2, 0.3, 0.4]  # FL, RL, FR, RR

        rm = MockRobotManager(position, orientation, wheel_angles)

        # Bind receiver
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", 0))
        recv_sock.settimeout(3.0)
        port = recv_sock.getsockname()[1]

        received = {}

        def receiver():
            try:
                data, _ = recv_sock.recvfrom(65536)
                received["data"] = data
            except socket.timeout:
                received["error"] = "timeout"
            finally:
                recv_sock.close()

        t = threading.Thread(target=receiver, daemon=True)
        t.start()

        # Create bridge and send
        bridge = UDPBridge("127.0.0.1", port, robot_manager=rm)
        bridge.setup()
        bridge.send_rover_data()
        bridge.shutdown()

        t.join(timeout=5.0)

        assert "data" in received, f"No packet received: {received.get('error')}"

        result = decode_packet(received["data"])
        assert result["crc_ok"] is True

        rover = result["rover_data"]
        # Position
        assert abs(rover["pos_x"] - 5.0) < 1e-5
        assert abs(rover["pos_y"] - 10.0) < 1e-5
        assert abs(rover["pos_z"] - 1.5) < 1e-5
        # RPY from identity quaternion → ~0
        assert abs(rover["roll_deg"]) < 1.0 or abs(rover["roll_deg"] - 360) < 1.0
        assert abs(rover["pitch_deg"]) < 1.0 or abs(rover["pitch_deg"] - 360) < 1.0
        assert abs(rover["yaw_deg"]) < 1.0 or abs(rover["yaw_deg"] - 360) < 1.0
        # Wheels: FL=0.1, RL=0.2, FR=0.3, RR=0.4
        # UDPBridge maps: angles[0]→FL, angles[1]→RL, angles[2]→FR, angles[3]→RR
        # build_rover_data(wheel_fl, wheel_fr, wheel_rl, wheel_rr)
        assert abs(rover["wheel_fl"] - 0.1) < 1e-5
        assert abs(rover["wheel_fr"] - 0.3) < 1e-5
        assert abs(rover["wheel_rl"] - 0.2) < 1e-5
        assert abs(rover["wheel_rr"] - 0.4) < 1e-5

    def test_send_rover_data_with_yaw_rotation(self):
        """Verify a 90° yaw rotation is correctly transmitted."""
        angle = math.radians(90)
        w = math.cos(angle / 2)
        z = math.sin(angle / 2)
        position = [1.0, 2.0, 3.0]
        orientation = [w, 0.0, 0.0, z]  # 90° yaw, Isaac (w,x,y,z)
        wheel_angles = [0.0, 0.0, 0.0, 0.0]

        rm = MockRobotManager(position, orientation, wheel_angles)

        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", 0))
        recv_sock.settimeout(3.0)
        port = recv_sock.getsockname()[1]

        received = {}

        def receiver():
            try:
                data, _ = recv_sock.recvfrom(65536)
                received["data"] = data
            except socket.timeout:
                received["error"] = "timeout"
            finally:
                recv_sock.close()

        t = threading.Thread(target=receiver, daemon=True)
        t.start()

        bridge = UDPBridge("127.0.0.1", port, robot_manager=rm)
        bridge.setup()
        bridge.send_rover_data()
        bridge.shutdown()

        t.join(timeout=5.0)

        assert "data" in received
        result = decode_packet(received["data"])
        rover = result["rover_data"]

        assert abs(rover["pos_x"] - 1.0) < 1e-5
        assert abs(rover["pos_y"] - 2.0) < 1e-5
        assert abs(rover["pos_z"] - 3.0) < 1e-5
        assert abs(rover["yaw_deg"] - 90.0) < 1.5

    def test_no_crash_without_robot_manager(self):
        """UDPBridge should silently skip if robot_manager is None."""
        bridge = UDPBridge("127.0.0.1", 9999, robot_manager=None)
        bridge.setup()
        bridge.send_rover_data()  # Should not raise
        bridge.shutdown()

    def test_no_crash_empty_rigid_groups(self):
        """UDPBridge should silently skip if no robots."""
        rm = MockRobotManager([0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0])
        rm.rigid_groups = {}  # Empty
        bridge = UDPBridge("127.0.0.1", 9999, robot_manager=rm)
        bridge.setup()
        bridge.send_rover_data()  # Should not raise
        bridge.shutdown()
