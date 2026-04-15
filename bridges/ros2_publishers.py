"""ROS2 publishers for sensor data, subsystem telemetry, and diagnostics.

Manages per-robot publisher lifecycle with lazy creation:
publishers are created on first use for each robot, avoiding
topic creation for robots that may never be spawned.

Ported from OmniLRS ros_manager.py publisher pattern.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64
    from geometry_msgs.msg import Vector3, TransformStamped
    from sensor_msgs.msg import BatteryState, Imu, PointCloud2
    from tf2_msgs.msg import TFMessage
    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False


# Sensor QoS: best-effort for high-frequency data
_SENSOR_QOS = (
    QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    if _HAS_ROS2 else None
)
# Reliable QoS for TF and state
_RELIABLE_QOS = QoSProfile(depth=10) if _HAS_ROS2 else None


class SensorPublishers:
    """Manages per-robot sensor data publishers.

    Creates publishers lazily on first publish call for each robot.
    All publishers are attached to the provided ROS2 node.

    Topics per robot:
        /{robot}/sun_vector          (Vector3)      sun direction
        /{robot}/battery_state       (BatteryState)  battery telemetry
        /{robot}/sensor_temperature  (Float64)       temperature (Kelvin)
        /{robot}/imu                 (Imu)           IMU readings
    """

    def __init__(self, node) -> None:
        self._node = node
        self._sun_pubs: Dict[str, Any] = {}
        self._battery_pubs: Dict[str, Any] = {}
        self._temp_pubs: Dict[str, Any] = {}
        self._imu_pubs: Dict[str, Any] = {}

    def _get_or_create(self, registry: dict, robot_name: str, msg_type, topic_suffix: str, qos) -> Any:
        """Get existing publisher or create a new one."""
        if robot_name not in registry:
            topic = f"/{robot_name}/{topic_suffix}"
            registry[robot_name] = self._node.create_publisher(msg_type, topic, qos)
            logger.debug("Created publisher: %s", topic)
        return registry[robot_name]

    def publish_sun_vector(self, robot_name: str, vector: np.ndarray) -> None:
        """Publish sun direction vector relative to robot."""
        if not _HAS_ROS2:
            return
        pub = self._get_or_create(self._sun_pubs, robot_name, Vector3, "sun_vector", _RELIABLE_QOS)
        msg = Vector3(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))
        pub.publish(msg)

    def publish_battery_state(
        self,
        robot_name: str,
        voltage: float,
        percentage: float,
        current: float,
    ) -> None:
        """Publish battery telemetry.

        Args:
            robot_name: Robot identifier.
            voltage: Battery voltage (V).
            percentage: State of charge (0-100%).
            current: Net current (A), positive = charging.
        """
        if not _HAS_ROS2:
            return
        pub = self._get_or_create(self._battery_pubs, robot_name, BatteryState, "battery_state", _RELIABLE_QOS)
        msg = BatteryState()
        msg.voltage = voltage
        msg.percentage = percentage / 100.0  # ROS BatteryState uses 0..1
        msg.current = current
        msg.present = True
        pub.publish(msg)

    def publish_temperature(self, robot_name: str, temperature_celsius: float) -> None:
        """Publish sensor temperature in Kelvin.

        Args:
            robot_name: Robot identifier.
            temperature_celsius: Temperature in Celsius (converted to K for publishing).
        """
        if not _HAS_ROS2:
            return
        pub = self._get_or_create(self._temp_pubs, robot_name, Float64, "sensor_temperature", _RELIABLE_QOS)
        msg = Float64(data=temperature_celsius + 273.15)
        pub.publish(msg)

    def publish_imu(
        self,
        robot_name: str,
        linear_acceleration: np.ndarray,
        angular_velocity: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """Publish IMU readings.

        Args:
            robot_name: Robot identifier.
            linear_acceleration: (ax, ay, az) in m/s^2.
            angular_velocity: (gx, gy, gz) in rad/s.
            orientation: Optional quaternion (x, y, z, w) in ROS format.
        """
        if not _HAS_ROS2:
            return
        pub = self._get_or_create(self._imu_pubs, robot_name, Imu, "imu", _SENSOR_QOS)
        msg = Imu()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = f"{robot_name}/imu_link"

        msg.linear_acceleration.x = float(linear_acceleration[0])
        msg.linear_acceleration.y = float(linear_acceleration[1])
        msg.linear_acceleration.z = float(linear_acceleration[2])

        msg.angular_velocity.x = float(angular_velocity[0])
        msg.angular_velocity.y = float(angular_velocity[1])
        msg.angular_velocity.z = float(angular_velocity[2])

        if orientation is not None:
            msg.orientation.x = float(orientation[0])
            msg.orientation.y = float(orientation[1])
            msg.orientation.z = float(orientation[2])
            msg.orientation.w = float(orientation[3])
            msg.orientation_covariance[0] = -1.0  # Mark as valid
        else:
            msg.orientation_covariance[0] = -1.0  # Unknown

        pub.publish(msg)


class TFPublisher:
    """Publishes ground-truth transforms on /tf_gt.

    Handles single-robot (map -> base_link) and
    multi-robot (map -> {name}/base_link) frame conventions.
    """

    def __init__(self, node) -> None:
        self._node = node
        self._pub = None
        if _HAS_ROS2:
            self._pub = node.create_publisher(TFMessage, "/tf_gt", _RELIABLE_QOS)

    def publish(self, robot_transforms: Dict[str, tuple]) -> None:
        """Publish ground-truth TF for all robots.

        Args:
            robot_transforms: Dict of robot_name -> (position[3], orientation[4])
                where orientation is Isaac format (w, x, y, z).
        """
        if self._pub is None or not robot_transforms:
            return

        multi = len(robot_transforms) > 1
        now = self._node.get_clock().now().to_msg()
        transforms = []

        for robot_name, (position, orientation) in robot_transforms.items():
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "map"
            t.child_frame_id = f"{robot_name}/base_link" if multi else "base_link"

            t.transform.translation.x = float(position[0])
            t.transform.translation.y = float(position[1])
            t.transform.translation.z = float(position[2])

            # Isaac (w,x,y,z) -> ROS TransformStamped
            t.transform.rotation.w = float(orientation[0])
            t.transform.rotation.x = float(orientation[1])
            t.transform.rotation.y = float(orientation[2])
            t.transform.rotation.z = float(orientation[3])

            transforms.append(t)

        self._pub.publish(TFMessage(transforms=transforms))
