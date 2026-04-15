"""
IMU sensor for acceleration, angular velocity, and orientation.

Wraps Isaac Sim's IMU sensor interface to provide structured readings
decoupled from the robot class.
"""

import dataclasses
from typing import Tuple

import numpy as np

from sensors.base import Sensor
from sensors.config import IMUConf

# Deferred Isaac Sim imports
try:
    from isaacsim.sensors.physics import _sensor
    _HAS_ISAAC = True
except ImportError:
    _HAS_ISAAC = False


@dataclasses.dataclass
class IMUReading:
    """Structured IMU sensor reading."""

    linear_acceleration: np.ndarray  # [ax, ay, az] m/s^2
    angular_velocity: np.ndarray     # [gx, gy, gz] rad/s
    orientation: np.ndarray          # [roll, pitch, yaw] radians

    @property
    def accel(self) -> np.ndarray:
        return self.linear_acceleration

    @property
    def gyro(self) -> np.ndarray:
        return self.angular_velocity

    @property
    def rpy(self) -> Tuple[float, float, float]:
        return (float(self.orientation[0]), float(self.orientation[1]), float(self.orientation[2]))


def _quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).

    Args:
        q: Quaternion as [x, y, z, w].

    Returns:
        Euler angles [roll, pitch, yaw] in radians.
    """
    x, y, z, w = q[0], q[1], q[2], q[3]

    # Roll (rotation about x)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation about y)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (rotation about z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


class IMUSensor(Sensor):
    """
    IMU sensor providing linear acceleration, angular velocity, and orientation.

    Wraps the Isaac Sim physics-based IMU interface.
    """

    def __init__(self, name: str, config: IMUConf) -> None:
        """
        Args:
            name: Unique sensor name.
            config: IMU configuration with sensor path.
        """
        super().__init__(name, config.sensor_path)
        self._config = config
        self._interface = None

    def initialize(self) -> None:
        """Acquire the Isaac Sim IMU sensor interface."""
        if not _HAS_ISAAC:
            raise RuntimeError("Isaac Sim not available - cannot initialize IMU sensor")

        if not self._prim_path:
            raise ValueError(
                "IMU sensor_path is not defined. Check your YAML configuration."
            )

        self._interface = _sensor.acquire_imu_sensor_interface()
        self._initialized = True

    def get_reading(self) -> IMUReading:
        """
        Read the latest IMU data.

        Returns:
            IMUReading with acceleration, angular velocity, and orientation.

        Raises:
            RuntimeError: If the sensor is not initialized.
        """
        if not self._initialized or self._interface is None:
            raise RuntimeError(f"IMU sensor '{self._name}' not initialized")

        reading = self._interface.get_sensor_reading(
            self._prim_path,
            use_latest_data=self._config.use_latest_data,
            read_gravity=self._config.read_gravity,
        )

        linear_acceleration = np.array([
            reading.lin_acc_x, reading.lin_acc_y, reading.lin_acc_z
        ])
        angular_velocity = np.array([
            reading.ang_vel_x, reading.ang_vel_y, reading.ang_vel_z
        ])

        # Convert quaternion orientation to Euler (roll, pitch, yaw)
        # Isaac Sim returns orientation as (x, y, z, w)
        q = np.array(reading.orientation)
        euler = _quaternion_to_euler(q)
        # Match OmniLRS convention: negate roll and pitch
        orientation = np.array([-euler[0], -euler[1], euler[2]])

        return IMUReading(
            linear_acceleration=linear_acceleration,
            angular_velocity=angular_velocity,
            orientation=orientation,
        )
