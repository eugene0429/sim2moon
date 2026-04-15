"""
Unified sensor registry and manager.

Provides a single point of access for registering, initializing, and
querying all sensor instances across the simulation. Decouples sensor
management from robot code.
"""

from typing import Dict, List, Optional, Union

import numpy as np

from sensors.base import Sensor
from sensors.camera import Camera
from sensors.config import CameraConf, IMUConf, LiDARConf
from sensors.imu import IMUReading, IMUSensor
from sensors.lidar import LiDAR


class SensorManager:
    """
    Central registry for all sensors in the simulation.

    Sensors are registered by unique name and can be queried generically
    or by type. The manager handles bulk initialization and provides
    typed convenience methods.

    Interface contract:
        register_camera(name, config) -> Camera
        register_imu(name, config) -> IMUSensor
        register_lidar(name, config) -> LiDAR
        get_sensor(name) -> Sensor
        initialize_all() -> None
    """

    def __init__(self) -> None:
        self._sensors: Dict[str, Sensor] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register_camera(self, name: str, config: CameraConf) -> Camera:
        """
        Register and return a new camera sensor.

        Args:
            name: Unique sensor name.
            config: Camera configuration.

        Returns:
            The registered Camera instance.
        """
        if name in self._sensors:
            raise ValueError(f"Sensor '{name}' is already registered")
        camera = Camera(name, config)
        self._sensors[name] = camera
        return camera

    def register_imu(self, name: str, config: IMUConf) -> IMUSensor:
        """
        Register and return a new IMU sensor.

        Args:
            name: Unique sensor name.
            config: IMU configuration.

        Returns:
            The registered IMUSensor instance.
        """
        if name in self._sensors:
            raise ValueError(f"Sensor '{name}' is already registered")
        imu = IMUSensor(name, config)
        self._sensors[name] = imu
        return imu

    def register_lidar(self, name: str, config: LiDARConf) -> LiDAR:
        """
        Register and return a new LiDAR sensor.

        Args:
            name: Unique sensor name.
            config: LiDAR configuration.

        Returns:
            The registered LiDAR instance.
        """
        if name in self._sensors:
            raise ValueError(f"Sensor '{name}' is already registered")
        lidar = LiDAR(name, config)
        self._sensors[name] = lidar
        return lidar

    def register(self, sensor: Sensor) -> None:
        """
        Register a pre-constructed sensor.

        Args:
            sensor: Any Sensor subclass instance.
        """
        if sensor.name in self._sensors:
            raise ValueError(f"Sensor '{sensor.name}' is already registered")
        self._sensors[sensor.name] = sensor

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def initialize_all(self) -> None:
        """Initialize all registered sensors that are not yet initialized."""
        for sensor in self._sensors.values():
            if not sensor.initialized:
                sensor.initialize()

    def initialize(self, name: str) -> None:
        """Initialize a specific sensor by name."""
        self._sensors[name].initialize()

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def get_sensor(self, name: str) -> Sensor:
        """
        Get a sensor by name.

        Args:
            name: The sensor's registered name.

        Returns:
            The Sensor instance.

        Raises:
            KeyError: If no sensor with that name exists.
        """
        if name not in self._sensors:
            raise KeyError(
                f"Sensor '{name}' not found. Registered: {list(self._sensors.keys())}"
            )
        return self._sensors[name]

    def get_camera(self, name: str) -> Camera:
        """Get a registered camera by name (type-checked)."""
        sensor = self.get_sensor(name)
        if not isinstance(sensor, Camera):
            raise TypeError(f"Sensor '{name}' is {type(sensor).__name__}, not Camera")
        return sensor

    def get_imu(self, name: str) -> IMUSensor:
        """Get a registered IMU by name (type-checked)."""
        sensor = self.get_sensor(name)
        if not isinstance(sensor, IMUSensor):
            raise TypeError(f"Sensor '{name}' is {type(sensor).__name__}, not IMUSensor")
        return sensor

    def get_lidar(self, name: str) -> LiDAR:
        """Get a registered LiDAR by name (type-checked)."""
        sensor = self.get_sensor(name)
        if not isinstance(sensor, LiDAR):
            raise TypeError(f"Sensor '{name}' is {type(sensor).__name__}, not LiDAR")
        return sensor

    # ------------------------------------------------------------------ #
    # Bulk queries
    # ------------------------------------------------------------------ #

    def get_all_cameras(self) -> List[Camera]:
        """Return all registered camera sensors."""
        return [s for s in self._sensors.values() if isinstance(s, Camera)]

    def get_all_imus(self) -> List[IMUSensor]:
        """Return all registered IMU sensors."""
        return [s for s in self._sensors.values() if isinstance(s, IMUSensor)]

    def get_all_lidars(self) -> List[LiDAR]:
        """Return all registered LiDAR sensors."""
        return [s for s in self._sensors.values() if isinstance(s, LiDAR)]

    @property
    def sensor_names(self) -> List[str]:
        """List of all registered sensor names."""
        return list(self._sensors.keys())

    @property
    def count(self) -> int:
        """Total number of registered sensors."""
        return len(self._sensors)

    def __contains__(self, name: str) -> bool:
        return name in self._sensors

    def __repr__(self) -> str:
        cameras = len(self.get_all_cameras())
        imus = len(self.get_all_imus())
        lidars = len(self.get_all_lidars())
        return f"SensorManager(cameras={cameras}, imus={imus}, lidars={lidars})"
