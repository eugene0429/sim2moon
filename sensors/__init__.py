"""
Sensor system for the new lunar simulation.

Provides a decoupled, uniform interface for all sensor types:
cameras (RGB/depth), IMU, LiDAR, and fixed monitoring cameras.
Sensors are registered and managed independently from robot code.
"""

from sensors.sensor_manager import SensorManager
from sensors.camera import Camera
from sensors.imu import IMUSensor, IMUReading
from sensors.lidar import LiDAR
from sensors.monitoring_cameras import MonitoringCamerasManager

__all__ = [
    "SensorManager",
    "Camera",
    "IMUSensor",
    "IMUReading",
    "LiDAR",
    "MonitoringCamerasManager",
]
