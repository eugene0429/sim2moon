"""Configuration dataclasses for the robot system."""

import dataclasses
import os
from typing import Dict, List, Optional


@dataclasses.dataclass
class Pose:
    """Robot pose: position (x, y, z) and orientation quaternion (w, x, y, z)."""

    position: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.5])
    orientation: List[float] = dataclasses.field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    def __post_init__(self):
        if len(self.position) != 3:
            raise ValueError(f"position must have 3 elements, got {len(self.position)}")
        if len(self.orientation) != 4:
            raise ValueError(f"orientation must have 4 elements, got {len(self.orientation)}")


@dataclasses.dataclass
class RobotParameters:
    """Configuration for a single robot instance.

    The ``cameras_ros2`` field allows per-camera ROS2 publisher overrides.
    Keys are camera prim names or parent link names. Example::

        cameras_ros2:
          left_camera:
            topic: "/husky/left/image_raw"
            image_type: rgb
            resolution: [1280, 720]
          right_camera:
            topic: "/husky/right/image_raw"
    """

    robot_name: str = "husky"
    usd_path: str = ""
    pose: Pose = dataclasses.field(default_factory=Pose)
    domain_id: int = 0
    target_links: List[str] = dataclasses.field(default_factory=list)
    base_link: str = "base_link"
    wheel_joints: Dict = dataclasses.field(default_factory=dict)
    camera: Dict = dataclasses.field(default_factory=dict)
    cameras_ros2: Dict = dataclasses.field(default_factory=dict)
    imu_sensor_path: str = ""
    dimensions: Dict = dataclasses.field(default_factory=dict)
    turn_speed_coef: float = 1.0
    pos_relative_to_prim: str = ""
    solar_panel_joint: str = ""

    def __post_init__(self):
        if isinstance(self.pose, dict):
            self.pose = Pose(**self.pose)
        if self.usd_path and not os.path.isabs(self.usd_path):
            self.usd_path = os.path.join(os.getcwd(), self.usd_path)
        # Resolve cameras_ros2 dicts into CameraROS2Conf
        from sensors.config import CameraROS2Conf
        resolved = {}
        for key, val in self.cameras_ros2.items():
            if isinstance(val, dict):
                resolved[key] = CameraROS2Conf(**val)
            else:
                resolved[key] = val
        self.cameras_ros2 = resolved


@dataclasses.dataclass
class RobotManagerConf:
    """Top-level robot manager configuration."""

    uses_nucleus: bool = False
    is_ROS2: bool = True
    max_robots: int = 5
    robots_root: str = "/Robots"
    parameters: List[RobotParameters] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        resolved = []
        for param in self.parameters:
            if isinstance(param, dict):
                resolved.append(RobotParameters(**param))
            else:
                resolved.append(param)
        self.parameters = resolved

        if len(self.parameters) > self.max_robots:
            raise ValueError(
                f"Number of robots ({len(self.parameters)}) exceeds max_robots ({self.max_robots})"
            )
