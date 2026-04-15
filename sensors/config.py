"""
Sensor configuration dataclasses.

All sensor-related configuration is defined here as typed dataclasses
with validation.
"""

import dataclasses
from typing import Dict, List, Optional, Tuple


@dataclasses.dataclass
class CameraConf:
    """Configuration for a single camera sensor."""

    prim_path: str = ""
    focal_length: float = 1.93
    horizontal_aperture: float = 2.4
    vertical_aperture: float = 1.8
    fstop: float = 0.0
    focus_distance: float = 10.0
    clipping_range: Tuple[float, float] = (0.01, 1000000.0)
    resolutions: Dict[str, List[int]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        assert self.focal_length > 0, "focal_length must be positive"
        assert self.horizontal_aperture > 0, "horizontal_aperture must be positive"
        assert self.vertical_aperture > 0, "vertical_aperture must be positive"
        assert self.fstop >= 0, "fstop must be non-negative"
        assert self.focus_distance > 0, "focus_distance must be positive"
        assert len(self.clipping_range) == 2, "clipping_range must have 2 elements"
        assert self.clipping_range[1] > self.clipping_range[0], (
            "clipping_range far must be greater than near"
        )


@dataclasses.dataclass
class IMUConf:
    """Configuration for an IMU sensor."""

    sensor_path: str = ""
    use_latest_data: bool = True
    read_gravity: bool = True


@dataclasses.dataclass
class LiDARConf:
    """Configuration for a LiDAR sensor."""

    prim_path: str = ""
    rotation_rate: float = 20.0
    horizontal_fov: float = 360.0
    vertical_fov_upper: float = 15.0
    vertical_fov_lower: float = -15.0
    horizontal_resolution: float = 0.4
    vertical_resolution: float = 2.0
    max_range: float = 100.0
    min_range: float = 0.4


@dataclasses.dataclass
class MonitoringCameraConf:
    """Configuration for a single fixed monitoring camera."""

    name: str = ""
    prim_path: str = ""
    resolution: List[int] = dataclasses.field(default_factory=lambda: [640, 480])
    camera_params: Dict = dataclasses.field(default_factory=dict)
    pose: Dict = dataclasses.field(default_factory=dict)
    sensor_type: str = "rgb"
    ros2: Optional[Dict] = None

    def __post_init__(self):
        assert len(self.resolution) == 2, "resolution must have 2 elements"
        assert self.sensor_type in ("rgb", "depth", "semantic_segmentation"), (
            f"unsupported sensor_type: {self.sensor_type}"
        )


@dataclasses.dataclass
class MonitoringCamerasConf:
    """Configuration for all fixed monitoring cameras."""

    root_path: str = "/MonitoringCameras"
    camera_definitions: List[MonitoringCameraConf] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        resolved = []
        for cd in self.camera_definitions:
            if isinstance(cd, dict):
                resolved.append(MonitoringCameraConf(**cd))
            else:
                resolved.append(cd)
        self.camera_definitions = resolved


@dataclasses.dataclass
class CameraROS2Conf:
    """Configuration for a single camera's ROS2 OmniGraph publisher.

    Used to override auto-discovered cameras or specify cameras explicitly.
    If prim_path is empty, the camera will be discovered from the robot USD.
    """

    prim_path: str = ""
    topic: str = ""
    image_type: str = "rgb"  # "rgb", "depth", "semantic_segmentation"
    resolution: List[int] = dataclasses.field(default_factory=lambda: [640, 480])
    frame_id: str = ""
    enabled: bool = True

    def __post_init__(self):
        assert self.image_type in ("rgb", "depth", "semantic_segmentation"), (
            f"unsupported image_type: {self.image_type}"
        )
        assert len(self.resolution) == 2, "resolution must have 2 elements [width, height]"
        assert self.resolution[0] > 0 and self.resolution[1] > 0, (
            "resolution must be positive"
        )
