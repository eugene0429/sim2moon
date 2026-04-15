"""
Fixed monitoring cameras manager.

Spawns and manages fixed observation cameras in the scene that are
independent of any robot. Supports RGB, depth, and semantic segmentation
output types, with optional ROS2 publishing.
"""

import os
from typing import Dict, Optional

from sensors.config import MonitoringCamerasConf

# Deferred Isaac Sim imports
try:
    import omni
    import omni.replicator.core as rep
    import omni.syntheticdata as sd
    from omni.isaac.sensor import Camera as IsaacCamera
    from pxr import UsdGeom, Gf
    _HAS_ISAAC = True
except ImportError:
    _HAS_ISAAC = False


class MonitoringCamerasManager:
    """
    Manages fixed observation cameras in the scene.

    These cameras are scene-level, not attached to any robot. They are used
    for:
    - Scene monitoring / debugging
    - External viewpoint capture
    - Synthetic data generation from fixed viewpoints
    """

    def __init__(
        self,
        config: MonitoringCamerasConf,
        is_ros2: bool = False,
        utils_module=None,
    ) -> None:
        """
        Args:
            config: Monitoring cameras configuration.
            is_ros2: Whether ROS2 publishers should be set up.
            utils_module: USD utility module with set_xform_pose (injected).
        """
        self._config = config
        self._is_ros2 = is_ros2
        self._utils = utils_module
        self._cameras: Dict[str, object] = {}
        self._stage = None

    def initialize(self, stage=None) -> None:
        """Set up the USD stage and create the root Xform."""
        if not _HAS_ISAAC:
            return
        self._stage = stage or omni.usd.get_context().get_stage()
        self._stage.DefinePrim(self._config.root_path, "Xform")

    def spawn(self) -> None:
        """
        Spawn all configured monitoring cameras.

        Creates USD camera prims, sets their attributes (focal length,
        aperture, clipping), positions them, and optionally sets up
        ROS2 publishers.
        """
        if not _HAS_ISAAC or self._stage is None:
            return

        for cam_def in self._config.camera_definitions:
            name = cam_def.name
            prim_path = os.path.join(self._config.root_path, name)

            # Create camera USD prim
            prim = self._stage.DefinePrim(prim_path, "Camera")
            cam_usd = UsdGeom.Camera(prim)

            # Set position and orientation
            if cam_def.pose and self._utils is not None:
                xform = UsdGeom.Xformable(self._stage.GetPrimAtPath(prim_path))
                self._utils.set_xform_pose(
                    xform,
                    cam_def.pose.get("position", [0, 0, 0]),
                    cam_def.pose.get("orientation", [0, 0, 0, 1]),
                )

            # Set camera attributes
            if cam_def.camera_params:
                self._set_camera_attributes(cam_usd, cam_def.camera_params)

            # Create Isaac Sim camera wrapper
            resolution = (cam_def.resolution[0], cam_def.resolution[1])
            isaac_cam = IsaacCamera(prim_path, resolution=resolution)
            isaac_cam.initialize()
            self._cameras[name] = isaac_cam

            # Set up ROS2 publisher if in ROS2 mode
            if self._is_ros2 and cam_def.ros2 is not None:
                rp = rep.create.render_product(prim_path, resolution)
                self._set_ros2_publisher(rp, cam_def.ros2, cam_def.sensor_type)

    def _set_camera_attributes(self, camera: "UsdGeom.Camera", params: Dict) -> None:
        """Apply camera parameters from configuration."""
        if "focal_length" in params:
            camera.CreateFocalLengthAttr().Set(float(params["focal_length"]))
        if "horizontal_aperture" in params:
            camera.CreateHorizontalApertureAttr().Set(float(params["horizontal_aperture"]))
        if "vertical_aperture" in params:
            camera.CreateVerticalApertureAttr().Set(float(params["vertical_aperture"]))
        if "fstop" in params:
            camera.CreateFStopAttr().Set(float(params["fstop"]))
        if "focus_distance" in params:
            camera.CreateFocusDistanceAttr().Set(float(params["focus_distance"]))
        if "clipping_range" in params:
            clip = params["clipping_range"]
            camera.CreateClippingRangeAttr().Set(Gf.Vec2f(float(clip[0]), float(clip[1])))

    def _set_ros2_publisher(self, render_product, ros2_cfg: Dict, sensor_type: str) -> None:
        """Set up a ROS2 image publisher for the camera."""
        rv = sd.SyntheticData.convert_sensor_type_to_rendervar(sensor_type)
        writer = rep.writers.get(rv + "ROS2PublishImage")
        writer.initialize(
            topicName=ros2_cfg.get("topic", "/monitoring/image"),
            frameId=ros2_cfg.get("frame_id", "monitoring_camera"),
            queueSize=ros2_cfg.get("queue_size", 1),
            nodeNamespace=ros2_cfg.get("namespace", ""),
        )
        writer.attach([render_product])

    def get_camera(self, name: str) -> Optional[object]:
        """Get a monitoring camera by name."""
        return self._cameras.get(name)

    def get_rgb(self, name: str):
        """Get an RGBA image from a monitoring camera."""
        cam = self._cameras.get(name)
        if cam is None:
            raise KeyError(f"Monitoring camera '{name}' not found")
        return cam.get_rgba()

    def get_depth(self, name: str):
        """Get a depth image from a monitoring camera."""
        cam = self._cameras.get(name)
        if cam is None:
            raise KeyError(f"Monitoring camera '{name}' not found")
        return cam.get_depth()

    @property
    def camera_names(self):
        """List of spawned camera names."""
        return list(self._cameras.keys())
