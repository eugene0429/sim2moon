"""
Camera sensor for RGB and depth image capture.

Wraps Isaac Sim's Camera class to provide resolution-independent access
to RGBA and depth images. Multiple resolution presets can be registered
and accessed by name (e.g., "low", "high").
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from sensors.base import Sensor
from sensors.config import CameraConf

# Deferred Isaac Sim imports
try:
    from omni.isaac.sensor import Camera as IsaacCamera
    _HAS_ISAAC = True
except ImportError:
    _HAS_ISAAC = False


class Camera(Sensor):
    """
    Camera sensor providing RGB and depth images.

    Supports multiple resolution presets. Each preset maintains its own
    Isaac Sim Camera wrapper for RGBA and depth independently.
    """

    def __init__(self, name: str, config: CameraConf) -> None:
        """
        Args:
            name: Unique sensor name.
            config: Camera configuration with prim_path and resolutions.
        """
        super().__init__(name, config.prim_path)
        self._config = config
        self._rgba_cameras: Dict[str, object] = {}
        self._depth_cameras: Dict[str, object] = {}

    def initialize(self) -> None:
        """
        Create Isaac Sim camera wrappers for each configured resolution.

        Must be called after the USD stage is ready and the camera prim exists.
        """
        if not _HAS_ISAAC:
            raise RuntimeError("Isaac Sim not available - cannot initialize camera sensor")

        for preset_name, res in self._config.resolutions.items():
            resolution = (res[0], res[1])

            # RGBA camera
            rgba_cam = IsaacCamera(self._prim_path, resolution=resolution)
            rgba_cam.initialize()
            self._rgba_cameras[preset_name] = rgba_cam

            # Depth camera (separate instance for depth-to-image-plane)
            depth_cam = IsaacCamera(self._prim_path, resolution=resolution)
            depth_cam.initialize()
            depth_cam.add_distance_to_image_plane_to_frame()
            self._depth_cameras[preset_name] = depth_cam

        self._initialized = True

    def get_reading(self) -> Dict[str, np.ndarray]:
        """
        Return RGBA images from all resolution presets.

        Returns:
            Dict mapping preset name to RGBA numpy array.
        """
        return {name: cam.get_rgba() for name, cam in self._rgba_cameras.items()}

    def get_rgb(self, resolution: str = "low") -> np.ndarray:
        """
        Get an RGBA image at the specified resolution preset.

        Args:
            resolution: Resolution preset name (e.g., "low", "high").

        Returns:
            RGBA image as numpy array [H, W, 4].
        """
        if resolution not in self._rgba_cameras:
            available = list(self._rgba_cameras.keys())
            raise KeyError(
                f"Resolution preset '{resolution}' not found. Available: {available}"
            )
        return self._rgba_cameras[resolution].get_rgba()

    def get_depth(self, resolution: str = "low") -> np.ndarray:
        """
        Get a depth image at the specified resolution preset.

        Args:
            resolution: Resolution preset name (e.g., "low", "high").

        Returns:
            Depth image as float32 numpy array [H, W] in meters.
        """
        if resolution not in self._depth_cameras:
            available = list(self._depth_cameras.keys())
            raise KeyError(
                f"Resolution preset '{resolution}' not found. Available: {available}"
            )
        return self._depth_cameras[resolution].get_depth()

    @property
    def resolution_presets(self) -> List[str]:
        """List of available resolution preset names."""
        return list(self._config.resolutions.keys())

    def get_resolution(self, preset: str) -> Tuple[int, int]:
        """Get the (width, height) for a resolution preset."""
        res = self._config.resolutions[preset]
        return (res[0], res[1])
