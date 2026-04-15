"""
LiDAR sensor wrapper.

Wraps Isaac Sim's LiDAR/RTX Lidar sensor. LiDAR configuration is
primarily defined in the robot's USD file; this wrapper provides
a uniform interface for point cloud access and ROS2 integration setup.
"""

from typing import Optional

import numpy as np

from sensors.base import Sensor
from sensors.config import LiDARConf

# Deferred Isaac Sim imports
try:
    import omni
    from pxr import UsdGeom
    _HAS_ISAAC = True
except ImportError:
    _HAS_ISAAC = False


class LiDAR(Sensor):
    """
    LiDAR sensor providing point cloud data.

    In Isaac Sim, LiDAR sensors are typically configured via OmniGraph
    within the robot USD file. This wrapper provides a uniform interface
    and manages the render product for point cloud access.
    """

    def __init__(self, name: str, config: LiDARConf) -> None:
        """
        Args:
            name: Unique sensor name.
            config: LiDAR configuration.
        """
        super().__init__(name, config.prim_path)
        self._config = config
        self._render_product = None
        self._writer = None

    def initialize(self) -> None:
        """
        Set up the LiDAR render product for point cloud access.

        The LiDAR prim itself is expected to exist in the USD scene
        (loaded from the robot USD file). This method creates the
        render product pipeline for data extraction.
        """
        if not _HAS_ISAAC:
            raise RuntimeError("Isaac Sim not available - cannot initialize LiDAR sensor")

        if not self._prim_path:
            raise ValueError("LiDAR prim_path not defined")

        try:
            import omni.replicator.core as rep

            self._render_product = rep.create.render_product(
                self._prim_path, resolution=(1, 1)  # LiDAR doesn't use pixel resolution
            )
            self._writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
            self._writer.attach([self._render_product])
        except (ImportError, Exception):
            # LiDAR may be handled entirely through OmniGraph / ROS2 bridge
            pass

        self._initialized = True

    def get_reading(self) -> Optional[np.ndarray]:
        """
        Get the latest point cloud.

        Returns:
            Point cloud as [N, 3] float32 array (x, y, z in sensor frame),
            or None if point cloud is not available through this interface.

        Note:
            In many setups, the LiDAR point cloud is published directly
            to ROS2 via OmniGraph and not read through this method.
        """
        if not self._initialized:
            raise RuntimeError(f"LiDAR sensor '{self._name}' not initialized")

        # Point cloud access depends on the specific LiDAR setup
        # In most Isaac Sim configurations, the point cloud goes directly
        # to ROS2 via OmniGraph. This method provides an alternative
        # programmatic access path.
        if self._writer is not None:
            try:
                data = self._writer.get_data()
                if data and "data" in data:
                    return np.array(data["data"], dtype=np.float32).reshape(-1, 3)
            except Exception:
                pass

        return None

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Alias for get_reading()."""
        return self.get_reading()

    @property
    def max_range(self) -> float:
        return self._config.max_range

    @property
    def min_range(self) -> float:
        return self._config.min_range
