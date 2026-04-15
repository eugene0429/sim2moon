"""Automatic camera discovery and ROS2 OmniGraph publisher setup.

Provides a hybrid approach for camera ROS2 publishing:
1. Auto-discover Camera prims by traversing the robot's USD hierarchy
2. Allow config overrides for topic names, resolution, and image type
3. Programmatically create OmniGraph nodes for each camera

This removes the need to manually configure OmniGraph in the Isaac Sim GUI.

Usage:
    publisher = CameraROS2Publisher()
    # Auto-discover + config overrides
    publisher.setup_robot_cameras(
        robot_prim_path="/Robots/husky",
        robot_name="husky",
        overrides={"left_camera": CameraROS2Conf(topic="/husky/left/image_raw")},
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sensors.config import CameraROS2Conf

logger = logging.getLogger(__name__)

try:
    from pxr import Usd, UsdGeom
    _HAS_USD = True
except ImportError:
    _HAS_USD = False

try:
    import omni.graph.core as og
    _HAS_OMNIGRAPH = True
except ImportError:
    _HAS_OMNIGRAPH = False


@dataclass
class DiscoveredCamera:
    """A camera discovered from the robot USD hierarchy."""

    prim_path: str
    prim_name: str
    parent_link: str  # The parent link name (for frame_id)


class CameraROS2Publisher:
    """Manages automatic camera discovery and OmniGraph ROS2 publisher creation.

    Workflow:
        1. discover_cameras() — Traverse robot USD to find all Camera prims
        2. merge_overrides() — Apply user config on top of discovered cameras
        3. create_omnigraph() — Build OmniGraph nodes for each camera
        4. setup_robot_cameras() — All-in-one convenience method
    """

    def __init__(self) -> None:
        self._graphs: Dict[str, str] = {}  # camera_name -> graph_path

    # ── Discovery ───────────────────────────────────────────────────────

    @staticmethod
    def discover_cameras(
        robot_prim_path: str,
        stage: Optional[object] = None,
    ) -> List[DiscoveredCamera]:
        """Find all Camera prims under a robot prim path.

        Recursively traverses the USD hierarchy and identifies prims
        of type UsdGeom.Camera.

        Args:
            robot_prim_path: USD path of the robot root prim.
            stage: USD stage. If None, uses the current stage.

        Returns:
            List of DiscoveredCamera with prim paths and names.
        """
        if not _HAS_USD:
            logger.warning("USD not available, cannot discover cameras")
            return []

        if stage is None:
            try:
                import omni.usd
                stage = omni.usd.get_context().get_stage()
            except ImportError:
                logger.warning("omni.usd not available, cannot auto-detect stage")
                return []

        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        if not robot_prim.IsValid():
            logger.warning("Robot prim not found at '%s'", robot_prim_path)
            return []

        cameras = []
        for prim in Usd.PrimRange(robot_prim):
            if prim.IsA(UsdGeom.Camera):
                prim_path = str(prim.GetPath())
                prim_name = prim.GetName()
                # Parent link is the prim's parent (usually a link frame)
                parent = prim.GetParent()
                parent_link = parent.GetName() if parent else ""
                cameras.append(DiscoveredCamera(
                    prim_path=prim_path,
                    prim_name=prim_name,
                    parent_link=parent_link,
                ))

        logger.info(
            "Discovered %d cameras under '%s': %s",
            len(cameras), robot_prim_path,
            [c.prim_name for c in cameras],
        )
        return cameras

    # ── Config merge ────────────────────────────────────────────────────

    @staticmethod
    def build_camera_configs(
        discovered: List[DiscoveredCamera],
        robot_name: str,
        overrides: Optional[Dict[str, CameraROS2Conf]] = None,
    ) -> Dict[str, CameraROS2Conf]:
        """Merge auto-discovered cameras with user config overrides.

        For each discovered camera:
        - If an override exists (matched by prim_name or parent_link), use it
        - Otherwise, generate a default config with auto-derived topic name

        Override keys are matched against both prim_name and parent_link,
        allowing flexible config like:
            overrides={"left_camera": CameraROS2Conf(...)}  # match by prim name
            overrides={"left_camera_link": CameraROS2Conf(...)}  # match by link

        Args:
            discovered: List of discovered cameras.
            robot_name: Robot name for topic namespace.
            overrides: Optional dict of config overrides keyed by camera/link name.

        Returns:
            Dict of camera_key -> CameraROS2Conf (merged).
        """
        overrides = overrides or {}
        configs: Dict[str, CameraROS2Conf] = {}

        for cam in discovered:
            # Try to match override by prim_name or parent_link
            override = overrides.get(cam.prim_name) or overrides.get(cam.parent_link)

            if override is not None:
                # Use override, filling in missing fields from discovery
                conf = CameraROS2Conf(
                    prim_path=override.prim_path or cam.prim_path,
                    topic=override.topic or f"/{robot_name}/{cam.prim_name}/image_raw",
                    image_type=override.image_type,
                    resolution=override.resolution,
                    frame_id=override.frame_id or f"{robot_name}/{cam.parent_link}",
                    enabled=override.enabled,
                )
            else:
                # Auto-generate default config
                conf = CameraROS2Conf(
                    prim_path=cam.prim_path,
                    topic=f"/{robot_name}/{cam.prim_name}/image_raw",
                    image_type="rgb",
                    resolution=[640, 480],
                    frame_id=f"{robot_name}/{cam.parent_link}",
                    enabled=True,
                )

            configs[cam.prim_name] = conf

        # Add explicit overrides for cameras not discovered (manual config)
        for key, override in overrides.items():
            if key not in configs and override.prim_path:
                configs[key] = override
                logger.info("Added manual camera config: %s -> %s", key, override.topic)

        return configs

    # ── OmniGraph creation ──────────────────────────────────────────────

    def create_omnigraph(
        self,
        camera_name: str,
        conf: CameraROS2Conf,
        graph_path_prefix: str = "/ROS2_Cameras",
    ) -> Optional[str]:
        """Create an OmniGraph pipeline for a single camera publisher.

        Pipeline:
            OnPlaybackTick → IsaacCreateRenderProduct → ROS2CameraHelper

        Args:
            camera_name: Unique name for this camera graph.
            conf: Camera publisher configuration.
            graph_path_prefix: USD path prefix for the graph.

        Returns:
            The graph path if created, None if OmniGraph is unavailable.
        """
        if not _HAS_OMNIGRAPH:
            logger.warning("OmniGraph not available, cannot create camera publisher for '%s'", camera_name)
            return None

        if not conf.enabled:
            logger.info("Camera '%s' is disabled, skipping OmniGraph", camera_name)
            return None

        graph_path = f"{graph_path_prefix}/{camera_name}"
        tick_node = f"{graph_path}/on_tick"
        render_node = f"{graph_path}/render_product"
        helper_node = f"{graph_path}/ros2_helper"

        try:
            keys = og.Controller.Keys
            og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        (tick_node, "omni.graph.action.OnPlaybackTick"),
                        (render_node, "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                        (helper_node, "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ],
                    keys.SET_VALUES: [
                        (f"{render_node}.inputs:cameraPrim", conf.prim_path),
                        (f"{render_node}.inputs:width", conf.resolution[0]),
                        (f"{render_node}.inputs:height", conf.resolution[1]),
                        (f"{helper_node}.inputs:topicName", conf.topic),
                        (f"{helper_node}.inputs:type", conf.image_type),
                        (f"{helper_node}.inputs:frameId", conf.frame_id),
                    ],
                    keys.CONNECT: [
                        (f"{tick_node}.outputs:tick", f"{render_node}.inputs:execIn"),
                        (f"{render_node}.outputs:execOut", f"{helper_node}.inputs:execIn"),
                        (f"{render_node}.outputs:renderProductPath", f"{helper_node}.inputs:renderProductPath"),
                    ],
                },
            )
            self._graphs[camera_name] = graph_path
            logger.info(
                "OmniGraph created: %s -> %s (%s, %dx%d)",
                camera_name, conf.topic, conf.image_type,
                conf.resolution[0], conf.resolution[1],
            )
            return graph_path

        except Exception as e:
            logger.error("Failed to create OmniGraph for '%s': %s", camera_name, e)
            return None

    # ── All-in-one convenience ──────────────────────────────────────────

    def setup_robot_cameras(
        self,
        robot_prim_path: str,
        robot_name: str,
        overrides: Optional[Dict[str, CameraROS2Conf]] = None,
        stage: Optional[object] = None,
    ) -> Dict[str, CameraROS2Conf]:
        """Discover cameras, merge config, and create OmniGraph publishers.

        This is the main entry point. Call once per robot after spawning.

        Args:
            robot_prim_path: USD path of the robot root prim.
            robot_name: Robot name for topic namespacing.
            overrides: Optional config overrides per camera name.
            stage: USD stage (auto-detected if None).

        Returns:
            Dict of camera_name -> final CameraROS2Conf used.

        Example YAML config for overrides:
            cameras:
              left_camera:
                topic: "/husky/left/image_raw"
                image_type: rgb
                resolution: [1280, 720]
              right_camera:
                topic: "/husky/right/image_raw"
                image_type: rgb
                resolution: [1280, 720]
              depth_camera:
                topic: "/husky/depth/image_raw"
                image_type: depth
        """
        # Step 1: Discover cameras from USD
        discovered = self.discover_cameras(robot_prim_path, stage)

        # Step 2: Merge with overrides
        configs = self.build_camera_configs(discovered, robot_name, overrides)

        if not configs:
            logger.warning(
                "No cameras found or configured for robot '%s' at '%s'",
                robot_name, robot_prim_path,
            )
            return {}

        # Step 3: Create OmniGraph for each camera
        for cam_name, conf in configs.items():
            self.create_omnigraph(cam_name, conf)

        logger.info(
            "Camera ROS2 setup complete for '%s': %d cameras",
            robot_name, len(configs),
        )
        return configs

    @property
    def active_graphs(self) -> Dict[str, str]:
        """Dict of camera_name -> OmniGraph path for all created graphs."""
        return dict(self._graphs)
