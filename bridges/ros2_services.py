"""ROS2 service handlers for environment and robot control.

Provides request/response services for operations that require
acknowledgment (reset, terrain generation, etc.), complementing
the topic-based fire-and-forget control in ros2_node.py.

Ported from OmniLRS ros_manager.py service pattern.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rclpy.node import Node
    from std_srvs.srv import Trigger, SetBool
    from geometry_msgs.msg import PoseStamped
    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False


class EnvironmentServiceHandler:
    """Registers and handles ROS2 services for environment control.

    Services:
        /OmniLRS/ResetEnvironment  (Trigger)   re-generate terrain + rocks
        /OmniLRS/EnableDeformation (SetBool)   toggle terrain deformation

    Designed to be attached to an existing ROS2 node.
    """

    def __init__(self, node, environment=None) -> None:
        self._node = node
        self._env = environment
        self._deformation_enabled = False
        self._reset_requested = False

        if not _HAS_ROS2:
            return

        node.create_service(
            Trigger,
            "/OmniLRS/ResetEnvironment",
            self._handle_reset,
        )
        node.create_service(
            SetBool,
            "/OmniLRS/EnableDeformation",
            self._handle_deformation_toggle,
        )
        logger.info("EnvironmentServiceHandler: 2 services registered")

    @property
    def deformation_enabled(self) -> bool:
        return self._deformation_enabled

    @property
    def reset_requested(self) -> bool:
        return self._reset_requested

    def clear_reset(self) -> None:
        self._reset_requested = False

    def _handle_reset(self, request, response) -> Any:
        """Handle /OmniLRS/ResetEnvironment service call."""
        self._reset_requested = True
        response.success = True
        response.message = "Environment reset queued"
        logger.info("Service: environment reset requested")
        return response

    def _handle_deformation_toggle(self, request, response) -> Any:
        """Handle /OmniLRS/EnableDeformation service call."""
        self._deformation_enabled = request.data
        response.success = True
        response.message = f"Deformation {'enabled' if request.data else 'disabled'}"
        logger.info("Service: deformation %s", "enabled" if request.data else "disabled")
        return response


class RobotServiceHandler:
    """Registers and handles ROS2 services for per-robot control.

    For each robot, creates:
        /{robot_name}/reset_pose  (Trigger)  reset to initial pose

    Also provides a topic-based reset with target pose:
        /{robot_name}/reset_pose_target  (PoseStamped)  reset to specific pose

    Designed to be attached to an existing ROS2 node.
    """

    def __init__(self, node, robot_manager=None) -> None:
        self._node = node
        self._rm = robot_manager
        self._reset_requests: Dict[str, Optional[Tuple[List[float], List[float]]]] = {}
        self._services: Dict[str, Any] = {}
        self._subscriptions: Dict[str, Any] = {}

    def register_robot(self, robot_name: str) -> None:
        """Create reset service and target subscription for a robot."""
        if not _HAS_ROS2:
            return

        # Reset service
        srv = self._node.create_service(
            Trigger,
            f"/{robot_name}/reset_pose",
            lambda req, resp, name=robot_name: self._handle_reset(name, req, resp),
        )
        self._services[robot_name] = srv

        # Reset with target pose
        sub = self._node.create_subscription(
            PoseStamped,
            f"/{robot_name}/reset_pose_target",
            lambda msg, name=robot_name: self._handle_reset_target(name, msg),
            10,
        )
        self._subscriptions[robot_name] = sub
        logger.info("RobotServiceHandler: registered services for '%s'", robot_name)

    def is_reset_requested(self, robot_name: str) -> Tuple[bool, Optional[Tuple]]:
        """Check if a reset was requested for the given robot.

        Returns:
            (requested, target_pose) where target_pose is None for default reset
            or (position, orientation) for specific pose reset.
        """
        if robot_name in self._reset_requests:
            target = self._reset_requests.pop(robot_name)
            return True, target
        return False, None

    def _handle_reset(self, robot_name: str, request, response) -> Any:
        """Handle /{robot_name}/reset_pose trigger."""
        self._reset_requests[robot_name] = None  # None = reset to initial
        response.success = True
        response.message = f"Reset queued for {robot_name}"
        logger.info("Service: reset requested for '%s'", robot_name)
        return response

    def _handle_reset_target(self, robot_name: str, msg: "PoseStamped") -> None:
        """Handle /{robot_name}/reset_pose_target topic."""
        from bridges.ros2_node import ros_to_isaac_quat

        p = msg.pose.position
        o = msg.pose.orientation
        position = [p.x, p.y, p.z]
        orientation = ros_to_isaac_quat(o.x, o.y, o.z, o.w)
        self._reset_requests[robot_name] = (position, orientation)
        logger.info("Reset target received for '%s': [%.1f, %.1f, %.1f]",
                     robot_name, p.x, p.y, p.z)
