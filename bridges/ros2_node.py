"""ROS2 nodes for environment and robot control.

Two nodes with clear responsibilities:
- EnvironmentNode: subscribes to sun/terrain/rendering topics, queues modifications
- RobotNode: subscribes to spawn/teleport/reset topics, publishes TF and sensor data

Both use the modification queue pattern for thread-safe stage access:
    ROS callbacks append (callable, kwargs) to a list.
    The main thread calls apply_modifications() after world.step().

Ported from OmniLRS base_wrapper_ros2.py, lunaryard_ros2.py, robot_manager_ros2.py
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False

# ROS2 message types — guarded import
try:
    from std_msgs.msg import Bool, Empty, Float32, Int32, String, Float64
    from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Vector3
    from sensor_msgs.msg import BatteryState
    from tf2_msgs.msg import TFMessage
    _HAS_ROS2_MSGS = True
except ImportError:
    _HAS_ROS2_MSGS = False


# Isaac Sim uses (w, x, y, z), ROS uses (x, y, z, w)
def isaac_to_ros_quat(q_isaac: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert quaternion from Isaac (w,x,y,z) to ROS (x,y,z,w)."""
    return (float(q_isaac[1]), float(q_isaac[2]), float(q_isaac[3]), float(q_isaac[0]))


def ros_to_isaac_quat(x: float, y: float, z: float, w: float) -> List[float]:
    """Convert quaternion from ROS (x,y,z,w) to Isaac (w,x,y,z)."""
    return [w, x, y, z]


# Default QoS for control topics
_CONTROL_QOS = QoSProfile(depth=10) if _HAS_ROS2 else None

# Latched QoS for state topics (last value available to new subscribers)
_LATCHED_QOS = (
    QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    if _HAS_ROS2
    else None
)


class _ModificationMixin:
    """Mixin providing the thread-safe modification queue pattern."""

    def __init__(self) -> None:
        self._modifications: List[Tuple[Callable, Dict[str, Any]]] = []

    def _queue(self, func: Callable, **kwargs: Any) -> None:
        """Queue a modification to be applied on the main thread."""
        self._modifications.append((func, kwargs))

    def apply_modifications(self) -> None:
        """Apply all queued modifications (called from main thread)."""
        for func, kwargs in self._modifications:
            try:
                func(**kwargs)
            except Exception as e:
                logger.error("Modification failed: %s(%s) -> %s", func.__name__, kwargs, e)
        self._modifications.clear()

    def clear_modifications(self) -> None:
        """Discard all pending modifications."""
        self._modifications.clear()


# ── Stub base for when ROS2 is not available ────────────────────────────────

class _StubNode:
    """Minimal stub when rclpy is not installed."""

    def __init__(self, name: str, **kwargs):
        self._name = name

    def get_logger(self):
        return logger

    def create_subscription(self, *args, **kwargs):
        pass

    def create_publisher(self, *args, **kwargs):
        return None

    def get_clock(self):
        return None

    def destroy_node(self):
        pass


_BaseNode = Node if _HAS_ROS2 else _StubNode


# ── EnvironmentNode ─────────────────────────────────────────────────────────

class EnvironmentNode(_BaseNode, _ModificationMixin):
    """ROS2 node for environment control (sun, terrain, rendering).

    Subscriptions:
        /OmniLRS/Sun/Intensity          (Float32)
        /OmniLRS/Sun/Color              (Vector3) - r, g, b
        /OmniLRS/Sun/ColorTemperature   (Float32)
        /OmniLRS/Sun/AngularSize        (Float32)
        /OmniLRS/Terrain/Switch         (Int32)   -> triggers reset
        /OmniLRS/Terrain/EnableRocks    (Bool)    -> triggers reset
        /OmniLRS/Terrain/RandomizeRocks (Int32)   -> triggers reset
        /OmniLRS/Render/EnableRTXRealTime    (Empty)
        /OmniLRS/Render/EnableRTXInteractive (Empty)
        /OmniLRS/LensFlare/Enable       (Bool)
        /OmniLRS/LensFlare/Scale        (Float32)
    """

    def __init__(self, environment=None, cfg: dict = None) -> None:
        _BaseNode.__init__(self, "omnilrs_environment")
        _ModificationMixin.__init__(self)

        self._env = environment
        self._cfg = cfg or {}
        self.trigger_reset: bool = False

        if not _HAS_ROS2_MSGS:
            logger.warning("ROS2 message types not available, subscriptions disabled")
            return

        # Sun control
        self.create_subscription(Float32, "/OmniLRS/Sun/Intensity", self._cb_sun_intensity, _CONTROL_QOS)
        self.create_subscription(Vector3, "/OmniLRS/Sun/Color", self._cb_sun_color, _CONTROL_QOS)
        self.create_subscription(Float32, "/OmniLRS/Sun/ColorTemperature", self._cb_sun_temperature, _CONTROL_QOS)
        self.create_subscription(Float32, "/OmniLRS/Sun/AngularSize", self._cb_sun_angle, _CONTROL_QOS)

        # Terrain control
        self.create_subscription(Int32, "/OmniLRS/Terrain/Switch", self._cb_terrain_switch, _CONTROL_QOS)
        self.create_subscription(Bool, "/OmniLRS/Terrain/EnableRocks", self._cb_enable_rocks, _CONTROL_QOS)
        self.create_subscription(Int32, "/OmniLRS/Terrain/RandomizeRocks", self._cb_randomize_rocks, _CONTROL_QOS)

        # Rendering control
        self.create_subscription(Empty, "/OmniLRS/Render/EnableRTXRealTime", self._cb_rtx_realtime, _CONTROL_QOS)
        self.create_subscription(Empty, "/OmniLRS/Render/EnableRTXInteractive", self._cb_rtx_interactive, _CONTROL_QOS)

        # Lens flare control
        self.create_subscription(Bool, "/OmniLRS/LensFlare/Enable", self._cb_lensflare_enable, _CONTROL_QOS)
        self.create_subscription(Float32, "/OmniLRS/LensFlare/Scale", self._cb_lensflare_scale, _CONTROL_QOS)

        logger.info("EnvironmentNode: 12 subscriptions created")

    def periodic_update(self, dt: float) -> None:
        """Called each physics step for time-dependent updates."""
        pass

    # -- Sun callbacks --

    def _cb_sun_intensity(self, msg: "Float32") -> None:
        if self._env is not None and hasattr(self._env, "_sun_controller"):
            ctrl = self._env._sun_controller
            if ctrl is not None:
                self._queue(ctrl.set_intensity, intensity=msg.data)

    def _cb_sun_color(self, msg: "Vector3") -> None:
        if self._env is not None and hasattr(self._env, "_sun_controller"):
            ctrl = self._env._sun_controller
            if ctrl is not None:
                self._queue(ctrl.set_color, color=[msg.x, msg.y, msg.z])

    def _cb_sun_temperature(self, msg: "Float32") -> None:
        if self._env is not None and hasattr(self._env, "_sun_controller"):
            ctrl = self._env._sun_controller
            if ctrl is not None:
                self._queue(ctrl.set_color_temperature, temperature=msg.data)

    def _cb_sun_angle(self, msg: "Float32") -> None:
        if self._env is not None and hasattr(self._env, "_sun_controller"):
            ctrl = self._env._sun_controller
            if ctrl is not None:
                self._queue(ctrl.set_angle, angle=msg.data)

    # -- Terrain callbacks --

    def _cb_terrain_switch(self, msg: "Int32") -> None:
        self.trigger_reset = True
        logger.info("Terrain switch requested (id=%d)", msg.data)

    def _cb_enable_rocks(self, msg: "Bool") -> None:
        if self._env is not None and hasattr(self._env, "_rock_manager"):
            rm = self._env._rock_manager
            if rm is not None:
                self._queue(rm.set_visible, flag=msg.data)
        self.trigger_reset = True

    def _cb_randomize_rocks(self, msg: "Int32") -> None:
        if self._env is not None and hasattr(self._env, "_rock_manager"):
            rm = self._env._rock_manager
            if rm is not None:
                self._queue(rm.randomize, num_rocks=msg.data)
        self.trigger_reset = True

    # -- Rendering callbacks --

    def _cb_rtx_realtime(self, msg: "Empty") -> None:
        if self._env is not None and hasattr(self._env, "_renderer"):
            renderer = self._env._renderer
            if renderer is not None:
                self._queue(renderer.enable_rtx_real_time)

    def _cb_rtx_interactive(self, msg: "Empty") -> None:
        if self._env is not None and hasattr(self._env, "_renderer"):
            renderer = self._env._renderer
            if renderer is not None:
                self._queue(renderer.enable_rtx_interactive)

    def _cb_lensflare_enable(self, msg: "Bool") -> None:
        if self._env is not None and hasattr(self._env, "_post_processing"):
            pp = self._env._post_processing
            if pp is not None:
                self._queue(pp.set_lens_flare_enabled, enabled=msg.data)

    def _cb_lensflare_scale(self, msg: "Float32") -> None:
        if self._env is not None and hasattr(self._env, "_post_processing"):
            pp = self._env._post_processing
            if pp is not None:
                self._queue(pp.set_lens_flare_scale, scale=msg.data)


# ── RobotNode ───────────────────────────────────────────────────────────────

class RobotNode(_BaseNode, _ModificationMixin):
    """ROS2 node for robot management and ground-truth publishing.

    Subscriptions:
        /OmniLRS/Robots/Spawn    (PoseStamped)  frame_id = "name:usd_path"
        /OmniLRS/Robots/Teleport (PoseStamped)  frame_id = robot_name
        /OmniLRS/Robots/Reset    (String)       data = robot_name
        /OmniLRS/Robots/ResetAll (Empty)

    Publishers:
        /tf_gt                   (TFMessage)    ground-truth transforms
        /{robot}/sun_vector      (Vector3)      sun direction per robot
        /{robot}/battery_state   (BatteryState) battery status per robot
    """

    def __init__(
        self,
        robot_manager=None,
        sensor_manager=None,
        cfg: dict = None,
    ) -> None:
        _BaseNode.__init__(self, "omnilrs_robots")
        _ModificationMixin.__init__(self)

        self._rm = robot_manager
        self._sm = sensor_manager
        self._cfg = cfg or {}

        # Publishers
        self._tf_gt_pub = None
        self._sun_vector_pubs: Dict[str, Any] = {}
        self._battery_pubs: Dict[str, Any] = {}

        if not _HAS_ROS2_MSGS:
            logger.warning("ROS2 message types not available, robot node disabled")
            return

        # TF ground-truth publisher
        self._tf_gt_pub = self.create_publisher(TFMessage, "/tf_gt", _CONTROL_QOS)

        # Robot control subscriptions
        self.create_subscription(PoseStamped, "/OmniLRS/Robots/Spawn", self._cb_spawn, _CONTROL_QOS)
        self.create_subscription(PoseStamped, "/OmniLRS/Robots/Teleport", self._cb_teleport, _CONTROL_QOS)
        self.create_subscription(String, "/OmniLRS/Robots/Reset", self._cb_reset, _CONTROL_QOS)
        self.create_subscription(Empty, "/OmniLRS/Robots/ResetAll", self._cb_reset_all, _CONTROL_QOS)

        logger.info("RobotNode: 4 subscriptions, TF publisher created")

    # -- Robot callbacks --

    def _cb_spawn(self, msg: "PoseStamped") -> None:
        """Spawn a robot. frame_id format: 'robot_name:usd_path'."""
        parts = msg.header.frame_id.split(":", 1)
        if len(parts) != 2:
            logger.error("Spawn: frame_id must be 'name:usd_path', got '%s'", msg.header.frame_id)
            return

        robot_name, usd_path = parts
        p = msg.pose.position
        o = msg.pose.orientation
        position = [p.x, p.y, p.z]
        orientation = ros_to_isaac_quat(o.x, o.y, o.z, o.w)

        if self._rm is not None:
            self._queue(
                self._rm.add_robot_at_pose,
                robot_name=robot_name,
                usd_path=usd_path,
                position=position,
                orientation=orientation,
            )
        logger.info("Spawn queued: %s at [%.1f, %.1f, %.1f]", robot_name, *position)

    def _cb_teleport(self, msg: "PoseStamped") -> None:
        """Teleport a robot. frame_id = robot_name."""
        robot_name = msg.header.frame_id
        p = msg.pose.position
        o = msg.pose.orientation
        position = [p.x, p.y, p.z]
        orientation = ros_to_isaac_quat(o.x, o.y, o.z, o.w)

        if self._rm is not None:
            self._queue(
                self._rm.teleport_robot,
                name=robot_name,
                position=position,
                orientation=orientation,
            )

    def _cb_reset(self, msg: "String") -> None:
        """Reset a single robot by name."""
        if self._rm is not None:
            self._queue(self._rm.reset_robot, name=msg.data)
        logger.info("Reset queued: %s", msg.data)

    def _cb_reset_all(self, msg: "Empty") -> None:
        """Reset all robots."""
        if self._rm is not None:
            self._queue(self._rm.reset_robots)
        logger.info("Reset all robots queued")

    def reset_all_robots(self) -> None:
        """Direct reset (called from main thread on terrain switch)."""
        if self._rm is not None:
            self._rm.reset_robots()

    # -- Ground-truth TF publishing --

    def publish_gt_tf(self) -> None:
        """Publish ground-truth transforms for all robots on /tf_gt.

        Single robot:  map -> base_link
        Multi-robot:   map -> {robot_name}/base_link
        """
        if self._tf_gt_pub is None or self._rm is None:
            return

        robots_rg = getattr(self._rm, "rigid_groups", {})
        if not robots_rg:
            return

        multi = len(robots_rg) > 1
        now = self.get_clock().now().to_msg()
        transforms = []

        for robot_name, rrg in robots_rg.items():
            try:
                position, orientation = rrg.get_pose_of_base_link()
            except Exception:
                continue

            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "map"
            t.child_frame_id = f"{robot_name}/base_link" if multi else "base_link"

            t.transform.translation.x = float(position[0])
            t.transform.translation.y = float(position[1])
            t.transform.translation.z = float(position[2])

            # Isaac format (w,x,y,z) -> ROS format
            t.transform.rotation.w = float(orientation[0])
            t.transform.rotation.x = float(orientation[1])
            t.transform.rotation.y = float(orientation[2])
            t.transform.rotation.z = float(orientation[3])

            transforms.append(t)

        if transforms:
            msg = TFMessage(transforms=transforms)
            self._tf_gt_pub.publish(msg)

    # -- Per-robot publishers (lazy creation) --

    def publish_sun_vector(self, robot_name: str, vector: np.ndarray) -> None:
        """Publish sun direction vector for a specific robot."""
        if not _HAS_ROS2_MSGS:
            return

        if robot_name not in self._sun_vector_pubs:
            self._sun_vector_pubs[robot_name] = self.create_publisher(
                Vector3, f"/{robot_name}/sun_vector", _CONTROL_QOS,
            )

        msg = Vector3(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))
        self._sun_vector_pubs[robot_name].publish(msg)

    def publish_battery_state(
        self, robot_name: str, voltage: float, percentage: float, current: float,
    ) -> None:
        """Publish battery state for a specific robot."""
        if not _HAS_ROS2_MSGS:
            return

        if robot_name not in self._battery_pubs:
            self._battery_pubs[robot_name] = self.create_publisher(
                BatteryState, f"/{robot_name}/battery_state", _CONTROL_QOS,
            )

        msg = BatteryState()
        msg.voltage = voltage
        msg.percentage = percentage / 100.0  # BatteryState expects 0..1
        msg.current = current
        msg.present = True
        self._battery_pubs[robot_name].publish(msg)
