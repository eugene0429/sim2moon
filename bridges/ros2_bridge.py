"""Main ROS2 simulation manager.

Orchestrates the simulation loop with ROS2 integration:
- Runs two ROS2 executors in background threads (environment + robots)
- Applies queued modifications from ROS callbacks after each physics step
- Publishes ground-truth transforms each frame

Thread-safety pattern:
    ROS callbacks queue (function, kwargs) tuples.
    The main simulation thread applies them after world.step().
    This prevents concurrent USD stage mutations.

Ported from OmniLRS src/environments_wrappers/ros2/simulation_manager_ros2.py
"""

import logging
import signal
import threading
import time
from typing import Optional

from bridges.udp_bridge import UDPBridge

logger = logging.getLogger(__name__)

_HAS_ROS2 = False


def enable_ros2_extension(bridge_name: str = "humble") -> None:
    """Enable ROS2 bridge extension in Isaac Sim.

    Enabling the extension adds Isaac Sim's bundled rclpy (Python 3.11
    compatible) to the module search path, so rclpy must be imported
    *after* this function is called.

    Args:
        bridge_name: ROS2 distribution name ('humble' or 'foxy').
    """
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
        logger.info("ROS2 bridge extension enabled (%s)", bridge_name)
    except Exception as e:
        logger.warning("Could not enable ROS2 extension: %s", e)


def _ensure_rclpy():
    """Import rclpy after the ROS2 extension has been enabled.

    Returns True if rclpy is available, False otherwise.
    """
    global _HAS_ROS2
    if _HAS_ROS2:
        return True
    try:
        import rclpy  # noqa: F401
        _HAS_ROS2 = True
        return True
    except ImportError:
        return False


class ROS2Bridge:
    """Main ROS2 simulation manager.

    Manages the simulation loop with two ROS2 executor threads:
    - EnvironmentNode: sun, terrain, rendering control
    - RobotNode: spawn, teleport, reset, ground-truth TF

    Usage:
        bridge = ROS2Bridge(cfg, environment, robot_manager, sensor_manager)
        bridge.setup()
        bridge.run()   # blocking main loop
        bridge.shutdown()
    """

    def __init__(
        self,
        cfg: dict,
        environment=None,
        robot_manager=None,
        sensor_manager=None,
        world=None,
    ) -> None:
        self._cfg = cfg
        self._environment = environment
        self._robot_manager = robot_manager
        self._sensor_manager = sensor_manager
        self._world = world

        self._running = False
        self._env_node = None
        self._robot_node = None
        self._env_executor: Optional[object] = None
        self._robot_executor: Optional[object] = None
        self._env_thread: Optional[threading.Thread] = None
        self._robot_thread: Optional[threading.Thread] = None
        self._udp_bridge: Optional[UDPBridge] = None

        # Config
        mode_cfg = cfg.get("mode", {})
        if isinstance(mode_cfg, dict):
            self._domain_id = mode_cfg.get("ROS_DOMAIN_ID", 0)
            self._bridge_name = mode_cfg.get("bridge_name", "humble")
            self._publish_gt_tf = mode_cfg.get("publish_gt_tf", True)
        else:
            self._domain_id = 0
            self._bridge_name = "humble"
            self._publish_gt_tf = True

        # UDP ICD config
        udp_cfg = cfg.get("udp_bridge", {})
        if isinstance(udp_cfg, dict):
            self._udp_enabled = udp_cfg.get("enable", True)
            self._udp_target_ip = udp_cfg.get("target_ip", "172.18.0.8")
            self._udp_target_port = udp_cfg.get("target_port", 7777)
        else:
            self._udp_enabled = True
            self._udp_target_ip = "172.18.0.8"
            self._udp_target_port = 7777

        self._physics_dt = cfg.get("physics_dt", 0.0333)
        self._enforce_realtime = cfg.get("enforce_realtime", True)

    def setup(self) -> None:
        """Initialize ROS2 context, create nodes, and start executor threads."""
        # Enable Isaac Sim ROS2 extension first — this adds the bundled
        # rclpy (Python 3.11 compatible) to the module search path.
        enable_ros2_extension(self._bridge_name)

        if not _ensure_rclpy():
            raise RuntimeError(
                "rclpy is not available. Enable the isaacsim.ros2.bridge "
                "extension or source a ROS2 workspace."
            )

        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        # Import ROS2 node classes after extension is enabled so that
        # ros2_node.py sees rclpy and uses Node (not _StubNode) as base.
        from bridges.ros2_node import EnvironmentNode, RobotNode

        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init()
            logger.info("rclpy initialized")

        # Create nodes
        self._env_node = EnvironmentNode(
            environment=self._environment,
            cfg=self._cfg,
        )
        self._robot_node = RobotNode(
            robot_manager=self._robot_manager,
            sensor_manager=self._sensor_manager,
            cfg=self._cfg,
        )
        logger.info("ROS2 nodes created")

        # Create executors and start background threads
        self._env_executor = SingleThreadedExecutor()
        self._env_executor.add_node(self._env_node)
        self._env_thread = threading.Thread(
            target=self._spin_executor,
            args=(self._env_executor, "env"),
            daemon=True,
        )

        self._robot_executor = SingleThreadedExecutor()
        self._robot_executor.add_node(self._robot_node)
        self._robot_thread = threading.Thread(
            target=self._spin_executor,
            args=(self._robot_executor, "robot"),
            daemon=True,
        )

        self._env_thread.start()
        self._robot_thread.start()
        logger.info("ROS2 executor threads started")

        # UDP ICD bridge
        if self._udp_enabled:
            self._udp_bridge = UDPBridge(
                self._udp_target_ip,
                self._udp_target_port,
                robot_manager=self._robot_manager,
            )
            self._udp_bridge.setup()
            logger.info("UDP ICD bridge enabled → %s:%d",
                         self._udp_target_ip, self._udp_target_port)

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def run(self) -> None:
        """Run the main simulation loop with ROS2 integration.

        Loop pattern (each frame):
            1. world.step(render=True)
            2. environment.update(dt)
            3. Apply environment modifications from ROS callbacks
            4. Handle terrain reset triggers
            5. Apply robot modifications from ROS callbacks
            6. Publish ground-truth transforms
            7. Terrain deformation (if enabled)
            8. Rate-limit to enforce realtime
        """
        self._running = True
        logger.info(
            "ROS2 simulation loop starting (dt=%.4f, realtime=%s)",
            self._physics_dt, self._enforce_realtime,
        )

        while self._running:
            loop_start = time.perf_counter()

            # Physics + render step
            if self._world is not None:
                self._world.step(render=True)

            if self._world is None or self._world.is_playing():
                # Update environment (stellar engine, etc.)
                if self._environment is not None:
                    self._environment.update(dt=self._physics_dt)

                # Apply queued environment modifications (sun, terrain, rendering)
                if self._env_node is not None:
                    self._env_node.apply_modifications()

                    # Handle terrain reset trigger
                    if self._env_node.trigger_reset:
                        if self._environment is not None:
                            self._environment.reset()
                        if self._robot_node is not None:
                            self._robot_node.reset_all_robots()
                        self._env_node.trigger_reset = False

                # Apply queued robot modifications (spawn, teleport)
                if self._robot_node is not None:
                    self._robot_node.apply_modifications()

                    # Publish ground-truth TF
                    if self._publish_gt_tf:
                        self._robot_node.publish_gt_tf()

                # Send rover data via UDP ICD
                if self._udp_bridge is not None:
                    self._udp_bridge.send_rover_data()

                # Terramechanics + terrain deformation
                if self._environment is not None:
                    self._environment.apply_terramechanics()
                    self._environment.deform_terrain()

            # Check executor threads are alive
            if not self._check_threads():
                logger.error("ROS2 executor thread died, shutting down")
                break

            # Realtime enforcement
            if self._enforce_realtime:
                elapsed = time.perf_counter() - loop_start
                sleep_time = self._physics_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown of ROS2 nodes and executors."""
        self._running = False
        logger.info("Shutting down ROS2 bridge")

        # Shutdown executors
        if self._env_executor is not None:
            self._env_executor.shutdown()
        if self._robot_executor is not None:
            self._robot_executor.shutdown()

        # Wait for threads
        if self._env_thread is not None and self._env_thread.is_alive():
            self._env_thread.join(timeout=2.0)
        if self._robot_thread is not None and self._robot_thread.is_alive():
            self._robot_thread.join(timeout=2.0)

        # Shutdown UDP bridge
        if self._udp_bridge is not None:
            self._udp_bridge.shutdown()
            self._udp_bridge = None

        # Destroy nodes
        if self._env_node is not None:
            self._env_node.destroy_node()
            self._env_node = None
        if self._robot_node is not None:
            self._robot_node.destroy_node()
            self._robot_node = None

        # Shutdown rclpy
        if _HAS_ROS2:
            import rclpy
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

        logger.info("ROS2 bridge shutdown complete")

    def stop(self) -> None:
        """Signal the main loop to stop."""
        self._running = False

    # -- Internal --

    def _spin_executor(self, executor, name: str) -> None:
        """Background thread target: spin a ROS2 executor."""
        try:
            executor.spin()
        except Exception as e:
            logger.error("ROS2 executor '%s' failed: %s", name, e)

    def _check_threads(self) -> bool:
        """Return True if both executor threads are alive."""
        env_ok = self._env_thread is None or self._env_thread.is_alive()
        robot_ok = self._robot_thread is None or self._robot_thread.is_alive()
        return env_ok and robot_ok

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Signal %d received, stopping ROS2 bridge", signum)
        self.stop()
