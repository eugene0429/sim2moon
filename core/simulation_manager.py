import logging
import signal
import time

from core.config_factory import PhysicsConf
from core.enums import SimulatorMode
from rendering.config import RenderingConf, RendererConf
from physics.physics_manager import PhysicsManager

logger = logging.getLogger(__name__)

# Registry of environment name -> class. Populated by later phases.
_environment_registry: dict[str, type] = {}


def register_environment(name: str, env_class: type) -> None:
    """Register a concrete environment class by name."""
    _environment_registry[name] = env_class


class SimulationManager:
    """Manages the full simulation lifecycle: setup, run, shutdown.

    Lifecycle:
        sm = SimulationManager(cfg)
        sm.setup()    # Creates SimulationApp, World, environment, physics, robots
        sm.run()      # Main loop (world.step each frame)
        sm.shutdown() # Teardown
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._running = False
        self._environment = None
        self._physics_manager = None
        self._simulation_app = None
        self._world = None
        self._timeline = None
        self._robot_manager = None
        self._ros2_bridge = None
        self._mode = SimulatorMode(cfg.get("mode", "ROS2"))

    def setup(self) -> None:
        """Initialize Isaac Sim, create World, environment, physics, and robots.

        Follows the OmniLRS lifecycle:
        1. Create SimulationApp
        2. Create World (physics_dt, rendering_dt)
        3. Build environment (terrain, celestial, rocks)
        4. Configure PhysicsContext (gravity, solver, CCD)
        5. Warm-up stepping (stabilize physics)
        6. Load robots into World
        7. Stabilization stepping
        8. Setup ROS2Bridge (if ROS2 mode)
        """
        from isaacsim import SimulationApp

        # Extract renderer config for SimulationApp launch
        rendering_cfg = self._cfg.get("rendering")
        if hasattr(rendering_cfg, "renderer"):
            renderer_cfg = rendering_cfg.renderer
            if isinstance(renderer_cfg, RendererConf):
                launch_config = {
                    "renderer": renderer_cfg.renderer,
                    "headless": renderer_cfg.headless,
                }
            elif isinstance(renderer_cfg, dict):
                launch_config = renderer_cfg
            else:
                launch_config = {}
        elif isinstance(rendering_cfg, dict):
            renderer_cfg = rendering_cfg.get("renderer", {})
            launch_config = renderer_cfg if isinstance(renderer_cfg, dict) else {}
        else:
            launch_config = {}

        self._simulation_app = SimulationApp(launch_config)
        self._simulation_app.update()
        logger.info("SimulationApp created")

        # Hide the default ground grid
        try:
            import carb
            settings = carb.settings.get_settings()
            settings.set("/app/viewport/grid/enabled", False)
            settings.set("/persistent/app/viewport/displayOptions", 0)
            import omni.kit.viewport.utility as vp_util
            vp = vp_util.get_active_viewport()
            if vp:
                from omni.kit.viewport.utility import disable_selection
                vp.updates_enabled = False
                self._simulation_app.update()
                vp.updates_enabled = True
        except Exception:
            pass
        try:
            import omni.kit.commands
            omni.kit.commands.execute("ChangeSetting", path="/app/viewport/grid/enabled", value=False)
            omni.kit.commands.execute("ChangeSetting", path="/persistent/app/viewport/displayOptions", value=0)
        except Exception:
            pass

        # ── Enable extensions (OmniLRS enables these BEFORE World creation) ─
        import os
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.graph.bundle.action")
        enable_extension("omni.graph.window.action")

        # ROS2 bridge requires env vars set BEFORE process start (for
        # LD_LIBRARY_PATH to be picked up by the dynamic linker).
        # If not set, the extension fails async and corrupts physics.
        # Use the launch script: run_ros2.sh
        _ros2_ready = bool(os.environ.get("RMW_IMPLEMENTATION"))
        if _ros2_ready:
            enable_extension("isaacsim.ros2.bridge")
            enable_extension("omni.kit.viewport.actions")
            logger.info("ROS2 bridge extension enabled")
        else:
            logger.warning(
                "ROS2 env vars not set — ROS2 bridge disabled. "
                "Use: ./run_ros2.sh environment=..."
            )

        self._simulation_app.update()
        logger.info("Extensions enabled")

        # ── Create Isaac Sim World ──────────────────────────────────────
        from isaacsim.core.api import World
        import omni

        physics_dt = self._cfg.get("physics_dt", 0.0333)
        rendering_dt = self._cfg.get("rendering_dt", physics_dt)

        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
        )
        self._timeline = omni.timeline.get_timeline_interface()
        logger.info("World created (physics_dt=%.4f, rendering_dt=%.4f)",
                     physics_dt, rendering_dt)

        # ── Create environment ──────────────────────────────────────────
        env_name = self._cfg.get("name", "")
        if env_name in _environment_registry:
            from pxr import Usd

            stage = omni.usd.get_context().get_stage()
            env_class = _environment_registry[env_name]
            self._environment = env_class(stage=stage, mode=self._mode, cfg=self._cfg)
            self._environment.build_scene()
            self._environment.load()
            logger.info("Environment '%s' created and loaded", env_name)
        else:
            logger.warning(
                "No environment registered for '%s'. Running without environment. "
                "Available: %s",
                env_name,
                list(_environment_registry.keys()),
            )

        # ── Configure physics on World's built-in PhysicsContext ────────
        # World already creates a PhysicsContext; configure it directly
        # instead of creating a second one (which can cause conflicts).
        physics_cfg = self._cfg.get("physics_scene")
        if isinstance(physics_cfg, PhysicsConf):
            pc = physics_cfg
        elif isinstance(physics_cfg, dict):
            pc = PhysicsConf(**physics_cfg)
        else:
            pc = PhysicsConf(dt=physics_dt)

        phys_ctx = self._world.get_physics_context()
        phys_ctx.set_gravity(value=pc.gravity[2])  # Z-component for Z-up stage
        if pc.enable_ccd:
            phys_ctx.enable_ccd(True)
        if pc.solver_type:
            phys_ctx.set_solver_type(pc.solver_type)
        if pc.broadphase_type:
            phys_ctx.set_broadphase_type(pc.broadphase_type)
        logger.info("Physics configured: gravity=%.2f, solver=%s, ccd=%s",
                     pc.gravity[2], pc.solver_type, pc.enable_ccd)

        # ── Warm-up stepping (stabilize physics before robot spawn) ─────
        # render=False avoids flashing an incomplete scene to screen.
        logger.info("Physics warm-up stepping (100 frames)")
        for _ in range(100):
            self._world.step(render=False)
        self._world.reset()

        # ── Load robots ─────────────────────────────────────────────────
        robots_cfg = self._cfg.get("robots_settings")
        if robots_cfg is not None:
            from robots.robot_manager import RobotManager
            from robots.config import RobotManagerConf

            if isinstance(robots_cfg, RobotManagerConf):
                self._robot_manager = RobotManager(robots_cfg)
            elif isinstance(robots_cfg, dict):
                self._robot_manager = RobotManager(RobotManagerConf(**robots_cfg))
            else:
                logger.warning("Unexpected robots_settings type: %s", type(robots_cfg))

            if self._robot_manager is not None:
                # Apply terrain mesh_position offset so robots spawn
                # at the correct world-space height
                tm_cfg = self._cfg.get("terrain_manager", {})
                if isinstance(tm_cfg, dict):
                    mesh_pos = tm_cfg.get("mesh_position", [0, 0, 0])
                else:
                    mesh_pos = getattr(tm_cfg, "mesh_position", (0, 0, 0))
                self._robot_manager.set_mesh_position(mesh_pos)


                self._robot_manager.preload_robot(world=self._world)
                logger.info("Robots loaded")

                # Register robot manager with environment for terramechanics
                if self._environment is not None and hasattr(self._environment, "set_robot_manager"):
                    self._environment.set_robot_manager(self._robot_manager)

        # ── Post-renderer initialization ────────────────────────────────
        if self._environment is not None:
            self._environment.instantiate_scene()

        # ── Stabilization stepping (settle robots on terrain) ───────────
        # render=False to avoid flashing incomplete scene, then a few
        # render=True frames to prime the GPU texture cache.
        logger.info("Stabilization stepping (100 frames)")
        for _ in range(95):
            self._world.step(render=False)
        for _ in range(5):
            self._world.step(render=True)
        self._world.reset()

        # ── Setup ROS2Bridge if in ROS2 mode and ROS2 env is available ───
        if self._mode == SimulatorMode.ROS2 and _ros2_ready:
            from bridges.ros2_bridge import ROS2Bridge

            self._ros2_bridge = ROS2Bridge(
                cfg=self._cfg,
                environment=self._environment,
                robot_manager=self._robot_manager,
                sensor_manager=None,
                world=self._world,
            )
            try:
                self._ros2_bridge.setup()
                logger.info("ROS2Bridge initialized")
            except RuntimeError as e:
                logger.warning("ROS2Bridge setup failed: %s", e)
                self._ros2_bridge = None
        elif self._mode == SimulatorMode.ROS2:
            logger.info(
                "ROS2 mode but env vars not set — running without ROS2Bridge. "
                "Use: ./run_ros2.sh environment=..."
            )

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def run(self) -> None:
        """Run the main simulation loop.

        In ROS2 mode, delegates to ROS2Bridge which handles the full loop
        with ROS2 executor threads, robot services, TF publishing, and UDP.
        """
        # Matches OmniLRS: timeline.play() then world.step loop.
        # The first step with current_time_step_index==0 triggers
        # world.reset() which does stop→play internally, properly
        # initializing the physics state machine.
        self._timeline.play()

        if self._ros2_bridge is not None:
            logger.info("Delegating run loop to ROS2Bridge")
            self._ros2_bridge.run()
            return

        # ── Fallback loop (no ROS2Bridge) ───────────────────────────────
        self._running = True
        enforce_realtime = self._cfg.get("enforce_realtime", False)
        physics_dt = self._cfg.get("physics_dt", 0.0333)

        # Create standalone UDP bridge for fallback loop
        _udp_bridge = None
        udp_cfg = self._cfg.get("udp_bridge", {})
        if isinstance(udp_cfg, dict) and udp_cfg.get("enable", False):
            from bridges.udp_bridge import UDPBridge
            _udp_bridge = UDPBridge(
                udp_cfg.get("target_ip", "127.0.0.1"),
                udp_cfg.get("target_port", 7777),
                robot_manager=self._robot_manager,
            )
            _udp_bridge.setup()
            logger.info("UDP bridge enabled (fallback) → %s:%d",
                         udp_cfg.get("target_ip"), udp_cfg.get("target_port"))

        logger.info("Simulation loop starting (realtime=%s, dt=%.4f)", enforce_realtime, physics_dt)

        while self._running:
            loop_start = time.perf_counter()

            # Physics + render step
            self._world.step(render=True)

            if self._world.is_playing():
                # First frame after timeline.play(): reset to initialize
                # physics handles (OmniLRS pattern)
                if self._world.current_time_step_index == 0:
                    self._world.reset()

                if self._environment is not None:
                    self._environment.update(dt=physics_dt)
                    self._environment.deform_terrain()
                    self._environment.apply_terramechanics()

                # Send rover telemetry via UDP
                if _udp_bridge is not None:
                    _udp_bridge.send_rover_data()

            if enforce_realtime:
                elapsed = time.perf_counter() - loop_start
                sleep_time = physics_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def shutdown(self) -> None:
        """Clean shutdown of the simulation."""
        self._running = False

        if self._ros2_bridge is not None:
            self._ros2_bridge.shutdown()
            self._ros2_bridge = None

        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            except Exception as e:
                logger.warning("Error during SimulationApp close: %s", e)
            self._simulation_app = None

        logger.info("Simulation shutdown complete")

    def stop(self) -> None:
        """Stop the simulation loop (can be called from signal handler)."""
        self._running = False
        if self._ros2_bridge is not None:
            self._ros2_bridge.stop()

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Signal %d received, stopping simulation", signum)
        self.stop()
