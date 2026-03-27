import logging
import signal
import time

from core.config_factory import PhysicsConf, RendererConf
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
        sm.setup()    # Creates SimulationApp, environment, physics
        sm.run()      # Main loop
        sm.shutdown() # Teardown
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._running = False
        self._environment = None
        self._physics_manager = None
        self._simulation_app = None

    def setup(self) -> None:
        """Initialize Isaac Sim, create environment and physics."""
        from isaacsim import SimulationApp

        # Extract renderer config for SimulationApp launch
        renderer_cfg = self._cfg.get("rendering", {}).get("renderer")
        if isinstance(renderer_cfg, RendererConf):
            launch_config = {
                "renderer": renderer_cfg.renderer,
                "headless": renderer_cfg.headless,
                "width": renderer_cfg.width,
                "height": renderer_cfg.height,
            }
        elif isinstance(renderer_cfg, dict):
            launch_config = renderer_cfg
        else:
            launch_config = {}

        self._simulation_app = SimulationApp(launch_config)
        self._simulation_app.update()
        logger.info("SimulationApp created")

        # Create environment if registered
        env_name = self._cfg.get("name", "")
        if env_name in _environment_registry:
            import omni
            from pxr import Usd

            from core.enums import SimulatorMode

            stage = omni.usd.get_context().get_stage()
            mode_str = self._cfg.get("mode", "ROS2")
            mode = SimulatorMode(mode_str)
            env_class = _environment_registry[env_name]
            self._environment = env_class(stage=stage, mode=mode, cfg=self._cfg)
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

        # Setup physics
        physics_cfg = self._cfg.get("physics_scene")
        if isinstance(physics_cfg, PhysicsConf):
            self._physics_manager = PhysicsManager(physics_cfg)
            self._physics_manager.setup()
            logger.info("Physics configured")
        elif isinstance(physics_cfg, dict):
            self._physics_manager = PhysicsManager(PhysicsConf(**physics_cfg))
            self._physics_manager.setup()
            logger.info("Physics configured from dict")
        else:
            logger.warning("No physics_scene config found, skipping physics setup")

        # Post-renderer initialization
        if self._environment is not None:
            self._environment.instantiate_scene()

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def run(self) -> None:
        """Run the main simulation loop."""
        self._running = True
        enforce_realtime = self._cfg.get("enforce_realtime", False)
        physics_dt = self._cfg.get("physics_dt", 0.0333)

        logger.info("Simulation loop starting (realtime=%s, dt=%.4f)", enforce_realtime, physics_dt)

        while self._running:
            loop_start = time.perf_counter()

            if self._environment is not None:
                self._environment.update(dt=physics_dt)
                self._environment.deform_terrain()
                self._environment.apply_terramechanics()

            if self._simulation_app is not None:
                self._simulation_app.update()

            if enforce_realtime:
                elapsed = time.perf_counter() - loop_start
                sleep_time = physics_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def shutdown(self) -> None:
        """Clean shutdown of the simulation."""
        self._running = False

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

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Signal %d received, stopping simulation", signum)
        self.stop()
