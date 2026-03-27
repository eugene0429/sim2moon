import logging

from core.config_factory import PhysicsConf

logger = logging.getLogger(__name__)


class PhysicsManager:
    """Manages the PhysX physics scene configuration.

    Wraps Isaac Sim's PhysicsContext. Config is stored in __init__,
    actual PhysicsContext creation happens in setup().
    """

    def __init__(self, config: PhysicsConf) -> None:
        self._config = config
        self._physics_context = None

    def setup(self) -> None:
        """Create and configure the PhysicsContext.

        Requires Isaac Sim runtime to be initialized (SimulationApp must exist).
        """
        from isaacsim.core.api.physics_context.physics_context import PhysicsContext

        self._physics_context = PhysicsContext(
            sim_params=self._config.physics_scene_args,
            set_defaults=True,
        )
        if self._config.enable_ccd:
            self._physics_context.enable_ccd(True)
        if self._config.broadphase_type is not None:
            self._physics_context.set_broadphase_type(self._config.broadphase_type)
        if self._config.solver_type is not None:
            self._physics_context.set_solver_type(self._config.solver_type)

        logger.info(
            "Physics configured: dt=%.6f, gravity=%s, solver=%s, broadphase=%s, ccd=%s",
            self._config.dt,
            self._config.gravity,
            self._config.solver_type,
            self._config.broadphase_type,
            self._config.enable_ccd,
        )

    @property
    def physics_context(self):
        return self._physics_context
