import pytest
from physics.physics_manager import PhysicsManager
from core.config_factory import PhysicsConf


class TestPhysicsManager:
    def test_init_stores_config(self):
        conf = PhysicsConf(dt=0.01)
        pm = PhysicsManager(conf)
        assert pm._config is conf

    def test_init_with_default_config(self):
        conf = PhysicsConf()
        pm = PhysicsManager(conf)
        assert pm._config.gravity == (0.0, 0.0, -1.62)
        assert pm._config.enable_ccd is True

    def test_setup_requires_isaac_sim(self):
        """setup() calls Isaac Sim APIs — skip if not available."""
        conf = PhysicsConf()
        pm = PhysicsManager(conf)
        try:
            from isaacsim.core.api.physics_context.physics_context import PhysicsContext
        except ImportError:
            pytest.skip("Isaac Sim not available")
