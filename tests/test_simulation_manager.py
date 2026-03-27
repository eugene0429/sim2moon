import pytest
from core.simulation_manager import SimulationManager


class TestSimulationManagerInit:
    def test_stores_config(self):
        cfg = {"name": "LunarYard", "seed": 42, "enforce_realtime": True}
        sm = SimulationManager(cfg)
        assert sm._cfg is cfg

    def test_initial_state(self):
        cfg = {"name": "LunarYard", "enforce_realtime": False}
        sm = SimulationManager(cfg)
        assert sm._running is False
        assert sm._environment is None
        assert sm._physics_manager is None
        assert sm._simulation_app is None

    def test_stop_sets_running_false(self):
        cfg = {"name": "test", "enforce_realtime": False}
        sm = SimulationManager(cfg)
        sm._running = True
        sm.stop()
        assert sm._running is False


class TestSimulationManagerSetup:
    def test_setup_requires_isaac_sim(self):
        """setup() needs Isaac Sim runtime -- skip if not available."""
        cfg = {
            "name": "LunarYard",
            "seed": 42,
            "enforce_realtime": True,
            "physics_dt": 0.0333,
            "rendering_dt": 0.0333,
            "rendering": {"renderer": {"renderer": "RayTracedLighting", "headless": True}},
            "physics_scene": {"dt": 0.016666, "gravity": [0.0, 0.0, -1.62]},
        }
        sm = SimulationManager(cfg)
        try:
            from isaacsim import SimulationApp
        except ImportError:
            pytest.skip("Isaac Sim not available")
