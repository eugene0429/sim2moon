import pytest
from environments.base_environment import BaseEnvironment
from core.enums import SimulatorMode


class TestBaseEnvironmentABC:
    def test_cannot_instantiate_directly(self):
        """BaseEnvironment is abstract — instantiating without implementing methods raises TypeError."""
        with pytest.raises(TypeError):
            BaseEnvironment(stage=None, mode=SimulatorMode.ROS2)

    def test_concrete_subclass_must_implement_all(self):
        """A subclass missing any abstract method raises TypeError."""

        class IncompleteEnv(BaseEnvironment):
            def build_scene(self):
                pass
            # Missing: load, instantiate_scene, update, reset

        with pytest.raises(TypeError):
            IncompleteEnv(stage=None, mode=SimulatorMode.ROS2)

    def test_concrete_subclass_works(self):
        """A subclass implementing all abstract methods can be instantiated."""

        class ConcreteEnv(BaseEnvironment):
            def build_scene(self):
                pass

            def load(self):
                pass

            def instantiate_scene(self):
                pass

            def update(self, dt: float):
                pass

            def reset(self):
                pass

        env = ConcreteEnv(stage=None, mode=SimulatorMode.ROS2)
        assert env._mode is SimulatorMode.ROS2
        assert env._stage is None

    def test_deform_terrain_default_noop(self):
        """deform_terrain has a default no-op implementation."""

        class ConcreteEnv(BaseEnvironment):
            def build_scene(self): pass
            def load(self): pass
            def instantiate_scene(self): pass
            def update(self, dt: float): pass
            def reset(self): pass

        env = ConcreteEnv(stage=None, mode=SimulatorMode.SDG)
        env.deform_terrain()  # Should not raise

    def test_apply_terramechanics_default_noop(self):
        """apply_terramechanics has a default no-op implementation."""

        class ConcreteEnv(BaseEnvironment):
            def build_scene(self): pass
            def load(self): pass
            def instantiate_scene(self): pass
            def update(self, dt: float): pass
            def reset(self): pass

        env = ConcreteEnv(stage=None, mode=SimulatorMode.SDG)
        env.apply_terramechanics()  # Should not raise
