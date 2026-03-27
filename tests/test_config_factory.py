import pytest
from core.config_factory import PhysicsConf, RendererConf


class TestPhysicsConf:
    def test_defaults(self):
        conf = PhysicsConf()
        assert conf.dt == 0.016666
        assert conf.gravity == (0.0, 0.0, -1.62)
        assert conf.enable_ccd is True
        assert conf.solver_type == "PGS"
        assert conf.broadphase_type == "SAP"

    def test_custom_values(self):
        conf = PhysicsConf(dt=0.01, gravity=(0.0, 0.0, -9.81), solver_type="TGS")
        assert conf.dt == 0.01
        assert conf.gravity == (0.0, 0.0, -9.81)
        assert conf.solver_type == "TGS"

    def test_invalid_solver_type(self):
        with pytest.raises(ValueError, match="solver_type"):
            PhysicsConf(solver_type="INVALID")

    def test_invalid_broadphase_type(self):
        with pytest.raises(ValueError, match="broadphase_type"):
            PhysicsConf(broadphase_type="INVALID")

    def test_invalid_dt(self):
        with pytest.raises(ValueError, match="dt"):
            PhysicsConf(dt=-1.0)

    def test_physics_scene_args_dict(self):
        conf = PhysicsConf(dt=0.01, enable_ccd=False, substeps=None)
        args = conf.physics_scene_args
        assert args["dt"] == 0.01
        assert args["enable_ccd"] is False
        assert "substeps" not in args


class TestRendererConf:
    def test_defaults(self):
        conf = RendererConf()
        assert conf.renderer == "RayTracedLighting"
        assert conf.headless is False
        assert conf.width == 1280
        assert conf.height == 720

    def test_invalid_renderer(self):
        with pytest.raises(ValueError, match="renderer"):
            RendererConf(renderer="OpenGL")

    def test_path_tracing(self):
        conf = RendererConf(renderer="PathTracing")
        assert conf.renderer == "PathTracing"


from omegaconf import OmegaConf, DictConfig
from core.config_factory import (
    ConfigFactory,
    create_config_factory,
    omegaconf_to_dict,
    instantiate_configs,
)


class TestConfigFactory:
    def test_register_and_create(self):
        factory = ConfigFactory()
        factory.register("physics_scene", PhysicsConf)
        conf = factory.create("physics_scene", dt=0.01)
        assert isinstance(conf, PhysicsConf)
        assert conf.dt == 0.01

    def test_create_unregistered_raises(self):
        factory = ConfigFactory()
        with pytest.raises(KeyError, match="unknown_config"):
            factory.create("unknown_config")

    def test_registered_names(self):
        factory = ConfigFactory()
        factory.register("physics_scene", PhysicsConf)
        factory.register("renderer", RendererConf)
        names = factory.registered_names()
        assert "physics_scene" in names
        assert "renderer" in names

    def test_create_config_factory_has_defaults(self):
        factory = create_config_factory()
        names = factory.registered_names()
        assert "physics_scene" in names
        assert "renderer" in names


class TestOmegaconfToDict:
    def test_simple_dict(self):
        cfg = OmegaConf.create({"a": 1, "b": "hello"})
        result = omegaconf_to_dict(cfg)
        assert result == {"a": 1, "b": "hello"}

    def test_nested_dict(self):
        cfg = OmegaConf.create({"outer": {"inner": 42}})
        result = omegaconf_to_dict(cfg)
        assert result == {"outer": {"inner": 42}}

    def test_list_values(self):
        cfg = OmegaConf.create({"items": [1, 2, 3]})
        result = omegaconf_to_dict(cfg)
        assert result == {"items": [1, 2, 3]}

    def test_scalar_passthrough(self):
        assert omegaconf_to_dict(42) == 42
        assert omegaconf_to_dict("hello") == "hello"


class TestInstantiateConfigs:
    def test_instantiates_registered_key(self):
        factory = ConfigFactory()
        factory.register("physics_scene", PhysicsConf)
        cfg = {
            "physics_scene": {"dt": 0.01, "gravity": [0.0, 0.0, -1.62]},
            "name": "test",
        }
        result = instantiate_configs(cfg, factory)
        assert isinstance(result["physics_scene"], PhysicsConf)
        assert result["physics_scene"].dt == 0.01
        assert result["name"] == "test"

    def test_nested_instantiation(self):
        factory = ConfigFactory()
        factory.register("renderer", RendererConf)
        cfg = {
            "rendering": {
                "renderer": {"renderer": "PathTracing", "headless": True},
            }
        }
        result = instantiate_configs(cfg, factory)
        assert isinstance(result["rendering"]["renderer"], RendererConf)
        assert result["rendering"]["renderer"].renderer == "PathTracing"

    def test_leaves_unknown_keys_as_dicts(self):
        factory = ConfigFactory()
        cfg = {"unknown": {"a": 1, "b": 2}}
        result = instantiate_configs(cfg, factory)
        assert result["unknown"] == {"a": 1, "b": 2}
