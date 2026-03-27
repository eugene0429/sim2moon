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
