import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from core.config_factory import (
    omegaconf_to_dict,
    instantiate_configs,
    create_config_factory,
    PhysicsConf,
)


class TestMainHelpers:
    def test_full_config_pipeline(self):
        """Test the complete YAML -> dict -> dataclass pipeline."""
        raw = OmegaConf.create({
            "name": "LunarYard",
            "seed": 42,
            "physics_scene": {
                "dt": 0.016666,
                "gravity": [0.0, 0.0, -1.62],
                "enable_ccd": True,
                "solver_type": "PGS",
                "broadphase_type": "SAP",
            },
            "rendering": {
                "renderer": {
                    "renderer": "RayTracedLighting",
                    "headless": False,
                    "width": 1280,
                    "height": 720,
                    "samples_per_pixel_per_frame": 32,
                    "max_bounces": 6,
                    "subdiv_refinement_level": 0,
                },
            },
        })
        cfg_dict = omegaconf_to_dict(raw)
        factory = create_config_factory()
        cfg_dict = instantiate_configs(cfg_dict, factory)

        assert isinstance(cfg_dict["physics_scene"], PhysicsConf)
        assert cfg_dict["physics_scene"].dt == 0.016666
        assert cfg_dict["physics_scene"].gravity == [0.0, 0.0, -1.62]
        assert cfg_dict["name"] == "LunarYard"
