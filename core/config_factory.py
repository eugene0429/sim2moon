from __future__ import annotations

import dataclasses
from typing import Optional, Tuple


@dataclasses.dataclass
class PhysicsConf:
    dt: float = 0.016666
    gravity: Tuple[float, float, float] = (0.0, 0.0, -1.62)
    enable_ccd: bool = True
    solver_type: str = "PGS"
    broadphase_type: str = "SAP"
    substeps: Optional[int] = None
    use_gpu_pipeline: Optional[bool] = None
    worker_thread_count: Optional[int] = None
    use_fabric: Optional[bool] = None
    enable_scene_query_support: Optional[bool] = None
    gpu_max_rigid_contact_count: Optional[int] = None
    gpu_max_rigid_patch_contact_count: Optional[int] = None
    gpu_found_lost_pairs_capacity: Optional[int] = None
    gpu_total_aggregate_pairs_capacity: Optional[int] = None
    gpu_max_soft_body_contacts: Optional[int] = None
    gpu_max_particle_contacts: Optional[int] = None
    gpu_heap_capacity: Optional[int] = None
    gpu_temp_buffer_capacity: Optional[int] = None
    gpu_max_num_partitions: Optional[int] = None
    gpu_collision_stack_size: Optional[int] = None
    enable_stabilization: Optional[bool] = None
    bounce_threshold_velocity: Optional[float] = None
    friction_offset_threshold: Optional[float] = None
    friction_correlation_distance: Optional[float] = None

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.solver_type not in ("PGS", "TGS"):
            raise ValueError(
                f"solver_type must be 'PGS' or 'TGS', got '{self.solver_type}'"
            )
        if self.broadphase_type not in ("SAP", "MBP", "GPU"):
            raise ValueError(
                f"broadphase_type must be 'SAP', 'MBP', or 'GPU', got '{self.broadphase_type}'"
            )
        # Build physics_scene_args dict with only non-None values
        self.physics_scene_args = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None:
                self.physics_scene_args[field.name] = value


@dataclasses.dataclass
class RendererConf:
    renderer: str = "RayTracedLighting"
    headless: bool = False
    width: int = 1280
    height: int = 720
    samples_per_pixel_per_frame: int = 32
    max_bounces: int = 6
    subdiv_refinement_level: int = 0

    def __post_init__(self):
        if self.renderer not in ("RayTracedLighting", "PathTracing"):
            raise ValueError(
                f"renderer must be 'RayTracedLighting' or 'PathTracing', got '{self.renderer}'"
            )


from omegaconf import DictConfig, ListConfig, OmegaConf


class ConfigFactory:
    def __init__(self):
        self._configs: dict[str, type] = {}

    def register(self, name: str, config_class: type) -> None:
        self._configs[name] = config_class

    def create(self, name: str, **kwargs):
        if name not in self._configs:
            raise KeyError(f"Config '{name}' not registered. Available: {list(self._configs.keys())}")
        return self._configs[name](**kwargs)

    def registered_names(self) -> list[str]:
        return list(self._configs.keys())


def omegaconf_to_dict(d):
    """Recursively convert OmegaConf DictConfig/ListConfig to plain Python dict/list."""
    if isinstance(d, DictConfig):
        return {k: omegaconf_to_dict(v) for k, v in d.items()}
    elif isinstance(d, ListConfig):
        return [omegaconf_to_dict(item) for item in d]
    else:
        return d


def instantiate_configs(cfg: dict, factory: ConfigFactory) -> dict:
    """Walk a config dict and instantiate any keys that match registered config names."""
    registered = factory.registered_names()
    result = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            if k in registered:
                result[k] = factory.create(k, **v)
            else:
                result[k] = instantiate_configs(v, factory)
        else:
            result[k] = v
    return result


def create_config_factory() -> ConfigFactory:
    """Create a ConfigFactory populated with all known config types."""
    from terrain.config import TerrainManagerConf, MoonYardConf
    from celestial.config import StellarEngineConf, SunConf, EarthConf
    from objects.config import RockManagerConf
    from rendering.config import RenderingConf
    from environments.lunar_yard_config import LunarYardConf

    factory = ConfigFactory()
    # Core configs
    factory.register("physics_scene", PhysicsConf)
    factory.register("renderer", RendererConf)
    # Terrain configs
    factory.register("terrain_manager", TerrainManagerConf)
    factory.register("moon_yard", MoonYardConf)
    # Celestial configs
    factory.register("stellar_engine_settings", StellarEngineConf)
    factory.register("sun_settings", SunConf)
    factory.register("earth_settings", EarthConf)
    # Objects configs
    factory.register("rocks_settings", RockManagerConf)
    # Rendering configs
    factory.register("rendering", RenderingConf)
    # Environment configs
    factory.register("lunaryard_settings", LunarYardConf)
    return factory


# Register the as_tuple resolver for OmegaConf
def _resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("as_tuple", _resolve_tuple, replace=True)
