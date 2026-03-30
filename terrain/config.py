"""
Terrain configuration dataclasses.

All terrain-related configuration is defined here as typed dataclasses
with validation in __post_init__.
"""

import dataclasses
from typing import List, Tuple


@dataclasses.dataclass
class CraterGeneratorConf:
    """Configuration for crater shape generation from spline profiles."""

    profiles_path: str = ""
    min_xy_ratio: float = 0.85
    max_xy_ratio: float = 1.0
    resolution: float = 0.02
    pad_size: int = 500
    random_rotation: bool = True
    z_scale: float = 0.2
    seed: int = 42

    def __post_init__(self):
        assert 0 < self.min_xy_ratio <= self.max_xy_ratio <= 1.0, (
            f"xy_ratio must satisfy 0 < min ({self.min_xy_ratio}) <= max ({self.max_xy_ratio}) <= 1.0"
        )
        assert self.resolution > 0, "resolution must be positive"
        assert self.pad_size >= 0, "pad_size must be non-negative"
        assert self.z_scale > 0, "z_scale must be positive"


@dataclasses.dataclass
class CraterDistributionConf:
    """Configuration for crater placement using hardcore Poisson process."""

    x_size: float = 40.0
    y_size: float = 40.0
    densities: List[float] = dataclasses.field(default_factory=lambda: [0.025, 0.05, 0.5])
    radius: List[List[float]] = dataclasses.field(
        default_factory=lambda: [[1.5, 2.5], [0.75, 1.5], [0.25, 0.5]]
    )
    num_repeat: int = 0
    seed: int = 42

    def __post_init__(self):
        assert self.x_size > 0, "x_size must be positive"
        assert self.y_size > 0, "y_size must be positive"
        assert len(self.densities) == len(self.radius), (
            "densities and radius must have the same length"
        )
        assert self.num_repeat >= 0, "num_repeat must be non-negative"


@dataclasses.dataclass
class BaseTerrainGeneratorConf:
    """Configuration for Perlin/interpolated noise base terrain."""

    x_size: float = 40.0
    y_size: float = 40.0
    resolution: float = 0.02
    max_elevation: float = 0.5
    min_elevation: float = -0.5
    z_scale: float = 0.8
    seed: int = 42

    def __post_init__(self):
        assert self.x_size > 0, "x_size must be positive"
        assert self.y_size > 0, "y_size must be positive"
        assert self.resolution > 0, "resolution must be positive"
        assert self.max_elevation > self.min_elevation, (
            "max_elevation must be greater than min_elevation"
        )
        assert self.z_scale > 0, "z_scale must be positive"


@dataclasses.dataclass
class FootprintConf:
    """Wheel footprint dimensions (FLU coordinate system)."""

    width: float = 0.25
    height: float = 0.2
    shape: str = "rectangle"

    def __post_init__(self):
        assert self.width > 0, "width must be positive"
        assert self.height > 0, "height must be positive"
        assert self.shape in ("rectangle",), f"unsupported shape: {self.shape}"


@dataclasses.dataclass
class DeformConstrainConf:
    """Deformation constraint parameters."""

    x_deform_offset: float = 0.0
    y_deform_offset: float = 0.0
    deform_decay_ratio: float = 0.01

    def __post_init__(self):
        assert self.deform_decay_ratio > 0, "deform_decay_ratio must be positive"


@dataclasses.dataclass
class BoundaryDistributionConf:
    """Boundary distribution for wheel deformation cross-section."""

    distribution: str = "trapezoidal"
    angle_of_repose: float = 1.047  # pi/3

    def __post_init__(self):
        assert self.distribution in ("uniform", "parabolic", "trapezoidal"), (
            f"unsupported distribution: {self.distribution}"
        )
        assert self.angle_of_repose > 0, "angle_of_repose must be positive"


@dataclasses.dataclass
class DepthDistributionConf:
    """Depth distribution for wheel deformation pattern."""

    distribution: str = "sinusoidal"
    wave_frequency: float = 4.14  # num_grouser / pi

    def __post_init__(self):
        assert self.distribution in ("uniform", "sinusoidal", "trapezoidal"), (
            f"unsupported distribution: {self.distribution}"
        )
        assert self.wave_frequency > 0, "wave_frequency must be positive"


@dataclasses.dataclass
class ForceDepthRegressionConf:
    """Linear regression parameters mapping contact force to sinkage depth."""

    amplitude_slope: float = 0.00006
    amplitude_intercept: float = 0.008
    mean_slope: float = -0.00046
    mean_intercept: float = -0.0013


@dataclasses.dataclass
class DeformationEngineConf:
    """Full deformation engine configuration."""

    enable: bool = False
    delay: float = 2.0
    terrain_resolution: float = 0.02
    terrain_width: float = 40.0
    terrain_height: float = 40.0
    num_links: int = 4
    footprint: FootprintConf = dataclasses.field(default_factory=FootprintConf)
    deform_constrain: DeformConstrainConf = dataclasses.field(default_factory=DeformConstrainConf)
    boundary_distribution: BoundaryDistributionConf = dataclasses.field(
        default_factory=BoundaryDistributionConf
    )
    depth_distribution: DepthDistributionConf = dataclasses.field(
        default_factory=DepthDistributionConf
    )
    force_depth_regression: ForceDepthRegressionConf = dataclasses.field(
        default_factory=ForceDepthRegressionConf
    )

    def __post_init__(self):
        assert self.delay >= 0, "delay must be non-negative"
        assert self.terrain_resolution > 0, "terrain_resolution must be positive"
        assert self.terrain_width > 0, "terrain_width must be positive"
        assert self.terrain_height > 0, "terrain_height must be positive"
        assert self.num_links > 0, "num_links must be positive"
        # Support dict-based construction from YAML
        if isinstance(self.footprint, dict):
            self.footprint = FootprintConf(**self.footprint)
        if isinstance(self.deform_constrain, dict):
            self.deform_constrain = DeformConstrainConf(**self.deform_constrain)
        if isinstance(self.boundary_distribution, dict):
            self.boundary_distribution = BoundaryDistributionConf(**self.boundary_distribution)
        if isinstance(self.depth_distribution, dict):
            self.depth_distribution = DepthDistributionConf(**self.depth_distribution)
        if isinstance(self.force_depth_regression, dict):
            self.force_depth_regression = ForceDepthRegressionConf(**self.force_depth_regression)


@dataclasses.dataclass
class MoonYardConf:
    """Configuration for the procedural moonyard terrain pipeline."""

    crater_generator: CraterGeneratorConf = dataclasses.field(
        default_factory=CraterGeneratorConf
    )
    crater_distribution: CraterDistributionConf = dataclasses.field(
        default_factory=CraterDistributionConf
    )
    base_terrain_generator: BaseTerrainGeneratorConf = dataclasses.field(
        default_factory=BaseTerrainGeneratorConf
    )
    deformation_engine: DeformationEngineConf = dataclasses.field(
        default_factory=DeformationEngineConf
    )
    is_yard: bool = True
    is_lab: bool = False

    def __post_init__(self):
        if isinstance(self.crater_generator, dict):
            self.crater_generator = CraterGeneratorConf(**self.crater_generator)
        if isinstance(self.crater_distribution, dict):
            self.crater_distribution = CraterDistributionConf(**self.crater_distribution)
        if isinstance(self.base_terrain_generator, dict):
            self.base_terrain_generator = BaseTerrainGeneratorConf(**self.base_terrain_generator)
        if isinstance(self.deformation_engine, dict):
            self.deformation_engine = DeformationEngineConf(**self.deformation_engine)


@dataclasses.dataclass
class LandscapeConf:
    """Configuration for SouthPole DEM-based outer landscape."""

    enable: bool = False
    dem_path: str = ""
    crop_size: float = 500.0
    target_resolution: float = 5.0
    hole_margin: float = 2.0
    texture_path: str = ""
    mesh_prim_path: str = "/LunarYard/Landscape"

    def __post_init__(self):
        assert self.crop_size > 0, "crop_size must be positive"
        assert self.target_resolution > 0, "target_resolution must be positive"
        assert self.hole_margin >= 0, "hole_margin must be non-negative"


@dataclasses.dataclass
class TerrainManagerConf:
    """Top-level terrain manager configuration."""

    moon_yard: MoonYardConf = dataclasses.field(default_factory=MoonYardConf)
    root_path: str = "/LunarYard"
    texture_path: str = "/LunarYard/Looks/Basalt"
    dems_path: str = "Terrains/Lunaryard"
    mesh_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    sim_length: float = 40.0
    sim_width: float = 40.0
    resolution: float = 0.02
    augmentation: bool = False

    def __post_init__(self):
        if isinstance(self.moon_yard, dict):
            self.moon_yard = MoonYardConf(**self.moon_yard)
        assert self.sim_length > 0, "sim_length must be positive"
        assert self.sim_width > 0, "sim_width must be positive"
        assert self.resolution > 0, "resolution must be positive"
        assert len(self.mesh_position) == 3, "mesh_position must have 3 elements"
        assert len(self.mesh_orientation) == 4, "mesh_orientation must have 4 elements"
        assert len(self.mesh_scale) == 3, "mesh_scale must have 3 elements"
