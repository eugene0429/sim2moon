"""
Configuration dataclasses for the objects (rock/asset) system.

Covers rock distribution parameters and static asset placement.
"""

import dataclasses
from typing import Dict, List, Optional


@dataclasses.dataclass
class RealisticDistributionConf:
    """Configuration for realistic (power-law + crater-aware) rock distribution."""

    method: str = "background_scatter"
    resolution: float = 0.02
    min_gap: float = 0.02
    native_rock_size: float = 0.3

    # Power-law size distribution parameters
    d_min: float = 0.3
    d_max: float = 2.0
    alpha: float = 2.5

    # Crater ejecta parameters
    rocks_per_crater: int = 15
    rim_inner_ratio: float = 0.8
    rim_outer_ratio: float = 2.5
    decay_power: float = 2.0
    min_crater_radius: float = 1.0

    # Crater wall debris parameters
    inner_radius_ratio: float = 0.1
    outer_radius_ratio: float = 0.9
    wall_bias: float = 2.0

    # Background scatter parameters
    density: float = 0.5

    # Clustered scatter parameters
    lambda_parent: float = 0.3
    lambda_daughter: int = 5
    sigma: float = 8.0

    def __post_init__(self):
        assert self.method in (
            "crater_ejecta", "crater_wall_debris",
            "background_scatter", "clustered_scatter",
        ), f"unsupported method: {self.method}"
        assert self.resolution >= 0
        assert self.d_min > 0
        assert self.d_max >= self.d_min
        assert self.alpha > 1.0, "alpha must be > 1 for convergent power-law"


@dataclasses.dataclass
class RockGroupConf:
    """Configuration for a single rock group (collection + distribution)."""

    seed: int = 42
    collections: List[str] = dataclasses.field(default_factory=lambda: ["apollo_rocks"])
    use_point_instancer: bool = True
    semantic_class: str = "rock"
    realistic_distribution: RealisticDistributionConf = dataclasses.field(
        default_factory=RealisticDistributionConf
    )

    def __post_init__(self):
        if isinstance(self.realistic_distribution, dict):
            self.realistic_distribution = RealisticDistributionConf(**self.realistic_distribution)


@dataclasses.dataclass
class RockManagerConf:
    """Top-level rock manager configuration."""

    enable: bool = True
    instancers_path: str = "/Lunaryard/Rocks"
    rocks_settings: Dict[str, RockGroupConf] = dataclasses.field(default_factory=dict)
    assets_root: str = "assets/USD_Assets/rocks"

    def __post_init__(self):
        resolved = {}
        for name, group in self.rocks_settings.items():
            if isinstance(group, dict):
                resolved[name] = RockGroupConf(**group)
            else:
                resolved[name] = group
        self.rocks_settings = resolved


@dataclasses.dataclass
class StaticAssetConf:
    """Configuration for a single static asset placement."""

    usd_path: str = ""
    prim_path: str = ""
    position: tuple = (0.0, 0.0, 0.0)
    orientation: tuple = (0.0, 0.0, 0.0, 1.0)
    scale: tuple = (1.0, 1.0, 1.0)
    semantic_class: str = "prop"
