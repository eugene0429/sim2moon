"""
Configuration dataclasses for the rendering pipeline.

Covers renderer settings, lens flares, chromatic aberration, and motion blur.
"""

import dataclasses
from typing import Tuple


@dataclasses.dataclass
class RendererConf:
    """RTX renderer configuration."""

    renderer: str = "RayTracedLighting"
    headless: bool = False
    samples_per_pixel_per_frame: int = 32
    max_bounces: int = 6
    max_specular_transmission_bounces: int = 6
    max_volume_bounces: int = 4
    subdiv_refinement_level: int = 0

    def __post_init__(self):
        assert self.renderer in ("RayTracedLighting", "PathTracing"), (
            f"renderer must be 'RayTracedLighting' or 'PathTracing', got '{self.renderer}'"
        )
        assert self.samples_per_pixel_per_frame > 0, "samples_per_pixel_per_frame must be positive"
        assert self.max_bounces > 0, "max_bounces must be positive"
        assert self.max_specular_transmission_bounces > 0
        assert self.max_volume_bounces > 0
        assert self.subdiv_refinement_level >= 0


@dataclasses.dataclass
class FlaresConf:
    """Lens flare post-processing configuration."""

    enable: bool = False
    scale: float = 0.5
    blades: int = 9
    aperture_rotation: float = 0.0
    sensor_diagonal: float = 28.0
    sensor_aspect_ratio: float = 1.5
    fstop: float = 2.8
    focal_length: float = 12.0

    def __post_init__(self):
        if self.enable:
            assert self.scale > 0, "scale must be positive"
            assert self.blades > 0, "blades must be positive"
            assert 0.0 <= self.aperture_rotation <= 360.0
            assert self.sensor_diagonal > 0
            assert self.sensor_aspect_ratio > 0
            assert self.fstop > 0
            assert self.focal_length > 0


@dataclasses.dataclass
class MotionBlurConf:
    """Motion blur post-processing configuration."""

    enable: bool = False
    max_blur_diameter_fraction: float = 0.02
    exposure_fraction: float = 1.0
    num_samples: int = 8

    def __post_init__(self):
        if self.enable:
            assert self.max_blur_diameter_fraction > 0
            assert self.exposure_fraction > 0
            assert self.num_samples > 0


@dataclasses.dataclass
class ChromaticAberrationConf:
    """Chromatic aberration post-processing configuration."""

    enable: bool = False
    strength: Tuple[float, float, float] = (-0.055, -0.075, 0.015)
    model: Tuple[str, str, str] = ("Radial", "Radial", "Radial")
    enable_lanczos: bool = False

    def __post_init__(self):
        assert len(self.strength) == 3, "strength must be a 3-tuple"
        assert len(self.model) == 3, "model must be a 3-tuple"
        assert all(m in ("Radial", "Barrel") for m in self.model), (
            "model values must be 'Radial' or 'Barrel'"
        )
        assert all(-1.0 <= s <= 1.0 for s in self.strength), (
            "strength values must be in [-1, 1]"
        )
        # Support list-based construction from YAML
        if isinstance(self.strength, list):
            self.strength = tuple(self.strength)
        if isinstance(self.model, list):
            self.model = tuple(self.model)


@dataclasses.dataclass
class RenderingConf:
    """Top-level rendering configuration combining all sub-configs."""

    renderer: RendererConf = dataclasses.field(default_factory=RendererConf)
    lens_flares: FlaresConf = dataclasses.field(default_factory=FlaresConf)
    motion_blur: MotionBlurConf = dataclasses.field(default_factory=MotionBlurConf)
    chromatic_aberration: ChromaticAberrationConf = dataclasses.field(
        default_factory=ChromaticAberrationConf
    )

    def __post_init__(self):
        if isinstance(self.renderer, dict):
            self.renderer = RendererConf(**self.renderer)
        if isinstance(self.lens_flares, dict):
            self.lens_flares = FlaresConf(**self.lens_flares)
        if isinstance(self.motion_blur, dict):
            self.motion_blur = MotionBlurConf(**self.motion_blur)
        if isinstance(self.chromatic_aberration, dict):
            self.chromatic_aberration = ChromaticAberrationConf(**self.chromatic_aberration)
