"""
Post-processing effects for lunar simulation.

Provides class-based management of lens flares, chromatic aberration,
and motion blur via Omniverse carb.settings.

Reference: OmniLRS src/environments/rendering.py (post-processing sections)
"""

import logging
from typing import Optional, Tuple

from rendering.config import (
    ChromaticAberrationConf,
    FlaresConf,
    MotionBlurConf,
    RenderingConf,
)

logger = logging.getLogger(__name__)

try:
    import carb
    _HAS_CARB = True
except ImportError:
    _HAS_CARB = False


def _set(path: str, value) -> None:
    """Set a carb setting value."""
    if _HAS_CARB:
        carb.settings.get_settings().set(path, value)


class LensFlareEffect:
    """Controls lens flare post-processing."""

    def apply(self, cfg: FlaresConf) -> None:
        _set("/rtx/post/lensFlares/enabled", cfg.enable)
        if cfg.enable:
            _set("/rtx/post/lensFlares/flareScale", cfg.scale)
            _set("/rtx/post/lensFlares/blades", int(cfg.blades))
            _set("/rtx/post/lensFlares/apertureRotation", cfg.aperture_rotation)
            _set("/rtx/post/lensFlares/sensorDiagonal", cfg.sensor_diagonal)
            _set("/rtx/post/lensFlares/sensorAspectRatio", cfg.sensor_aspect_ratio)
            _set("/rtx/post/lensFlares/fNumber", cfg.fstop)
            _set("/rtx/post/lensFlares/focalLength", cfg.focal_length)
        logger.info("LensFlare: %s", "enabled" if cfg.enable else "disabled")

    def set_enabled(self, enable: bool) -> None:
        _set("/rtx/post/lensFlares/enabled", enable)

    def set_scale(self, value: float) -> None:
        _set("/rtx/post/lensFlares/flareScale", value)

    def set_blades(self, value: int) -> None:
        _set("/rtx/post/lensFlares/blades", int(value))

    def set_fstop(self, value: float) -> None:
        _set("/rtx/post/lensFlares/fNumber", value)

    def set_focal_length(self, value: float) -> None:
        _set("/rtx/post/lensFlares/focalLength", value)


class ChromaticAberrationEffect:
    """Controls chromatic aberration post-processing."""

    def apply(self, cfg: ChromaticAberrationConf) -> None:
        _set("/rtx/post/chromaticAberration/enabled", cfg.enable)
        if cfg.enable:
            _set("/rtx/post/chromaticAberration/strengthR", cfg.strength[0])
            _set("/rtx/post/chromaticAberration/strengthG", cfg.strength[1])
            _set("/rtx/post/chromaticAberration/strengthB", cfg.strength[2])
            _set("/rtx/post/chromaticAberration/modelR", cfg.model[0])
            _set("/rtx/post/chromaticAberration/modelG", cfg.model[1])
            _set("/rtx/post/chromaticAberration/modelB", cfg.model[2])
            _set("/rtx/post/chromaticAberration/enableLanczos", cfg.enable_lanczos)
        logger.info("ChromaticAberration: %s", "enabled" if cfg.enable else "disabled")

    def set_enabled(self, enable: bool) -> None:
        _set("/rtx/post/chromaticAberration/enabled", enable)

    def set_strength(self, r: float, g: float, b: float) -> None:
        _set("/rtx/post/chromaticAberration/strengthR", r)
        _set("/rtx/post/chromaticAberration/strengthG", g)
        _set("/rtx/post/chromaticAberration/strengthB", b)


class MotionBlurEffect:
    """Controls motion blur post-processing."""

    def apply(self, cfg: MotionBlurConf) -> None:
        _set("/rtx/post/motionblur/enabled", cfg.enable)
        if cfg.enable:
            _set(
                "/rtx/post/motionblur/maxBlurDiameterFraction",
                cfg.max_blur_diameter_fraction,
            )
            _set("/rtx/post/motionblur/exposureFraction", cfg.exposure_fraction)
            _set("/rtx/post/motionblur/numSamples", cfg.num_samples)
        logger.info("MotionBlur: %s", "enabled" if cfg.enable else "disabled")

    def set_enabled(self, enable: bool) -> None:
        _set("/rtx/post/motionblur/enabled", enable)

    def set_diameter_fraction(self, value: float) -> None:
        _set("/rtx/post/motionblur/maxBlurDiameterFraction", value)

    def set_num_samples(self, value: int) -> None:
        _set("/rtx/post/motionblur/numSamples", value)


class PostProcessingManager:
    """
    Manages all post-processing effects as a unified interface.

    Interface contract:
        apply(cfg) -> None
        lens_flares -> LensFlareEffect
        chromatic_aberration -> ChromaticAberrationEffect
        motion_blur -> MotionBlurEffect
    """

    def __init__(self) -> None:
        self.lens_flares = LensFlareEffect()
        self.chromatic_aberration = ChromaticAberrationEffect()
        self.motion_blur = MotionBlurEffect()

    def apply(self, cfg: RenderingConf) -> None:
        """
        Apply all post-processing effects from a rendering config.

        Args:
            cfg: Full rendering configuration containing sub-configs
                 for flares, chromatic aberration, and motion blur.
        """
        self.lens_flares.apply(cfg.lens_flares)
        self.chromatic_aberration.apply(cfg.chromatic_aberration)
        self.motion_blur.apply(cfg.motion_blur)
        logger.info("PostProcessingManager: all effects applied")

    def disable_all(self) -> None:
        """Disable all post-processing effects."""
        self.lens_flares.set_enabled(False)
        self.chromatic_aberration.set_enabled(False)
        self.motion_blur.set_enabled(False)
