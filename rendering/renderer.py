"""
RTX renderer configuration for lunar simulation.

Provides a class-based API for switching between RayTracedLighting
and PathTracing modes, and configuring rendering quality parameters.

Reference: OmniLRS src/environments/rendering.py (renderer control section)
"""

import logging
from typing import Optional

from rendering.config import RendererConf

logger = logging.getLogger(__name__)

try:
    import carb
    import omni.kit.actions.core
    _HAS_OMNI = True
except ImportError:
    _HAS_OMNI = False


class Renderer:
    """
    Manages the Omniverse RTX renderer configuration.

    Switches between RayTracedLighting and PathTracing modes and
    applies quality settings (samples, bounces, etc.) via carb.settings.

    Interface contract:
        apply(cfg) -> None
        set_mode(mode) -> None
        set_samples_per_pixel(n) -> None
        set_max_bounces(n) -> None
    """

    def __init__(self, cfg: Optional[RendererConf] = None) -> None:
        self._cfg = cfg or RendererConf()

    def apply(self, cfg: Optional[RendererConf] = None) -> None:
        """
        Apply all renderer settings from config.

        Args:
            cfg: Renderer configuration. Uses stored config if None.
        """
        if cfg is not None:
            self._cfg = cfg

        self.set_mode(self._cfg.renderer)
        self.set_samples_per_pixel(self._cfg.samples_per_pixel_per_frame)
        self.set_max_bounces(self._cfg.max_bounces)
        self.set_max_specular_transmission_bounces(
            self._cfg.max_specular_transmission_bounces
        )
        self.set_max_volume_bounces(self._cfg.max_volume_bounces)
        self.set_subdiv_refinement_level(self._cfg.subdiv_refinement_level)

        logger.info(
            "Renderer: applied %s (spp=%d, bounces=%d)",
            self._cfg.renderer,
            self._cfg.samples_per_pixel_per_frame,
            self._cfg.max_bounces,
        )

    def set_mode(self, mode: str) -> None:
        """
        Switch renderer mode.

        Args:
            mode: 'RayTracedLighting' or 'PathTracing'.
        """
        if not _HAS_OMNI:
            logger.warning("Renderer: Omniverse not available, skipping mode switch")
            return

        action_registry = omni.kit.actions.core.get_action_registry()
        if mode == "RayTracedLighting":
            action = action_registry.get_action(
                "omni.kit.viewport.actions", "set_renderer_rtx_realtime"
            )
        elif mode == "PathTracing":
            action = action_registry.get_action(
                "omni.kit.viewport.actions", "set_renderer_rtx_pathtracing"
            )
        else:
            raise ValueError(f"Unknown renderer mode: {mode}")

        action.execute()
        logger.info("Renderer: switched to %s", mode)

    def set_samples_per_pixel(self, value: int) -> None:
        """Set samples per pixel per frame."""
        self._set_setting("/rtx/pathtracing/spp", value)
        self._set_setting(
            "/rtx/directLighting/sampledLighting/autoNumberOfRays", value
        )

    def set_max_bounces(self, value: int) -> None:
        """Set maximum ray bounces."""
        self._set_setting("/rtx/pathtracing/maxBounces", value)

    def set_max_specular_transmission_bounces(self, value: int) -> None:
        """Set maximum specular transmission bounces."""
        self._set_setting(
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces", value
        )

    def set_max_volume_bounces(self, value: int) -> None:
        """Set maximum volume bounces."""
        self._set_setting("/rtx/pathtracing/maxVolumeBounces", value)

    def set_subdiv_refinement_level(self, value: int) -> None:
        """Set subdivision refinement level."""
        self._set_setting("/rtx/hydra/subdivision/refinementLevel", value)

    @property
    def current_mode(self) -> str:
        return self._cfg.renderer

    @staticmethod
    def _set_setting(path: str, value) -> None:
        """Set a carb setting value."""
        if not _HAS_OMNI:
            return
        settings = carb.settings.get_settings()
        settings.set(path, value)
