"""
Rendering pipeline for lunar simulation.

Provides RTX renderer configuration and post-processing effects
(lens flares, chromatic aberration, motion blur).
"""

__all__ = ["Renderer", "PostProcessingManager"]


def __getattr__(name):
    if name == "Renderer":
        from rendering.renderer import Renderer
        return Renderer
    if name == "PostProcessingManager":
        from rendering.post_processing import PostProcessingManager
        return PostProcessingManager
    raise AttributeError(f"module 'rendering' has no attribute {name!r}")
