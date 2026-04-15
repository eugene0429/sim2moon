"""
Object placement system for lunar simulation.

Provides rock instancing with power-law distributions,
crater-aware placement, and static asset management.
"""

__all__ = ["RockManager", "RockDistributor", "StaticAssetManager"]


def __getattr__(name):
    if name == "RockManager":
        from objects.rock_manager import RockManager
        return RockManager
    if name == "RockDistributor":
        from objects.rock_distribution import RockDistributor
        return RockDistributor
    if name == "StaticAssetManager":
        from objects.static_assets import StaticAssetManager
        return StaticAssetManager
    raise AttributeError(f"module 'objects' has no attribute {name!r}")
