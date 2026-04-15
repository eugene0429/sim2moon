"""
Celestial body system for lunar simulation.

Provides ephemeris-based sun/earth positioning using JPL DE421 data,
USD light management, and Earth sphere rendering.
"""

__all__ = ["StellarEngine", "SunController", "EarthController"]


def __getattr__(name):
    if name == "StellarEngine":
        from celestial.stellar_engine import StellarEngine
        return StellarEngine
    if name == "SunController":
        from celestial.sun_controller import SunController
        return SunController
    if name == "EarthController":
        from celestial.earth_controller import EarthController
        return EarthController
    raise AttributeError(f"module 'celestial' has no attribute {name!r}")
