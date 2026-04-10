# tests/conftest.py
"""
Pytest configuration for the lunar sim test suite.

Many terrain sub-modules pull in heavy Isaac Sim / OpenCV dependencies that
are not available in a plain Python environment (CI, unit-test runners, etc.).
This conftest pre-populates sys.modules with lightweight stubs so that
pure-numpy modules such as terrain.static_transition can be imported without
triggering the full dependency chain.
"""

import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    """Return an empty module stub registered under the given dotted name."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# Stub out cv2 before anything else tries to import it.
if "cv2" not in sys.modules:
    _make_stub("cv2")

# Stub the heavy terrain sub-packages so that terrain/__init__.py does not
# attempt to import TerrainManager (which would cascade into cv2 / Isaac Sim).
for _name in (
    "terrain.terrain_manager",
    "terrain.procedural",
    "terrain.procedural.base_terrain",
    "terrain.procedural.crater_generator",
    "terrain.procedural.realistic_crater_generator",
    "terrain.procedural.crater_distribution",
    "terrain.materials",
    "terrain.mesh",
    "terrain.deformation",
):
    if _name not in sys.modules:
        _make_stub(_name)

# Provide a minimal TerrainManager stub so the terrain package __init__ import
# succeeds without executing real code.
_tm_stub = _make_stub("terrain.terrain_manager")
_tm_stub.TerrainManager = type("TerrainManager", (), {})
