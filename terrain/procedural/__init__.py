"""Procedural terrain generation: base terrain, crater placement, and crater shapes."""

from terrain.procedural.base_terrain import BaseTerrainGenerator
from terrain.procedural.crater_distribution import CraterDistributor
from terrain.procedural.crater_generator import CraterGenerator, CraterData
from terrain.procedural.moonyard_generator import MoonyardGenerator
from terrain.procedural.noise import perlin_1d, perlin_2d
from terrain.procedural.realistic_crater_generator import (
    RealisticCraterGenerator,
    RealisticCraterData,
)

__all__ = [
    "BaseTerrainGenerator",
    "CraterDistributor",
    "CraterGenerator",
    "CraterData",
    "MoonyardGenerator",
    "RealisticCraterGenerator",
    "RealisticCraterData",
    "perlin_1d",
    "perlin_2d",
]
