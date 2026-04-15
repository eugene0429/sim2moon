"""
Configuration dataclasses for the celestial system.

Covers stellar engine (ephemeris), sun light, and Earth rendering.
"""

import dataclasses
import datetime
import os
from typing import Tuple


@dataclasses.dataclass
class Date:
    """Calendar date with validation, for specifying simulation start time."""

    year: int = 2024
    month: int = 5
    day: int = 11
    hour: int = 16
    minute: int = 30

    def __post_init__(self):
        self.year = int(self.year)
        self.month = int(self.month)
        self.day = int(self.day)
        self.hour = int(self.hour)
        self.minute = int(self.minute)

        assert 1970 <= self.year <= 2050, "Year must be between 1970 and 2050."
        assert 1 <= self.month <= 12, "Month must be between 1 and 12."
        assert 1 <= self.day <= 31, "Day must be between 1 and 31."
        assert 0 <= self.hour <= 23, "Hour must be between 0 and 23."
        assert 0 <= self.minute <= 59, "Minute must be between 0 and 59."

    def to_datetime(self) -> datetime.datetime:
        return datetime.datetime(
            self.year, self.month, self.day, self.hour, self.minute,
            tzinfo=datetime.timezone.utc,
        )


@dataclasses.dataclass
class StellarEngineConf:
    """Configuration for the stellar engine (ephemeris-based celestial positioning)."""

    enable: bool = True
    start_date: datetime.datetime = dataclasses.field(default_factory=dict)
    time_scale: float = 36000.0
    update_interval: float = 600.0
    distance_scale: float = 1.0
    ephemeris_path: str = "assets/Ephemeris"
    ephemeris: str = "de421.bsp"
    moon_pa: str = "moon_pa_de421_1900-2050.bpc"
    moon_tf: str = "moon_080317.tf"
    pck: str = "pck00008.tpc"
    frame: str = "MOON_ME_DE421"

    def __post_init__(self):
        # Convert dict/Date to datetime
        if isinstance(self.start_date, dict):
            d = Date(**self.start_date)
            self.start_date = d.to_datetime()
        elif isinstance(self.start_date, Date):
            self.start_date = self.start_date.to_datetime()

        assert self.time_scale > 0, "time_scale must be positive."
        assert self.update_interval > 0, "update_interval must be positive."

        # Resolve file paths
        self.ephemeris = os.path.join(self.ephemeris_path, self.ephemeris)
        self.moon_pa = os.path.join(self.ephemeris_path, self.moon_pa)
        self.moon_tf = os.path.join(self.ephemeris_path, self.moon_tf)
        self.pck = os.path.join(self.ephemeris_path, self.pck)


@dataclasses.dataclass
class SunConf:
    """Configuration for the sun distant light."""

    intensity: float = 1750.0
    angle: float = 0.53
    diffuse_multiplier: float = 1.0
    specular_multiplier: float = 1.0
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    temperature: float = 6500.0
    azimuth: float = 180.0
    elevation: float = 45.0

    def __post_init__(self):
        assert self.intensity > 0, "intensity must be positive."
        assert self.angle > 0, "angle must be positive."
        assert self.diffuse_multiplier > 0, "diffuse_multiplier must be positive."
        assert self.specular_multiplier > 0, "specular_multiplier must be positive."
        assert self.temperature > 0, "temperature must be positive."
        assert len(self.color) == 3, "color must be a 3-tuple."
        assert all(0.0 <= c <= 1.0 for c in self.color), "color values must be in [0, 1]."


@dataclasses.dataclass
class EarthConf:
    """Configuration for the Earth visual sphere."""

    texture_path: str = "assets/Textures/Earth/earth_color_with_clouds.tif"
    scale: float = 50.0
    enable: bool = True
