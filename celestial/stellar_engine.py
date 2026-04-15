"""
Stellar engine for computing celestial body positions.

Uses JPL DE421 ephemeris data via the Skyfield library to compute
the altitude, azimuth, and distance of celestial bodies (sun, earth, venus)
as seen from an observer on the Moon's surface.

Reference: OmniLRS src/stellar/stellar_engine.py
"""

import datetime
import logging
import math
from typing import Dict, Optional, Tuple

from scipy.spatial.transform import Rotation as SSTR
from skyfield.api import PlanetaryConstants, load

from celestial.config import StellarEngineConf

logger = logging.getLogger(__name__)


class StellarEngine:
    """
    Computes realistic celestial body positions using JPL ephemeris data.

    After construction, call set_coordinates() to place the observer on the Moon,
    then call update(dt) each simulation step. Positions are only recomputed
    when the accumulated time exceeds update_interval.

    Interface contract (exported to other agents):
        set_coordinates(lat, lon) -> None
        set_time(timestamp) -> None
        update(dt) -> bool
        get_sun_alt_az() -> Tuple[float, float, float]
        get_earth_alt_az() -> Tuple[float, float, float]
        get_sun_quaternion() -> np.ndarray
    """

    SUPPORTED_BODIES = ("earth", "moon", "sun", "venus")

    def __init__(self, cfg: StellarEngineConf) -> None:
        self._cfg = cfg
        self._ts = load.timescale()
        self._current_time = cfg.start_date
        self._last_update = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)
        self._observer = None

        self._load_ephemeris()
        self._t = self._ts.from_datetime(self._current_time)

    def _load_ephemeris(self) -> None:
        """Load JPL ephemeris and lunar frame data."""
        self._eph = load(self._cfg.ephemeris)

        self._bodies: Dict[str, object] = {
            "earth": self._eph["earth"],
            "moon": self._eph["moon"],
            "sun": self._eph["sun"],
            "venus": self._eph["venus"],
        }

        self._pc = PlanetaryConstants()
        self._pc.read_text(load(self._cfg.moon_tf))
        self._pc.read_text(load(self._cfg.pck))
        self._pc.read_binary(load(self._cfg.moon_pa))
        self._frame = self._pc.build_frame_named(self._cfg.frame)

        self._moon = self._bodies["moon"]

    # -- Observer setup --

    def set_coordinates(self, lat: float, lon: float) -> None:
        """
        Set the observer's position on the Moon's surface.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
        """
        self._observer = self._moon + self._pc.build_latlon_degrees(
            self._frame, lat, lon,
        )
        logger.info("StellarEngine: observer set at lat=%.2f, lon=%.2f", lat, lon)

    # -- Time control --

    def set_time(self, timestamp: float) -> None:
        """
        Set the current simulation time directly (UTC seconds since epoch).

        Args:
            timestamp: UTC timestamp in seconds.
        """
        self._current_time = datetime.datetime.fromtimestamp(
            timestamp, datetime.timezone.utc,
        )
        self._last_update = self._current_time
        self._t = self._ts.from_datetime(self._current_time)
        logger.debug("StellarEngine: time set to %s", self._current_time)

    def set_time_scale(self, time_scale: float) -> None:
        """Set the time acceleration factor."""
        assert time_scale > 0, "time_scale must be positive"
        self._cfg.time_scale = time_scale

    def update(self, dt: float) -> bool:
        """
        Advance simulation time by dt * time_scale seconds.

        Returns True if the ephemeris was recomputed (i.e., enough time
        has elapsed since the last update to exceed update_interval).

        Args:
            dt: Wall-clock delta time in seconds.

        Returns:
            True if celestial positions were recomputed.
        """
        self._current_time += datetime.timedelta(seconds=dt * self._cfg.time_scale)
        time_delta = self._current_time - self._last_update

        if time_delta.total_seconds() >= self._cfg.update_interval:
            self._last_update = self._current_time
            self._t = self._ts.from_datetime(self._current_time)
            logger.debug(
                "StellarEngine: recomputed at %s (delta=%.0fs)",
                self._current_time, time_delta.total_seconds(),
            )
            return True
        return False

    # -- Position queries --

    def get_alt_az(self, body: str) -> Tuple[float, float, float]:
        """
        Get altitude, azimuth, and distance of a body from the observer.

        Args:
            body: One of 'earth', 'sun', 'venus'.

        Returns:
            (altitude_deg, azimuth_deg, distance_m)
        """
        self._check_observer()
        apparent = self._observer.at(self._t).observe(self._bodies[body]).apparent()
        alt, az, distance = apparent.altaz()
        return alt.degrees, az.degrees, distance.m

    def get_sun_alt_az(self) -> Tuple[float, float, float]:
        """Convenience: get sun altitude, azimuth, distance."""
        return self.get_alt_az("sun")

    def get_earth_alt_az(self) -> Tuple[float, float, float]:
        """Convenience: get earth altitude, azimuth, distance."""
        return self.get_alt_az("earth")

    def get_position(self, body: str) -> Tuple[float, float, float]:
        """
        Get the 3D position of a body in the observer's frame (meters).

        Args:
            body: One of 'earth', 'sun', 'venus'.

        Returns:
            (x, y, z) position in meters.
        """
        self._check_observer()
        apparent = self._observer.at(self._t).observe(self._bodies[body]).apparent()
        return tuple(apparent.position.to("m").value)

    def get_local_position(self, body: str) -> Tuple[float, float, float]:
        """
        Get the local ENU position of a body from alt/az conversion.

        Args:
            body: One of 'earth', 'sun', 'venus'.

        Returns:
            (x, y, z) local position in meters.
        """
        alt, az, dist = self.get_alt_az(body)
        x = dist * math.cos(math.radians(alt)) * math.cos(math.radians(az))
        y = dist * math.cos(math.radians(alt)) * math.sin(math.radians(az))
        z = dist * math.sin(math.radians(alt))
        return (x, y, z)

    # -- Quaternion conversion --

    def get_sun_quaternion(self) -> Tuple[float, float, float, float]:
        """
        Get the quaternion that rotates a default [0,0,-1] direction
        to point toward the sun's current alt/az position.

        Returns:
            (w, x, y, z) quaternion.
        """
        alt, az, _ = self.get_sun_alt_az()
        return self.convert_alt_az_to_quat(alt, az)

    @staticmethod
    def convert_alt_az_to_quat(alt: float, az: float) -> Tuple[float, float, float, float]:
        """
        Convert altitude/azimuth to a quaternion for a directional light.

        Assumes the default light direction is [0, 0, -1].

        Args:
            alt: Altitude in degrees.
            az: Azimuth in degrees.

        Returns:
            (w, x, y, z) quaternion.
        """
        x, y, z, w = SSTR.from_euler("xyz", [0, alt, az - 90], degrees=True).as_quat()
        return (w, x, y, z)

    # -- Properties --

    @property
    def current_time(self) -> datetime.datetime:
        return self._current_time

    @property
    def time_scale(self) -> float:
        return self._cfg.time_scale

    # -- Internal --

    def _check_observer(self) -> None:
        if self._observer is None:
            raise RuntimeError(
                "Observer not set. Call set_coordinates(lat, lon) first."
            )
