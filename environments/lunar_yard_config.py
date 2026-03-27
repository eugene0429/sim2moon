import dataclasses


@dataclasses.dataclass
class Coordinates:
    """Lunar surface coordinates for the observer location."""

    latitude: float = 46.8
    longitude: float = -26.3

    def __post_init__(self):
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude must be in [-90, 90], got {self.latitude}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude must be in [-180, 180], got {self.longitude}")


@dataclasses.dataclass
class LunarYardConf:
    """Configuration for the LunarYard outdoor environment."""

    lab_length: float = 40.0
    lab_width: float = 40.0
    resolution: float = 0.02
    root_path: str = "/LunarYard"
    coordinates: Coordinates = dataclasses.field(default_factory=Coordinates)

    def __post_init__(self):
        if isinstance(self.coordinates, dict):
            self.coordinates = Coordinates(**self.coordinates)
        if self.lab_length <= 0:
            raise ValueError(f"lab_length must be positive, got {self.lab_length}")
        if self.lab_width <= 0:
            raise ValueError(f"lab_width must be positive, got {self.lab_width}")
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
