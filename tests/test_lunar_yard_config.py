import pytest
from environments.lunar_yard_config import Coordinates, LunarYardConf


class TestCoordinates:
    def test_defaults(self):
        c = Coordinates()
        assert c.latitude == 46.8
        assert c.longitude == -26.3

    def test_custom(self):
        c = Coordinates(latitude=0.0, longitude=180.0)
        assert c.latitude == 0.0
        assert c.longitude == 180.0

    def test_invalid_latitude(self):
        with pytest.raises(ValueError, match="latitude"):
            Coordinates(latitude=91.0)

    def test_invalid_longitude(self):
        with pytest.raises(ValueError, match="longitude"):
            Coordinates(longitude=181.0)


class TestLunarYardConf:
    def test_defaults(self):
        conf = LunarYardConf()
        assert conf.lab_length == 40.0
        assert conf.lab_width == 40.0
        assert conf.resolution == 0.02
        assert conf.root_path == "/LunarYard"
        assert isinstance(conf.coordinates, Coordinates)

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="lab_length"):
            LunarYardConf(lab_length=-1.0)

    def test_invalid_resolution(self):
        with pytest.raises(ValueError, match="resolution"):
            LunarYardConf(resolution=0.0)

    def test_coordinates_from_dict(self):
        conf = LunarYardConf(coordinates={"latitude": 10.0, "longitude": 20.0})
        assert conf.coordinates.latitude == 10.0
        assert conf.coordinates.longitude == 20.0
