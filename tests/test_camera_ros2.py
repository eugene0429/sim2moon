"""Tests for camera auto-discovery, config merging, and ROS2 publisher setup."""

import pytest

from sensors.config import CameraROS2Conf
from sensors.camera_ros2 import CameraROS2Publisher, DiscoveredCamera


# ── CameraROS2Conf ──────────────────────────────────────────────────────────

class TestCameraROS2Conf:
    def test_defaults(self):
        conf = CameraROS2Conf()
        assert conf.image_type == "rgb"
        assert conf.resolution == [640, 480]
        assert conf.enabled is True

    def test_custom(self):
        conf = CameraROS2Conf(
            prim_path="/Robot/cam",
            topic="/robot/cam/image_raw",
            image_type="depth",
            resolution=[1280, 720],
            frame_id="robot/cam_link",
        )
        assert conf.prim_path == "/Robot/cam"
        assert conf.image_type == "depth"
        assert conf.resolution == [1280, 720]

    def test_invalid_image_type(self):
        with pytest.raises(AssertionError):
            CameraROS2Conf(image_type="invalid")

    def test_invalid_resolution(self):
        with pytest.raises(AssertionError):
            CameraROS2Conf(resolution=[0, 480])

    def test_resolution_length(self):
        with pytest.raises(AssertionError):
            CameraROS2Conf(resolution=[640])


# ── Config merge (build_camera_configs) ─────────────────────────────────────

class TestBuildCameraConfigs:
    def _make_discovered(self):
        return [
            DiscoveredCamera(
                prim_path="/Robots/husky/left_cam_link/Camera",
                prim_name="Camera",
                parent_link="left_cam_link",
            ),
            DiscoveredCamera(
                prim_path="/Robots/husky/right_cam_link/DepthCamera",
                prim_name="DepthCamera",
                parent_link="right_cam_link",
            ),
        ]

    def test_auto_generates_defaults(self):
        discovered = self._make_discovered()
        configs = CameraROS2Publisher.build_camera_configs(
            discovered, robot_name="husky",
        )
        assert "Camera" in configs
        assert "DepthCamera" in configs
        assert configs["Camera"].topic == "/husky/Camera/image_raw"
        assert configs["Camera"].prim_path == "/Robots/husky/left_cam_link/Camera"
        assert configs["Camera"].frame_id == "husky/left_cam_link"
        assert configs["Camera"].image_type == "rgb"

    def test_override_by_prim_name(self):
        discovered = self._make_discovered()
        overrides = {
            "Camera": CameraROS2Conf(
                topic="/husky/left/image_raw",
                resolution=[1920, 1080],
            ),
        }
        configs = CameraROS2Publisher.build_camera_configs(
            discovered, "husky", overrides,
        )
        assert configs["Camera"].topic == "/husky/left/image_raw"
        assert configs["Camera"].resolution == [1920, 1080]
        # prim_path should be filled from discovery
        assert configs["Camera"].prim_path == "/Robots/husky/left_cam_link/Camera"

    def test_override_by_parent_link(self):
        discovered = self._make_discovered()
        overrides = {
            "right_cam_link": CameraROS2Conf(
                topic="/husky/right/depth",
                image_type="depth",
            ),
        }
        configs = CameraROS2Publisher.build_camera_configs(
            discovered, "husky", overrides,
        )
        assert configs["DepthCamera"].topic == "/husky/right/depth"
        assert configs["DepthCamera"].image_type == "depth"

    def test_manual_camera_added(self):
        """Config for a camera not discovered (explicitly specified prim_path)."""
        discovered = []
        overrides = {
            "thermal_cam": CameraROS2Conf(
                prim_path="/Robots/husky/thermal_link/Camera",
                topic="/husky/thermal/image_raw",
            ),
        }
        configs = CameraROS2Publisher.build_camera_configs(
            discovered, "husky", overrides,
        )
        assert "thermal_cam" in configs
        assert configs["thermal_cam"].prim_path == "/Robots/husky/thermal_link/Camera"

    def test_manual_camera_without_prim_path_ignored(self):
        """Config without prim_path and no discovery match is skipped."""
        discovered = []
        overrides = {
            "ghost_cam": CameraROS2Conf(topic="/ghost"),
            # prim_path is empty, so should not be added
        }
        configs = CameraROS2Publisher.build_camera_configs(
            discovered, "husky", overrides,
        )
        assert "ghost_cam" not in configs

    def test_disabled_camera(self):
        discovered = [
            DiscoveredCamera("/Robot/cam", "cam", "cam_link"),
        ]
        overrides = {
            "cam": CameraROS2Conf(enabled=False),
        }
        configs = CameraROS2Publisher.build_camera_configs(
            discovered, "robot", overrides,
        )
        assert configs["cam"].enabled is False

    def test_no_cameras(self):
        configs = CameraROS2Publisher.build_camera_configs([], "robot")
        assert configs == {}


# ── OmniGraph creation (stub, since no Isaac) ──────────────────────────────

class TestCreateOmnigraph:
    def test_returns_none_without_omnigraph(self):
        pub = CameraROS2Publisher()
        conf = CameraROS2Conf(
            prim_path="/Robot/cam",
            topic="/robot/image_raw",
        )
        result = pub.create_omnigraph("test_cam", conf)
        # OmniGraph not available in test environment
        assert result is None

    def test_disabled_camera_skipped(self):
        pub = CameraROS2Publisher()
        conf = CameraROS2Conf(
            prim_path="/Robot/cam",
            topic="/robot/image_raw",
            enabled=False,
        )
        result = pub.create_omnigraph("test_cam", conf)
        assert result is None
        assert "test_cam" not in pub.active_graphs


# ── setup_robot_cameras (no USD available) ──────────────────────────────────

class TestSetupRobotCameras:
    def test_no_discovery_returns_empty(self):
        pub = CameraROS2Publisher()
        configs = pub.setup_robot_cameras(
            robot_prim_path="/Robots/nonexistent",
            robot_name="test_robot",
        )
        assert configs == {}


# ── RobotParameters cameras_ros2 field ──────────────────────────────────────

class TestRobotParametersCamerasROS2:
    def test_empty_default(self):
        from robots.config import RobotParameters
        params = RobotParameters()
        assert params.cameras_ros2 == {}

    def test_dict_resolution(self):
        from robots.config import RobotParameters
        params = RobotParameters(
            cameras_ros2={
                "left": {
                    "prim_path": "/Robot/left_cam/Camera",
                    "topic": "/robot/left/image_raw",
                    "resolution": [1280, 720],
                },
            },
        )
        assert "left" in params.cameras_ros2
        conf = params.cameras_ros2["left"]
        assert isinstance(conf, CameraROS2Conf)
        assert conf.topic == "/robot/left/image_raw"
        assert conf.resolution == [1280, 720]

    def test_already_conf_passthrough(self):
        from robots.config import RobotParameters
        conf = CameraROS2Conf(topic="/test")
        params = RobotParameters(cameras_ros2={"cam": conf})
        assert params.cameras_ros2["cam"] is conf
