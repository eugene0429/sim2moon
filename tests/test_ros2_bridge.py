"""Tests for ROS2 bridge module.

Tests the non-ROS2 parts: modification queue, quaternion conversion,
configuration parsing, and stub behavior when rclpy is unavailable.
"""

import numpy as np
import pytest

from bridges.ros2_node import (
    _ModificationMixin,
    isaac_to_ros_quat,
    ros_to_isaac_quat,
    EnvironmentNode,
    RobotNode,
)


# ── Quaternion conversion ───────────────────────────────────────────────────

class TestQuaternionConversion:
    def test_isaac_to_ros_identity(self):
        # Isaac (w=1, x=0, y=0, z=0) -> ROS (x=0, y=0, z=0, w=1)
        q_isaac = np.array([1.0, 0.0, 0.0, 0.0])
        x, y, z, w = isaac_to_ros_quat(q_isaac)
        assert w == pytest.approx(1.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(0.0)

    def test_ros_to_isaac_identity(self):
        result = ros_to_isaac_quat(0.0, 0.0, 0.0, 1.0)
        assert result == [1.0, 0.0, 0.0, 0.0]

    def test_roundtrip(self):
        q_isaac = np.array([0.5, 0.5, 0.5, 0.5])
        x, y, z, w = isaac_to_ros_quat(q_isaac)
        q_back = ros_to_isaac_quat(x, y, z, w)
        np.testing.assert_allclose(q_back, q_isaac)

    def test_arbitrary_quat(self):
        # Isaac (w=0.7, x=0.1, y=0.2, z=0.3)
        q_isaac = np.array([0.7, 0.1, 0.2, 0.3])
        x, y, z, w = isaac_to_ros_quat(q_isaac)
        assert x == pytest.approx(0.1)
        assert y == pytest.approx(0.2)
        assert z == pytest.approx(0.3)
        assert w == pytest.approx(0.7)


# ── Modification queue ──────────────────────────────────────────────────────

class TestModificationMixin:
    def test_queue_and_apply(self):
        mixin = _ModificationMixin()
        results = []
        mixin._queue(lambda value: results.append(value), value=42)
        mixin._queue(lambda value: results.append(value), value=99)
        assert len(results) == 0
        mixin.apply_modifications()
        assert results == [42, 99]

    def test_apply_clears_queue(self):
        mixin = _ModificationMixin()
        mixin._queue(lambda: None)
        mixin.apply_modifications()
        # Second apply should be no-op
        mixin.apply_modifications()

    def test_clear_modifications(self):
        mixin = _ModificationMixin()
        mixin._queue(lambda: None)
        mixin._queue(lambda: None)
        mixin.clear_modifications()
        assert len(mixin._modifications) == 0

    def test_error_in_modification_continues(self):
        mixin = _ModificationMixin()
        results = []

        def bad():
            raise RuntimeError("oops")

        mixin._queue(bad)
        mixin._queue(lambda value: results.append(value), value="ok")
        mixin.apply_modifications()  # Should not raise
        assert results == ["ok"]

    def test_empty_apply(self):
        mixin = _ModificationMixin()
        mixin.apply_modifications()  # Should not raise


# ── EnvironmentNode (stub mode, no rclpy) ───────────────────────────────────

class TestEnvironmentNodeStub:
    def test_creation_without_ros2(self):
        node = EnvironmentNode(environment=None, cfg={})
        assert node.trigger_reset is False

    def test_trigger_reset_settable(self):
        node = EnvironmentNode(environment=None, cfg={})
        node.trigger_reset = True
        assert node.trigger_reset is True

    def test_apply_modifications_empty(self):
        node = EnvironmentNode(environment=None, cfg={})
        node.apply_modifications()  # Should not raise

    def test_periodic_update(self):
        node = EnvironmentNode(environment=None, cfg={})
        node.periodic_update(dt=0.033)  # Should not raise


# ── RobotNode (stub mode, no rclpy) ────────────────────────────────────────

class TestRobotNodeStub:
    def test_creation_without_ros2(self):
        node = RobotNode(robot_manager=None, sensor_manager=None, cfg={})
        assert node._rm is None

    def test_apply_modifications_empty(self):
        node = RobotNode(robot_manager=None, sensor_manager=None, cfg={})
        node.apply_modifications()  # Should not raise

    def test_reset_all_robots_no_manager(self):
        node = RobotNode(robot_manager=None, sensor_manager=None, cfg={})
        node.reset_all_robots()  # Should not raise

    def test_publish_gt_tf_no_manager(self):
        node = RobotNode(robot_manager=None, sensor_manager=None, cfg={})
        node.publish_gt_tf()  # Should not raise


# ── ROS2Bridge config parsing ───────────────────────────────────────────────

class TestROS2BridgeConfig:
    def test_default_config(self):
        from bridges.ros2_bridge import ROS2Bridge

        bridge = ROS2Bridge(cfg={})
        assert bridge._domain_id == 0
        assert bridge._bridge_name == "humble"
        assert bridge._publish_gt_tf is True
        assert bridge._physics_dt == pytest.approx(0.0333)
        assert bridge._enforce_realtime is True

    def test_custom_config(self):
        from bridges.ros2_bridge import ROS2Bridge

        cfg = {
            "mode": {
                "ROS_DOMAIN_ID": 5,
                "bridge_name": "foxy",
                "publish_gt_tf": False,
            },
            "physics_dt": 0.016,
            "enforce_realtime": False,
        }
        bridge = ROS2Bridge(cfg=cfg)
        assert bridge._domain_id == 5
        assert bridge._bridge_name == "foxy"
        assert bridge._publish_gt_tf is False
        assert bridge._physics_dt == pytest.approx(0.016)
        assert bridge._enforce_realtime is False

    def test_setup_without_rclpy_raises(self):
        from bridges.ros2_bridge import ROS2Bridge, _HAS_ROS2

        bridge = ROS2Bridge(cfg={})
        if not _HAS_ROS2:
            with pytest.raises(RuntimeError, match="rclpy"):
                bridge.setup()

    def test_stop_sets_running_false(self):
        from bridges.ros2_bridge import ROS2Bridge

        bridge = ROS2Bridge(cfg={})
        bridge._running = True
        bridge.stop()
        assert bridge._running is False


# ── ROS2 Services (stub mode) ──────────────────────────────────────────────

class TestEnvironmentServiceHandler:
    def test_creation_without_ros2(self):
        from bridges.ros2_services import EnvironmentServiceHandler

        class FakeNode:
            def create_service(self, *args, **kwargs):
                pass

        handler = EnvironmentServiceHandler(FakeNode(), environment=None)
        assert handler.deformation_enabled is False
        assert handler.reset_requested is False

    def test_clear_reset(self):
        from bridges.ros2_services import EnvironmentServiceHandler

        class FakeNode:
            def create_service(self, *args, **kwargs):
                pass

        handler = EnvironmentServiceHandler(FakeNode(), environment=None)
        handler._reset_requested = True
        handler.clear_reset()
        assert handler.reset_requested is False


class TestRobotServiceHandler:
    def test_creation(self):
        from bridges.ros2_services import RobotServiceHandler

        class FakeNode:
            def create_service(self, *args, **kwargs):
                pass
            def create_subscription(self, *args, **kwargs):
                pass

        handler = RobotServiceHandler(FakeNode(), robot_manager=None)
        assert len(handler._reset_requests) == 0

    def test_is_reset_requested_no_request(self):
        from bridges.ros2_services import RobotServiceHandler

        class FakeNode:
            pass

        handler = RobotServiceHandler(FakeNode())
        requested, target = handler.is_reset_requested("husky")
        assert requested is False
        assert target is None

    def test_is_reset_requested_with_request(self):
        from bridges.ros2_services import RobotServiceHandler

        class FakeNode:
            pass

        handler = RobotServiceHandler(FakeNode())
        handler._reset_requests["husky"] = None
        requested, target = handler.is_reset_requested("husky")
        assert requested is True
        assert target is None
        # Should be consumed
        requested2, _ = handler.is_reset_requested("husky")
        assert requested2 is False

    def test_is_reset_requested_with_target(self):
        from bridges.ros2_services import RobotServiceHandler

        class FakeNode:
            pass

        handler = RobotServiceHandler(FakeNode())
        handler._reset_requests["husky"] = ([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        requested, target = handler.is_reset_requested("husky")
        assert requested is True
        assert target[0] == [1.0, 2.0, 3.0]
