from core.enums import SimulatorMode


def test_simulator_mode_has_ros2():
    assert SimulatorMode.ROS2.value == "ROS2"


def test_simulator_mode_has_sdg():
    assert SimulatorMode.SDG.value == "SDG"


def test_simulator_mode_from_string():
    assert SimulatorMode("ROS2") is SimulatorMode.ROS2
    assert SimulatorMode("SDG") is SimulatorMode.SDG
