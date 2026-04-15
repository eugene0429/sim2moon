#!/bin/bash
# Launch new_lunar_sim with ROS2 support.
#
# Usage:
#   ./run_ros2.sh environment=lunar_yard_40m_workshop_full_husky
#
# This sets the env vars required for Isaac Sim's bundled ROS2 Humble
# libraries (Python 3.11 compatible) so the isaacsim.ros2.bridge
# extension starts up correctly.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim}"
ROS2_LIB="$ISAAC_PATH/exts/isaacsim.ros2.bridge/humble/lib"

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$ROS2_LIB:${LD_LIBRARY_PATH:-}"

echo "ROS2 env: ROS_DISTRO=$ROS_DISTRO  RMW=$RMW_IMPLEMENTATION"
echo "LD_LIBRARY_PATH includes: $ROS2_LIB"

cd "$SCRIPT_DIR"
exec "$ISAAC_PATH/python.sh" ./main.py "$@"
