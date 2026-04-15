"""Individual robot control.

Handles spawning a single robot from USD, managing its pose, driving,
and accessing sensors (cameras, IMU). Each robot has its own ROS2
namespace and domain ID for multi-robot support.

Reference: OmniLRS src/robots/robot.py (Robot class)
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import omni
    from pxr import Gf, Usd
    from omni.isaac.dynamic_control import _dynamic_control
    _HAS_ISAAC = True
except ImportError:
    _HAS_ISAAC = False

try:
    from core.pxr_utils import createObject
    _HAS_WORLDBUILDERS = True
except ImportError:
    _HAS_WORLDBUILDERS = False


class Robot:
    """Single robot instance in the simulation.

    Manages spawning, pose, driving, and sensor access for one robot.

    Interface:
        load(position, orientation) -> None
        get_pose() -> (position, orientation)
        teleport(position, orientation) -> None
        reset() -> None
        drive_straight(velocity) -> None
        drive_turn(wheel_speed) -> None
        stop_drive() -> None
        get_wheels_joint_angles() -> List[float]
        deploy_solar_panel() -> None
        stow_solar_panel() -> None
    """

    def __init__(
        self,
        usd_path: str,
        robot_name: str,
        robots_root: str = "/Robots",
        is_on_nucleus: bool = False,
        is_ROS2: bool = False,
        domain_id: int = 0,
        wheel_joints: Dict = None,
        camera_conf: Dict = None,
        imu_sensor_path: str = "",
        dimensions: Dict = None,
        turn_speed_coef: float = 1.0,
        pos_relative_to_prim: str = "",
        solar_panel_joint: str = "",
    ) -> None:
        self._usd_path = str(usd_path)
        self._robots_root = robots_root
        self._robot_name = robot_name if robot_name.startswith("/") else f"/{robot_name}"
        self._robot_path = os.path.join(self._robots_root, self._robot_name.strip("/"))
        self._is_on_nucleus = is_on_nucleus
        self._is_ROS2 = is_ROS2
        self._domain_id = int(domain_id)
        self._wheel_joint_names = wheel_joints or {}
        self._camera_conf = camera_conf or {}
        self._imu_sensor_path = imu_sensor_path
        self._dimensions = dimensions or {}
        self._turn_speed_coef = turn_speed_coef
        self._solar_panel_joint = solar_panel_joint

        self._reset_position = None
        self._reset_orientation = None
        self._root_body_id = None
        self._dofs: Dict[str, list] = {}
        self._solar_panel_dof = None
        self._dc = None
        self._stage = None

    def load(self, position: List[float], orientation: List[float]) -> None:
        """Spawn the robot USD in the scene at the given pose."""
        if not _HAS_ISAAC:
            logger.warning("Isaac Sim not available, cannot load robot %s", self._robot_name)
            return

        self._stage = omni.usd.get_context().get_stage()
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._reset_position = list(position)
        self._reset_orientation = list(orientation)

        if self._is_on_nucleus:
            from isaacsim.core.utils.nucleus import get_assets_root_path
            nucleus = get_assets_root_path()
            self._usd_path = os.path.join(nucleus, self._usd_path)

        if _HAS_WORLDBUILDERS:
            createObject(
                self._robot_path,
                self._stage,
                self._usd_path,
                is_instance=False,
                position=Gf.Vec3d(*position),
                rotation=Gf.Quatd(*orientation),
            )
        else:
            logger.warning("WorldBuilders not available, robot USD not spawned")

        if self._is_ROS2:
            self._edit_graphs()

        logger.info("Robot %s loaded at %s", self._robot_name, position)

    def _edit_graphs(self) -> None:
        """Set ROS2 namespace and domain_id on OmniGraph nodes."""
        if self._stage is None:
            return
        for prim in Usd.PrimRange(self._stage.GetPrimAtPath(self._robot_path)):
            attrs = [a for a in prim.GetAttributes() if a.GetName().split(":")[0] == "graph"]
            if attrs:
                prim.GetAttribute("graph:variable:Namespace").Set(self._robot_name)
                prim.GetAttribute("graph:variable:Context").Set(self._domain_id)

    def get_pose(self) -> Tuple[List[float], List[float]]:
        """Return current (position, orientation) of the robot base."""
        if self._root_body_id is None:
            self._get_root_body()
        pose = self._dc.get_rigid_body_pose(self._root_body_id)
        return list(pose.p), list(pose.r)

    def teleport(self, position: List[float], orientation: List[float]) -> None:
        """Teleport robot to position with orientation (w, x, y, z) Isaac convention."""
        if self._root_body_id is None:
            self._get_root_body()
        # Dynamic Control expects (x, y, z, w), convert from Isaac (w, x, y, z)
        w, x, y, z = orientation
        transform = _dynamic_control.Transform(position, [x, y, z, w])
        self._dc.set_rigid_body_pose(self._root_body_id, transform)
        self._dc.set_rigid_body_linear_velocity(self._root_body_id, [0, 0, 0])
        self._dc.set_rigid_body_angular_velocity(self._root_body_id, [0, 0, 0])

    def reset(self) -> None:
        """Reset robot to its initial spawn pose."""
        self._root_body_id = None
        self.teleport(list(self._reset_position), list(self._reset_orientation))

    def drive_straight(self, velocity: float) -> None:
        """Drive both sides at the same velocity."""
        self._set_wheels_velocity(velocity, "left")
        self._set_wheels_velocity(velocity, "right")

    def drive_turn(self, wheel_speed: float) -> None:
        """Turn by driving sides in opposite directions."""
        self._set_wheels_velocity(-wheel_speed, "left")
        self._set_wheels_velocity(wheel_speed, "right")

    def stop_drive(self) -> None:
        """Stop all wheels."""
        self._set_wheels_velocity(0, "left")
        self._set_wheels_velocity(0, "right")

    def get_wheels_joint_angles(self) -> List[float]:
        """Return current joint angles for all wheels."""
        self._init_dofs()
        angles = []
        for side in ["left", "right"]:
            for dof in self._dofs.get(side, []):
                angles.append(self._dc.get_dof_position(dof))
        return angles

    def deploy_solar_panel(self) -> None:
        """Deploy solar panel to 0 degrees."""
        self._init_solar_panel_dof()
        if self._solar_panel_dof is not None:
            self._dc.set_dof_position_target(self._solar_panel_dof, math.radians(0))

    def stow_solar_panel(self) -> None:
        """Stow solar panel to -80 degrees."""
        self._init_solar_panel_dof()
        if self._solar_panel_dof is not None:
            self._dc.set_dof_position_target(self._solar_panel_dof, math.radians(-80))

    @property
    def name(self) -> str:
        return self._robot_name

    @property
    def path(self) -> str:
        return self._robot_path

    def _get_root_body(self) -> None:
        art = self._dc.get_articulation(self._robot_path)
        self._root_body_id = self._dc.get_articulation_root_body(art)

    def _set_wheels_velocity(self, velocity: float, side: str) -> None:
        self._init_dofs()
        for dof in self._dofs.get(side, []):
            self._dc.set_dof_velocity_target(dof, velocity)

    def _init_dofs(self) -> None:
        if not self._wheel_joint_names or "left" in self._dofs:
            return
        self._dofs = {"left": [], "right": []}
        art = self._dc.get_articulation(self._robot_path)
        for side in ["left", "right"]:
            for joint_name in self._wheel_joint_names.get(side, []):
                dof = self._dc.find_articulation_dof(art, joint_name)
                self._dofs[side].append(dof)

    def _init_solar_panel_dof(self) -> None:
        if self._solar_panel_dof is not None or not self._solar_panel_joint:
            return
        art = self._dc.get_articulation(self._robot_path)
        self._solar_panel_dof = self._dc.find_articulation_dof(art, self._solar_panel_joint)
