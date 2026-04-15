"""Multi-robot manager.

Spawns, resets, and teleports robots. Manages the mapping between
robot names and their Robot + RobotRigidGroup instances.

Reference: OmniLRS src/robots/robot.py (RobotManager class)
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from robots.config import RobotManagerConf, RobotParameters
from robots.robot import Robot
from robots.rigid_body_group import RobotRigidGroup

logger = logging.getLogger(__name__)

try:
    import omni
    from core.pxr_utils import createXform
    _HAS_ISAAC = True
except ImportError:
    _HAS_ISAAC = False


class RobotManager:
    """Manages multiple robots in the simulation.

    Spawns robots from config, creates rigid body groups for physics,
    and provides reset/teleport operations.

    Interface:
        preload_robot(world) -> None
        add_robot(...) -> None
        add_rigid_group(name, target_links, base_link, world) -> None
        reset_robots() -> None
        reset_robot(name) -> None
        teleport_robot(name, position, orientation) -> None
        get_robot(name) -> Robot
        get_rigid_group(name) -> RobotRigidGroup
    """

    def __init__(self, cfg: RobotManagerConf) -> None:
        if isinstance(cfg, dict):
            cfg = RobotManagerConf(**cfg)
        self._cfg = cfg
        self._robots: Dict[str, Robot] = {}
        self._rigid_groups: Dict[str, RobotRigidGroup] = {}
        self._num_robots = 0
        self._mesh_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Create root Xform in USD stage
        if _HAS_ISAAC:
            stage = omni.usd.get_context().get_stage()
            createXform(stage, cfg.robots_root)

        logger.info("RobotManager initialized (max_robots=%d, root=%s)",
                     cfg.max_robots, cfg.robots_root)

    def set_mesh_position(self, mesh_position) -> None:
        """Set terrain mesh world-space offset for robot placement.

        Args:
            mesh_position: (x, y, z) terrain mesh offset in world space.
        """
        mp = [float(v) for v in mesh_position]
        self._mesh_position = (mp[0], mp[1], mp[2])

    def preload_robot(self, world=None) -> None:
        """Spawn all robots defined in config and create their rigid body groups.

        Args:
            world: Isaac Sim World object. If None, robots are spawned but
                   rigid body groups are not initialized (no physics).
        """
        for param in self._cfg.parameters:
            self.add_robot(param)
            if world is not None:
                self.add_rigid_group(
                    param.robot_name,
                    param.target_links,
                    param.base_link,
                    world,
                )

    def add_robot(self, param: RobotParameters) -> None:
        """Add a single robot to the scene from parameters."""
        name = param.robot_name if param.robot_name.startswith("/") else f"/{param.robot_name}"

        if self._num_robots >= self._cfg.max_robots:
            warnings.warn(f"Max robots ({self._cfg.max_robots}) reached, ignoring {name}")
            return

        if name in self._robots:
            warnings.warn(f"Robot {name} already exists, ignoring")
            return

        robot = Robot(
            usd_path=param.usd_path,
            robot_name=param.robot_name,
            robots_root=self._cfg.robots_root,
            is_on_nucleus=self._cfg.uses_nucleus,
            is_ROS2=self._cfg.is_ROS2,
            domain_id=param.domain_id,
            wheel_joints=param.wheel_joints,
            camera_conf=param.camera,
            imu_sensor_path=param.imu_sensor_path,
            dimensions=param.dimensions,
            turn_speed_coef=param.turn_speed_coef,
            solar_panel_joint=param.solar_panel_joint,
        )
        pos = list(param.pose.position)
        pos[0] += self._mesh_position[0]
        pos[1] += self._mesh_position[1]
        pos[2] += self._mesh_position[2]
        robot.load(pos, param.pose.orientation)
        self._robots[name] = robot
        self._num_robots += 1
        logger.info("Robot %s added (%d/%d)", name, self._num_robots, self._cfg.max_robots)

    def add_rigid_group(
        self,
        robot_name: str,
        target_links: List[str],
        base_link: str,
        world,
    ) -> None:
        """Create a RobotRigidGroup for physics interaction."""
        name = robot_name if robot_name.startswith("/") else f"/{robot_name}"
        rrg = RobotRigidGroup(
            root_path=self._cfg.robots_root,
            robot_name=robot_name,
            target_links=target_links,
            base_link=base_link,
        )
        rrg.initialize(world)
        self._rigid_groups[name] = rrg

    def reset_robots(self) -> None:
        """Reset all robots to their initial poses."""
        for robot in self._robots.values():
            robot.reset()

    def reset_robot(self, name: str) -> None:
        """Reset a specific robot by name."""
        name = name if name.startswith("/") else f"/{name}"
        if name in self._robots:
            self._robots[name].reset()
        else:
            warnings.warn(f"Robot {name} not found")

    def teleport_robot(
        self,
        name: str,
        position: List[float],
        orientation: List[float],
    ) -> None:
        """Teleport a specific robot to a new pose."""
        name = name if name.startswith("/") else f"/{name}"
        if name in self._robots:
            self._robots[name].teleport(position, orientation)
        else:
            warnings.warn(f"Robot {name} not found. Available: {list(self._robots.keys())}")

    def get_robot(self, name: str) -> Optional[Robot]:
        """Get a robot by name."""
        name = name if name.startswith("/") else f"/{name}"
        return self._robots.get(name)

    def get_rigid_group(self, name: str) -> Optional[RobotRigidGroup]:
        """Get a rigid body group by robot name."""
        name = name if name.startswith("/") else f"/{name}"
        return self._rigid_groups.get(name)

    @property
    def robots(self) -> Dict[str, Robot]:
        return self._robots

    @property
    def rigid_groups(self) -> Dict[str, RobotRigidGroup]:
        return self._rigid_groups

    @property
    def num_robots(self) -> int:
        return self._num_robots
