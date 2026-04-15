"""Physics interface for a single robot's rigid body links.

Provides access to world poses, contact forces, and force/torque application
for wheel links. Used by the terramechanics pipeline.

Reference: OmniLRS src/robots/robot.py (RobotRigidGroup class)
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

try:
    from isaacsim.core.prims import SingleRigidPrim, RigidPrim
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    _HAS_ISAAC = True
except ImportError:
    _HAS_ISAAC = False


class RobotRigidGroup:
    """Rigid body group for a single robot's target links (typically wheels).

    Tracks contact forces, world poses, and velocities for physics interaction.
    Provides force/torque application for terramechanics.

    Interface:
        initialize(world) -> None
        get_world_poses() -> np.ndarray          # (N, 4, 4) transforms
        get_pose() -> (positions, orientations)   # (N,3), (N,4) w,x,y,z
        get_pose_of_base_link() -> (pos, orient)
        get_velocities() -> (linear, angular)
        get_net_contact_forces() -> np.ndarray    # (N, 3)
        apply_force_torque(forces, torques) -> None
    """

    def __init__(
        self,
        root_path: str,
        robot_name: str,
        target_links: List[str],
        base_link: str,
    ) -> None:
        self._root_path = root_path
        self._robot_name = robot_name
        self._target_links = target_links
        self._base_link = base_link
        self._prims: List = []
        self._prim_views: List = []
        self._base_prim = None
        self._dt: float = 0.016666

    def initialize(self, world) -> None:
        """Initialize rigid prims for all target links and base link."""
        if not _HAS_ISAAC:
            logger.warning("Isaac Sim not available, RobotRigidGroup cannot initialize")
            return

        self._dt = world.get_physics_dt()
        world.reset()

        for link in self._target_links:
            prim, view = self._init_link(link)
            self._prims.append(prim)
            self._prim_views.append(view)

        if self._base_link:
            self._base_prim, _ = self._init_link(self._base_link)

        world.reset()
        logger.info("RobotRigidGroup initialized for %s (%d links)",
                     self._robot_name, len(self._target_links))

    def _init_link(self, link_name: str):
        prim_path = os.path.join(self._root_path, self._robot_name.strip("/"), link_name)
        prim = SingleRigidPrim(prim_path=prim_path, name=f"{self._robot_name}/{link_name}")
        view = RigidPrim(
            prim_paths_expr=prim_path,
            name=f"{self._robot_name}/{link_name}_view",
            track_contact_forces=True,
        )
        view.initialize()
        return prim, view

    def get_world_poses(self) -> np.ndarray:
        """Return (N, 4, 4) homogeneous transform matrices for target links."""
        n = len(self._target_links)
        poses = np.zeros((n, 4, 4))
        for i, prim in enumerate(self._prims):
            position, orientation = prim.get_world_pose()
            rot = quat_to_rot_matrix(orientation)
            poses[i, :3, :3] = rot
            poses[i, :3, 3] = position
            poses[i, 3, 3] = 1.0
        return poses

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return positions (N,3) and orientations (N,4) in (w,x,y,z) format.

        Removes wheel pitch rotation to align with global frame.
        """
        n = len(self._target_links)
        positions = np.zeros((n, 3))
        orientations = np.zeros((n, 4))
        for i, prim in enumerate(self._prims):
            position, orientation = prim.get_world_pose()
            # orientation is (w, x, y, z), convert to scipy (x, y, z, w)
            quat_xyzw = [orientation[1], orientation[2], orientation[3], orientation[0]]
            rotation = R.from_quat(quat_xyzw)

            # Remove pitch rotation to align wheel frame with global frame
            pitch_angle = 2 * np.arctan2(rotation.as_quat()[1], rotation.as_quat()[3])
            pitch_correction = R.from_quat([0, -np.sin(pitch_angle / 2), 0, np.cos(pitch_angle / 2)])
            corrected = rotation * pitch_correction

            q = corrected.as_quat()  # (x, y, z, w)
            positions[i] = position
            orientations[i] = [q[3], q[0], q[1], q[2]]  # back to (w, x, y, z)
        return positions, orientations

    def get_pose_of_base_link(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, orientation) of the base link."""
        if self._base_prim is None:
            raise RuntimeError("Base link not initialized")
        return self._base_prim.get_world_pose()

    def get_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return linear (N,3) and angular (N,3) velocities of target links."""
        n = len(self._target_links)
        linear = np.zeros((n, 3))
        angular = np.zeros((n, 3))
        for i, prim in enumerate(self._prims):
            linear[i] = prim.get_linear_velocity()
            angular[i] = prim.get_angular_velocity()
        return linear, angular

    def get_net_contact_forces(self) -> np.ndarray:
        """Return net contact forces (N, 3) on each target link."""
        n = len(self._target_links)
        forces = np.zeros((n, 3))
        for i, view in enumerate(self._prim_views):
            forces[i] = view.get_net_contact_forces(dt=self._dt).squeeze()
        return forces

    def apply_force_torque(
        self,
        forces: np.ndarray,
        torques: np.ndarray,
        is_global: bool = False,
    ) -> None:
        """Apply forces and torques to target link bodies.

        Args:
            forces: (N, 3) force vectors per link.
            torques: (N, 3) torque vectors per link.
            is_global: If True, vectors are in world frame; if False, in body-local frame.
        """
        n = len(self._target_links)
        assert forces.shape[0] == n, f"forces shape {forces.shape} != {n} links"
        assert torques.shape[0] == n, f"torques shape {torques.shape} != {n} links"
        for i, view in enumerate(self._prim_views):
            view.apply_forces_and_torques_at_pos(
                forces=forces[i], torques=torques[i], is_global=is_global
            )
