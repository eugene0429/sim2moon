"""SDG (Synthetic Data Generation) simulation manager.

Orchestrates the scene randomization loop:
    1. Randomize camera pose on the terrain
    2. Randomize sun/earth positions
    3. Periodically randomize terrain and rocks
    4. Capture annotated frame via AutonomousLabeling
    5. Repeat until num_images reached

Ported from OmniLRS SDG_SimulationManager + SDG_Lunaryard with composition.
"""

import logging
import signal
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as SSTR

from data_generation.auto_labeling import AutonomousLabeling
from data_generation.config import AutoLabelingConf, SDGCameraConf

logger = logging.getLogger(__name__)

try:
    import omni
    from pxr import Gf, UsdGeom
    _HAS_USD = True
except ImportError:
    _HAS_USD = False


class SceneRandomizer:
    """Randomizes sun, earth, and camera positions for SDG diversity.

    Operates directly on USD stage prims without depending on
    environment controllers, making it usable with any environment type.
    """

    def __init__(
        self,
        stage=None,
        scene_root: str = "/LunarYard",
        terrain_resolution: float = 0.02,
        seed: int = 42,
    ) -> None:
        self._stage = stage
        self._scene_root = scene_root
        self._resolution = terrain_resolution
        self._rng = np.random.default_rng(seed)

        self._sun_prim = None
        self._earth_prim = None
        self._camera_xform = None
        self._dem: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None

    def set_terrain_data(self, dem: np.ndarray, mask: np.ndarray) -> None:
        """Register the current DEM and mask for camera height sampling."""
        self._dem = dem
        self._mask = mask

    def find_prims(self) -> None:
        """Locate sun and earth prims on the stage."""
        if self._stage is None:
            return
        sun_path = f"{self._scene_root}/Sun/sun"
        earth_path = f"{self._scene_root}/Earth"
        sun = self._stage.GetPrimAtPath(sun_path)
        earth = self._stage.GetPrimAtPath(earth_path)
        if sun.IsValid():
            self._sun_prim = UsdGeom.Xformable(sun)
        if earth.IsValid():
            self._earth_prim = UsdGeom.Xformable(earth)

    def create_sdg_camera(self, cfg: SDGCameraConf) -> str:
        """Create a camera prim for SDG capture.

        Returns:
            USD path of the camera prim.
        """
        if not _HAS_USD or self._stage is None:
            return ""

        xform_path = f"{self._scene_root}/Camera"
        cam_path = f"{xform_path}/{cfg.camera_path.split('/')[-1]}"

        # Create container Xform
        self._stage.DefinePrim(xform_path, "Xform")
        self._camera_xform = UsdGeom.Xformable(self._stage.GetPrimAtPath(xform_path))

        # Create camera
        camera = UsdGeom.Camera.Define(self._stage, cam_path)
        camera.CreateFocalLengthAttr().Set(cfg.focal_length)
        camera.CreateFocusDistanceAttr().Set(cfg.focus_distance)
        camera.CreateHorizontalApertureAttr().Set(cfg.horizontal_aperture)
        camera.CreateVerticalApertureAttr().Set(cfg.vertical_aperture)
        camera.CreateFStopAttr().Set(cfg.fstop)
        camera.CreateClippingRangeAttr().Set(
            Gf.Vec2f(cfg.clipping_range[0], cfg.clipping_range[1])
        )

        # Set camera offset: 20cm above ground, looking forward
        cam_xformable = UsdGeom.Xformable(camera.GetPrim())
        cam_xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.2))
        cam_xformable.AddOrientOp().Set(Gf.Quatd(0.5, -0.5, -0.5, 0.5))

        logger.info("SDG camera created at %s", cam_path)
        return cam_path

    def randomize_sun(self) -> None:
        """Randomize sun direction (azimuth 0-360, elevation 20-90)."""
        if self._sun_prim is None:
            return
        theta = self._rng.uniform(0, 360)
        phi = self._rng.uniform(20, 90)
        R = SSTR.from_euler("xyz", (phi, 0, theta), degrees=True)
        quat = R.as_quat()  # (x, y, z, w)
        self._set_xform(self._sun_prim, (0, 0, 0), quat)

    def randomize_earth(self) -> None:
        """Randomize earth position on the sky dome."""
        if self._earth_prim is None:
            return
        r = 348000
        theta = self._rng.uniform(0, 360)
        phi = self._rng.uniform(15, 55)
        x = np.cos(np.radians(theta)) * r
        y = np.sin(np.radians(theta)) * r
        z = np.cos(np.radians(phi)) * r
        rot = self._rng.uniform(0, 360)
        R = SSTR.from_euler("xyz", (0, 0, rot), degrees=True)
        quat = R.as_quat()
        self._set_xform(self._earth_prim, (x, y, z), quat)

    def randomize_camera(self) -> None:
        """Place camera at a random valid position on the terrain."""
        if self._camera_xform is None or self._dem is None:
            return

        h, w = self._dem.shape
        res = self._resolution

        # Sample a valid position from mask
        for _ in range(100):
            xi = self._rng.integers(0, w)
            yi = self._rng.integers(0, h)
            if self._mask is not None and self._mask[yi, xi] <= 0:
                continue
            break

        x = xi * res
        y = (h - yi) * res
        z = float(self._dem[yi, xi])

        # Random yaw
        yaw = self._rng.uniform(0, 2 * np.pi)
        R = SSTR.from_euler("z", yaw)
        quat = R.as_quat()  # (x, y, z, w)

        self._set_xform(self._camera_xform, (x, y, z), quat)

    @staticmethod
    def _set_xform(xformable, position, quat_xyzw) -> None:
        """Set position and orientation on a UsdGeom.Xformable."""
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            name = op.GetOpName()
            if "translate" in name:
                op.Set(Gf.Vec3d(*position))
            elif "orient" in name:
                op.Set(Gf.Quatd(float(quat_xyzw[3]), float(quat_xyzw[0]),
                                float(quat_xyzw[1]), float(quat_xyzw[2])))


class SDGSimulationManager:
    """Manages the SDG simulation loop.

    Usage:
        manager = SDGSimulationManager(cfg)
        manager.setup()
        manager.run()     # Blocks until num_images captured
        manager.shutdown()
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._running = False
        self._environment = None
        self._labeling: Optional[AutonomousLabeling] = None
        self._randomizer: Optional[SceneRandomizer] = None
        self._simulation_app = None

        # Extract SDG-specific config
        mode_cfg = cfg.get("mode", {})
        gen_cfg = mode_cfg.get("generation_settings", {})
        cam_cfg = mode_cfg.get("camera_settings", {})

        if isinstance(gen_cfg, dict):
            self._gen_conf = AutoLabelingConf(**gen_cfg)
        elif isinstance(gen_cfg, AutoLabelingConf):
            self._gen_conf = gen_cfg
        else:
            self._gen_conf = AutoLabelingConf()

        if isinstance(cam_cfg, dict):
            self._cam_conf = SDGCameraConf(**cam_cfg)
        elif isinstance(cam_cfg, SDGCameraConf):
            self._cam_conf = cam_cfg
        else:
            self._cam_conf = SDGCameraConf()

        # Randomization intervals
        self._rock_interval = 100
        self._terrain_interval = 1000

    def setup(self) -> None:
        """Initialize Isaac Sim, environment, camera, and labeling system."""
        from isaacsim import SimulationApp
        from isaacsim.core.api.world import World

        # Launch SimulationApp
        self._simulation_app = SimulationApp({"headless": True})
        self._simulation_app.update()

        # Create world
        self._world = World(stage_units_in_meters=1.0)
        self._world.reset()

        # Create and load environment
        from core.simulation_manager import _environment_registry
        env_name = self._cfg.get("name", "LunarYard")
        if env_name in _environment_registry:
            import omni
            from core.enums import SimulatorMode
            stage = omni.usd.get_context().get_stage()
            env_class = _environment_registry[env_name]
            self._environment = env_class(stage=stage, mode=SimulatorMode.SDG, cfg=self._cfg)
            self._environment.build_scene()
            self._environment.load()
            logger.info("SDG environment '%s' loaded", env_name)

        # Warm up physics
        for _ in range(100):
            self._world.step(render=True)

        # Create SDG camera
        stage = omni.usd.get_context().get_stage()
        scene_root = getattr(self._environment, "_root_path", "/LunarYard")
        self._randomizer = SceneRandomizer(
            stage=stage,
            scene_root=scene_root,
            terrain_resolution=self._cfg.get("resolution", 0.02),
            seed=self._cfg.get("seed", 42),
        )
        self._randomizer.find_prims()
        self._randomizer.create_sdg_camera(self._cam_conf)

        # Pass terrain data to randomizer
        if self._environment is not None:
            tm = getattr(self._environment, "_terrain_manager", None)
            if tm is not None:
                dem = tm.get_dem()
                mask = tm.get_mask()
                if dem is not None:
                    self._randomizer.set_terrain_data(dem, mask)

        # Resolve prim_path relative to scene root
        self._gen_conf.prim_path = f"{scene_root}/{self._gen_conf.prim_path}"

        # Initialize labeling
        self._labeling = AutonomousLabeling(self._gen_conf)
        self._labeling.load()

        # Initial randomization
        self._randomize_all()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("SDG setup complete, will generate %d images", self._gen_conf.num_images)

    def run(self) -> None:
        """Run the SDG capture loop."""
        self._running = True
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        count = 0
        while self._running and count < self._gen_conf.num_images:
            self._world.step(render=True)

            if self._world.is_playing():
                try:
                    self._labeling.record()
                    count += 1
                except Exception as e:
                    logger.warning("Record failed at frame %d: %s", count, e)
                    continue

                # Periodic randomization
                if count % self._rock_interval == 0 and self._environment is not None:
                    rock_mgr = getattr(self._environment, "_rock_manager", None)
                    if rock_mgr is not None:
                        rock_mgr.randomize()

                if count % self._terrain_interval == 0 and self._environment is not None:
                    self._environment.reset()
                    tm = getattr(self._environment, "_terrain_manager", None)
                    if tm is not None:
                        dem = tm.get_dem()
                        mask = tm.get_mask()
                        if dem is not None:
                            self._randomizer.set_terrain_data(dem, mask)

                self._randomize_all()

                if count % 100 == 0:
                    logger.info("SDG progress: %d / %d images", count, self._gen_conf.num_images)

        timeline.stop()
        logger.info("SDG complete: %d images captured to %s", count, self._labeling.data_dir)

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._running = False
        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            except Exception:
                pass
        logger.info("SDG shutdown complete")

    def _randomize_all(self) -> None:
        """Randomize sun, earth, and camera for the next frame."""
        if self._randomizer is not None:
            self._randomizer.randomize_sun()
            self._randomizer.randomize_earth()
            self._randomizer.randomize_camera()

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Signal %d received, stopping SDG", signum)
        self._running = False
