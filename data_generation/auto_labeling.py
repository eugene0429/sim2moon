"""Autonomous labeling engine for synthetic data generation.

Uses Omniverse Replicator annotators to capture RGB, depth,
semantic/instance segmentation, IR, and camera pose data.

Workflow:
    1. AutonomousLabeling(cfg) — configure cameras and annotators
    2. labeling.load() — find camera prims, create render products, attach annotators
    3. labeling.record() — capture one frame of data from all cameras

Ported from OmniLRS src/labeling/auto_label.py with improved structure.
"""

import logging
import os
import random
import string
from typing import Dict, List, Tuple

import numpy as np

from data_generation.config import AutoLabelingConf
from data_generation.writers import create_writer

logger = logging.getLogger(__name__)

try:
    import omni
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom
    _HAS_REPLICATOR = True
except ImportError:
    _HAS_REPLICATOR = False


class PoseAnnotator:
    """Extracts camera pose (position + quaternion) from USD prim transform."""

    def __init__(self, prim) -> None:
        self._prim = prim
        self._xform = UsdGeom.Xformable(prim)

    def get_data(self) -> Dict[str, List[float]]:
        time = Usd.TimeCode.Default()
        world_transform = self._xform.ComputeLocalToWorldTransform(time)
        t = world_transform.ExtractTranslation()
        r = world_transform.ExtractRotationQuat()
        return {
            "position_x": [t[0]],
            "position_y": [t[1]],
            "position_z": [t[2]],
            "quaternion_x": [r.GetImaginary()[0]],
            "quaternion_y": [r.GetImaginary()[1]],
            "quaternion_z": [r.GetImaginary()[2]],
            "quaternion_w": [r.GetReal()],
        }


class AutonomousLabeling:
    """Manages annotators, render products, and data writers for SDG.

    Supports annotators: rgb, depth, ir, semantic_segmentation,
    instance_segmentation, pose.

    Each camera gets its own render product and set of annotators.
    Data is written to disk using the writer framework in writers.py.
    """

    def __init__(self, cfg: AutoLabelingConf) -> None:
        self._cfg = cfg

        # Unique output directory per run
        data_hash = "".join(random.sample(string.ascii_letters + string.digits, 16))
        self._data_dir = os.path.join(cfg.data_dir, data_hash)
        os.makedirs(self._data_dir, exist_ok=True)

        self._camera_prims: Dict[str, object] = {}
        self._camera_paths: Dict[str, str] = {}
        self._render_products: Dict[str, object] = {}
        self._annotators: Dict[str, Tuple[str, str, object]] = {}
        self._writers: Dict[str, Dict[str, object]] = {}

        # Pre-build writer configs
        self._init_writers()

        # Annotator enabler registry
        self._enablers = {
            "rgb": self._enable_rgb,
            "depth": self._enable_depth,
            "ir": self._enable_ir,
            "semantic_segmentation": self._enable_semantic,
            "instance_segmentation": self._enable_instance,
            "pose": self._enable_pose,
        }

        self._frame_count = 0
        logger.info("AutonomousLabeling: output dir = %s", self._data_dir)

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ── Initialization ──────────────────────────────────────────────────

    def _init_writers(self) -> None:
        """Pre-create writers for each camera × annotator pair."""
        cfg = self._cfg
        for i, cam_name in enumerate(cfg.camera_names):
            self._writers[cam_name] = {}
            for ann_name in cfg.annotators_list[i]:
                self._writers[cam_name][ann_name] = create_writer(
                    ann_name,
                    root_path=self._data_dir,
                    prefix=cam_name + "_",
                    element_per_folder=cfg.element_per_folder,
                    image_format=cfg.image_formats[i],
                    annot_format=cfg.annot_formats[i],
                )

    def load(self) -> None:
        """Find cameras in scene, create render products, attach annotators.

        Must be called after the USD stage is ready.
        """
        if not _HAS_REPLICATOR:
            raise RuntimeError("Omniverse Replicator not available")

        stage = omni.usd.get_context().get_stage()
        meta_prim = stage.GetPrimAtPath(self._cfg.prim_path)

        # Discover cameras
        self._find_cameras(meta_prim)

        cfg = self._cfg
        for cam_name in cfg.camera_names:
            # Save intrinsics
            if cfg.save_intrinsics:
                self._save_intrinsics(cam_name)

            # Create render product
            idx = cfg.camera_names.index(cam_name)
            res = tuple(cfg.camera_resolutions[idx])
            self._render_products[cam_name] = rep.create.render_product(
                self._camera_paths[cam_name], res,
            )

            # Enable annotators
            for ann_name in cfg.annotators_list[idx]:
                self._enablers[ann_name](cam_name)

        logger.info(
            "AutonomousLabeling: loaded %d cameras, %d annotators",
            len(self._camera_prims), len(self._annotators),
        )

    def _find_cameras(self, meta_prim) -> None:
        """Find camera prims matching configured names."""
        for prim in Usd.PrimRange(meta_prim):
            if prim.GetName() in self._cfg.camera_names:
                self._camera_prims[prim.GetName()] = prim
                self._camera_paths[prim.GetName()] = str(prim.GetPath())

        missing = set(self._cfg.camera_names) - set(self._camera_prims.keys())
        if missing:
            raise ValueError(f"Cameras not found in scene: {missing}")

    # ── Intrinsics ──────────────────────────────────────────────────────

    def get_intrinsics_matrix(self, camera_prim, width: int = 1280, height: int = 720) -> np.ndarray:
        """Compute 3x3 camera intrinsics matrix from USD camera attributes."""
        focal_length = camera_prim.GetAttribute("focalLength").Get() / 10.0
        h_aperture = camera_prim.GetAttribute("horizontalAperture").Get() / 10.0
        v_aperture = h_aperture * (float(height) / width)
        fx = width * focal_length / h_aperture
        fy = height * focal_length / v_aperture
        cx = width * 0.5
        cy = height * 0.5
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def _save_intrinsics(self, camera_name: str) -> None:
        """Save camera intrinsics to CSV and NPY."""
        idx = self._cfg.camera_names.index(camera_name)
        res = self._cfg.camera_resolutions[idx]
        K = self.get_intrinsics_matrix(self._camera_prims[camera_name], res[0], res[1])
        np.savetxt(os.path.join(self._data_dir, f"{camera_name}_intrinsics.csv"), K, delimiter=",")
        np.save(os.path.join(self._data_dir, f"{camera_name}_intrinsics.npy"), K)
        logger.info("Saved intrinsics for '%s'", camera_name)

    # ── Annotator enablers ──────────────────────────────────────────────

    def _enable_rgb(self, cam_name: str) -> None:
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([self._render_products[cam_name]])
        self._annotators[f"{cam_name}_rgb"] = (cam_name, "rgb", ann)

    def _enable_depth(self, cam_name: str) -> None:
        ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        ann.attach([self._render_products[cam_name]])
        self._annotators[f"{cam_name}_depth"] = (cam_name, "depth", ann)

    def _enable_ir(self, cam_name: str) -> None:
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([self._render_products[cam_name]])
        self._annotators[f"{cam_name}_ir"] = (cam_name, "ir", ann)

    def _enable_semantic(self, cam_name: str) -> None:
        ann = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": True},
        )
        ann.attach([self._render_products[cam_name]])
        self._annotators[f"{cam_name}_semantic_segmentation"] = (
            cam_name, "semantic_segmentation", ann,
        )

    def _enable_instance(self, cam_name: str) -> None:
        ann = rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"colorize": True},
        )
        ann.attach([self._render_products[cam_name]])
        self._annotators[f"{cam_name}_instance_segmentation"] = (
            cam_name, "instance_segmentation", ann,
        )

    def _enable_pose(self, cam_name: str) -> None:
        pose_ann = PoseAnnotator(self._camera_prims[cam_name])
        self._annotators[f"{cam_name}_pose"] = (cam_name, "pose", pose_ann)

    # ── Recording ───────────────────────────────────────────────────────

    def record(self) -> None:
        """Capture one frame of data from all annotators and write to disk."""
        for cam_name, ann_name, annotator in self._annotators.values():
            try:
                data = annotator.get_data()
                self._writers[cam_name][ann_name].write(data)
            except Exception as e:
                logger.warning("Failed to record %s/%s: %s", cam_name, ann_name, e)
        self._frame_count += 1
