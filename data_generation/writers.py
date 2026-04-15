"""Data writers for synthetic data output.

Each writer handles a specific data type (RGB, depth, segmentation, pose, IR)
and manages folder partitioning for large datasets.

Ported from OmniLRS src/labeling/rep_utils.py with improved structure.
"""

import json
import logging
import os
from typing import Any, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseWriter:
    """Base class for all data writers with folder partitioning."""

    def __init__(
        self,
        root_path: str,
        name: str = "",
        element_per_folder: int = 1000,
        prefix: str = "",
        **kwargs,
    ) -> None:
        self.root_path = root_path
        self.data_path = os.path.join(root_path, prefix + name)
        self.element_per_folder = element_per_folder
        self.counter = 0
        self.folder_counter = 0
        self._name_width = len(str(element_per_folder))
        self.current_folder = ""

    def _make_folder(self) -> None:
        """Create a new subfolder when the current one is full."""
        if self.counter % self.element_per_folder == 0:
            self.current_folder = os.path.join(self.data_path, str(self.folder_counter))
            os.makedirs(self.current_folder, exist_ok=True)
            self.counter = 0
            self.folder_counter += 1

    def _filename(self, ext: str) -> str:
        """Generate a zero-padded filename."""
        return f"{self.counter:0{self._name_width}d}.{ext}"

    def write(self, data: Any) -> None:
        raise NotImplementedError


class WriteRGBData(BaseWriter):
    """Write RGBA images as BGR(A) files (png/jpg/tiff)."""

    def __init__(self, image_format: str = "png", **kwargs) -> None:
        super().__init__(name="rgb", **kwargs)
        self.image_format = image_format

    def write(self, data: np.ndarray) -> None:
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape, -1)
        rgb = np.squeeze(rgb)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGRA)
        self._make_folder()
        cv2.imwrite(os.path.join(self.current_folder, self._filename(self.image_format)), rgb)
        self.counter += 1


class WriteIRData(BaseWriter):
    """Write IR images as grayscale files."""

    def __init__(self, image_format: str = "png", **kwargs) -> None:
        super().__init__(name="ir", **kwargs)
        self.image_format = image_format

    def write(self, data: np.ndarray) -> None:
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape, -1)
        rgb = np.squeeze(rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self._make_folder()
        cv2.imwrite(os.path.join(self.current_folder, self._filename(self.image_format)), gray)
        self.counter += 1


class WriteDepthData(BaseWriter):
    """Write depth maps as compressed numpy archives."""

    def __init__(self, **kwargs) -> None:
        super().__init__(name="depth", **kwargs)

    def write(self, data: np.ndarray) -> None:
        depth = np.frombuffer(data, dtype=np.float32).reshape(*data.shape, -1)
        depth = np.squeeze(depth)
        self._make_folder()
        np.savez_compressed(os.path.join(self.current_folder, self._filename("npz")), depth=depth)
        self.counter += 1


class WriteSemanticData(BaseWriter):
    """Write semantic segmentation images + id-to-label JSON."""

    def __init__(self, image_format: str = "png", annot_format: str = "json", **kwargs) -> None:
        super().__init__(name="semantic_segmentation", **kwargs)
        self.image_format = image_format
        self.annot_format = annot_format
        self._folder_image = ""
        self._folder_labels = ""

    def _make_folder(self) -> None:
        if self.counter % self.element_per_folder == 0:
            self._folder_image = os.path.join(self.data_path, str(self.folder_counter))
            self._folder_labels = os.path.join(self.data_path + "_id_label", str(self.folder_counter))
            os.makedirs(self._folder_image, exist_ok=True)
            os.makedirs(self._folder_labels, exist_ok=True)
            self.counter = 0
            self.folder_counter += 1

    def write(self, data: Dict[str, Any]) -> None:
        id_to_labels = data["info"]["idToLabels"]
        sem = np.frombuffer(data["data"], dtype=np.uint8).reshape(*data["data"].shape, -1)
        sem = cv2.cvtColor(np.squeeze(sem), cv2.COLOR_RGBA2BGRA)
        self._make_folder()
        with open(os.path.join(self._folder_labels, self._filename(self.annot_format)), "w") as f:
            json.dump(id_to_labels, f)
        cv2.imwrite(os.path.join(self._folder_image, self._filename(self.image_format)), sem)
        self.counter += 1


class WriteInstanceData(BaseWriter):
    """Write instance segmentation images + id-to-label + id-to-semantic JSON."""

    def __init__(self, image_format: str = "png", annot_format: str = "json", **kwargs) -> None:
        super().__init__(name="instance_segmentation", **kwargs)
        self.image_format = image_format
        self.annot_format = annot_format
        self._folder_image = ""
        self._folder_labels = ""
        self._folder_semantics = ""

    def _make_folder(self) -> None:
        if self.counter % self.element_per_folder == 0:
            self._folder_image = os.path.join(self.data_path, str(self.folder_counter))
            self._folder_labels = os.path.join(self.data_path + "_id_label", str(self.folder_counter))
            self._folder_semantics = os.path.join(self.data_path + "_id_semantic", str(self.folder_counter))
            os.makedirs(self._folder_image, exist_ok=True)
            os.makedirs(self._folder_labels, exist_ok=True)
            os.makedirs(self._folder_semantics, exist_ok=True)
            self.counter = 0
            self.folder_counter += 1

    def write(self, data: Dict[str, Any]) -> None:
        id_to_labels = data["info"]["idToLabels"]
        id_to_semantic = data["info"]["idToSemantics"]
        inst = np.frombuffer(data["data"], dtype=np.uint8).reshape(*data["data"].shape, -1)
        inst = cv2.cvtColor(np.squeeze(inst), cv2.COLOR_RGBA2BGRA)
        self._make_folder()
        with open(os.path.join(self._folder_labels, self._filename(self.annot_format)), "w") as f:
            json.dump(id_to_labels, f)
        with open(os.path.join(self._folder_semantics, self._filename(self.annot_format)), "w") as f:
            json.dump(id_to_semantic, f)
        cv2.imwrite(os.path.join(self._folder_image, self._filename(self.image_format)), inst)
        self.counter += 1


class WritePoseData(BaseWriter):
    """Write camera pose data to CSV (appending each frame)."""

    def __init__(self, annot_format: str = "csv", **kwargs) -> None:
        super().__init__(name="pose", **kwargs)
        self.annot_format = annot_format
        self._path = os.path.join(self.data_path, f"pose.{annot_format}")
        os.makedirs(self.data_path, exist_ok=True)
        self._header_written = False

    def write(self, data: Dict[str, Any]) -> None:
        import csv
        fieldnames = list(data.keys())
        row = {k: v[0] if isinstance(v, list) else v for k, v in data.items()}

        write_header = not os.path.exists(self._path)
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.counter += 1


# ── Writer factory ──────────────────────────────────────────────────────────

_WRITER_REGISTRY: Dict[str, type] = {
    "rgb": WriteRGBData,
    "ir": WriteIRData,
    "depth": WriteDepthData,
    "semantic_segmentation": WriteSemanticData,
    "instance_segmentation": WriteInstanceData,
    "pose": WritePoseData,
}


def create_writer(name: str, **kwargs) -> BaseWriter:
    """Create a data writer by annotator name.

    Args:
        name: Annotator type ('rgb', 'depth', 'semantic_segmentation', etc.).
        **kwargs: Passed to writer constructor (root_path, prefix, element_per_folder, etc.).

    Returns:
        Configured BaseWriter instance.
    """
    if name not in _WRITER_REGISTRY:
        raise ValueError(f"Unknown writer '{name}'. Available: {list(_WRITER_REGISTRY.keys())}")
    return _WRITER_REGISTRY[name](**kwargs)
