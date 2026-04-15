"""Configuration dataclasses for Synthetic Data Generation."""

import dataclasses
from typing import List, Tuple


@dataclasses.dataclass
class SDGCameraConf:
    """Camera parameters for SDG capture."""

    camera_path: str = "Camera/camera_annotations"
    focal_length: float = 1.93
    horizontal_aperture: float = 3.6
    vertical_aperture: float = 2.7
    fstop: float = 0.0
    focus_distance: float = 10.0
    clipping_range: Tuple[float, float] = (0.01, 1000000.0)

    def __post_init__(self):
        assert self.focal_length > 0, "focal_length must be positive"
        assert self.horizontal_aperture > 0, "horizontal_aperture must be positive"
        assert self.vertical_aperture > 0, "vertical_aperture must be positive"
        assert self.fstop >= 0, "fstop must be non-negative"
        assert self.focus_distance > 0, "focus_distance must be positive"
        assert len(self.clipping_range) == 2, "clipping_range must have 2 elements"
        assert self.clipping_range[1] > self.clipping_range[0], (
            "clipping_range far must be > near"
        )
        if isinstance(self.clipping_range, list):
            self.clipping_range = tuple(self.clipping_range)


VALID_ANNOTATORS = {"rgb", "depth", "ir", "semantic_segmentation", "instance_segmentation", "pose"}
VALID_IMAGE_FORMATS = {"png", "jpeg", "jpg", "tiff", "tif"}
VALID_ANNOT_FORMATS = {"json", "csv", "yaml"}


@dataclasses.dataclass
class AutoLabelingConf:
    """Configuration for the autonomous labeling pipeline."""

    num_images: int = 1000
    prim_path: str = "Camera"
    camera_names: List[str] = dataclasses.field(default_factory=lambda: ["camera_annotations"])
    camera_resolutions: List[List[int]] = dataclasses.field(
        default_factory=lambda: [[640, 480]]
    )
    data_dir: str = "data"
    annotators_list: List[List[str]] = dataclasses.field(
        default_factory=lambda: [["rgb", "semantic_segmentation", "instance_segmentation"]]
    )
    image_formats: List[str] = dataclasses.field(default_factory=lambda: ["png"])
    annot_formats: List[str] = dataclasses.field(default_factory=lambda: ["json"])
    element_per_folder: int = 1000
    save_intrinsics: bool = True

    def __post_init__(self):
        n = len(self.camera_names)
        assert n > 0, "camera_names must have at least one element"
        assert len(self.camera_resolutions) == n, (
            f"camera_resolutions length ({len(self.camera_resolutions)}) must match camera_names ({n})"
        )
        assert len(self.annotators_list) == n, (
            f"annotators_list length ({len(self.annotators_list)}) must match camera_names ({n})"
        )
        assert len(self.image_formats) == n, (
            f"image_formats length ({len(self.image_formats)}) must match camera_names ({n})"
        )
        assert len(self.annot_formats) == n, (
            f"annot_formats length ({len(self.annot_formats)}) must match camera_names ({n})"
        )
        assert self.num_images > 0, "num_images must be positive"
        assert self.element_per_folder > 0, "element_per_folder must be positive"
        assert self.data_dir, "data_dir must not be empty"

        for res in self.camera_resolutions:
            assert len(res) == 2, f"camera_resolution must be [w, h], got {res}"
        for annotator_group in self.annotators_list:
            for ann in annotator_group:
                assert ann in VALID_ANNOTATORS, (
                    f"Unknown annotator '{ann}'. Valid: {VALID_ANNOTATORS}"
                )
        for fmt in self.image_formats:
            assert fmt in VALID_IMAGE_FORMATS, (
                f"Unknown image format '{fmt}'. Valid: {VALID_IMAGE_FORMATS}"
            )
        for fmt in self.annot_formats:
            assert fmt in VALID_ANNOT_FORMATS, (
                f"Unknown annot format '{fmt}'. Valid: {VALID_ANNOT_FORMATS}"
            )
