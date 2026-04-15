"""Tests for Synthetic Data Generation module."""

import os
import tempfile

import numpy as np
import pytest

from data_generation.config import AutoLabelingConf, SDGCameraConf
from data_generation.writers import (
    BaseWriter,
    WriteRGBData,
    WriteDepthData,
    WritePoseData,
    WriteSemanticData,
    WriteInstanceData,
    WriteIRData,
    create_writer,
)


# ── Config validation ───────────────────────────────────────────────────────

class TestAutoLabelingConf:
    def test_defaults(self):
        conf = AutoLabelingConf()
        assert conf.num_images == 1000
        assert conf.camera_names == ["camera_annotations"]
        assert conf.element_per_folder == 1000

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError, match="camera_resolutions length"):
            AutoLabelingConf(
                camera_names=["cam1", "cam2"],
                camera_resolutions=[[640, 480]],
            )

    def test_invalid_annotator_raises(self):
        with pytest.raises(AssertionError, match="Unknown annotator"):
            AutoLabelingConf(annotators_list=[["rgb", "invalid_type"]])

    def test_invalid_image_format_raises(self):
        with pytest.raises(AssertionError, match="Unknown image format"):
            AutoLabelingConf(image_formats=["bmp"])

    def test_invalid_annot_format_raises(self):
        with pytest.raises(AssertionError, match="Unknown annot format"):
            AutoLabelingConf(annot_formats=["xml"])

    def test_zero_images_raises(self):
        with pytest.raises(AssertionError, match="num_images must be positive"):
            AutoLabelingConf(num_images=0)

    def test_multi_camera_valid(self):
        conf = AutoLabelingConf(
            camera_names=["left", "right"],
            camera_resolutions=[[1280, 720], [640, 480]],
            annotators_list=[["rgb"], ["depth"]],
            image_formats=["png", "png"],
            annot_formats=["json", "json"],
        )
        assert len(conf.camera_names) == 2


class TestSDGCameraConf:
    def test_defaults(self):
        conf = SDGCameraConf()
        assert conf.focal_length == pytest.approx(1.93)
        assert conf.clipping_range == (0.01, 1000000.0)

    def test_list_to_tuple_conversion(self):
        conf = SDGCameraConf(clipping_range=[0.1, 500.0])
        assert isinstance(conf.clipping_range, tuple)

    def test_invalid_focal_length(self):
        with pytest.raises(AssertionError):
            SDGCameraConf(focal_length=-1.0)


# ── Writer factory ──────────────────────────────────────────────────────────

class TestWriterFactory:
    def test_create_rgb_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = create_writer("rgb", root_path=tmpdir, prefix="cam1_")
            assert isinstance(writer, WriteRGBData)

    def test_create_depth_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = create_writer("depth", root_path=tmpdir, prefix="cam1_")
            assert isinstance(writer, WriteDepthData)

    def test_create_pose_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = create_writer("pose", root_path=tmpdir, prefix="cam1_")
            assert isinstance(writer, WritePoseData)

    def test_create_semantic_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = create_writer("semantic_segmentation", root_path=tmpdir, prefix="cam_")
            assert isinstance(writer, WriteSemanticData)

    def test_create_instance_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = create_writer("instance_segmentation", root_path=tmpdir, prefix="cam_")
            assert isinstance(writer, WriteInstanceData)

    def test_create_ir_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = create_writer("ir", root_path=tmpdir, prefix="cam_")
            assert isinstance(writer, WriteIRData)

    def test_unknown_writer_raises(self):
        with pytest.raises(ValueError, match="Unknown writer"):
            create_writer("unknown_type", root_path="/tmp")


# ── Pose writer (CSV append, no Isaac dependency) ───────────────────────────

class TestWritePoseData:
    def test_write_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WritePoseData(root_path=tmpdir, prefix="cam_")
            data = {
                "position_x": [1.0],
                "position_y": [2.0],
                "position_z": [3.0],
                "quaternion_x": [0.0],
                "quaternion_y": [0.0],
                "quaternion_z": [0.0],
                "quaternion_w": [1.0],
            }
            writer.write(data)
            writer.write(data)

            csv_path = os.path.join(tmpdir, "cam_pose", "pose.csv")
            assert os.path.exists(csv_path)

            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert float(rows[0]["position_x"]) == pytest.approx(1.0)

    def test_counter_increments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WritePoseData(root_path=tmpdir, prefix="")
            data = {"x": [0.0], "y": [0.0], "z": [0.0]}
            writer.write(data)
            assert writer.counter == 1
            writer.write(data)
            assert writer.counter == 2


# ── RGB writer (folder partitioning) ────────────────────────────────────────

class TestWriteRGBData:
    def test_folder_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WriteRGBData(root_path=tmpdir, prefix="cam_", element_per_folder=3)
            # Simulate 4 RGBA images
            for _ in range(4):
                img = np.zeros((480, 640, 4), dtype=np.uint8)
                writer.write(img)

            # Should have 2 folders: folder 0 (3 images) and folder 1 (1 image)
            folder_0 = os.path.join(tmpdir, "cam_rgb", "0")
            folder_1 = os.path.join(tmpdir, "cam_rgb", "1")
            assert os.path.isdir(folder_0)
            assert os.path.isdir(folder_1)
            assert len(os.listdir(folder_0)) == 3
            assert len(os.listdir(folder_1)) == 1


# ── Depth writer ────────────────────────────────────────────────────────────

class TestWriteDepthData:
    def test_writes_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WriteDepthData(root_path=tmpdir, prefix="cam_", element_per_folder=10)
            depth = np.random.rand(480, 640).astype(np.float32)
            writer.write(depth)

            folder = os.path.join(tmpdir, "cam_depth", "0")
            assert os.path.isdir(folder)
            files = os.listdir(folder)
            assert len(files) == 1
            assert files[0].endswith(".npz")

            loaded = np.load(os.path.join(folder, files[0]))
            assert "depth" in loaded


# ── SceneRandomizer (no USD, logic tests only) ─────────────────────────────

class TestSceneRandomizer:
    def test_set_terrain_data(self):
        from data_generation.sdg_manager import SceneRandomizer

        randomizer = SceneRandomizer(stage=None, seed=42)
        dem = np.random.rand(100, 100).astype(np.float32)
        mask = np.ones((100, 100), dtype=np.uint8)
        randomizer.set_terrain_data(dem, mask)
        assert randomizer._dem is not None
        assert randomizer._mask is not None

    def test_randomize_without_stage_is_noop(self):
        from data_generation.sdg_manager import SceneRandomizer

        randomizer = SceneRandomizer(stage=None)
        # Should not raise
        randomizer.randomize_sun()
        randomizer.randomize_earth()
        randomizer.randomize_camera()


# ── SDGSimulationManager config ─────────────────────────────────────────────

class TestSDGSimulationManagerConfig:
    def test_default_config_parsing(self):
        from data_generation.sdg_manager import SDGSimulationManager

        cfg = {
            "mode": {
                "name": "SDG",
                "generation_settings": {
                    "num_images": 500,
                    "prim_path": "Camera",
                    "camera_names": ["cam1"],
                    "camera_resolutions": [[1280, 720]],
                    "data_dir": "/tmp/sdg_test",
                    "annotators_list": [["rgb", "depth"]],
                    "image_formats": ["png"],
                    "annot_formats": ["json"],
                    "element_per_folder": 100,
                    "save_intrinsics": False,
                },
                "camera_settings": {
                    "camera_path": "Camera/cam1",
                    "focal_length": 2.5,
                },
            },
            "seed": 123,
        }
        mgr = SDGSimulationManager(cfg)
        assert mgr._gen_conf.num_images == 500
        assert mgr._gen_conf.camera_names == ["cam1"]
        assert mgr._cam_conf.focal_length == pytest.approx(2.5)

    def test_empty_mode_uses_defaults(self):
        from data_generation.sdg_manager import SDGSimulationManager

        mgr = SDGSimulationManager({"mode": {}})
        assert mgr._gen_conf.num_images == 1000
        assert mgr._cam_conf.focal_length == pytest.approx(1.93)
