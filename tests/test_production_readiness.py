"""Tests for production config, projects, and I/O reliability."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig, load_config, save_config, set_config
from src.image_processing.compositor import Compositor
from src.image_processing.template_cache import TemplateCache
from src.persistence.project_store import ProjectError, ProjectStore
from src.utils.image_loader import ImageLoadError, ImageLoader
from test_mesh_geometry import synthetic_phone


class ConfigTests(unittest.TestCase):
    def test_round_trip_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            cfg = AppConfig(export_quality=88, preview_max=1000, log_level="WARNING")
            save_config(cfg, path)
            loaded = load_config(path)
            self.assertEqual(loaded.export_quality, 88)
            self.assertEqual(loaded.preview_max, 1000)
            self.assertEqual(loaded.log_level, "WARNING")
            set_config(AppConfig())


class ImageLoaderReliabilityTests(unittest.TestCase):
    def test_rejects_empty_and_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.png"
            empty.write_bytes(b"")
            with self.assertRaises(ImageLoadError):
                ImageLoader.load_image(empty)

            bad = Path(tmp) / "bad.png"
            bad.write_bytes(b"not-a-png")
            with self.assertRaises(ImageLoadError):
                ImageLoader.load_image(bad)

    def test_unique_path_and_save_ex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "out.png"
            img = np.full((20, 10, 3), 40, np.uint8)
            ok, err = ImageLoader.save_image_ex(img, first, quality=90)
            self.assertTrue(ok, err)
            second = ImageLoader.unique_path(first)
            self.assertNotEqual(second, first)
            self.assertFalse(second.exists())


class ProjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = TemplateCache(self.root / "templates")
        self.compositor = Compositor(template_cache=self.cache)
        phone = synthetic_phone()
        self.assertTrue(self.compositor.set_phone_image(phone))
        self.phone_path = self.root / "phone.png"
        ok, buf = cv2.imencode(".png", phone)
        self.assertTrue(ok)
        self.phone_path.write_bytes(buf.tobytes())

        design = np.full((100, 50, 3), (30, 120, 200), np.uint8)
        self.design_path = self.root / "design.png"
        ok, buf = cv2.imencode(".png", design)
        self.assertTrue(ok)
        self.design_path.write_bytes(buf.tobytes())
        self.compositor.set_design_image(design)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_load_preserves_settings_and_paths(self) -> None:
        self.compositor.update_settings({"design_scale": 123.0, "opacity": 90.0})
        project = self.root / "session.pcms"
        ProjectStore.save(
            project, self.compositor, self.phone_path, self.design_path
        )
        raw = json.loads(project.read_text(encoding="utf-8"))
        self.assertEqual(raw["format_version"], 1)
        self.assertIn("metadata", raw)

        other = Compositor(template_cache=TemplateCache(self.root / "tpl2"))
        document, phone, design = ProjectStore.load(project, other)
        self.assertEqual(Path(phone), self.phone_path)
        self.assertEqual(Path(design), self.design_path)
        self.assertAlmostEqual(other.settings["design_scale"], 123.0, places=1)
        self.assertEqual(document.material_name, other.material_name)

    def test_missing_phone_raises(self) -> None:
        project = self.root / "broken.pcms"
        ProjectStore.save(
            project, self.compositor, self.root / "missing.png", self.design_path
        )
        other = Compositor(template_cache=TemplateCache(self.root / "tpl3"))
        with self.assertRaises(ProjectError):
            ProjectStore.load(project, other)

    def test_autosave_writes_file(self) -> None:
        # Point autosave into temp via config override.
        cfg = AppConfig(autosave_dir=str(self.root / "autosave"))
        set_config(cfg)
        path = ProjectStore.autosave(
            self.compositor, self.phone_path, self.design_path
        )
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        set_config(AppConfig())


if __name__ == "__main__":
    unittest.main()
