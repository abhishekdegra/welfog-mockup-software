"""Regression tests for the Smart Cover Surface Engine and template cache."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.image_processing.compositor import Compositor
from src.image_processing.cover_surface import CoverSurfaceEngine
from src.image_processing.mesh import ControlMesh
from src.image_processing.template_cache import TemplateCache
from test_mesh_geometry import synthetic_phone


class CoverSurfaceEngineTests(unittest.TestCase):
    def test_cover_surface_is_inside_phone_and_excludes_hardware(self) -> None:
        phone = synthetic_phone()
        engine = CoverSurfaceEngine(
            TemplateCache(Path(tempfile.mkdtemp(prefix="cover-tpl-")))
        )
        surface = engine.analyze(phone, use_templates=False)

        self.assertIsNotNone(surface.cover_mask)
        self.assertIsNotNone(surface.phone_mask)
        self.assertIsNotNone(surface.printable_mask)
        self.assertGreater(surface.confidence, 0.3)
        self.assertGreater(surface.corner_radius_percent, 0.0)

        # Printable cover must stay inside the phone reference, and the final
        # printable mask must already exclude hardware cutouts.
        phone_pixels = surface.phone_mask > 0
        cover_pixels = surface.cover_mask > 0
        printable_pixels = surface.printable_mask > 0
        exclusion_core = surface.exclusion_mask > 96
        self.assertGreater(int(np.count_nonzero(cover_pixels)), 0)
        self.assertEqual(
            int(np.count_nonzero(cover_pixels & ~phone_pixels)), 0
        )
        self.assertEqual(
            int(np.count_nonzero(printable_pixels & ~cover_pixels)), 0
        )
        self.assertEqual(
            int(np.count_nonzero(printable_pixels & exclusion_core)), 0
        )

        # Adaptive snap can sit on the outer rim; allow a 2px halo.
        dilated = cv2.dilate(
            surface.printable_mask, np.ones((5, 5), np.uint8), iterations=1
        )
        for point in surface.mesh.boundary_points():
            x = int(np.clip(round(point[0]), 0, phone.shape[1] - 1))
            y = int(np.clip(round(point[1]), 0, phone.shape[0] - 1))
            self.assertGreater(dilated[y, x], 0)

        centers = [(125, 125), (190, 130), (135, 210), (198, 220)]
        self.assertTrue(
            all(surface.exclusion_mask[y, x] > 150 for x, y in centers)
        )

    def test_smart_fit_returns_rotation_and_margins(self) -> None:
        phone = synthetic_phone()
        compositor = Compositor(
            TemplateCache(Path(tempfile.mkdtemp(prefix="cover-tpl-")))
        )
        design = np.full((700, 350, 3), (40, 90, 210), np.uint8)
        self.assertTrue(compositor.set_phone_image(phone))
        compositor.set_design_image(design)
        settings = compositor.get_settings()
        self.assertIn("rotation", settings)
        self.assertIn("region_inset", settings)
        self.assertIn("corner_radius", settings)
        self.assertGreaterEqual(settings["corner_radius"], 0.0)

    def test_template_cache_reuses_manual_correction(self) -> None:
        phone = synthetic_phone()
        cache_dir = Path(tempfile.mkdtemp(prefix="cover-tpl-"))
        cache = TemplateCache(cache_dir)
        engine = CoverSurfaceEngine(cache)

        first = engine.analyze(phone, use_templates=False)
        edited = first.mesh.copy()
        edited.points[edited.index(0, 0)] += (12.0, 8.0)
        engine.remember_correction(
            phone, edited, first.exclusion_mask,
            margin_percent=first.margin_percent,
            corner_radius_percent=first.corner_radius_percent,
        )

        reused = engine.analyze(phone, use_templates=True)
        self.assertTrue(reused.from_template)
        self.assertIsNotNone(reused.template_id)
        np.testing.assert_allclose(
            reused.mesh.points[reused.mesh.index(0, 0)],
            edited.points[edited.index(0, 0)],
            atol=1.5,
        )

        # Compositor should also pick up the template on a fresh instance.
        compositor = Compositor(TemplateCache(cache_dir))
        self.assertTrue(compositor.set_phone_image(phone))
        self.assertTrue(compositor.from_template)
        compositor.set_design_image(
            np.full((500, 300, 3), (30, 100, 210), np.uint8)
        )
        output = compositor.export()
        self.assertIsNotNone(output)
        self.assertEqual(output.shape[:2], phone.shape[:2])

    def test_manual_edit_writes_template_json(self) -> None:
        phone = synthetic_phone()
        cache_dir = Path(tempfile.mkdtemp(prefix="cover-tpl-"))
        compositor = Compositor(TemplateCache(cache_dir))
        compositor.set_phone_image(phone)
        mesh = compositor.get_control_mesh()
        self.assertIsNotNone(mesh)
        mesh.points[mesh.index(1, 1)] += (6.0, -4.0)
        compositor.set_control_mesh(mesh)
        self.assertTrue(any(cache_dir.glob("*.json")))

    def test_wrap_target_not_clipped_to_raw_stairs(self) -> None:
        """Smooth phone rim must not be clipped back onto a jagged raw mask."""
        phone = np.zeros((200, 120), dtype=np.uint8)
        cv2.rectangle(phone, (20, 20), (100, 180), 255, -1)
        phone[20:28, 20:28] = 0
        gate = CoverSurfaceEngine.wrap_target_mask(None, phone)
        self.assertIsNotNone(gate)
        self.assertGreaterEqual(int(np.count_nonzero(gate[20:28, 20:28])), 12)


if __name__ == "__main__":
    unittest.main()
