"""Regression checks for Chapter 3.5 precision auto-fit refinements."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.image_processing.cover_surface import CoverSurfaceEngine
from src.image_processing.mesh import AdaptiveMeshBuilder, ControlMesh
from src.image_processing.region_detector import (
    HardwareRegionDetector,
    PrintableRegionDetector,
)
from src.image_processing.smart_fit import SmartFitEstimator
from src.image_processing.template_cache import TemplateManager
from test_mesh_geometry import synthetic_phone


class PrecisionCoverTests(unittest.TestCase):
    def test_cover_bbox_does_not_grow_into_spikes(self) -> None:
        phone = synthetic_phone()
        engine = CoverSurfaceEngine(
            template_manager=TemplateManager(Path(tempfile.mkdtemp()))
        )
        surface = engine.analyze(phone, use_templates=False)
        self.assertIsNotNone(surface.cover_mask)
        self.assertIsNotNone(surface.printable_mask)

        cover = surface.cover_mask > 0
        phone_mask = surface.phone_mask > 0
        # Cover must stay inside the phone reference.
        self.assertEqual(int(np.count_nonzero(cover & ~phone_mask)), 0)

        ys, xs = np.nonzero(cover)
        cover_w = float(xs.max() - xs.min() + 1)
        cover_h = float(ys.max() - ys.min() + 1)
        pys, pxs = np.nonzero(phone_mask)
        phone_w = float(pxs.max() - pxs.min() + 1)
        phone_h = float(pys.max() - pys.min() + 1)
        # No triangular lobe that approaches full phone width from a thin spike.
        self.assertLess(cover_w / phone_w, 1.02)
        self.assertLess(cover_h / phone_h, 1.02)

    def test_mesh_cells_stay_positively_oriented(self) -> None:
        phone = synthetic_phone()
        engine = CoverSurfaceEngine(
            template_manager=TemplateManager(Path(tempfile.mkdtemp()))
        )
        surface = engine.analyze(phone, use_templates=False)
        mesh = AdaptiveMeshBuilder.refine(
            surface.mesh, surface.printable_mask, surface.corner_radius_percent
        )
        grid = mesh.points.reshape(mesh.rows, mesh.cols, 2)
        for row in range(mesh.rows - 1):
            for col in range(mesh.cols - 1):
                tl, tr = grid[row, col], grid[row, col + 1]
                bl, br = grid[row + 1, col], grid[row + 1, col + 1]
                self.assertGreater(float(np.cross(tr - tl, bl - tl)), 0.0)
                self.assertGreater(float(np.cross(br - tr, bl - tr)), 0.0)

    def test_camera_merge_prefers_rounded_stadium(self) -> None:
        mask = np.zeros((200, 120), np.uint8)
        circles = [(30, 40, 12), (30, 70, 12), (48, 55, 6)]
        for x, y, r in circles:
            cv2.circle(mask, (x, y), r, 255, -1)
        HardwareRegionDetector._merge_camera_cluster(
            mask, width=120, top_height=120, circles=circles
        )
        # Island should exist and not fill the whole top band.
        self.assertGreater(int(np.count_nonzero(mask[:120])), 400)
        self.assertLess(int(np.count_nonzero(mask[:120])), 120 * 120 * 0.35)

    def test_fill_autofit_starts_near_full_bleed(self) -> None:
        design = np.full((400, 200, 3), (40, 90, 210), np.uint8)
        content = np.ones((400, 200), np.uint8) * 255
        scale = SmartFitEstimator._initial_scale(
            content, isolated_content=False,
            base_crop_w=200.0, base_crop_h=400.0, fit_mode="fill",
        )
        # Geometry fit uses scale 1.0 — no hardcoded overscan.
        self.assertAlmostEqual(scale, 1.0, places=5)

    def test_geometry_fit_is_centered_with_zero_inset(self) -> None:
        from src.image_processing.compositor import Compositor

        phone = synthetic_phone()
        design = np.full((800, 400, 3), (30, 80, 200), np.uint8)
        compositor = Compositor()
        compositor.set_phone_image(phone)
        compositor.set_design_image(design)
        fit = compositor.auto_fit_design()
        self.assertAlmostEqual(float(fit["offset_x"]), 0.0, places=1)
        self.assertAlmostEqual(float(fit["offset_y"]), 0.0, places=1)
        self.assertAlmostEqual(float(fit["rotation"]), 0.0, places=1)
        self.assertAlmostEqual(float(fit["region_inset"]), 0.0, places=1)
        self.assertAlmostEqual(float(fit["design_scale"]), 100.0, places=0)

    def test_flash_satellite_is_detected_near_lenses(self) -> None:
        phone = synthetic_phone()
        region = PrintableRegionDetector.detect(phone)
        # Bright flash at (198, 220) on the synthetic phone.
        self.assertGreater(int(region.exclusion_mask[220, 198]), 150)
        # Exclusion outlines keep editable finishing dots (circles are denser).
        self.assertTrue(any(len(c) >= 3 for c in region.hardware_contours))
        self.assertTrue(
            any(3 <= len(c.reshape(-1, 2)) <= 48 for c in region.hardware_contours)
        )

    def test_manual_exclusion_contours_rebuild_mask(self) -> None:
        from src.image_processing.compositor import Compositor

        phone = synthetic_phone()
        compositor = Compositor()
        compositor.set_phone_image(phone)
        circle = HardwareRegionDetector._sample_circle(200.0, 140.0, 18.0)
        compositor.set_hardware_exclusions([circle])
        self.assertGreater(int(compositor.exclusion_mask[140, 200]), 150)
        self.assertEqual(int(compositor.exclusion_mask[500, 260]), 0)

    def test_shape_polygons_cover_expected_centres(self) -> None:
        for shape in ("circle", "square", "triangle", "free"):
            poly = HardwareRegionDetector.make_shape_polygon(
                shape, (0.5, 0.4), 0.05
            ).reshape(-1, 2)
            self.assertGreaterEqual(len(poly), 3)
            if shape == "circle":
                self.assertGreaterEqual(len(poly), 24)
            else:
                self.assertLessEqual(len(poly), 8)
            center = poly.mean(axis=0)
            self.assertAlmostEqual(float(center[0]), 0.5, delta=0.08)
            self.assertAlmostEqual(float(center[1]), 0.4, delta=0.12)

    def test_perfect_finish_makes_round_and_stadium_curves(self) -> None:
        # Jagged almost-circle (flash) + lumpy island (camera module).
        flash = np.array(
            [[50, 40], [58, 42], [60, 50], [55, 58], [45, 57], [40, 48]],
            np.float32,
        )
        island = np.array(
            [
                [20, 20], [45, 18], [48, 55], [46, 90], [22, 92],
                [15, 70], [14, 40],
            ],
            np.float32,
        )
        button = np.array(
            [[100, 10], [112, 12], [114, 60], [110, 108], [102, 110], [96, 55]],
            np.float32,
        )
        finished = HardwareRegionDetector.perfect_finish_contours(
            [flash, island, button], phone_image=None
        )
        self.assertEqual(len(finished), 3)
        # Flash becomes a dense true circle.
        self.assertGreaterEqual(len(finished[0]), 24)
        circ = finished[0].reshape(-1, 2)
        center = circ.mean(axis=0)
        radii = np.linalg.norm(circ - center, axis=1)
        self.assertLess(float(radii.std()), 1.5)
        # Camera island → smooth rounded rect.
        stadium = finished[1].reshape(-1, 2)
        self.assertGreaterEqual(len(stadium), 16)
        # Side button → capsule with many corner samples.
        pill = finished[2].reshape(-1, 2)
        self.assertGreaterEqual(len(pill), 16)
        kind, _ = HardwareRegionDetector._classify_cutout(pill)
        self.assertIn(kind, ("stadium", "rounded_rect"))

    def test_perfect_finish_snaps_mesh_to_cover(self) -> None:
        from src.image_processing.compositor import Compositor

        phone = synthetic_phone()
        compositor = Compositor()
        compositor.set_phone_image(phone)
        before = compositor.get_control_mesh()
        self.assertIsNotNone(before)
        count = compositor.perfect_finish_cutouts()
        after = compositor.get_control_mesh()
        self.assertIsNotNone(after)
        self.assertEqual(before.rows, after.rows)
        self.assertEqual(before.cols, after.cols)
        self.assertGreaterEqual(count, 1)
        # Boundary should stay inside the cover silhouette.
        cover = compositor.cover_engine.last_cover_mask
        self.assertIsNotNone(cover)
        for point in after.boundary_points():
            x = int(np.clip(round(point[0]), 0, phone.shape[1] - 1))
            y = int(np.clip(round(point[1]), 0, phone.shape[0] - 1))
            self.assertGreater(int(cover[y, x]), 0)


if __name__ == "__main__":
    unittest.main()
