"""Model-agnostic wrap: any camera layout + any design must stay upright."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.image_processing.compositor import Compositor
from src.image_processing.cover_surface import CoverSurfaceEngine
from src.image_processing.mesh import AdaptiveMeshBuilder
from src.image_processing.template_cache import TemplateManager


def _synthetic_phone(camera_side: str = "left") -> np.ndarray:
    """Upright product shot with a raised camera plate on left/center/right."""
    image = np.full((900, 480, 3), (240, 240, 240), np.uint8)
    cv2.rectangle(image, (50, 40), (430, 860), (70, 74, 82), -1, cv2.LINE_AA)
    # Rounded corners approx.
    for cx, cy in ((70, 60), (410, 60), (70, 840), (410, 840)):
        cv2.circle(image, (cx, cy), 22, (70, 74, 82), -1, cv2.LINE_AA)

    if camera_side == "left":
        plate = (70, 70, 210, 230)
    elif camera_side == "right":
        plate = (270, 70, 410, 230)
    else:
        plate = (160, 70, 320, 220)

    x1, y1, x2, y2 = plate
    cv2.rectangle(image, (x1, y1), (x2, y2), (18, 20, 24), -1, cv2.LINE_AA)
    # Lenses inside the plate (layout follows the plate, not a fixed corner).
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    lenses = [
        (mid_x - 28, mid_y - 28, 22),
        (mid_x - 28, mid_y + 30, 20),
        (mid_x + 32, mid_y - 20, 12),
        (mid_x + 32, mid_y + 28, 10),
    ]
    for cx, cy, r in lenses:
        cv2.circle(image, (cx, cy), r, (5, 5, 6), -1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), r, (120, 120, 120), 2, cv2.LINE_AA)
    return image


def _design(h: int = 800, w: int = 400) -> np.ndarray:
    design = np.zeros((h, w, 3), np.uint8)
    design[:] = (40, 30, 140)
    cv2.circle(design, (w // 2, h // 3), 80, (20, 180, 220), -1)
    cv2.rectangle(design, (40, h // 2), (w - 40, h - 40), (30, 30, 30), -1)
    return design


class ModelAgnosticWrapTests(unittest.TestCase):
    def _engine(self) -> CoverSurfaceEngine:
        return CoverSurfaceEngine(
            template_manager=TemplateManager(Path(tempfile.mkdtemp()))
        )

    def test_upright_mesh_for_left_center_right_cameras(self) -> None:
        for side in ("left", "center", "right"):
            with self.subTest(side=side):
                phone = _synthetic_phone(side)
                surface = self._engine().analyze(phone, use_templates=False)
                self.assertIsNotNone(surface.mesh)
                tilt = AdaptiveMeshBuilder._quad_axis_deviation_deg(
                    surface.mesh.corner_points()
                )
                self.assertLess(tilt, 2.5, f"{side} mesh tilt {tilt}")
                excl = surface.exclusion_mask
                self.assertIsNotNone(excl)
                self.assertGreater(int(np.count_nonzero(excl)), 200)
                # Camera exclusion must sit in the upper half.
                ys, xs = np.where(excl > 96)
                self.assertLess(float(ys.mean()), phone.shape[0] * 0.45)
                cx = float(xs.mean())
                if side == "left":
                    self.assertLess(cx, phone.shape[1] * 0.48)
                elif side == "right":
                    self.assertGreater(cx, phone.shape[1] * 0.52)
                else:
                    self.assertGreater(cx, phone.shape[1] * 0.35)
                    self.assertLess(cx, phone.shape[1] * 0.65)

    def test_any_design_fills_without_stale_offsets(self) -> None:
        phone = _synthetic_phone("center")
        design = _design()
        comp = Compositor()
        comp.cover_engine = self._engine()
        self.assertTrue(comp.set_phone_image(phone))
        comp.set_design_image(design)
        self.assertEqual(float(comp.settings.get("offset_x", 99)), 0.0)
        self.assertEqual(float(comp.settings.get("offset_y", 99)), 0.0)
        self.assertEqual(float(comp.settings.get("rotation", 99)), 0.0)
        out = comp.render()
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[:2], phone.shape[:2])
        # Design must change a meaningful share of the printable mesh area.
        diff = cv2.absdiff(out, phone).mean(axis=2) > 10
        self.assertGreater(float(diff.mean()), 0.12)


if __name__ == "__main__":
    unittest.main()
