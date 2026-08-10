"""Regression tests for the Material Rendering Engine."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.image_processing.compositor import PRESETS, Compositor
from src.image_processing.materials import (
    LIGHTING, MATERIALS, MaterialRenderingEngine, material_settings,
)
from src.image_processing.region_detector import HardwareRegionDetector
from src.image_processing.template_cache import TemplateCache
from test_mesh_geometry import synthetic_phone


class MaterialCatalogTests(unittest.TestCase):
    def test_all_materials_and_lighting_are_registered(self) -> None:
        expected_materials = {
            "Glossy", "Matte", "Silicon", "Transparent TPU",
            "Frosted", "Leather", "Carbon Fiber",
        }
        expected_lighting = {"Studio", "Soft", "Outdoor", "Premium"}
        self.assertEqual(set(MATERIALS), expected_materials)
        self.assertEqual(set(LIGHTING), expected_lighting)
        for name in expected_materials | expected_lighting:
            self.assertIn(name, PRESETS)

    def test_material_profiles_expose_required_fields(self) -> None:
        for profile in MATERIALS.values():
            self.assertGreaterEqual(profile.reflection, 0.0)
            self.assertGreaterEqual(profile.highlight, 0.0)
            self.assertGreaterEqual(profile.shadow_softness, 0.0)
            self.assertGreaterEqual(profile.surface_contrast, 0.0)
            self.assertGreaterEqual(profile.texture_strength, 0.0)
            self.assertGreater(profile.opacity, 0.0)
            self.assertLessEqual(profile.opacity, 1.0)


class MaterialEngineTests(unittest.TestCase):
    def test_engine_returns_shaded_design_and_contact_shadow(self) -> None:
        h, w = 120, 80
        design = np.full((h, w, 3), 0.55, np.float32)
        phone = np.linspace(0.2, 0.9, h * w, dtype=np.float32).reshape(h, w)
        phone = np.stack([phone, phone * 0.95, phone * 0.9], axis=-1)
        mask = np.zeros((h, w), np.float32)
        mask[20:100, 15:65] = 1.0

        engine = MaterialRenderingEngine()
        shaded, contact = engine.apply(
            design, phone, mask,
            material=MATERIALS["Carbon Fiber"],
            lighting=LIGHTING["Premium"],
        )
        self.assertEqual(shaded.shape, design.shape)
        self.assertEqual(contact.shape, (h, w))
        self.assertTrue(np.isfinite(shaded).all())
        self.assertGreater(float(np.max(contact)), 0.0)
        # Artwork chroma should remain finite and in range.
        self.assertGreaterEqual(float(shaded.min()), 0.0)
        self.assertLessEqual(float(shaded.max()), 1.0)

    def test_opaque_specular_does_not_bleach_print(self) -> None:
        """Corner specular must keep cover colour — no chalk white wash."""
        h, w = 160, 100
        # Dark maroon print (typical cover art).
        design = np.zeros((h, w, 3), np.float32)
        design[:, :, 0] = 0.12
        design[:, :, 1] = 0.08
        design[:, :, 2] = 0.28
        # Bright studio phone (would previously chalk the BR corner).
        phone = np.full((h, w, 3), 0.92, np.float32)
        mask = np.zeros((h, w), np.float32)
        mask[10:150, 12:88] = 1.0
        # Punch a camera hole for bevel path.
        cv2.circle(mask, (35, 35), 12, 0.0, -1)

        engine = MaterialRenderingEngine()
        shaded, _ = engine.apply(
            design, phone, mask,
            material=MATERIALS["Glossy"],
            lighting=LIGHTING["Premium"],
            settings={
                "reflection_strength": 80.0,
                "shadow_strength": 40.0,
                "texture_strength": 50.0,
                "opacity": 100.0,
            },
        )
        # Bottom-right print region should stay chromatically dark-red, not white.
        patch = shaded[120:145, 65:85]
        mean = patch.mean(axis=(0, 1))
        self.assertLess(float(mean.max()), 0.72)
        # Red channel should still dominate blue/green (cover hue preserved).
        self.assertGreater(float(mean[2]), float(mean[0]))
        self.assertGreater(float(mean[2]), float(mean[1]))

    def test_camera_bump_ridge_keeps_cutout_empty(self) -> None:
        """Bump shades the border; cutout interior stays punched (no design)."""
        h, w = 140, 100
        design = np.zeros((h, w, 3), np.float32)
        design[:, :, 0] = 0.08
        design[:, :, 1] = 0.05
        design[:, :, 2] = 0.42  # deep red wrap
        phone = np.full((h, w, 3), 0.88, np.float32)
        mask = np.ones((h, w), np.float32)
        # Full cutout punched — nothing inside.
        mask[25:110, 18:58] = 0.0
        module = np.zeros((h, w), np.float32)
        module[25:110, 18:58] = 1.0

        shaded, new_mask = MaterialRenderingEngine.apply_camera_bump(
            design,
            mask,
            phone,
            module,
            np.zeros((h, w), np.float32),
            LIGHTING["Studio"],
            wrap_mask=np.ones((h, w), np.float32),
        )
        # Interior of cutout stays empty.
        self.assertLess(float(new_mask[60, 38]), 0.05)
        # Ridge just outside the cutout still has wrap coverage + shading.
        self.assertGreater(float(new_mask[60, 14]), 0.5)
        # Design chroma preserved on the ridge (not bleached white).
        ridge = shaded[58:63, 12:17]
        self.assertGreater(float(ridge[:, :, 2].mean()), 0.20)
        self.assertLess(float(ridge.max()), 0.95)

    def test_camera_bump_corners_shade_as_smoothly_as_straight_edges(self) -> None:
        """Rounded corners of the lip must not shade in visible stair steps."""
        h, w = 460, 380
        design = np.zeros((h, w, 3), np.float32)
        design[:, :] = (0.10, 0.06, 0.34)
        phone = np.full((h, w, 3), 0.85, np.float32)
        mask = np.ones((h, w), np.float32)
        module_u8 = np.zeros((h, w), np.uint8)
        x1, y1, x2, y2 = 110.0, 120.0, 250.0, 340.0
        radius = 62.0
        HardwareRegionDetector._paint_rounded_rect_aa(
            module_u8, x1, y1, x2, y2, radius, expand_px=0.9
        )
        module = module_u8.astype(np.float32) / 255.0

        shaded, _ = MaterialRenderingEngine.apply_camera_bump(
            design, mask, phone, module, np.zeros((h, w), np.float32),
            LIGHTING["Studio"],
        )
        lift = shaded.mean(axis=2) - design.mean(axis=2)
        # There must be a readable raised lip at all.
        self.assertGreater(float(np.max(lift)), 0.02)

        # Walk the top-left corner arc a couple of pixels outside the hole and
        # check the shading has no high-frequency jitter (aliasing signature).
        cx, cy = x1 + radius, y1 + radius
        offset = 4.0
        angles = np.linspace(np.pi, 1.5 * np.pi, 220)
        xs = cx + (radius + offset) * np.cos(angles)
        ys = cy + (radius + offset) * np.sin(angles)
        arc = np.array(
            [
                float(
                    cv2.getRectSubPix(lift, (1, 1), (float(x), float(y)))[0, 0]
                )
                for x, y in zip(xs, ys)
            ],
            dtype=np.float32,
        )
        smooth = cv2.GaussianBlur(arc.reshape(1, -1), (0, 0), 4.0).ravel()
        # An aliased (beaded) corner highlight lands around 0.10 here; a
        # band-limited one stays well under 0.04.
        residual = float(np.max(np.abs(arc[10:-10] - smooth[10:-10])))
        self.assertLess(residual, 0.04)
        # And the corner should be lit in the same range as the straight edge.
        edge = lift[int(y1 + 0.5 * (y2 - y1)), int(x1 - offset)]
        self.assertGreater(float(np.max(arc)), 0.25 * float(edge) - 1e-6)

    def test_stabilize_wrap_texture_reduces_cutout_border_streaks(self) -> None:
        """Tangential re-blend should lower high-frequency warp tear at arcs."""
        h, w = 420, 360
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        design = np.zeros((h, w, 3), np.float32)
        design[:, :, 2] = 0.55 + 0.18 * np.sin(xx / 5.5) * np.cos(yy / 7.0)
        print_mask = np.ones((h, w), np.float32)
        excl_u8 = np.zeros((h, w), np.uint8)
        HardwareRegionDetector._paint_rounded_rect_aa(
            excl_u8, 120.0, 130.0, 250.0, 340.0, 62.0, expand_px=0.9
        )
        excl = excl_u8.astype(np.float32) / 255.0
        # Simulate warp tear: smear along outward normal only.
        hole = (excl > 0.45).astype(np.uint8)
        outside = (1 - hole).astype(np.uint8)
        dist = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
        gnorm = np.sqrt(gx * gx + gy * gy) + 1e-6
        tx, ty = -gy / gnorm, gx / gnorm
        torn = cv2.remap(
            design,
            (xx + tx * 9.0).astype(np.float32),
            (yy + ty * 9.0).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        fixed = MaterialRenderingEngine.stabilize_wrap_texture(
            torn, print_mask, excl
        )
        band = (dist > 0.5) & (dist < 10.0) & (print_mask > 0.2)
        if int(np.count_nonzero(band)) < 32:
            self.skipTest("band too small")
        lap_torn = cv2.Laplacian(torn.mean(axis=2), cv2.CV_32F)
        lap_fixed = cv2.Laplacian(fixed.mean(axis=2), cv2.CV_32F)
        e_torn = float(np.mean(np.abs(lap_torn[band])))
        e_fixed = float(np.mean(np.abs(lap_fixed[band])))
        self.assertLess(e_fixed, e_torn * 0.72)

    def test_edge_finish_does_not_darken_cutout_arcs(self) -> None:
        """Interior holes must not pick up the outer-product charcoal rim."""
        h, w = 220, 180
        design = np.full((h, w, 3), 0.62, np.float32)
        mask = np.ones((h, w), np.float32)
        excl_u8 = np.zeros((h, w), np.uint8)
        HardwareRegionDetector._paint_rounded_rect_aa(
            excl_u8, 60.0, 70.0, 130.0, 170.0, 28.0, expand_px=0.9, aa=1.35
        )
        excl = excl_u8.astype(np.float32) / 255.0
        mask = mask * (1.0 - excl)
        shaded = MaterialRenderingEngine._edge_finish(
            design,
            mask,
            0.35,
            LIGHTING["Studio"],
            exclusion=excl,
        )
        hole = (excl > 0.45).astype(np.uint8)
        outside = (1 - hole).astype(np.uint8)
        dist = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        band = (dist > 0.4) & (dist < 7.0) & (mask > 0.2)
        if int(np.count_nonzero(band)) < 24:
            self.skipTest("cutout band too small")
        base = float(design[band].mean())
        edge = float(shaded[band].mean())
        self.assertGreater(edge, base * 0.88)


class CameraCutoutGeometryTests(unittest.TestCase):
    """Camera openings must hug the hardware and stay perfectly round."""

    @staticmethod
    def _phone_with_stack_and_flash() -> np.ndarray:
        image = np.full((1000, 520, 3), (228, 228, 228), np.uint8)
        cv2.rectangle(image, (55, 35), (465, 965), (70, 74, 80), -1)
        for cy in (110, 185, 260):
            cv2.circle(image, (120, cy), 30, (6, 6, 8), -1)
            cv2.circle(image, (120, cy), 30, (150, 150, 150), 3)
        cv2.circle(image, (215, 150), 17, (240, 240, 230), -1)
        cv2.circle(image, (215, 150), 17, (150, 150, 150), 3)
        return image

    def _compositor(self) -> Compositor:
        compositor = Compositor(
            TemplateCache(Path(tempfile.mkdtemp(prefix="cam-tpl-")))
        )
        self.assertTrue(
            compositor.set_phone_image(self._phone_with_stack_and_flash())
        )
        compositor.set_design_image(np.full((900, 500, 3), (40, 30, 160), np.uint8))
        return compositor

    def test_camera_hole_hugs_the_module(self) -> None:
        """The punch may not balloon past the edited border like a button."""
        compositor = self._compositor()
        stack = HardwareRegionDetector._sample_rounded_rect(
            92.0, 82.0, 150.0, 290.0, 28.0, samples_per_corner=4
        )
        self.assertIsNotNone(stack)
        compositor.set_hardware_exclusions(
            [np.asarray(stack, np.float32).reshape(-1, 2)], allow_clear=True
        )
        mask = compositor.exclusion_mask
        self.assertIsNotNone(mask)
        painted = int(np.count_nonzero(mask > 128))
        poly = np.asarray(stack, np.float32).reshape(-1, 2)
        reference = np.zeros_like(mask)
        cv2.fillPoly(reference, [np.round(poly).astype(np.int32)], 255)
        wanted = int(np.count_nonzero(reference > 128))
        self.assertGreater(painted, wanted * 0.9)
        # Mild expand clears the plate rim without ballooning the hole.
        self.assertLess(painted, wanted * 1.40)

    def test_perfect_finish_keeps_round_flash_round(self) -> None:
        compositor = self._compositor()
        stack = HardwareRegionDetector._sample_rounded_rect(
            88.0, 76.0, 152.0, 294.0, 30.0, samples_per_corner=4
        )
        flash = HardwareRegionDetector._sample_circle(215.0, 150.0, 19.0, samples=16)
        self.assertIsNotNone(stack)
        self.assertIsNotNone(flash)
        compositor.set_hardware_exclusions(
            [
                np.asarray(stack, np.float32).reshape(-1, 2),
                np.asarray(flash, np.float32).reshape(-1, 2),
            ],
            allow_clear=True,
        )
        compositor.perfect_finish_cutouts("camera")
        shapes = [
            np.asarray(c, np.float32).reshape(-1, 2)
            for c in (compositor.hardware_contours or [])
        ]
        # Stack and flash stay separate openings.
        self.assertGreaterEqual(len(shapes), 2)
        small = min(
            shapes,
            key=lambda p: (p[:, 0].max() - p[:, 0].min())
            * (p[:, 1].max() - p[:, 1].min()),
        )
        width = float(small[:, 0].max() - small[:, 0].min())
        height = float(small[:, 1].max() - small[:, 1].min())
        self.assertAlmostEqual(width / max(height, 1e-6), 1.0, delta=0.08)
        centre = small.mean(axis=0)
        radii = np.linalg.norm(small - centre, axis=1)
        self.assertLess(float(radii.std() / max(radii.mean(), 1e-6)), 0.05)


class MaterialCompositorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compositor = Compositor(
            TemplateCache(Path(tempfile.mkdtemp(prefix="mat-tpl-")))
        )
        phone = synthetic_phone()
        design = np.full((400, 200, 3), (30, 120, 220), np.uint8)
        self.assertTrue(self.compositor.set_phone_image(phone))
        self.compositor.set_design_image(design)

    def test_presets_apply_without_changing_placement(self) -> None:
        self.compositor.settings["design_scale"] = 112.0
        self.compositor.settings["offset_x"] = 4.0
        for name in ("Leather", "Frosted", "Outdoor", "Studio Product"):
            settings = self.compositor.apply_preset(name)
            self.assertAlmostEqual(settings["design_scale"], 112.0)
            self.assertAlmostEqual(settings["offset_x"], 4.0)
            self.assertIn(name if name in MATERIALS or name in LIGHTING else "Studio Product", PRESETS)

    def test_material_preset_sets_texture_kind_via_name(self) -> None:
        self.compositor.apply_preset("Carbon Fiber")
        self.assertEqual(self.compositor.material_name, "Carbon Fiber")
        floats = material_settings("Carbon Fiber")
        self.assertAlmostEqual(
            self.compositor.settings["texture_strength"],
            floats["texture_strength"],
            places=3,
        )

    def test_lighting_preset_preserves_material(self) -> None:
        self.compositor.apply_preset("Leather")
        self.compositor.apply_preset("Outdoor")
        self.assertEqual(self.compositor.material_name, "Leather")
        self.assertEqual(self.compositor.lighting_name, "Outdoor")

    def test_hardware_exclusions_remain_intact(self) -> None:
        self.compositor.apply_preset("Glossy")
        result = self.compositor.render(max_size=400)
        self.assertIsNotNone(result)
        exclusion = self.compositor.exclusion_mask
        self.assertIsNotNone(exclusion)
        # Scale exclusion to result size.
        h, w = result.shape[:2]
        excl = cv2.resize(
            exclusion, (w, h), interpolation=cv2.INTER_AREA
        )
        phone = self.compositor.phone_image
        phone_s = cv2.resize(phone, (w, h), interpolation=cv2.INTER_AREA)
        core = excl > 160
        if not np.any(core):
            self.skipTest("no hard exclusion core in synthetic phone")
        # Inside camera holes, composite should match the phone closely.
        delta = np.abs(
            result.astype(np.float32) - phone_s.astype(np.float32)
        )
        mean_delta = float(np.mean(delta[core]))
        self.assertLess(mean_delta, 12.0)

    def test_transparent_tpu_reduces_opacity(self) -> None:
        self.compositor.apply_preset("Transparent TPU")
        self.assertLess(self.compositor.settings["opacity"], 95.0)

    def test_export_still_returns_full_frame(self) -> None:
        self.compositor.apply_preset("Frosted")
        exported = self.compositor.export()
        self.assertIsNotNone(exported)
        self.assertEqual(
            exported.shape[:2], self.compositor.phone_image.shape[:2]
        )


if __name__ == "__main__":
    unittest.main()
