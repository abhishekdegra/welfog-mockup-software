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
        ridge = shaded[58:63, 16:19]
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
        offset = 1.25
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

    def test_molded_lip_follows_independent_shapes_and_stays_thin(self) -> None:
        """One renderer: pill + circle keep their own contours; lip is thin."""
        h, w = 240, 180
        design = np.full((h, w, 3), (0.08, 0.04, 0.28), np.float32)
        hole_u8 = np.zeros((h, w), np.uint8)
        cv2.ellipse(hole_u8, (55, 110), (18, 62), 0, 0, 360, 255, -1)
        cv2.circle(hole_u8, (108, 70), 8, 255, -1)
        hole = hole_u8.astype(np.float32) / 255.0
        wrap = (1.0 - (hole > 0.5).astype(np.float32))
        out = MaterialRenderingEngine.apply_molded_cutout_lip(
            design, hole, wrap, LIGHTING["Studio"],
            shade_inner=True, shade_outer=True,
        )
        lift = np.abs(out - design).max(axis=2)
        # Lip exists around both openings (≈1px outside the painted edge).
        self.assertGreater(float(lift[110, 55 - 19]), 0.004)
        self.assertGreater(float(lift[70, 108 + 9]), 0.004)
        # Far from either opening the wrap is unchanged.
        self.assertLess(float(lift[200, 160]), 0.002)
        # Lip does not become a bulky ring: 6px out is nearly flat.
        self.assertLess(float(lift[110, 55 - 25]), float(lift[110, 55 - 19]))
        # Interior of the pill is not flooded with wrap-colored fill.
        self.assertLess(float(np.mean(lift[110, 55])), 0.12)

    def test_molded_lip_does_not_replace_hardware_with_solid_fill(self) -> None:
        """Inner opening keeps phone RGB; rim is a thin annulus, not a plate."""
        h, w = 240, 180
        wrap_rgb = np.array([0.08, 0.04, 0.32], np.float32)
        island = np.array([0.78, 0.76, 0.74], np.float32)
        hole_u8 = np.zeros((h, w), np.uint8)
        HardwareRegionDetector._paint_rounded_rect_aa(
            hole_u8, 48.0, 40.0, 120.0, 170.0, 22.0, expand_px=0.0, aa=1.2
        )
        hole = hole_u8.astype(np.float32) / 255.0
        img = np.broadcast_to(wrap_rgb, (h, w, 3)).copy()
        solid = hole >= 0.50
        img[solid] = island
        opening, outer_lip, inner_lip = MaterialRenderingEngine.cutout_rim_geometry(
            hole
        )
        # Opening is inset: centre is open, selected-path pixels are not.
        self.assertGreater(float(opening[105, 84]), 0.95)
        edge = cv2.Canny((solid.astype(np.uint8) * 255), 40, 120) > 0
        if int(np.count_nonzero(edge)) >= 8:
            self.assertLess(float(opening[edge].mean()), 0.35)
        annulus = np.maximum(outer_lip, inner_lip)
        self.assertGreater(float(np.max(annulus)), 0.4)
        # Annulus must not cover the lens/island interior.
        self.assertLess(float(annulus[105, 84]), 0.08)

        wrap = (1.0 - opening).astype(np.float32)
        out = MaterialRenderingEngine.apply_molded_cutout_lip(
            img, hole, wrap, LIGHTING["Studio"],
            shade_inner=True, shade_outer=True,
        )
        core = opening > 0.85
        np.testing.assert_allclose(out[core], img[core], atol=1e-5)
        # Opening stays the island, not a wrap-coloured or grey plate.
        self.assertGreater(
            float(np.mean(np.abs(out[core] - wrap_rgb))), 0.40
        )
        np.testing.assert_allclose(
            out[core].reshape(-1, 3).mean(axis=0), island, atol=1e-4
        )
        # Rim pixels just outside the path moved (shaded wrap), not a fill.
        dist_out = cv2.distanceTransform(
            (~solid).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        ring = (dist_out > 0.4) & (dist_out < 3.0) & (~solid)
        self.assertGreater(float(np.max(np.abs(out[ring] - img[ring]))), 0.004)

    def test_rim_geometry_is_identical_for_every_locked_shape(self) -> None:
        """One engine: every editor shape yields an annulus from its own path."""
        h, w = 200, 200

        def circle(m):
            cv2.circle(m, (100, 100), 36, 255, -1)

        def square(m):
            cv2.rectangle(m, (64, 64), (136, 136), 255, -1)

        def rrect(m):
            HardwareRegionDetector._paint_rounded_rect_aa(
                m, 60.0, 62.0, 140.0, 138.0, 18.0, expand_px=0.0, aa=1.2
            )

        def oval(m):
            cv2.ellipse(m, (100, 100), (28, 50), 0, 0, 360, 255, -1)

        def diamond(m):
            cv2.fillConvexPoly(
                m, np.array([[100, 50], [150, 100], [100, 150], [50, 100]], np.int32), 255
            )

        def triangle(m):
            cv2.fillConvexPoly(
                m, np.array([[100, 48], [152, 150], [48, 150]], np.int32), 255
            )

        for painter in (circle, square, rrect, oval, diamond, triangle):
            mask = np.zeros((h, w), np.uint8)
            painter(mask)
            hole = mask.astype(np.float32) / 255.0
            opening, outer_lip, inner_lip = MaterialRenderingEngine.cutout_rim_geometry(
                hole
            )
            solid = hole >= 0.50
            self.assertGreater(float(opening[solid].max()), 0.95)
            ys, xs = np.where(solid)
            cy, cx = int(ys.mean()), int(xs.mean())
            self.assertGreater(float(opening[cy, cx]), 0.9)
            rim = np.maximum(outer_lip, inner_lip)
            self.assertGreater(float(np.max(rim)), 0.35)
            self.assertLess(float(rim[cy, cx]), 0.08)
            # Rotate/scale identity: geometry comes from this raster, not AABB.
            rot = cv2.warpAffine(
                mask,
                cv2.getRotationMatrix2D((100.0, 100.0), 25.0, 1.15),
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            hole_r = np.clip(rot.astype(np.float32) / 255.0, 0.0, 1.0)
            op_r, out_r, in_r = MaterialRenderingEngine.cutout_rim_geometry(hole_r)
            solid_r = hole_r >= 0.50
            if int(np.count_nonzero(solid_r)) < 40:
                continue
            ys, xs = np.where(solid_r)
            self.assertGreater(float(op_r[int(ys.mean()), int(xs.mean())]), 0.85)
            self.assertGreater(float(np.max(np.maximum(out_r, in_r))), 0.30)

    def test_compositor_show_through_keeps_hardware_not_a_fill_plate(self) -> None:
        """Wrap is punched at the inset opening; camera RGB is not replaced."""
        h, w = 220, 180
        phone = np.full((h, w, 3), 255, np.uint8)
        body = np.zeros((h, w), np.uint8)
        cv2.rectangle(body, (24, 16), (156, 204), 255, -1)
        phone[body > 0] = (48, 50, 54)
        island = (196, 198, 202)
        excl = np.zeros((h, w), np.uint8)
        HardwareRegionDetector._paint_rounded_rect_aa(
            excl, 46.0, 38.0, 118.0, 168.0, 20.0, expand_px=0.0, aa=1.2
        )
        phone[excl > 160] = island
        cv2.circle(phone, (70, 78), 16, (18, 18, 22), -1)
        cv2.circle(phone, (70, 128), 16, (16, 16, 20), -1)
        wrap_bgr = np.array([18, 28, 170], np.uint8)
        output = phone.copy()
        output[body > 0] = wrap_bgr
        # Artwork currently covers the island — the cutout must punch it.
        comp = Compositor()
        shown = comp._soft_phone_through_cutouts(output, phone, excl)
        rimmed = comp._apply_manufactured_cutout_rim(shown, phone, excl)
        opening, outer_lip, inner_lip = MaterialRenderingEngine.cutout_rim_geometry(
            excl.astype(np.float32) / 255.0
        )
        core = cv2.erode(
            (opening > 0.5).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        ) > 0
        self.assertGreater(int(np.count_nonzero(core)), 40)
        delta = np.abs(
            rimmed.astype(np.int16) - phone.astype(np.int16)
        )[core]
        self.assertLessEqual(int(delta.max()), 2)
        # Not a wrap-coloured plate.
        wrap_delta = np.abs(
            rimmed.astype(np.int16) - wrap_bgr.astype(np.int16)
        )[core]
        self.assertGreater(float(wrap_delta.mean()), 80.0)
        # Lenses survive.
        self.assertLess(float(rimmed[78, 70].mean()), 40.0)
        self.assertLess(float(rimmed[128, 70].mean()), 40.0)
        # Thin rim exists around the selected path (outer wrap side).
        self.assertGreater(float(np.max(outer_lip)), 0.35)
        self.assertLess(float((outer_lip > 0.2)[core].mean()), 0.02)

    def test_nearby_flash_keeps_its_own_thin_lip(self) -> None:
        """A small circle beside a large pill must not inherit a fat halo."""
        h, w = 200, 160
        design = np.full((h, w, 3), (0.07, 0.04, 0.30), np.float32)
        hole_u8 = np.zeros((h, w), np.uint8)
        cv2.ellipse(hole_u8, (50, 100), (20, 70), 0, 0, 360, 255, -1)
        cv2.circle(hole_u8, (86, 100), 8, 255, -1)
        hole = hole_u8.astype(np.float32) / 255.0
        wrap = (1.0 - (hole > 0.5).astype(np.float32))
        out = MaterialRenderingEngine.apply_molded_cutout_lip(
            design, hole, wrap, LIGHTING["Studio"],
            shade_inner=False, shade_outer=True,
        )
        lift = np.abs(out - design).max(axis=2)
        # 1px outside the flash: readable lip.
        self.assertGreater(float(lift[100, 86 + 9]), 0.003)
        # 5px outside the flash: gone — not a capsule-width ring.
        self.assertLess(float(lift[100, 86 + 14]), 0.003)

    def test_molded_lip_follows_every_cutout_shape(self) -> None:
        """Circle, square, pill, diamond and triangle share one geometry path."""
        h, w = 180, 180

        def _lip_hugs(mask_u8: np.ndarray) -> None:
            hole = mask_u8.astype(np.float32) / 255.0
            design = np.full((h, w, 3), (0.10, 0.06, 0.32), np.float32)
            wrap = (1.0 - (hole > 0.5).astype(np.float32))
            out = MaterialRenderingEngine.apply_molded_cutout_lip(
                design, hole, wrap, LIGHTING["Studio"],
                shade_inner=True, shade_outer=True,
            )
            lift = np.abs(out - design).max(axis=2)
            solid = (mask_u8 > 127).astype(np.uint8)
            dist_out = cv2.distanceTransform(
                (1 - solid), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            dist_in = cv2.distanceTransform(
                solid, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            near = ((dist_out > 0.15) & (dist_out < 8.0)) | (
                (dist_in > 0.15) & (dist_in < 3.0)
            )
            far = (dist_out > 10.0) & (wrap > 0.5)
            self.assertGreater(float(np.max(lift[near])), 0.006)
            self.assertLess(float(np.max(lift[far])), 0.003)

        circle = np.zeros((h, w), np.uint8)
        cv2.circle(circle, (90, 90), 28, 255, -1)
        _lip_hugs(circle)

        square = np.zeros((h, w), np.uint8)
        cv2.rectangle(square, (50, 50), (130, 130), 255, -1)
        _lip_hugs(square)

        rrect = np.zeros((h, w), np.uint8)
        HardwareRegionDetector._paint_rounded_rect_aa(
            rrect, 48.0, 55.0, 132.0, 125.0, 18.0, expand_px=0.0, aa=1.2
        )
        _lip_hugs(rrect)

        pill = np.zeros((h, w), np.uint8)
        cv2.ellipse(pill, (90, 90), (22, 58), 0, 0, 360, 255, -1)
        _lip_hugs(pill)

        diamond = np.zeros((h, w), np.uint8)
        pts = np.array([[90, 40], [140, 90], [90, 140], [40, 90]], np.int32)
        cv2.fillConvexPoly(diamond, pts, 255)
        _lip_hugs(diamond)

        tri = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(
            tri, np.array([[90, 38], [145, 140], [35, 140]], np.int32), 255
        )
        _lip_hugs(tri)

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
