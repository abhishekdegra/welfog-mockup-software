"""Regression tests for the offline mesh geometry pipeline."""

import unittest

import cv2
import numpy as np

from src.image_processing.compositor import Compositor
from src.image_processing.mesh import (
    ControlMesh, MeshWarper, create_mesh_mask, mesh_aspect,
)
from src.image_processing.region_detector import (
    HardwareRegionDetector,
    PrintableRegionDetector,
)
from src.image_processing.smart_fit import SmartFitEstimator


def synthetic_phone() -> np.ndarray:
    """Phone-like cover with camera cutouts and side buttons."""
    image = np.full((1000, 520, 3), (225, 225, 225), np.uint8)
    cv2.rectangle(image, (55, 35), (465, 965), (65, 70, 78), -1)
    cv2.rectangle(image, (78, 72), (235, 285), (22, 24, 28), -1)
    hardware = [
        ((125, 125), 31, (4, 4, 5)),
        ((190, 130), 29, (5, 5, 6)),
        ((135, 210), 30, (4, 4, 5)),
        ((198, 220), 15, (245, 245, 235)),
    ]
    for center, radius, colour in hardware:
        cv2.circle(image, center, radius, colour, -1)
        cv2.circle(image, center, radius, (140, 140, 140), 3)
    # Right volume / power button ridges (brighter than body).
    cv2.rectangle(image, (458, 320), (472, 410), (200, 205, 210), -1)
    cv2.rectangle(image, (458, 430), (472, 490), (195, 200, 205), -1)
    # Left mute switch (on the cover rim, not outside the body).
    cv2.rectangle(image, (55, 300), (70, 345), (190, 195, 200), -1)
    return image


class MeshGeometryTests(unittest.TestCase):
    def test_automatic_mesh_is_ordered_and_inside_printable_surface(self) -> None:
        phone = synthetic_phone()
        region = PrintableRegionDetector.detect(phone)
        grid = region.mesh.points.reshape(
            region.mesh.rows, region.mesh.cols, 2
        )

        self.assertGreater(region.margin_percent, 0.0)
        self.assertGreater(region.confidence, 0.3)
        for point in region.mesh.boundary_points():
            x = int(np.clip(round(point[0]), 0, phone.shape[1] - 1))
            y = int(np.clip(round(point[1]), 0, phone.shape[0] - 1))
            self.assertGreater(region.printable_mask[y, x], 0)

        for row in range(region.mesh.rows - 1):
            for col in range(region.mesh.cols - 1):
                tl = grid[row, col]
                tr = grid[row, col + 1]
                bl = grid[row + 1, col]
                br = grid[row + 1, col + 1]
                first = np.cross(tr - tl, bl - tl)
                second = np.cross(br - tr, bl - tr)
                self.assertGreater(float(first), 0.0)
                self.assertGreater(float(second), 0.0)

    def test_smart_fit_scales_logo_and_avoids_camera_cluster(self) -> None:
        phone = synthetic_phone()
        region = PrintableRegionDetector.detect(phone)
        design = np.zeros((600, 360, 4), np.uint8)
        cv2.rectangle(
            design, (120, 220), (240, 380), (30, 80, 230, 255), -1
        )
        cv2.putText(
            design, "A", (150, 335), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (255, 255, 255, 255), 4,
        )

        result = SmartFitEstimator.estimate(
            design, region.mesh, region.exclusion_mask,
            printable_mask=region.printable_mask,
        )
        self.assertGreater(result.scale, 100.0)
        # Camera cluster is upper-left, so the source crop moves upper-left,
        # placing important artwork lower-right on the physical cover.
        self.assertLessEqual(result.offset_x, 0.0)
        self.assertLessEqual(result.offset_y, 0.0)

    def test_pan_and_zoom_never_expose_blank_print(self) -> None:
        design = np.full((900, 620, 4), (40, 60, 200, 255), np.uint8)
        quad = np.array(
            [[60, 40], [460, 55], [455, 950], [65, 940]], np.float32
        )
        mesh = ControlMesh.from_quad(quad)
        interior = cv2.erode(
            (create_mesh_mask(mesh, (1000, 520)) > 0.5).astype(np.uint8),
            np.ones((5, 5), np.uint8),
        ) > 0

        def warp_with(scale: float, offset_x: float, offset_y: float):
            source = MeshWarper.source_points(
                design.shape[:2], mesh.rows, mesh.cols, mesh_aspect(mesh),
                scale=scale, offset_x=offset_x, offset_y=offset_y,
            )
            return MeshWarper.warp(design, source, mesh, (1000, 520))

        for offset_x, offset_y, scale in (
            (-0.9, -0.9, 1.0), (0.9, 0.9, 1.0),
            (0.6, -0.9, 1.0), (0.5, -0.5, 1.4),
        ):
            warped = warp_with(scale, offset_x, offset_y)
            self.assertEqual(
                int(warped[:, :, 3][interior].min()), 255,
                f"pan {offset_x},{offset_y} at {scale} left unprinted pixels",
            )

        # Zooming out below the cover still letterboxes, so the phone stays
        # visible instead of the artwork edge being smeared outwards.
        letterboxed = warp_with(0.6, 0.0, 0.0)
        self.assertEqual(int(letterboxed[:, :, 3][interior].min()), 0)

    def test_smart_fit_keeps_sampling_window_inside_artwork(self) -> None:
        phone = synthetic_phone()
        region = PrintableRegionDetector.detect(phone)
        design = np.full((1063, 736, 4), (30, 40, 200, 255), np.uint8)

        fit = SmartFitEstimator.estimate(
            design, region.mesh, region.exclusion_mask, fit_mode="fill",
            printable_mask=region.printable_mask,
        )
        target = mesh_aspect(region.mesh)
        scale = fit.scale / 100.0
        base_w, base_h = SmartFitEstimator._base_crop(
            design.shape[1], design.shape[0], target, "fill"
        )
        self.assertAlmostEqual(
            (base_w / scale) / (base_h / scale), target, places=3
        )

        source = MeshWarper.source_points(
            design.shape[:2], region.mesh.rows, region.mesh.cols, target,
            scale=scale,
            offset_x=fit.offset_x / 100.0,
            offset_y=fit.offset_y / 100.0,
            rotation=fit.rotation,
        )
        self.assertGreaterEqual(float(source[:, 0].min()), -0.5)
        self.assertGreaterEqual(float(source[:, 1].min()), -0.5)
        self.assertLessEqual(float(source[:, 0].max()), design.shape[1] + 0.5)
        self.assertLessEqual(float(source[:, 1].max()), design.shape[0] + 0.5)

    def test_single_vertex_change_is_local(self) -> None:
        # Non-uniform design so a mesh edit produces a visible pixel change.
        design = np.zeros((700, 350, 4), np.uint8)
        yy, xx = np.indices((700, 350))
        design[:, :, 0] = (xx * 2) % 256
        design[:, :, 1] = (yy * 2) % 256
        design[:, :, 2] = 180
        design[:, :, 3] = 255
        quad = np.array(
            [[80, 60], [440, 60], [440, 940], [80, 940]], np.float32
        )
        base = ControlMesh.from_quad(quad)
        source = MeshWarper.source_points(
            design.shape[:2], base.rows, base.cols, mesh_aspect(base)
        )
        original = MeshWarper.warp(design, source, base, (1000, 520))

        edited = base.copy()
        row = max(2, base.rows // 3)
        col = max(2, base.cols // 3)
        edited.points[edited.index(row, col)] += (35, -25)
        deformed = MeshWarper.warp(design, source, edited, (1000, 520))

        difference = np.max(
            np.abs(
                original.astype(np.int16) - deformed.astype(np.int16)
            ),
            axis=2,
        )
        # Far corner of the cover should stay untouched.
        self.assertEqual(int(difference[80:180, 90:160].max()), 0)
        # Local neighbourhood of the moved vertex must change.
        cx, cy = edited.points[edited.index(row, col)]
        y0 = max(0, int(cy) - 80)
        y1 = min(1000, int(cy) + 80)
        x0 = max(0, int(cx) - 80)
        x1 = min(520, int(cx) + 80)
        self.assertGreater(int(np.count_nonzero(difference[y0:y1, x0:x1])), 100)

    def test_hardware_is_detected_and_never_printed(self) -> None:
        phone = synthetic_phone()
        region = PrintableRegionDetector.detect(phone)
        centers = [(125, 125), (190, 130), (135, 210), (198, 220)]
        self.assertTrue(
            all(region.exclusion_mask[y, x] > 150 for x, y in centers)
        )
        # Side buttons must be excluded so artwork cannot wrap onto them.
        self.assertGreater(int(region.exclusion_mask[360, 465]), 100)
        self.assertGreater(int(region.exclusion_mask[320, 60]), 80)

        compositor = Compositor()
        compositor.set_phone_image(phone)
        # Fresh detect — template cache from earlier tests can leave a soft
        # exclusion fringe on side buttons.
        compositor.redetect_cover()
        compositor.exclude_side_buttons()
        design = np.full((600, 360, 4), (40, 90, 210, 255), np.uint8)
        compositor.set_design_image(design)
        result = compositor.render(None)
        self.assertIsNotNone(result)
        # Button pixels stay near the original phone (not solid design blue).
        phone_btn = phone[360, 465].astype(np.int16)
        out_btn = result[360, 465].astype(np.int16)
        self.assertLess(float(np.mean(np.abs(out_btn - phone_btn))), 40.0)

        design_bgr = np.full((700, 350, 3), (20, 40, 245), np.uint8)
        compositor2 = Compositor()
        self.assertTrue(compositor2.set_phone_image(phone))
        compositor2.redetect_cover()
        compositor2.exclude_side_buttons()
        compositor2.set_design_image(design_bgr)
        output = compositor2.export()

        # Full exclusion cores (camera cutout + buttons) must match the phone —
        # cover design never fills inside the cutout border.
        excl = compositor2.exclusion_mask
        self.assertIsNotNone(excl)
        core = excl >= 200
        difference = np.abs(
            output.astype(np.int16) - phone.astype(np.int16)
        )[core]
        self.assertGreater(int(np.count_nonzero(core)), 0)
        # Allow 1–2 levels of AA / colour rounding at soft exclusion edges.
        self.assertLessEqual(int(difference.max()), 2)

        # Side buttons must still match the phone body.
        phone_btn = phone[360, 465].astype(np.int16)
        out_btn = output[360, 465].astype(np.int16)
        self.assertLess(float(np.mean(np.abs(out_btn - phone_btn))), 40.0)

    def test_perfect_finish_preserves_mesh_vertices(self) -> None:
        phone = synthetic_phone()
        compositor = Compositor()
        self.assertTrue(compositor.set_phone_image(phone))
        mesh = compositor.get_control_mesh()
        self.assertIsNotNone(mesh)
        # Nudge a few vertices like a manual edit.
        edited = mesh.copy()
        edited.points[edited.index(2, 2)] += (12.0, -8.0)
        edited.points[edited.index(4, 1)] += (-6.0, 10.0)
        compositor.set_control_mesh(edited)
        before = edited.points.copy()
        before_cutouts = len(compositor.hardware_contours)

        count = compositor.perfect_finish_cutouts()
        self.assertGreaterEqual(count, 1)
        after = compositor.get_control_mesh()
        self.assertIsNotNone(after)
        # Topology preserved; production perimeter may rebuild from a stable
        # quad (corners/edges tighten) but must not scramble the grid.
        self.assertEqual(before.shape, after.points.shape)
        span = float(np.linalg.norm(before.max(axis=0) - before.min(axis=0)))
        move = np.linalg.norm(after.points - before, axis=1)
        self.assertLess(float(np.max(move)), max(80.0, span * 0.45))
        # Edges finish must not wipe existing cutouts.
        self.assertGreaterEqual(
            len(compositor.hardware_contours), max(0, before_cutouts - 1)
        )
        # Must not invent a pile of new red circles.
        self.assertLessEqual(
            len(compositor.hardware_contours), max(before_cutouts, 1) + 4
        )

    def test_erase_wrap_dabs_clear_print(self) -> None:
        phone = synthetic_phone()
        compositor = Compositor()
        self.assertTrue(compositor.set_phone_image(phone))
        design = np.full((600, 360, 4), (40, 90, 210, 255), np.uint8)
        compositor.set_design_image(design)
        painted = compositor.paint_exclusion_dabs([(465.0, 360.0, 14.0)])
        self.assertGreater(painted, 0)
        self.assertGreater(int(compositor.exclusion_mask[360, 465]), 100)
        result = compositor.render(None)
        phone_btn = phone[360, 465].astype(np.int16)
        out_btn = result[360, 465].astype(np.int16)
        self.assertLess(float(np.mean(np.abs(out_btn - phone_btn))), 40.0)

    def test_camera_island_finish_is_sparse_stadium(self) -> None:
        from src.image_processing.region_detector import HardwareRegionDetector

        # Rough jagged island like a bad auto-trace.
        rough = np.array(
            [
                [80, 70], [200, 68], [210, 90], [205, 200],
                [190, 250], [100, 255], [70, 220], [75, 100],
            ],
            dtype=np.float32,
        )
        finished = HardwareRegionDetector.perfect_finish_contours(
            [rough], synthetic_phone()
        )
        self.assertEqual(len(finished), 1)
        pts = finished[0].reshape(-1, 2)
        self.assertGreaterEqual(len(pts), 8)
        self.assertLessEqual(len(pts), 48)
        # Smooth stadium — no extreme spike from centroid.
        center = pts.mean(axis=0)
        radii = np.linalg.norm(pts - center, axis=1)
        self.assertLess(float(radii.max() / max(radii.min(), 1.0)), 2.8)

    def test_legacy_quad_presets_and_export_still_work(self) -> None:
        compositor = Compositor()
        phone = synthetic_phone()
        design = np.full((500, 300, 3), (30, 100, 210), np.uint8)
        compositor.set_phone_image(phone)
        compositor.set_design_image(design)

        legacy_quad = np.array(
            [[80, 60], [440, 60], [440, 940], [80, 940]], np.float32
        )
        compositor.set_cover_points(legacy_quad)
        self.assertEqual(compositor.get_control_mesh().points.shape, (99, 2))

        settings = compositor.apply_preset("Glossy Glass")
        self.assertEqual(settings["reflection_strength"], 36.0)
        self.assertIsNotNone(compositor.get_preview(500))
        self.assertEqual(compositor.export(include_alpha=True).shape[2], 4)

    def test_create_mesh_mask_follows_phone_silhouette(self) -> None:
        quad = np.array(
            [[25, 25], [195, 25], [195, 375], [25, 375]], dtype=np.float32
        )
        mesh = ControlMesh.from_quad(quad, 11, 9)
        phone = np.zeros((400, 220), dtype=np.uint8)
        cv2.rectangle(phone, (20, 20), (200, 380), 255, -1)
        mask = create_mesh_mask(
            mesh, (400, 220), corner_radius_percent=18.0, phone_silhouette=phone
        )
        phone_bin = phone > 0
        covered = float(np.count_nonzero((mask > 0.35) & phone_bin)) / float(
            np.count_nonzero(phone_bin)
        )
        # Product rim is rounded — sharp phone apexes stay slightly uncovered
        # (envelope gap-fill used to paint those and jagged the bottom).
        self.assertGreaterEqual(covered, 0.88)
        # Bottom corners must keep a soft AA tip like the top (not binary stairs).
        tip = (mask > 0.08) & (mask < 0.85)
        top_tip = int(np.count_nonzero(tip[:80, :]))
        bot_tip = int(np.count_nonzero(tip[-80:, :]))
        self.assertGreater(top_tip, 20)
        self.assertGreater(bot_tip, 20)
        # Bottom tip mass should be in the same ballpark as top.
        self.assertGreaterEqual(bot_tip, int(top_tip * 0.45))

    def test_detect_buttons_from_silhouette_finds_bumps(self) -> None:
        mask = np.zeros((300, 160), dtype=np.uint8)
        cv2.rectangle(mask, (40, 30), (120, 270), 255, -1)
        cv2.ellipse(mask, (38, 110), (6, 22), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, (38, 175), (6, 14), 0, 0, 360, 255, -1)
        quad = np.array(
            [[40, 30], [120, 30], [120, 270], [40, 270]], dtype=np.float32
        )
        hits = HardwareRegionDetector.detect_buttons_from_silhouette(mask, quad)
        self.assertGreater(int(np.count_nonzero(hits)), 40)

    def test_prune_keeps_compact_side_fingerprint(self) -> None:
        """Side FP ovals must survive orphan prune (volume rockers already did)."""
        h, w = 400, 220
        mask = np.zeros((h, w), dtype=np.uint8)
        # Camera island (upper).
        cv2.rectangle(mask, (70, 40), (150, 160), 255, -1)
        # Tall left volume rocker.
        cv2.rectangle(mask, (8, 140), (18, 220), 255, -1)
        # Compact right side fingerprint / power pill (not skinny enough
        # for the old rocker-only keep rule).
        cv2.ellipse(mask, (210, 200), (8, 14), 0, 0, 360, 255, -1)
        before_fp = int(np.count_nonzero(mask[186:214, 202:218]))
        self.assertGreater(before_fp, 20)
        HardwareRegionDetector._prune_orphan_exclusions(mask, w, h)
        after_vol = int(np.count_nonzero(mask[140:220, 8:18]))
        after_fp = int(np.count_nonzero(mask[186:214, 202:218]))
        self.assertGreater(after_vol, 20)
        self.assertGreater(after_fp, 15)


if __name__ == "__main__":
    unittest.main()
