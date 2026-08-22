"""Outer-rim finishing must follow each silhouette, not one phone model."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.image_processing.compositor import Compositor


def _rounded_rect_mask(
    h: int, w: int, x0: int, y0: int, x1: int, y1: int, radius: int
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x0 + radius, y0), (x1 - radius, y1), 255, -1)
    cv2.rectangle(mask, (x0, y0 + radius), (x1, y1 - radius), 255, -1)
    for cx, cy in (
        (x0 + radius, y0 + radius),
        (x1 - radius, y0 + radius),
        (x0 + radius, y1 - radius),
        (x1 - radius, y1 - radius),
    ):
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def _ellipse_mask(h: int, w: int, box: tuple) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, box, 255, -1)
    return mask


class ModelAgnosticRimTests(unittest.TestCase):
    def test_straight_rounded_rect_is_detected_straight(self) -> None:
        mask = _rounded_rect_mask(500, 280, 40, 30, 240, 470, 18)
        state = Compositor._silhouette_wall_state(mask)
        self.assertIsNotNone(state)
        self.assertGreater(state["s_l"], 0.75)
        self.assertGreater(state["s_r"], 0.75)
        self.assertGreater(state["s_t"], 0.75)
        self.assertGreater(state["s_b"], 0.75)
        self.assertGreaterEqual(state["corner_frac"], 0.10)
        self.assertLessEqual(state["corner_frac"], 0.34)

    def test_ellipse_sides_are_not_forced_straight(self) -> None:
        mask = _ellipse_mask(420, 300, ((150, 210), (160, 300), 0))
        state = Compositor._silhouette_wall_state(mask)
        self.assertIsNotNone(state)
        # An ellipse has no long straight wall — do not treat it as a slab.
        self.assertLess(state["s_l"], 0.55)
        self.assertLess(state["s_r"], 0.55)

    def test_tapered_side_keeps_photo_contour(self) -> None:
        mask = np.zeros((400, 240), dtype=np.uint8)
        for y in range(40, 360):
            inset = int(round((y - 40) * 0.12))
            mask[y, 50 + inset : 190] = 255
        state = Compositor._silhouette_wall_state(mask)
        self.assertIsNotNone(state)
        self.assertLess(state["s_l"], 0.45)
        self.assertGreater(state["s_r"], 0.70)

    def test_nub_mask_skips_curved_sides(self) -> None:
        mask = _ellipse_mask(420, 300, ((150, 210), (160, 300), 0))
        state = Compositor._silhouette_wall_state(mask)
        nubs = Compositor._straight_wall_nub_mask(state, mask.shape, eps=0.50)
        self.assertEqual(int(np.count_nonzero(nubs)), 0)

    def test_nub_mask_skips_button_side(self) -> None:
        mask = _rounded_rect_mask(400, 220, 30, 20, 190, 380, 14)
        mask[120:124, 191:194] = 255
        state = Compositor._silhouette_wall_state(mask)
        nubs = Compositor._straight_wall_nub_mask(state, mask.shape, eps=0.50)
        skipped = Compositor._straight_wall_nub_mask(
            state, mask.shape, eps=0.50, skip_sides={"right"}
        )
        self.assertGreater(int(np.count_nonzero(nubs[:, 190:])), 0)
        self.assertEqual(int(np.count_nonzero(skipped[:, 190:])), 0)

    def test_button_sides_detect_left_relief(self) -> None:
        body = _rounded_rect_mask(400, 220, 40, 20, 190, 380, 14)
        tips = np.zeros_like(body)
        tips[200:220, 36:40] = 255
        sides, prot = Compositor._sides_with_buttons(tips, body, body.shape)
        self.assertIn("left", sides)
        self.assertGreater(int(np.count_nonzero(prot)), int(np.count_nonzero(tips)))
        # Protect follows the tip contour — never a full-height left box.
        self.assertEqual(int(np.count_nonzero(prot[:, :30])), 0)
        self.assertGreater(int(np.count_nonzero(prot[198:222, 33:43])), 0)

    def test_wall_reference_ignores_left_fringe_majority(self) -> None:
        edge = np.full(500, 26.0, dtype=np.float32)
        edge[120:300] = 24.0  # fringe/buttons occupy the mid-band majority
        wall = Compositor._straight_wall_reference(
            edge, 20, 480, 460.0, side="left"
        )
        self.assertGreater(wall, 25.4)
        self.assertLess(wall, 26.6)

    def test_grow_spans_fills_tall_rocker(self) -> None:
        h = 80
        seed = np.zeros(h, dtype=bool)
        past = np.zeros(h, dtype=bool)
        past[10:54] = True  # 44px rocker
        seed[10:13] = True  # start cap (too short alone)
        seed[48:54] = True  # end cap (>= min_span)
        spans = Compositor._grow_side_button_spans(
            seed, past, 0, h - 1, min_span=4, max_span=64
        )
        self.assertEqual(spans, [(10, 54)])

    def test_grow_spans_drops_long_fringe_without_seed(self) -> None:
        h = 80
        seed = np.zeros(h, dtype=bool)
        past = np.zeros(h, dtype=bool)
        past[10:53] = True
        seed[10:13] = True  # 3px only — below min_span
        spans = Compositor._grow_side_button_spans(
            seed, past, 0, h - 1, min_span=4, max_span=64
        )
        self.assertEqual(spans, [])

    def test_nub_mask_fires_on_straight_rect_outliers(self) -> None:
        mask = _rounded_rect_mask(400, 220, 30, 20, 190, 380, 14)
        mask[120:124, 191:194] = 255  # 1–3px right-side ticks
        state = Compositor._silhouette_wall_state(mask)
        self.assertGreater(state["s_r"], 0.70)
        nubs = Compositor._straight_wall_nub_mask(state, mask.shape, eps=0.50)
        self.assertGreater(int(np.count_nonzero(nubs)), 0)

    def test_small_and_large_phones_both_get_state(self) -> None:
        tiny = _rounded_rect_mask(180, 100, 12, 10, 88, 170, 8)
        huge = _rounded_rect_mask(1400, 700, 80, 60, 620, 1340, 48)
        for mask in (tiny, huge):
            state = Compositor._silhouette_wall_state(mask)
            self.assertIsNotNone(state)
            self.assertGreater(state["s_r"], 0.65)
            self.assertGreater(state["r"] - state["l"], 8.0)

    def test_fill_bool_gaps_closes_short_holes(self) -> None:
        flags = np.zeros(20, dtype=bool)
        flags[4:8] = True
        flags[10:14] = True  # 2-row gap
        filled = Compositor._fill_bool_gaps(flags, max_gap=2)
        self.assertTrue(np.all(filled[4:14]))
        flags2 = np.zeros(20, dtype=bool)
        flags2[2:5] = True
        flags2[10:13] = True  # 5-row gap stays
        filled2 = Compositor._fill_bool_gaps(flags2, max_gap=2)
        self.assertFalse(np.any(filled2[5:10]))

    def test_photo_buttons_ignore_silhouette_aa_nicks(self) -> None:
        """Buttons exist only where the photo rail actually changes."""
        h, w = 500, 260
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        x0, x1, y0, y1 = 40, 220, 30, 470
        phone[y0:y1, x0:x1] = (48, 48, 48)
        # Quiet bezel AA (included in silhouette, not a key).
        phone[y0:y1, x0 - 2 : x0] = (148, 148, 148)
        # Two real left keys: outer lip shifts + darker rail.
        phone[110:155, x0 - 3 : x0 + 2] = (210, 210, 210)
        phone[110:155, x0 - 2 : x0 + 1] = (70, 70, 70)
        phone[200:255, x0 - 2 : x0 + 2] = (120, 120, 120)
        phone[200:255, x0 - 1 : x0 + 1] = (55, 55, 55)
        # Silhouette-only nick at mid-body (bright AA, same lip as quiet).
        raw = np.zeros((h, w), dtype=np.uint8)
        raw[y0:y1, x0 - 2 : x1] = 255
        raw[110:155, x0 - 3 : x0] = 255
        raw[200:255, x0 - 2 : x0] = 255
        raw[310:320, x0 - 4 : x0] = 255  # fake nick, no photo lip change

        comp = Compositor()
        comp.phone_image = phone
        comp.exclusion_mask = None
        body, btn = comp._derive_clean_body_and_button_masks(
            raw, phone, exclusion_mask=None
        )
        self.assertIsNotNone(btn)
        n, _, st, _ = cv2.connectedComponentsWithStats(
            (btn > 127).astype(np.uint8), connectivity=8
        )
        boxes = [
            (
                int(st[i, cv2.CC_STAT_LEFT]),
                int(st[i, cv2.CC_STAT_TOP]),
                int(st[i, cv2.CC_STAT_HEIGHT]),
            )
            for i in range(1, n)
        ]
        self.assertEqual(len(boxes), 2)
        tops = sorted(b[1] for b in boxes)
        self.assertTrue(any(abs(t - 110) <= 6 for t in tops))
        self.assertTrue(any(abs(t - 200) <= 8 for t in tops))
        self.assertEqual(int(np.count_nonzero(btn[308:322])), 0)
        # Body stays a straight wall at the nick — no synthetic nub.
        nick_xmin = []
        quiet_xmin = []
        for y in range(312, 318):
            xs = np.where(body[y] > 127)[0]
            if xs.size:
                nick_xmin.append(int(xs.min()))
        for y in range(80, 90):
            xs = np.where(body[y] > 127)[0]
            if xs.size:
                quiet_xmin.append(int(xs.min()))
        self.assertTrue(nick_xmin and quiet_xmin)
        self.assertLessEqual(abs(int(np.median(nick_xmin)) - int(np.median(quiet_xmin))), 1)

    def test_button_wrap_samples_body_only_inside_mask(self) -> None:
        h, w = 180, 120
        output = np.zeros((h, w, 3), dtype=np.uint8)
        output[:, 50:] = (30, 50, 90)  # navy wrap on body
        output[70:120, 46:50] = (90, 90, 95)  # original gray key
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 46:] = (40, 40, 40)
        phone[70:120, 46:50] = (120, 125, 130)  # brighter lip
        tips = np.zeros((h, w), dtype=bool)
        tips[70:120, 46:50] = True
        body = np.zeros((h, w), dtype=bool)
        body[:, 50:] = True
        wrapped = Compositor._wrap_validated_button_surface(
            output, phone, tips, body, left_side=True
        )
        btn = wrapped[70:120, 46:50]
        body_keep = wrapped[:, 50:]
        # Artwork from the body, not the original gray hardware.
        self.assertLess(float(btn.mean()), 80.0)
        self.assertGreater(float(btn[:, :, 2].mean()), float(btn[:, :, 0].mean()))
        # Body pixels unchanged.
        self.assertTrue(np.array_equal(body_keep, output[:, 50:]))
        # Nothing written outside the mask.
        outside = wrapped.copy()
        outside[tips] = output[tips]
        self.assertTrue(np.array_equal(outside[~tips], output[~tips]))

    def test_button_wrap_samples_design_not_composite_rim(self) -> None:
        """Body rim in the composite is still original bezel; artwork is in wrap_src."""
        h, w = 180, 120
        output = np.zeros((h, w, 3), dtype=np.uint8)
        output[:, 50:54] = (88, 90, 94)  # gray show-through at the wall
        output[:, 54:] = (30, 50, 90)
        output[70:120, 46:50] = (110, 112, 118)
        wrap_src = np.zeros((h, w, 3), dtype=np.float32)
        wrap_src[:, 50:] = (30 / 255.0, 50 / 255.0, 90 / 255.0)
        phone = np.full((h, w, 3), 40, dtype=np.uint8)
        phone[70:120, 46:50] = (130, 128, 120)
        phone[70:120, 46] = (170, 168, 160)  # outer lip highlight
        tips = np.zeros((h, w), dtype=bool)
        tips[70:120, 46:50] = True
        body = np.zeros((h, w), dtype=bool)
        body[:, 50:] = True
        wrapped = Compositor._wrap_validated_button_surface(
            output,
            phone,
            tips,
            body,
            left_side=True,
            wrap_src=wrap_src,
        )
        btn = wrapped[70:120, 46:50]
        # Must be navy wrap, not the gray composite rim / hardware.
        self.assertLess(float(btn.mean()), 95.0)
        self.assertGreater(float(btn[:, :, 2].mean()), float(btn[:, :, 0].mean()) + 15.0)
        self.assertTrue(np.array_equal(wrapped[:, 50:], output[:, 50:]))
        self.assertTrue(np.array_equal(wrapped[~tips], output[~tips]))
        # Contour from the photo: outer lip stays brighter than the inner face.
        self.assertGreater(float(btn[:, 0].mean()), float(btn[:, -1].mean()))

    def test_button_wrap_skips_empty_uv_rim(self) -> None:
        """UV often has no ink at the wall; keys must sample the print inward."""
        h, w = 180, 120
        output = np.zeros((h, w, 3), dtype=np.uint8)
        output[:, 50:56] = (8, 8, 8)
        output[:, 56:] = (30, 50, 90)
        output[70:120, 46:50] = (12, 12, 12)
        wrap_src = np.zeros((h, w, 3), dtype=np.float32)
        wrap_src[:, 56:] = (30 / 255.0, 50 / 255.0, 90 / 255.0)
        phone = np.full((h, w, 3), 40, dtype=np.uint8)
        phone[70:120, 46:50] = (130, 128, 120)
        tips = np.zeros((h, w), dtype=bool)
        tips[70:120, 46:50] = True
        body = np.zeros((h, w), dtype=bool)
        body[:, 50:] = True
        wrapped = Compositor._wrap_validated_button_surface(
            output,
            phone,
            tips,
            body,
            left_side=True,
            wrap_src=wrap_src,
        )
        btn = wrapped[70:120, 46:50]
        self.assertGreater(float(btn.mean()), 20.0)
        self.assertLess(float(btn.mean()), 95.0)
        self.assertGreater(
            float(btn[:, :, 2].mean()), float(btn[:, :, 0].mean()) + 12.0
        )
        self.assertTrue(np.array_equal(wrapped[~tips], output[~tips]))

    def test_button_wrap_layer_covers_every_component(self) -> None:
        h, w = 160, 100
        output = np.full((h, w, 3), 20, dtype=np.uint8)
        output[:, 40:] = (28, 48, 88)
        output[40:55, 36:40] = (100, 100, 100)
        output[70:110, 36:40] = (100, 100, 100)
        wrap_src = np.zeros((h, w, 3), dtype=np.uint8)
        wrap_src[:, 40:] = (28, 48, 88)
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 40:] = (50, 50, 50)
        phone[40:55, 36:40] = (80, 80, 85)
        phone[70:110, 36:40] = (80, 80, 85)
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[40:55, 36:40] = 255
        tips[70:110, 36:40] = 255
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 40:] = 255
        comp = Compositor()
        out = comp._composite_side_button_layer(
            output,
            phone,
            tips,
            phone_mask=body,
            wrap_src=wrap_src,
        )
        for y0, y1 in ((40, 55), (70, 110)):
            btn = out[y0:y1, 36:40]
            self.assertLess(float(btn.mean()), 70.0)
            self.assertGreater(
                float(btn[:, :, 2].mean()), float(btn[:, :, 0].mean())
            )
        self.assertTrue(np.array_equal(out[:, 40:], output[:, 40:]))
        self.assertTrue(np.array_equal(out[tips == 0], output[tips == 0]))

    def test_button_wrap_stays_inside_validated_contour(self) -> None:
        """Wrap must not fill a rectangular stem outside the detected keys."""
        h, w = 160, 100
        output = np.full((h, w, 3), 255, dtype=np.uint8)
        output[:, 40:] = (28, 48, 88)
        output[50:90, 36:40] = (110, 110, 112)
        wrap_src = np.zeros((h, w, 3), dtype=np.uint8)
        wrap_src[:, 40:] = (28, 48, 88)
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 40:] = (50, 50, 50)
        phone[50:90, 36:40] = (80, 80, 85)
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[50:90, 36:40] = 255
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 40:] = 255
        comp = Compositor()
        out = comp._composite_side_button_layer(
            output,
            phone,
            tips,
            phone_mask=body,
            wrap_src=wrap_src,
        )
        # Nothing outside the validated tip mask is rewritten.
        self.assertTrue(np.array_equal(out[tips == 0], output[tips == 0]))
        btn = out[50:90, 36:40]
        self.assertLess(float(btn.mean()), 80.0)
        self.assertGreater(float(btn[:, :, 2].mean()), float(btn[:, :, 0].mean()))

    def test_button_wrap_clips_oversized_rectangle_to_photo_protrusion(self) -> None:
        """A rectangular detection strip is clipped to the real outer lip."""
        h, w = 160, 100
        output = np.full((h, w, 3), 20, dtype=np.uint8)
        output[:, 40:] = (28, 48, 88)
        output[50:90, 36:40] = (100, 100, 100)
        wrap_src = np.zeros((h, w, 3), dtype=np.uint8)
        wrap_src[:, 38:] = (28, 48, 88)
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 38:] = (50, 50, 50)  # real case edge
        phone[50:90, 36:38] = (90, 90, 95)  # real 2px button
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[50:90, 36:40] = 255  # oversized rectangle to the wall
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 40:] = 255
        comp = Compositor()
        out = comp._composite_side_button_layer(
            output,
            phone,
            tips,
            phone_mask=body,
            wrap_src=wrap_src,
        )
        face = out[50:90, 36:38]
        stem = out[50:90, 38:40]
        self.assertLess(float(face.mean()), 80.0)
        self.assertGreater(float(face[:, :, 2].mean()), float(face[:, :, 0].mean()))
        # Cover continues over the key stem; studio and body stay put.
        self.assertLess(float(stem.mean()), 80.0)
        self.assertGreater(float(stem[:, :, 2].mean()), float(stem[:, :, 0].mean()))
        self.assertTrue(np.array_equal(out[:, 40:], output[:, 40:]))
        self.assertTrue(np.array_equal(out[50:90, :36], output[50:90, :36]))

    def test_button_wrap_removes_bright_aa_fringe(self) -> None:
        """Original photo AA must not remain as a light bar beside the wrap."""
        h, w = 160, 100
        output = np.full((h, w, 3), 255, dtype=np.uint8)
        output[:, 40:] = (28, 48, 88)
        output[50:90, 35] = (250, 250, 250)
        output[50:90, 36:40] = (110, 110, 112)
        wrap_src = np.zeros((h, w, 3), dtype=np.uint8)
        wrap_src[:, 40:] = (28, 48, 88)
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 40:] = (50, 50, 50)
        phone[50:90, 35] = (250, 250, 250)
        phone[50:90, 36:39] = (70, 70, 75)
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[50:90, 35:40] = 255
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 40:] = 255
        comp = Compositor()
        out = comp._composite_side_button_layer(
            output,
            phone,
            tips,
            phone_mask=body,
            wrap_src=wrap_src,
        )
        fringe = out[50:90, 35]
        face = out[50:90, 36:39]
        self.assertGreater(float(fringe.mean()), 240.0)
        self.assertLess(float(face.mean()), 80.0)
        self.assertGreater(float(face[:, :, 2].mean()), float(face[:, :, 0].mean()))
        self.assertTrue(np.array_equal(out[:, 40:], output[:, 40:]))

    def test_button_wrap_follows_photo_not_component_bbox(self) -> None:
        """AA-only cap rows in the component bbox are not wrapped as a bar."""
        h, w = 160, 100
        output = np.full((h, w, 3), 20, dtype=np.uint8)
        output[:, 40:] = (28, 48, 88)
        output[50:90, 36:40] = (100, 100, 100)
        wrap_src = np.zeros((h, w, 3), dtype=np.uint8)
        wrap_src[:, 40:] = (28, 48, 88)
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 40:] = (50, 50, 50)
        phone[50:90, 36:39] = (70, 70, 75)
        phone[50:56, 36:40] = (230, 230, 230)
        phone[84:90, 36:40] = (230, 230, 230)
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[50:90, 36:40] = 255
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 40:] = 255
        paint, alpha, _ = Compositor._button_photo_wrap_alpha(
            tips > 127,
            tips > 127,
            body > 127,
            phone,
            left_side=True,
        )
        self.assertGreater(int(np.count_nonzero(paint[60:80, 36:39])), 20)
        self.assertEqual(int(np.count_nonzero(paint[50:56])), 0)
        self.assertEqual(int(np.count_nonzero(paint[84:90])), 0)
        self.assertEqual(int(np.count_nonzero(paint[:, 39:])), 0)

    def test_button_wrap_photo_contour_drops_aa_and_bbox_caps(self) -> None:
        """Quiet-bezel AA lip: wrap the solid key, not the detection rectangle."""
        h, w = 200, 80
        output = np.full((h, w, 3), 255, dtype=np.uint8)
        output[:, 28:] = (20, 30, 50)
        wrap_src = np.zeros((h, w, 3), dtype=np.uint8)
        wrap_src[:, 28:] = (20, 30, 50)
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 27] = (145, 145, 145)
        phone[:, 28:] = (70, 70, 70)
        # Mid key: outer AA + solid face. Caps in the bbox are rounded AA.
        phone[50:90, 25] = (209, 209, 209)
        phone[50:90, 26] = (72, 72, 72)
        phone[50:90, 27] = (69, 69, 69)
        phone[48, 25] = (234, 234, 234)
        phone[48, 26] = (186, 186, 186)
        phone[48, 27] = (117, 117, 117)
        phone[90, 25] = (223, 223, 223)
        phone[90, 26] = (119, 119, 119)
        phone[90, 27] = (84, 84, 84)
        # Thinner key whose outer face is mid-gray (not studio AA).
        phone[110:140, 26] = (126, 126, 126)
        phone[110:140, 27] = (57, 57, 57)
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[48:91, 25:28] = 255
        tips[110:140, 26:28] = 255
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 28:] = 255
        paint, _, _ = Compositor._button_photo_wrap_alpha(
            tips > 127,
            tips > 127,
            body > 127,
            phone,
            left_side=True,
        )
        self.assertEqual(int(np.count_nonzero(paint[50:90, 25])), 0)
        self.assertEqual(int(np.count_nonzero(paint[50:90, 26])), 40)
        self.assertEqual(int(np.count_nonzero(paint[50:90, 27])), 40)
        self.assertEqual(int(np.count_nonzero(paint[48, 25:27])), 0)
        self.assertTrue(bool(paint[48, 27]))
        self.assertEqual(int(np.count_nonzero(paint[90, 25])), 0)
        self.assertTrue(bool(paint[90, 26]) and bool(paint[90, 27]))
        self.assertEqual(int(np.count_nonzero(paint[110:140, 26])), 30)
        self.assertEqual(int(np.count_nonzero(paint[110:140, 27])), 30)
        self.assertLess(int(np.count_nonzero(paint[48])), int(np.count_nonzero(paint[70])))
        comp = Compositor()
        out = comp._composite_side_button_layer(
            output,
            phone,
            tips,
            phone_mask=body,
            wrap_src=wrap_src,
        )
        self.assertGreater(float(out[50:90, 25].mean()), 240.0)
        self.assertLess(float(out[50:90, 26].mean()), 80.0)
        self.assertGreater(
            float(out[50:90, 26, 2].mean()), float(out[50:90, 26, 0].mean())
        )
        self.assertTrue(np.array_equal(out[:, 28:], output[:, 28:]))

    def test_button_row_keeps_body_wall_isolated(self) -> None:
        """Button wrap must not rewrite the body wall (no dark rectangular strip)."""
        h, w = 160, 100
        output = np.full((h, w, 3), 255, dtype=np.uint8)
        output[:, 40:] = (28, 48, 88)
        output[50:90, 40] = (153, 150, 149)
        output[50:90, 36:40] = (110, 110, 112)
        wrap_src = np.zeros((h, w, 3), dtype=np.uint8)
        wrap_src[:, 40:] = (28, 48, 88)
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 40:] = (50, 50, 50)
        phone[50:90, 36:39] = (70, 70, 75)
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[50:90, 36:40] = 255
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 40:] = 255
        comp = Compositor()
        out = comp._composite_side_button_layer(
            output,
            phone,
            tips,
            phone_mask=body,
            wrap_src=wrap_src,
        )
        self.assertTrue(np.array_equal(out[:, 40:], output[:, 40:]))
        self.assertTrue(np.array_equal(out[tips == 0], output[tips == 0]))
        btn = out[50:90, 36:39]
        self.assertLess(float(btn.mean()), 80.0)
        self.assertGreater(float(btn[:, :, 2].mean()), float(btn[:, :, 0].mean()))

    def test_button_wrap_covers_silver_keys(self) -> None:
        """Light / silver side keys wrap with the print, not leftover hardware."""
        h, w = 160, 100
        output = np.full((h, w, 3), 255, dtype=np.uint8)
        output[:, 40:] = (28, 48, 88)
        output[50:90, 36:40] = (210, 212, 214)
        wrap_src = np.zeros((h, w, 3), dtype=np.uint8)
        wrap_src[:, 40:] = (28, 48, 88)
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 40:] = (50, 50, 50)
        phone[50:90, 36:40] = (208, 210, 214)
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[50:90, 36:40] = 255
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 40:] = 255
        paint, _, _ = Compositor._button_photo_wrap_alpha(
            tips > 127,
            tips > 127,
            body > 127,
            phone,
            left_side=True,
        )
        self.assertGreater(int(np.count_nonzero(paint[50:90, 36:40])), 80)
        comp = Compositor()
        out = comp._composite_side_button_layer(
            output,
            phone,
            tips,
            phone_mask=body,
            wrap_src=wrap_src,
        )
        btn = out[50:90, 36:40]
        self.assertLess(float(btn.mean()), 90.0)
        self.assertGreater(float(btn[:, :, 2].mean()), float(btn[:, :, 0].mean()))

    def test_snap_keeps_nub_without_on_body_strip(self) -> None:
        """Stem to the wall is kept; a wide on-body rectangle is not invented."""
        h, w = 200, 120
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 40:] = 255
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[80:120, 36:40] = 255
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 36:] = (60, 60, 62)
        phone[80:120, 36:40] = (90, 88, 86)
        comp = Compositor()
        snapped = comp._snap_button_mask_to_device_surface(tips, body, phone)
        self.assertGreater(int(np.count_nonzero(snapped[80:120, 36:40])), 20)
        self.assertEqual(int(np.count_nonzero(snapped[80:120, 42:50])), 0)
        self.assertEqual(int(np.count_nonzero(snapped[20:40])), 0)

    def test_silhouette_fallback_needs_real_bump_not_aa(self) -> None:
        h = 200
        el = np.full(h, 40.0, dtype=np.float32)
        er = np.full(h, 180.0, dtype=np.float32)
        el[80:84] = 39.0  # 1px AA tick
        raw = np.zeros((h, 220), dtype=np.uint8)
        for y in range(h):
            raw[y, int(el[y]) : int(er[y]) + 1] = 255
        comp = Compositor()
        out = comp._silhouette_side_button_mask(
            raw, el, er, 40.0, 180.0, 10, 190, 180.0
        )
        self.assertTrue(out is None or int(np.count_nonzero(out)) == 0)

    def test_hard_hole_weight_keeps_sdf_ramp(self) -> None:
        """Inset punch keeps a soft ramp and opens only deep inside the hole."""
        yy, xx = np.mgrid[0:80, 0:80].astype(np.float32)
        dist = np.hypot(xx - 40.0, yy - 40.0) - 18.0
        excl = np.clip(0.5 - dist / 3.0, 0.0, 1.0)
        hole = Compositor._hard_hole_weight(excl)
        mid = int(np.count_nonzero((hole > 0.08) & (hole < 0.92)))
        self.assertGreater(mid, 40)
        # Punch is inset vs painted coverage — solid hole area shrinks.
        self.assertLess(
            int(np.count_nonzero(hole > 0.50)),
            int(np.count_nonzero(excl > 0.50)),
        )
        # Deep center still fully open.
        self.assertGreater(float(hole[40, 40]), 0.95)

    def test_button_wrap_replaces_black_voids_from_body_print(self) -> None:
        h, w = 160, 100
        output = np.full((h, w, 3), 20, dtype=np.uint8)
        output[:, 40:] = (28, 48, 88)
        output[50:90, 36:40] = (4, 4, 4)
        wrap_src = np.zeros((h, w, 3), dtype=np.float32)
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        phone[:, 40:] = (50, 50, 50)
        phone[50:90, 36:40] = (40, 40, 42)
        tips = np.zeros((h, w), dtype=np.uint8)
        tips[50:90, 36:40] = 255
        body = np.zeros((h, w), dtype=np.uint8)
        body[:, 40:] = 255
        comp = Compositor()
        out = comp._composite_side_button_layer(
            output,
            phone,
            tips,
            phone_mask=body,
            wrap_src=wrap_src,
        )
        btn = out[50:90, 36:40]
        self.assertGreater(float(btn.mean()), 20.0)
        self.assertGreater(float(btn[:, :, 2].mean()), float(btn[:, :, 0].mean()))
        self.assertTrue(np.array_equal(out[tips == 0], output[tips == 0]))

    def test_wrap_silhouette_excludes_studio_aabb_wedges(self) -> None:
        """Detected wrap mask follows the rounded phone, not the AABB."""
        from src.image_processing.cover_surface import CoverSurfaceEngine

        h, w = 320, 220
        phone = np.full((h, w, 3), 255, dtype=np.uint8)
        body = _rounded_rect_mask(h, w, 40, 20, 180, 300, 28)
        phone[body > 0] = (190, 190, 192)
        sil = CoverSurfaceEngine.detect_phone_wrap_silhouette(phone)
        self.assertIsNotNone(sil)
        # AABB corner wedges (outside the round phone) must stay studio.
        for x, y in ((40, 20), (180, 20), (40, 300), (180, 300)):
            self.assertEqual(int(sil[y, x]), 0)
        # Interior of the phone stays covered.
        self.assertGreater(int(sil[160, 110]), 127)

    def test_seal_maps_clips_coverage_outside_body(self) -> None:
        """UV coverage cannot remain outside the phone silhouette."""
        from src.image_processing.mesh import MeshWarper

        h, w = 40, 40
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        coverage = np.zeros((h, w), dtype=np.uint8)
        coverage[2:38, 2:38] = 255  # oversized rectangle
        body = np.zeros((h, w), dtype=np.uint8)
        body[8:32, 8:32] = 255
        map_x[coverage > 0] = 10.0
        mx, my, cov = MeshWarper.seal_maps_to_mask(map_x, map_y, coverage, body)
        self.assertEqual(int(np.count_nonzero(cov[body == 0])), 0)
        self.assertGreater(int(np.count_nonzero(cov[body > 0])), 0)
        self.assertEqual(float(mx[0, 0]), 0.0)

    def test_camera_face_bite_is_filled_to_the_wall(self) -> None:
        """GrabCut pac-man left of the camera must not stay in the wrap mask."""
        body = _rounded_rect_mask(400, 240, 40, 20, 200, 380, 22)
        # Vertical bite from the top-left like a camera-to-edge GrabCut notch.
        body[20:160, 40:110] = 0
        excl = np.zeros((400, 240), dtype=np.uint8)
        cv2.circle(excl, (130, 90), 22, 255, -1)
        phone = np.full((400, 240, 3), 40, dtype=np.uint8)
        phone[body > 0] = (180, 180, 182)
        filled = Compositor._fill_camera_face_bites(body, excl, phone)
        self.assertGreater(int(filled[90, 55]), 127)
        self.assertGreater(int(filled[40, 70]), 127)

    def test_seal_body_row_spans_fills_interior_channel(self) -> None:
        mask = _rounded_rect_mask(200, 120, 20, 15, 100, 185, 16)
        mask[40:80, 28:45] = 0  # vertical channel inside the body
        sealed = Compositor._seal_body_row_spans(mask)
        self.assertGreater(int(sealed[60, 36]), 127)
        self.assertEqual(int(sealed[10, 10]), 0)


if __name__ == "__main__":
    unittest.main()
