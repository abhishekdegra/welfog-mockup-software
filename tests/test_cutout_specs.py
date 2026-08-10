"""Phase 3 hardware-true cutouts — freeze geom, photo silhouette, paint specs."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.image_processing.device_template import (
    CutoutSpec,
    build_cutout_specs,
    classify_cutout_kind,
)
from src.image_processing.region_detector import HardwareRegionDetector


def _stadium_pts(x1, y1, x2, y2, n=24):
    """Simple axis-aligned stadium outline."""
    return HardwareRegionDetector._sample_rounded_rect(
        x1, y1, x2, y2, min(x2 - x1, y2 - y1) * 0.48, samples_per_corner=3
    )


def test_cutout_spec_roundtrip_with_geom():
    spec = CutoutSpec(
        kind="camera",
        contour=[[0.2, 0.1], [0.4, 0.1], [0.4, 0.25], [0.2, 0.25]],
        geom="rounded_rect",
        params=[40.0, 30.0, 120.0, 90.0, 18.0],
        expand_px=0.9,
        authoritative=True,
    )
    restored = CutoutSpec.from_dict(spec.to_dict())
    assert restored.geom == "rounded_rect"
    assert restored.authoritative is True
    assert restored.params[4] == pytest.approx(18.0)
    assert restored.resolved_expand() == pytest.approx(0.9)


def test_paint_from_frozen_circle_ignores_aabb_reclassify():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Jagged-ish polygon that would classify oddly, but frozen as circle.
    poly = np.array(
        [[80, 60], [120, 55], [130, 90], [115, 120], [85, 115], [70, 85]],
        dtype=np.float32,
    )
    HardwareRegionDetector.paint_cutout_mask(
        mask,
        poly,
        geom="circle",
        params=(100.0, 90.0, 25.0),
        expand_override=0.9,
    )
    assert int(np.count_nonzero(mask > 128)) > 100
    # Centre of frozen circle must be punched.
    assert mask[90, 100] >= 200


def test_paint_contour_geom_uses_polygon():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # L-shaped island — AABB stadium would fill the missing corner.
    L = np.array(
        [
            [40, 40], [120, 40], [120, 70], [70, 70],
            [70, 130], [40, 130],
        ],
        dtype=np.float32,
    )
    HardwareRegionDetector.paint_cutout_mask(
        mask, L, force_contour=True, expand_override=0.5
    )
    assert mask[50, 80] >= 128  # top bar
    assert mask[100, 50] >= 128  # stem
    # The hollow of the L must stay empty (AABB stadium would fill it).
    assert mask[100, 100] < 64


def test_paint_from_cutout_spec_authoritative_contour():
    h, w = 240, 180
    spec = CutoutSpec(
        kind="camera",
        contour=[
            [0.25, 0.15], [0.55, 0.15], [0.55, 0.28],
            [0.40, 0.28], [0.40, 0.42], [0.25, 0.42],
        ],
        geom="contour",
        expand_px=0.9,
        authoritative=True,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    HardwareRegionDetector.paint_from_cutout_spec(mask, spec, w, h)
    assert int(np.count_nonzero(mask > 100)) > 50


def test_freeze_circle_flash():
    pts = HardwareRegionDetector._sample_circle(80, 70, 12, samples=24)
    gray = np.full((160, 160), 180, dtype=np.uint8)
    cv2.circle(gray, (80, 70), 12, 40, -1)
    spec = HardwareRegionDetector.freeze_cutout_spec(
        pts, kind="flash", gray=gray, width=160, height=160
    )
    assert spec.authoritative is True
    assert spec.kind == "flash"
    assert spec.geom in ("circle", "contour", "rounded_rect", "stadium")


def test_large_flash_stays_perfect_circle():
    """Hi-res flash selections used to demote past 64px into rounded_rect."""
    pts = HardwareRegionDetector._sample_circle(220, 160, 42, samples=64)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    kind, params = HardwareRegionDetector._classify_cutout(pts)
    assert kind == "circle"
    assert len(params) >= 3
    assert abs(float(params[2]) - 42.0) < 1.5

    # Even if a bad rounded_rect freeze sneaks in, flash paint stays round.
    from src.image_processing.device_template import CutoutSpec

    h = w = 400
    spec = CutoutSpec(
        kind="flash",
        contour=[[float(x) / w, float(y) / h] for x, y in pts],
        geom="rounded_rect",
        params=[178.0, 118.0, 262.0, 202.0, 38.0],
        expand_px=1.15,
        authoritative=True,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    HardwareRegionDetector.paint_from_cutout_spec(mask, spec, w, h)
    edge = cv2.Canny(mask, 40, 120)
    ey, ex = np.where(edge > 0)
    assert len(ex) > 40
    cx = float(ex.mean())
    cy = float(ey.mean())
    radii = np.sqrt((ex.astype(np.float64) - cx) ** 2 + (ey.astype(np.float64) - cy) ** 2)
    assert float(radii.std() / max(radii.mean(), 1e-6)) < 0.04


def test_aabb_square_still_not_promoted_to_circle():
    box = np.array(
        [[100, 100], [180, 100], [180, 180], [100, 180]], dtype=np.float32
    )
    kind, _params = HardwareRegionDetector._classify_cutout(box)
    assert kind != "circle"


def test_capsule_shape_tag_locks_stadium_paint():
    """User-selected capsule must paint as stadium, never mild rectangle."""
    from src.image_processing.device_template import build_cutout_specs

    cover = np.array(
        [[10, 10], [200, 10], [200, 400], [10, 400]], dtype=np.float32
    )
    # Tall camera-island AABB (would classify mild RR without the tag).
    cam = np.array(
        [[40, 40], [110, 40], [110, 200], [40, 200]], dtype=np.float32
    )
    specs = build_cutout_specs(
        [cam],
        cover,
        220,
        420,
        authoritative=False,
        shape_tags=["capsule"],
    )
    assert len(specs) == 1
    assert specs[0].geom == "stadium"
    assert specs[0].shape_tag == "capsule"
    assert len(specs[0].params) >= 5
    short = min(
        float(specs[0].params[2] - specs[0].params[0]),
        float(specs[0].params[3] - specs[0].params[1]),
    )
    assert float(specs[0].params[4]) >= short * 0.40

    mask = np.zeros((420, 220), dtype=np.uint8)
    HardwareRegionDetector.paint_from_cutout_spec(mask, specs[0], 220, 420)
    assert mask[120, 75] >= 200
    # Full stadium ends punch near the short-side midpoints.
    assert mask[45, 75] >= 180


def test_camera_rectangle_box_is_mild_rounded_not_stadium():
    """User rectangle over a tall camera island must not become a pill."""
    box = np.array(
        [[40, 40], [140, 40], [140, 200], [40, 200]], dtype=np.float32
    )
    kind, params = HardwareRegionDetector._classify_cutout(box)
    assert kind == "rounded_rect"
    assert len(params) >= 5
    short = min(float(params[2] - params[0]), float(params[3] - params[1]))
    corner = float(params[4])
    assert corner <= short * 0.25
    assert corner >= short * 0.10

    from src.image_processing.device_template import CutoutSpec

    h = w = 280
    spec = CutoutSpec(
        kind="camera",
        contour=[[float(x) / w, float(y) / h] for x, y in box],
        geom="rectangle",
        params=[40.0, 40.0, 140.0, 200.0, 16.0],
        expand_px=2.25,
        authoritative=False,
        shape_tag="rectangle",
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    HardwareRegionDetector.paint_from_cutout_spec(mask, spec, w, h)
    assert mask[120, 90] >= 200
    assert mask[48, 48] >= 128


def test_build_cutout_specs_authoritative():
    cover = np.array(
        [[10, 10], [200, 10], [200, 400], [10, 400]], dtype=np.float32
    )
    cam = np.array(
        [[40, 40], [110, 40], [110, 100], [40, 100]], dtype=np.float32
    )
    btn = np.array(
        [[190, 160], [205, 160], [205, 230], [190, 230]], dtype=np.float32
    )
    specs = build_cutout_specs(
        [cam, btn], cover, 220, 420, authoritative=True
    )
    kinds = {s.kind for s in specs}
    assert "camera" in kinds or "other" in kinds
    assert any(s.authoritative for s in specs)
    assert any(s.geom for s in specs)


def test_classify_aligns_with_camera_zone():
    cover = np.array(
        [[20, 20], [200, 20], [200, 400], [20, 400]], dtype=np.float32
    )
    camera = np.array(
        [[40, 40], [100, 40], [100, 95], [40, 95]], dtype=np.float32
    )
    button = np.array(
        [[192, 170], [208, 170], [208, 240], [192, 240]], dtype=np.float32
    )
    assert classify_cutout_kind(camera, cover) in ("camera", "flash")
    assert classify_cutout_kind(button, cover) == "button"


def test_exclusion_from_specs_combines():
    specs = [
        CutoutSpec(
            kind="flash",
            contour=[[0.3, 0.2], [0.4, 0.2], [0.4, 0.28], [0.3, 0.28]],
            geom="circle",
            params=[70.0, 55.0, 10.0],
            expand_px=0.9,
            authoritative=True,
        ),
        CutoutSpec(
            kind="button",
            contour=[[0.85, 0.4], [0.95, 0.4], [0.95, 0.55], [0.85, 0.55]],
            geom="stadium",
            params=[170.0, 100.0, 190.0, 140.0, 18.0],
            expand_px=-1.0,
            authoritative=True,
        ),
    ]
    mask = HardwareRegionDetector.paint_exclusion_from_specs(specs, 200, 250)
    assert int(np.count_nonzero(mask > 128)) > 80


def test_rebuilt_rejects_ballooned_stadium():
    user = [
        np.array(
            [[40, 40], [120, 40], [120, 160], [40, 160]], dtype=np.float32
        )
    ]
    balloon = [
        HardwareRegionDetector._sample_rounded_rect(
            10, 10, 200, 220, 40, samples_per_corner=12
        ).reshape(-1, 2)
    ]
    assert not HardwareRegionDetector._rebuilt_agrees_with_user(user, balloon)


def test_rebuilt_accepts_close_stadium():
    user = [
        np.array(
            [[40, 40], [120, 40], [120, 160], [40, 160]], dtype=np.float32
        )
    ]
    close = [
        HardwareRegionDetector._sample_rounded_rect(
            38, 38, 122, 162, 20, samples_per_corner=12
        ).reshape(-1, 2)
    ]
    assert HardwareRegionDetector._rebuilt_agrees_with_user(user, close)


def test_freeze_irregular_island_uses_contour(monkeypatch):
    """D-shaped photo sil must freeze as contour, not forced stadium/circle."""
    pts = []
    for y in range(40, 161):
        pts.append([40, y])
    for x in range(40, 121):
        pts.append([x, 160])
    for y in range(160, 39, -1):
        pts.append([120, y])
    angles = np.linspace(-np.pi / 2, np.pi / 2, 40)
    for a in angles:
        pts.append([120 + 35 * np.cos(a), 100 + 35 * np.sin(a)])
    for x in range(120, 39, -1):
        pts.append([x, 40])
    sil = np.asarray(pts, dtype=np.float32)

    monkeypatch.setattr(
        HardwareRegionDetector,
        "extract_photo_silhouette",
        staticmethod(lambda gray, seed, **kwargs: sil),
    )
    user_box = np.array(
        [[35, 35], [160, 35], [160, 165], [35, 165]], dtype=np.float32
    )
    gray = np.zeros((220, 200), dtype=np.uint8)
    spec = HardwareRegionDetector.freeze_cutout_spec(
        user_box, kind="camera", gray=gray, width=200, height=220
    )
    assert spec.geom == "contour"
    assert len(spec.contour) >= 16
