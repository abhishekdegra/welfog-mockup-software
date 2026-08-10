"""Phase 1 device template schema, catalog, and per-corner mesh support."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.image_processing.device_template import (
    CornerRadii,
    CutoutSpec,
    DeviceTemplate,
    DeviceTemplateCatalog,
    UVBounds,
    build_cutout_specs,
    classify_cutout_kind,
    estimate_corner_radii,
    slugify_model_name,
)
from src.image_processing.mesh import (
    AdaptiveMeshBuilder,
    ControlMesh,
    _sample_rounded_quad_perimeter,
)
from src.image_processing.template_cache import CoverTemplate, TemplateCache


def _rounded_rect_mask(h=400, w=220, radius=28, pad=20):
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = pad, pad, w - pad - 1, h - pad - 1
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


def test_corner_radii_roundtrip():
    radii = CornerRadii(tl=8.0, tr=9.0, br=7.5, bl=8.5)
    restored = CornerRadii.from_dict(radii.to_dict())
    assert restored.as_tuple() == pytest.approx(radii.as_tuple())
    assert 7.0 < restored.median() < 9.5


def test_device_template_json_roundtrip(tmp_path: Path):
    catalog = DeviceTemplateCatalog(tmp_path)
    template = DeviceTemplate(
        model_id="redmi-note-demo",
        display_name="Redmi Note Demo",
        aspect=0.48,
        corner_radii=CornerRadii(tl=10, tr=11, br=9, bl=10),
        phone_contours=[[[0.1, 0.05], [0.9, 0.05], [0.9, 0.95], [0.1, 0.95]]],
        cover_contours=[[[0.12, 0.07], [0.88, 0.07], [0.88, 0.93], [0.12, 0.93]]],
        cutouts=[
            CutoutSpec(
                kind="camera",
                contour=[[0.2, 0.12], [0.45, 0.12], [0.45, 0.28], [0.2, 0.28]],
            ),
            CutoutSpec(
                kind="button",
                contour=[[0.88, 0.35], [0.92, 0.35], [0.92, 0.48], [0.88, 0.48]],
            ),
        ],
        uv_bounds=UVBounds(
            quad=[[0.12, 0.07], [0.88, 0.07], [0.88, 0.93], [0.12, 0.93]]
        ),
        rows=13,
        cols=11,
        fingerprint="abc123",
    )
    catalog.save(template)
    loaded = catalog.load("redmi-note-demo")
    assert loaded is not None
    assert loaded.display_name == "Redmi Note Demo"
    assert loaded.corner_radii.tr == pytest.approx(11.0)
    assert len(loaded.camera_cutouts()) == 1
    assert len(loaded.button_cutouts()) == 1
    assert len(loaded.uv_bounds.quad) == 4


def test_cover_template_v3_fields(tmp_path: Path):
    cache = TemplateCache(tmp_path)
    image = np.full((400, 220, 3), 40, dtype=np.uint8)
    mask = _rounded_rect_mask()
    image[mask > 0] = (180, 180, 180)
    mesh = ControlMesh.from_quad(
        np.array(
            [[20, 20], [199, 20], [199, 379], [20, 379]], dtype=np.float32
        ),
        7,
        5,
    )
    radii = CornerRadii(tl=8, tr=10, br=7, bl=9)
    saved = cache.save(
        image,
        mesh,
        exclusion_mask=None,
        silhouette=mask,
        cover_mask=mask,
        printable_mask=mask,
        corner_radii=radii,
        phone_mask=mask,
        model_id="demo-phone",
        hardware_contours=[
            np.array(
                [[40, 40], [100, 40], [100, 100], [40, 100]], dtype=np.float32
            )
        ],
    )
    assert saved.version >= 3
    assert saved.radii().tr == pytest.approx(10.0)
    assert saved.phone_contours
    assert saved.uv_bounds
    assert saved.model_id == "demo-phone"

    loaded = cache.load(saved.fingerprint)
    assert loaded is not None
    assert loaded.radii().tl == pytest.approx(8.0)
    region = cache.materialise(loaded, image.shape[:2])
    assert region.mesh.rows == 7
    assert getattr(region, "corner_radii").tr == pytest.approx(10.0)


def test_estimate_corner_radii_near_uniform():
    mask = _rounded_rect_mask(radius=30)
    radii = estimate_corner_radii(mask)
    vals = radii.as_tuple()
    assert all(2.5 <= v <= 22.0 for v in vals)
    # Four corners of a symmetric rounded rect should be similar.
    assert max(vals) - min(vals) < 6.0


def test_force_rounded_perimeter_per_corner():
    quad = np.array(
        [[10, 10], [210, 10], [210, 410], [10, 410]], dtype=np.float32
    )
    mesh = ControlMesh.from_quad(quad, 9, 7)
    uneven = AdaptiveMeshBuilder.force_rounded_perimeter(
        mesh, 8.0, corner_radii=(6.0, 14.0, 6.0, 14.0)
    )
    even = AdaptiveMeshBuilder.force_rounded_perimeter(mesh, 8.0)
    # Uneven radii must move the top-right / bottom-left relative to uniform.
    assert not np.allclose(uneven.points, even.points, atol=0.5)

    outline = _sample_rounded_quad_perimeter(
        quad, 8.0, corner_radii=(6.0, 14.0, 6.0, 14.0)
    )
    assert outline.shape[0] >= 40


def test_classify_cutout_kind_camera_vs_button():
    cover = np.array(
        [[20, 20], [200, 20], [200, 400], [20, 400]], dtype=np.float32
    )
    camera = np.array(
        [[40, 40], [110, 40], [110, 110], [40, 110]], dtype=np.float32
    )
    button = np.array(
        [[195, 160], [205, 160], [205, 220], [195, 220]], dtype=np.float32
    )
    assert classify_cutout_kind(camera, cover) in ("camera", "flash")
    assert classify_cutout_kind(button, cover) == "button"
    specs = build_cutout_specs([camera, button], cover, 220, 420)
    kinds = {s.kind for s in specs}
    assert "camera" in kinds or "flash" in kinds
    assert "button" in kinds


def test_slugify_and_legacy_cover_template_load(tmp_path: Path):
    assert slugify_model_name("Redmi Note 13 Pro+") == "redmi-note-13-pro"
    # Legacy v2 without Phase 1 fields still loads.
    legacy = {
        "fingerprint": "deadbeefcafebabe01234567",
        "aspect": 0.5,
        "rows": 7,
        "cols": 5,
        "mesh_points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        "exclusion_contours": [],
        "cover_contours": [],
        "printable_contours": [],
        "margin_percent": 1.0,
        "corner_radius_percent": 8.0,
        "confidence": 0.9,
        "updated_at": 1.0,
        "version": 2,
    }
    path = tmp_path / f"{legacy['fingerprint']}.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = CoverTemplate.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    assert loaded.radii().median() == pytest.approx(8.0)
    assert loaded.version == 2


def test_catalog_capture_from_session(tmp_path: Path):
    catalog = DeviceTemplateCatalog(tmp_path)
    phone = np.full((400, 220, 3), 30, dtype=np.uint8)
    mask = _rounded_rect_mask()
    phone[mask > 0] = (160, 160, 160)
    mesh = ControlMesh.from_quad(
        np.array(
            [[20, 20], [199, 20], [199, 379], [20, 379]], dtype=np.float32
        ),
        11,
        9,
    )
    cam = np.array(
        [[40, 40], [100, 40], [100, 95], [40, 95]], dtype=np.float32
    )
    saved = catalog.capture_from_session(
        phone_image=phone,
        mesh=mesh,
        phone_mask=mask,
        cover_mask=mask,
        printable_mask=mask,
        hardware_contours=[cam],
        corner_radii=CornerRadii(tl=9, tr=9, br=8, bl=8),
        model_id="session-capture",
        display_name="Session Capture",
        fingerprint="fp-session-1",
    )
    assert saved.model_id == "session-capture"
    assert Path(tmp_path / "session-capture.json").is_file()
    assert saved.phone_contours
    assert saved.uv_bounds.quad
    assert any(c.kind in ("camera", "flash", "other") for c in saved.cutouts)


def test_calibrate_corner_radii_shrinks_over_rounded():
    """Geometric arc radius must shrink to fit the real phone silhouette."""
    mask = np.zeros((200, 120), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (100, 180), 255, -1)
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.erode(mask, erode, iterations=1)
    quad = np.array(
        [[20, 20], [100, 20], [100, 180], [20, 180]], dtype=np.float32
    )
    percent, _radii = AdaptiveMeshBuilder.calibrate_corner_radii_from_silhouette(
        mask, quad, 18.0
    )
    assert percent < 16.0
    assert percent >= 2.0
