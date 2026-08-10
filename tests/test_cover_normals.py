"""Phase 4 cover normals — Blinn lighting, soft AO, micro-displacement."""

from __future__ import annotations

import numpy as np
import pytest

from src.image_processing.materials import (
    CoverNormalField,
    LightingProfile,
    MaterialProfile,
    MaterialRenderingEngine,
    MATERIALS,
)
from src.utils.helpers import luminance


def _rounded_cover_mask(h=320, w=180, pad=18, radius=22):
    mask = np.zeros((h, w), dtype=np.float32)
    x0, y0, x1, y1 = pad, pad, w - pad - 1, h - pad - 1
    import cv2
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(binary, (x0 + radius, y0), (x1 - radius, y1), 255, -1)
    cv2.rectangle(binary, (x0, y0 + radius), (x1, y1 - radius), 255, -1)
    for cx, cy in (
        (x0 + radius, y0 + radius),
        (x1 - radius, y0 + radius),
        (x0 + radius, y1 - radius),
        (x1 - radius, y1 - radius),
    ):
        cv2.circle(binary, (cx, cy), radius, 255, -1)
    return binary.astype(np.float32) / 255.0


def test_build_cover_normals_flat_face_up():
    mask = _rounded_cover_mask()
    field = MaterialRenderingEngine.build_cover_normals(
        mask, None, bevel_amp=0.55, micro_disp=0.0
    )
    assert isinstance(field, CoverNormalField)
    # Centre of the cover should face the camera (nz ≈ 1, nx/ny ≈ 0).
    cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    assert field.nz[cy, cx] == pytest.approx(1.0, abs=0.08)
    assert abs(field.nx[cy, cx]) < 0.12
    assert abs(field.ny[cy, cx]) < 0.12
    # Rim band should tilt (nonzero slope somewhere near edge).
    assert float(np.max(np.abs(field.nx))) > 0.05
    assert float(np.max(field.height)) > 0.05


def test_build_cover_normals_with_cutout_well():
    import cv2
    mask = _rounded_cover_mask()
    excl = np.zeros_like(mask)
    cv2.circle(excl, (70, 70), 22, 1.0, -1)
    field = MaterialRenderingEngine.build_cover_normals(
        mask, excl, bevel_amp=0.5, cutout_amp=0.4
    )
    # Height near the cutout lip should rise.
    assert float(np.max(field.height)) > float(
        np.max(
            MaterialRenderingEngine.build_cover_normals(
                mask, None, bevel_amp=0.5, cutout_amp=0.0
            ).height
        )
    ) * 0.85


def test_shade_from_normals_lifts_lit_rim_not_center():
    mask = _rounded_cover_mask()
    design = np.full((*mask.shape, 3), 0.45, dtype=np.float32)
    lighting = LightingProfile(
        "Studio", direction=(-0.35, -0.55), highlight_scale=1.0, softness=0.4
    )
    material = MATERIALS["Glossy"]
    normals = MaterialRenderingEngine.build_cover_normals(mask, None)
    shaded = MaterialRenderingEngine.shade_from_normals(
        design,
        normals,
        lighting,
        material,
        mask,
        specular_gain=0.40,
        diffuse_gain=0.25,
        ao_strength=0.12,
        opaque=True,
    )
    cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    center = float(luminance(shaded)[cy, cx])
    # Find a rim pixel with coverage.
    binary = (mask > 0.5).astype(np.uint8)
    import cv2
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    rim = (dist > 1.5) & (dist < 6.0) & (mask > 0.5)
    if int(np.count_nonzero(rim)) > 10:
        rim_lum = float(np.mean(luminance(shaded)[rim]))
        # Rim should not collapse darker than a charcoal outline.
        assert rim_lum > 0.25
    # Centre stays roughly the design colour (no bleach).
    assert 0.30 < center < 0.62


def test_apply_uses_normal_lighting_by_default():
    engine = MaterialRenderingEngine()
    h, w = 240, 140
    mask = _rounded_cover_mask(h, w)
    design = np.full((h, w, 3), 0.5, dtype=np.float32)
    phone = np.full((h, w, 3), 0.35, dtype=np.float32)
    out, contact = engine.apply(
        design,
        phone,
        mask,
        material=MATERIALS["Glossy"],
        lighting=LightingProfile("Studio"),
        settings={
            "normal_lighting": 1.0,
            "rim_bevel": 55.0,
            "ao_strength": 12.0,
            "micro_disp": 8.0,
        },
    )
    assert out.shape == design.shape
    assert contact.shape == mask.shape
    assert float(np.mean(out[mask > 0.5])) > 0.2


def test_apply_legacy_path_when_normals_off():
    engine = MaterialRenderingEngine()
    h, w = 200, 120
    mask = _rounded_cover_mask(h, w)
    design = np.full((h, w, 3), 0.5, dtype=np.float32)
    phone = np.full((h, w, 3), 0.4, dtype=np.float32)
    out, _ = engine.apply(
        design,
        phone,
        mask,
        material=MATERIALS["Matte"],
        lighting=LightingProfile("Soft"),
        settings={"normal_lighting": 0.0},
    )
    assert out.shape == design.shape


def test_opaque_no_bleach_with_normals():
    """Regression: glossy + normals must not wash print to near-white."""
    engine = MaterialRenderingEngine()
    h, w = 260, 150
    mask = _rounded_cover_mask(h, w)
    design = np.zeros((h, w, 3), dtype=np.float32)
    design[:] = (0.15, 0.35, 0.75)  # strong chroma blue
    phone = np.full((h, w, 3), 0.5, dtype=np.float32)
    out, _ = engine.apply(
        design,
        phone,
        mask,
        material=MATERIALS["Glossy"],
        lighting=LightingProfile("Premium"),
        settings={"normal_lighting": 1.0, "rim_bevel": 60.0},
    )
    covered = out[mask > 0.55]
    mean = covered.mean(axis=0)
    # Still recognisably blue — not washed to grey/white.
    assert mean[0] < mean[2]
    assert float(mean.max()) < 0.92
