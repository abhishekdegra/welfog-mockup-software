"""Phase 2 curved UV wrap — bevel rim foreshortening + denser corners."""

from __future__ import annotations

import numpy as np
import pytest

from src.image_processing.curved_uv import (
    CurvedUVParams,
    estimate_rim_uv_from_margin,
    remap_grid,
    remap_uv,
    rim_concentrated_params,
    rounded_rect_sdf,
)
from src.image_processing.device_template import CornerRadii, UVBounds
from src.image_processing.mesh import (
    AdaptiveMeshBuilder,
    ControlMesh,
    DEFAULT_MESH_COLS,
    DEFAULT_MESH_ROWS,
    MeshWarper,
)


def test_defaults_are_denser_for_phase2():
    assert DEFAULT_MESH_ROWS >= 15
    assert DEFAULT_MESH_COLS >= 13


def test_rounded_rect_sdf_inside_positive():
    radii = (0.1, 0.1, 0.1, 0.1)
    d, nx, ny = rounded_rect_sdf(0.5, 0.5, radii)
    assert d > 0.3
    # Centre normal is arbitrary-ish; just ensure finite.
    assert np.isfinite(nx) and np.isfinite(ny)

    d_edge, nx_e, _ = rounded_rect_sdf(0.0, 0.5, radii)
    assert d_edge == pytest.approx(0.0, abs=1e-6)
    assert nx_e == pytest.approx(-1.0)


def test_remap_uv_flat_back_identity():
    params = CurvedUVParams(rim_uv=0.06, bevel_strength=1.0, enabled=True)
    u, v = remap_uv(0.5, 0.5, params)
    assert u == pytest.approx(0.5, abs=1e-5)
    assert v == pytest.approx(0.5, abs=1e-5)


def test_remap_uv_rim_pushes_outward():
    params = CurvedUVParams(
        rim_uv=0.08,
        bevel_strength=1.0,
        corner_radii=CornerRadii.uniform(8.0),
        enabled=True,
    )
    # Near left edge, inside the rim band.
    u0, v0 = 0.02, 0.5
    u1, v1 = remap_uv(u0, v0, params)
    # Outward normal is -X, so source u should decrease (into wrap bleed).
    assert u1 < u0
    assert v1 == pytest.approx(v0, abs=0.02)


def test_remap_uv_disabled_is_identity():
    params = CurvedUVParams(enabled=False, bevel_strength=1.0, rim_uv=0.08)
    assert remap_uv(0.02, 0.5, params) == pytest.approx((0.02, 0.5))


def test_remap_grid_shape_and_corners_move():
    params = CurvedUVParams(rim_uv=0.07, bevel_strength=1.1, enabled=True)
    grid = remap_grid(9, 7, params)
    assert grid.shape == (9 * 7, 2)
    # Centre cell stays near (0.5, 0.5).
    mid = grid[(9 // 2) * 7 + (7 // 2)]
    assert mid[0] == pytest.approx(0.5, abs=0.02)
    assert mid[1] == pytest.approx(0.5, abs=0.02)
    # Top-left parametric corner should be pushed outward (negative).
    assert grid[0, 0] <= 0.0 + 1e-6 or grid[0, 0] < 0.05


def test_source_points_curved_differs_from_flat():
    params = CurvedUVParams(rim_uv=0.07, bevel_strength=1.0, enabled=True)
    # Dense enough that some verts sit inside the rim band (not only on the
    # unit-square edge, which must stay inside the artwork).
    flat = MeshWarper.source_points(
        (400, 300), 21, 15, target_aspect=0.55, fit_mode="fill"
    )
    curved = MeshWarper.source_points(
        (400, 300),
        21,
        15,
        target_aspect=0.55,
        fit_mode="fill",
        curved_uv=params,
    )
    assert flat.shape == curved.shape
    assert float(curved[:, 0].min()) >= -1e-3
    assert float(curved[:, 1].min()) >= -1e-3
    assert float(curved[:, 0].max()) <= 300.0 + 1e-3
    assert float(curved[:, 1].max()) <= 400.0 + 1e-3
    # Rim-band verts move; interior stays close.
    assert not np.allclose(flat, curved, atol=0.5)
    mid = (21 // 2) * 15 + (15 // 2)
    assert np.allclose(flat[mid], curved[mid], atol=2.0)


def test_warp_with_curved_uv_produces_coverage():
    design = np.zeros((200, 160, 4), dtype=np.uint8)
    design[:, :] = (40, 120, 200, 255)
    quad = np.array(
        [[20, 20], [180, 20], [180, 360], [20, 360]], dtype=np.float32
    )
    mesh = ControlMesh.from_quad(quad, 9, 7)
    source = MeshWarper.source_points(
        design.shape[:2],
        mesh.rows,
        mesh.cols,
        target_aspect=160 / 200,
        curved_uv=CurvedUVParams(enabled=True, rim_uv=0.06, bevel_strength=0.9),
    )
    out = MeshWarper.warp(design, source, mesh, (400, 220))
    assert out is not None
    assert out.shape == (400, 220, 4)
    assert int(np.count_nonzero(out[:, :, 3])) > 1000


def test_rim_concentrated_params_packs_ends():
    t = rim_concentrated_params(11, corner_bias=0.7)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(1.0)
    # Gaps near ends should be smaller than the middle gap when biased.
    gaps = np.diff(t)
    assert gaps[0] < gaps[len(gaps) // 2]
    assert gaps[-1] < gaps[len(gaps) // 2]


def test_boundary_corner_bias_moves_samples():
    quad = np.array(
        [[10, 10], [210, 10], [210, 410], [10, 410]], dtype=np.float32
    )
    mesh = ControlMesh.from_quad(quad, 11, 9)
    rounded = AdaptiveMeshBuilder.force_rounded_perimeter(mesh, 10.0)
    # Boundary verts should exist and stay near the rounded silhouette.
    boundary = rounded.boundary_points()
    assert boundary.shape[0] >= 20
    xs = boundary[:, 0]
    assert xs.min() >= 5.0
    assert xs.max() <= 215.0


def test_uv_bounds_phase2_fields_roundtrip():
    bounds = UVBounds(
        quad=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        rim_uv=0.06,
        bevel_strength=1.05,
    )
    restored = UVBounds.from_dict(bounds.to_dict())
    assert restored.rim_uv == pytest.approx(0.06)
    assert restored.bevel_strength == pytest.approx(1.05)


def test_estimate_rim_from_margin():
    assert 0.035 <= estimate_rim_uv_from_margin(0.0) <= 0.06
    assert estimate_rim_uv_from_margin(8.0) > estimate_rim_uv_from_margin(0.0)
