"""Phase 5 adaptive mesh — corner-packed UV, density from curvature."""

from __future__ import annotations

import numpy as np
import pytest

from src.image_processing.mesh import (
    AdaptiveMeshBuilder,
    ControlMesh,
    DEFAULT_CORNER_BIAS,
    DEFAULT_MESH_COLS,
    DEFAULT_MESH_ROWS,
    MeshWarper,
    adaptive_axis_samples,
    adaptive_density_for_corners,
)
from src.image_processing.cover_surface import CoverSurfaceEngine
from src.image_processing.curved_uv import CurvedUVParams, remap_grid


def test_defaults_are_dense_for_phase5():
    assert DEFAULT_MESH_ROWS >= 21
    assert DEFAULT_MESH_COLS >= 15
    assert DEFAULT_CORNER_BIAS >= 0.65


def test_adaptive_axis_samples_pack_ends():
    t = adaptive_axis_samples(11, corner_bias=0.7)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(1.0)
    # Strictly increasing.
    assert np.all(np.diff(t) > 0)
    gaps = np.diff(t)
    # End gaps smaller than middle when biased.
    assert gaps[0] < gaps[len(gaps) // 2]
    assert gaps[-1] < gaps[len(gaps) // 2]


def test_adaptive_axis_uniform_when_bias_zero():
    t = adaptive_axis_samples(9, corner_bias=0.0)
    expected = np.linspace(0.0, 1.0, 9)
    assert np.allclose(t, expected, atol=1e-5)


def test_from_quad_adaptive_differs_from_uniform():
    quad = np.array(
        [[10, 10], [210, 10], [210, 410], [10, 410]], dtype=np.float32
    )
    adaptive = ControlMesh.from_quad(quad, 11, 9, adaptive=True, corner_bias=0.7)
    uniform = ControlMesh.from_quad(quad, 11, 9, adaptive=False)
    assert adaptive.points.shape == uniform.points.shape
    assert not np.allclose(adaptive.points, uniform.points, atol=0.5)
    # Corners identical.
    assert np.allclose(
        adaptive.corner_points(), uniform.corner_points(), atol=1e-3
    )


def test_force_rounded_adaptive_packs_near_corners():
    quad = np.array(
        [[20, 20], [200, 20], [200, 400], [20, 400]], dtype=np.float32
    )
    base = ControlMesh.from_quad(quad, 13, 11, adaptive=False)
    packed = AdaptiveMeshBuilder.force_rounded_perimeter(
        base, 12.0, adaptive=True, corner_bias=0.75
    )
    uniform = AdaptiveMeshBuilder.force_rounded_perimeter(
        base, 12.0, adaptive=False
    )
    assert not np.allclose(packed.points, uniform.points, atol=0.5)
    # Top-edge: first interior vert should sit closer to TL than uniform.
    pg = packed.points.reshape(13, 11, 2)
    ug = uniform.points.reshape(13, 11, 2)
    # Distance along top from TL to first interior sample.
    d_packed = float(np.linalg.norm(pg[0, 1] - pg[0, 0]))
    d_uniform = float(np.linalg.norm(ug[0, 1] - ug[0, 0]))
    assert d_packed < d_uniform


def test_densify_for_curvature_preserves_topology():
    quad = np.array(
        [[15, 15], [185, 15], [185, 385], [15, 385]], dtype=np.float32
    )
    mesh = ControlMesh.from_quad(quad, 15, 11)
    out = AdaptiveMeshBuilder.densify_for_curvature(mesh, 10.0)
    assert out.rows == mesh.rows
    assert out.cols == mesh.cols
    assert out.points.shape == mesh.points.shape


def test_adaptive_density_grows_with_soft_corners():
    r_soft = adaptive_density_for_corners(18.0)
    r_sharp = adaptive_density_for_corners(4.0)
    assert r_soft[0] >= r_sharp[0]
    assert r_soft[1] >= r_sharp[1]


def test_recommend_mesh_density_uses_corners():
    mask = np.zeros((400, 220), dtype=np.uint8)
    mask[30:370, 25:195] = 255
    soft = CoverSurfaceEngine.recommend_mesh_density(mask, 16.0)
    sharp = CoverSurfaceEngine.recommend_mesh_density(mask, 4.0)
    assert soft[0] >= sharp[0]


def test_source_points_adaptive_aligns_with_mesh():
    """Source + dest both use adaptive UV — centre sample still near centre."""
    design_shape = (300, 200)
    rows, cols = 11, 9
    source = MeshWarper.source_points(
        design_shape, rows, cols, target_aspect=0.55, fit_mode="fill"
    )
    mid = (rows // 2) * cols + (cols // 2)
    # Design centre ≈ (100, 150)
    assert abs(source[mid, 0] - 100.0) < 25.0
    assert abs(source[mid, 1] - 150.0) < 25.0


def test_remap_grid_adaptive_shape():
    grid = remap_grid(
        9, 7, CurvedUVParams(enabled=True, rim_uv=0.06), adaptive=True
    )
    assert grid.shape == (9 * 7, 2)
    mid = (9 // 2) * 7 + (7 // 2)
    assert grid[mid, 0] == pytest.approx(0.5, abs=0.05)
