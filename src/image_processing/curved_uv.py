"""
Phase 2 curved cover UV — bevel rim + flat back foreshortening.

Destination mesh stays the photo-space cage (handles / Perfect Finish intact).
Only the *source* sample grid is remapped so artwork wraps the moulded rim
instead of stretching like a flat sticker.

Model:
  - Flat back: identity UV inside an inset rounded rectangle
  - Bevel rim: quarter-cylinder unwrap along the outward normal
  - Corners: same distance field uses Phase 1 TL/TR/BR/BL radii
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .device_template import CornerRadii


# Default rim band as a fraction of cover UV short-edge (face + wrap).
DEFAULT_RIM_UV = 0.055
# How strongly the rim pulls source samples into the wrap bleed (0–1.5).
DEFAULT_BEVEL_STRENGTH = 0.92


@dataclass
class CurvedUVParams:
    """Tunable wrap foreshortening for one cover."""

    rim_uv: float = DEFAULT_RIM_UV
    bevel_strength: float = DEFAULT_BEVEL_STRENGTH
    corner_radii: Optional[CornerRadii] = None
    enabled: bool = True

    def clamped(self) -> "CurvedUVParams":
        return CurvedUVParams(
            rim_uv=float(np.clip(self.rim_uv, 0.02, 0.14)),
            bevel_strength=float(np.clip(self.bevel_strength, 0.0, 1.5)),
            corner_radii=self.corner_radii,
            enabled=bool(self.enabled),
        )


def _radii_uv(
    radii: Optional[CornerRadii], fallback: float = 0.08
) -> Tuple[float, float, float, float]:
    """Corner radii as UV fractions of the unit square short edge."""
    if radii is None:
        r = float(np.clip(fallback, 0.02, 0.42))
        return r, r, r, r
    # CornerRadii are percent of short edge (same as settings slider).
    tl, tr, br, bl = radii.as_tuple()
    return (
        float(np.clip(tl / 100.0, 0.02, 0.42)),
        float(np.clip(tr / 100.0, 0.02, 0.42)),
        float(np.clip(br / 100.0, 0.02, 0.42)),
        float(np.clip(bl / 100.0, 0.02, 0.42)),
    )


def rounded_rect_sdf(
    u: float,
    v: float,
    radii: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    """
    Signed distance to a unit-square rounded rect (positive inside).

    Also returns a unit outward normal in UV (points toward exterior).
    """
    rtl, rtr, rbr, rbl = radii
    uu = float(np.clip(u, 0.0, 1.0))
    vv = float(np.clip(v, 0.0, 1.0))

    # Pick the local corner radius for the nearest corner quadrant.
    if uu < 0.5 and vv < 0.5:
        r = rtl
        cx, cy = r, r
        in_corner = uu < r and vv < r
        qx, qy = uu - r, vv - r
    elif uu >= 0.5 and vv < 0.5:
        r = rtr
        cx, cy = 1.0 - r, r
        in_corner = uu > 1.0 - r and vv < r
        qx, qy = uu - (1.0 - r), vv - r
    elif uu >= 0.5 and vv >= 0.5:
        r = rbr
        cx, cy = 1.0 - r, 1.0 - r
        in_corner = uu > 1.0 - r and vv > 1.0 - r
        qx, qy = uu - (1.0 - r), vv - (1.0 - r)
    else:
        r = rbl
        cx, cy = r, 1.0 - r
        in_corner = uu < r and vv > 1.0 - r
        qx, qy = uu - r, vv - (1.0 - r)

    if in_corner:
        dist_from_center = float(np.hypot(qx, qy))
        # Inside the disc → positive; outside (past arc) → negative.
        sdf = r - dist_from_center
        if dist_from_center > 1e-8:
            # Outward = away from corner centre.
            nx, ny = qx / dist_from_center, qy / dist_from_center
        else:
            nx, ny = 0.0, 0.0
        return sdf, nx, ny

    # Straight-edge region: distance to nearest outer edge.
    d_left = uu
    d_right = 1.0 - uu
    d_top = vv
    d_bot = 1.0 - vv
    sdf = min(d_left, d_right, d_top, d_bot)
    # Outward normal toward the closest exterior.
    if sdf == d_left:
        return sdf, -1.0, 0.0
    if sdf == d_right:
        return sdf, 1.0, 0.0
    if sdf == d_top:
        return sdf, 0.0, -1.0
    return sdf, 0.0, 1.0


def remap_uv(
    u: float,
    v: float,
    params: CurvedUVParams,
) -> Tuple[float, float]:
    """
    Map cover parametric UV → design sample UV with rim wrap foreshortening.

    Flat back stays near-identity. On the bevel band, source UV is pushed
    outward along the silhouette normal so the rim shows wrap bleed instead
    of affine-stretched face art.
    """
    p = params.clamped()
    if not p.enabled or p.bevel_strength < 1e-4:
        return float(u), float(v)

    radii = _radii_uv(p.corner_radii)
    sdf, nx, ny = rounded_rect_sdf(float(u), float(v), radii)
    rim = p.rim_uv

    # Outside / on the boundary: stay on the unit-square edge.
    # Pushing UV past 0/1 requested pixels outside the artwork and the
    # sampler invented a stretched rim from those out-of-bounds hits.
    if sdf <= 0.0:
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

    if sdf >= rim:
        # Flat back — mild inset so the face doesn't fight the wrap band.
        # Keep identity for Smart Fit stability.
        return float(u), float(v)

    # t: 0 at flat/rim junction → 1 at outer silhouette.
    t = float(np.clip(1.0 - sdf / rim, 0.0, 1.0))
    # Quarter-cylinder: screen foreshortening ~ 1 - cos(theta).
    # Source advances into the wrap as R * (1 - cos(t * π/2)).
    theta = t * (0.5 * np.pi)
    wrap = rim * p.bevel_strength * (1.0 - float(np.cos(theta)))
    # Soften mid-rim to avoid a hard kink at the flat junction.
    wrap *= 0.35 + 0.65 * (t ** 0.85)
    return (
        float(np.clip(u + nx * wrap, 0.0, 1.0)),
        float(np.clip(v + ny * wrap, 0.0, 1.0)),
    )


def remap_grid(
    rows: int,
    cols: int,
    params: CurvedUVParams,
    *,
    adaptive: bool = True,
) -> np.ndarray:
    """
    (rows*cols, 2) array of remapped UV coords for a regular mesh topology.

    Phase 5: ``adaptive`` uses the same corner-packed axis samples as the
    destination ControlMesh so wrap foreshortening matches dense corner cells.
    """
    if adaptive and rows >= 3 and cols >= 3:
        from .mesh import adaptive_axis_samples
        u_s = adaptive_axis_samples(cols)
        v_s = adaptive_axis_samples(rows)
    else:
        u_s = np.linspace(0.0, 1.0, cols, dtype=np.float32)
        v_s = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    pts = np.zeros((rows * cols, 2), dtype=np.float32)
    idx = 0
    for row in range(rows):
        v = float(v_s[row])
        for col in range(cols):
            u = float(u_s[col])
            uu, vv = remap_uv(u, v, params)
            pts[idx, 0] = float(np.clip(uu, 0.0, 1.0))
            pts[idx, 1] = float(np.clip(vv, 0.0, 1.0))
            idx += 1
    return pts


def rim_concentrated_params(
    count: int, *, corner_bias: float = 0.55
) -> np.ndarray:
    """
    Parameter samples in [0, 1] denser near 0 and 1 (corners of each edge).

    Used when assigning mesh boundary verts so high-curvature corners get
    more affine triangles without changing rows×cols topology.
    """
    n = max(2, int(count))
    if n == 2:
        return np.array([0.0, 1.0], dtype=np.float32)
    # Smoothstep ease toward ends.
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    bias = float(np.clip(corner_bias, 0.0, 0.9))
    # Map through a raised-cosine that packs samples near 0 and 1.
    eased = 0.5 - 0.5 * np.cos(np.pi * t)
    return ((1.0 - bias) * t + bias * eased).astype(np.float32)


def estimate_rim_uv_from_margin(margin_percent: float) -> float:
    """Derive a sensible rim band from the detected print margin."""
    # margin_percent is typically 0–6; map into UV rim width.
    m = float(np.clip(margin_percent, 0.0, 12.0))
    return float(np.clip(0.038 + m * 0.0045, 0.035, 0.10))
