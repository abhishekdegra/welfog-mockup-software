"""Editable control meshes and piecewise-affine image warping."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..utils.helpers import order_points, to_bgra


DEFAULT_MESH_ROWS = 21
DEFAULT_MESH_COLS = 15

# Phase 5: pack parametric samples toward cover corners (0 = uniform).
DEFAULT_CORNER_BIAS = 0.72


def adaptive_axis_samples(
    count: int, *, corner_bias: float = DEFAULT_CORNER_BIAS
) -> np.ndarray:
    """
    ``count`` samples in [0, 1] denser near both ends (high-curvature corners).

    Topology (rows×cols) stays fixed — only *where* verts sit along each axis
    changes, so edit handles and triangle indices stay compatible.
    """
    n = max(2, int(count))
    if n == 2:
        return np.array([0.0, 1.0], dtype=np.float32)
    bias = float(np.clip(corner_bias, 0.0, 0.90))
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    # Raised-cosine packs mass toward 0 and 1 without crossing.
    eased = 0.5 - 0.5 * np.cos(np.pi * t)
    samples = (1.0 - bias) * t + bias * eased
    samples[0] = 0.0
    samples[-1] = 1.0
    # Enforce strict monotonicity (numerical safety).
    for i in range(1, n):
        if samples[i] <= samples[i - 1]:
            samples[i] = samples[i - 1] + 1e-4
    samples[-1] = 1.0
    samples = (samples - samples[0]) / max(samples[-1] - samples[0], 1e-9)
    return samples.astype(np.float32)


def adaptive_density_for_corners(
    corner_radius_percent: float,
    *,
    base_rows: int = DEFAULT_MESH_ROWS,
    base_cols: int = DEFAULT_MESH_COLS,
) -> Tuple[int, int]:
    """
    Softer (larger) corner radii need more verts to approximate the arc.

    Returns clamped (rows, cols) for production warp cost.
    """
    r = float(np.clip(corner_radius_percent, 2.0, 28.0))
    # Extra rows/cols when corners are very rounded.
    bonus = int(round(max(0.0, (r - 7.0) / 4.0)))
    rows = int(np.clip(base_rows + bonus, 15, 25))
    cols = int(np.clip(base_cols + max(0, bonus - 1), 13, 19))
    return rows, cols



@dataclass
class ControlMesh:
    """Regular-topology mesh whose destination vertices are freely editable."""

    points: np.ndarray
    rows: int = DEFAULT_MESH_ROWS
    cols: int = DEFAULT_MESH_COLS

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        if points.shape == (self.rows, self.cols, 2):
            points = points.reshape(-1, 2)
        if points.shape != (self.rows * self.cols, 2):
            raise ValueError(
                f"Expected {(self.rows * self.cols, 2)} mesh points, "
                f"received {points.shape}"
            )
        self.points = points.copy()

    @classmethod
    def from_quad(
        cls,
        quad: np.ndarray,
        rows: int = DEFAULT_MESH_ROWS,
        cols: int = DEFAULT_MESH_COLS,
        *,
        adaptive: bool = True,
        corner_bias: float = DEFAULT_CORNER_BIAS,
    ) -> "ControlMesh":
        """
        Create a bilinearly interpolated grid inside TL/TR/BR/BL corners.

        Phase 5: ``adaptive=True`` packs UV samples toward the four corners so
        piecewise-affine triangles better follow rounded silhouettes.
        """
        tl, tr, br, bl = order_points(quad)
        if adaptive and rows >= 3 and cols >= 3:
            u_s = adaptive_axis_samples(cols, corner_bias=corner_bias)
            v_s = adaptive_axis_samples(rows, corner_bias=corner_bias)
        else:
            u_s = np.linspace(0.0, 1.0, cols, dtype=np.float32)
            v_s = np.linspace(0.0, 1.0, rows, dtype=np.float32)
        points = []
        for row in range(rows):
            v = float(v_s[row])
            left = tl * (1.0 - v) + bl * v
            right = tr * (1.0 - v) + br * v
            for col in range(cols):
                u = float(u_s[col])
                points.append(left * (1.0 - u) + right * u)
        return cls(np.asarray(points, dtype=np.float32), rows, cols)

    def copy(self) -> "ControlMesh":
        """Independent copy of this mesh."""
        return ControlMesh(self.points.copy(), self.rows, self.cols)

    def scaled(self, factor: float) -> "ControlMesh":
        """Mesh scaled uniformly into another image resolution."""
        return ControlMesh(self.points * float(factor), self.rows, self.cols)

    def inset(self, percent: float) -> "ControlMesh":
        """Shrink or grow the complete mesh around its centroid."""
        center = self.points.mean(axis=0)
        factor = 1.0 - float(np.clip(percent, -50.0, 50.0)) / 100.0
        return ControlMesh(
            (self.points - center) * factor + center, self.rows, self.cols
        )

    def boundary_indices(self) -> np.ndarray:
        """Clockwise perimeter vertex indices, without repeating corners."""
        top = [self.index(0, col) for col in range(self.cols)]
        right = [
            self.index(row, self.cols - 1) for row in range(1, self.rows)
        ]
        bottom = [
            self.index(self.rows - 1, col)
            for col in range(self.cols - 2, -1, -1)
        ]
        left = [
            self.index(row, 0) for row in range(self.rows - 2, 0, -1)
        ]
        return np.asarray(top + right + bottom + left, dtype=np.int32)

    def boundary_points(self) -> np.ndarray:
        """Clockwise perimeter coordinates."""
        return self.points[self.boundary_indices()]

    def corner_points(self) -> np.ndarray:
        """Legacy four corners ordered TL/TR/BR/BL."""
        return self.points[
            [
                self.index(0, 0),
                self.index(0, self.cols - 1),
                self.index(self.rows - 1, self.cols - 1),
                self.index(self.rows - 1, 0),
            ]
        ].copy()

    def triangles(self) -> np.ndarray:
        """Two consistently wound triangles for every mesh cell."""
        triangles = []
        for row in range(self.rows - 1):
            for col in range(self.cols - 1):
                top_left = self.index(row, col)
                top_right = self.index(row, col + 1)
                bottom_left = self.index(row + 1, col)
                bottom_right = self.index(row + 1, col + 1)

                # Alternating diagonals avoid a visible directional bias.
                if (row + col) % 2 == 0:
                    triangles.extend(
                        [
                            (top_left, top_right, bottom_right),
                            (top_left, bottom_right, bottom_left),
                        ]
                    )
                else:
                    triangles.extend(
                        [
                            (top_left, top_right, bottom_left),
                            (top_right, bottom_right, bottom_left),
                        ]
                    )
        return np.asarray(triangles, dtype=np.int32)

    def normalized_points(self, width: int, height: int) -> np.ndarray:
        """Vertices represented in 0-1 image coordinates for the Qt editor."""
        result = self.points.copy()
        result[:, 0] /= max(int(width), 1)
        result[:, 1] /= max(int(height), 1)
        return result

    @classmethod
    def from_normalized(
        cls,
        points: np.ndarray,
        width: int,
        height: int,
        rows: int,
        cols: int,
    ) -> "ControlMesh":
        """Create a mesh from editor-space 0-1 coordinates."""
        result = np.asarray(points, dtype=np.float32).copy()
        result[:, 0] *= max(int(width), 1)
        result[:, 1] *= max(int(height), 1)
        return cls(result, rows, cols)

    def index(self, row: int, col: int) -> int:
        """Flat vertex index for a row and column."""
        return row * self.cols + col

    def insert_column_after(self, col: int) -> "ControlMesh":
        """Insert a new column between `col` and `col + 1`."""
        if col < 0 or col >= self.cols - 1:
            raise ValueError("Column insert requires an interior edge")
        grid = self.points.reshape(self.rows, self.cols, 2)
        new_cols = self.cols + 1
        new_grid = np.zeros((self.rows, new_cols, 2), dtype=np.float32)
        new_grid[:, : col + 1] = grid[:, : col + 1]
        new_grid[:, col + 1] = 0.5 * (grid[:, col] + grid[:, col + 1])
        new_grid[:, col + 2 :] = grid[:, col + 1 :]
        return ControlMesh(new_grid.reshape(-1, 2), self.rows, new_cols)

    def insert_row_after(self, row: int) -> "ControlMesh":
        """Insert a new row between `row` and `row + 1`."""
        if row < 0 or row >= self.rows - 1:
            raise ValueError("Row insert requires an interior edge")
        grid = self.points.reshape(self.rows, self.cols, 2)
        new_rows = self.rows + 1
        new_grid = np.zeros((new_rows, self.cols, 2), dtype=np.float32)
        new_grid[: row + 1] = grid[: row + 1]
        new_grid[row + 1] = 0.5 * (grid[row] + grid[row + 1])
        new_grid[row + 2 :] = grid[row + 1 :]
        return ControlMesh(new_grid.reshape(-1, 2), new_rows, self.cols)

    def remove_column(self, col: int) -> "ControlMesh":
        """Remove an interior column (corners of the mesh stay intact)."""
        # Keep production density — never collapse to a coarse 3×3 cage.
        if col <= 0 or col >= self.cols - 1 or self.cols <= 7:
            raise ValueError("Cannot remove this column")
        grid = self.points.reshape(self.rows, self.cols, 2)
        kept = np.concatenate([grid[:, :col], grid[:, col + 1 :]], axis=1)
        return ControlMesh(kept.reshape(-1, 2), self.rows, self.cols - 1)

    def remove_row(self, row: int) -> "ControlMesh":
        """Remove an interior row (corners of the mesh stay intact)."""
        if row <= 0 or row >= self.rows - 1 or self.rows <= 7:
            raise ValueError("Cannot remove this row")
        grid = self.points.reshape(self.rows, self.cols, 2)
        kept = np.concatenate([grid[:row], grid[row + 1 :]], axis=0)
        return ControlMesh(kept.reshape(-1, 2), self.rows - 1, self.cols)


class AdaptiveMeshBuilder:
    """
    Build and refine a fixed-topology mesh that hugs the printable cover.

    Topology stays editor-friendly (rows×cols). Phase 5 adapts *where*
    vertices sit: corner-packed UV, edge snap, and rounded-arc sampling so
    high-curvature regions get more affine triangles without new UI tools.
    """

    @staticmethod
    def densify_for_curvature(
        mesh: ControlMesh,
        corner_radius_percent: float = 8.0,
        corner_radii: Optional[Tuple[float, float, float, float]] = None,
        *,
        corner_bias: float = DEFAULT_CORNER_BIAS,
    ) -> ControlMesh:
        """
        Rebuild a mesh with adaptive UV packing toward rounded corners.

        Same rows×cols — safe to call from Perfect Finish / detect.
        """
        if mesh is None or mesh.rows < 3 or mesh.cols < 3:
            return mesh
        return AdaptiveMeshBuilder.force_rounded_perimeter(
            mesh,
            corner_radius_percent,
            corner_radii=corner_radii,
            adaptive=True,
            corner_bias=corner_bias,
        )

    @staticmethod
    def build(
        printable_mask: np.ndarray,
        rows: int = DEFAULT_MESH_ROWS,
        cols: int = DEFAULT_MESH_COLS,
        corner_radius_percent: float = 6.0,
    ) -> ControlMesh:
        """Sample then refine a mesh on a printable cover mask."""
        from .region_detector import PrintableRegionDetector

        mesh = PrintableRegionDetector._sample_mesh(
            printable_mask, rows, cols
        )
        return AdaptiveMeshBuilder.refine(
            mesh, printable_mask, corner_radius_percent
        )

    @staticmethod
    def settle_edges(
        mesh: ControlMesh,
        cover_mask: Optional[np.ndarray] = None,
        corner_radius_percent: float = 6.0,
        *,
        max_move_fraction: float = 0.045,
    ) -> ControlMesh:
        """
        Production edge settle: stable quad → rounded perimeter → mid-side snap.

        Corners stay geometric arcs (never snapped to pixel stairs). Mid-side
        verts may soft-snap to a manufactured-smooth cover silhouette.
        """
        if mesh is None or mesh.points.size == 0:
            return mesh
        return AdaptiveMeshBuilder.production_perimeter(
            mesh,
            cover_mask,
            corner_radius_percent,
            max_move_fraction=max_move_fraction,
        )

    @staticmethod
    def calibrate_corner_radii_from_silhouette(
        mask: np.ndarray,
        quad: np.ndarray,
        corner_radius_percent: float,
        corner_radii: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        """
        Shrink geometric corner radii until the rounded perimeter fits inside
        the photo silhouette.

        Over-estimated radii tuck corners inward and leave white gaps at the
        phone rim — common when smooth gates balloon past the real hardware arc.
        """
        if mask is None or np.count_nonzero(mask) < 64:
            return corner_radius_percent, corner_radii
        binary = (mask > 0).astype(np.uint8)
        ordered = order_points(np.asarray(quad, dtype=np.float32))

        def outline_fits(
            percent: float,
            radii: Optional[Tuple[float, float, float, float]],
        ) -> bool:
            try:
                outline = _sample_rounded_quad_perimeter(
                    ordered,
                    percent,
                    samples_per_edge=36,
                    corner_radii=radii,
                )
            except Exception:
                return True
            h, w = binary.shape[:2]
            for point in outline:
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
                if not (0 <= x < w and 0 <= y < h):
                    return False
                if binary[y, x] == 0:
                    return False
            return True

        scale = 1.0
        best_percent = float(corner_radius_percent)
        best_radii = corner_radii
        for _ in range(14):
            percent = float(np.clip(corner_radius_percent * scale, 2.0, 28.0))
            radii = None
            if corner_radii is not None:
                radii = tuple(
                    float(np.clip(v * scale, 2.0, 28.0)) for v in corner_radii
                )
            if outline_fits(percent, radii):
                best_percent = percent
                best_radii = radii
                break
            scale *= 0.90
        else:
            best_percent = float(np.clip(corner_radius_percent * scale, 2.0, 28.0))
            if corner_radii is not None:
                best_radii = tuple(
                    float(np.clip(v * scale, 2.0, 28.0)) for v in corner_radii
                )
        return best_percent, best_radii

    @staticmethod
    def _expand_boundary_to_silhouette(
        mesh: ControlMesh,
        mask: np.ndarray,
        *,
        corner_only: bool = False,
        corner_span: int = 3,
    ) -> None:
        """
        Pull boundary verts outward when geometric arcs sit inset of the rim.

        Walks from the mask centroid through each boundary point to the
        outermost in-mask pixel along that ray — reaches corner apexes without
        relying on contour start order (which breaks arc-length mapping).

        ``corner_only``: only move verts near the four mesh corners so mid-side
        snaps (and side-button zones) stay put.
        """
        if mesh is None or mask is None or np.count_nonzero(mask) == 0:
            return
        binary = (mask > 0).astype(np.uint8)
        h, w = binary.shape[:2]
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return
        cx = float(xs.mean())
        cy = float(ys.mean())
        short = float(min(h, w))
        # Reach corner apexes that geometric UV arcs leave inset — typically a
        # few percent of the short edge on soft modern phones.
        max_push = max(8.0, short * 0.08)

        allowed: Optional[set] = None
        if corner_only and mesh.rows >= 2 and mesh.cols >= 2:
            boundary = list(mesh.boundary_indices())
            n = len(boundary)
            corner_ids = {
                mesh.index(0, 0),
                mesh.index(0, mesh.cols - 1),
                mesh.index(mesh.rows - 1, mesh.cols - 1),
                mesh.index(mesh.rows - 1, 0),
            }
            allowed = set()
            span = max(1, int(corner_span))
            for i, idx in enumerate(boundary):
                if idx not in corner_ids:
                    continue
                for d in range(-span, span + 1):
                    allowed.add(boundary[(i + d) % n])

        for index in mesh.boundary_indices():
            if allowed is not None and int(index) not in allowed:
                continue
            x = float(mesh.points[index, 0])
            y = float(mesh.points[index, 1])
            dx = x - cx
            dy = y - cy
            norm = float(np.hypot(dx, dy))
            if norm < 1e-3:
                continue
            ux, uy = dx / norm, dy / norm
            best_t = norm
            for t in np.linspace(norm, norm + max_push, 36):
                sx = cx + ux * float(t)
                sy = cy + uy * float(t)
                xi = int(np.clip(round(sx), 0, w - 1))
                yi = int(np.clip(round(sy), 0, h - 1))
                if binary[yi, xi]:
                    best_t = float(t)
                else:
                    break
            if best_t > norm + 0.25:
                mesh.points[index, 0] = cx + ux * best_t
                mesh.points[index, 1] = cy + uy * best_t

    @staticmethod
    def fit_mesh_to_mask(
        mesh: ControlMesh,
        cover_mask: np.ndarray,
        corner_radius_percent: float = 10.0,
        corner_radii: Optional[Tuple[float, float, float, float]] = None,
        *,
        expand_frac: float = 0.004,
        settle_first: bool = True,
    ) -> ControlMesh:
        """
        Rebuild mesh so the printable face fills ``cover_mask`` (full-bleed).

        Calibrates corner radii against the photo silhouette on the live mesh
        frame, then expands boundary verts to the outer rim without over-rounding.
        """
        if mesh is None or cover_mask is None or np.count_nonzero(cover_mask) < 64:
            return mesh
        quad = AdaptiveMeshBuilder._stable_quad_from_mask(cover_mask)
        if quad is None:
            return mesh
        cal_percent, cal_radii = (
            AdaptiveMeshBuilder.calibrate_corner_radii_from_silhouette(
                cover_mask, quad, corner_radius_percent, corner_radii
            )
        )
        settled = mesh
        if settle_first:
            settled = AdaptiveMeshBuilder.production_perimeter(
                mesh,
                cover_mask,
                corner_radius_percent=cal_percent,
                max_move_fraction=0.22,
                corner_radii=cal_radii,
                preserve_corner_arcs=True,
            )
        fitted = AdaptiveMeshBuilder.force_rounded_perimeter(
            settled,
            cal_percent,
            corner_radii=cal_radii,
            adaptive=True,
        )
        # Follow the real phone outline — geometric arcs alone leave white
        # corner gaps when the user's cage is larger than the hardware.
        AdaptiveMeshBuilder.conform_boundary_to_silhouette(fitted, cover_mask)
        AdaptiveMeshBuilder._expand_boundary_to_silhouette(
            fitted, cover_mask, corner_only=True
        )
        AdaptiveMeshBuilder._pull_boundary_inside(fitted, cover_mask)
        AdaptiveMeshBuilder._reinterpolate_interior(fitted)
        return fitted

    @staticmethod
    def production_perimeter(
        mesh: ControlMesh,
        cover_mask: Optional[np.ndarray] = None,
        corner_radius_percent: float = 10.0,
        *,
        max_move_fraction: float = 0.05,
        corner_radii: Optional[Tuple[float, float, float, float]] = None,
        preserve_corner_arcs: bool = False,
    ) -> ControlMesh:
        """
        Rebuild a product-smooth cover mesh from a stable perspective quad.

        Ignores jagged corner verts on the incoming mesh — those are the main
        reason Perfect Finish looked fake. Fits a clean oriented rect, applies
        true UV rounded corners, then optionally nudges mid-sides only.

        ``preserve_corner_arcs``: keep geometric quarter-circles — never map
        corners onto a jagged photo silhouette (kills stair-step finishes).
        """
        if mesh is None or mesh.rows < 2 or mesh.cols < 2:
            return mesh

        quad = None
        if cover_mask is not None and np.count_nonzero(cover_mask) > 64:
            quad = AdaptiveMeshBuilder._stable_quad_from_mask(cover_mask)
        if quad is None:
            quad = AdaptiveMeshBuilder._stable_quad_from_mesh(mesh)

        base = ControlMesh.from_quad(quad, mesh.rows, mesh.cols, adaptive=True)
        rounded = AdaptiveMeshBuilder.force_rounded_perimeter(
            base,
            corner_radius_percent,
            corner_radii=corner_radii,
            adaptive=True,
        )

        if cover_mask is None or np.count_nonzero(cover_mask) == 0:
            AdaptiveMeshBuilder._reinterpolate_interior(rounded)
            return rounded

        upright = AdaptiveMeshBuilder._quad_axis_deviation_deg(quad) <= 3.5
        snapped = rounded.copy()
        AdaptiveMeshBuilder._snap_midsides_to_mask(
            snapped,
            cover_mask,
            smooth=True,
            max_move_fraction=max_move_fraction,
        )
        AdaptiveMeshBuilder._straighten_sides(snapped, passes=2)
        AdaptiveMeshBuilder._smooth_edges(snapped, passes=2)
        if preserve_corner_arcs:
            if upright:
                # Product shots: geometric rounded rect on the AABB is the wrap
                # cage — then hug a lightly dilated rim so the four arcs reach
                # the visible product corners (not inset chords).
                snapped = AdaptiveMeshBuilder.force_rounded_perimeter(
                    ControlMesh.from_quad(
                        quad, mesh.rows, mesh.cols, adaptive=True
                    ),
                    corner_radius_percent,
                    corner_radii=corner_radii,
                    adaptive=True,
                )
                rim = cover_mask
                if cover_mask is not None and np.count_nonzero(cover_mask):
                    pad = max(2, int(round(min(cover_mask.shape[:2]) * 0.004)))
                    rim = cv2.dilate(
                        (cover_mask > 127).astype(np.uint8) * 255
                        if float(np.max(cover_mask)) > 1.5
                        else (cover_mask > 0.18).astype(np.uint8) * 255,
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
                        ),
                        iterations=1,
                    )
                    AdaptiveMeshBuilder._expand_boundary_to_silhouette(
                        snapped, rim, corner_only=True, corner_span=6
                    )
                    # Pull only into the dilated rim — never raw stair mask.
                    AdaptiveMeshBuilder._pull_boundary_inside(snapped, rim)
                AdaptiveMeshBuilder._reinterpolate_interior(snapped)
                return snapped
            # Perspective / rotated phones: expand to silhouette, then
            # rebuild arcs on the corrected cage.
            AdaptiveMeshBuilder._expand_boundary_to_silhouette(
                snapped, cover_mask, corner_only=False
            )
            AdaptiveMeshBuilder._pull_boundary_inside(snapped, cover_mask)
            snapped = AdaptiveMeshBuilder.force_rounded_perimeter(
                snapped,
                corner_radius_percent,
                corner_radii=corner_radii,
                adaptive=True,
            )
            AdaptiveMeshBuilder._expand_boundary_to_silhouette(
                snapped, cover_mask, corner_only=True
            )
            AdaptiveMeshBuilder._pull_boundary_inside(snapped, cover_mask)
        else:
            # Follow the real phone/case silhouette — do NOT re-lock UV rounded arcs
            # here or corners drift away from the photo (bottom gap + corner tear).
            AdaptiveMeshBuilder.conform_boundary_to_silhouette(snapped, cover_mask)
        AdaptiveMeshBuilder._reinterpolate_interior(snapped)
        if not preserve_corner_arcs:
            AdaptiveMeshBuilder._pull_boundary_inside(snapped, cover_mask)
        return snapped

    @staticmethod
    def outer_contour_polyline(
        mask: np.ndarray, *, smooth: bool = True
    ) -> Optional[np.ndarray]:
        """Dense outer silhouette polyline from a phone / cover mask."""
        if mask is None or mask.size == 0:
            return None
        if float(np.max(mask)) <= 1.5:
            binary = (mask > 0.18).astype(np.uint8)
        else:
            binary = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32
        )
        if pts.shape[0] < 8:
            return None
        if smooth and pts.shape[0] >= 16:
            pts = AdaptiveMeshBuilder._smooth_closed_polyline(
                pts,
                window=max(
                    9, min(35, int(pts.shape[0] // 22) * 2 + 1)
                ),
            )
        return pts

    @staticmethod
    def conform_boundary_to_silhouette(
        mesh: ControlMesh, mask: np.ndarray
    ) -> None:
        """
        Map the mesh boundary onto the photo silhouette by arc length.

        Works for any phone model — not a fixed rounded rectangle.
        """
        outline = AdaptiveMeshBuilder.outer_contour_polyline(mask, smooth=True)
        if outline is None or outline.shape[0] < 16:
            return
        AdaptiveMeshBuilder._assign_boundary_from_outline(mesh, outline)
        AdaptiveMeshBuilder._pull_boundary_inside(mesh, mask)

    @staticmethod
    def _rect_axis_deviation_deg(rect) -> float:
        """How far a minAreaRect is from axis-aligned (0 = upright)."""
        _center, _size, angle = rect
        a = abs(float(angle)) % 90.0
        return float(min(a, 90.0 - a))

    @staticmethod
    def _quad_axis_deviation_deg(quad: np.ndarray) -> float:
        """Top-edge tilt of an ordered TL-TR-BR-BL quad (degrees from horizontal)."""
        pts = order_points(np.asarray(quad, dtype=np.float32).reshape(4, 2))
        edge = pts[1] - pts[0]
        angle = abs(float(np.degrees(np.arctan2(edge[1], edge[0]))))
        a = angle % 90.0
        return float(min(a, 90.0 - a))

    @staticmethod
    def _aabb_quad_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
        """Axis-aligned bounding quad of a silhouette (equal L/R and T/B)."""
        binary = (mask > 0).astype(np.uint8)
        ys, xs = np.nonzero(binary)
        if ys.size < 16:
            return None
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        if (x2 - x1) < 8.0 or (y2 - y1) < 8.0:
            return None
        return order_points(
            np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )
        )

    @staticmethod
    def _tight_aabb_quad_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Phone AABB from dense row/column mass — ignores thin shadow spikes.

        Raw min/max AABB grows when GrabCut leaves a soft halo or a tilted
        blob; projection thresholds hug the solid device body instead.
        """
        binary = (mask > 127).astype(np.uint8)
        if np.count_nonzero(binary) < 64:
            return AdaptiveMeshBuilder._aabb_quad_from_mask(mask)
        # Light close so camera bites don't split the column mass.
        short = min(binary.shape[:2])
        k = max(3, (short // 90) | 1)
        solid = cv2.morphologyEx(
            binary * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
        solid = (solid > 0).astype(np.uint8)
        col = solid.sum(axis=0).astype(np.float64)
        row = solid.sum(axis=1).astype(np.float64)
        if float(col.max()) < 8.0 or float(row.max()) < 8.0:
            return AdaptiveMeshBuilder._aabb_quad_from_mask(mask)
        # Keep columns/rows that carry real phone mass (not 1–2 px fringe).
        thr_c = max(6.0, float(col.max()) * 0.18)
        thr_r = max(6.0, float(row.max()) * 0.18)
        xs = np.where(col >= thr_c)[0]
        ys = np.where(row >= thr_r)[0]
        if xs.size < 8 or ys.size < 8:
            return AdaptiveMeshBuilder._aabb_quad_from_mask(mask)
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        if (x2 - x1) < 8.0 or (y2 - y1) < 8.0:
            return AdaptiveMeshBuilder._aabb_quad_from_mask(mask)
        # Never expand past the true silhouette extrema.
        raw = AdaptiveMeshBuilder._aabb_quad_from_mask(mask)
        if raw is not None:
            x1 = max(x1, float(raw[0, 0]))
            y1 = max(y1, float(raw[0, 1]))
            x2 = min(x2, float(raw[2, 0]))
            y2 = min(y2, float(raw[2, 1]))
        return order_points(
            np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )
        )

    @staticmethod
    def _stable_quad_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Stable outer quad of a cover/phone silhouette.

        Upright product shots use an axis-aligned AABB so the wrap cage is not
        tilted by minAreaRect noise or jagged template masks. True perspective
        / rotated shots keep the oriented box.
        """
        binary = (mask > 0).astype(np.uint8)
        # Light close so tiny notches don't skew minAreaRect.
        short = min(binary.shape[:2])
        k = max(3, (short // 80) | 1)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        outer = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(outer))
        if area < 64:
            return None
        aabb = AdaptiveMeshBuilder._aabb_quad_from_mask(binary)
        rect = cv2.minAreaRect(outer)
        deviation = AdaptiveMeshBuilder._rect_axis_deviation_deg(rect)
        # Prefer AABB when the silhouette fills its axis box (typical upright
        # phone on white) — even if minAreaRect angle is noisy.
        if aabb is not None:
            x1, y1 = float(aabb[0, 0]), float(aabb[0, 1])
            x2, y2 = float(aabb[2, 0]), float(aabb[2, 1])
            aabb_area = max((x2 - x1) * (y2 - y1), 1.0)
            fill = area / aabb_area
            if fill >= 0.78 or deviation <= 5.0:
                return aabb
        box = cv2.boxPoints(rect).astype(np.float32)
        return order_points(box)

    @staticmethod
    def _stable_quad_from_mesh(mesh: ControlMesh) -> np.ndarray:
        """Oriented rect from mesh boundary — ignores exploded corner spikes."""
        pts = mesh.boundary_points().astype(np.float32).reshape(-1, 1, 2)
        if pts.shape[0] < 4:
            return order_points(mesh.corner_points())
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.float32)
        # If minAreaRect collapses (degenerate), fall back to mesh corners.
        w, h = rect[1]
        if w < 8 or h < 8:
            return order_points(mesh.corner_points())
        if AdaptiveMeshBuilder._rect_axis_deviation_deg(rect) <= 3.5:
            ys = pts.reshape(-1, 2)[:, 1]
            xs = pts.reshape(-1, 2)[:, 0]
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            return order_points(
                np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                )
            )
        return order_points(box)

    @staticmethod
    def _snap_midsides_to_mask(
        mesh: ControlMesh,
        mask: np.ndarray,
        *,
        smooth: bool = True,
        max_move_fraction: float = 0.05,
    ) -> None:
        """
        Soft-snap mid-edge verts only. Corner verts stay put so arcs survive.
        """
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return
        edge_pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32
        )
        if edge_pts.shape[0] == 0:
            return
        if smooth and edge_pts.shape[0] >= 16:
            edge_pts = AdaptiveMeshBuilder._smooth_closed_polyline(
                edge_pts,
                window=max(9, min(31, edge_pts.shape[0] // 30 * 2 + 1)),
            )

        span = float(
            np.linalg.norm(mesh.points.max(axis=0) - mesh.points.min(axis=0))
        )
        snap_limit = max(6.0, span * float(max_move_fraction))
        corners = {
            mesh.index(0, 0),
            mesh.index(0, mesh.cols - 1),
            mesh.index(mesh.rows - 1, 0),
            mesh.index(mesh.rows - 1, mesh.cols - 1),
        }
        for index in mesh.boundary_indices():
            if int(index) in corners:
                continue
            point = mesh.points[index]
            delta = edge_pts - point
            nearest = int(np.argmin((delta * delta).sum(axis=1)))
            target = edge_pts[nearest]
            if float(np.linalg.norm(target - point)) <= snap_limit:
                # Blend — never fully jump onto residual stairs.
                mesh.points[index] = point * 0.35 + target * 0.65

    @staticmethod
    def force_rounded_perimeter(
        mesh: ControlMesh,
        corner_radius_percent: float = 8.0,
        corner_radii: Optional[Tuple[float, float, float, float]] = None,
        *,
        adaptive: bool = True,
        corner_bias: float = DEFAULT_CORNER_BIAS,
    ) -> ControlMesh:
        """
        Rebuild mesh verts as a perspective rounded rectangle.

        Long sides stay straight (bilinear UV). Corner verts sit on true
        quarter-circle arcs in cover UV — the product-smooth silhouette
        Perfect Finish needs.

        ``corner_radii`` optional (tl, tr, br, bl) percents override the
        single ``corner_radius_percent`` when Phase 1 device templates exist.

        Phase 5: ``adaptive`` packs grid UV toward corners for denser arcs.
        """
        if mesh is None or mesh.rows < 2 or mesh.cols < 2:
            return mesh
        result = mesh.copy()
        # Recover a sharp outer quad first. Using already-rounded corner verts
        # as the bilinear frame shrinks every Perfect Finish into chamfers.
        try:
            tl, tr, br, bl = [
                p.astype(np.float32).copy()
                for p in order_points(_sharp_quad_from_mesh(mesh))
            ]
        except Exception:
            grid0 = result.points.reshape(result.rows, result.cols, 2)
            tl = grid0[0, 0].astype(np.float32).copy()
            tr = grid0[0, -1].astype(np.float32).copy()
            br = grid0[-1, -1].astype(np.float32).copy()
            bl = grid0[-1, 0].astype(np.float32).copy()
        grid = result.points.reshape(result.rows, result.cols, 2)

        rows, cols = result.rows, result.cols
        if corner_radii is not None:
            rtl, rtr, rbr, rbl = [
                float(np.clip(v, 2.0, 28.0)) / 100.0 for v in corner_radii
            ]
        else:
            r = float(np.clip(corner_radius_percent, 2.0, 28.0)) / 100.0
            rtl = rtr = rbr = rbl = r
        # Keep arcs from colliding on short phone edges.
        rtl = min(rtl, 0.42)
        rtr = min(rtr, 0.42)
        rbr = min(rbr, 0.42)
        rbl = min(rbl, 0.42)
        radii_uv = (rtl, rtr, rbr, rbl)

        if adaptive and rows >= 3 and cols >= 3:
            u_s = adaptive_axis_samples(cols, corner_bias=corner_bias)
            v_s = adaptive_axis_samples(rows, corner_bias=corner_bias)
        else:
            u_s = np.linspace(0.0, 1.0, cols, dtype=np.float32)
            v_s = np.linspace(0.0, 1.0, rows, dtype=np.float32)

        def bilinear(u: float, v: float) -> np.ndarray:
            top = tl * (1.0 - u) + tr * u
            bot = bl * (1.0 - u) + br * u
            return top * (1.0 - v) + bot * v

        for row in range(rows):
            v = float(v_s[row])
            for col in range(cols):
                u = float(u_s[col])
                on_border = row in (0, rows - 1) or col in (0, cols - 1)
                if on_border:
                    uu, vv = AdaptiveMeshBuilder._rounded_rect_boundary_uv(
                        u, v, radii_uv, row, col, rows, cols
                    )
                else:
                    uu, vv = AdaptiveMeshBuilder._clamp_uv_to_rounded_rect(
                        u, v, radii_uv
                    )
                grid[row, col] = bilinear(uu, vv)

        result.points = grid.reshape(-1, 2).astype(np.float32)
        # Overwrite sparse boundary with dense true arcs so Perfect Finish
        # corners never read as flat chamfers between two UV samples.
        # Skip on upright AABB cages: adaptive arc-length packing mis-assigns
        # top-row verts onto the right edge and skews the wrap warp.
        try:
            sharp = order_points(
                np.array([tl, tr, br, bl], dtype=np.float32)
            )
            upright = AdaptiveMeshBuilder._quad_axis_deviation_deg(sharp) <= 3.5
            if not upright:
                outline = _sample_rounded_quad_perimeter(
                    sharp,
                    corner_radius_percent,
                    samples_per_edge=max(48, cols * 4),
                    corner_radii=corner_radii,
                )
                outline_bias = max(corner_bias, 0.55) if adaptive else 0.0
                AdaptiveMeshBuilder._assign_boundary_from_outline(
                    result, outline, corner_bias=outline_bias
                )
        except Exception:
            pass
        AdaptiveMeshBuilder._reinterpolate_interior(result)
        return result

    @staticmethod
    def _assign_boundary_from_outline(
        mesh: ControlMesh, outline: np.ndarray, *, corner_bias: float = 0.65
    ) -> None:
        """
        Map a dense closed outline onto the mesh boundary by arc length.

        Phase 2: ``corner_bias`` packs samples near the four corner regions
        so high-curvature arcs get more affine triangles (same rows×cols).
        """
        outline = np.asarray(outline, dtype=np.float32).reshape(-1, 2)
        if outline.shape[0] < 8 or mesh.rows < 2 or mesh.cols < 2:
            return
        indices = list(mesh.boundary_indices())
        if len(indices) < 4:
            return
        # Rotate outline so index 0 is nearest the mesh top-left corner —
        # otherwise arc-length mapping scrambles TL/TR/BR/BL roles.
        tl = mesh.points[mesh.index(0, 0)]
        start = int(np.argmin(((outline - tl) ** 2).sum(axis=1)))
        outline = np.vstack([outline[start:], outline[:start]])
        diffs = np.diff(np.vstack([outline, outline[:1]]), axis=0)
        seg = np.sqrt((diffs ** 2).sum(axis=1))
        total = float(seg.sum())
        if total < 1e-3:
            return
        cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        n = len(indices)
        bias = float(np.clip(corner_bias, 0.0, 0.85))
        # Build a density that peaks four times around the loop (corners).
        # Integrate → inverse-CDF sample for each boundary index.
        samples = max(256, n * 8)
        s = np.linspace(0.0, 1.0, samples, endpoint=False, dtype=np.float64)
        density = 1.0 + bias * (np.cos(2.0 * np.pi * 4.0 * s) ** 2)
        cdf = np.cumsum(density)
        cdf /= cdf[-1]
        for i, index in enumerate(indices):
            u = (i + 0.0) / float(n)
            # Inverse CDF: find s where cdf(s) ≈ u
            j = int(np.searchsorted(cdf, u, side="left"))
            j = int(np.clip(j, 0, samples - 1))
            target = float(s[j]) * total
            k = int(np.searchsorted(cum, target, side="right") - 1)
            k = int(np.clip(k, 0, len(outline) - 1))
            k2 = (k + 1) % len(outline)
            span = float(seg[k]) if seg[k] > 1e-8 else 1.0
            t = float(np.clip((target - cum[k]) / span, 0.0, 1.0))
            mesh.points[index] = (1.0 - t) * outline[k] + t * outline[k2]

        # Re-pin the four mesh corners to the outline extremes so handles
        # stay TL/TR/BR/BL after arc-length mapping.
        ordered = order_points(outline)
        mesh.points[mesh.index(0, 0)] = ordered[0]
        mesh.points[mesh.index(0, mesh.cols - 1)] = ordered[1]
        mesh.points[mesh.index(mesh.rows - 1, mesh.cols - 1)] = ordered[2]
        mesh.points[mesh.index(mesh.rows - 1, 0)] = ordered[3]

    @staticmethod
    def _rounded_rect_boundary_uv(
        u: float,
        v: float,
        r,
        row: int,
        col: int,
        rows: int,
        cols: int,
    ) -> Tuple[float, float]:
        """
        Map a boundary grid sample onto the rounded-rect perimeter in UV.

        Place on the unit-square edge first, then radially project any corner
        notch onto the quarter-circle (true product corners).

        ``r`` may be a single float or (tl, tr, br, bl) UV fractions.
        """
        top = row == 0
        bottom = row == rows - 1
        left = col == 0
        right = col == cols - 1

        uu = float(np.clip(u, 0.0, 1.0))
        vv = float(np.clip(v, 0.0, 1.0))
        if top and left:
            uu, vv = 0.0, 0.0
        elif top and right:
            uu, vv = 1.0, 0.0
        elif bottom and right:
            uu, vv = 1.0, 1.0
        elif bottom and left:
            uu, vv = 0.0, 1.0
        elif top:
            vv = 0.0
        elif bottom:
            vv = 1.0
        elif left:
            uu = 0.0
        elif right:
            uu = 1.0

        return AdaptiveMeshBuilder._clamp_uv_to_rounded_rect(uu, vv, r)

    @staticmethod
    def _normalize_corner_radii_uv(r) -> Tuple[float, float, float, float]:
        """Accept scalar or (tl,tr,br,bl) → UV fractions."""
        if isinstance(r, (tuple, list, np.ndarray)) and len(r) >= 4:
            vals = [float(max(1e-6, min(0.42, float(v)))) for v in r[:4]]
            return vals[0], vals[1], vals[2], vals[3]
        val = float(max(1e-6, min(0.42, float(r))))
        return val, val, val, val

    @staticmethod
    def _clamp_uv_to_rounded_rect(
        u: float, v: float, r
    ) -> Tuple[float, float]:
        """Pull UV points in the four corner notches onto circular arcs."""
        uu, vv = float(u), float(v)
        rtl, rtr, rbr, rbl = AdaptiveMeshBuilder._normalize_corner_radii_uv(r)
        # TL
        if uu < rtl and vv < rtl:
            dx, dy = uu - rtl, vv - rtl
            dist = float(np.hypot(dx, dy))
            if dist > 1e-8 and dist > rtl:
                s = rtl / dist
                return rtl + dx * s, rtl + dy * s
            return uu, vv
        # TR
        if uu > 1.0 - rtr and vv < rtr:
            dx, dy = uu - (1.0 - rtr), vv - rtr
            dist = float(np.hypot(dx, dy))
            if dist > 1e-8 and dist > rtr:
                s = rtr / dist
                return (1.0 - rtr) + dx * s, rtr + dy * s
            return uu, vv
        # BR
        if uu > 1.0 - rbr and vv > 1.0 - rbr:
            dx, dy = uu - (1.0 - rbr), vv - (1.0 - rbr)
            dist = float(np.hypot(dx, dy))
            if dist > 1e-8 and dist > rbr:
                s = rbr / dist
                return (1.0 - rbr) + dx * s, (1.0 - rbr) + dy * s
            return uu, vv
        # BL
        if uu < rbl and vv > 1.0 - rbl:
            dx, dy = uu - rbl, vv - (1.0 - rbl)
            dist = float(np.hypot(dx, dy))
            if dist > 1e-8 and dist > rbl:
                s = rbl / dist
                return rbl + dx * s, (1.0 - rbl) + dy * s
            return uu, vv
        return uu, vv

    @staticmethod
    def refine(
        mesh: ControlMesh,
        printable_mask: Optional[np.ndarray] = None,
        corner_radius_percent: float = 6.0,
        *,
        light: bool = False,
    ) -> ControlMesh:
        """
        Local edge / corner refinement with stable Coons re-interpolation.

        Snaps to a *smoothed* outer silhouette so mid-side verts don't lock onto
        pixel stair-steps (the main cause of jagged wrap edges).
        """
        refined = mesh.copy()
        if not light:
            AdaptiveMeshBuilder._smooth_edges(refined, passes=2)
        if printable_mask is not None and np.count_nonzero(printable_mask) > 0:
            AdaptiveMeshBuilder._snap_boundary_to_mask(
                refined, printable_mask, smooth=True
            )
            if not light:
                AdaptiveMeshBuilder._refine_corners(
                    refined, printable_mask, corner_radius_percent
                )
                # Kill mid-side zig-zags (phone cases have nearly straight sides).
                AdaptiveMeshBuilder._straighten_sides(refined, passes=2)
                # Smooth mid-sides after corner polish — do not re-snap to raw
                # stairs (that undoes rounding and recreates zig-zags).
                AdaptiveMeshBuilder._smooth_edges(refined, passes=3)
                AdaptiveMeshBuilder._snap_boundary_to_mask(
                    refined, printable_mask, smooth=True
                )
        if not light:
            AdaptiveMeshBuilder._reinterpolate_interior(refined)
            AdaptiveMeshBuilder._smooth_edges(refined, passes=2)
        if printable_mask is not None and np.count_nonzero(printable_mask) > 0:
            AdaptiveMeshBuilder.clamp_to_mask(refined, printable_mask)
            AdaptiveMeshBuilder._straighten_sides(refined, passes=1)
            AdaptiveMeshBuilder._smooth_edges(refined, passes=1)
            AdaptiveMeshBuilder._reinterpolate_interior(refined)
        return refined

    @staticmethod
    def clamp_to_mask(mesh: ControlMesh, mask: np.ndarray) -> None:
        """Pull any outside boundary vertex onto the nearest *smoothed* mask edge."""
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return
        edge_pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32
        )
        if edge_pts.shape[0] == 0:
            return
        if edge_pts.shape[0] >= 16:
            edge_pts = AdaptiveMeshBuilder._smooth_closed_polyline(
                edge_pts, window=max(7, min(21, edge_pts.shape[0] // 40 * 2 + 1))
            )
        height, width = mask.shape[:2]
        for index in mesh.boundary_indices():
            point = mesh.points[index]
            x = int(np.clip(round(point[0]), 0, width - 1))
            y = int(np.clip(round(point[1]), 0, height - 1))
            if mask[y, x] > 0:
                continue
            delta = edge_pts - point
            nearest = int(np.argmin((delta * delta).sum(axis=1)))
            mesh.points[index] = edge_pts[nearest]

    @staticmethod
    def _smooth_edges(mesh: ControlMesh, passes: int = 2) -> None:
        """Moving-average each boundary edge for local refinement."""
        grid = mesh.points.reshape(mesh.rows, mesh.cols, 2)
        for _ in range(max(1, passes)):
            # Top / bottom: smooth along columns, keep corners anchored.
            for row in (0, mesh.rows - 1):
                edge = grid[row].copy()
                for col in range(1, mesh.cols - 1):
                    edge[col] = (
                        grid[row, col - 1]
                        + grid[row, col]
                        + grid[row, col + 1]
                    ) / 3.0
                edge[0] = grid[row, 0]
                edge[-1] = grid[row, -1]
                grid[row] = edge
            # Left / right: smooth along rows.
            for col in (0, mesh.cols - 1):
                edge = grid[:, col].copy()
                for row in range(1, mesh.rows - 1):
                    edge[row] = (
                        grid[row - 1, col]
                        + grid[row, col]
                        + grid[row + 1, col]
                    ) / 3.0
                edge[0] = grid[0, col]
                edge[-1] = grid[-1, col]
                grid[:, col] = edge
        mesh.points = grid.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _straighten_sides(mesh: ControlMesh, passes: int = 2) -> None:
        """
        Pull mid-side verts toward the chord between near-corner samples.

        Manufactured covers have nearly straight long edges; residual zig-zags
        from silhouette noise make wrap look fake. Corner verts stay put so
        rounded corners survive.
        """
        if mesh.rows < 5 and mesh.cols < 5:
            return
        grid = mesh.points.reshape(mesh.rows, mesh.cols, 2)
        blend = 0.62
        for _ in range(max(1, passes)):
            if mesh.rows >= 5:
                for col in (0, mesh.cols - 1):
                    p0 = grid[1, col]
                    p1 = grid[mesh.rows - 2, col]
                    for row in range(2, mesh.rows - 2):
                        t = (row - 1) / max(mesh.rows - 3, 1)
                        target = p0 * (1.0 - t) + p1 * t
                        grid[row, col] = (
                            grid[row, col] * (1.0 - blend) + target * blend
                        )
            if mesh.cols >= 5:
                for row in (0, mesh.rows - 1):
                    p0 = grid[row, 1]
                    p1 = grid[row, mesh.cols - 2]
                    for col in range(2, mesh.cols - 2):
                        t = (col - 1) / max(mesh.cols - 3, 1)
                        target = p0 * (1.0 - t) + p1 * t
                        grid[row, col] = (
                            grid[row, col] * (1.0 - blend) + target * blend
                        )
        mesh.points = grid.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _pull_boundary_inside(
        mesh: ControlMesh, mask: np.ndarray
    ) -> None:
        """Nudge any boundary vert that landed just outside the cover mask."""
        if mesh is None or mask is None or np.count_nonzero(mask) == 0:
            return
        binary = (mask > 0).astype(np.uint8)
        height, width = binary.shape[:2]
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return
        cx = float(xs.mean())
        cy = float(ys.mean())
        for index in mesh.boundary_indices():
            x, y = float(mesh.points[index, 0]), float(mesh.points[index, 1])
            xi = int(np.clip(round(x), 0, width - 1))
            yi = int(np.clip(round(y), 0, height - 1))
            if binary[yi, xi]:
                continue
            placed = False
            for t in np.linspace(0.02, 0.40, 16):
                nx = x * (1.0 - t) + cx * t
                ny = y * (1.0 - t) + cy * t
                xi = int(np.clip(round(nx), 0, width - 1))
                yi = int(np.clip(round(ny), 0, height - 1))
                if binary[yi, xi]:
                    mesh.points[index, 0] = nx
                    mesh.points[index, 1] = ny
                    placed = True
                    break
            if not placed:
                # Last resort: nearest cover pixel.
                dist = (xs.astype(np.float32) - x) ** 2 + (
                    ys.astype(np.float32) - y
                ) ** 2
                j = int(np.argmin(dist))
                mesh.points[index, 0] = float(xs[j])
                mesh.points[index, 1] = float(ys[j])

    @staticmethod
    def _snap_boundary_to_mask(
        mesh: ControlMesh,
        mask: np.ndarray,
        *,
        smooth: bool = False,
    ) -> None:
        """Pull boundary vertices onto the nearest outer printable edge."""
        binary = (mask > 0).astype(np.uint8)
        # Outer perimeter only — never snap onto camera/flash holes.
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return
        outer = max(contours, key=cv2.contourArea)
        edge_pts = outer.reshape(-1, 2).astype(np.float32)
        if edge_pts.shape[0] == 0:
            return
        if smooth and edge_pts.shape[0] >= 16:
            edge_pts = AdaptiveMeshBuilder._smooth_closed_polyline(
                edge_pts, window=max(7, min(21, edge_pts.shape[0] // 40 * 2 + 1))
            )

        # Scale snap radius with image size — fixed 12px fails on large photos.
        short = float(min(mask.shape[0], mask.shape[1]))
        snap_limit = max(12.0, short * 0.022)

        for index in mesh.boundary_indices():
            point = mesh.points[index]
            delta = edge_pts - point
            nearest = int(np.argmin((delta * delta).sum(axis=1)))
            if float(np.linalg.norm(edge_pts[nearest] - point)) <= snap_limit:
                mesh.points[index] = edge_pts[nearest]

    @staticmethod
    def _smooth_closed_polyline(
        points: np.ndarray, window: int = 9
    ) -> np.ndarray:
        """Circular moving-average of a closed contour (kills pixel stairs)."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        n = pts.shape[0]
        if n < 8:
            return pts
        window = int(max(3, window))
        if window % 2 == 0:
            window += 1
        window = min(window, n if n % 2 == 1 else n - 1)
        half = window // 2
        # Pad cyclically then convolve.
        padded = np.concatenate([pts[-half:], pts, pts[:half]], axis=0)
        kernel = np.ones((window, 1), dtype=np.float32) / float(window)
        sm_x = np.convolve(padded[:, 0], kernel.ravel(), mode="valid")
        sm_y = np.convolve(padded[:, 1], kernel.ravel(), mode="valid")
        return np.stack([sm_x, sm_y], axis=1).astype(np.float32)

    @staticmethod
    def _refine_corners(
        mesh: ControlMesh,
        mask: np.ndarray,
        corner_radius_percent: float,
    ) -> None:
        """
        Nudge near-corner edge samples along the rounded printable silhouette.

        Corners stay at diagonal extrema; the adjacent edge points track the
        arc so piecewise-affine warps follow rounded covers smoothly.
        """
        if corner_radius_percent <= 0:
            return
        binary = mask > 0
        if not np.count_nonzero(binary):
            return
        ys, xs = np.nonzero(binary)
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        short = max(min(x_max - x_min, y_max - y_min), 1.0)
        radius = short * float(np.clip(corner_radius_percent, 0.0, 30.0)) / 100.0
        if radius < 2.0:
            return

        grid = mesh.points.reshape(mesh.rows, mesh.cols, 2)
        # Outer contour for projecting corners onto the real rounded silhouette.
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        edge_pts = None
        if contours:
            edge_pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
                np.float32
            )

        center = np.array(
            [(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], np.float32
        )
        corners = {
            (0, 0): (x_min, y_min),
            (0, mesh.cols - 1): (x_max, y_min),
            (mesh.rows - 1, 0): (x_min, y_max),
            (mesh.rows - 1, mesh.cols - 1): (x_max, y_max),
        }
        for (r, c), (cx, cy) in corners.items():
            # Prefer silhouette points in the same quadrant as this corner so
            # opposite arcs cannot steal a vertex (which inverts mesh cells).
            if edge_pts is not None and edge_pts.shape[0] > 0:
                target_corner = np.array([cx, cy], np.float32)
                sx = -1.0 if cx <= center[0] else 1.0
                sy = -1.0 if cy <= center[1] else 1.0
                rel = edge_pts - center
                quadrant = (rel[:, 0] * sx >= -1.0) & (rel[:, 1] * sy >= -1.0)
                candidates = edge_pts[quadrant] if np.count_nonzero(quadrant) >= 4 else edge_pts
                delta = candidates - target_corner
                nearest = int(np.argmin((delta * delta).sum(axis=1)))
                # Blend — never jump a corner all the way across the silhouette.
                grid[r, c] = grid[r, c] * 0.25 + candidates[nearest] * 0.75
                cx, cy = float(grid[r, c][0]), float(grid[r, c][1])

            neighbors = []
            if c + 1 < mesh.cols and r in (0, mesh.rows - 1):
                neighbors.append((r, c + 1 if c == 0 else c - 1))
            if r + 1 < mesh.rows and c in (0, mesh.cols - 1):
                neighbors.append((r + 1 if r == 0 else r - 1, c))
            for nr, nc in neighbors:
                point = grid[nr, nc]
                away = point - np.array([cx, cy], np.float32)
                length = float(np.linalg.norm(away))
                if length < 1e-3:
                    continue
                target = np.array([cx, cy], np.float32) + away * (
                    min(radius, length) / length
                )
                if edge_pts is not None and edge_pts.shape[0] > 0:
                    delta = edge_pts - target
                    nearest = int(np.argmin((delta * delta).sum(axis=1)))
                    if float(np.linalg.norm(edge_pts[nearest] - target)) <= radius * 1.5:
                        target = edge_pts[nearest]
                tx = int(np.clip(round(target[0]), 0, mask.shape[1] - 1))
                ty = int(np.clip(round(target[1]), 0, mask.shape[0] - 1))
                if mask[ty, tx] == 0:
                    continue
                grid[nr, nc] = grid[nr, nc] * 0.35 + target * 0.65
        mesh.points = grid.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _reinterpolate_interior(mesh: ControlMesh) -> None:
        """Rebuild interior vertices from the refined boundary (Coons patch)."""
        grid = mesh.points.reshape(mesh.rows, mesh.cols, 2)
        top = grid[0].copy()
        bottom = grid[-1].copy()
        left = grid[:, 0].copy()
        right = grid[:, -1].copy()
        tl, tr = top[0], top[-1]
        bl, br = bottom[0], bottom[-1]

        for row in range(1, mesh.rows - 1):
            v = row / max(mesh.rows - 1, 1)
            for col in range(1, mesh.cols - 1):
                u = col / max(mesh.cols - 1, 1)
                edge_blend = (
                    (1.0 - v) * top[col]
                    + v * bottom[col]
                    + (1.0 - u) * left[row]
                    + u * right[row]
                )
                bilinear = (
                    (1.0 - u) * (1.0 - v) * tl
                    + u * (1.0 - v) * tr
                    + u * v * br
                    + (1.0 - u) * v * bl
                )
                grid[row, col] = edge_blend - bilinear

        # Coons extrapolates wildly when mid-side boundary verts sit outside
        # the corner quad (common after silhouette conform on rounded phones).
        # Fall back to bilinear so wrap triangles stay on the cover face.
        boundary = np.vstack([top, bottom, left, right])
        bmin = boundary.min(axis=0) - 4.0
        bmax = boundary.max(axis=0) + 4.0
        interior = grid[1:-1, 1:-1].reshape(-1, 2)
        if interior.size and (
            float(interior[:, 0].min()) < float(bmin[0])
            or float(interior[:, 0].max()) > float(bmax[0])
            or float(interior[:, 1].min()) < float(bmin[1])
            or float(interior[:, 1].max()) > float(bmax[1])
        ):
            for row in range(1, mesh.rows - 1):
                v = row / max(mesh.rows - 1, 1)
                for col in range(1, mesh.cols - 1):
                    u = col / max(mesh.cols - 1, 1)
                    grid[row, col] = (
                        (1.0 - u) * (1.0 - v) * tl
                        + u * (1.0 - v) * tr
                        + u * v * br
                        + (1.0 - u) * v * bl
                    )

        mesh.points = grid.reshape(-1, 2).astype(np.float32)


class MeshWarper:
    """Piecewise-affine BGRA warping over a fixed-topology control mesh."""

    EDGE_TOLERANCE = 2

    @staticmethod
    def source_points(
        design_shape: Tuple[int, int],
        rows: int,
        cols: int,
        target_aspect: float,
        fit_mode: str = "fill",
        scale: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        rotation: float = 0.0,
        curved_uv: Optional[object] = None,
    ) -> np.ndarray:
        """
        Regular source grid after fit, crop, pan, scale, and rotation.

        Optional ``curved_uv`` (CurvedUVParams) remaps parametric UV so the
        bevel rim foreshortens like a moulded cover instead of a flat sticker.
        """
        design_h, design_w = design_shape[:2]
        design_aspect = design_w / max(float(design_h), 1e-6)
        target_aspect = max(float(target_aspect), 1e-6)

        if fit_mode == "stretch":
            crop_w, crop_h = float(design_w), float(design_h)
        elif fit_mode == "fit":
            if design_aspect > target_aspect:
                crop_w = float(design_w)
                crop_h = crop_w / target_aspect
            else:
                crop_h = float(design_h)
                crop_w = crop_h * target_aspect
        else:
            if design_aspect > target_aspect:
                crop_h = float(design_h)
                crop_w = crop_h * target_aspect
            else:
                crop_w = float(design_w)
                crop_h = crop_w / target_aspect

        scale = max(float(scale), 0.05)
        crop_w /= scale
        crop_h /= scale

        # Phase 2 + 5: optional curved UV table (rows*cols, 2) using the same
        # adaptive axis samples as the destination mesh so foreshortening
        # aligns with corner-dense triangles.
        uv_table = None
        if curved_uv is not None:
            try:
                from .curved_uv import CurvedUVParams, remap_grid
                params = curved_uv
                if not isinstance(params, CurvedUVParams):
                    params = CurvedUVParams(
                        rim_uv=float(getattr(params, "rim_uv", 0.055)),
                        bevel_strength=float(
                            getattr(params, "bevel_strength", 0.92)
                        ),
                        corner_radii=getattr(params, "corner_radii", None),
                        enabled=bool(getattr(params, "enabled", True)),
                    )
                if params.enabled:
                    uv_table = remap_grid(
                        rows, cols, params, adaptive=True
                    )
            except Exception:
                uv_table = None

        # CSS object-fit: cover — when curved UV pushes samples past the unit
        # square, shrink the crop (zoom) so every dest vert still reads artwork.
        # Extent comes from the live remap table, not a fixed bleed constant.
        if (
            uv_table is not None
            and fit_mode in ("fill", "fit")
            and fit_mode != "stretch"
        ):
            u_ext = float(
                max(np.max(np.abs(uv_table[:, 0] - 0.5)), 0.5)
            )
            v_ext = float(
                max(np.max(np.abs(uv_table[:, 1] - 0.5)), 0.5)
            )
            cover = max(u_ext, v_ext) / 0.5
            if cover > 1.0:
                crop_w /= cover
                crop_h /= cover

        center = np.array(
            [
                design_w / 2.0 + offset_x * crop_w * 0.5,
                design_h / 2.0 + offset_y * crop_h * 0.5,
            ],
            dtype=np.float32,
        )
        # Soft clamp: keep most of the crop on artwork, but allow enough travel
        # so Move Design / offset sliders actually move the print (hard clamp
        # used to lock the window when cover-fit made crop ≈ design).
        center = MeshWarper._clamp_window(
            center, crop_w, crop_h, design_w, design_h, rotation,
            soft=True,
        )

        if uv_table is None:
            u_s = adaptive_axis_samples(cols)
            v_s = adaptive_axis_samples(rows)
            points = []
            for row in range(rows):
                v = float(v_s[row])
                for col in range(cols):
                    u = float(u_s[col])
                    points.append(
                        [
                            center[0] + (u - 0.5) * crop_w,
                            center[1] + (v - 0.5) * crop_h,
                        ]
                    )
            points = np.asarray(points, dtype=np.float32)
        else:
            points = np.zeros((rows * cols, 2), dtype=np.float32)
            for idx in range(rows * cols):
                u = float(uv_table[idx, 0])
                v = float(uv_table[idx, 1])
                points[idx, 0] = center[0] + (u - 0.5) * crop_w
                points[idx, 1] = center[1] + (v - 0.5) * crop_h

        if abs(rotation) > 1e-6:
            theta = np.deg2rad(-float(rotation))
            matrix = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ],
                dtype=np.float32,
            )
            points = (points - center) @ matrix.T + center

        return points.astype(np.float32)

    @staticmethod
    def _clamp_window(
        center: np.ndarray,
        crop_w: float,
        crop_h: float,
        design_w: int,
        design_h: int,
        rotation: float,
        *,
        soft: bool = False,
    ) -> np.ndarray:
        """
        Keep the sampled window mostly inside the artwork.

        Hard mode (``soft=False``): window stays fully on artwork (auto-fit).
        Soft mode: up to ~40% of the crop may hang off so the user can pan
        which part of the design prints (Move Design / offset sliders).
        """
        theta = np.deg2rad(float(rotation))
        cos, sin = abs(float(np.cos(theta))), abs(float(np.sin(theta)))
        extents = (
            (cos * crop_w + sin * crop_h) / 2.0,
            (sin * crop_w + cos * crop_h) / 2.0,
        )
        clamped = np.asarray(center, dtype=np.float32).copy()
        for axis, (half, size) in enumerate(
            zip(extents, (float(design_w), float(design_h)))
        ):
            if half * 2.0 >= size:
                if soft:
                    # Still allow sliding along an oversized axis.
                    pad = max(1.0, size * 0.08)
                    clamped[axis] = float(np.clip(clamped[axis], pad, size - pad))
                else:
                    clamped[axis] = size / 2.0
            else:
                if soft:
                    # Allow the crop to hang off so pan is not dead.
                    travel = half * 0.95
                    lo = half - travel
                    hi = size - (half - travel)
                else:
                    lo = half
                    hi = size - half
                clamped[axis] = float(np.clip(clamped[axis], lo, hi))
        return clamped

    @staticmethod
    def warp(
        design_image: np.ndarray,
        source_points: np.ndarray,
        destination_mesh: ControlMesh,
        output_shape: Tuple[int, int],
        mirror: bool = False,
    ) -> Optional[np.ndarray]:
        """Warp each source triangle independently into its destination."""
        if design_image is None or design_image.ndim < 2:
            return None

        design = to_bgra(design_image)
        if mirror:
            design = cv2.flip(design, 1)

        output_h, output_w = map(int, output_shape)
        destination = np.zeros((output_h, output_w, 4), dtype=np.uint8)
        source_points = np.asarray(source_points, dtype=np.float32)

        for triangle in destination_mesh.triangles():
            MeshWarper._warp_triangle(
                design,
                destination,
                source_points[triangle],
                destination_mesh.points[triangle],
            )
        return destination

    @staticmethod
    def _warp_triangle(
        source: np.ndarray,
        destination: np.ndarray,
        source_triangle: np.ndarray,
        destination_triangle: np.ndarray,
    ) -> None:
        """Affine-warp one triangle and alpha-compose it into the output."""
        source_rect = cv2.boundingRect(source_triangle.astype(np.float32))
        destination_rect = cv2.boundingRect(
            destination_triangle.astype(np.float32)
        )
        sx, sy, sw, sh = source_rect
        dx, dy, dw, dh = destination_rect

        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            return

        # Clamp destination ROI to the output while preserving triangle offsets.
        out_h, out_w = destination.shape[:2]
        x0, y0 = max(dx, 0), max(dy, 0)
        x1, y1 = min(dx + dw, out_w), min(dy + dh, out_h)
        if x1 <= x0 or y1 <= y0:
            return

        source_local = source_triangle - np.array([sx, sy], dtype=np.float32)
        destination_local = destination_triangle - np.array(
            [dx, dy], dtype=np.float32
        )

        # BORDER_TRANSPARENT is not reliable for affine BGRA on all OpenCV
        # builds, so warp an explicitly padded source crop and mask afterwards.
        source_crop = MeshWarper._source_patch(source, sx, sy, sw, sh)
        if source_crop is None:
            return

        transform = cv2.getAffineTransform(source_local, destination_local)
        # Bilinear sampling on the triangle's own outline reaches just past the
        # crop. Replicating keeps alpha at full strength there; a constant zero
        # border would instead darken a hairline along every triangle edge.
        warped = cv2.warpAffine(
            source_crop,
            transform,
            (dw, dh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Soft triangle coverage from LINE_AA — do NOT promote every partial
        # pixel to 255. That binary threshold was the main stair-step on
        # rounded outer corners (internal seams still use max-alpha compose).
        triangle_mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillConvexPoly(
            triangle_mask,
            np.round(destination_local).astype(np.int32),
            255,
            lineType=cv2.LINE_AA,
        )
        warped[:, :, 3] = np.minimum(warped[:, :, 3], triangle_mask)

        crop_x0, crop_y0 = x0 - dx, y0 - dy
        crop_x1, crop_y1 = crop_x0 + (x1 - x0), crop_y0 + (y1 - y0)
        patch = warped[crop_y0:crop_y1, crop_x0:crop_x1]
        target = destination[y0:y1, x0:x1]

        # Keep straight (not premultiplied) colour because the compositor
        # applies alpha later. Prefer the triangle with greater coverage on
        # antialiased shared edges to prevent dark seams.
        take = patch[:, :, 3] > target[:, :, 3]
        target[:, :, :3][take] = patch[:, :, :3][take]
        target[:, :, 3] = np.maximum(target[:, :, 3], patch[:, :, 3])

    @staticmethod
    def _source_patch(
        source: np.ndarray, x: int, y: int, w: int, h: int
    ) -> Optional[np.ndarray]:
        """
        Read a source rectangle, extending edge pixels where it leaves the image.

        Sub-pixel triangle bounds routinely reach a pixel or two outside the
        artwork, which would punch transparent notches into an otherwise solid
        print. Replication is capped at that tolerance: a window reaching much
        further is genuinely outside the artwork, as in fit mode or a deliberate
        zoom-out, and the cover has to stay visible there.
        """
        height, width = source.shape[:2]
        x0 = int(np.clip(x, 0, width - 1))
        y0 = int(np.clip(y, 0, height - 1))
        x1 = int(np.clip(x + w, x0 + 1, width))
        y1 = int(np.clip(y + h, y0 + 1, height))

        patch = source[y0:y1, x0:x1]
        pads = [
            max(0, min(h, y0 - y)),
            0,
            max(0, min(w, x0 - x)),
            0,
        ]
        pads[1] = max(0, h - pads[0] - (y1 - y0))
        pads[3] = max(0, w - pads[2] - (x1 - x0))
        if not any(pads):
            return patch

        replicated = [min(pad, MeshWarper.EDGE_TOLERANCE) for pad in pads]
        if any(replicated):
            patch = cv2.copyMakeBorder(
                patch, *replicated, borderType=cv2.BORDER_REPLICATE
            )
        blank = [pad - done for pad, done in zip(pads, replicated)]
        if any(blank):
            patch = cv2.copyMakeBorder(
                patch, *blank, borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0, 0),
            )
        return patch


def mesh_aspect(mesh: ControlMesh) -> float:
    """Average boundary width divided by average boundary height."""
    points = mesh.points.reshape(mesh.rows, mesh.cols, 2)
    top = np.linalg.norm(np.diff(points[0], axis=0), axis=1).sum()
    bottom = np.linalg.norm(np.diff(points[-1], axis=0), axis=1).sum()
    left = np.linalg.norm(np.diff(points[:, 0], axis=0), axis=1).sum()
    right = np.linalg.norm(np.diff(points[:, -1], axis=0), axis=1).sum()
    return float((top + bottom) / max(left + right, 1e-6))


def _chaikin_closed(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner-cutting — densifies and rounds a closed polyline."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return pts
    for _ in range(max(0, iterations)):
        n = pts.shape[0]
        out = np.empty((n * 2, 2), dtype=np.float32)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            out[i * 2] = 0.75 * p0 + 0.25 * p1
            out[i * 2 + 1] = 0.25 * p0 + 0.75 * p1
        pts = out
    return pts


def _sample_rounded_quad_perimeter(
    quad: np.ndarray,
    corner_radius_percent: float,
    samples_per_edge: int = 48,
    corner_radii: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """
    Dense perspective rounded-rect outline from a SHARP TL/TR/BR/BL quad.

    Angle-parameterized quarter-circles in UV (then bilinear to pixels) so
    corners stay true curves — never flat 45° chamfer chords.

    Optional ``corner_radii`` are percents (tl, tr, br, bl).
    """
    tl, tr, br, bl = order_points(np.asarray(quad, dtype=np.float32))
    if corner_radii is not None:
        rtl, rtr, rbr, rbl = [
            min(0.42, float(np.clip(v, 2.0, 28.0)) / 100.0)
            for v in corner_radii
        ]
    else:
        r = float(np.clip(corner_radius_percent, 2.0, 28.0)) / 100.0
        r = min(r, 0.42)
        rtl = rtr = rbr = rbl = r
    n_side = max(12, int(samples_per_edge))
    n_arc = max(48, int(samples_per_edge * 1.15))

    def bilinear(u: float, v: float) -> np.ndarray:
        top = tl * (1.0 - u) + tr * u
        bot = bl * (1.0 - u) + br * u
        return top * (1.0 - v) + bot * v

    pts: list = []

    def add_arc(cx: float, cy: float, radius: float, a0: float, a1: float) -> None:
        for i in range(n_arc + 1):
            t = i / float(n_arc)
            ang = a0 + (a1 - a0) * t
            pts.append(
                bilinear(
                    cx + radius * float(np.cos(ang)),
                    cy + radius * float(np.sin(ang)),
                )
            )

    for i in range(n_side):
        u = rtl + (1.0 - rtl - rtr) * (i / max(n_side - 1, 1))
        if i == n_side - 1:
            break
        u = float(np.clip(u, rtl, 1.0 - rtr))
        pts.append(bilinear(u, 0.0))
    add_arc(1.0 - rtr, rtr, rtr, -0.5 * np.pi, 0.0)
    for i in range(1, n_side):
        v = rtr + (1.0 - rtr - rbr) * (i / max(n_side - 1, 1))
        if i == n_side - 1:
            break
        v = float(np.clip(v, rtr, 1.0 - rbr))
        pts.append(bilinear(1.0, v))
    add_arc(1.0 - rbr, 1.0 - rbr, rbr, 0.0, 0.5 * np.pi)
    for i in range(1, n_side):
        u = (1.0 - rbr) - (1.0 - rbr - rbl) * (i / max(n_side - 1, 1))
        if i == n_side - 1:
            break
        u = float(np.clip(u, rbl, 1.0 - rbr))
        pts.append(bilinear(u, 1.0))
    add_arc(rbl, 1.0 - rbl, rbl, 0.5 * np.pi, np.pi)
    for i in range(1, n_side):
        v = (1.0 - rbl) - (1.0 - rbl - rtl) * (i / max(n_side - 1, 1))
        if i == n_side - 1:
            break
        v = float(np.clip(v, rtl, 1.0 - rbl))
        pts.append(bilinear(0.0, v))
    add_arc(rtl, rtl, rtl, np.pi, 1.5 * np.pi)

    arr = np.asarray(pts, dtype=np.float32)
    # Drop near-duplicate joints.
    cleaned = [arr[0]]
    for p in arr[1:]:
        if float(np.linalg.norm(p - cleaned[-1])) > 0.25:
            cleaned.append(p)
    if (
        len(cleaned) >= 2
        and float(np.linalg.norm(cleaned[0] - cleaned[-1])) < 0.35
    ):
        cleaned.pop()
    return np.asarray(cleaned, dtype=np.float32)


def _corner_proximity_map(
    shape: Tuple[int, int],
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    corner_frac: float = 0.22,
) -> np.ndarray:
    """
    1 near the four rounded corners, 0 along mid-edge straight sides.

    Used so AA / blur softens curves without making L/R vertical borders cloudy.
    """
    height, width = map(int, shape)
    bw = max(float(x1 - x0), 1.0)
    bh = max(float(y1 - y0), 1.0)
    band = float(np.clip(corner_frac, 0.08, 0.40))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = np.clip((xx - float(x0)) / bw, 0.0, 1.0)
    v = np.clip((yy - float(y0)) / bh, 0.0, 1.0)
    near_lr = np.minimum(u, 1.0 - u)
    near_tb = np.minimum(v, 1.0 - v)
    # High only when BOTH axes are near an edge (true corner pockets).
    along_lr = np.clip(1.0 - near_lr / band, 0.0, 1.0)
    along_tb = np.clip(1.0 - near_tb / band, 0.0, 1.0)
    return (along_lr * along_tb).astype(np.float32)


def _sharp_quad_from_mesh(mesh: ControlMesh) -> np.ndarray:
    """
    Recover a sharp TL/TR/BR/BL quad from a (possibly already-rounded) mesh.

    Using already-inset corner_points() as the quad re-applies rounding on a
    smaller rectangle and creates flat chamfer / octagon corners.
    """
    stable = AdaptiveMeshBuilder._stable_quad_from_mesh(mesh)
    if stable is not None and len(stable) == 4:
        return order_points(np.asarray(stable, dtype=np.float32))
    return order_points(mesh.corner_points().astype(np.float32))


def _fill_closed_polyline_aa(
    points: np.ndarray,
    shape: Tuple[int, int],
    *,
    scale: int = 4,
    expand_px: float = 0.85,
) -> np.ndarray:
    """
    Supersampled closed fill → float coverage with product-smooth AA.

    expand_px grows the silhouette slightly so Chaikin/rounding cannot leave a
    chalk white rim of bare phone / studio plate on the outer edge.
    """
    height, width = map(int, shape)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return np.zeros((height, width), dtype=np.float32)

    s = max(2, min(int(scale), 10))
    x0 = float(pts[:, 0].min())
    y0 = float(pts[:, 1].min())
    x1 = float(pts[:, 0].max())
    y1 = float(pts[:, 1].max())
    pad = max(3, int(np.ceil(expand_px + 2.0)))
    ix0 = max(0, int(np.floor(x0)) - pad)
    iy0 = max(0, int(np.floor(y0)) - pad)
    ix1 = min(width, int(np.ceil(x1)) + pad + 1)
    iy1 = min(height, int(np.ceil(y1)) + pad + 1)
    if ix1 <= ix0 or iy1 <= iy0:
        return np.zeros((height, width), dtype=np.float32)

    rw = ix1 - ix0
    rh = iy1 - iy0
    big = np.zeros((rh * s, rw * s), dtype=np.uint8)
    local = (pts - np.array([ix0, iy0], dtype=np.float32)) * float(s)
    local_i = np.round(local).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(big, [local_i], 255)
    grow = max(0, int(round(float(expand_px) * s)))
    if grow > 0:
        big = cv2.dilate(
            big,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)
            ),
            iterations=1,
        )
    small = cv2.resize(big, (rw, rh), interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width), dtype=np.float32)
    out[iy0:iy1, ix0:ix1] = small.astype(np.float32) / 255.0
    return out


def create_mesh_mask(
    mesh: ControlMesh,
    shape: Tuple[int, int],
    feather_radius: int = 0,
    corner_radius_percent: float = 0.0,
    *,
    smooth_boundary: bool = True,
    phone_silhouette: Optional[np.ndarray] = None,
    corner_radii: Optional[Tuple[float, float, float, float]] = None,
    prefer_live_boundary: bool = False,
) -> np.ndarray:
    """
    Rasterise the editable perimeter with product-smooth edges.

    Prefers a dense perspective rounded-rect outline (true manufactured
    corners) when a corner radius is set; otherwise Chaikin-smooths the live
    mesh boundary. Always supersamples so outer edges/corners match cutout AA.

    ``prefer_live_boundary`` fills from the editable mesh perimeter (same as
    the blue on-screen cage) so wrap matches the selected area on any phone.
    """
    height, width = map(int, shape)
    outline: Optional[np.ndarray] = None

    # Live mesh perimeter = blue selection cage (dynamic for every phone).
    if prefer_live_boundary:
        raw = mesh.boundary_points().astype(np.float32)
        if raw.shape[0] >= 8:
            outline = _chaikin_closed(raw, iterations=1)

    # Prefer true manufactured quarter-circles on all four corners. Chaikin of
    # the live mesh boundary inherits bottom silhouette stairs (contact shadow)
    # while the top stays clean — that made BL/BR look unfinished vs TL/TR.
    if outline is None and smooth_boundary and float(corner_radius_percent) > 0.5:
        try:
            outline = _sample_rounded_quad_perimeter(
                _sharp_quad_from_mesh(mesh),
                float(corner_radius_percent),
                samples_per_edge=128,
                corner_radii=corner_radii,
            )
        except Exception:
            outline = None

    if outline is None and (
        phone_silhouette is not None
        and np.count_nonzero(phone_silhouette)
        and smooth_boundary
    ):
        raw = mesh.boundary_points().astype(np.float32)
        if raw.shape[0] >= 8:
            # Light pass only — heavy Chaikin rounds corners inward.
            outline = _chaikin_closed(raw, iterations=1)

    if outline is None or outline.shape[0] < 8:
        raw = mesh.boundary_points().astype(np.float32)
        if smooth_boundary:
            smooth = _chaikin_closed(raw, iterations=4)
            if float(corner_radius_percent) > 0.5 and smooth.shape[0] >= 16:
                window = max(
                    5,
                    min(
                        17,
                        int(
                            round(
                                smooth.shape[0]
                                * float(corner_radius_percent)
                                / 500.0
                            )
                        )
                        * 2
                        + 1,
                    ),
                )
                smooth = AdaptiveMeshBuilder._smooth_closed_polyline(
                    smooth, window=window
                )
            outline = smooth
        else:
            outline = raw

    cover = _fill_closed_polyline_aa(
        outline,
        (height, width),
        # Higher supersample = smooth quarter-circles (no stair-step "katna").
        scale=10,
        expand_px=(
            0.20
            if prefer_live_boundary
            else (0.95 if smooth_boundary else 0.0)
        ),
    )

    # Live-boundary wrap must not re-inset to a smaller photo silhouette
    # (silver phones on white cards often detect ~70–85% of the blue cage).
    if (
        not prefer_live_boundary
        and phone_silhouette is not None
        and np.count_nonzero(phone_silhouette)
    ):
        try:
            from .cover_surface import CoverSurfaceEngine
            from .device_template import CornerRadii

            cr = None
            if corner_radii is not None:
                cr = CornerRadii(
                    tl=corner_radii[0],
                    tr=corner_radii[1],
                    br=corner_radii[2],
                    bl=corner_radii[3],
                )
            sym = CoverSurfaceEngine.symmetric_rim_gate(
                phone_silhouette,
                _sharp_quad_from_mesh(mesh),
                float(corner_radius_percent),
                corner_radii=cr,
            )
            if sym is not None and float(np.max(sym)) > 0.05:
                # Clip sharp cages to the product rounded rim — a rectangular
                # mesh fill must never stick past the phone's real curvature.
                cover = np.minimum(cover, sym)
                mesh_bin = cover > 0.35
                sym_bin = sym > 0.35
                sym_area = float(np.count_nonzero(sym_bin))
                if sym_area > 0 and (
                    float(np.count_nonzero(mesh_bin & sym_bin)) / sym_area
                    >= 0.90
                ):
                    # Product rim owns the tip so bottom corners match top AA.
                    deep = sym > 0.90
                    tip = (sym > 0.02) & (sym < 0.90)
                    cover = np.where(deep, np.maximum(cover, sym), cover)
                    cover = np.where(
                        tip, np.maximum(cover * 0.12 + sym * 0.88, cover), cover
                    )
                    cover = np.minimum(cover, np.clip(sym + 0.03, 0.0, 1.0))
            else:
                sil_outline = AdaptiveMeshBuilder.outer_contour_polyline(
                    phone_silhouette, smooth=False
                )
                if sil_outline is not None and sil_outline.shape[0] >= 16:
                    sil_outline = AdaptiveMeshBuilder._smooth_closed_polyline(
                        sil_outline,
                        window=max(
                            5, min(11, (sil_outline.shape[0] // 70) * 2 + 1)
                        ),
                    )
                    cover = _fill_closed_polyline_aa(
                        sil_outline,
                        (height, width),
                        scale=6,
                        expand_px=0.45,
                    )
        except Exception:
            pass

    if feather_radius > 0:
        radius = max(1, min(int(feather_radius), 48))
        binary = (cover > 0.45).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            return cover
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        soft = np.clip(dist / float(radius), 0.0, 1.0)
        # Keep a hard opaque core so white studio plates cannot chalk through.
        soft = np.where(soft > 0.40, np.maximum(soft, 0.97), soft)
        return soft.astype(np.float32)

    # Corner-weighted AA: supersample owns the curve — light corner soften only.
    binary = (cover > 0.35).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return cover
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return cover
    corner_w = _corner_proximity_map(
        (height, width),
        x0=float(xs.min()),
        y0=float(ys.min()),
        x1=float(xs.max()),
        y1=float(ys.max()),
        corner_frac=max(0.18, float(corner_radius_percent) / 100.0 * 1.85),
    )
    soft = cv2.GaussianBlur(cover, (0, 0), 0.45)
    edge = cover * (1.0 - 0.30 * corner_w) + soft * (0.30 * corner_w)
    edge = np.where(edge > 0.58, np.maximum(edge, 0.975), edge)
    return np.clip(edge, 0.0, 1.0).astype(np.float32)
