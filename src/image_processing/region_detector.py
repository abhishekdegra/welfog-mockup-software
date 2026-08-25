"""Offline phone boundary, printable surface, and hardware estimation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..utils.helpers import order_points, quad_size, to_bgr
from .mesh import (
    DEFAULT_MESH_COLS,
    DEFAULT_MESH_ROWS,
    ControlMesh,
)
from .transform import PerspectiveTransform


@dataclass
class PrintableRegion:
    """Initial editable geometry and hardware regions excluded from printing."""

    mesh: ControlMesh
    exclusion_mask: np.ndarray
    hardware_contours: List[np.ndarray] = field(default_factory=list)
    confidence: float = 0.0
    silhouette_mask: np.ndarray = None
    printable_mask: np.ndarray = None
    margin_percent: float = 0.0


@dataclass
class BoundaryEstimate:
    """Stable outer silhouette estimate in source-image coordinates."""

    quad: np.ndarray
    contour: np.ndarray
    mask: np.ndarray
    confidence: float


class PrintableRegionDetector:
    """Estimate a useful mesh and hardware mask using local OpenCV analysis."""

    ANALYSIS_LONG_EDGE = 900

    @staticmethod
    def detect(
        phone_image: np.ndarray,
        rows: int = DEFAULT_MESH_ROWS,
        cols: int = DEFAULT_MESH_COLS,
    ) -> PrintableRegion:
        """Detect a printable surface, initialise its mesh, and mask hardware."""
        source = np.asarray(phone_image)
        phone = to_bgr(source)
        boundary = PhoneBoundaryDetector.detect(source)
        mesh, printable_mask, margin_percent = (
            PrintableRegionDetector._mesh_from_boundary(
                boundary, rows, cols
            )
        )
        exclusion_mask, contours, hardware_confidence = (
            HardwareRegionDetector.detect(phone, boundary.quad)
        )

        confidence = min(
            1.0, boundary.confidence * 0.72 + hardware_confidence * 0.28
        )
        return PrintableRegion(
            mesh=mesh,
            exclusion_mask=exclusion_mask,
            hardware_contours=contours,
            confidence=confidence,
            silhouette_mask=boundary.mask,
            printable_mask=printable_mask,
            margin_percent=margin_percent,
        )

    @staticmethod
    def centered(
        phone_image: np.ndarray,
        rows: int = DEFAULT_MESH_ROWS,
        cols: int = DEFAULT_MESH_COLS,
    ) -> PrintableRegion:
        """Safe centered mesh while still retaining automatic hardware masks."""
        phone = to_bgr(phone_image)
        outer = PerspectiveTransform.default_cover(phone)
        mesh = ControlMesh.from_quad(outer, rows, cols)
        exclusion_mask, contours, confidence = HardwareRegionDetector.detect(
            phone, outer
        )
        silhouette = np.zeros(phone.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(
            silhouette, np.round(outer).astype(np.int32), 255, cv2.LINE_AA
        )
        return PrintableRegion(
            mesh, exclusion_mask, contours, confidence,
            silhouette, silhouette.copy(), 0.0,
        )

    @staticmethod
    def _mesh_from_boundary(
        boundary: BoundaryEstimate, rows: int, cols: int
    ) -> Tuple[ControlMesh, np.ndarray, float]:
        """
        Rectify the silhouette, calculate a safe printable margin, then sample
        all four edges into a Coons-patch mesh.
        """
        quad = order_points(boundary.quad)
        quad_w, quad_h = quad_size(quad)
        rect_w = max(120, int(round(quad_w)))
        rect_h = max(220, int(round(quad_h)))
        max_edge = max(rect_w, rect_h)
        if max_edge > PrintableRegionDetector.ANALYSIS_LONG_EDGE:
            scale = PrintableRegionDetector.ANALYSIS_LONG_EDGE / max_edge
            rect_w = max(120, int(round(rect_w * scale)))
            rect_h = max(220, int(round(rect_h * scale)))

        rect = np.array(
            [
                [0, 0], [rect_w - 1, 0],
                [rect_w - 1, rect_h - 1], [0, rect_h - 1],
            ],
            dtype=np.float32,
        )
        to_rect = cv2.getPerspectiveTransform(quad, rect)
        rectified = cv2.warpPerspective(
            boundary.mask, to_rect, (rect_w, rect_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rectified = PrintableRegionDetector._largest_component(rectified)

        margin = PrintableRegionDetector._automatic_margin(rectified)
        distance = cv2.distanceTransform(
            (rectified > 127).astype(np.uint8), cv2.DIST_L2, 5
        )
        printable_rect = (distance >= margin).astype(np.uint8) * 255
        if np.count_nonzero(printable_rect) < rect_w * rect_h * 0.15:
            printable_rect = rectified
            margin = 0

        rect_mesh = PrintableRegionDetector._sample_mesh(
            printable_rect, rows, cols
        )
        from_rect = cv2.getPerspectiveTransform(rect, quad)
        points = cv2.perspectiveTransform(
            rect_mesh.points.reshape(1, -1, 2), from_rect
        ).reshape(-1, 2)
        mesh = ControlMesh(points, rows, cols)

        image_h, image_w = boundary.mask.shape[:2]
        printable_full = cv2.warpPerspective(
            printable_rect, from_rect, (image_w, image_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        margin_percent = 100.0 * margin / max(min(rect_w, rect_h), 1)
        return mesh, printable_full, margin_percent

    @staticmethod
    def _automatic_margin(mask: np.ndarray) -> int:
        """
        Tiny print-safe inset — keep wrap near the real cover edge.

        A large margin was leaving a visible grey phone/case border where
        artwork should reach. Rough silhouettes still get a little more inset.
        """
        height, width = mask.shape
        perimeter_contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        roughness = 0.0
        if perimeter_contours:
            contour = max(perimeter_contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            hull_perimeter = cv2.arcLength(cv2.convexHull(contour), True)
            if hull_perimeter > 0:
                roughness = np.clip(
                    perimeter / hull_perimeter - 1.0, 0.0, 0.5
                )
        base = min(height, width) * (0.0015 + roughness * 0.005)
        return max(0, int(round(base)))

    @staticmethod
    def _sample_mesh(mask: np.ndarray, rows: int, cols: int) -> ControlMesh:
        """Sample curved mask edges and interpolate a stable interior patch."""
        height, width = mask.shape
        binary = mask > 0
        fallback = ControlMesh.from_quad(
            np.array(
                [[0, 0], [width - 1, 0],
                 [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            ),
            rows, cols,
        )
        if np.count_nonzero(binary) == 0:
            return fallback

        ys, xs = np.nonzero(binary)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        span_x = max(float(x_max - x_min), 1.0)
        span_y = max(float(y_max - y_min), 1.0)

        # Diagonal extrema locate the 45-degree points of rounded corners.
        # Using absolute x/y extrema would place a "corner" halfway along a
        # curved side, producing the large zig-zags this editor must avoid.
        normal_x = (xs.astype(np.float32) - x_min) / span_x
        normal_y = (ys.astype(np.float32) - y_min) / span_y
        tl = np.array(
            [xs[np.argmin(normal_x + normal_y)],
             ys[np.argmin(normal_x + normal_y)]], np.float32
        )
        tr = np.array(
            [xs[np.argmin((1.0 - normal_x) + normal_y)],
             ys[np.argmin((1.0 - normal_x) + normal_y)]], np.float32
        )
        br = np.array(
            [xs[np.argmin((1.0 - normal_x) + (1.0 - normal_y))],
             ys[np.argmin((1.0 - normal_x) + (1.0 - normal_y))]],
            np.float32,
        )
        bl = np.array(
            [xs[np.argmin(normal_x + (1.0 - normal_y))],
             ys[np.argmin(normal_x + (1.0 - normal_y))]], np.float32
        )

        left = np.zeros((rows, 2), np.float32)
        right = np.zeros((rows, 2), np.float32)
        top = np.zeros((cols, 2), np.float32)
        bottom = np.zeros((cols, 2), np.float32)

        for index in range(rows):
            v = index / max(rows - 1, 1)
            left_y = tl[1] * (1.0 - v) + bl[1] * v
            right_y = tr[1] * (1.0 - v) + br[1] * v
            y_left = PrintableRegionDetector._nearest_nonempty_row(
                binary, int(round(left_y))
            )
            y_right = PrintableRegionDetector._nearest_nonempty_row(
                binary, int(round(right_y))
            )
            left_xs = np.flatnonzero(binary[y_left])
            right_xs = np.flatnonzero(binary[y_right])
            left[index] = (left_xs.min(), y_left)
            right[index] = (right_xs.max(), y_right)

        for index in range(cols):
            u = index / max(cols - 1, 1)
            top_x = tl[0] * (1.0 - u) + tr[0] * u
            bottom_x = bl[0] * (1.0 - u) + br[0] * u
            x_top = PrintableRegionDetector._nearest_nonempty_col(
                binary, int(round(top_x))
            )
            x_bottom = PrintableRegionDetector._nearest_nonempty_col(
                binary, int(round(bottom_x))
            )
            top_ys = np.flatnonzero(binary[:, x_top])
            bottom_ys = np.flatnonzero(binary[:, x_bottom])
            top[index] = (x_top, top_ys.min())
            bottom[index] = (x_bottom, bottom_ys.max())

        # Shared, diagonally estimated corners keep the Coons patch continuous.
        top[0], left[0] = tl, tl
        top[-1], right[0] = tr, tr
        bottom[-1], right[-1] = br, br
        bottom[0], left[-1] = bl, bl

        points = []
        for row in range(rows):
            v = row / max(rows - 1, 1)
            for col in range(cols):
                u = col / max(cols - 1, 1)
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
                points.append(edge_blend - bilinear)
        return ControlMesh(np.asarray(points, np.float32), rows, cols)

    @staticmethod
    def _nearest_nonempty_row(mask: np.ndarray, row: int) -> int:
        """Closest row containing silhouette pixels."""
        row = int(np.clip(row, 0, mask.shape[0] - 1))
        for radius in range(mask.shape[0]):
            for candidate in (row - radius, row + radius):
                if 0 <= candidate < mask.shape[0] and mask[candidate].any():
                    return candidate
        return row

    @staticmethod
    def _nearest_nonempty_col(mask: np.ndarray, col: int) -> int:
        """Closest column containing silhouette pixels."""
        col = int(np.clip(col, 0, mask.shape[1] - 1))
        for radius in range(mask.shape[1]):
            for candidate in (col - radius, col + radius):
                if 0 <= candidate < mask.shape[1] and mask[:, candidate].any():
                    return candidate
        return col

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        """Retain only the largest connected foreground component."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), 8
        )
        if count <= 1:
            return mask
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == label).astype(np.uint8) * 255


class PhoneBoundaryDetector:
    """Multi-candidate, confidence-scored phone silhouette estimator."""

    MAX_ANALYSIS_EDGE = 1000

    @staticmethod
    def detect(image: np.ndarray) -> BoundaryEstimate:
        """Estimate the outer phone/cover silhouette without learned models."""
        source = np.asarray(image)
        bgr = to_bgr(source)
        original_h, original_w = bgr.shape[:2]
        scale = min(
            1.0,
            PhoneBoundaryDetector.MAX_ANALYSIS_EDGE
            / max(original_h, original_w),
        )
        small = (
            cv2.resize(
                bgr,
                (max(1, int(round(original_w * scale))),
                 max(1, int(round(original_h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0 else bgr
        )

        candidates = []
        if source.ndim == 3 and source.shape[2] == 4:
            alpha = source[:, :, 3]
            if scale < 1.0:
                alpha = cv2.resize(
                    alpha, (small.shape[1], small.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            if alpha.max() - alpha.min() > 32:
                candidates.append(("alpha", (alpha > 12).astype(np.uint8) * 255))

        candidates.extend(
            [
                ("border", PhoneBoundaryDetector._border_colour_mask(small)),
                ("grabcut", PhoneBoundaryDetector._grabcut_mask(small)),
                ("edges", PhoneBoundaryDetector._edge_mask(small)),
            ]
        )

        best = None
        for name, mask in candidates:
            estimate = PhoneBoundaryDetector._score_mask(mask, name)
            if estimate is not None and (
                best is None or estimate[0] > best[0]
            ):
                best = estimate

        if best is None:
            quad = PerspectiveTransform.detect_cover(bgr)
            mask = np.zeros((original_h, original_w), np.uint8)
            cv2.fillConvexPoly(
                mask, np.round(quad).astype(np.int32), 255, cv2.LINE_AA
            )
            contour = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
            return BoundaryEstimate(quad, contour, mask, 0.35)

        score, contour, mask = best
        contour = contour.astype(np.float32) / scale
        contour[:, :, 0] = np.clip(contour[:, :, 0], 0, original_w - 1)
        contour[:, :, 1] = np.clip(contour[:, :, 1], 0, original_h - 1)
        quad = PhoneBoundaryDetector._contour_quad(contour)
        full_mask = np.zeros((original_h, original_w), np.uint8)
        cv2.drawContours(
            full_mask, [np.round(contour).astype(np.int32)],
            -1, 255, -1, cv2.LINE_AA,
        )
        return BoundaryEstimate(quad, contour, full_mask, score)

    @staticmethod
    def _border_colour_mask(image: np.ndarray) -> np.ndarray:
        """Foreground from robust Lab distance to the image border."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        height, width = image.shape[:2]
        band = max(2, int(round(min(height, width) * 0.035)))
        border = np.concatenate(
            [
                lab[:band].reshape(-1, 3),
                lab[-band:].reshape(-1, 3),
                lab[:, :band].reshape(-1, 3),
                lab[:, -band:].reshape(-1, 3),
            ]
        )
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab - background, axis=2)
        distance = cv2.GaussianBlur(distance, (7, 7), 0)
        distance_u8 = np.clip(
            distance / max(np.percentile(distance, 98), 1e-6) * 255,
            0, 255,
        ).astype(np.uint8)
        _, mask = cv2.threshold(
            distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return PhoneBoundaryDetector._clean_mask(mask)

    @staticmethod
    def _grabcut_mask(image: np.ndarray) -> np.ndarray:
        """GrabCut with border-background and central-subject priors."""
        height, width = image.shape[:2]
        mask = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
        border = max(2, int(round(min(height, width) * 0.025)))
        mask[:border] = cv2.GC_BGD
        mask[-border:] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        center_x1, center_x2 = int(width * 0.28), int(width * 0.72)
        center_y1, center_y2 = int(height * 0.12), int(height * 0.88)
        mask[center_y1:center_y2, center_x1:center_x2] = cv2.GC_PR_FGD
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                image, mask, None, bg_model, fg_model, 3,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            return np.zeros((height, width), np.uint8)
        foreground = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        return PhoneBoundaryDetector._clean_mask(foreground)

    @staticmethod
    def _edge_mask(image: np.ndarray) -> np.ndarray:
        """Closed external silhouette inferred from strong edges."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 50, 50)
        median = float(np.median(gray))
        edges = cv2.Canny(
            gray, int(max(15, median * 0.55)),
            int(min(255, median * 1.45)),
        )
        kernel_size = max(5, int(min(image.shape[:2]) * 0.025) | 1)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
            iterations=2,
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros(gray.shape, np.uint8)
        if contours:
            cv2.drawContours(
                mask, [max(contours, key=cv2.contourArea)], -1, 255, -1
            )
        return PhoneBoundaryDetector._clean_mask(mask)

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """Close holes, remove speckles, and retain the main component."""
        size = max(3, int(min(mask.shape) * 0.018) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        return PrintableRegionDetector._largest_component(cleaned)

    @staticmethod
    def _score_mask(
        mask: np.ndarray, source_name: str
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Reject implausible masks and score phone-like silhouettes."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        height, width = mask.shape
        total = height * width
        fraction = area / max(total, 1)
        if fraction < 0.06 or fraction > 0.96:
            return None

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        if min(rect_w, rect_h) < 5:
            return None
        aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        if aspect < 1.15 or aspect > 4.2:
            return None

        rect_area = rect_w * rect_h
        rectangularity = np.clip(area / max(rect_area, 1), 0.0, 1.0)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = np.clip(area / max(hull_area, 1), 0.0, 1.0)
        moments = cv2.moments(contour)
        center_x = moments["m10"] / max(moments["m00"], 1e-6)
        center_y = moments["m01"] / max(moments["m00"], 1e-6)
        center_distance = np.hypot(
            (center_x - width / 2) / max(width / 2, 1),
            (center_y - height / 2) / max(height / 2, 1),
        )
        x, y, box_w, box_h = cv2.boundingRect(contour)
        touches = sum(
            [
                x <= 1, y <= 1,
                x + box_w >= width - 1,
                y + box_h >= height - 1,
            ]
        )
        aspect_score = 1.0 - min(abs(aspect - 2.0) / 2.5, 1.0)
        # Product shots often fill ~30–60% of the frame. The old 0.48 target
        # preferred GrabCut "interior islands" on white phones and left the
        # real rim uncovered.
        if 0.28 <= fraction <= 0.62:
            area_score = 0.88 + 0.12 * (
                1.0 - min(abs(fraction - 0.45) / 0.25, 1.0)
            )
        else:
            area_score = 1.0 - min(abs(fraction - 0.42) / 0.42, 1.0)
        source_bonus = 0.12 if source_name == "alpha" else 0.0
        # Prefer complete outer silhouettes over half-phone interiors.
        fullness_bonus = 0.0
        if rectangularity >= 0.86 and solidity >= 0.90 and fraction >= 0.32:
            fullness_bonus = 0.10 * min(fraction / 0.50, 1.0)
        if source_name == "border" and fullness_bonus > 0:
            fullness_bonus += 0.04
        score = (
            rectangularity * 0.25
            + solidity * 0.18
            + aspect_score * 0.18
            + area_score * 0.16
            + (1.0 - min(center_distance, 1.0)) * 0.14
            + fullness_bonus
            + source_bonus
            - touches * 0.08
        )
        return float(np.clip(score, 0.0, 1.0)), contour, mask

    @staticmethod
    def _contour_quad(contour: np.ndarray) -> np.ndarray:
        """Prefer true perspective corners; fall back to a stable upright box."""
        contour_i = np.round(contour).astype(np.int32)
        pts2 = contour_i.reshape(-1, 2)

        def _aabb() -> np.ndarray:
            x1, x2 = float(pts2[:, 0].min()), float(pts2[:, 0].max())
            y1, y2 = float(pts2[:, 1].min()), float(pts2[:, 1].max())
            return order_points(
                np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                )
            )

        def _tilt_deg(quad: np.ndarray) -> float:
            edge = quad[1] - quad[0]
            angle = abs(float(np.degrees(np.arctan2(edge[1], edge[0]))))
            a = angle % 90.0
            return float(min(a, 90.0 - a))

        perimeter = cv2.arcLength(contour_i, True)
        for epsilon_factor in (0.018, 0.025, 0.035, 0.05):
            approx = cv2.approxPolyDP(
                contour_i, epsilon_factor * perimeter, True
            )
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = order_points(
                    approx.reshape(4, 2).astype(np.float32)
                )
                # Approx can still be a skewed trap on upright phones.
                if _tilt_deg(quad) <= 3.5:
                    return _aabb()
                return quad
        rect = cv2.minAreaRect(contour_i)
        _c, _s, angle = rect
        a = abs(float(angle)) % 90.0
        if min(a, 90.0 - a) <= 3.5:
            return _aabb()
        return order_points(
            cv2.boxPoints(rect).astype(np.float32)
        )


class HardwareRegionDetector:
    """
    Detect camera modules, lenses, flash/sensor circles, and side controls.

    Detection runs in a fronto-parallel rectification of the cover. This makes
    size/aspect rules stable even when the source photograph has perspective.
    """

    @staticmethod
    def detect(
        phone_image: np.ndarray, outer_quad: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], float]:
        """Return full-resolution exclusion mask, contours, and confidence."""
        phone = to_bgr(phone_image)
        image_h, image_w = phone.shape[:2]
        quad = order_points(outer_quad)
        quad_w, quad_h = quad_size(quad)

        if quad_w < 8 or quad_h < 8:
            return (
                np.zeros((image_h, image_w), dtype=np.uint8),
                [],
                0.0,
            )

        scale = min(
            1.0,
            HardwareRegionDetector._analysis_scale(quad_w, quad_h),
        )
        rect_w = max(80, int(round(quad_w * scale)))
        rect_h = max(160, int(round(quad_h * scale)))
        rect = np.array(
            [
                [0, 0],
                [rect_w - 1, 0],
                [rect_w - 1, rect_h - 1],
                [0, rect_h - 1],
            ],
            dtype=np.float32,
        )

        to_rect = cv2.getPerspectiveTransform(quad, rect)
        rectified = cv2.warpPerspective(
            phone,
            to_rect,
            (rect_w, rect_h),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_REPLICATE,
        )

        rect_mask, scores = HardwareRegionDetector._detect_rectified(rectified)
        # Tiny safety pad only — feature detectors already include mild padding.
        # Prefer slightly smaller exclusions over oversized islands.
        safety = max(1, int(round(min(rect_w, rect_h) * 0.002)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (safety * 2 + 1, safety * 2 + 1)
        )
        rect_mask = cv2.dilate(rect_mask, kernel, iterations=1)
        # Mild AA without ballooning the cutout.
        if safety >= 1:
            rect_mask = cv2.GaussianBlur(rect_mask, (3, 3), 0)

        from_rect = cv2.getPerspectiveTransform(rect, quad)
        full_mask = cv2.warpPerspective(
            rect_mask,
            from_rect,
            (image_w, image_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Re-binarise after warp so soft edges do not become jagged polygons.
        full_mask = (full_mask > 96).astype(np.uint8) * 255
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, close_k)
        # Full-resolution side pass — thin button ridges survive perspective
        # better in the original photo than in the rectified strip alone.
        # Side buttons painted after the soft AA blur so their cores stay hard
        # (pixel-identical phone restore in the compositor).
        soft = cv2.GaussianBlur(full_mask, (3, 3), 0)
        side_only = np.zeros_like(full_mask)
        HardwareRegionDetector._detect_side_hardware_fullres(
            phone, side_only, quad
        )
        if np.count_nonzero(side_only):
            # Slightly dilate so wrap faces clear the whole button ridge.
            dilate = max(1, int(round(min(image_w, image_h) * 0.0018)))
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)
            )
            side_only = cv2.dilate(side_only, k, iterations=1)
            full_mask = cv2.max(soft, side_only)
        else:
            full_mask = soft

        # Drop corner glare / false side hits after full-res merge, then snap
        # the primary camera island to a clean rounded rectangle — but only
        # when the plate is still a jagged freeform (not already square).
        HardwareRegionDetector._prune_orphan_exclusions(
            full_mask, image_w, image_h
        )
        HardwareRegionDetector._snap_camera_island(
            full_mask, image_w, image_h, only_if_jagged=True
        )
        HardwareRegionDetector._refine_camera_to_dark_plate(
            phone, full_mask, quad
        )
        HardwareRegionDetector._clip_face_openings_to_cover(
            full_mask, quad, image_w, image_h
        )

        # Dense / circular contours → smooth overlay & editable cutouts.
        contours = HardwareRegionDetector._smooth_exclusion_contours(full_mask)
        confidence = min(0.45, sum(scores) / max(len(scores), 1) * 0.45)
        return full_mask, contours, confidence

    @staticmethod
    def _analysis_scale(width: float, height: float) -> float:
        """Scale so the rectified long edge stays inexpensive to analyse."""
        return PrintableRegionDetector.ANALYSIS_LONG_EDGE / max(width, height)

    @staticmethod
    def _detect_rectified(image: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Detect hardware in a fronto-parallel cover image."""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        scores: List[float] = []

        # Camera hardware is overwhelmingly in the upper portion.
        top_height = max(1, int(height * 0.50))
        top = gray[:top_height]
        blur = cv2.GaussianBlur(top, (7, 7), 1.4)

        circles = HardwareRegionDetector._detect_circles(
            blur, mask, scores, width, top_height
        )
        # Drop orphan lenses far from the dominant camera island (works for
        # left / right / center modules — never assume a fixed corner).
        circles = HardwareRegionDetector._primary_circle_cluster(
            circles, width, top_height
        )
        # Rebuild mask cores from the pruned circle set only.
        mask[:top_height] = 0
        scores.clear()
        for x, y, radius in circles:
            padding = max(1, int(radius * 0.04))
            cv2.circle(mask, (x, y), radius + padding, 255, -1, cv2.LINE_AA)
            scores.append(0.95)
        satellites = HardwareRegionDetector._detect_flash_spots(
            blur, mask, scores, width, top_height, circles
        )
        circles = circles + satellites
        # Discrete rings on a flat back must stay separate openings. A raised
        # plate (Redmi-style island) still merges; cover-colored gaps do not.
        discrete_face = False
        if len(circles) >= 2:
            lens_xyr = [
                (float(x), float(y), float(r)) for x, y, r in circles
            ]
            xs = [c[0] for c in lens_xyr]
            ys = [c[1] for c in lens_xyr]
            rs = [c[2] for c in lens_xyr]
            discrete_face = HardwareRegionDetector._lenses_sit_on_cover_face(
                gray,
                lens_xyr,
                min(x - r for x, r in zip(xs, rs)),
                min(y - r for y, r in zip(ys, rs)),
                max(x + r for x, r in zip(xs, rs)),
                max(y + r for y, r in zip(ys, rs)),
            )
        plate = None
        if not discrete_face:
            plate = HardwareRegionDetector._detect_square_camera_plate(
                blur, width, top_height, circles
            )
        if discrete_face:
            pass
        elif plate is not None:
            x1, y1, x2, y2 = plate
            corner = max(3, int(min(x2 - x1, y2 - y1) * 0.14))
            HardwareRegionDetector._rounded_rectangle(
                mask, x1, y1, x2, y2, corner, expand_px=0.5
            )
            scores.append(0.96)
        else:
            HardwareRegionDetector._expand_cluster_to_module(
                blur, mask, scores, circles, width, top_height
            )
            if len(circles) < 2:
                HardwareRegionDetector._detect_camera_modules(
                    blur, mask, scores, width, top_height
                )
        HardwareRegionDetector._detect_side_hardware(
            gray, mask, scores, width, height
        )

        if not discrete_face:
            HardwareRegionDetector._merge_camera_cluster(
                mask, width, top_height, circles if plate is None else []
            )
            # Tightening dissolves clean plate corners into jagged freeforms —
            # only run when we fell back to circle/blob unions.
            if plate is None:
                HardwareRegionDetector._tighten_to_hardware(
                    mask, gray, top_height, circles
                )
        HardwareRegionDetector._prune_orphan_exclusions(
            mask, width, height
        )

        return mask, scores

    @staticmethod
    def _detect_circles(
        gray: np.ndarray,
        mask: np.ndarray,
        scores: List[float],
        width: int,
        top_height: int,
    ) -> List[Tuple[int, int, int]]:
        """Detect round lenses, flash, microphones, and sensor holes."""
        min_dim = min(width, top_height)
        min_radius = max(3, int(min_dim * 0.018))
        max_radius = max(min_radius + 2, int(min_dim * 0.16))
        found: List[Tuple[int, int, int]] = []

        # Lens pass + sensitive small-circle pass for flash / mic / sensors.
        passes = (
            (1.2, 18, min_radius, max_radius, max(8, int(min_radius * 1.6))),
            (0.8, 22, min_radius, max_radius, max(8, int(min_radius * 1.6))),
            (
                0.6, 12,
                max(2, int(min_dim * 0.008)),
                max(4, int(min_dim * 0.07)),
                max(5, int(min_dim * 0.025)),
            ),
        )
        for blur_sigma, param2, r_min, r_max, min_dist in passes:
            source = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
            circles = cv2.HoughCircles(
                source,
                cv2.HOUGH_GRADIENT,
                dp=1.15,
                minDist=min_dist,
                param1=85,
                param2=param2,
                minRadius=r_min,
                maxRadius=r_max,
            )
            if circles is None:
                continue
            for x, y, radius in np.round(circles[0]).astype(int):
                # Reject dead-center logo graphics; keep left/right camera islands
                # and slightly off-center modern bumps.
                center_x = abs((x / max(width, 1)) - 0.5)
                if center_x < 0.07 and y > top_height * 0.38:
                    continue
                if y > top_height * 0.92:
                    continue
                # Rounded case corners look circular to Hough and sit on the
                # rectified frame origin. Unioning them with nearby lenses
                # paints one blob over the whole array.
                if (
                    (x - radius) <= 1
                    and (y - radius) <= 1
                ) or (
                    (x + radius) >= (width - 2)
                    and (y - radius) <= 1
                ):
                    continue
                # Reject circles whose interior looks like plain cover.
                if not HardwareRegionDetector._circle_looks_like_hardware(
                    gray, x, y, radius
                ):
                    continue
                # Deduplicate near-identical detections from the two passes.
                if any(
                    (x - ox) ** 2 + (y - oy) ** 2 < (max(radius, oradius) * 0.6) ** 2
                    for ox, oy, oradius in found
                ):
                    continue
                found.append((x, y, radius))

        for x, y, radius in found:
            # Prefer a tight circle; uncertainty → slightly smaller pad.
            padding = max(1, int(radius * 0.04))
            cv2.circle(mask, (x, y), radius + padding, 255, -1, cv2.LINE_AA)
            scores.append(0.95)
        return found

    @staticmethod
    def _detect_square_camera_plate(
        gray: np.ndarray,
        width: int,
        top_height: int,
        circles: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate a raised camera plate anywhere in the upper cover.

        Model-agnostic: seeds from the lens cluster when present, otherwise
        scores every compact upper plate (left / center / right). Accepts
        near-square islands and moderate stadiums.
        """
        if top_height < 24 or width < 24:
            return None
        roi = gray[:top_height]
        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(blur, 36, 110)
        edges = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Seed from lenses when available; else neutral top-centre prior only
        # as a soft bias (never a hard left/right assumption).
        if circles:
            seed_cx = float(np.mean([c[0] for c in circles]))
            seed_cy = float(np.mean([c[1] for c in circles]))
            has_seed = True
        else:
            seed_cx = width * 0.5
            seed_cy = top_height * 0.28
            has_seed = False
        # Aspect gate: multi-lens columns may be taller stadiums.
        max_aspect = 1.85
        if circles and len(circles) >= 2:
            ys = [float(c[1]) for c in circles]
            xs = [float(c[0]) for c in circles]
            span_y = max(ys) - min(ys)
            span_x = max(xs) - min(xs)
            if span_y > span_x * 1.15:
                max_aspect = 2.25
            elif span_x > span_y * 1.15:
                max_aspect = 2.25
        best = None
        best_score = 0.0
        min_side = max(28, int(min(width, top_height) * 0.12))
        max_side = int(min(width, top_height) * 0.78)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_side * min_side * 0.30:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_side or h < min_side or w > max_side or h > max_side:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > max_aspect:
                continue
            cx, cy = x + w * 0.5, y + h * 0.5
            if cy > top_height * 0.68:
                continue
            # Tiny dead-center logos (not camera islands) — reject only when
            # we have no lens seed and the blob is small + perfectly centered.
            if (
                not has_seed
                and abs(cx / max(width, 1) - 0.5) < 0.06
                and w < width * 0.22
                and h < top_height * 0.22
            ):
                continue
            fill = area / max(float(w * h), 1.0)
            if fill < 0.28:
                continue
            dist = ((cx - seed_cx) ** 2 + (cy - seed_cy) ** 2) ** 0.5
            near = 1.0 / (1.0 + dist / max(min_side, 1))
            # Lens seed dominates; without it prefer larger upper plates.
            seed_weight = 0.55 if has_seed else 0.20
            score = (
                (w * h)
                * (1.55 - min(aspect, 1.55))
                * (0.55 + 0.45 * fill)
                * ((1.0 - seed_weight) + seed_weight * near)
            )
            if score > best_score:
                best_score = score
                pad = max(1, int(min(w, h) * 0.012))
                rx1 = max(0, x - pad)
                ry1 = max(0, y - pad)
                rx2 = min(width - 1, x + w + pad)
                ry2 = min(top_height - 1, y + h + pad)
                patch = blur[ry1 : ry2 + 1, rx1 : rx2 + 1]
                if patch.size >= 64:
                    # Plate vs local cover — works for dark OR light modules.
                    cover_med = float(np.median(blur))
                    diff = np.abs(patch.astype(np.float32) - cover_med)
                    thr = max(7.0, float(np.percentile(diff, 60)))
                    solid = (diff >= thr).astype(np.uint8) * 255
                    solid = cv2.morphologyEx(
                        solid,
                        cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                        iterations=2,
                    )
                    scnts, _ = cv2.findContours(
                        solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if scnts:
                        sc = max(scnts, key=cv2.contourArea)
                        sx, sy, sw, sh = cv2.boundingRect(sc)
                        if (
                            sw >= min_side * 0.70
                            and sh >= min_side * 0.70
                            and max(sw, sh) / max(min(sw, sh), 1) <= max_aspect
                        ):
                            best = (
                                rx1 + sx,
                                ry1 + sy,
                                rx1 + sx + sw,
                                ry1 + sy + sh,
                            )
                            continue
                best = (rx1, ry1, rx2, ry2)
        return best

    @staticmethod
    def _primary_circle_cluster(
        circles: List[Tuple[int, int, int]],
        width: int,
        top_height: int,
    ) -> List[Tuple[int, int, int]]:
        """Keep only the dominant lens/flash cluster; drop far orphans."""
        if len(circles) <= 1:
            return list(circles)
        rs = [float(c[2]) for c in circles]
        link = max(14.0, float(np.median(rs)) * 2.85)
        n = len(circles)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            xi, yi, ri = circles[i]
            for j in range(i + 1, n):
                xj, yj, rj = circles[j]
                dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                if dist <= link + 0.35 * (ri + rj):
                    union(i, j)

        groups: dict = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(circles[i])

        def _score(group: List[Tuple[int, int, int]]) -> float:
            area = float(sum(np.pi * (c[2] ** 2) for c in group))
            # Prefer multi-lens islands and upper placements.
            cy = float(np.mean([c[1] for c in group]))
            height_bias = 1.0 + 0.35 * (1.0 - cy / max(top_height, 1))
            return area * height_bias * (1.0 + 0.15 * len(group))

        best = max(groups.values(), key=_score)
        # Reject a lone small circle on the opposite side of a real island.
        if len(best) == 1 and len(circles) >= 2:
            alone = best[0]
            others = [c for c in circles if c != alone]
            ox = float(np.mean([c[0] for c in others]))
            if abs(alone[0] - ox) > width * 0.28 and alone[2] < np.median(
                [c[2] for c in others]
            ) * 1.15:
                return list(others)
        return list(best)

    @staticmethod
    def _expand_cluster_to_module(
        gray: np.ndarray,
        mask: np.ndarray,
        scores: List[float],
        circles: List[Tuple[int, int, int]],
        width: int,
        top_height: int,
    ) -> None:
        """Grow lens circles into the raised square camera plate when present."""
        if len(circles) < 2:
            return
        xs = [c[0] for c in circles]
        ys = [c[1] for c in circles]
        rs = [c[2] for c in circles]
        med_r = float(np.median(rs))
        pad = max(4, int(med_r * 0.85))
        x1 = max(0, int(min(xs) - max(rs) - pad))
        y1 = max(0, int(min(ys) - max(rs) - pad))
        x2 = min(width - 1, int(max(xs) + max(rs) + pad))
        y2 = min(top_height - 1, int(max(ys) + max(rs) + pad))
        # Search a slightly larger ROI for the plate edge.
        grow = max(6, int(med_r * 1.1))
        rx1 = max(0, x1 - grow)
        ry1 = max(0, y1 - grow)
        rx2 = min(width - 1, x2 + grow)
        ry2 = min(top_height - 1, y2 + grow)
        roi = gray[ry1 : ry2 + 1, rx1 : rx2 + 1]
        if roi.size < 64:
            corner = max(3, int(min(x2 - x1, y2 - y1) * 0.18))
            HardwareRegionDetector._rounded_rectangle(
                mask, x1, y1, x2, y2, corner
            )
            scores.append(0.8)
            return
        blur = cv2.GaussianBlur(roi, (0, 0), 1.2)
        # Plate usually differs from cover; edges form a compact rectangle.
        edges = cv2.Canny(blur, 40, 110)
        edges = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best = None
        best_score = 0.0
        cluster_cx = float(np.mean(xs) - rx1)
        cluster_cy = float(np.mean(ys) - ry1)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < med_r * med_r * 2.5:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < med_r * 2.2 or bh < med_r * 2.2:
                continue
            if bw > roi.shape[1] * 0.95 or bh > roi.shape[0] * 0.95:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 1.85:
                continue
            cx = bx + bw * 0.5
            cy = by + bh * 0.5
            if abs(cx - cluster_cx) > med_r * 2.8 or abs(cy - cluster_cy) > med_r * 2.8:
                continue
            score = area / max(aspect, 1.0)
            if score > best_score:
                best_score = score
                best = (bx + rx1, by + ry1, bx + bw + rx1, by + bh + ry1)
        if best is None:
            # Fallback: padded AABB around lenses (square-ish island).
            best = (x1, y1, x2, y2)
        bx1, by1, bx2, by2 = best
        # Mild pad so artwork clears the raised plate edge.
        extra = max(2, int(med_r * 0.12))
        bx1 = max(0, bx1 - extra)
        by1 = max(0, by1 - extra)
        bx2 = min(width - 1, bx2 + extra)
        by2 = min(top_height - 1, by2 + extra)
        corner = max(3, int(min(bx2 - bx1, by2 - by1) * 0.16))
        HardwareRegionDetector._rounded_rectangle(
            mask, bx1, by1, bx2, by2, corner
        )
        scores.append(0.88)

    @staticmethod
    def _prune_orphan_exclusions(
        mask: np.ndarray, width: int, height: int
    ) -> None:
        """
        Keep the main camera island + rim side-buttons; drop stray blobs.

        Prevents fake holes (logo reflections, corner glare) from punching
        through a finished mockup.
        """
        binary = (mask > 32).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            return
        count, labels, stats, cents = cv2.connectedComponentsWithStats(
            binary, 8
        )
        if count <= 2:
            return
        edge = max(4, int(width * 0.055))
        top_band = int(height * 0.55)
        # Score camera candidates in the upper half.
        cam_label = -1
        cam_score = -1.0
        keep = np.zeros(count, dtype=bool)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            # Speckles glued to the analysis-frame edge are not hardware.
            on_frame = y <= 1 or x <= 1 or (x + w) >= (width - 1)
            if on_frame and area < 180 and max(w, h) <= 22:
                continue
            cx, cy = float(cents[label][0]), float(cents[label][1])
            touches_side = x <= edge or (x + w) >= (width - edge)
            aspect = max(w, h) / max(min(w, h), 1)
            # Tall volume / power rockers.
            skinny_rocker = (
                w <= max(10, int(width * 0.09)) and h >= w * 1.4
            )
            # Compact mid-side fingerprint / power / mute pills — not skinny
            # enough for the rocker rule, but must keep a visible cutout gap.
            compact_side = (
                touches_side
                and area >= 40
                and w <= max(14, int(width * 0.12))
                and height * 0.025 <= h <= height * 0.20
                and aspect <= 3.2
                and height * 0.12 < cy < height * 0.90
            )
            if touches_side and area >= 40 and (skinny_rocker or compact_side):
                keep[label] = True
                continue
            if cy <= top_band and area >= 80:
                # Prefer large, somewhat square upper components.
                aspect = max(w, h) / max(min(w, h), 1)
                score = float(area) * (1.35 if aspect < 1.75 else 1.0)
                score *= 1.0 + 0.25 * (1.0 - cy / max(top_band, 1))
                if score > cam_score:
                    cam_score = score
                    cam_label = label
        if cam_label > 0:
            keep[cam_label] = True
            # Discrete lens arrays are several compact upper circles.
            # Keeping only the largest (plus satellites within its own
            # radius) deleted the rest of the stack.
            HardwareRegionDetector._keep_clustered_upper_openings(
                keep, stats, cents, cam_label, top_band
            )
        # If nothing classified, keep the largest component only.
        if not np.any(keep[1:]):
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            keep[largest] = True
        for label in range(1, count):
            if not keep[label]:
                mask[labels == label] = 0

    @staticmethod
    def _keep_clustered_upper_openings(
        keep: np.ndarray,
        stats: np.ndarray,
        cents: np.ndarray,
        cam_label: int,
        top_band: int,
    ) -> None:
        """
        Keep every opening in the same upper cluster as the primary camera.

        A raised island is one blob (unchanged). Separate lenses / flash on a
        flat back are several compact circles — they must stay together.
        """
        count = int(stats.shape[0])
        members: List[int] = []
        radii: List[float] = []
        for label in range(1, count):
            _x, _y, bw, bh, area = stats[label]
            cy = float(cents[label][1])
            if cy > top_band or area < 24:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            fill = float(area) / max(float(bw * bh), 1.0)
            compact = aspect <= 1.55 and fill >= 0.55
            if compact or label == cam_label:
                members.append(label)
                radii.append(0.5 * float(max(bw, bh)))
        if len(members) <= 1:
            # Single island: keep nearby smaller satellites (flash / mic).
            cx0, cy0 = float(cents[cam_label][0]), float(cents[cam_label][1])
            cam_r = 0.5 * max(stats[cam_label][2], stats[cam_label][3])
            for label in range(1, count):
                if keep[label]:
                    continue
                area = int(stats[label][4])
                if area < 24 or area > stats[cam_label][4] * 0.55:
                    continue
                cx, cy = float(cents[label][0]), float(cents[label][1])
                if ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 <= cam_r * 1.35:
                    keep[label] = True
            return
        link = max(14.0, float(np.median(radii)) * 2.85)
        parent = list(range(len(members)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, li in enumerate(members):
            xi, yi = float(cents[li][0]), float(cents[li][1])
            ri = radii[i]
            for j in range(i + 1, len(members)):
                lj = members[j]
                xj, yj = float(cents[lj][0]), float(cents[lj][1])
                dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                if dist <= link + 0.35 * (ri + radii[j]):
                    ri_i, rj_i = find(i), find(j)
                    if ri_i != rj_i:
                        parent[rj_i] = ri_i
        groups: dict = {}
        for i, li in enumerate(members):
            groups.setdefault(find(i), []).append(i)

        def _gscore(idxs: List[int]) -> float:
            labs = [members[k] for k in idxs]
            area = float(sum(int(stats[lab][4]) for lab in labs))
            has_cam = 1.25 if cam_label in labs else 1.0
            return area * has_cam * (1.0 + 0.12 * len(idxs))

        best = max(groups.values(), key=_gscore)
        for k in best:
            keep[members[k]] = True
        xs = [float(cents[members[k]][0]) for k in best]
        ys = [float(cents[members[k]][1]) for k in best]
        rs = [radii[k] for k in best]
        cx0 = float(np.mean(xs))
        cy0 = float(np.mean(ys))
        reach = max(rs) * 2.2 + 0.55 * max(
            max(xs) - min(xs), max(ys) - min(ys)
        )
        cluster_area = max(int(stats[members[k]][4]) for k in best)
        for label in range(1, count):
            if keep[label]:
                continue
            area = int(stats[label][4])
            if area < 16 or area > cluster_area * 0.85:
                continue
            if float(cents[label][1]) > top_band:
                continue
            cx, cy = float(cents[label][0]), float(cents[label][1])
            if ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 <= reach:
                keep[label] = True

    @staticmethod
    def _snap_camera_island(
        mask: np.ndarray,
        width: int,
        height: int,
        *,
        only_if_jagged: bool = False,
    ) -> None:
        """
        Replace the primary upper exclusion with a clean rounded rectangle.

        Jagged freeform camera holes are the #1 tell that a wrap is fake —
        production mockups use a smooth island matching the raised module.
        """
        binary = (mask > 32).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            return
        top_band = int(height * 0.55)
        count, labels, stats, cents = cv2.connectedComponentsWithStats(
            binary, 8
        )
        cam_label = -1
        cam_score = -1.0
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            cy = float(cents[label][1])
            if cy > top_band or area < 200:
                continue
            # Side capsules are thin — skip them.
            if w <= max(12, int(width * 0.10)) and h >= w * 1.6:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            score = float(area) * (1.4 if aspect <= 1.65 else 1.0)
            if score > cam_score:
                cam_score = score
                cam_label = label
        if cam_label < 0:
            return
        # Two or more compact upper circles = discrete lenses on a flat back.
        # Absorbing them into one AABB is the rectangular hole on S-style shots.
        circular_n = 0
        for label in range(1, count):
            x, y, bw, bh, area = stats[label]
            if float(cents[label][1]) > top_band or area < 80:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            fill = float(area) / max(float(bw * bh), 1.0)
            if aspect <= 1.35 and fill >= 0.70:
                circular_n += 1
        if circular_n >= 2:
            return
        x, y, w, h, area = stats[cam_label]
        if only_if_jagged:
            # Already a clean rounded plate — leave it (avoid AABB inflation).
            component = (labels == cam_label).astype(np.uint8) * 255
            peri = float(
                cv2.arcLength(
                    max(
                        cv2.findContours(
                            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )[0],
                        key=cv2.contourArea,
                    ),
                    True,
                )
            )
            circularity = (4.0 * np.pi * float(area)) / max(peri * peri, 1e-3)
            rect_fill = float(area) / max(float(w * h), 1.0)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect <= 1.28 and rect_fill >= 0.82 and circularity >= 0.72:
                return
        # Absorb nearby upper satellites into the island AABB.
        x1, y1, x2, y2 = x, y, x + w, y + h
        cx0 = float(cents[cam_label][0])
        cy0 = float(cents[cam_label][1])
        reach = 0.55 * max(w, h)
        for label in range(1, count):
            if label == cam_label:
                continue
            sx, sy, sw, sh, area = stats[label]
            if float(cents[label][1]) > top_band:
                continue
            if area > stats[cam_label][4] * 0.65:
                continue
            if sw <= max(12, int(width * 0.10)) and sh >= sw * 1.6:
                continue  # side button
            cx, cy = float(cents[label][0]), float(cents[label][1])
            if ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 > reach * 1.4:
                continue
            x1 = min(x1, sx)
            y1 = min(y1, sy)
            x2 = max(x2, sx + sw)
            y2 = max(y2, sy + sh)
            mask[labels == label] = 0
        mask[labels == cam_label] = 0
        pad = max(1, int(min(width, height) * 0.003))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width - 1, x2 + pad)
        y2 = min(height - 1, y2 + pad)
        # Prefer near-square / stadium plates when aspect is camera-like.
        bw, bh = x2 - x1, y2 - y1
        if bw > 0 and bh > 0 and max(bw, bh) / max(min(bw, bh), 1) < 2.15:
            corner = max(3, int(min(bw, bh) * 0.14))
        else:
            corner = max(3, int(min(bw, bh) * 0.22))
        HardwareRegionDetector._rounded_rectangle(
            mask, x1, y1, x2, y2, corner, expand_px=0.5
        )

    @staticmethod
    def _refine_camera_to_dark_plate(
        phone_bgr: np.ndarray,
        mask: np.ndarray,
        outer_quad: np.ndarray,
    ) -> None:
        """
        Shrink an oversized camera hole onto the photo's raised module plate.

        Model-agnostic: edge density + local contrast inside the current hole
        (dark or light plates, any corner). Oversized holes leave a body-colour
        ring — the classic fake-mockup tell.
        """
        del outer_quad  # reserved for perspective-aware refine later
        if phone_bgr is None or mask is None:
            return
        h, w = mask.shape[:2]
        binary = (mask > 32).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            return
        top_band = int(h * 0.58)
        count, labels, stats, cents = cv2.connectedComponentsWithStats(
            binary, 8
        )
        cam_label = -1
        cam_score = -1.0
        for label in range(1, count):
            x, y, bw, bh, area = stats[label]
            cy = float(cents[label][1])
            if cy > top_band:
                continue
            if bw <= max(12, int(w * 0.10)) and bh >= bw * 1.6:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            score = float(area) * (1.35 if aspect <= 1.75 else 1.0)
            score *= 1.0 + 0.2 * (1.0 - cy / max(top_band, 1))
            if score > cam_score:
                cam_score = score
                cam_label = label
        if cam_label < 0:
            return
        # Discrete lenses on a flat back must not be replaced by one plate.
        circular_n = 0
        for label in range(1, count):
            x0, y0, cw, ch, area = stats[label]
            if float(cents[label][1]) > top_band or area < 80:
                continue
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            fill = float(area) / max(float(cw * ch), 1.0)
            if aspect <= 1.35 and fill >= 0.70:
                circular_n += 1
        if circular_n >= 2:
            return
        x, y, bw, bh, _ = stats[cam_label]
        # Search slightly inside the current hole for the dark plate.
        pad = max(2, int(min(bw, bh) * 0.04))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        roi = phone_bgr[y1:y2, x1:x2]
        if roi.size < 64:
            return
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 36, 110)
        edges = cv2.dilate(
            edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best = None
        best_score = 0.0
        min_side = max(24, int(min(bw, bh) * 0.50))
        max_aspect = 2.15
        for contour in contours:
            sx, sy, sw, sh = cv2.boundingRect(contour)
            if sw < min_side or sh < min_side:
                continue
            # Accept any real shrink (either axis may tighten).
            if sw * sh > bw * bh * 0.98:
                continue
            if sw > bw * 0.995 and sh > bh * 0.995:
                continue
            if sw >= bw * 0.99 and sh >= bh * 0.96:
                continue
            aspect = max(sw, sh) / max(min(sw, sh), 1)
            if aspect > max_aspect:
                continue
            # Edge contours are hollow — score by bbox + edge density.
            patch = edges[sy : sy + sh, sx : sx + sw]
            dens = float(np.count_nonzero(patch)) / max(float(sw * sh), 1.0)
            if dens < 0.02:
                continue
            cx = sx + sw * 0.5
            cy = sy + sh * 0.5
            dist = ((cx - bw * 0.5) ** 2 + (cy - bh * 0.5) ** 2) ** 0.5
            score = (sw * sh) * (1.45 - min(aspect, 1.45)) * (
                0.45 + 2.5 * min(dens, 0.2)
            ) / (1.0 + dist / max(min_side, 1))
            score *= 1.0 + 0.40 * max(0.0, (bw - sw) / max(bw, 1))
            score *= 1.0 + 0.40 * max(0.0, (bh - sh) / max(bh, 1))
            if score > best_score:
                best_score = score
                best = (sx, sy, sw, sh)
        if best is None:
            # Contrast vs local cover — dark OR bright modules.
            cover_ref = float(np.percentile(L, 55))
            contrast = (np.abs(L - cover_ref) >= 5.0).astype(np.uint8) * 255
            contrast = cv2.morphologyEx(
                contrast,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=2,
            )
            dcnts, _ = cv2.findContours(
                contrast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in dcnts:
                area = float(cv2.contourArea(contour))
                sx, sy, sw, sh = cv2.boundingRect(contour)
                if sw * sh > bw * bh * 0.98 or sw < min_side or sh < min_side:
                    continue
                aspect = max(sw, sh) / max(min(sw, sh), 1)
                if aspect > max_aspect:
                    continue
                score = area / max(aspect, 1.0)
                score *= 1.0 + 0.45 * max(
                    0.0, ((bw * bh) - (sw * sh)) / max(float(bw * bh), 1.0)
                )
                if score > best_score:
                    best_score = score
                    best = (sx, sy, sw, sh)
        if best is None:
            return
        sx, sy, sw, sh = best
        lip = max(1, int(min(sw, sh) * 0.012))
        nx1 = max(0, x1 + sx - lip)
        ny1 = max(0, y1 + sy - lip)
        nx2 = min(w - 1, x1 + sx + sw + lip)
        ny2 = min(h - 1, y1 + sy + sh + lip)
        if (nx2 - nx1) < 20 or (ny2 - ny1) < 20:
            return
        if (nx2 - nx1) * (ny2 - ny1) > bw * bh * 0.985:
            return
        mask[labels == cam_label] = 0
        corner = max(3, int(min(nx2 - nx1, ny2 - ny1) * 0.13))
        # No expand_px — grey body rings come from oversizing the hole.
        HardwareRegionDetector._rounded_rectangle(
            mask, nx1, ny1, nx2, ny2, corner, expand_px=0.4
        )

    @staticmethod
    def _circle_looks_like_hardware(
        gray: np.ndarray, x: int, y: int, radius: int
    ) -> bool:
        """True when a candidate circle is darker/brighter than nearby cover."""
        height, width = gray.shape
        if radius < 2:
            return False
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        disk = xx * xx + yy * yy <= radius * radius
        ring = (
            (xx * xx + yy * yy <= (radius * 1.7) ** 2)
            & (xx * xx + yy * yy > (radius * 1.15) ** 2)
        )

        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        dy0, dx0 = y0 - (y - radius), x0 - (x - radius)
        local_disk = disk[
            dy0 : dy0 + patch.shape[0], dx0 : dx0 + patch.shape[1]
        ]
        if np.count_nonzero(local_disk) < 8:
            return False
        interior = float(np.median(patch[local_disk]))

        ry = max(radius + 2, int(radius * 1.7))
        y0r, y1r = max(0, y - ry), min(height, y + ry + 1)
        x0r, x1r = max(0, x - ry), min(width, x + ry + 1)
        surround = gray[y0r:y1r, x0r:x1r].astype(np.float32)
        yy, xx = np.ogrid[y0r - y : y1r - y, x0r - x : x1r - x]
        local_ring = (xx * xx + yy * yy <= (radius * 1.7) ** 2) & (
            xx * xx + yy * yy > (radius * 1.15) ** 2
        )
        if np.count_nonzero(local_ring) < 8:
            return True
        exterior = float(np.median(surround[local_ring]))
        return abs(interior - exterior) >= 9.0

    @staticmethod
    def _detect_flash_spots(
        gray: np.ndarray,
        mask: np.ndarray,
        scores: List[float],
        width: int,
        top_height: int,
        lenses: List[Tuple[int, int, int]],
    ) -> List[Tuple[int, int, int]]:
        """
        Detect flash, mic, and sensor disks near already-found lenses.

        Covers bright flash LEDs and darker microphone/sensor pits that Hough
        often misses at phone-photo resolution.
        """
        if not lenses:
            return []

        found: List[Tuple[int, int, int]] = []
        lx = float(np.median([c[0] for c in lenses]))
        ly = float(np.median([c[1] for c in lenses]))
        mean_r = float(np.median([c[2] for c in lenses]))
        search = max(12, int(mean_r * 4.8))

        x0 = int(np.clip(lx - search, 0, width - 1))
        x1 = int(np.clip(lx + search, 0, width))
        y0 = int(np.clip(ly - search * 1.6, 0, top_height - 1))
        y1 = int(np.clip(ly + search * 1.6, 0, top_height))
        roi = gray[y0:y1, x0:x1]
        if roi.size < 16:
            return []

        blurred = cv2.GaussianBlur(roi, (0, 0), 0.9)
        max_flash_r = max(3, int(mean_r * 0.70))
        min_flash_r = 2

        def _accept(fx: int, fy: int, radius: float, score: float) -> None:
            radius = float(radius)
            if radius < min_flash_r or radius > max_flash_r:
                return
            if any(
                (fx - lx_) ** 2 + (fy - ly_) ** 2 < (lr * 0.80) ** 2
                for lx_, ly_, lr in lenses
            ):
                return
            if any(
                (fx - ox) ** 2 + (fy - oy) ** 2
                < (max(radius, or_) * 0.75) ** 2
                for ox, oy, or_ in lenses + found
            ):
                return
            # Prefer openings beside the lens column, not far below the island.
            dist = ((fx - lx) ** 2 + (fy - ly) ** 2) ** 0.5
            if dist > mean_r * 4.2:
                return
            ir = max(2, int(round(radius + 1.0)))
            found.append((fx, fy, ir))
            cv2.circle(mask, (fx, fy), ir, 255, -1, cv2.LINE_AA)
            scores.append(score)

        # Bright flash / LED peaks.
        bright_thr = max(
            float(np.percentile(blurred, 88)),
            float(np.median(blurred)) + 22.0,
        )
        bright = (blurred >= bright_thr).astype(np.uint8) * 255
        bright = cv2.morphologyEx(
            bright, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        for contour in cv2.findContours(
            bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]:
            area = cv2.contourArea(contour)
            if area < 3 or area > np.pi * (max_flash_r + 1) ** 2:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            circularity = (
                (4.0 * np.pi * area) / max(cv2.arcLength(contour, True) ** 2, 1e-3)
            )
            if circularity < 0.45 and radius > 4:
                continue
            _accept(
                int(round(cx + x0)), int(round(cy + y0)), radius, 0.92
            )

        # Dark mic / sensor pits beside the cluster.
        dark_thr = min(
            float(np.percentile(blurred, 10)),
            float(np.median(blurred)) - 20.0,
        )
        dark = (blurred <= dark_thr).astype(np.uint8) * 255
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        for contour in cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]:
            area = cv2.contourArea(contour)
            if area < 3 or area > np.pi * (max_flash_r + 1) ** 2:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius > mean_r * 0.55:
                continue
            _accept(
                int(round(cx + x0)), int(round(cy + y0)), radius, 0.88
            )

        # Sensitive Hough inside the camera ROI for leftover disks.
        roi_hough = cv2.GaussianBlur(roi, (0, 0), 0.7)
        hough = cv2.HoughCircles(
            roi_hough,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(4, int(mean_r * 0.45)),
            param1=70,
            param2=11,
            minRadius=min_flash_r,
            maxRadius=max_flash_r,
        )
        if hough is not None:
            for x, y, radius in np.round(hough[0]).astype(int):
                fx, fy = int(x + x0), int(y + y0)
                if HardwareRegionDetector._circle_looks_like_hardware(
                    gray, fx, fy, max(2, int(radius))
                ) or float(
                    gray[
                        max(0, fy - 1) : fy + 2, max(0, fx - 1) : fx + 2
                    ].mean()
                ) >= bright_thr * 0.85:
                    _accept(fx, fy, float(radius), 0.9)

        return found

    @staticmethod
    def _detect_camera_modules(
        gray: np.ndarray,
        mask: np.ndarray,
        scores: List[float],
        width: int,
        top_height: int,
    ) -> None:
        """Detect dark/bright camera islands that may contain multiple holes."""
        smoothed = cv2.GaussianBlur(gray, (0, 0), max(2.0, width * 0.012))
        local = cv2.GaussianBlur(
            gray, (0, 0), max(8.0, width * 0.07)
        )
        difference = cv2.absdiff(smoothed, local)
        _, contrast = cv2.threshold(
            difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contrast = cv2.morphologyEx(
            contrast,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    max(3, int(width * 0.028) | 1),
                    max(3, int(width * 0.028) | 1),
                ),
            ),
            iterations=2,
        )

        contours, _ = cv2.findContours(
            contrast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        total = width * top_height
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < total * 0.0012 or area > total * 0.14:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w / 2.0
            near_side = center_x < width * 0.42 or center_x > width * 0.58
            compact = max(w, h) / max(min(w, h), 1) < 2.6
            if not near_side or not compact:
                continue

            hull = cv2.convexHull(contour)
            padding = max(1, int(min(width, top_height) * 0.003))
            component = np.zeros_like(mask)
            x, y, w, h = cv2.boundingRect(hull)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width - 1, x + w + padding)
            y2 = min(top_height - 1, y + h + padding)
            corner = max(3, int(min(max(w, 1), max(h, 1)) * 0.22))
            HardwareRegionDetector._rounded_rectangle(
                component, x1, y1, x2, y2, corner,
            )
            mask[:] = cv2.max(mask, component)
            scores.append(0.65)

    @staticmethod
    def _detect_side_hardware(
        gray: np.ndarray,
        mask: np.ndarray,
        scores: List[float],
        width: int,
        height: int,
    ) -> None:
        """
        Exclude volume / power buttons, mute switches, and side openings.

        Print must never wrap onto these — the original phone hardware should
        show through for a realistic mockup.
        """
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        band = max(12, int(width * 0.14))
        sobel = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        # Vertical structure (buttons) also shows in y-gradient of the strip.
        sobel_y = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))

        for x0 in (0, max(0, width - band)):
            x1 = min(width, x0 + band)
            strip = sobel[:, x0:x1]
            strip_y = sobel_y[:, x0:x1]
            energy = cv2.normalize(
                strip + 0.35 * strip_y, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            # Local bright ridges vs quiet cover plastic.
            thr = max(
                22.0,
                float(np.percentile(energy, 74)),
                float(np.median(energy)) + 14.0,
            )
            binary = (energy >= thr).astype(np.uint8) * 255
            # Prefer tall thin button capsules.
            v_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(3, band // 5), max(9, int(height * 0.028))),
            )
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, v_kernel, iterations=2
            )
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
            )

            # Intensity profile peaks (raised button plastic on clear cases).
            profile = blur[:, x0:x1].astype(np.float32).mean(axis=1)
            smooth = cv2.GaussianBlur(
                profile.reshape(-1, 1), (0, 0), max(2.0, height * 0.004)
            ).ravel()
            baseline = float(np.median(smooth))
            deviant = (np.abs(smooth - baseline) >= 5.0).astype(np.uint8) * 255
            deviant = cv2.morphologyEx(
                deviant.reshape(-1, 1), cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, (1, max(7, int(height * 0.02)))
                ),
            ).ravel()
            profile_mask = np.zeros_like(binary)
            profile_mask[deviant > 0, :] = 255
            binary = cv2.max(binary, profile_mask)

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Volume / power: tall. Mute / SIM: shorter pill.
                # Side fingerprint sensors: mid-height ovals (~1:1–2:1).
                elongated = h / max(w, 1) >= 1.6
                compact_switch = (
                    height * 0.012 <= h <= height * 0.08
                    and w <= band * 0.85
                )
                side_fingerprint = (
                    height * 0.04 <= h <= height * 0.18
                    and w <= band * 0.95
                    and 0.85 <= (h / max(w, 1)) <= 2.45
                )
                plausible = (
                    width * 0.004 <= w <= band
                    and height * 0.015 <= h <= height * 0.28
                    and y > height * 0.08
                    and y + h < height * 0.92
                )
                if not plausible or not (
                    elongated or compact_switch or side_fingerprint
                ):
                    continue

                pad_x = max(2, int(width * 0.008))
                pad_y = max(2, int(height * 0.004))
                bx1 = max(0, x0 + x - pad_x)
                by1 = max(0, y - pad_y)
                bx2 = min(width - 1, x0 + x + w + pad_x)
                by2 = min(height - 1, y + h + pad_y)
                # Capsule so exclusions follow real button silhouettes.
                corner = max(2, min((bx2 - bx1), (by2 - by1)) // 2)
                HardwareRegionDetector._rounded_rectangle(
                    mask, bx1, by1, bx2, by2, corner
                )
                scores.append(0.82 if elongated else 0.7)

        HardwareRegionDetector._detect_bottom_speaker(
            gray, mask, scores, width, height
        )

    @staticmethod
    def _detect_bottom_speaker(
        gray: np.ndarray,
        mask: np.ndarray,
        scores: List[float],
        width: int,
        height: int,
    ) -> None:
        """Exclude bottom speaker / mic grille openings from the print."""
        band = max(10, int(height * 0.10))
        y0 = max(0, height - band)
        roi = gray[y0:height, :]
        if roi.size < 16:
            return
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 110)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(7, int(width * 0.04)), 3)
            ),
            iterations=2,
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            wide = w / max(h, 1) >= 2.0
            plausible = (
                width * 0.04 <= w <= width * 0.55
                and 2 <= h <= band * 0.55
            )
            if not wide or not plausible:
                continue
            pad = max(2, int(height * 0.004))
            HardwareRegionDetector._rounded_rectangle(
                mask,
                max(0, x - pad),
                max(0, y0 + y - pad),
                min(width - 1, x + w + pad),
                min(height - 1, y0 + y + h + pad),
                max(2, h // 2 + 1),
            )
            scores.append(0.68)

    @staticmethod
    def _detect_side_hardware_fullres(
        phone_bgr: np.ndarray,
        mask: np.ndarray,
        outer_quad: np.ndarray,
        *,
        relaxed: bool = False,
    ) -> None:
        """
        Find side buttons on the original photo along the cover's left/right.

        Clear-case mockups show physical button ridges that must stay original
        (no artwork wrap). Sample on the rim and slightly inward — wrap faces
        put volume/power/mute/fingerprint on the side wall, not only outside.

        ``relaxed=True`` lowers thresholds for flush / dark phones where the
        first pass often misses left volume + side FP.
        """
        gray = cv2.cvtColor(phone_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        sobel_y = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
        h, w = gray.shape
        quad = order_points(outer_quad)
        left_a, left_b = quad[0], quad[3]
        right_a, right_b = quad[1], quad[2]
        band = max(14, int(min(w, h) * (0.072 if relaxed else 0.060)))

        # Dual inward depths catch flush buttons that sit slightly inside.
        depth_fracs = (0.28, 0.55, 0.78) if relaxed else (0.28, 0.55)

        for a, b in (
            (left_a, left_b),
            (right_a, right_b),
        ):
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length < 20:
                continue
            tangent = edge / length
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            centroid = quad.mean(axis=0)
            mid = (a + b) * 0.5
            if np.dot(normal, mid - centroid) < 0:
                normal = -normal

            samples = max(64, int(length / 2.0))
            profile = []
            edge_energy = []
            coords = []
            for i in range(samples):
                t = i / max(samples - 1, 1)
                p = a * (1.0 - t) + b * t
                vals = []
                e_vals = []
                xs_c, ys_c = [], []
                for df in depth_fracs:
                    q = p - normal * (band * float(df))
                    x = int(np.clip(round(q[0]), 0, w - 1))
                    y = int(np.clip(round(q[1]), 0, h - 1))
                    ox = int(
                        np.clip(round((p + normal * (band * 0.22))[0]), 0, w - 1)
                    )
                    oy = int(
                        np.clip(round((p + normal * (band * 0.22))[1]), 0, h - 1)
                    )
                    ix = int(
                        np.clip(round((p - normal * (band * 0.70))[0]), 0, w - 1)
                    )
                    iy = int(
                        np.clip(round((p - normal * (band * 0.70))[1]), 0, h - 1)
                    )
                    x0, x1 = min(ox, ix, x), max(ox, ix, x) + 1
                    y0 = int(np.clip(y - 3, 0, h - 1))
                    y1 = int(np.clip(y + 4, 0, h))
                    x0 = int(np.clip(x0, 0, w - 1))
                    x1 = int(np.clip(x1, 0, w))
                    patch = gray[y0:y1, x0:x1]
                    e_patch = (sobel + 0.45 * sobel_y)[y0:y1, x0:x1]
                    if patch.size:
                        vals.append(float(patch.mean()))
                        e_vals.append(float(e_patch.mean()))
                        xs_c.append(x)
                        ys_c.append(y)
                profile.append(float(np.mean(vals)) if vals else 0.0)
                edge_energy.append(float(np.mean(e_vals)) if e_vals else 0.0)
                if xs_c:
                    coords.append(
                        (int(round(np.mean(xs_c))), int(round(np.mean(ys_c))))
                    )
                else:
                    q = p - normal * (band * 0.35)
                    coords.append(
                        (
                            int(np.clip(round(q[0]), 0, w - 1)),
                            int(np.clip(round(q[1]), 0, h - 1)),
                        )
                    )

            arr = np.asarray(profile, dtype=np.float32)
            energy = np.asarray(edge_energy, dtype=np.float32)
            smooth = cv2.GaussianBlur(arr.reshape(-1, 1), (0, 0), 1.4).ravel()
            e_smooth = cv2.GaussianBlur(
                energy.reshape(-1, 1), (0, 0), 1.2
            ).ravel()
            baseline = float(np.median(smooth))
            trend = cv2.GaussianBlur(
                smooth.reshape(-1, 1), (0, 0), max(4.0, samples * 0.04)
            ).ravel()
            prominence = np.abs(smooth - trend)

            # Adaptive per-edge thresholds (fixed floors miss dark flush bezels).
            inten_floor = 2.4 if relaxed else 3.2
            prom_floor = 1.05 if relaxed else 1.45
            inten_thr = max(
                inten_floor,
                float(np.percentile(np.abs(smooth - baseline), 62)) * 0.85,
            )
            prom_thr = max(
                prom_floor,
                float(np.percentile(prominence, 70)) * 0.72,
            )
            intensity_hit = np.abs(smooth - baseline) >= inten_thr
            e_base = float(np.median(e_smooth))
            e_thr = max(
                e_base * (1.22 if relaxed else 1.32),
                e_base + (5.0 if relaxed else 7.0),
                float(np.percentile(e_smooth, 72)),
            )
            edge_hit = e_smooth >= e_thr
            # Edge-only allowed when the ridge is locally strong (flush buttons).
            active = intensity_hit & (
                (prominence >= prom_thr)
                | (edge_hit & (prominence >= prom_thr * 0.55))
            )
            if relaxed:
                active = active | (
                    edge_hit & (prominence >= prom_thr * 0.35)
                )

            i = 0
            while i < len(active):
                if not active[i]:
                    i += 1
                    continue
                j = i
                while j < len(active) and active[j]:
                    j += 1
                span = j - i
                min_span = max(2, int(samples * (0.015 if relaxed else 0.02)))
                max_span = int(samples * (0.34 if relaxed else 0.30))
                if span >= min_span and span <= max_span:
                    t_mid = 0.5 * (i + j - 1) / max(samples - 1, 1)
                    # Keep upper-mid power / side fingerprint (~0.08–0.25).
                    if t_mid < 0.04 or t_mid > 0.96:
                        i = j
                        continue
                    pts = coords[i:j]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    pad_out = max(2, band // 5)
                    pad_in = max(5, int(band * (0.42 if relaxed else 0.36)))
                    pad_y = max(4, band // 5)
                    mid_x = int(round(0.5 * (min(xs) + max(xs))))
                    mid_y = int(round(0.5 * (min(ys) + max(ys))))
                    if mid_y < h * 0.06 or mid_y > h * 0.94:
                        i = j
                        continue
                    c = np.array([mid_x, mid_y], dtype=np.float32)
                    inward = c - normal * pad_in
                    outward = c + normal * pad_out
                    x1 = int(
                        np.clip(
                            min(inward[0], outward[0], min(xs)) - 2, 0, w - 1
                        )
                    )
                    x2 = int(
                        np.clip(
                            max(inward[0], outward[0], max(xs)) + 2, 0, w - 1
                        )
                    )
                    y1 = int(np.clip(min(ys) - pad_y, 0, h - 1))
                    y2 = int(np.clip(max(ys) + pad_y, 0, h - 1))
                    max_w = band * (3.4 if relaxed else 3.0)
                    if (y2 - y1) >= 7 and (x2 - x1) <= max_w:
                        corner = max(2, min(x2 - x1, y2 - y1) // 2)
                        HardwareRegionDetector._rounded_rectangle(
                            mask, x1, y1, x2, y2, corner
                        )
                i = j

    @staticmethod
    def detect_verified_side_hardware(
        phone_bgr: np.ndarray,
        outer_quad: np.ndarray,
        *,
        phone_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Dynamic side volume / power / mute / fingerprint openings.

        Photo-driven only: ridge profiles along each live bezel + optional
        silhouette bumps. No fixed height seeds (those invented ghost wraps on
        empty sides). Corner glare and bottom contact-shadow are rejected.
        """
        phone = to_bgr(phone_bgr)
        h, w = phone.shape[:2]
        quad = order_points(np.asarray(outer_quad, dtype=np.float32))
        out = np.zeros((h, w), dtype=np.uint8)

        strict = np.zeros((h, w), dtype=np.uint8)
        HardwareRegionDetector._detect_side_hardware_fullres(
            phone, strict, quad, relaxed=False
        )
        # Prefer strict mid-bezel hits. Fat relaxed merges climb into the
        # camera island and later collapse with the camera cutout.
        combined = strict.copy()
        relaxed = np.zeros((h, w), dtype=np.uint8)
        HardwareRegionDetector._detect_side_hardware_fullres(
            phone, relaxed, quad, relaxed=True
        )
        if np.count_nonzero(relaxed):
            gray = cv2.cvtColor(phone, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(
                gray
            )
            sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            y_min = float(quad[:, 1].min())
            y_max = float(quad[:, 1].max())
            height = max(y_max - y_min, 1.0)
            max_bh = height * 0.38
            near_strict = cv2.dilate(
                strict,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
                iterations=1,
            )
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                (relaxed > 0).astype(np.uint8), connectivity=8
            )
            for label in range(1, num):
                area = int(stats[label, cv2.CC_STAT_AREA])
                bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                bw = int(stats[label, cv2.CC_STAT_WIDTH])
                if area < 40 or bh > max_bh:
                    continue
                comp = labels == label
                touches = np.count_nonzero(near_strict[comp]) > 0
                e_mean = float(sobel[comp].mean()) if np.any(comp) else 0.0
                contrast = float(gray[comp].std()) if np.any(comp) else 0.0
                # Compact flush FP / power only — never tall wall slabs.
                if touches or (e_mean >= 20.0 and contrast >= 7.0 and bw <= bh * 1.35):
                    combined[comp] = 255

        if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
            sil = phone_mask
            if sil.shape[:2] != (h, w):
                sil = cv2.resize(
                    sil, (w, h), interpolation=cv2.INTER_NEAREST
                )
            bumps = HardwareRegionDetector.detect_buttons_from_silhouette(
                sil, quad
            )
            if np.count_nonzero(bumps):
                combined = cv2.max(combined, bumps)

        return HardwareRegionDetector._filter_verified_side_blobs(
            combined, phone, quad, phone_mask=phone_mask
        )

    @staticmethod
    def _filter_verified_side_blobs(
        mask: np.ndarray,
        phone_bgr: np.ndarray,
        outer_quad: np.ndarray,
        *,
        phone_mask: Optional[np.ndarray] = None,
        max_per_side: int = 4,
    ) -> np.ndarray:
        """Keep compact mid-bezel pills; drop corners / face / studio spill."""
        binary = (mask > 0).astype(np.uint8) * 255
        out = np.zeros_like(binary)
        if np.count_nonzero(binary) < 24:
            return out
        h, w = binary.shape[:2]
        phone = to_bgr(phone_bgr)
        if phone.shape[:2] != (h, w):
            phone = cv2.resize(phone, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(phone, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        quad = order_points(np.asarray(outer_quad, dtype=np.float32))
        x_min = float(quad[:, 0].min())
        x_max = float(quad[:, 0].max())
        y_min = float(quad[:, 1].min())
        y_max = float(quad[:, 1].max())
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)
        mid_x = 0.5 * (x_min + x_max)
        side_band = width * 0.16
        gate = None
        if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
            gate = phone_mask
            if gate.shape[:2] != (h, w):
                gate = cv2.resize(
                    gate, (w, h), interpolation=cv2.INTER_NEAREST
                )
            gate = cv2.dilate(
                (gate > 127).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        scored_l: List[Tuple[float, int]] = []
        scored_r: List[Tuple[float, int]] = []
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 28 or area > int(h * w * 0.04):
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            t = (cy - y_min) / height
            # Corners are glare / contact shadow — never side hardware.
            if t < 0.09 or t > 0.88:
                continue
            near_side = (cx - x_min) <= side_band or (x_max - cx) <= side_band
            if not near_side:
                continue
            # Stay on the thin bezel — face/camera islands sit further inward.
            edge_dist = min(cx - x_min, x_max - cx)
            if edge_dist > width * 0.085:
                continue
            if bw > width * 0.12 or bh > height * 0.38:
                continue
            # Camera module lives on the upper face, not the outer bezel.
            if t < 0.20 and edge_dist > width * 0.06:
                continue
            # Long thin wall slabs are ghosts, not buttons.
            if bh > height * 0.38 and bw < width * 0.045:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1e-3)
            if aspect < 1.05 and max(bw, bh) > width * 0.12:
                continue
            comp = labels == label
            if gate is not None:
                overlap = float(np.count_nonzero(comp & (gate > 0)))
                if overlap < area * 0.35:
                    continue
            contrast = float(gray[comp].std())
            edge_mean = float(sobel[comp].mean())
            if contrast < 3.0 and edge_mean < 10.0:
                continue
            score = (
                float(area)
                + 3.0 * float(bh)
                + 1.5 * edge_mean
                + 2.0 * contrast
            )
            # Prefer mid-upper bezel (volume / power / side FP cluster).
            score += 40.0 * (1.0 - abs(t - 0.28))
            if cx < mid_x:
                scored_l.append((score, label))
            else:
                scored_r.append((score, label))

        for scored in (scored_l, scored_r):
            scored.sort(reverse=True)
            for _, label in scored[: max(1, int(max_per_side))]:
                # Paint a tight capsule from the blob bounds (not the fat pad).
                x = int(stats[label, cv2.CC_STAT_LEFT])
                y = int(stats[label, cv2.CC_STAT_TOP])
                bw = int(stats[label, cv2.CC_STAT_WIDTH])
                bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                cx = x + bw * 0.5
                # Slight shrink so wrap wall remains around the opening.
                pad = max(1, int(round(min(bw, bh) * 0.08)))
                x1 = max(0, x + pad)
                y1 = max(0, y + pad)
                x2 = min(w - 1, x + bw - pad)
                y2 = min(h - 1, y + bh - pad)
                # Snap to the outer bezel so volume/FP cut on the rim, not the face.
                box_w = max(4, x2 - x1)
                if cx < mid_x:
                    x1 = int(np.clip(round(x_min + width * 0.004), 0, w - 2))
                    x2 = int(np.clip(x1 + box_w, x1 + 4, w - 1))
                else:
                    x2 = int(np.clip(round(x_max - width * 0.004), 1, w - 1))
                    x1 = int(np.clip(x2 - box_w, 0, x2 - 4))
                if x2 - x1 < 4 or y2 - y1 < 6:
                    continue
                corner = max(2, min(x2 - x1, y2 - y1) // 2)
                HardwareRegionDetector._rounded_rectangle(
                    out, x1, y1, x2, y2, corner
                )
        return out

    @staticmethod
    def filter_volume_button_mask(
        mask: np.ndarray,
        outer_quad: np.ndarray,
        *,
        allow_compact: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Keep volume-rocker-like side openings (elongated bezel pills).

        Shape is judged relative to the live phone quad — no fixed millimetre
        sizes. Compact power / FP pills are dropped unless ``allow_compact``.
        """
        binary = (mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 24:
            return None
        h, w = binary.shape[:2]
        quad = order_points(np.asarray(outer_quad, dtype=np.float32))
        x_min = float(quad[:, 0].min())
        x_max = float(quad[:, 0].max())
        y_min = float(quad[:, 1].min())
        y_max = float(quad[:, 1].max())
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)
        out = np.zeros((h, w), dtype=np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        kept = 0
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 20:
                continue
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            cy = float(
                stats[label, cv2.CC_STAT_TOP]
                + stats[label, cv2.CC_STAT_HEIGHT] * 0.5
            )
            t = (cy - y_min) / height
            if t < 0.08 or t > 0.90:
                continue
            # Volume rockers are elongated along the bezel. Thresholds are
            # fractions of the live phone height — no fixed pixel sizes.
            long_edge = max(bw, bh)
            short_edge = max(min(bw, bh), 1)
            aspect = long_edge / float(short_edge)
            span_frac = long_edge / height
            is_volume = aspect >= 1.25 and span_frac >= 0.045
            # Compact FP / power pills — skip unless caller allows mid-bezel
            # ridges that are only mildly elongated.
            is_compact_fp = aspect < 1.20 and span_frac < 0.055
            if is_volume:
                out[labels == label] = 255
                kept += 1
            elif allow_compact and not is_compact_fp and aspect >= 1.18:
                out[labels == label] = 255
                kept += 1
        if kept == 0 or np.count_nonzero(out) < 24:
            return None
        return out

    @staticmethod
    def detect_buttons_from_silhouette(
        phone_mask: np.ndarray,
        outer_quad: np.ndarray,
    ) -> np.ndarray:
        """
        Find volume / power bumps from outward protrusions on the phone outline.

        Works on any phone photo where side buttons create a visible edge bump.
        """
        out = np.zeros_like(phone_mask, dtype=np.uint8)
        binary = (phone_mask > 0).astype(np.uint8)
        if np.count_nonzero(binary) < 64:
            return out
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return out
        outer = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32
        )
        if outer.shape[0] < 24:
            return out
        h, w = binary.shape[:2]
        quad = order_points(np.asarray(outer_quad, dtype=np.float32))
        centroid = quad.mean(axis=0)
        band = max(10.0, float(min(w, h)) * 0.05)

        for a, b in ((quad[0], quad[3]), (quad[1], quad[2])):
            edge = b.astype(np.float32) - a.astype(np.float32)
            length = float(np.linalg.norm(edge))
            if length < 30:
                continue
            tangent = edge / length
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            mid = (a + b) * 0.5
            if float(np.dot(normal, mid - centroid)) < 0:
                normal = -normal

            samples: list = []
            for pt in outer:
                rel = pt - a
                t = float(np.dot(rel, tangent) / max(length, 1e-6))
                if t < 0.06 or t > 0.94:
                    continue
                proj = a + tangent * (t * length)
                outward = float(np.dot(pt - proj, normal))
                if outward < -1.0:
                    continue
                dist_edge = float(np.linalg.norm(pt - proj))
                if dist_edge > band * 1.8:
                    continue
                samples.append((t, outward, pt))

            if len(samples) < 8:
                continue
            samples.sort(key=lambda s: s[0])
            ts = np.array([s[0] for s in samples], dtype=np.float32)
            outs = np.array([s[1] for s in samples], dtype=np.float32)
            smooth = cv2.GaussianBlur(
                outs.reshape(-1, 1), (0, 0), max(1.2, len(outs) * 0.06)
            ).ravel()
            baseline = float(np.median(smooth))
            trend = cv2.GaussianBlur(
                smooth.reshape(-1, 1), (0, 0), max(3.0, len(smooth) * 0.12)
            ).ravel()
            prominence = smooth - trend
            active = prominence >= max(0.55, baseline * 0.08 + 0.35)
            i = 0
            while i < len(active):
                if not active[i]:
                    i += 1
                    continue
                j = i
                while j < len(active) and active[j]:
                    j += 1
                span = j - i
                if span >= 2 and span <= max(6, int(len(active) * 0.35)):
                    seg = samples[i:j]
                    xs = [float(p[2][0]) for p in seg]
                    ys = [float(p[2][1]) for p in seg]
                    pad_out = max(4, int(band * 0.35))
                    # Keep punch tight to the button ridge — large inward pad
                    # erased side-wall wrap and hid volume/power shape.
                    pad_in = max(6, int(band * 0.55))
                    pad_y = max(4, int(band * 0.28))
                    mid_x = 0.5 * (min(xs) + max(xs))
                    mid_y = 0.5 * (min(ys) + max(ys))
                    c = np.array([mid_x, mid_y], dtype=np.float32)
                    inward = c - normal * pad_in
                    outward_pt = c + normal * pad_out
                    x1 = int(
                        np.clip(
                            min(inward[0], outward_pt[0], min(xs)) - 2, 0, w - 1
                        )
                    )
                    x2 = int(
                        np.clip(
                            max(inward[0], outward_pt[0], max(xs)) + 2, 0, w - 1
                        )
                    )
                    y1 = int(np.clip(min(ys) - pad_y, 0, h - 1))
                    y2 = int(np.clip(max(ys) + pad_y, 0, h - 1))
                    if (y2 - y1) >= 8 and (x2 - x1) <= band * 3.2:
                        corner = max(2, min(x2 - x1, y2 - y1) // 2)
                        HardwareRegionDetector._rounded_rectangle(
                            out, x1, y1, x2, y2, corner
                        )
                i = j
        return out

    @staticmethod
    def _peripheral_only_mask(
        mask: np.ndarray, outer_quad: np.ndarray
    ) -> np.ndarray:
        """
        Keep only side / bottom hardware from a full exclusion mask.

        Drops camera-island blobs so Perfect Finish can merge button/speaker
        openings without reopening or duplicating the camera cutout.
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return np.zeros_like(mask)
        h, w = mask.shape[:2]
        quad = order_points(outer_quad)
        left_a, left_b = quad[0], quad[3]
        right_a, right_b = quad[1], quad[2]
        bottom_a, bottom_b = quad[3], quad[2]
        band = max(10.0, float(min(w, h)) * 0.07)

        def _dist_to_segment(
            point: np.ndarray, a: np.ndarray, b: np.ndarray
        ) -> float:
            ab = b - a
            length2 = float(np.dot(ab, ab))
            if length2 < 1e-6:
                return float(np.linalg.norm(point - a))
            t = float(np.clip(np.dot(point - a, ab) / length2, 0.0, 1.0))
            proj = a + t * ab
            return float(np.linalg.norm(point - proj))

        out = np.zeros_like(mask)
        binary = (mask > 96).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            m = cv2.moments(contour)
            if m["m00"] < 1e-3:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            center = np.array([cx, cy], dtype=np.float32)
            near_side = (
                _dist_to_segment(center, left_a, left_b) <= band
                or _dist_to_segment(center, right_a, right_b) <= band
            )
            near_bottom = _dist_to_segment(center, bottom_a, bottom_b) <= band
            # Camera islands sit upper and inward — reject those.
            upper = cy < (float(quad[0][1] + quad[1][1]) * 0.5 + (h * 0.28))
            inward = (
                _dist_to_segment(center, left_a, left_b) > band * 1.4
                and _dist_to_segment(center, right_a, right_b) > band * 1.4
            )
            if upper and inward and not near_bottom:
                continue
            if near_side or near_bottom:
                cv2.drawContours(out, [contour], -1, 255, -1)
        return out

    @staticmethod
    def _merge_camera_cluster(
        mask: np.ndarray,
        width: int,
        top_height: int,
        circles: Optional[List[Tuple[int, int, int]]] = None,
    ) -> None:
        """
        Join nearby lens/flash openings into one camera-island exclusion.

        Prefer a tight rounded stadium around validated circles over a convex
        hull (hulls fill plain cover between lenses and look jagged).
        """
        circles = circles or []
        if len(circles) >= 2:
            rs = [int(c[2]) for c in circles]
            pad = max(2, int(np.median(rs) * 0.10))
            # Small satellites (flash/mic) need a bit more enclosure than lenses.
            extents = []
            for x, y, radius in circles:
                extra = pad + (2 if radius < np.median(rs) * 0.75 else 0)
                extents.append((x - radius - extra, y - radius - extra,
                                x + radius + extra, y + radius + extra))
            x1 = max(0, min(e[0] for e in extents))
            y1 = max(0, min(e[1] for e in extents))
            x2 = min(width - 1, max(e[2] for e in extents))
            y2 = min(top_height - 1, max(e[3] for e in extents))
            if (x2 - x1) <= width * 0.48 and (y2 - y1) <= top_height * 0.72:
                corner = int(np.clip(np.median(rs) * 0.95, 3, min(x2 - x1, y2 - y1) // 2))
                HardwareRegionDetector._rounded_rectangle(
                    mask, x1, y1, x2, y2, corner
                )
                # Reinforce each lens/flash tightly so AA does not leave gaps.
                for x, y, radius in circles:
                    cv2.circle(
                        mask, (x, y),
                        radius + max(1, int(radius * 0.05)),
                        255, -1, cv2.LINE_AA,
                    )
            return

        top_mask = mask[:top_height]
        contours, _ = cv2.findContours(
            (top_mask > 32).astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) < 2:
            return

        centers = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            centers.append((x + w / 2.0, y + h / 2.0))

        left = [i for i, center in enumerate(centers) if center[0] < width / 2]
        right = [i for i, center in enumerate(centers) if center[0] >= width / 2]
        cluster = left if len(left) >= len(right) else right
        if len(cluster) < 2:
            return

        selected = np.vstack([contours[index] for index in cluster])
        x, y, w, h = cv2.boundingRect(selected)
        if w > width * 0.52 or h > top_height * 0.75:
            return

        pad = max(1, int(width * 0.004))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(width - 1, x + w + pad)
        y2 = min(top_height - 1, y + h + pad)
        corner = max(3, int(min(w, h) * 0.22))
        HardwareRegionDetector._rounded_rectangle(mask, x1, y1, x2, y2, corner)

    @staticmethod
    def _tighten_to_hardware(
        mask: np.ndarray,
        gray: np.ndarray,
        top_height: int,
        circles: Optional[List[Tuple[int, int, int]]] = None,
    ) -> None:
        """
        Shrink camera exclusions onto real hardware pixels only.

        Strictly subtractive: never uncovers a validated lens/flash circle.
        """
        top = mask[:top_height]
        detected = (top > 32).astype(np.uint8)
        if np.count_nonzero(detected) == 0:
            return

        region = gray[:top_height].astype(np.float32)
        outside = region[detected == 0]
        if outside.size < 64:
            return

        cover = float(np.median(outside))
        spread = float(np.median(np.abs(outside - cover)))
        tolerance = max(10.0, spread * 2.0)
        distinct = np.abs(region - cover) > tolerance

        # Always preserve validated circular openings.
        protected = np.zeros_like(detected)
        for x, y, radius in circles or []:
            if y >= top_height:
                continue
            pad = max(1, int(radius * 0.04))
            cv2.circle(
                protected, (x, y), radius + pad, 1, -1, cv2.LINE_AA
            )

        count, labels, _, _ = cv2.connectedComponentsWithStats(detected, 8)
        for label in range(1, count):
            selected = labels == label
            core = selected & (distinct | (protected > 0))
            selected_area = int(np.count_nonzero(selected))
            core_area = int(np.count_nonzero(core))
            if core_area < max(12, int(selected_area * 0.08)):
                continue

            ys, xs = np.nonzero(core)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            corner = max(2, int(min(x2 - x1, y2 - y1) * 0.22))
            kept = np.zeros_like(detected)
            HardwareRegionDetector._rounded_rectangle(
                kept, x1, y1, x2, y2, corner
            )
            # Close tiny gaps between lenses without restoring discarded spill.
            kept = cv2.morphologyEx(
                kept, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            # Keep protected circles even if outside the stadium.
            kept = np.maximum(kept, protected)
            top[selected & (kept == 0)] = 0

    @staticmethod
    def _smooth_exclusion_contours(mask: np.ndarray) -> List[np.ndarray]:
        """
        Emit editable polygons that match hardware shape.

        Round openings become dense circles; pills become stadiums so Perfect
        Finish and the overlay both look production-clean (not octagons).
        """
        binary = (mask > 32).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        smoothed: List[np.ndarray] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 16:
                continue
            pts = contour.reshape(-1, 2).astype(np.float32)
            kind, params = HardwareRegionDetector._classify_cutout(pts)
            if kind == "circle":
                cx, cy, radius = params
                smoothed.append(
                    HardwareRegionDetector._sample_circle(
                        cx, cy, radius, samples=64
                    )
                )
                continue
            if kind in ("stadium", "rounded_rect"):
                x1, y1, x2, y2, corner = params
                stadium = HardwareRegionDetector._sample_rounded_rect(
                    x1, y1, x2, y2, corner, samples_per_corner=16
                )
                if stadium is not None:
                    smoothed.append(stadium.reshape(-1, 1, 2))
                    continue

            peri = float(cv2.arcLength(contour, True))
            # Keep free shapes dense — ≤12-vert approx looks polygonal on export.
            for eps_factor in (0.008, 0.004, 0.0025):
                approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
                n = int(approx.shape[0])
                if 8 <= n <= 64:
                    smoothed.append(approx)
                    break
            else:
                # Chaikin-smooth the raw chain for a soft outline.
                dense = contour.reshape(-1, 2).astype(np.float32)
                if dense.shape[0] > 96:
                    step = max(1, dense.shape[0] // 72)
                    dense = dense[::step]
                smoothed.append(dense.reshape(-1, 1, 2))
        return smoothed

    @staticmethod
    def _circle_params_from_pts(
        pts: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Stable (cx, cy, r) for a round cutout — AABB short-side preferred so
        editor-locked 1:1 circles stay perfectly round after resize.
        """
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return 0.0, 0.0, -1.0
        x1 = float(pts[:, 0].min())
        y1 = float(pts[:, 1].min())
        x2 = float(pts[:, 0].max())
        y2 = float(pts[:, 1].max())
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        radius = 0.5 * min(x2 - x1, y2 - y1)
        cx2, cy2, r2 = HardwareRegionDetector._fit_circle_least_squares(pts)
        if r2 > 1.0:
            # Prefer LS fit when it agrees with the AABB (true disk samples).
            if abs(r2 - radius) / max(radius, 1.0) < 0.22:
                return float(cx2), float(cy2), float(r2)
            # Dense circular polylines: trust fit even if AABB is slightly long.
            n = int(pts.shape[0])
            if n >= 24 and abs(r2 - radius) / max(radius, 1.0) < 0.40:
                return float(cx2), float(cy2), float(r2)
        return float(cx), float(cy), float(max(radius, 1.0))

    @staticmethod
    def _looks_like_true_disk(pts: np.ndarray) -> bool:
        """True flash/lens disks vs axis-aligned selection boxes."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return False
        contour = pts.reshape(-1, 1, 2)
        area = float(cv2.contourArea(contour))
        if area < 4:
            return False
        peri = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area) / max(peri * peri, 1e-3)
        (_cx, _cy), radius = cv2.minEnclosingCircle(pts)
        radius = float(max(radius, 1.0))
        fill = area / max(float(np.pi * radius ** 2), 1.0)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        aspect = max(w, h) / max(min(w, h), 1)
        n_verts = int(pts.shape[0])
        bbox_fill = area / max(float(w * h), 1.0)
        # Sparse AABB (4–8 verts filling the box) must never promote to circle.
        if n_verts <= 8 and bbox_fill >= 0.82:
            return False
        try:
            (_c, (rw, rh), _a) = cv2.minAreaRect(pts)
            rot_aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        except Exception:
            rot_aspect = aspect
        return (
            aspect <= 1.22
            and rot_aspect <= 1.28
            and fill >= 0.82
            and circularity >= 0.78
        )

    @staticmethod
    def _classify_cutout(
        pts: np.ndarray,
    ) -> Tuple[str, Tuple[float, ...]]:
        """
        Decide circle / stadium / rounded_rect / free for a cutout polygon.

        Returns (kind, params). Circle params = (cx, cy, r). Box kinds =
        (x1, y1, x2, y2, corner_radius).
        """
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return "free", ()
        contour = pts.reshape(-1, 1, 2)
        area = float(cv2.contourArea(contour))
        if area < 4:
            return "free", ()
        peri = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area) / max(peri * peri, 1e-3)
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        radius = float(max(radius, 1.0))
        circle_area = float(np.pi * radius ** 2)
        fill = area / max(circle_area, 1.0)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        aspect = max(w, h) / max(min(w, h), 1)
        short = float(min(w, h))
        # Rotated aspect catches side buttons that are diagonal in the photo.
        try:
            (_c, (rw, rh), _a) = cv2.minAreaRect(pts)
            rot_aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        except Exception:
            rot_aspect = aspect

        # Compact round → circle ONLY for real flash / lens disks.
        # User AABB squares used to pass (fill≈0.64) and paint giant circles
        # over whole camera modules (Redmi / any square selection).
        # Size caps removed: editor flash on hi-res phones often exceeds 64px
        # short-side; those must stay perfect SDF circles, not rounded boxes.
        n_verts = int(pts.shape[0])
        bbox_fill = area / max(float(w * h), 1.0)
        looks_like_box = n_verts <= 8 and bbox_fill >= 0.82
        circular = (
            not looks_like_box
            and HardwareRegionDetector._looks_like_true_disk(pts)
        )
        if circular:
            cx2, cy2, r2 = HardwareRegionDetector._circle_params_from_pts(pts)
            if r2 > 0:
                cx, cy, radius = cx2, cy2, r2
            return "circle", (float(cx), float(cy), float(radius))

        # Side buttons / speakers — only THIN elongated capsules (volume
        # rockers). Tall camera-island AABBs must NOT become stadium pills.
        skinny = float(min(w, h)) <= max(14.0, float(max(w, h)) * 0.38)
        if skinny and aspect >= 2.0 and (fill >= 0.35 or bbox_fill >= 0.40):
            pad = max(0.5, 0.015 * short)
            x1, y1 = float(x) - pad * 0.15, float(y) - pad * 0.15
            x2, y2 = float(x + w) + pad * 0.15, float(y + h) + pad * 0.15
            corner = float(np.clip(short * 0.48, 2.0, short * 0.5 - 0.5))
            return "stadium", (x1, y1, x2, y2, corner)

        # Camera / module boxes — mild "halka" round corners (not stadium ends).
        # User rectangle selections are 4-pt AABBs (looks_like_box); keep the
        # hole close to the editor box so wrap cannot bleed onto the plate.
        if (area >= 350 or (w >= 28 and h >= 28) or looks_like_box) and aspect < 3.5:
            if bbox_fill >= 0.42 or area >= 700 or looks_like_box:
                pad = max(0.5, 0.015 * short)
                x1, y1 = float(x) - pad * 0.15, float(y) - pad * 0.15
                x2, y2 = float(x + w) + pad * 0.15, float(y + h) + pad * 0.15
                mild = float(np.clip(short * 0.16, 3.0, short * 0.22))
                return "rounded_rect", (x1, y1, x2, y2, mild)

        return "free", ()

    @staticmethod
    def _fit_circle_least_squares(
        pts: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Algebraic circle fit (stable for jagged flash / lens outlines)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            return 0.0, 0.0, -1.0
        x = pts[:, 0]
        y = pts[:, 1]
        x_m = float(x.mean())
        y_m = float(y.mean())
        u = x - x_m
        v = y - y_m
        suu = float(np.dot(u, u))
        suv = float(np.dot(u, v))
        svv = float(np.dot(v, v))
        suuu = float(np.dot(u, u * u))
        svvv = float(np.dot(v, v * v))
        suvv = float(np.dot(u, v * v))
        svuu = float(np.dot(v, u * u))
        denom = 2.0 * (suu * svv - suv * suv)
        if abs(denom) < 1e-8:
            (cx, cy), radius = cv2.minEnclosingCircle(pts.astype(np.float32))
            return float(cx), float(cy), float(radius)
        uc = (suuu + suvv) * svv - (svvv + svuu) * suv
        vc = (svvv + svuu) * suu - (suuu + suvv) * suv
        uc /= denom
        vc /= denom
        cx = uc + x_m
        cy = vc + y_m
        radius = float(np.sqrt(uc * uc + vc * vc + (suu + svv) / len(pts)))
        if not np.isfinite(radius) or radius < 1.0:
            (cx2, cy2), r2 = cv2.minEnclosingCircle(pts.astype(np.float32))
            return float(cx2), float(cy2), float(r2)
        return float(cx), float(cy), float(radius)

    @staticmethod
    def _sample_circle(
        cx: float, cy: float, radius: float, samples: int = 64
    ) -> np.ndarray:
        """Dense polygon approximating a true circle (reads round on screen)."""
        samples = max(48, int(samples))
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        pts = np.stack(
            [cx + radius * np.cos(angles), cy + radius * np.sin(angles)],
            axis=1,
        ).astype(np.float32)
        return pts.reshape(-1, 1, 2)

    @staticmethod
    def _sample_ellipse(
        ellipse, samples: int = 12
    ) -> Optional[np.ndarray]:
        """Polygon approximating an OpenCV fitEllipse result."""
        (cx, cy), (axis_w, axis_h), angle = ellipse
        if axis_w < 2 or axis_h < 2:
            return None
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        x = (axis_w / 2.0) * np.cos(t)
        y = (axis_h / 2.0) * np.sin(t)
        xr = cx + x * cos_a - y * sin_a
        yr = cy + x * sin_a + y * cos_a
        pts = np.stack([xr, yr], axis=1).astype(np.float32)
        return pts.reshape(-1, 1, 2)

    @staticmethod
    def make_shape_polygon(
        shape: str,
        center: Tuple[float, float],
        size: float,
        *,
        aspect: float = 1.0,
        corner_frac: float = -1.0,
        rotation_deg: float = 0.0,
        sides: int = 6,
    ) -> np.ndarray:
        """
        Build a normalised cutout polygon for the Shift+click shape tool.

        Existing: circle, square, triangle, capsule/button, free.
        Added: rounded_rect, rounded_square, rectangle, oval, pill_h, pill_v,
        squircle, superellipse, polygon, custom_path.
        """
        cx, cy = float(center[0]), float(center[1])
        size = max(float(size), 0.008)
        shape = (shape or "circle").lower().strip()
        # Wide range so elongated pills/ovals survive rebuild without
        # axis drift when aspect would otherwise clamp.
        aspect = float(np.clip(aspect, 0.08, 12.0))
        half_w = size * float(np.sqrt(aspect))
        half_h = size / float(np.sqrt(max(aspect, 1e-6)))

        def _box_rr(
            x1: float, y1: float, x2: float, y2: float, corner: float
        ) -> np.ndarray:
            # `_sample_rounded_rect` expects pixel-scale spans (≥4). Cutout
            # tool coords are normalised 0–1 — scale up, sample, scale down.
            span = max(float(x2 - x1), float(y2 - y1), 1e-6)
            if span < 4.0:
                scale = 200.0 / span
                stadium = HardwareRegionDetector._sample_rounded_rect(
                    x1 * scale,
                    y1 * scale,
                    x2 * scale,
                    y2 * scale,
                    max(corner * scale, 1.0),
                    samples_per_corner=12,
                )
                if stadium is not None:
                    return (stadium.reshape(-1, 2) / scale).astype(np.float32)
            else:
                stadium = HardwareRegionDetector._sample_rounded_rect(
                    x1, y1, x2, y2, corner, samples_per_corner=12
                )
                if stadium is not None:
                    return stadium.reshape(-1, 2)
            return np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )

        def _rotate(pts: np.ndarray) -> np.ndarray:
            ang = float(rotation_deg) * (np.pi / 180.0)
            if abs(ang) < 1e-6:
                return pts
            c, s = float(np.cos(ang)), float(np.sin(ang))
            rel = pts - np.array([cx, cy], dtype=np.float32)
            rot = np.column_stack(
                [rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c]
            )
            return (rot + np.array([cx, cy], dtype=np.float32)).astype(np.float32)

        def _superellipse(n: float, samples: int = 64) -> np.ndarray:
            # |x/a|^n + |y/b|^n = 1
            a, b = half_w, half_h
            t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
            # Continuous parameterization for even n-like shapes.
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            exp = 2.0 / max(float(n), 0.5)
            x = np.sign(cos_t) * a * (np.abs(cos_t) ** exp)
            y = np.sign(sin_t) * b * (np.abs(sin_t) ** exp)
            return np.stack([cx + x, cy + y], axis=1).astype(np.float32)

        if shape == "square":
            half = size
            pts = np.array(
                [
                    [cx - half, cy - half],
                    [cx + half, cy - half],
                    [cx + half, cy + half],
                    [cx - half, cy + half],
                ],
                dtype=np.float32,
            )
        elif shape == "rectangle":
            # Mild corner round — matches camera-module plates (not sharp 90°).
            frac = (
                float(corner_frac)
                if corner_frac >= 0.0
                else 0.16
            )
            frac = float(np.clip(frac, 0.08, 0.22))
            short = min(half_w, half_h) * 2.0
            pts = _box_rr(
                cx - half_w,
                cy - half_h,
                cx + half_w,
                cy + half_h,
                short * frac,
            )
        elif shape in ("rounded_square", "rounded-square"):
            # Independent W×H (creation usually starts at aspect=1).
            short = min(half_w, half_h) * 2.0
            corner = (
                short * float(corner_frac)
                if corner_frac >= 0.0
                else short * 0.28
            )
            pts = _box_rr(
                cx - half_w, cy - half_h, cx + half_w, cy + half_h, corner
            )
        elif shape in ("rounded_rect", "rounded-rect", "rounded_rectangle"):
            short = min(half_w, half_h) * 2.0
            corner = (
                short * float(corner_frac)
                if corner_frac >= 0.0
                else short * 0.30
            )
            pts = _box_rr(
                cx - half_w, cy - half_h, cx + half_w, cy + half_h, corner
            )
        elif shape in ("oval", "ellipse"):
            pts = _superellipse(2.0, samples=64)
        elif shape in ("pill_h", "pill-horizontal", "capsule_h"):
            # Horizontal stadium — AABB from half_w/half_h (no extra stretch;
            # stretch on create comes from aspect so rebuild stays stable).
            short = half_h * 2.0
            corner = short * 0.48
            pts = _box_rr(
                cx - half_w, cy - half_h,
                cx + half_w, cy + half_h, corner
            )
        elif shape in ("pill_v", "pill-vertical", "capsule_v"):
            short = half_w * 2.0
            corner = short * 0.48
            pts = _box_rr(
                cx - half_w, cy - half_h,
                cx + half_w, cy + half_h, corner
            )
        elif shape in ("squircle", "ios"):
            # iPhone-style squircle ≈ superellipse n=5.
            pts = _superellipse(5.0, samples=72)
        elif shape == "superellipse":
            pts = _superellipse(3.5, samples=72)
        elif shape == "polygon":
            n = int(np.clip(sides, 3, 16))
            ang0 = -np.pi / 2.0
            t = ang0 + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            pts = np.stack(
                [cx + size * np.cos(t), cy + size * np.sin(t)], axis=1
            ).astype(np.float32)
        elif shape in ("custom_path", "custom", "path"):
            # Editable freeform seed (irregular hex) — refine with Ctrl+click.
            t = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
            radii = size * np.array([1.0, 0.78, 1.05, 0.82, 1.0, 0.88])
            pts = np.stack(
                [cx + radii * np.cos(t), cy + radii * np.sin(t)], axis=1
            ).astype(np.float32)
        elif shape == "triangle":
            h = size * 1.15
            pts = np.array(
                [
                    [cx, cy - h],
                    [cx + h, cy + h * 0.7],
                    [cx - h, cy + h * 0.7],
                ],
                dtype=np.float32,
            )
        elif shape in ("capsule", "button"):
            # Side-button stadium — aspect-driven so resize can change W and H.
            short = min(half_w, half_h) * 2.0
            corner = min(half_w, half_h) * 0.95
            pts = _box_rr(
                cx - half_w, cy - half_h,
                cx + half_w, cy + half_h, corner
            )
        elif shape == "free":
            half = size
            pts = np.array(
                [
                    [cx, cy - half],
                    [cx + half, cy],
                    [cx, cy + half],
                    [cx - half, cy],
                ],
                dtype=np.float32,
            )
        else:
            pts = HardwareRegionDetector._sample_circle(
                cx, cy, size, samples=64
            ).reshape(-1, 2)
        return _rotate(pts).reshape(-1, 1, 2)

    @staticmethod
    def merge_overlapping_contours(
        contours: List[np.ndarray],
        *,
        overlap_ratio: float = 0.25,
        center_frac: float = 0.08,
    ) -> List[np.ndarray]:
        """
        Merge only near-duplicate / nested cutouts — keep disjoint ones apart.

        Previously every contour was painted into one mask, so Perfect Finish
        collapsed separate lenses, flash, and buttons into a single outline.
        """
        if not contours:
            return []
        items: List[np.ndarray] = []
        for contour in contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 3:
                items.append(pts)
        if len(items) <= 1:
            return [p.reshape(-1, 1, 2) for p in items]

        n = len(items)
        parent = list(range(n))

        def _find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(i: int, j: int) -> None:
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[rj] = ri

        areas = [
            float(max(cv2.contourArea(p.reshape(-1, 1, 2)), 1.0))
            for p in items
        ]
        centers = [p.mean(axis=0) for p in items]
        bboxes = []
        for p in items:
            mn = p.min(axis=0)
            mx = p.max(axis=0)
            bboxes.append((float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])))

        def _pair_overlap(i: int, j: int) -> float:
            """Intersection / min(area) — 1.0 means nested or identical."""
            ax1, ay1, ax2, ay2 = bboxes[i]
            bx1, by1, bx2, by2 = bboxes[j]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            # Cheap bbox gate, then precise mask IoU on the local crop.
            pad = 3.0
            x0 = int(np.floor(min(ax1, bx1) - pad))
            y0 = int(np.floor(min(ay1, by1) - pad))
            x1 = int(np.ceil(max(ax2, bx2) + pad))
            y1 = int(np.ceil(max(ay2, by2) + pad))
            w = max(2, x1 - x0)
            h = max(2, y1 - y0)
            if w * h > 1_500_000:
                inter_bbox = (ix2 - ix1) * (iy2 - iy1)
                return float(inter_bbox / max(min(areas[i], areas[j]), 1.0))
            ma = np.zeros((h, w), dtype=np.uint8)
            mb = np.zeros((h, w), dtype=np.uint8)
            oa = np.round(items[i] - np.array([x0, y0], np.float32)).astype(
                np.int32
            )
            ob = np.round(items[j] - np.array([x0, y0], np.float32)).astype(
                np.int32
            )
            cv2.fillPoly(ma, [oa.reshape(-1, 1, 2)], 255)
            cv2.fillPoly(mb, [ob.reshape(-1, 1, 2)], 255)
            inter = float(np.count_nonzero((ma > 0) & (mb > 0)))
            if inter <= 0:
                return 0.0
            return inter / max(min(areas[i], areas[j]), 1.0)

        all_pts = np.vstack(items)
        diag = float(
            np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
        )
        center_lim = max(8.0, diag * float(center_frac))
        ratio = float(np.clip(overlap_ratio, 0.05, 0.95))

        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(centers[i] - centers[j]))
                # Far apart and no bbox overlap → keep separate.
                ax1, ay1, ax2, ay2 = bboxes[i]
                bx1, by1, bx2, by2 = bboxes[j]
                bbox_hit = not (
                    ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1
                )
                if dist > center_lim and not bbox_hit:
                    continue
                overlap = _pair_overlap(i, j)
                # Merge near-duplicates / nested holes only.
                if overlap >= ratio or (
                    dist <= center_lim * 0.55 and overlap >= ratio * 0.55
                ):
                    _union(i, j)

        groups: dict = {}
        for i in range(n):
            root = _find(i)
            groups.setdefault(root, []).append(i)

        result: List[np.ndarray] = []
        for indices in groups.values():
            if len(indices) == 1:
                result.append(items[indices[0]].reshape(-1, 1, 2))
                continue
            # Union only this cluster.
            cluster = [items[k] for k in indices]
            pts = np.vstack(cluster)
            mn = pts.min(axis=0) - 4
            mx = pts.max(axis=0) + 4
            span = np.maximum(mx - mn, 1.0)
            w = int(np.ceil(span[0])) + 2
            h = int(np.ceil(span[1])) + 2
            if w * h > 2_000_000:
                # Keep the largest member if union mask is too big.
                best = max(indices, key=lambda k: areas[k])
                result.append(items[best].reshape(-1, 1, 2))
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            offset = mn.astype(np.float32)
            for p in cluster:
                local = np.round(p - offset).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [local], 255)
            merged, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not merged:
                best = max(indices, key=lambda k: areas[k])
                result.append(items[best].reshape(-1, 1, 2))
                continue
            outer = max(merged, key=cv2.contourArea)
            if float(cv2.contourArea(outer)) < 12:
                continue
            out = outer.reshape(-1, 2).astype(np.float32) + offset
            result.append(out.reshape(-1, 1, 2))
        return result

    @staticmethod
    def perfect_finish_contours(
        contours: List[np.ndarray],
        phone_image: Optional[np.ndarray] = None,
        *,
        lock_bounds: bool = False,
    ) -> List[np.ndarray]:
        """
        Snap rough cutouts to clean geometric finishes.

        Circles / flash / fingerprint → true circles. Side buttons → stadium
        capsules. Camera bumps → smooth rounded rectangles. Uses optional local
        edge cues from the phone photo when available.

        lock_bounds=True (manual drag release): keep the user's size — only
        clean the shape. Prevents shrinks from being undone by edge snap.
        """
        finished: List[np.ndarray] = []
        gray = None
        if phone_image is not None and phone_image.size:
            phone = to_bgr(phone_image)
            gray = cv2.cvtColor(phone, cv2.COLOR_BGR2GRAY)

        for contour in contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            polished = HardwareRegionDetector._perfect_one_contour(
                pts, gray, lock_bounds=lock_bounds
            )
            if polished is not None and len(polished) >= 3:
                finished.append(polished.reshape(-1, 1, 2))
        return finished

    @staticmethod
    def rebuild_camera_cutouts(
        contours: List[np.ndarray],
        phone_image: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Snap camera / flash openings to the photo — never a hardcoded shape.

        Prefers the real hardware silhouette inside the user's selection
        (Redmi D-islands, L-shapes, odd modules). Only falls back to
        stadium/circle when the photo outline is already that clean.
        """
        parts = [
            np.asarray(c, dtype=np.float32).reshape(-1, 2)
            for c in contours
            if len(np.asarray(c).reshape(-1, 2)) >= 3
        ]
        if not parts:
            return []

        gray = None
        if phone_image is not None and phone_image.size:
            phone = to_bgr(phone_image)
            gray = cv2.cvtColor(phone, cv2.COLOR_BGR2GRAY)

        def _pack(items: List[np.ndarray]) -> List[np.ndarray]:
            cleaned = HardwareRegionDetector._finalize_camera_openings(
                items, gray
            )
            return [
                np.asarray(c, np.float32).reshape(-1, 1, 2) for c in cleaned
            ]

        # Already-separate compact openings must not be union-Houghed into
        # one plate (that rebuilt a rectangular hole around discrete lenses).
        if HardwareRegionDetector._parts_are_discrete_openings(parts):
            # Overlapping enclosing disks (one per lens) Hough the same
            # strongest ring when processed one-at-a-time and drop the
            # middle opening. Cluster Hough once, then keep disks.
            union = HardwareRegionDetector._rebuild_camera_from_lenses(
                parts, gray
            )
            if union and HardwareRegionDetector._parts_are_discrete_openings(
                union
            ):
                finished_u: List[np.ndarray] = []
                for item in union:
                    ip = np.asarray(item, np.float32).reshape(-1, 2)
                    if gray is not None:
                        ip = HardwareRegionDetector._shrink_disk_to_lens(
                            gray, ip
                        )
                    finished_u.append(ip)
                packed = _pack(finished_u)
                if len(packed) >= 2:
                    return packed
            finished: List[np.ndarray] = []
            for pts in parts:
                one = HardwareRegionDetector._rebuild_camera_from_lenses(
                    [pts], gray
                )
                seed_area = float(
                    cv2.contourArea(
                        np.asarray(pts, np.float32).reshape(-1, 1, 2)
                    )
                )
                if one:
                    cands = []
                    for item in one:
                        ip = np.asarray(item, np.float32).reshape(-1, 2)
                        ia = float(cv2.contourArea(ip.reshape(-1, 1, 2)))
                        if ia <= seed_area * 1.08 and ia >= 12:
                            cands.append((ia, ip))
                    if cands:
                        biggest = max(t[0] for t in cands)
                        compact = [
                            t for t in cands if t[0] >= biggest * 0.48
                        ]
                        disks = [
                            t
                            for t in compact
                            if HardwareRegionDetector._looks_like_true_disk(
                                t[1]
                            )
                        ]
                        pool = disks or compact
                        # Prefer the tight ring over the enclosing blob.
                        picked = min(pool, key=lambda t: t[0])[1]
                        if gray is not None:
                            picked = HardwareRegionDetector._shrink_disk_to_lens(
                                gray, picked
                            )
                        finished.append(picked)
                        continue
                polished = HardwareRegionDetector._perfect_one_contour(
                    pts, gray, lock_bounds=False
                )
                if polished is not None and len(polished) >= 3:
                    finished.append(polished)
                else:
                    finished.append(pts)
            if finished:
                return _pack(finished)

        # 1) Lens Hough first. Snap-to-silhouette used to win with the
        # cluster AABB and paint a rectangular hole around discrete lenses.
        rebuilt = HardwareRegionDetector._rebuild_camera_from_lenses(
            parts, gray
        )
        if rebuilt:
            # Smaller-than-seed is always OK (tight rings / island).
            # Larger is rejected so Hough cannot balloon past the detect box.
            if HardwareRegionDetector._rebuilt_agrees_with_user(parts, rebuilt):
                return _pack(rebuilt)
            r = np.vstack(
                [np.asarray(p, np.float32).reshape(-1, 2) for p in rebuilt]
            )
            u = np.vstack(
                [np.asarray(p, np.float32).reshape(-1, 2) for p in parts]
            )
            rw = float(r[:, 0].max() - r[:, 0].min())
            rh = float(r[:, 1].max() - r[:, 1].min())
            uw = float(u[:, 0].max() - u[:, 0].min())
            uh = float(u[:, 1].max() - u[:, 1].min())
            if rw * rh <= uw * uh * 1.02 and len(rebuilt) >= 1:
                return _pack(rebuilt)

        # 2) Dynamic: each user selection → photo silhouette when irregular.
        snapped = HardwareRegionDetector._snap_parts_to_photo(parts, gray)
        if snapped:
            return _pack(snapped)

        # 3) Per-contour circle/stadium polish (flash / simple pills).
        finished: List[np.ndarray] = []
        for pts in parts:
            polished = HardwareRegionDetector._perfect_one_contour(
                pts, gray, lock_bounds=True
            )
            if polished is not None and len(polished) >= 3:
                finished.append(polished)
            else:
                finished.append(pts)
        return _pack(finished)

    @staticmethod
    def _rebuilt_agrees_with_user(
        user_parts: List[np.ndarray],
        rebuilt: List[np.ndarray],
    ) -> bool:
        """Reject lens-stadium rebuilds that balloon past the user's selection."""
        if not user_parts or not rebuilt:
            return False
        u = np.vstack([np.asarray(p, np.float32).reshape(-1, 2) for p in user_parts])
        r = np.vstack([np.asarray(p, np.float32).reshape(-1, 2) for p in rebuilt])
        uw = float(u[:, 0].max() - u[:, 0].min())
        uh = float(u[:, 1].max() - u[:, 1].min())
        rw = float(r[:, 0].max() - r[:, 0].min())
        rh = float(r[:, 1].max() - r[:, 1].min())
        if uw < 4 or uh < 4:
            return False
        # Rebuilt must stay near the user footprint (not a giant semicircle).
        if rw > uw * 1.35 or rh > uh * 1.35:
            return False
        if rw * rh > uw * uh * 1.55:
            return False
        return True

    @staticmethod
    def _snap_parts_to_photo(
        parts: List[np.ndarray],
        gray: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Trace each cutout against the photo. Irregular islands stay contours;
        clean circles/stadiums stay analytical.
        """
        if gray is None or not parts:
            return []
        out: List[np.ndarray] = []
        for pts in parts:
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            kind, params = HardwareRegionDetector._classify_cutout(pts)
            # Tiny round flash / lens — keep as circle.
            if kind == "circle" and params:
                circ = HardwareRegionDetector._sample_circle(
                    float(params[0]), float(params[1]), float(params[2]),
                    samples=64,
                )
                out.append(np.asarray(circ, dtype=np.float32).reshape(-1, 2))
                continue

            sil = HardwareRegionDetector.extract_photo_silhouette(gray, pts)
            if sil is None or sil.shape[0] < 8:
                # No photo evidence — keep user geometry (lock bounds polish).
                polished = HardwareRegionDetector._perfect_one_contour(
                    pts, gray, lock_bounds=True
                )
                out.append(
                    polished if polished is not None else pts
                )
                continue

            g2, p2 = HardwareRegionDetector._classify_cutout(sil)
            sil_area = float(cv2.contourArea(sil.reshape(-1, 1, 2)))
            sx = float(sil[:, 0].max() - sil[:, 0].min())
            sy = float(sil[:, 1].max() - sil[:, 1].min())
            box_area = max(sx * sy, 1.0)
            rect_fill = sil_area / box_area

            # Irregular / D-shaped when AABB fill is low — dense photo contour.
            if rect_fill < 0.82 or g2 == "free":
                step = max(1, len(sil) // 72)
                dense = sil[::step]
                if dense.shape[0] < 16:
                    dense = sil
                out.append(dense.astype(np.float32))
                continue

            if g2 == "circle" and p2 and rect_fill >= 0.72:
                circ = HardwareRegionDetector._sample_circle(
                    float(p2[0]), float(p2[1]), float(p2[2]), samples=64
                )
                out.append(np.asarray(circ, dtype=np.float32).reshape(-1, 2))
                continue

            # Clean stadium only when the photo outline fills the pill well.
            if (
                g2 in ("stadium", "rounded_rect")
                and len(p2) >= 5
                and rect_fill >= 0.84
            ):
                stadium = HardwareRegionDetector._sample_rounded_rect(
                    float(p2[0]), float(p2[1]), float(p2[2]), float(p2[3]),
                    float(p2[4]), samples_per_corner=16,
                )
                if stadium is not None:
                    out.append(stadium.reshape(-1, 2))
                    continue

            step = max(1, len(sil) // 72)
            dense = sil[::step]
            if dense.shape[0] < 16:
                dense = sil
            out.append(dense.astype(np.float32))

        return out if len(out) == len(parts) else []

    @staticmethod
    def _parts_are_discrete_openings(parts: List[np.ndarray]) -> bool:
        """
        True when detect already produced separate openings.

        A module plate plus nested holes is one parent bbox containing the
        rest — those still rebuild as a single island.
        """
        if len(parts) < 2:
            return False
        boxes = []
        for part in parts:
            pts = np.asarray(part, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 3:
                continue
            boxes.append(
                (
                    float(pts[:, 0].min()),
                    float(pts[:, 1].min()),
                    float(pts[:, 0].max()),
                    float(pts[:, 1].max()),
                )
            )
        if len(boxes) < 2:
            return False
        for i, a in enumerate(boxes):
            contained = 0
            for j, b in enumerate(boxes):
                if i == j:
                    continue
                if (
                    b[0] >= a[0] - 2
                    and b[1] >= a[1] - 2
                    and b[2] <= a[2] + 2
                    and b[3] <= a[3] + 2
                ):
                    contained += 1
            if contained >= len(boxes) - 1:
                return False
        return True

    @staticmethod
    def _merge_overlapping_openings(parts: List[np.ndarray]) -> List[np.ndarray]:
        """Collapse near-duplicate lens disks; keep the tighter contour."""
        items = [
            np.asarray(p, np.float32).reshape(-1, 2)
            for p in parts
            if len(np.asarray(p).reshape(-1, 2)) >= 3
        ]
        if len(items) < 2:
            return items
        circles = []
        for pts in items:
            (cx, cy), radius = cv2.minEnclosingCircle(pts)
            area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
            disk = HardwareRegionDetector._looks_like_true_disk(pts)
            circles.append(
                (area, float(cx), float(cy), float(radius), disk, pts)
            )
        used = [False] * len(circles)
        kept_idx: List[int] = []
        for i in sorted(range(len(circles)), key=lambda k: circles[k][0]):
            if used[i]:
                continue
            used[i] = True
            kept_idx.append(i)
            area, cx, cy, radius, disk, _pts = circles[i]
            if not disk:
                continue
            for j, other in enumerate(circles):
                if used[j]:
                    continue
                _aj, ox, oy, orr, od, _op = other
                if not od:
                    continue
                dist = float(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5)
                if dist <= 0.50 * (radius + orr) or dist <= 0.45 * max(
                    radius, orr
                ):
                    used[j] = True
        kept = [circles[i] for i in kept_idx]
        kept.sort(key=lambda t: (t[2], t[1]))
        return [t[5] for t in kept]

    @staticmethod
    def _drop_ghost_cluster_disks(parts: List[np.ndarray]) -> List[np.ndarray]:
        """
        Drop a Hough ghost that sits between real stacked lenses.

        A false circle on the cover, tangent to two true openings, punches a
        white hole through valid artwork. Isolated dual/quad cameras overlap
        at most one peer at this threshold and are kept.
        """
        items = [
            np.asarray(p, np.float32).reshape(-1, 2)
            for p in parts
            if len(np.asarray(p).reshape(-1, 2)) >= 3
        ]
        if len(items) < 3:
            return items
        meta = []
        for pts in items:
            (cx, cy), radius = cv2.minEnclosingCircle(pts)
            meta.append(
                (
                    float(cx),
                    float(cy),
                    float(radius),
                    HardwareRegionDetector._looks_like_true_disk(pts),
                    pts,
                )
            )
        disks = [i for i, m in enumerate(meta) if m[3]]
        if len(disks) < 3:
            return items
        drop = set()
        for i in disks:
            cx, cy, radius, _d, _p = meta[i]
            n_ov = 0
            for j in disks:
                if i == j:
                    continue
                ox, oy, orr, _od, _op = meta[j]
                dist = float(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5)
                if dist < 0.90 * (radius + orr):
                    n_ov += 1
            if n_ov >= 2:
                drop.add(i)
        if not drop or len(drop) >= len(disks):
            return items
        return [meta[i][4] for i in range(len(meta)) if i not in drop]

    @staticmethod
    def _drop_false_hardware_disks(
        parts: List[np.ndarray],
        gray: np.ndarray,
    ) -> List[np.ndarray]:
        """Drop compact disks that are not a lens/flash contrast ring."""
        items = [
            np.asarray(p, np.float32).reshape(-1, 2)
            for p in parts
            if len(np.asarray(p).reshape(-1, 2)) >= 3
        ]
        if not items or gray is None or gray.size == 0:
            return items
        kept: List[np.ndarray] = []
        for pts in items:
            if not HardwareRegionDetector._looks_like_true_disk(pts):
                kept.append(pts)
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(pts)
            ix = int(np.clip(round(float(cx)), 0, gray.shape[1] - 1))
            iy = int(np.clip(round(float(cy)), 0, gray.shape[0] - 1))
            ir = max(2, int(round(float(radius))))
            if HardwareRegionDetector._circle_looks_like_hardware(
                gray, ix, iy, ir
            ):
                kept.append(pts)
        return kept if kept else items

    @staticmethod
    def _finalize_camera_openings(
        parts: List[np.ndarray],
        gray: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """Merge duplicates, drop ghosts, keep real detected geometry."""
        items = [
            np.asarray(p, np.float32).reshape(-1, 2)
            for p in parts
            if len(np.asarray(p).reshape(-1, 2)) >= 3
        ]
        if not items:
            return []
        items = HardwareRegionDetector._merge_overlapping_openings(items)
        items = HardwareRegionDetector._drop_ghost_cluster_disks(items)
        items = HardwareRegionDetector._drop_nested_openings(items)
        if gray is not None:
            filtered = HardwareRegionDetector._drop_false_hardware_disks(
                items, gray
            )
            if filtered:
                items = filtered
        return items

    @staticmethod
    def _drop_nested_openings(parts: List[np.ndarray]) -> List[np.ndarray]:
        """Drop glint disks that sit inside a larger lens opening."""
        items = [np.asarray(p, np.float32).reshape(-1, 2) for p in parts]
        items = [p for p in items if p.shape[0] >= 3]
        if len(items) < 2:
            return items
        meta = []
        for pts in items:
            (cx, cy), radius = cv2.minEnclosingCircle(pts)
            area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
            meta.append((area, float(cx), float(cy), float(radius), pts))
        meta.sort(key=lambda t: -t[0])
        kept: List[np.ndarray] = []
        kept_meta: List[Tuple[float, float, float]] = []
        for area, cx, cy, radius, pts in meta:
            nested = False
            for kx, ky, kr in kept_meta:
                dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
                if dist + radius * 0.35 <= kr * 1.05 and area < kr * kr * 1.6:
                    nested = True
                    break
            if nested:
                continue
            kept.append(pts)
            kept_meta.append((cx, cy, radius))
        return kept if kept else items

    @staticmethod
    def _shrink_disk_to_lens(
        gray: np.ndarray, pts: np.ndarray
    ) -> np.ndarray:
        """
        Shrink an enclosing disk onto the dark lens / bright flash core.

        Oversized Hough hits include adjacent cover; contrast vs the ring
        peaks on the real opening.
        """
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        cx, cy, radius = HardwareRegionDetector._refine_circle(
            gray, float(cx), float(cy), float(radius)
        )
        best_r = float(radius)
        best_s = -1.0
        ix, iy = int(round(cx)), int(round(cy))
        for scale in (1.0, 0.92, 0.84, 0.76, 0.68, 0.60):
            rr = float(radius) * scale
            if rr < 4.0:
                break
            ir = max(2, int(round(rr)))
            if not HardwareRegionDetector._circle_looks_like_hardware(
                gray, ix, iy, ir
            ):
                continue
            yy, xx = np.ogrid[-ir : ir + 1, -ir : ir + 1]
            disk = xx * xx + yy * yy <= ir * ir
            h, w = gray.shape[:2]
            y0, y1 = max(0, iy - ir), min(h, iy + ir + 1)
            x0, x1 = max(0, ix - ir), min(w, ix + ir + 1)
            patch = gray[y0:y1, x0:x1].astype(np.float32)
            dy0, dx0 = y0 - (iy - ir), x0 - (ix - ir)
            local = disk[dy0 : dy0 + patch.shape[0], dx0 : dx0 + patch.shape[1]]
            if np.count_nonzero(local) < 8:
                continue
            interior = float(np.median(patch[local]))
            ry = max(ir + 2, int(ir * 1.55))
            y0r, y1r = max(0, iy - ry), min(h, iy + ry + 1)
            x0r, x1r = max(0, ix - ry), min(w, ix + ry + 1)
            surround = gray[y0r:y1r, x0r:x1r].astype(np.float32)
            yy, xx = np.ogrid[y0r - iy : y1r - iy, x0r - ix : x1r - ix]
            ring = (xx * xx + yy * yy <= (ir * 1.55) ** 2) & (
                xx * xx + yy * yy > (ir * 1.08) ** 2
            )
            if np.count_nonzero(ring) < 8:
                continue
            exterior = float(np.median(surround[ring]))
            score = abs(interior - exterior) - 0.15 * rr
            if score > best_s:
                best_s = score
                best_r = rr
        if best_s < 0:
            return pts
        return HardwareRegionDetector._sample_circle(
            cx, cy, best_r, samples=48
        ).reshape(-1, 2)

    @staticmethod
    def _clip_face_openings_to_cover(
        mask: np.ndarray,
        quad: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        """
        Camera holes live on the cover face, not the case rim.

        Rim-hugging disks (corner fillets warped into the mask) are clipped
        to an inset of the cover quad. Thin side buttons are left on the rim.
        """
        binary = (mask > 32).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            return
        cover = np.zeros((height, width), dtype=np.uint8)
        pts = np.round(order_points(quad)).astype(np.int32)
        cv2.fillConvexPoly(cover, pts, 255)
        inset = max(3, int(round(min(width, height) * 0.012)))
        inner = cv2.erode(
            cover,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1)
            ),
        )
        q = order_points(quad)
        qx0 = float(q[:, 0].min())
        qx1 = float(q[:, 0].max())
        cover_w = max(qx1 - qx0, 1.0)
        rim_band = max(6.0, 0.08 * cover_w)
        count, labels, stats, _cents = cv2.connectedComponentsWithStats(
            binary, 8
        )
        for label in range(1, count):
            x, y, bw, bh, area = stats[label]
            if area < 16:
                continue
            on_cover_rim = (
                float(x) <= qx0 + rim_band
                or float(x + bw) >= qx1 - rim_band
            )
            skinny = bw <= max(12, int(cover_w * 0.10)) and bh >= bw * 1.4
            if on_cover_rim and skinny:
                continue
            comp = labels == label
            outside = comp & (inner == 0)
            if np.count_nonzero(outside) == 0:
                continue
            if float(np.count_nonzero(outside)) / float(area) > 0.22:
                mask[comp] = 0
            else:
                mask[outside] = 0

    @staticmethod
    def _module_plate_encloses(
        gray: np.ndarray,
        lenses: List[Tuple[float, float, float]],
        bx1: float,
        by1: float,
        bx2: float,
        by2: float,
    ) -> bool:
        """
        True when a raised camera plate contains the lens cluster.

        Distinguishes a Redmi-style island (one hole) from discrete rings
        whose union bbox is tall but has no plate.
        """
        if gray is None or len(lenses) < 2:
            return False
        h, w = gray.shape[:2]
        top_h = int(min(h, max(int(h * 0.58), int(np.ceil(by2)) + 8)))
        circles = [
            (int(round(x)), int(round(y)), max(2, int(round(r))))
            for x, y, r in lenses
        ]
        plate = HardwareRegionDetector._detect_square_camera_plate(
            gray, w, top_h, circles
        )
        if plate is None:
            return False
        x1, y1, x2, y2 = plate
        med_d = 2.0 * float(np.median([c[2] for c in lenses]))
        if min(x2 - x1, y2 - y1) < med_d * 1.8:
            return False
        n_in = 0
        for x, y, r in lenses:
            if (x1 - 0.25 * r) <= x <= (x2 + 0.25 * r) and (
                y1 - 0.25 * r
            ) <= y <= (y2 + 0.25 * r):
                n_in += 1
        need = min(len(lenses), max(2, int(np.ceil(0.7 * len(lenses)))))
        return n_in >= need

    @staticmethod
    def _lenses_sit_on_cover_face(
        gray: np.ndarray,
        lenses: List[Tuple[float, float, float]],
        bx1: float,
        by1: float,
        bx2: float,
        by2: float,
    ) -> bool:
        """
        True when lenses are discrete rings on the cover face.

        A raised island plate is a different colour between the lenses;
        a flat back (separate cutouts) matches the surrounding cover.
        """
        if gray is None or len(lenses) < 2:
            return False
        h, w = gray.shape[:2]
        lens_m = np.zeros((h, w), dtype=np.uint8)
        for cx, cy, radius in lenses:
            rr = max(2, int(round(float(radius) * 0.82)))
            cv2.circle(
                lens_m,
                (int(round(cx)), int(round(cy))),
                rr,
                255,
                -1,
            )
        x0 = int(np.clip(np.floor(bx1), 0, w - 1))
        y0 = int(np.clip(np.floor(by1), 0, h - 1))
        x1 = int(np.clip(np.ceil(bx2), 0, w))
        y1 = int(np.clip(np.ceil(by2), 0, h))
        cluster = np.zeros((h, w), dtype=np.uint8)
        cluster[y0:y1, x0:x1] = 255
        gap = cv2.bitwise_and(cluster, cv2.bitwise_not(lens_m))
        pad = max(4, int(round(min(bx2 - bx1, by2 - by1) * 0.08)))
        ring = cv2.dilate(
            cluster,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
            ),
            iterations=1,
        )
        outside = cv2.bitwise_and(ring, cv2.bitwise_not(cluster))
        gpix = gap > 0
        opix = outside > 0
        lpix = lens_m > 0
        if (
            int(np.count_nonzero(gpix)) < 24
            or int(np.count_nonzero(opix)) < 24
            or int(np.count_nonzero(lpix)) < 16
        ):
            return False
        med_gap = float(np.median(gray[gpix]))
        med_body = float(np.median(gray[opix]))
        med_lens = float(np.median(gray[lpix]))
        # Gap looks like the cover, not like the dark glass / island plate.
        return abs(med_gap - med_body) + 6.0 < abs(med_gap - med_lens)

    @staticmethod
    def _rebuild_camera_from_lenses(
        parts: List[np.ndarray],
        gray: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """Detect lens circles in the cutout ROI and emit stadium + flash."""
        if gray is None or not parts:
            return []

        all_pts = np.vstack(parts)
        bx1 = float(all_pts[:, 0].min())
        by1 = float(all_pts[:, 1].min())
        bx2 = float(all_pts[:, 0].max())
        by2 = float(all_pts[:, 1].max())
        bw = max(bx2 - bx1, 1.0)
        bh = max(by2 - by1, 1.0)
        pad = max(8.0, min(bw, bh) * 0.18)
        height, width = gray.shape[:2]
        x0 = int(np.clip(bx1 - pad, 0, width - 1))
        y0 = int(np.clip(by1 - pad, 0, height - 1))
        x1 = int(np.clip(bx2 + pad, 0, width))
        y1 = int(np.clip(by2 + pad, 0, height))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []

        roi = gray[y0:y1, x0:x1]
        blur = cv2.GaussianBlur(roi, (0, 0), 1.15)
        # Lens radii relative to the user cutout — works across phone models.
        r_min = max(4, int(min(bw, bh) * 0.08))
        # Stacked discrete lenses: the cluster is tall/narrow and each
        # opening is ~the short side. A 0.28 cap only fitted square islands.
        short, long = min(bw, bh), max(bw, bh)
        aspect = long / max(short, 1.0)
        r_frac = 0.52 if (aspect >= 1.65 or len(parts) == 1) else 0.32
        r_max = max(r_min + 2, int(short * r_frac))
        min_dist = max(6, int(short * 0.16))
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=min_dist,
            param1=60,
            param2=16,
            minRadius=r_min,
            maxRadius=r_max,
        )
        if circles is None or len(circles[0]) < 1:
            # Softer pass for low-contrast silvery backs.
            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(4, min_dist // 2),
                param1=50,
                param2=12,
                minRadius=max(3, r_min - 2),
                maxRadius=r_max + 4,
            )
        if circles is None or len(circles[0]) < 1:
            return []

        found: List[Tuple[float, float, float]] = []
        for cx, cy, radius in circles[0]:
            fx = float(cx) + float(x0)
            fy = float(cy) + float(y0)
            fr = float(radius)
            # Keep circles that overlap the user's cutout bbox.
            if fx + fr < bx1 - 4 or fx - fr > bx2 + 4:
                continue
            if fy + fr < by1 - 4 or fy - fr > by2 + 4:
                continue
            found.append((fx, fy, fr))
        if not found:
            return []

        # Deduplicate near-duplicates.
        found.sort(key=lambda c: -c[2])
        unique: List[Tuple[float, float, float]] = []
        max_r = short * r_frac
        for cx, cy, radius in found:
            if radius > max_r:
                continue
            ix = int(np.clip(round(cx), 0, width - 1))
            iy = int(np.clip(round(cy), 0, height - 1))
            # Phone-corner fillets look circular to Hough. Real lenses are
            # dark; a bright circle in the cluster's top-left is the rim.
            near_tl = (cx - bx1) < 0.22 * bw and (cy - by1) < 0.22 * bh
            if near_tl and float(gray[iy, ix]) >= 130.0:
                continue
            if any(
                (cx - ux) ** 2 + (cy - uy) ** 2 < (0.45 * max(radius, ur)) ** 2
                for ux, uy, ur in unique
            ):
                continue
            unique.append((cx, cy, radius))
        if not unique:
            return []

        radii = np.array([c[2] for c in unique], dtype=np.float32)
        median_r = float(np.median(radii))
        # Primary lenses share a similar size; flash/mic are usually smaller.
        lenses = [
            c for c in unique if c[2] >= median_r * 0.72
        ]
        flashes = [
            c for c in unique if c[2] < median_r * 0.72
        ]
        if len(lenses) < 1:
            lenses = list(unique)
            flashes = []

        out: List[np.ndarray] = []
        if len(lenses) == 1:
            cx, cy, radius = lenses[0]
            # Slight pad so the ring / bezel is fully punched.
            out.append(
                HardwareRegionDetector._sample_circle(
                    cx, cy, radius * 1.06, samples=48
                )
            )
        elif (
            HardwareRegionDetector._lenses_sit_on_cover_face(
                gray, lenses, bx1, by1, bx2, by2
            )
            and not HardwareRegionDetector._module_plate_encloses(
                gray, lenses, bx1, by1, bx2, by2
            )
        ):
            # Discrete rings on a flat back — punch each lens,
            # never a rectangular cluster hole.
            for cx, cy, radius in lenses:
                out.append(
                    HardwareRegionDetector._sample_circle(
                        cx, cy, radius * 1.02, samples=48
                    )
                )
        else:
            xs = [c[0] for c in lenses]
            ys = [c[1] for c in lenses]
            rs = [c[2] for c in lenses]
            max_r = float(max(rs))
            # Grow so each lens ring clears; stadium hugs the stack tightly.
            grow = max_r * 1.08
            sx1 = float(min(xs) - grow)
            sy1 = float(min(ys) - grow)
            sx2 = float(max(xs) + grow)
            sy2 = float(max(ys) + grow)
            short = min(sx2 - sx1, sy2 - sy1)
            corner = float(np.clip(short * 0.48, 2.0, short * 0.5 - 0.5))
            stadium = HardwareRegionDetector._sample_rounded_rect(
                sx1, sy1, sx2, sy2, corner, samples_per_corner=16
            )
            if stadium is not None:
                out.append(stadium.reshape(-1, 2))
            else:
                for cx, cy, radius in lenses:
                    out.append(
                        HardwareRegionDetector._sample_circle(
                            cx, cy, radius * 1.06, samples=64
                        )
                    )

        for cx, cy, radius in flashes:
            # Flash / sensor — keep as its own clean circle.
            out.append(
                HardwareRegionDetector._sample_circle(
                    cx, cy, max(radius * 1.08, 3.0), samples=64
                )
            )

        # If Hough missed the flash but the user drew a separate small circle,
        # keep a polished version of any leftover compact part.
        if len(parts) > 1 and not flashes:
            lens_center = np.array(
                [
                    float(np.mean([c[0] for c in lenses])),
                    float(np.mean([c[1] for c in lenses])),
                ],
                dtype=np.float32,
            )
            for pts in parts:
                center = pts.mean(axis=0)
                bw_p = float(pts[:, 0].max() - pts[:, 0].min())
                bh_p = float(pts[:, 1].max() - pts[:, 1].min())
                short_p = min(bw_p, bh_p)
                aspect = max(bw_p, bh_p) / max(short_p, 1.0)
                dist = float(np.linalg.norm(center - lens_center))
                if aspect <= 1.35 and short_p < median_r * 2.4 and dist > median_r * 1.1:
                    polished = HardwareRegionDetector._perfect_one_contour(
                        pts, gray, lock_bounds=False
                    )
                    if polished is not None:
                        out.append(polished.reshape(-1, 2))

        return out

    @staticmethod
    def _perfect_one_contour(
        pts: np.ndarray,
        gray: Optional[np.ndarray],
        *,
        lock_bounds: bool = False,
    ) -> Optional[np.ndarray]:
        """Fit one cutout to a clean circle or stadium (dense, smooth verts)."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return None
        user_x, user_y, user_w, user_h = cv2.boundingRect(pts.astype(np.int32))
        user_box = (
            float(user_x),
            float(user_y),
            float(user_x + user_w),
            float(user_y + user_h),
        )
        kind, params = HardwareRegionDetector._classify_cutout(pts)

        if kind == "circle":
            cx, cy, radius = params
            original_radius = float(radius)
            if not lock_bounds:
                cx, cy, radius = HardwareRegionDetector._refine_circle(
                    gray, float(cx), float(cy), float(radius)
                )
            # Never explode a flash hole into a huge ring.
            if radius > original_radius * 1.15:
                radius = original_radius
            if radius < original_radius * 0.55:
                radius = original_radius
            if lock_bounds:
                # Honour the smaller of enclosing-circle vs user bbox.
                half = 0.5 * min(user_w, user_h)
                radius = min(float(radius), float(half) * 1.02)
                cx = 0.5 * (user_box[0] + user_box[2])
                cy = 0.5 * (user_box[1] + user_box[3])
            return HardwareRegionDetector._sample_circle(
                cx, cy, radius, samples=64
            ).reshape(-1, 2)

        if kind in ("stadium", "rounded_rect"):
            x1, y1, x2, y2, corner = params
            if lock_bounds:
                # Keep the exact size the user dragged to (allow shrink!).
                x1, y1, x2, y2 = user_box
            elif gray is not None and max(x2 - x1, y2 - y1) >= 14:
                x1, y1, x2, y2 = HardwareRegionDetector._snap_box_to_edges(
                    gray, x1, y1, x2, y2
                )
                fitted = HardwareRegionDetector._fit_hardware_box(
                    gray, x1, y1, x2, y2
                )
                if fitted is not None:
                    fx1, fy1, fx2, fy2 = fitted
                    # Stay close to the user's outline — no weird dimension jumps.
                    x1 = float(np.clip(fx1, user_box[0] - 0.06 * user_w, user_box[0] + 0.06 * user_w))
                    y1 = float(np.clip(fy1, user_box[1] - 0.06 * user_h, user_box[1] + 0.06 * user_h))
                    x2 = float(np.clip(fx2, user_box[2] - 0.06 * user_w, user_box[2] + 0.06 * user_w))
                    y2 = float(np.clip(fy2, user_box[3] - 0.06 * user_h, user_box[3] + 0.06 * user_h))
                    # Absolute size guard vs user AABB.
                    if (x2 - x1) > user_w * 1.12:
                        mid = 0.5 * (x1 + x2)
                        x1, x2 = mid - 0.56 * user_w, mid + 0.56 * user_w
                    if (y2 - y1) > user_h * 1.12:
                        mid = 0.5 * (y1 + y2)
                        y1, y2 = mid - 0.56 * user_h, mid + 0.56 * user_h
            short = min(x2 - x1, y2 - y1)
            if kind == "stadium":
                corner = float(np.clip(short * 0.48, 2.0, short * 0.5 - 0.5))
            else:
                corner = float(np.clip(short * 0.36, 6.0, short * 0.48))
            # Tiny AA pad only — never the old 0.8% grow that looked oversized.
            if not lock_bounds:
                gx = max(0.4, (x2 - x1) * 0.004)
                gy = max(0.4, (y2 - y1) * 0.004)
                x1, y1, x2, y2 = x1 - gx, y1 - gy, x2 + gx, y2 + gy
            stadium = HardwareRegionDetector._sample_rounded_rect(
                x1, y1, x2, y2, corner, samples_per_corner=16
            )
            if stadium is not None:
                return stadium
            return HardwareRegionDetector._simplify_editable(pts)

        # Free / irregular — soft stadium from the USER box (no expand).
        contour = pts.reshape(-1, 1, 2).astype(np.float32)
        area = float(cv2.contourArea(contour))
        if area < 8:
            return HardwareRegionDetector._simplify_editable(pts)
        x1, y1, x2, y2 = user_box
        if (not lock_bounds) and gray is not None and max(user_w, user_h) >= 22:
            sx1, sy1, sx2, sy2 = HardwareRegionDetector._snap_box_to_edges(
                gray, x1, y1, x2, y2
            )
            # Only accept snap if it does not enlarge a lot.
            if (sx2 - sx1) * (sy2 - sy1) <= (x2 - x1) * (y2 - y1) * 1.12:
                x1, y1, x2, y2 = sx1, sy1, sx2, sy2
        short = min(x2 - x1, y2 - y1)
        aspect = max(x2 - x1, y2 - y1) / max(short, 1.0)
        if aspect < 1.25 and short >= 28:
            corner = float(np.clip(short * 0.30, 6.0, short * 0.45))
        else:
            corner = float(np.clip(short * 0.45, 3.0, short * 0.5 - 0.5))
        stadium = HardwareRegionDetector._sample_rounded_rect(
            x1, y1, x2, y2, corner, samples_per_corner=16
        )
        if stadium is not None:
            return stadium
        return HardwareRegionDetector._simplify_editable(pts)

    @staticmethod
    def _fit_hardware_box(
        gray: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Fit a tight axis-aligned box to the dark camera-island / button body.

        Uses local adaptive threshold so frosted glass / pink bodies still work.
        """
        h, w = gray.shape[:2]
        bw = max(8.0, x2 - x1)
        bh = max(8.0, y2 - y1)
        pad = max(6, int(round(0.14 * max(bw, bh))))
        rx1 = int(np.clip(x1 - pad, 0, w - 1))
        ry1 = int(np.clip(y1 - pad, 0, h - 1))
        rx2 = int(np.clip(x2 + pad, 0, w))
        ry2 = int(np.clip(y2 + pad, 0, h))
        roi = gray[ry1:ry2, rx1:rx2]
        if roi.size < 64:
            return None
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        # Island is usually darker than surrounding glass.
        thr = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 6,
        )
        thr = cv2.morphologyEx(
            thr, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        thr = cv2.morphologyEx(
            thr, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _ = cv2.findContours(
            thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        # Prefer the blob that best overlaps the user's rough cutout.
        rough = np.zeros_like(thr)
        lx1 = int(np.clip(x1 - rx1, 0, rough.shape[1] - 1))
        ly1 = int(np.clip(y1 - ry1, 0, rough.shape[0] - 1))
        lx2 = int(np.clip(x2 - rx1, 0, rough.shape[1]))
        ly2 = int(np.clip(y2 - ry1, 0, rough.shape[0]))
        cv2.rectangle(rough, (lx1, ly1), (lx2, ly2), 255, -1)
        best = None
        best_score = -1.0
        rough_area = float(max(np.count_nonzero(rough), 1))
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < rough_area * 0.18 or area > rough_area * 2.8:
                continue
            blob = np.zeros_like(thr)
            cv2.drawContours(blob, [contour], -1, 255, -1)
            overlap = float(np.count_nonzero((blob > 0) & (rough > 0)))
            score = overlap / max(area, 1.0) + 0.15 * (overlap / rough_area)
            if score > best_score:
                best_score = score
                best = contour
        if best is None or best_score < 0.25:
            return None
        bx, by, bw2, bh2 = cv2.boundingRect(best)
        # Reject skinny false edges.
        if bw2 < 12 or bh2 < 12:
            return None
        aspect = max(bw2, bh2) / max(min(bw2, bh2), 1)
        if aspect > 3.2:
            return None
        out = (
            float(rx1 + bx),
            float(ry1 + by),
            float(rx1 + bx + bw2),
            float(ry1 + by + bh2),
        )
        # Keep within a reasonable distance of the user's placement.
        lim = 0.20 * max(bw, bh)
        if abs(out[0] - x1) > lim or abs(out[1] - y1) > lim:
            return None
        if abs(out[2] - x2) > lim or abs(out[3] - y2) > lim:
            return None
        return out

    @staticmethod
    def _simplify_editable(pts: np.ndarray, max_verts: int = 12) -> np.ndarray:
        """Keep a short editable polygon (no dense jagged outlines)."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) <= max_verts:
            return pts
        contour = pts.reshape(-1, 1, 2)
        peri = float(cv2.arcLength(contour, True))
        for factor in (0.02, 0.035, 0.05, 0.08):
            approx = cv2.approxPolyDP(contour, factor * peri, True)
            if 4 <= approx.shape[0] <= max_verts:
                return approx.reshape(-1, 2)
        # Even subsample as last resort.
        step = max(1, len(pts) // max_verts)
        return pts[::step][:max_verts].copy()

    @staticmethod
    def _snap_box_to_edges(
        gray: Optional[np.ndarray],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Tuple[float, float, float, float]:
        """Nudge a bbox onto strong local edges for a tighter camera cutout."""
        if gray is None:
            return x1, y1, x2, y2
        h, w = gray.shape[:2]
        bw = max(8.0, x2 - x1)
        bh = max(8.0, y2 - y1)
        pad = max(4, int(round(0.08 * max(bw, bh))))
        rx1 = int(np.clip(x1 - pad, 0, w - 1))
        ry1 = int(np.clip(y1 - pad, 0, h - 1))
        rx2 = int(np.clip(x2 + pad, 0, w))
        ry2 = int(np.clip(y2 + pad, 0, h))
        roi = gray[ry1:ry2, rx1:rx2]
        if roi.size < 64:
            return x1, y1, x2, y2
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blur, 22, 85)
        # Project edge energy onto each side and pick the strongest ridge.
        def _best_line(profile: np.ndarray, fallback: float, origin: float) -> float:
            if profile.size < 3:
                return fallback
            smooth = cv2.GaussianBlur(
                profile.astype(np.float32).reshape(-1, 1), (0, 0), 1.2
            ).ravel()
            idx = int(np.argmax(smooth))
            if float(smooth[idx]) < max(5.0, float(np.median(smooth)) * 1.45):
                return fallback
            return origin + float(idx)

        # Left / right columns.
        col_energy = edges.mean(axis=0)
        mid = max(1, col_energy.shape[0] // 2)
        left = _best_line(col_energy[:mid], x1, float(rx1))
        right_local = _best_line(col_energy[mid:], x2, float(rx1 + mid))
        right = right_local
        if right <= left + 8:
            left, right = x1, x2
        # Top / bottom rows.
        row_energy = edges.mean(axis=1)
        mid_r = max(1, row_energy.shape[0] // 2)
        top = _best_line(row_energy[:mid_r], y1, float(ry1))
        bottom = _best_line(row_energy[mid_r:], y2, float(ry1 + mid_r))
        if bottom <= top + 8:
            top, bottom = y1, y2
        # Allow a bigger snap so rough manual outlines can reach the real rim.
        lim = 0.24 * max(bw, bh)
        left = float(np.clip(left, x1 - lim, x1 + lim))
        right = float(np.clip(right, x2 - lim, x2 + lim))
        top = float(np.clip(top, y1 - lim, y1 + lim))
        bottom = float(np.clip(bottom, y2 - lim, y2 + lim))
        if right <= left + 8 or bottom <= top + 8:
            return x1, y1, x2, y2
        return left, top, right, bottom

    @staticmethod
    def _refine_circle(
        gray: Optional[np.ndarray], cx: float, cy: float, radius: float
    ) -> Tuple[float, float, float]:
        """Tighten a circle using local Hough evidence when possible."""
        if gray is None or radius < 2:
            return cx, cy, radius
        height, width = gray.shape[:2]
        search = max(8, int(radius * 1.8))
        x0 = int(np.clip(cx - search, 0, width - 1))
        y0 = int(np.clip(cy - search, 0, height - 1))
        x1 = int(np.clip(cx + search, 0, width))
        y1 = int(np.clip(cy + search, 0, height))
        roi = gray[y0:y1, x0:x1]
        if roi.size < 16:
            return cx, cy, radius
        blur = cv2.GaussianBlur(roi, (0, 0), 1.0)
        r_min = max(2, int(radius * 0.70))
        r_max = max(r_min + 1, int(radius * 1.35))
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(4, int(radius * 0.8)),
            param1=70,
            param2=14,
            minRadius=r_min,
            maxRadius=r_max,
        )
        if circles is None:
            return cx, cy, radius
        best = None
        best_score = 1e18
        for x, y, r in circles[0]:
            fx, fy = float(x + x0), float(y + y0)
            score = (fx - cx) ** 2 + (fy - cy) ** 2 + 0.35 * (float(r) - radius) ** 2
            if score < best_score:
                best_score = score
                best = (fx, fy, float(r))
        return best if best is not None else (cx, cy, radius)

    @staticmethod
    def _sample_rounded_rect(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
        samples_per_corner: int = 12,
    ) -> Optional[np.ndarray]:
        """
        Analytical rounded-rectangle polygon with dense corner arcs.

        Default 12 samples/corner keeps Perfect Finish overlays round;
        UI may pass 3 for Canva-style 4-handle boxes.
        """
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        radius = float(
            np.clip(radius, 1.0, min((x2 - x1) / 2.0, (y2 - y1) / 2.0) - 0.5)
        )
        n = max(2, int(samples_per_corner))
        pts: List[List[float]] = []

        def _arc(cx: float, cy: float, a0: float, a1: float) -> None:
            for i in range(n + 1):
                t = i / float(n)
                ang = a0 + (a1 - a0) * t
                pts.append(
                    [cx + radius * float(np.cos(ang)), cy + radius * float(np.sin(ang))]
                )

        # Clockwise: top-left arc → top-right → bottom-right → bottom-left.
        _arc(x1 + radius, y1 + radius, np.pi, np.pi * 1.5)
        _arc(x2 - radius, y1 + radius, np.pi * 1.5, np.pi * 2.0)
        _arc(x2 - radius, y2 - radius, 0.0, np.pi * 0.5)
        _arc(x1 + radius, y2 - radius, np.pi * 0.5, np.pi)

        arr = np.asarray(pts, dtype=np.float32)
        # Drop exact duplicate corner joints.
        cleaned = [arr[0]]
        for p in arr[1:]:
            if float(np.linalg.norm(p - cleaned[-1])) > 0.35:
                cleaned.append(p)
        if len(cleaned) >= 2 and float(np.linalg.norm(cleaned[0] - cleaned[-1])) < 0.35:
            cleaned.pop()
        if len(cleaned) < 4:
            return None
        return np.asarray(cleaned, dtype=np.float32)

    @staticmethod
    def _rounded_rectangle(
        mask: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        radius: int,
        *,
        expand_px: float = 2.0,
    ) -> None:
        """Filled rounded rect with sub-pixel AA (no stair-step / spike caps)."""
        HardwareRegionDetector._paint_rounded_rect_aa(
            mask,
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            float(max(0, radius)),
            expand_px=float(expand_px),
        )

    @staticmethod
    def _paint_circle_aa(
        mask: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        *,
        aa: float = 1.75,
        expand_px: float = 1.5,
    ) -> None:
        """Soft-edged filled circle via signed distance (production AA)."""
        if radius < 0.5 or mask.size == 0:
            return
        radius = float(radius) + float(max(0.0, expand_px))
        h, w = mask.shape[:2]
        pad = max(2, int(np.ceil(aa + expand_px + 1.0)))
        ix1 = max(0, int(np.floor(cx - radius)) - pad)
        iy1 = max(0, int(np.floor(cy - radius)) - pad)
        ix2 = min(w, int(np.ceil(cx + radius)) + pad + 1)
        iy2 = min(h, int(np.ceil(cy + radius)) + pad + 1)
        if ix2 <= ix1 or iy2 <= iy1:
            return
        yy, xx = np.mgrid[iy1:iy2, ix1:ix2].astype(np.float32)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - float(radius)
        cover = np.clip(0.5 - dist / max(float(aa), 0.35), 0.0, 1.0)
        patch = (cover * 255.0).astype(np.float32)
        existing = mask[iy1:iy2, ix1:ix2].astype(np.float32)
        mask[iy1:iy2, ix1:ix2] = np.clip(
            np.maximum(existing, patch), 0.0, 255.0
        ).astype(np.uint8)

    @staticmethod
    def _paint_rounded_rect_aa(
        mask: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
        *,
        aa: float = 1.75,
        expand_px: float = 2.0,
    ) -> None:
        """
        Soft-edged stadium / rounded-rect via SDF (Inigo Quilez style).

        Eliminates the triangular chord spikes from sparse fillPoly stadiums
        and the aliased bars of rectangle+circle construction.
        """
        if x2 - x1 < 2.0 or y2 - y1 < 2.0 or mask.size == 0:
            return
        # Grow slightly so side-button ridges on the cover rim stay punched.
        grow = float(max(0.0, expand_px))
        x1, y1, x2, y2 = x1 - grow, y1 - grow, x2 + grow, y2 + grow
        radius = float(
            np.clip(radius, 0.0, min((x2 - x1) * 0.5, (y2 - y1) * 0.5) - 0.25)
        )
        h, w = mask.shape[:2]
        pad = max(2, int(np.ceil(aa + grow + 1.0)))
        ix1 = max(0, int(np.floor(x1)) - pad)
        iy1 = max(0, int(np.floor(y1)) - pad)
        ix2 = min(w, int(np.ceil(x2)) + pad + 1)
        iy2 = min(h, int(np.ceil(y2)) + pad + 1)
        if ix2 <= ix1 or iy2 <= iy1:
            return
        yy, xx = np.mgrid[iy1:iy2, ix1:ix2].astype(np.float32)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        # Half-extents of the inner rectangle (before rounding).
        hx = max(0.0, 0.5 * (x2 - x1) - radius)
        hy = max(0.0, 0.5 * (y2 - y1) - radius)
        qx = np.abs(xx - cx) - hx
        qy = np.abs(yy - cy) - hy
        ox = np.maximum(qx, 0.0)
        oy = np.maximum(qy, 0.0)
        outside = np.sqrt(ox * ox + oy * oy + 1e-12)
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        dist = outside + inside - radius
        cover = np.clip(0.5 - dist / max(float(aa), 0.35), 0.0, 1.0)
        patch = (cover * 255.0).astype(np.float32)
        existing = mask[iy1:iy2, ix1:ix2].astype(np.float32)
        mask[iy1:iy2, ix1:ix2] = np.clip(
            np.maximum(existing, patch), 0.0, 255.0
        ).astype(np.uint8)

    @staticmethod
    def _densify_closed_polyline(
        pts: np.ndarray, *, max_edge: float = 1.25
    ) -> np.ndarray:
        """Insert verts so long edges cannot facet curved cutouts."""
        src = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if src.shape[0] < 3:
            return src
        out: List[np.ndarray] = []
        n = int(src.shape[0])
        step = float(max(0.55, max_edge))
        for i in range(n):
            a = src[i]
            b = src[(i + 1) % n]
            out.append(a)
            dist = float(np.linalg.norm(b - a))
            if dist <= step:
                continue
            segs = int(np.ceil(dist / step))
            for k in range(1, segs):
                t = float(k) / float(segs)
                out.append(a * (1.0 - t) + b * t)
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _fill_polygon_aa(
        mask: np.ndarray,
        poly: np.ndarray,
        *,
        scale: int = 12,
    ) -> None:
        """
        Supersampled polygon fill — exact path, sub-pixel AA, no expand.
        """
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3 or mask.size == 0:
            return
        pts = HardwareRegionDetector._densify_closed_polyline(pts, max_edge=1.15)
        h, w = mask.shape[:2]
        try:
            from .mesh import _fill_closed_polyline_aa

            cov = _fill_closed_polyline_aa(
                pts, (h, w), scale=max(8, min(int(scale), 16)), expand_px=0.0
            )
            if cov is not None and float(np.max(cov)) > 0.05:
                patch = np.clip(cov * 255.0, 0.0, 255.0).astype(np.float32)
                existing = mask.astype(np.float32)
                mask[:] = np.clip(
                    np.maximum(existing, patch), 0.0, 255.0
                ).astype(np.uint8)
                return
        except Exception:
            pass
        s = max(8, min(int(scale), 16))
        x0 = float(pts[:, 0].min())
        y0 = float(pts[:, 1].min())
        x1 = float(pts[:, 0].max())
        y1 = float(pts[:, 1].max())
        pad = 2
        ix0 = max(0, int(np.floor(x0)) - pad)
        iy0 = max(0, int(np.floor(y0)) - pad)
        ix1 = min(w, int(np.ceil(x1)) + pad + 1)
        iy1 = min(h, int(np.ceil(y1)) + pad + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        rw = ix1 - ix0
        rh = iy1 - iy0
        big = np.zeros((rh * s, rw * s), dtype=np.uint8)
        local = (pts - np.array([ix0, iy0], dtype=np.float32)) * float(s)
        local_i = np.round(local).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(big, [local_i], 255)
        small = cv2.resize(big, (rw, rh), interpolation=cv2.INTER_AREA)
        existing = mask[iy0:iy1, ix0:ix1].astype(np.float32)
        mask[iy0:iy1, ix0:ix1] = np.clip(
            np.maximum(existing, small.astype(np.float32)), 0.0, 255.0
        ).astype(np.uint8)

    @staticmethod
    def _contour_matches_aabb(
        pts: np.ndarray, *, tol_frac: float = 0.04
    ) -> bool:
        """True when verts sit on an axis-aligned box (safe for SDF AABB paint)."""
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if p.shape[0] < 4:
            return True
        x1 = float(p[:, 0].min())
        y1 = float(p[:, 1].min())
        x2 = float(p[:, 0].max())
        y2 = float(p[:, 1].max())
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        tol = float(max(0.75, tol_frac * min(bw, bh)))
        on_v = (np.abs(p[:, 0] - x1) <= tol) | (np.abs(p[:, 0] - x2) <= tol)
        on_h = (np.abs(p[:, 1] - y1) <= tol) | (np.abs(p[:, 1] - y2) <= tol)
        return bool(np.mean(on_v | on_h) >= 0.92)

    @staticmethod
    def paint_cutout_mask(
        mask: np.ndarray,
        poly: np.ndarray,
        *,
        analytical: bool = True,
        expand_override: Optional[float] = None,
        geom: Optional[str] = None,
        params: Optional[Tuple[float, ...]] = None,
        force_contour: bool = False,
    ) -> None:
        """
        Rasterise one cutout into ``mask`` with production-smooth edges.

        analytical=True: circle / stadium / rounded-rect via SDF.
        analytical=False / force_contour / geom="contour": supersampled fill
        of the exact polygon (photo-true hardware silhouette).

        When ``geom`` + ``params`` are provided (Phase 3 freeze), paint uses
        those instead of re-classifying the AABB — keeps export contour-true.
        """
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return
        if force_contour or (geom == "contour") or not analytical:
            HardwareRegionDetector._fill_polygon_aa(mask, pts, scale=12)
            return
        # Rotated / non-AABB contours must stay path-true (never axis box).
        if geom in (
            "rectangle",
            "square",
            "rounded_square",
            "rounded_rect",
            "stadium",
        ) and not HardwareRegionDetector._contour_matches_aabb(pts):
            HardwareRegionDetector._fill_polygon_aa(mask, pts, scale=12)
            return
        tight = None if expand_override is None else float(max(0.0, expand_override))
        # Soft sub-pixel AA — scales gently with hole size for clean zoom.
        edge_aa = 1.85 if tight is not None and tight <= 0.5 else 2.05

        kind = geom
        frozen = params
        if kind is None or kind == "" or frozen is None:
            kind, frozen = HardwareRegionDetector._classify_cutout(pts)
        if kind == "free":
            # Photo silhouette that isn't a clean stadium — keep the contour.
            HardwareRegionDetector._fill_polygon_aa(mask, pts, scale=12)
            return

        # Editor rectangle / square → mild-rounded AABB hole (exact box).
        if kind in ("rectangle", "square", "rounded_square"):
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())
            if frozen is not None and len(frozen) >= 5:
                x1, y1, x2, y2 = (
                    float(frozen[0]),
                    float(frozen[1]),
                    float(frozen[2]),
                    float(frozen[3]),
                )
                corner = float(frozen[4])
            else:
                short = min(x2 - x1, y2 - y1)
                if kind == "square":
                    corner = float(np.clip(short * 0.08, 1.5, short * 0.14))
                else:
                    # Rectangle: slight round to match camera-module plates.
                    corner = float(np.clip(short * 0.16, 2.5, short * 0.22))
            expand = 0.0 if tight is None else tight
            HardwareRegionDetector._paint_rounded_rect_aa(
                mask,
                x1,
                y1,
                x2,
                y2,
                corner,
                aa=edge_aa,
                expand_px=expand,
            )
            return

        if kind == "circle":
            # Prefer frozen (cx,cy,r). If params look like a box freeze
            # (x1,y1,x2,...) or are missing, rebuild a true disk from verts.
            cx = cy = radius = -1.0
            if frozen and len(frozen) >= 3:
                a0, a1, a2 = float(frozen[0]), float(frozen[1]), float(frozen[2])
                # Box params: x2 > x1 typically and "radius" ≫ half-span.
                looks_box_params = (
                    len(frozen) >= 5
                    and float(frozen[2]) > float(frozen[0]) + 1.0
                    and float(frozen[3]) > float(frozen[1]) + 1.0
                )
                if not looks_box_params and a2 > 0.5:
                    cx, cy, radius = a0, a1, a2
            if radius < 0.5 or geom is None:
                cx2, cy2, r2 = HardwareRegionDetector._circle_params_from_pts(pts)
                if r2 > 0.5:
                    cx, cy, radius = cx2, cy2, r2
            if radius >= 0.5:
                HardwareRegionDetector._paint_circle_aa(
                    mask, float(cx), float(cy), float(radius),
                    aa=max(edge_aa, 1.85),
                    expand_px=0.0 if tight is None else tight,
                )
                return
            HardwareRegionDetector._fill_polygon_aa(mask, pts, scale=12)
            return
        if kind in ("stadium", "rounded_rect"):
            if frozen is not None and len(frozen) >= 5:
                x1, y1, x2, y2, corner = (
                    float(frozen[0]),
                    float(frozen[1]),
                    float(frozen[2]),
                    float(frozen[3]),
                    float(frozen[4]),
                )
            else:
                x1 = float(pts[:, 0].min())
                y1 = float(pts[:, 1].min())
                x2 = float(pts[:, 0].max())
                y2 = float(pts[:, 1].max())
                short = min(x2 - x1, y2 - y1)
                if kind != "stadium":
                    corner = float(np.clip(short * 0.16, 2.5, short * 0.22))
                else:
                    corner = float(
                        np.clip(short * 0.48, 2.0, short * 0.5 - 0.5)
                    )
            short = min(x2 - x1, y2 - y1)
            long_side = max(x2 - x1, y2 - y1)
            aspect = long_side / max(short, 1.0)
            mh, mw = mask.shape[:2]
            near_side = x1 < mw * 0.14 or x2 > mw * 0.86
            if kind == "stadium":
                if corner <= 0:
                    corner = float(
                        np.clip(short * 0.48, 2.0, max(2.0, short * 0.5 - 0.5))
                    )
                # Camera/flash holes: exact path. Side buttons may still expand.
                expand = 0.0 if tight is None else tight
                if tight is None and near_side:
                    expand = 0.0
            else:
                if corner <= 0:
                    corner = float(np.clip(short * 0.16, 2.5, short * 0.22))
                expand = 0.0 if tight is None else tight
            if tight is not None:
                expand = tight
            HardwareRegionDetector._paint_rounded_rect_aa(
                mask,
                x1,
                y1,
                x2,
                y2,
                corner,
                aa=edge_aa,
                expand_px=expand,
            )
            return
        # Fallback: mild rounded rect from AABB (camera boxes — not stadium).
        x1 = float(pts[:, 0].min())
        y1 = float(pts[:, 1].min())
        x2 = float(pts[:, 0].max())
        y2 = float(pts[:, 1].max())
        short = min(x2 - x1, y2 - y1)
        if short < 2.0:
            HardwareRegionDetector._fill_polygon_aa(mask, pts, scale=12)
            return
        long_side = max(x2 - x1, y2 - y1)
        aspect = long_side / max(short, 1.0)
        mh, mw = mask.shape[:2]
        near_side = x1 < mw * 0.14 or x2 > mw * 0.86
        skinny = short <= max(14.0, long_side * 0.38)
        if near_side and skinny and aspect >= 2.0:
            corner = float(np.clip(short * 0.48, 3.0, short * 0.5 - 0.5))
            expand = 0.0 if tight is None else tight
        else:
            corner = float(np.clip(short * 0.16, 2.5, short * 0.22))
            expand = 0.0 if tight is None else tight
        if tight is not None:
            expand = tight
        HardwareRegionDetector._paint_rounded_rect_aa(
            mask,
            x1,
            y1,
            x2,
            y2,
            corner,
            aa=edge_aa,
            expand_px=expand,
        )

    @staticmethod
    def paint_from_cutout_spec(
        mask: np.ndarray,
        spec: "CutoutSpec",
        width: int,
        height: int,
    ) -> None:
        """Paint one Phase 3 CutoutSpec into an exclusion mask."""
        pts = spec.pixel_contour(width, height)
        if pts.shape[0] < 3:
            return
        expand = spec.resolved_expand()
        geom = spec.geom or None
        params = tuple(spec.params) if spec.params else None
        tag = str(getattr(spec, "shape_tag", "") or "").lower().strip()
        # Locked editor shapes must never be stolen by flash/disk heuristics.
        locked_box = geom in (
            "stadium",
            "rounded_rect",
            "rectangle",
            "square",
            "rounded_square",
        ) or tag in (
            "capsule",
            "button",
            "pill_h",
            "pill_v",
            "rectangle",
            "rounded_rect",
            "rounded_square",
            "square",
            "oval",
        )
        # Exact-path tools: always paint the editable polyline.
        exact_path = geom == "contour" or tag in (
            "squircle",
            "superellipse",
            "polygon",
            "triangle",
            "custom_path",
            "free",
            "diamond",
        )
        if exact_path:
            HardwareRegionDetector.paint_cutout_mask(
                mask,
                pts,
                analytical=False,
                expand_override=0.0 if expand is None else float(max(0.0, expand)),
                force_contour=True,
            )
            return
        # Explicit circle tool OR flash disks only. Camera modules must never
        # be forced to a disk from AABB / looks_like_true_disk heuristics.
        force_disk = (not locked_box) and (
            tag == "circle"
            or (spec.kind == "flash" and (geom == "circle" or tag in ("", "circle")))
        )
        if tag == "circle":
            # Always paint the tool circle from AABB (or frozen cx,cy,r).
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            radius = 0.5 * min(x2 - x1, y2 - y1)
            if params and len(params) >= 3 and len(params) < 5:
                p0, p1, p2 = float(params[0]), float(params[1]), float(params[2])
                if p2 > 0.5:
                    cx, cy, radius = p0, p1, p2
            if radius >= 0.5:
                HardwareRegionDetector._paint_circle_aa(
                    mask,
                    float(cx),
                    float(cy),
                    float(radius),
                    aa=1.85,
                    expand_px=float(expand) if expand is not None and expand >= 0 else 0.0,
                )
                return
        if (
            not force_disk
            and spec.kind in ("camera", "other")
            and geom == "circle"
            and tag not in ("circle",)
        ):
            # Demote frozen camera circles (legacy / mis-classify) to the
            # selection AABB rounded-rect so the user shape stays a module hole.
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())
            short = min(x2 - x1, y2 - y1)
            corner = float(np.clip(short * 0.16, 3.0, short * 0.22))
            geom = "rounded_rect"
            params = (x1, y1, x2, y2, corner)
        if force_disk and geom != "contour" and tag != "circle":
            cx, cy, radius = HardwareRegionDetector._circle_params_from_pts(pts)
            if params and len(params) >= 3 and len(params) < 5:
                # Trust frozen circle params when they are (cx,cy,r).
                p0, p1, p2 = float(params[0]), float(params[1]), float(params[2])
                if p2 > 0.5:
                    cx, cy, radius = p0, p1, p2
            if radius >= 0.5:
                HardwareRegionDetector._paint_circle_aa(
                    mask,
                    float(cx),
                    float(cy),
                    float(radius),
                    aa=1.85,
                    expand_px=float(expand) if expand is not None and expand >= 0 else 0.0,
                )
                return
            geom = "circle"
            params = (float(cx), float(cy), float(max(radius, 1.0)))
        # Authoritative contour holes (photo silhouette) → polygon AA.
        if spec.authoritative and geom == "contour":
            HardwareRegionDetector.paint_cutout_mask(
                mask,
                pts,
                analytical=False,
                expand_override=expand,
                force_contour=True,
            )
            return
        # Always honor frozen geom/params when present — live rectangle edits
        # used to drop geom and reclassify into heavy stadiums, leaving wrap
        # on the camera plate inside the user's red selection box.
        HardwareRegionDetector.paint_cutout_mask(
            mask,
            pts,
            analytical=True,
            expand_override=expand,
            geom=geom,
            params=params,
        )

    @staticmethod
    def paint_exclusion_from_specs(
        specs: List["CutoutSpec"],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Build a full exclusion mask from authoritative CutoutSpecs."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for spec in specs:
            kind = str(getattr(spec, "kind", "") or "")
            # Side keys are wrapped on a separate mask, not punched as holes.
            if kind == "button":
                continue
            HardwareRegionDetector.paint_from_cutout_spec(
                mask, spec, width, height
            )
        return mask

    @staticmethod
    def extract_photo_silhouette(
        gray: np.ndarray,
        seed_pts: np.ndarray,
        *,
        pad_frac: float = 0.18,
    ) -> Optional[np.ndarray]:
        """
        Trace the real hardware outline in the photo near ``seed_pts``.

        Used for camera islands that aren't a clean stadium — returns a dense
        smoothed contour in image pixels, or None when detection is weak.
        """
        if gray is None or gray.size == 0:
            return None
        pts = np.asarray(seed_pts, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            return None
        h, w = gray.shape[:2]
        x1 = float(pts[:, 0].min())
        y1 = float(pts[:, 1].min())
        x2 = float(pts[:, 0].max())
        y2 = float(pts[:, 1].max())
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        pad = max(6.0, min(bw, bh) * pad_frac)
        ix0 = int(np.clip(x1 - pad, 0, w - 1))
        iy0 = int(np.clip(y1 - pad, 0, h - 1))
        ix1 = int(np.clip(x2 + pad, 0, w))
        iy1 = int(np.clip(y2 + pad, 0, h))
        if ix1 - ix0 < 12 or iy1 - iy0 < 12:
            return None

        roi = gray[iy0:iy1, ix0:ix1]
        blur = cv2.GaussianBlur(roi, (0, 0), 1.1)
        # Dark hardware on lighter back — or inverse for silvery modules.
        thr_dark = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 4,
        )
        thr_light = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 4,
        )
        edges = cv2.Canny(blur, 40, 120)
        # Seed from the user's polygon filled in ROI coords.
        seed = np.zeros_like(thr_dark)
        local = pts.copy()
        local[:, 0] -= float(ix0)
        local[:, 1] -= float(iy0)
        cv2.fillPoly(
            seed,
            [np.round(local).astype(np.int32).reshape(-1, 1, 2)],
            255,
        )
        seed_area = float(np.count_nonzero(seed))
        if seed_area < 16:
            return None

        best = None
        best_score = 0.0
        for binary in (thr_dark, thr_light, edges):
            work = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            # Keep components that overlap the seed.
            overlap = cv2.bitwise_and(work, seed)
            if np.count_nonzero(overlap) < seed_area * 0.15:
                # Dilate edges so thin rings still catch the seed.
                work = cv2.dilate(
                    work,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
                overlap = cv2.bitwise_and(work, seed)
            if np.count_nonzero(overlap) < 8:
                continue
            # Flood from seed into work.
            flood = work.copy()
            flood[seed > 0] = 255
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                (flood > 0).astype(np.uint8), connectivity=8
            )
            for lab in range(1, n_labels):
                area = float(stats[lab, cv2.CC_STAT_AREA])
                if area < seed_area * 0.35 or area > seed_area * 3.8:
                    continue
                comp = (labels == lab).astype(np.uint8) * 255
                inter = float(np.count_nonzero(cv2.bitwise_and(comp, seed)))
                score = inter / max(area, 1.0)
                if score > best_score and inter >= seed_area * 0.25:
                    best_score = score
                    best = comp

        if best is None or best_score < 0.18:
            return None

        contours, _ = cv2.findContours(
            best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        outer = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32
        )
        if outer.shape[0] < 8:
            return None
        # Smooth lightly, keep density for polygon AA.
        if outer.shape[0] >= 16:
            from .mesh import AdaptiveMeshBuilder
            outer = AdaptiveMeshBuilder._smooth_closed_polyline(
                outer,
                window=max(5, min(15, (outer.shape[0] // 30) * 2 + 1)),
            )
        outer[:, 0] += float(ix0)
        outer[:, 1] += float(iy0)
        # Reject silhouettes that balloon far past the user's selection —
        # blank / low-contrast ROIs used to invent giant fake islands.
        ux1, uy1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        ux2, uy2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        uw, uh = max(ux2 - ux1, 1.0), max(uy2 - uy1, 1.0)
        sx1, sy1 = float(outer[:, 0].min()), float(outer[:, 1].min())
        sx2, sy2 = float(outer[:, 0].max()), float(outer[:, 1].max())
        sw, sh = max(sx2 - sx1, 1.0), max(sy2 - sy1, 1.0)
        if sw > uw * 1.28 or sh > uh * 1.28:
            return None
        if sw * sh > uw * uh * 1.45:
            return None
        # Must still overlap the seed box substantially.
        ox1, oy1 = max(ux1, sx1), max(uy1, sy1)
        ox2, oy2 = min(ux2, sx2), min(uy2, sy2)
        if ox2 <= ox1 or oy2 <= oy1:
            return None
        inter = (ox2 - ox1) * (oy2 - oy1)
        if inter < uw * uh * 0.35:
            return None
        return outer

    @staticmethod
    def freeze_cutout_spec(
        pts: np.ndarray,
        *,
        kind: str = "other",
        gray: Optional[np.ndarray] = None,
        width: int = 1,
        height: int = 1,
    ) -> "CutoutSpec":
        """
        Freeze geom + params for an authoritative Phase 3 hole.

        Circles / stadiums / rounded-rects keep SDF painters.
        Irregular camera islands snap to a photo silhouette (``geom=contour``).
        """
        from .device_template import CutoutSpec

        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        w = max(float(width), 1.0)
        h = max(float(height), 1.0)
        expand = 0.0 if kind in ("camera", "flash") else (
            2.05 if kind == "button" else -1.0
        )

        if kind == "flash":
            cx, cy, radius = HardwareRegionDetector._fit_circle_least_squares(pts)
            if radius < 1.0:
                cx = 0.5 * (float(pts[:, 0].min()) + float(pts[:, 0].max()))
                cy = 0.5 * (float(pts[:, 1].min()) + float(pts[:, 1].max()))
                radius = 0.25 * (
                    float(pts[:, 0].max() - pts[:, 0].min())
                    + float(pts[:, 1].max() - pts[:, 1].min())
                )
            radius = float(
                np.clip(radius, 1.0, max(width, height) * 0.075)
            )
            edit_pts = HardwareRegionDetector._sample_circle(
                float(cx), float(cy), radius, samples=64
            )
            edit_arr = np.asarray(edit_pts, dtype=np.float32).reshape(-1, 2)
            norm = [
                [float(x / w), float(y / h)] for x, y in edit_arr
            ]
            return CutoutSpec(
                kind="flash",
                contour=norm,
                geom="circle",
                params=[float(cx), float(cy), float(radius)],
                expand_px=0.0,
                authoritative=True,
            )

        # Camera / module islands: NEVER freeze a selection box as a circle.
        # Min-enclosing-circle of a Redmi AABB painted a giant gray disk over
        # the whole module and ate the surrounding wrap. User shape tags are
        # applied earlier; this path is auto / untagged only.
        if kind in ("camera", "other") and gray is not None:
            if HardwareRegionDetector._looks_like_true_disk(pts):
                cx, cy, radius = HardwareRegionDetector._circle_params_from_pts(
                    pts
                )
                # Only keep true small flash/lens disks — never module AABBs.
                short_img = float(min(width, height))
                if radius > 0.5 and radius <= short_img * 0.055:
                    edit = HardwareRegionDetector._sample_circle(
                        float(cx), float(cy), float(radius), samples=48
                    )
                    edit_arr = np.asarray(edit, dtype=np.float32).reshape(-1, 2)
                    norm = [
                        [float(x / w), float(y / h)] for x, y in edit_arr
                    ]
                    return CutoutSpec(
                        kind=kind if kind != "other" else "camera",
                        contour=norm,
                        geom="circle",
                        params=[float(cx), float(cy), float(radius)],
                        expand_px=0.0,
                        authoritative=True,
                    )
                # Large "disk" on a camera selection → rounded module hole.
                x1 = float(pts[:, 0].min())
                y1 = float(pts[:, 1].min())
                x2 = float(pts[:, 0].max())
                y2 = float(pts[:, 1].max())
                short = min(x2 - x1, y2 - y1)
                corner = float(np.clip(short * 0.16, 3.0, short * 0.22))
                edit = HardwareRegionDetector._sample_rounded_rect(
                    x1, y1, x2, y2, corner, samples_per_corner=14
                )
                edit_arr = (
                    np.asarray(edit, dtype=np.float32).reshape(-1, 2)
                    if edit is not None
                    else pts
                )
                norm = [
                    [float(x / w), float(y / h)] for x, y in edit_arr
                ]
                return CutoutSpec(
                    kind=kind if kind != "other" else "camera",
                    contour=norm,
                    geom="rounded_rect",
                    params=[x1, y1, x2, y2, corner],
                    expand_px=0.0,
                    authoritative=True,
                    shape_tag="rounded_rect",
                )
            sil = HardwareRegionDetector.extract_photo_silhouette(gray, pts)
            if sil is not None and sil.shape[0] >= 8:
                # Detected camera islands must keep the photo contour. A mild
                # AABB rounded-rect (straight vertical/horizontal stop) left a
                # silver gap around Samsung-style pills. User rectangle tools
                # never reach this branch (tagged_geom is applied first).
                step = max(1, len(sil) // 72)
                approx = sil[::step]
                if approx.shape[0] < 16:
                    approx = sil
                norm = [[float(x / w), float(y / h)] for x, y in approx]
                return CutoutSpec(
                    kind=kind if kind != "other" else "camera",
                    contour=norm,
                    geom="contour",
                    params=[],
                    expand_px=0.0,
                    authoritative=True,
                )

        # Analytical shapes — polish then freeze.
        # Camera islands: NEVER promote a selection box to a circle (that made
        # the giant gray semicircle over Redmi modules). Prefer user AABB
        # rounded-rect / photo contour.
        polished = HardwareRegionDetector._perfect_one_contour(
            pts, gray, lock_bounds=True
        )
        use = polished if polished is not None and len(polished) >= 4 else pts
        geom, params = HardwareRegionDetector._classify_cutout(use)
        if kind == "camera" and geom == "circle" and params:
            # Demote only square AABB promotions — never true disks (flash
            # mislabeled as camera, or large hi-res lens holes).
            bw = float(use[:, 0].max() - use[:, 0].min())
            bh = float(use[:, 1].max() - use[:, 1].min())
            short = min(bw, bh)
            n_verts = int(use.shape[0])
            bbox_fill = float(cv2.contourArea(use.reshape(-1, 1, 2))) / max(
                bw * bh, 1.0
            )
            true_disk = HardwareRegionDetector._looks_like_true_disk(use)
            fake_circle = (not true_disk) and (
                (n_verts <= 8 and bbox_fill >= 0.82)
            )
            if fake_circle:
                x1 = float(use[:, 0].min())
                y1 = float(use[:, 1].min())
                x2 = float(use[:, 0].max())
                y2 = float(use[:, 1].max())
                corner = float(np.clip(short * 0.16, 3.0, short * 0.22))
                geom = "rounded_rect"
                params = (x1, y1, x2, y2, corner)
            elif true_disk:
                cx, cy, radius = HardwareRegionDetector._circle_params_from_pts(
                    use
                )
                geom = "circle"
                params = (float(cx), float(cy), float(radius))
        if geom == "free":
            geom = "contour"
            params = ()
            edit_pts = use
        elif kind == "camera" and geom in ("rounded_rect", "rectangle", "square"):
            # Auto camera without a photo silhouette still must not become an
            # AABB hole (straight stop + silver gap around the island).
            geom = "contour"
            params = ()
            edit_pts = use
        elif geom == "circle" and params:
            edit_pts = HardwareRegionDetector._sample_circle(
                float(params[0]), float(params[1]), float(params[2]), samples=64
            )
        elif geom in ("stadium", "rounded_rect") and len(params) >= 5:
            sampled = HardwareRegionDetector._sample_rounded_rect(
                float(params[0]),
                float(params[1]),
                float(params[2]),
                float(params[3]),
                float(params[4]),
                samples_per_corner=16,
            )
            edit_pts = sampled if sampled is not None else use
        else:
            edit_pts = use
            geom = "contour"
            params = ()

        if kind == "flash" and geom != "circle":
            # Flash must stay a perfect round — force disk from selection.
            cx, cy, radius = HardwareRegionDetector._circle_params_from_pts(pts)
            # Reject absurd flash radii (would paint a module-sized circle).
            if radius > 0.5 and radius <= max(width, height) * 0.12:
                geom = "circle"
                params = (float(cx), float(cy), float(radius))
                edit_pts = HardwareRegionDetector._sample_circle(
                    cx, cy, radius, samples=72
                )
            else:
                g2, p2 = HardwareRegionDetector._classify_cutout(pts)
                if g2 == "circle":
                    geom, params = g2, p2
                    edit_pts = HardwareRegionDetector._sample_circle(
                        float(p2[0]), float(p2[1]), float(p2[2]), samples=72
                    )

        expand = 0.0 if kind in ("camera", "flash") else -1.0
        if kind == "button":
            bw = float(pts[:, 0].max() - pts[:, 0].min())
            bh = float(pts[:, 1].max() - pts[:, 1].min())
            aspect = max(bw, bh) / max(min(bw, bh), 1.0)
            # Tall volume → tight wrap; compact FP/power → visible cutout.
            expand = 1.25 if aspect >= 2.2 else 2.35
        edit_arr = np.asarray(edit_pts, dtype=np.float32).reshape(-1, 2)
        norm = [
            [float(float(x) / w), float(float(y) / h)]
            for x, y in edit_arr
        ]
        return CutoutSpec(
            kind=kind,
            contour=norm,
            geom=geom,
            params=[float(p) for p in params],
            expand_px=expand,
            authoritative=True,
        )
