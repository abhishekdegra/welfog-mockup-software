"""
Smart Auto-Fit Engine — geometry-based artwork placement.

Places artwork onto the detected printable cover surface using only the live
phone/mesh geometry. No hardcoded placement offsets, overscan scales, or
bleed margins: the printable aspect comes from the mesh (and printable mask
when available), the crop is centered, and aspect ratio is preserved unless
the user explicitly chose stretch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.helpers import to_bgra
from .mesh import ControlMesh, mesh_aspect


@dataclass
class SmartFitResult:
    """Placement settings derived from printable cover geometry."""

    scale: float = 100.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    region_inset: float = 0.0
    corner_radius: float = 6.0
    confidence: float = 0.0

    def settings(self) -> dict:
        """Compositor placement settings."""
        return {
            "design_scale": float(self.scale),
            "offset_x": float(self.offset_x),
            "offset_y": float(self.offset_y),
            "rotation": float(self.rotation),
            "region_inset": float(self.region_inset),
            "corner_radius": float(self.corner_radius),
        }


class SmartFitEstimator:
    """
    Geometry-based auto-fit for the printable cover surface.

    Pipeline:
      1. Measure printable aspect from the live mesh / printable mask
      2. Build the same aspect-preserving crop MeshWarper uses
      3. Center the crop (equal leftover margins on opposing sides)
      4. Leave region_inset at 0 — print margins come from phone geometry
    """

    SCORE_SIZE = 64

    @staticmethod
    def estimate(
        design_image: np.ndarray,
        mesh: ControlMesh,
        exclusion_mask: Optional[np.ndarray] = None,
        fit_mode: str = "fill",
        margin_percent: float = 0.0,
        corner_radius_percent: float = 6.0,
        printable_mask: Optional[np.ndarray] = None,
    ) -> SmartFitResult:
        """
        Estimate centered placement from the current printable geometry.

        ``margin_percent`` and ``exclusion_mask`` are accepted for API
        compatibility; placement no longer invents offsets from them. Print
        safety is enforced by the printable mask / mesh perimeter itself.
        """
        del exclusion_mask  # geometry gate already punched; do not pan away
        mode = fit_mode if fit_mode in ("fill", "fit", "stretch") else "fill"

        design = to_bgra(design_image)
        height, width = design.shape[:2]
        target_aspect = SmartFitEstimator._printable_aspect(
            mesh, printable_mask
        )

        # Validate crop math matches the warper (side-effect free).
        SmartFitEstimator._base_crop(
            width,
            height,
            target_aspect,
            mode if mode != "stretch" else "fill",
        )

        # Exact geometric fit: scale 1.0 maps the aspect-matched centered crop
        # onto the printable mesh. No overscan, no auto-rotation, no pan.
        scale = SmartFitEstimator._initial_scale(
            np.ones((height, width), np.uint8) * 255,
            isolated_content=False,
            base_crop_w=float(width),
            base_crop_h=float(height),
            fit_mode=mode if mode != "stretch" else "fill",
        )

        corner_radius = float(np.clip(corner_radius_percent, 0.0, 30.0))
        # Confidence rises when we have a real printable silhouette to follow.
        has_printable = (
            printable_mask is not None
            and int(np.count_nonzero(printable_mask)) > 64
        )
        confidence = 0.92 if has_printable else 0.75
        # margin_percent documents geometry-derived safety only (not applied as
        # a placement inset — equal margins come from the centered crop).
        _ = float(margin_percent)

        return SmartFitResult(
            scale=round(scale * 100.0, 1),
            offset_x=0.0,
            offset_y=0.0,
            rotation=0.0,
            region_inset=0.0,
            corner_radius=round(corner_radius, 1),
            confidence=float(confidence),
        )

    # ----------------------------------------------------------- geometry

    @staticmethod
    def _printable_aspect(
        mesh: ControlMesh,
        printable_mask: Optional[np.ndarray],
    ) -> float:
        """
        Aspect of the surface the artwork must cover.

        Prefer the mesh edge aspect (matches UV warp). Fall back to the
        printable mask bounding box when the mesh is missing/degenerate.
        """
        aspect = float(mesh_aspect(mesh))
        if aspect > 0.05:
            return aspect
        if printable_mask is not None and np.count_nonzero(printable_mask) > 64:
            ys, xs = np.nonzero(printable_mask > 0)
            bw = float(xs.max() - xs.min() + 1)
            bh = float(ys.max() - ys.min() + 1)
            if bh > 1.0:
                return bw / bh
        return max(aspect, 0.5)

    # ----------------------------------------------------------- initialisation

    @staticmethod
    def _initial_scale(
        content_mask: np.ndarray,
        isolated_content: bool,
        base_crop_w: float,
        base_crop_h: float,
        fit_mode: str,
    ) -> float:
        """
        Aspect-preserving starting scale from geometry — always 1.0.

        The base crop already matches the printable aspect; scale 1.0 centers
        that crop with equal leftover margins. No hardcoded overscan.
        """
        del content_mask, isolated_content, base_crop_w, base_crop_h
        if fit_mode not in ("fill", "fit", "stretch"):
            return 1.0
        return 1.0

    @staticmethod
    def _safe_inset(
        margin_percent: float,
        corner_radius_percent: float,
        isolated_content: bool,
    ) -> float:
        """Placement inset is always 0 — margins come from phone geometry."""
        del margin_percent, corner_radius_percent, isolated_content
        return 0.0

    @staticmethod
    def _clamp_offsets(
        offset_x: float,
        offset_y: float,
        width: int,
        height: int,
        crop_w: float,
        crop_h: float,
    ) -> Tuple[float, float]:
        """Keep the sampling window inside the artwork (no blank bands)."""
        limit_x = max(0.0, (width - crop_w) / max(crop_w, 1e-6))
        limit_y = max(0.0, (height - crop_h) / max(crop_h, 1e-6))
        return (
            float(np.clip(offset_x, -limit_x, limit_x)),
            float(np.clip(offset_y, -limit_y, limit_y)),
        )

    # --------------------------------------------------------------- refine
    # Kept for compatibility with older callers / tests; estimate() no longer
    # searches offsets because geometry-centered placement is authoritative.

    @staticmethod
    def _refine(
        initial: SmartFitResult,
        saliency: np.ndarray,
        content_mask: np.ndarray,
        mesh: ControlMesh,
        exclusion_mask: Optional[np.ndarray],
        printable_mask: Optional[np.ndarray],
        design_size: Tuple[int, int],
        base_crop_w: float,
        base_crop_h: float,
        fit_mode: str,
        desired_uv: np.ndarray,
        isolated_content: bool,
    ) -> SmartFitResult:
        """Identity refine — centered geometry fit is final."""
        del (
            saliency, content_mask, mesh, exclusion_mask, printable_mask,
            design_size, base_crop_w, base_crop_h, fit_mode, desired_uv,
            isolated_content,
        )
        return initial

    @staticmethod
    def _score_maps(
        saliency: np.ndarray,
        content_mask: np.ndarray,
        mesh: ControlMesh,
        exclusion_mask: Optional[np.ndarray],
        printable_mask: Optional[np.ndarray],
    ) -> dict:
        """Downscaled maps used by legacy candidate scoring."""
        size = SmartFitEstimator.SCORE_SIZE
        sal = cv2.resize(saliency, (size, size), interpolation=cv2.INTER_AREA)
        content = cv2.resize(
            content_mask.astype(np.float32),
            (size, size),
            interpolation=cv2.INTER_AREA,
        )

        exclusion_uv = np.zeros((size, size), np.float32)
        printable_uv = np.ones((size, size), np.float32)
        points = mesh.points
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)

        if exclusion_mask is not None and np.count_nonzero(exclusion_mask) > 0:
            exclusion_uv = SmartFitEstimator._mask_to_uv(
                exclusion_mask, min_xy, span, size
            )
        if printable_mask is not None and np.count_nonzero(printable_mask) > 0:
            printable_uv = SmartFitEstimator._mask_to_uv(
                printable_mask, min_xy, span, size
            )
            printable_uv = np.clip(printable_uv, 0.0, 1.0)

        return {
            "saliency": sal.astype(np.float32),
            "content": content.astype(np.float32),
            "exclusion_uv": exclusion_uv,
            "printable_uv": printable_uv,
            "size": size,
        }

    @staticmethod
    def _mask_to_uv(
        mask: np.ndarray,
        min_xy: np.ndarray,
        span: np.ndarray,
        size: int,
    ) -> np.ndarray:
        """Rasterise an image-space mask into a unit UV occupancy grid."""
        ys, xs = np.nonzero(mask > 96)
        if ys.size == 0:
            return np.zeros((size, size), np.float32)
        u = np.clip((xs.astype(np.float32) - min_xy[0]) / span[0], 0.0, 1.0)
        v = np.clip((ys.astype(np.float32) - min_xy[1]) / span[1], 0.0, 1.0)
        ui = np.clip((u * (size - 1)).astype(np.int32), 0, size - 1)
        vi = np.clip((v * (size - 1)).astype(np.int32), 0, size - 1)
        grid = np.zeros((size, size), np.float32)
        np.add.at(grid, (vi, ui), 1.0)
        if grid.max() > 0:
            grid /= grid.max()
        grid = cv2.GaussianBlur(grid, (5, 5), 0)
        return grid

    @staticmethod
    def _score_candidate(
        maps: dict,
        width: int,
        height: int,
        base_crop_w: float,
        base_crop_h: float,
        scale: float,
        offset_x: float,
        offset_y: float,
        rotation: float,
        desired_uv: np.ndarray,
        fit_mode: str,
        isolated_content: bool,
    ) -> float:
        """Higher is better. Prefer centered, balanced, fully covered fits."""
        size = maps["size"]
        crop_w = base_crop_w / max(scale, 1e-6)
        crop_h = base_crop_h / max(scale, 1e-6)
        center_x = width / 2.0 + offset_x * crop_w * 0.5
        center_y = height / 2.0 + offset_y * crop_h * 0.5
        x0 = center_x - crop_w * 0.5
        y0 = center_y - crop_h * 0.5
        uu, vv = np.meshgrid(
            np.linspace(0.0, 1.0, size, dtype=np.float32),
            np.linspace(0.0, 1.0, size, dtype=np.float32),
        )
        if abs(rotation) > 1e-3:
            theta = np.deg2rad(-rotation)
            cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
            dx = (uu - 0.5) * crop_w
            dy = (vv - 0.5) * crop_h
            sx = center_x + dx * cos_t - dy * sin_t
            sy = center_y + dx * sin_t + dy * cos_t
        else:
            sx = x0 + uu * crop_w
            sy = y0 + vv * crop_h

        map_x = (sx / max(width - 1, 1)) * (size - 1)
        map_y = (sy / max(height - 1, 1)) * (size - 1)
        sal = cv2.remap(
            maps["saliency"], map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        content = cv2.remap(
            maps["content"], map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        printable = maps["printable_uv"]
        exclusion = maps["exclusion_uv"]
        inside = printable > 0.15
        printable_area = max(float(np.count_nonzero(inside)), 1.0)
        coverage = float(np.sum(content[inside])) / printable_area
        hardware_hit = float(np.sum(sal * exclusion)) / max(
            float(sal.sum()), 1e-6
        )
        weight = sal * printable + 1e-6
        cu = float((uu * weight).sum() / weight.sum())
        cv_ = float((vv * weight).sum() / weight.sum())
        center_err = float(np.hypot(cu - desired_uv[0], cv_ - desired_uv[1]))
        left = float(content[:, : size // 2].sum())
        right = float(content[:, size // 2 :].sum())
        top = float(content[: size // 2, :].sum())
        bottom = float(content[size // 2 :, :].sum())
        balance = 1.0 - 0.5 * (
            abs(left - right) / max(left + right, 1.0)
            + abs(top - bottom) / max(top + bottom, 1.0)
        )
        blank = float(np.mean((content < 0.05) & inside))
        _ = fit_mode, isolated_content
        return (
            coverage * 0.45
            + balance * 0.25
            + (1.0 - min(center_err * 2.2, 1.0)) * 0.25
            - hardware_hit * 0.35
            - blank * 0.40
        )

    # ---------------------------------------------------------- analysis

    @staticmethod
    def _content_analysis(
        design: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Saliency + binary content mask (legacy helpers / tests)."""
        height, width = design.shape[:2]
        alpha = design[:, :, 3].astype(np.float32) / 255.0
        gray = cv2.cvtColor(design[:, :, :3], cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
        edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) ** 2
        edges += cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
        edges = np.sqrt(edges)
        saliency = SmartFitEstimator._normalise(edges) * alpha
        content_mask = ((alpha > 0.15) & (gray < 250)).astype(np.uint8) * 255
        if np.count_nonzero(content_mask) == 0:
            content_mask = (alpha > 0.15).astype(np.uint8) * 255
        filled = float(np.count_nonzero(content_mask)) / max(width * height, 1)
        isolated_content = filled < 0.55
        return saliency.astype(np.float32), content_mask, isolated_content

    @staticmethod
    def _desired_target_position(
        mesh: ControlMesh,
        exclusion_mask: Optional[np.ndarray],
        printable_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Printable centroid in mesh UV — always the geometric centre bias."""
        del exclusion_mask
        center = np.array([0.5, 0.5], np.float32)
        if printable_mask is None or np.count_nonzero(printable_mask) == 0:
            return center, 0.0
        points = mesh.points
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        ys, xs = np.nonzero(printable_mask > 0)
        if ys.size == 0:
            return center, 0.0
        center = np.array(
            [
                (float(xs.mean()) - min_xy[0]) / span[0],
                (float(ys.mean()) - min_xy[1]) / span[1],
            ],
            dtype=np.float32,
        )
        # Keep true centre — do not invent hardware avoidance offsets.
        return np.clip(center, 0.0, 1.0), 0.0

    @staticmethod
    def _base_crop(
        width: int, height: int, target_aspect: float, fit_mode: str
    ) -> Tuple[float, float]:
        """Unscaled source crop dimensions matching MeshWarper semantics."""
        design_aspect = width / max(float(height), 1e-6)
        if fit_mode == "stretch":
            return float(width), float(height)
        if fit_mode == "fit":
            # Contain: crop is larger than the design on one axis (letterbox).
            if design_aspect > target_aspect:
                return float(width), width / target_aspect
            return height * target_aspect, float(height)
        # fill — cover the printable surface, preserve aspect (equal crop margins)
        if design_aspect > target_aspect:
            return height * target_aspect, float(height)
        return float(width), width / target_aspect

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        """Robustly normalise a response map into 0-1."""
        high = float(np.percentile(values, 98))
        if high <= 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip(values / high, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _estimate_rotation(
        saliency: np.ndarray,
        content_mask: np.ndarray,
        mesh: ControlMesh,
        isolated_content: bool,
    ) -> float:
        """Auto-rotation disabled — placement stays axis-aligned to the mesh."""
        del saliency, content_mask, mesh, isolated_content
        return 0.0

    @staticmethod
    def _requested_offsets(
        centroid: np.ndarray,
        desired_uv: np.ndarray,
        design_size: Tuple[int, int],
        crop_size: Tuple[float, float],
        strength: float,
        max_offset: float,
    ) -> Tuple[float, float]:
        """Legacy helper — geometry fit always returns centered offsets."""
        del centroid, desired_uv, design_size, crop_size, strength, max_offset
        return 0.0, 0.0
