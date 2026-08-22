"""
Smart Cover Surface Engine.

The phone photograph is only a geometric reference. Artwork is fitted to the
printable face of the installed cover — never to the phone body, metal frame,
or hardware openings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..utils.helpers import order_points, quad_size, to_bgr
from .mesh import (
    DEFAULT_MESH_COLS,
    DEFAULT_MESH_ROWS,
    AdaptiveMeshBuilder,
    ControlMesh,
)
from .region_detector import (
    BoundaryEstimate,
    HardwareRegionDetector,
    PhoneBoundaryDetector,
    PrintableRegion,
    PrintableRegionDetector,
)
from .template_cache import TemplateCache, TemplateManager
from .device_template import (
    CornerRadii,
    CutoutSpec,
    DeviceTemplateCatalog,
    estimate_corner_radii,
)


@dataclass
class CoverSurfaceResult:
    """Detected printable cover surface ready for mesh warp and smart fit."""

    mesh: ControlMesh
    exclusion_mask: np.ndarray
    hardware_contours: List[np.ndarray] = field(default_factory=list)
    confidence: float = 0.0
    phone_mask: Optional[np.ndarray] = None
    cover_mask: Optional[np.ndarray] = None
    printable_mask: Optional[np.ndarray] = None
    margin_percent: float = 0.0
    corner_radius_percent: float = 6.0
    corner_radii: Optional[CornerRadii] = None
    from_template: bool = False
    template_id: Optional[str] = None
    model_id: str = ""

    def resolved_corner_radii(self) -> CornerRadii:
        if self.corner_radii is not None:
            return self.corner_radii
        return CornerRadii.uniform(self.corner_radius_percent)

    def as_printable_region(self) -> PrintableRegion:
        """Compatibility wrapper for callers that expect PrintableRegion."""
        return PrintableRegion(
            mesh=self.mesh,
            exclusion_mask=self.exclusion_mask,
            hardware_contours=self.hardware_contours,
            confidence=self.confidence,
            silhouette_mask=self.cover_mask,
            printable_mask=self.printable_mask,
            margin_percent=self.margin_percent,
        )


class CoverSurfaceEngine:
    """
    Estimate the installed cover's printable surface using classical CV only.

    Pipeline:
      phone boundary (reference)
        → installed cover face
        → printable inset + rounded corners
        → hardware exclusions
        → editable control mesh

    When a local template matches the phone layout, that saved geometry is
    reused instead of re-detecting.
    """

    ANALYSIS_LONG_EDGE = 900

    def __init__(
        self,
        template_cache: Optional[TemplateCache] = None,
        template_manager: Optional[TemplateManager] = None,
    ) -> None:
        if template_manager is not None:
            self.templates = template_manager
        elif template_cache is not None:
            self.templates = TemplateManager(template_cache.directory)
        else:
            self.templates = TemplateManager()
        # Back-compat alias used by older compositor wiring.
        self.template_cache = self.templates.cache
        self.device_catalog = DeviceTemplateCatalog()
        self.last_phone_mask: Optional[np.ndarray] = None
        self.last_cover_mask: Optional[np.ndarray] = None
        self.last_printable_mask: Optional[np.ndarray] = None
        self.last_corner_radius_percent: float = 6.0
        self.last_corner_radii: CornerRadii = CornerRadii.uniform(6.0)
        self.last_from_template: bool = False
        self.last_template_id: Optional[str] = None
        self.last_model_id: str = ""

    def analyze(
        self,
        phone_image: np.ndarray,
        rows: int = DEFAULT_MESH_ROWS,
        cols: int = DEFAULT_MESH_COLS,
        use_templates: bool = True,
    ) -> CoverSurfaceResult:
        """Detect or recall the printable cover surface for a phone photo."""
        source = np.asarray(phone_image)
        phone = to_bgr(source)

        if use_templates:
            # Cheap silhouette for matching only — avoids GrabCut when a
            # template will be restored instantly.
            cheap = self._cheap_silhouette(phone)
            cached = self.templates.find(phone, cheap)
            if cached is not None:
                region = self.templates.materialise(cached, phone.shape[:2])
                # Always refresh hardware exclusions (side buttons / speakers
                # especially). Older templates often only stored camera holes.
                fresh_excl, fresh_contours, hw_conf = (
                    HardwareRegionDetector.detect(
                        phone,
                        CoverSurfaceEngine._canonical_hardware_quad(
                            phone, region.mesh.corner_points()
                        ),
                    )
                )
                if np.count_nonzero(fresh_excl) > 0:
                    # Live photo hardware is canonical. OR-ing a cached island
                    # AABB with discrete lenses rebuilt a rectangular hole.
                    region.exclusion_mask = fresh_excl
                    region.hardware_contours = fresh_contours
                    region.confidence = min(
                        1.0, region.confidence * 0.7 + hw_conf * 0.3
                    )
                    # Keep printable clear of refreshed exclusions.
                    if region.printable_mask is not None:
                        hard = (
                            (region.exclusion_mask > 96).astype(np.uint8) * 255
                        )
                        region.printable_mask = cv2.bitwise_and(
                            region.printable_mask, cv2.bitwise_not(hard)
                        )
                # Upgrade coarse legacy templates (e.g. 7×5) to production
                # mesh density. Wrap cage always comes from the live phone
                # silhouette — frozen template meshes / skewed phone_masks are
                # what produced the tilted sticker look on upright shots.
                radii = cached.radii()
                corner = float(np.clip(radii.median(), 2.5, 22.0))
                radii_tuple = radii.as_tuple()
                phone_from_tpl = getattr(region, "phone_mask", None)
                live_boundary = PhoneBoundaryDetector.detect(source)
                live_mask = getattr(live_boundary, "mask", None)
                if (
                    live_mask is not None
                    and np.count_nonzero(live_mask) > 64
                    and live_mask.shape[:2] == phone.shape[:2]
                ):
                    phone_gate = live_mask
                elif (
                    phone_from_tpl is not None
                    and np.count_nonzero(phone_from_tpl)
                    and phone_from_tpl.shape[:2] == phone.shape[:2]
                ):
                    phone_gate = phone_from_tpl
                elif (
                    region.silhouette_mask is not None
                    and region.silhouette_mask.shape[:2] == phone.shape[:2]
                ):
                    phone_gate = region.silhouette_mask
                else:
                    phone_gate = cv2.resize(
                        cheap,
                        (phone.shape[1], phone.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                gate = phone_gate
                if gate is None or np.count_nonzero(gate) == 0:
                    gate = region.silhouette_mask
                if gate is None or np.count_nonzero(gate) == 0:
                    gate = region.printable_mask
                target_rows = max(int(rows), int(region.mesh.rows))
                target_cols = max(int(cols), int(region.mesh.cols))
                # Prefer live corner radii — legacy templates often stored ~2.5%
                # which makes the wrap look like a sharp sticker.
                seed_quad = None
                if gate is not None and np.count_nonzero(gate) > 64:
                    seed_quad = AdaptiveMeshBuilder._stable_quad_from_mask(gate)
                if seed_quad is not None:
                    live_radii = estimate_corner_radii(gate, seed_quad)
                    if live_radii.median() >= 3.5:
                        radii = live_radii
                        corner = float(np.clip(radii.median(), 4.0, 22.0))
                        radii_tuple = radii.as_tuple()
                    # Upright product shots need visible roundness — tiny
                    # cached radii look like a sharp sticker wrap on any model.
                    if (
                        AdaptiveMeshBuilder._quad_axis_deviation_deg(seed_quad)
                        <= 3.5
                    ):
                        corner = float(max(corner, 8.0))
                        if radii.median() < 7.5:
                            radii = CornerRadii.uniform(corner)
                            radii_tuple = radii.as_tuple()
                if gate is not None and np.count_nonzero(gate) > 64:
                    if seed_quad is None:
                        seed_quad = region.mesh.corner_points()
                    denser = ControlMesh.from_quad(
                        seed_quad, target_rows, target_cols, adaptive=True
                    )
                    region.mesh = AdaptiveMeshBuilder.production_perimeter(
                        denser,
                        gate,
                        corner_radius_percent=corner,
                        max_move_fraction=0.12,
                        corner_radii=radii_tuple,
                        preserve_corner_arcs=True,
                    )
                elif (
                    region.mesh.rows < rows
                    or region.mesh.cols < cols
                ):
                    denser = ControlMesh.from_quad(
                        region.mesh.corner_points(), rows, cols, adaptive=True
                    )
                    region.mesh = denser
                # Refresh exclusions against the upright live cage so orphan
                # template holes (false top-right circles) are not restored.
                fresh_excl2, fresh_contours2, _ = HardwareRegionDetector.detect(
                    phone,
                    CoverSurfaceEngine._canonical_hardware_quad(
                        phone, region.mesh.corner_points(), phone_gate
                    ),
                )
                if np.count_nonzero(fresh_excl2) > 0:
                    region.exclusion_mask = fresh_excl2
                    region.hardware_contours = fresh_contours2
                    if region.printable_mask is not None:
                        hard = (
                            (region.exclusion_mask > 96).astype(np.uint8) * 255
                        )
                        region.printable_mask = cv2.bitwise_and(
                            region.printable_mask, cv2.bitwise_not(hard)
                        )
                result = CoverSurfaceResult(
                    mesh=region.mesh,
                    exclusion_mask=region.exclusion_mask,
                    hardware_contours=region.hardware_contours,
                    confidence=region.confidence,
                    phone_mask=phone_gate,
                    cover_mask=region.silhouette_mask,
                    printable_mask=region.printable_mask,
                    margin_percent=region.margin_percent,
                    corner_radius_percent=corner,
                    corner_radii=radii,
                    from_template=True,
                    template_id=cached.fingerprint,
                    model_id=str(cached.model_id or ""),
                )
                self._remember(result)
                return result

        boundary = PhoneBoundaryDetector.detect(source)
        self.last_phone_mask = boundary.mask
        return self._detect_fresh(phone, boundary, rows, cols)

    @staticmethod
    def _canonical_hardware_quad(
        phone: np.ndarray,
        fallback_quad: Optional[np.ndarray],
        phone_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Axis-aligned quad of the photo silhouette for hardware detect.

        The edit cage is only a working bound — a larger/tilted cage warps
        camera and side-button samples off the real phone.
        """
        gate = phone_mask
        if gate is None or np.count_nonzero(gate) < 64:
            try:
                est = PhoneBoundaryDetector.detect(phone)
                gate = getattr(est, "mask", None)
            except Exception:
                gate = None
        if gate is not None and np.count_nonzero(gate) >= 64:
            if gate.shape[:2] != phone.shape[:2]:
                gate = cv2.resize(
                    (gate > 127).astype(np.uint8) * 255,
                    (phone.shape[1], phone.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            quad = AdaptiveMeshBuilder._aabb_quad_from_mask(gate)
            if quad is not None:
                return quad
        if fallback_quad is not None:
            return fallback_quad
        h, w = phone.shape[:2]
        return np.array(
            [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
            dtype=np.float32,
        )

    @staticmethod
    def _cheap_silhouette(phone: np.ndarray) -> np.ndarray:
        """Fast layout mask for template matching (no GrabCut)."""
        return PhoneBoundaryDetector._border_colour_mask(
            cv2.resize(
                phone,
                (
                    max(1, phone.shape[1] // 2),
                    max(1, phone.shape[0] // 2),
                ),
                interpolation=cv2.INTER_AREA,
            )
            if max(phone.shape[:2]) > 600
            else phone
        )

    def centered(
        self,
        phone_image: np.ndarray,
        rows: int = DEFAULT_MESH_ROWS,
        cols: int = DEFAULT_MESH_COLS,
    ) -> CoverSurfaceResult:
        """Safe centered cover estimate without template lookup."""
        region = PrintableRegionDetector.centered(phone_image, rows, cols)
        result = CoverSurfaceResult(
            mesh=region.mesh,
            exclusion_mask=region.exclusion_mask,
            hardware_contours=region.hardware_contours,
            confidence=region.confidence,
            phone_mask=region.silhouette_mask,
            cover_mask=region.silhouette_mask,
            printable_mask=region.printable_mask,
            margin_percent=region.margin_percent,
            corner_radius_percent=6.0,
            from_template=False,
        )
        self._remember(result)
        return result

    def remember_correction(
        self,
        phone_image: np.ndarray,
        mesh: ControlMesh,
        exclusion_mask: Optional[np.ndarray] = None,
        margin_percent: float = 0.0,
        corner_radius_percent: Optional[float] = None,
        cover_mask: Optional[np.ndarray] = None,
        printable_mask: Optional[np.ndarray] = None,
        phone_mask: Optional[np.ndarray] = None,
        corner_radii: Optional[CornerRadii] = None,
        hardware_contours: Optional[List[np.ndarray]] = None,
        cutouts: Optional[List[dict]] = None,
        model_id: str = "",
        save_named_model: bool = True,
        display_name: str = "",
    ) -> None:
        """Persist a manual mesh correction for the current phone layout."""
        phone = to_bgr(phone_image)
        # Fingerprint with the same cheap silhouette used by analyze() so
        # template hits stay consistent across sessions.
        fingerprint_mask = self._cheap_silhouette(phone)
        radii = corner_radii or self.last_corner_radii
        if corner_radius_percent is not None and corner_radii is None:
            # Slider moved uniformly — keep four corners synced.
            if abs(radii.median() - float(corner_radius_percent)) > 0.35:
                radii = CornerRadii.uniform(float(corner_radius_percent))
        mid = model_id or self.last_model_id
        saved = self.templates.save(
            phone,
            mesh,
            exclusion_mask,
            silhouette=fingerprint_mask,
            cover_mask=cover_mask if cover_mask is not None else self.last_cover_mask,
            printable_mask=(
                printable_mask
                if printable_mask is not None
                else self.last_printable_mask
            ),
            margin_percent=margin_percent,
            corner_radius_percent=(
                float(corner_radius_percent)
                if corner_radius_percent is not None
                else float(radii.median())
            ),
            confidence=0.95,
            phone_mask=phone_mask if phone_mask is not None else self.last_phone_mask,
            corner_radii=radii,
            cutouts=cutouts,
            hardware_contours=hardware_contours,
            model_id=mid,
        )
        self.last_corner_radii = radii
        self.last_corner_radius_percent = float(radii.median())
        self.last_template_id = saved.fingerprint
        if mid:
            self.last_model_id = mid
        # Also write a named DeviceTemplate so the catalog grows with edits.
        if save_named_model:
            try:
                fp, _ = self.templates.fingerprint(phone, fingerprint_mask)
                self.device_catalog.capture_from_session(
                    phone_image=phone,
                    mesh=mesh,
                    phone_mask=phone_mask if phone_mask is not None else self.last_phone_mask,
                    cover_mask=cover_mask if cover_mask is not None else self.last_cover_mask,
                    printable_mask=(
                        printable_mask
                        if printable_mask is not None
                        else self.last_printable_mask
                    ),
                    hardware_contours=hardware_contours,
                    cutout_specs=(
                        [CutoutSpec.from_dict(c) for c in cutouts]
                        if cutouts
                        else None
                    ),
                    corner_radii=radii,
                    corner_radius_percent=float(radii.median()),
                    margin_percent=margin_percent,
                    fingerprint=fp,
                    model_id=mid or fp,
                    display_name=display_name or mid or fp[:8],
                    confidence=0.95,
                )
                if not mid:
                    self.last_model_id = fp
            except OSError:
                pass

    # ---------------------------------------------------------------- detection

    def _detect_fresh(
        self,
        phone: np.ndarray,
        boundary: BoundaryEstimate,
        rows: int,
        cols: int,
    ) -> CoverSurfaceResult:
        """Full classical cover-surface estimation from a phone photo."""
        cover_mask, cover_quad, rim_fraction = self._estimate_cover_face(
            phone, boundary
        )
        # Expand toward outer case/phone rim so wrap fills the visible face
        # (clear MagSafe shots often under-estimate the cover and leave a gap).
        wrap_mask = self.wrap_target_mask(cover_mask, boundary.mask)
        if wrap_mask is not None and np.count_nonzero(wrap_mask):
            cover_mask = wrap_mask
            cover_quad = self._mask_quad(cover_mask, fallback=cover_quad)
        # Hard safety: cover never leaves the phone silhouette.
        if boundary.mask is not None and np.count_nonzero(boundary.mask):
            cover_mask = cv2.bitwise_and(cover_mask, boundary.mask)
            cover_quad = self._mask_quad(cover_mask, fallback=cover_quad)
        corner_radii = estimate_corner_radii(cover_mask, cover_quad)
        corner_radius = float(corner_radii.median())
        # Density follows this photo's aspect + corner roundness (Phase 5).
        if rows == DEFAULT_MESH_ROWS and cols == DEFAULT_MESH_COLS:
            use_rows, use_cols = self.recommend_mesh_density(
                cover_mask, corner_radius
            )
        else:
            use_rows, use_cols = rows, cols
        mesh, printable_mask, margin_percent = self._mesh_from_cover(
            cover_mask,
            cover_quad,
            use_rows,
            use_cols,
            corner_radius,
            corner_radii=corner_radii.as_tuple(),
        )
        exclusion_mask, contours, hw_confidence = HardwareRegionDetector.detect(
            phone,
            CoverSurfaceEngine._canonical_hardware_quad(
                phone, cover_quad, cover_mask
            ),
        )
        # Production polish on detect: stadiums / circles, merge overlaps.
        if contours:
            merged = HardwareRegionDetector.merge_overlapping_contours(contours)
            polished = HardwareRegionDetector.perfect_finish_contours(
                merged, phone
            )
            if polished:
                contours = polished
                # Rebuild exclusion from polished editable outlines.
                h, w = phone.shape[:2]
                rebuilt = np.zeros((h, w), dtype=np.uint8)
                for contour in contours:
                    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
                    if len(pts) < 3:
                        continue
                    HardwareRegionDetector.paint_cutout_mask(
                        rebuilt, pts, analytical=True
                    )
                if np.count_nonzero(rebuilt):
                    exclusion_mask = rebuilt

        # Final printable mask = cover face inset ∩ ¬ hardware. Artwork must
        # never leave this mask; the compositor still receives exclusion_mask
        # for a hard safety edge during feathering.
        if printable_mask is not None and exclusion_mask is not None:
            hard = (exclusion_mask > 96).astype(np.uint8) * 255
            printable_mask = cv2.bitwise_and(
                printable_mask, cv2.bitwise_not(hard)
            )
            # Do NOT pull mesh verts away from camera holes — that warped the
            # upright cage into a trapezoid and skewed the entire wrap. Holes
            # are punched at composite time via exclusion_mask only.

        confidence = min(
            1.0,
            boundary.confidence * 0.55
            + hw_confidence * 0.25
            + (0.20 if rim_fraction > 0.004 else 0.10),
        )
        result = CoverSurfaceResult(
            mesh=mesh,
            exclusion_mask=exclusion_mask,
            hardware_contours=contours,
            confidence=confidence,
            phone_mask=boundary.mask,
            cover_mask=cover_mask,
            printable_mask=printable_mask,
            margin_percent=margin_percent,
            corner_radius_percent=corner_radius,
            corner_radii=corner_radii,
            from_template=False,
        )
        self._remember(result)
        return result

    def _remember(self, result: CoverSurfaceResult) -> None:
        self.last_phone_mask = result.phone_mask
        self.last_cover_mask = result.cover_mask
        self.last_printable_mask = result.printable_mask
        self.last_corner_radius_percent = result.corner_radius_percent
        self.last_corner_radii = result.resolved_corner_radii()
        self.last_from_template = result.from_template
        self.last_template_id = result.template_id
        self.last_model_id = str(result.model_id or "")

    def _estimate_cover_face(
        self, phone: np.ndarray, boundary: BoundaryEstimate
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate the installed cover face inside the phone silhouette.

        The phone body is a reference only. A thin outer rim (frame / edge
        wrap / bezel) is peeled away so the printable object is the cover.
        """
        quad = order_points(boundary.quad)
        quad_w, quad_h = quad_size(quad)
        if quad_w < 8 or quad_h < 8:
            return boundary.mask.copy(), quad, 0.0

        scale = min(1.0, self.ANALYSIS_LONG_EDGE / max(quad_w, quad_h))
        rect_w = max(80, int(round(quad_w * scale)))
        rect_h = max(160, int(round(quad_h * scale)))
        rect = np.array(
            [
                [0, 0], [rect_w - 1, 0],
                [rect_w - 1, rect_h - 1], [0, rect_h - 1],
            ],
            dtype=np.float32,
        )
        to_rect = cv2.getPerspectiveTransform(quad, rect)
        rectified = cv2.warpPerspective(
            phone, to_rect, (rect_w, rect_h),
            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE,
        )
        mask_rect = cv2.warpPerspective(
            boundary.mask, to_rect, (rect_w, rect_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask_rect = PrintableRegionDetector._largest_component(mask_rect)

        rim_px = self._rim_width_pixels(rectified, mask_rect)
        rim_fraction = rim_px / max(min(rect_w, rect_h), 1)

        if rim_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (rim_px * 2 + 1, rim_px * 2 + 1)
            )
            cover_rect = cv2.erode(mask_rect, kernel, iterations=1)
            if np.count_nonzero(cover_rect) < rect_w * rect_h * 0.12:
                cover_rect = mask_rect
                rim_px = 0
                rim_fraction = 0.0
        else:
            cover_rect = mask_rect

        # Replace the noisy silhouette with a stable rounded-rectangle cover
        # face. Raw contours zig-zag on soft photo edges; a geometric cover
        # matches how printable cases are actually manufactured.
        cover_rect, _corner_px = self._fit_rounded_cover(cover_rect)
        # Two-pass snap + re-fit keeps edges tight without triangular lobes.
        for _ in range(2):
            cover_rect = self._snap_cover_to_edges(
                rectified, cover_rect, mask_rect
            )
            cover_rect, _corner_px = self._fit_rounded_cover(cover_rect)
        cover_rect = PrintableRegionDetector._largest_component(cover_rect)

        from_rect = cv2.getPerspectiveTransform(rect, quad)
        image_h, image_w = phone.shape[:2]
        cover_mask = cv2.warpPerspective(
            cover_rect, from_rect, (image_w, image_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Soft phone gate — never AND with the raw jagged phone silhouette
        # (that imprints pixel stairs onto the manufactured rounded cover).
        phone_gate = self._soft_phone_gate(boundary.mask)
        cover_mask = cv2.bitwise_and(cover_mask, phone_gate)
        cover_mask = self._manufacture_smooth_cover(cover_mask)
        if np.count_nonzero(cover_mask) < np.count_nonzero(boundary.mask) * 0.2:
            cover_mask = self._manufacture_smooth_cover(boundary.mask.copy())
        cover_quad = self._mask_quad(cover_mask, fallback=quad)
        return cover_mask, cover_quad, float(rim_fraction)

    @staticmethod
    def _refine_phone_inside_cage(
        img: np.ndarray,
        gray: np.ndarray,
        mesh_prior: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Separate the phone body from a bright studio card inside the cage.

        Silver / light phones on white cards fail plain GrabCut — the card and
        phone look similar. Compare centre vs border luminance and keep the
        darker (or brighter) island that matches a phone aspect ratio.
        """
        if mesh_prior is None or np.count_nonzero(mesh_prior) < 64:
            return None
        h, w = gray.shape[:2]
        prior = (mesh_prior > 0).astype(np.uint8)
        pad = max(5, int(round(min(h, w) * 0.04)))
        core = cv2.erode(
            prior * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
            ),
            iterations=1,
        )
        ring = (prior > 0) & (core == 0)
        if np.count_nonzero(core) < 64 or np.count_nonzero(ring) < 64:
            return None
        center_lum = float(np.median(gray[core > 0]))
        border_lum = float(np.median(gray[ring]))
        delta = border_lum - center_lum

        # Edge map helps lock the real rim even when luminance is close.
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.bitwise_and(edges, prior * 255)

        candidates: List[np.ndarray] = []

        if abs(delta) >= 12.0:
            # Card brighter than phone (common) or dark plate around bright phone.
            if delta > 0:
                lo = center_lum - 35.0
                hi = (center_lum + border_lum) * 0.5
                band = (gray >= lo) & (gray <= hi) & (prior > 0)
            else:
                lo = (center_lum + border_lum) * 0.5
                hi = center_lum + 35.0
                band = (gray >= lo) & (gray <= hi) & (prior > 0)
            band_u8 = band.astype(np.uint8) * 255
            band_u8 = cv2.morphologyEx(
                band_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=2,
            )
            candidates.append(band_u8)

        # Gradient rim fill: strong edges near cage → flood from centre.
        if np.count_nonzero(edges) > 80:
            ys, xs = np.where(core > 0)
            seed = (int(xs.mean()), int(ys.mean()))
            # Dilate edges to close gaps, invert → basins, flood fill.
            rim = cv2.dilate(
                edges,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            fill_src = np.where(rim > 0, 0, 255).astype(np.uint8)
            fill_src[prior == 0] = 0
            ff = fill_src.copy()
            mask_ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff, mask_ff, seed, 128, loDiff=0, upDiff=0)
            flooded = (ff == 128).astype(np.uint8) * 255
            flooded = cv2.bitwise_and(flooded, prior * 255)
            candidates.append(flooded)

        best = None
        best_score = -1.0
        prior_area = float(np.count_nonzero(prior))
        frame = float(h * w)
        for cand in candidates:
            contours, _ = cv2.findContours(
                cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for outer in contours:
                area = float(cv2.contourArea(outer))
                if area < frame * 0.05 or area > prior_area * 0.96:
                    continue
                # Prefer a real phone-shaped island — NOT "fill ~94% of cage".
                # When the blue box is oversized, the old target locked onto
                # the cage instead of the device.
                fill_ratio = area / max(prior_area, 1.0)
                if fill_ratio < 0.35 or fill_ratio > 0.995:
                    continue
                x, y, bw, bh = cv2.boundingRect(outer)
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect < 1.2 or aspect > 3.3:
                    continue
                # Phone-like aspect ~1.8–2.3 scores highest; mild fill OK.
                aspect_fit = 1.0 - min(abs(aspect - 2.05) / 1.2, 1.0)
                score = area * (0.55 + 0.45 * aspect_fit) * (
                    1.0 - abs(fill_ratio - 0.72) * 0.55
                )
                if score > best_score:
                    best_score = score
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask, [outer], -1, 255, -1)
                    best = mask
        return best

    @staticmethod
    def _expand_mask_to_visible_rim(
        img: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Dilate a phone mask to the visible product rim (not studio card).

        White phones often detect a slightly inset face; a 1–2% grow toward
        non-background pixels removes the bald side strip without painting
        onto the empty studio.
        """
        if mask is None or np.count_nonzero(mask) < 64:
            return mask
        from ..utils.helpers import to_bgr

        bgr = to_bgr(img)
        h, w = mask.shape[:2]
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        m = (mask > 127).astype(np.uint8) * 255
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Sample true studio from the four corners only (away from the phone).
        corner = max(2, int(round(min(h, w) * 0.06)))
        border = np.concatenate(
            [
                lab[:corner, :corner].reshape(-1, 3),
                lab[:corner, -corner:].reshape(-1, 3),
                lab[-corner:, :corner].reshape(-1, 3),
                lab[-corner:, -corner:].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(lab - bg, axis=2)
        dist = cv2.GaussianBlur(dist, (5, 5), 0)
        core_dist = dist[m > 0]
        thr = float(max(2.5, np.percentile(core_dist, 5) * 0.32))
        device = (dist >= thr).astype(np.uint8) * 255
        # High-contrast dark phones are already at the visible rim — a 1.2%
        # dilate swallowed the gray AA halo / studio card as "device".
        core_med = float(np.median(core_dist)) if core_dist.size else 0.0
        if core_med >= 25.0:
            pad = max(1, int(round(min(h, w) * 0.003)))
        else:
            pad = max(3, int(round(min(h, w) * 0.012)))
        grown = cv2.dilate(
            m,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
            ),
            iterations=1,
        )
        # Grow only toward pixels that differ from studio. Never hull.
        ring = cv2.bitwise_and(grown, device)
        out = cv2.bitwise_or(m, ring)
        out = cv2.morphologyEx(
            out,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return m
        outer = max(contours, key=cv2.contourArea)
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled, [outer], -1, 255, -1)
        overlap = float(np.count_nonzero((filled > 0) & (m > 0)))
        if overlap < float(np.count_nonzero(m)) * 0.85:
            return m
        if float(np.count_nonzero(filled)) > float(h * w) * 0.82:
            return m
        # Grow never paints the studio card / AABB corner wedges.
        device = CoverSurfaceEngine._device_pixels_from_photo(bgr)
        filled = cv2.bitwise_and(filled, device)
        if np.count_nonzero(filled) < np.count_nonzero(m) * 0.90:
            return cv2.bitwise_and(m, device)
        return filled

    @staticmethod
    def _fuller_phone_candidate(
        img: np.ndarray, primary: np.ndarray
    ) -> np.ndarray:
        """
        On light phones / white cards, GrabCut often returns a shrunken
        interior. Prefer a fuller border/edge silhouette when it still looks
        like the same phone (contains the primary core).
        """
        h, w = img.shape[:2]
        primary_bin = (primary > 127).astype(np.uint8) * 255
        primary_a = float(np.count_nonzero(primary_bin))
        if primary_a < 64:
            return primary_bin

        candidates = [primary_bin]
        try:
            border = PhoneBoundaryDetector._border_colour_mask(img)
            border = PhoneBoundaryDetector._clean_mask(border)
            if border is not None and np.count_nonzero(border) > 64:
                candidates.append((border > 127).astype(np.uint8) * 255)
        except Exception:
            pass
        try:
            edges = PhoneBoundaryDetector._edge_mask(img)
            edges = PhoneBoundaryDetector._clean_mask(edges)
            if edges is not None and np.count_nonzero(edges) > 64:
                candidates.append((edges > 127).astype(np.uint8) * 255)
        except Exception:
            pass

        best = primary_bin
        best_a = primary_a
        frame = float(h * w)
        for cand in candidates:
            # Outer contour fill only — convex hull squared photo corners
            # and painted wrap into the white wedges at the AABB corners.
            contours, _ = cv2.findContours(
                cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            outer = max(contours, key=cv2.contourArea)
            filled = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(filled, [outer], -1, 255, -1)
            area = float(np.count_nonzero(filled))
            if area < frame * 0.08 or area > frame * 0.85:
                continue
            x, y, bw, bh = cv2.boundingRect(filled)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect < 1.25 or aspect > 3.2:
                continue
            overlap = float(
                np.count_nonzero((filled > 0) & (primary_bin > 0))
            )
            # Must cover the primary core — never jump to a different blob.
            if overlap < primary_a * 0.80:
                continue
            if area > best_a * 1.08:
                best = filled
                best_a = area
            elif area >= best_a * 0.98 and area > best_a:
                best = filled
                best_a = area
        return best

    @staticmethod
    def detect_phone_body_mask(
        phone_bgr: Optional[np.ndarray],
        cover_quad: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Phone body silhouette from the photo — ignores a drifted edit cage.

        Perfect Finish used to trust the blue mesh as a GrabCut prior. When that
        cage was tilted / oversized, the "phone mask" became the cage and the
        wrap floated off the real device. Always recover the phone from the
        photo first; only use the cage when it already hugs that silhouette.
        """
        if phone_bgr is None or getattr(phone_bgr, "size", 0) == 0:
            return None
        from ..utils.helpers import to_bgr

        img = to_bgr(phone_bgr)
        h, w = img.shape[:2]
        frame = float(h * w)

        photo: Optional[np.ndarray] = None
        try:
            est = PhoneBoundaryDetector.detect(img)
            if (
                est is not None
                and getattr(est, "mask", None) is not None
                and np.count_nonzero(est.mask) >= frame * 0.04
            ):
                photo = (est.mask > 127).astype(np.uint8) * 255
        except Exception:
            photo = None
        if photo is None:
            photo = CoverSurfaceEngine.estimate_phone_mask_from_photo(
                img, cover_quad=None
            )
        if photo is None or np.count_nonzero(photo) < 64:
            return None

        # Upgrade shrunk GrabCut interiors to the full device rim when possible.
        photo = CoverSurfaceEngine._fuller_phone_candidate(img, photo)
        photo = CoverSurfaceEngine.seal_phone_body(photo, phone_bgr=img)
        if photo is None or np.count_nonzero(photo) < 64:
            return None
        # Nudge the silhouette out to the visible silver/product rim so wrap
        # does not leave a bald white strip along the sides.
        photo = CoverSurfaceEngine._expand_mask_to_visible_rim(img, photo)
        phone_a = float(np.count_nonzero(photo > 127))

        if cover_quad is None:
            return CoverSurfaceEngine._manufacture_smooth_cover(photo)

        try:
            pts = order_points(
                np.asarray(cover_quad, dtype=np.float32).reshape(-1, 2)
            )
        except Exception:
            return CoverSurfaceEngine._manufacture_smooth_cover(photo)
        if pts.shape[0] < 4:
            return CoverSurfaceEngine._manufacture_smooth_cover(photo)

        cage = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(cage, np.round(pts).astype(np.int32), 255)
        cage_a = float(np.count_nonzero(cage))
        if cage_a < 64:
            return CoverSurfaceEngine._manufacture_smooth_cover(photo)
        overlap = float(np.count_nonzero((cage > 0) & (photo > 127)))
        iou = overlap / max(cage_a + phone_a - overlap, 1.0)
        # Drifted / oversized blue box — never let it redefine the phone.
        if cage_a > phone_a * 1.15 or iou < 0.50:
            return CoverSurfaceEngine._manufacture_smooth_cover(photo)

        guided = CoverSurfaceEngine.estimate_phone_mask_from_photo(
            img, cover_quad=pts
        )
        if guided is None or np.count_nonzero(guided) < 64:
            return CoverSurfaceEngine._manufacture_smooth_cover(photo)
        guided = CoverSurfaceEngine.seal_phone_body(guided, phone_bgr=img)
        if guided is None or np.count_nonzero(guided) < 64:
            return CoverSurfaceEngine._manufacture_smooth_cover(photo)
        g_a = float(np.count_nonzero(guided > 127))
        if g_a < phone_a * 0.82 or g_a > phone_a * 1.18:
            return CoverSurfaceEngine._manufacture_smooth_cover(photo)
        g_overlap = float(np.count_nonzero((guided > 127) & (photo > 127)))
        if g_overlap / max(phone_a, 1.0) < 0.75:
            return CoverSurfaceEngine._manufacture_smooth_cover(photo)
        return CoverSurfaceEngine._manufacture_smooth_cover(guided)

    @staticmethod
    def detect_phone_wrap_silhouette(
        phone_bgr: Optional[np.ndarray],
        cover_quad: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Full photo silhouette for wrap clip (any phone / studio shot).

        Unlike ``detect_phone_body_mask``, this does **not** run
        ``_manufacture_smooth_cover`` — contour smoothing chords off round
        corners and leaves a silver rim gap under the wrap. Oversized edit
        cages are ignored; only the photo product rim matters.
        """
        if phone_bgr is None or getattr(phone_bgr, "size", 0) == 0:
            return None
        from ..utils.helpers import to_bgr

        img = to_bgr(phone_bgr)
        h, w = img.shape[:2]
        frame = float(h * w)

        photo: Optional[np.ndarray] = None
        try:
            est = PhoneBoundaryDetector.detect(img)
            if (
                est is not None
                and getattr(est, "mask", None) is not None
                and np.count_nonzero(est.mask) >= frame * 0.04
            ):
                photo = (est.mask > 127).astype(np.uint8) * 255
        except Exception:
            photo = None
        if photo is None:
            photo = CoverSurfaceEngine.estimate_phone_mask_from_photo(
                img, cover_quad=None
            )
        if photo is None or np.count_nonzero(photo) < 64:
            return None

        photo = CoverSurfaceEngine._fuller_phone_candidate(img, photo)
        photo = CoverSurfaceEngine.seal_phone_body(
            photo,
            phone_bgr=img,
            manufacture_smooth=False,
            hull_gaps=False,
        )
        if photo is None or np.count_nonzero(photo) < 64:
            return None
        photo = CoverSurfaceEngine._expand_mask_to_visible_rim(img, photo)
        if photo is None or np.count_nonzero(photo) < 64:
            return None

        # Optional: if the user cage already hugs the phone, refine once.
        if cover_quad is not None:
            try:
                pts = order_points(
                    np.asarray(cover_quad, dtype=np.float32).reshape(-1, 2)
                )
            except Exception:
                pts = None
            if pts is not None and pts.shape[0] >= 4:
                phone_a = float(np.count_nonzero(photo > 127))
                cage = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(cage, np.round(pts).astype(np.int32), 255)
                cage_a = float(np.count_nonzero(cage))
                if cage_a >= 64:
                    overlap = float(
                        np.count_nonzero((cage > 0) & (photo > 127))
                    )
                    iou = overlap / max(cage_a + phone_a - overlap, 1.0)
                    if cage_a <= phone_a * 1.12 and iou >= 0.55:
                        guided = CoverSurfaceEngine.estimate_phone_mask_from_photo(
                            img, cover_quad=pts
                        )
                        if guided is not None and np.count_nonzero(guided) >= 64:
                            guided = CoverSurfaceEngine.seal_phone_body(
                                guided,
                                phone_bgr=img,
                                manufacture_smooth=False,
                                hull_gaps=False,
                            )
                            guided = CoverSurfaceEngine._expand_mask_to_visible_rim(
                                img, guided
                            )
                            g_a = float(np.count_nonzero(guided > 127))
                            if phone_a * 0.85 <= g_a <= phone_a * 1.15:
                                photo = guided

        # Tiny dilate then keep only near-device pixels — closes chalk tips.
        tip = max(2, int(round(min(h, w) * 0.004)))
        grown = cv2.dilate(
            (photo > 127).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (tip * 2 + 1, tip * 2 + 1)
            ),
            iterations=1,
        )
        device = CoverSurfaceEngine._device_pixels_from_photo(img)
        # Grow only onto real device pixels. gray<254 kept near-white studio
        # in the rim ring and looked like over-wrap on the card.
        out = cv2.bitwise_or(photo, cv2.bitwise_and(grown, device))
        # Never keep studio-card pixels in the wrap silhouette.
        out = cv2.bitwise_and(out, device)
        # Fill holes only inside the outer contour.
        contours, _ = cv2.findContours(
            out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            outer = max(contours, key=cv2.contourArea)
            filled = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(filled, [outer], -1, 255, -1)
            filled = cv2.bitwise_and(filled, device)
            if float(np.count_nonzero(filled)) <= float(h * w) * 0.82:
                out = filled
        return out

    @staticmethod
    def polish_product_silhouette(mask: np.ndarray) -> np.ndarray:
        """
        Clean jagged photo-rim stairs into smooth product curves.

        Removes outward pixel speckles at corners / side buttons while keeping
        the same fill area (no bald gaps). Relative to mask size — no hardcode.
        """
        from .mesh import AdaptiveMeshBuilder, _fill_closed_polyline_aa

        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 64:
            return mask.astype(np.uint8) if mask.dtype == np.uint8 else binary

        h, w = binary.shape[:2]
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return binary

        outer = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32
        )
        if outer.shape[0] < 20:
            return binary

        short = float(min(h, w))
        win1 = max(13, min(37, int(outer.shape[0] // 14) * 2 + 1))
        win2 = max(9, min(23, win1 // 2 * 2 + 1))
        smooth = AdaptiveMeshBuilder._smooth_closed_polyline(
            outer, window=win1
        )
        smooth = AdaptiveMeshBuilder._smooth_closed_polyline(
            smooth, window=win2
        )
        expand = 0.0  # Keep exact silhouette size — AA without enlarge/reshape.
        gate = _fill_closed_polyline_aa(
            smooth, (h, w), scale=16, expand_px=expand
        )
        gate = np.clip(gate, 0.0, 1.0)

        # Interior core must stay solid — polish only replaces the noisy rim.
        core_k = max(3, int(round(short * 0.008)) | 1)
        core = cv2.erode(
            binary,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (core_k, core_k)
            ),
            iterations=1,
        )
        out_f = np.maximum(gate, core.astype(np.float32) / 255.0)
        # Drop outward spikes beyond the smooth gate.
        out_f = np.where(gate < 0.05, 0.0, out_f)
        out = (out_f > 0.50).astype(np.uint8) * 255
        out = cv2.bitwise_or(out, core)
        if np.count_nonzero(out) < np.count_nonzero(binary) * 0.97:
            out = cv2.bitwise_or(out, binary)
        return out

    @staticmethod
    def estimate_phone_mask_from_photo(
        phone_bgr: Optional[np.ndarray],
        cover_quad: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Recover a phone silhouette from the studio photo (any model / colour).

        ``cover_quad`` (user mesh corners) is a strong prior so light phones on
        white cards / dark plates are not confused with the backdrop.
        """
        if phone_bgr is None or getattr(phone_bgr, "size", 0) == 0:
            return None
        from ..utils.helpers import to_bgr

        img = to_bgr(phone_bgr)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        mesh_prior = None
        if cover_quad is not None:
            pts = order_points(
                np.asarray(cover_quad, dtype=np.float32).reshape(-1, 2)
            )
            if pts.shape[0] >= 4:
                mesh_prior = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(
                    mesh_prior, np.round(pts).astype(np.int32), 255
                )
                pad = max(6, int(round(min(h, w) * 0.02)))
                mesh_prior = cv2.dilate(
                    mesh_prior,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
                    ),
                    iterations=1,
                )

        # GrabCut seeded by the cage — best for silver phones on white cards.
        if mesh_prior is not None and np.count_nonzero(mesh_prior) > 64:
            try:
                refined = CoverSurfaceEngine._refine_phone_inside_cage(
                    img, gray, mesh_prior
                )
                if refined is not None and np.count_nonzero(refined) > 64:
                    return CoverSurfaceEngine._manufacture_smooth_cover(refined)

                gc = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
                ys, xs = np.where(mesh_prior > 0)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                margin = max(4, int(round(min(h, w) * 0.01)))
                x0 = max(0, x0 - margin)
                y0 = max(0, y0 - margin)
                x1 = min(w - 1, x1 + margin)
                y1 = min(h - 1, y1 + margin)
                gc[y0 : y1 + 1, x0 : x1 + 1] = cv2.GC_PR_BGD
                # Outer cage band = probable background (studio card spill).
                outer_band = mesh_prior.copy()
                core_pad = max(5, int(round(min(h, w) * 0.035)))
                core = cv2.erode(
                    mesh_prior,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (core_pad * 2 + 1, core_pad * 2 + 1)
                    ),
                    iterations=1,
                )
                gc[(mesh_prior > 0) & (core == 0)] = cv2.GC_PR_BGD
                # Inner core = definite / probable foreground.
                inner_pad = max(3, int(round(min(h, w) * 0.018)))
                inner = cv2.erode(
                    core,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (inner_pad * 2 + 1, inner_pad * 2 + 1)
                    ),
                    iterations=1,
                )
                gc[core > 0] = cv2.GC_PR_FGD
                gc[inner > 0] = cv2.GC_FGD
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
                cv2.grabCut(
                    img, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK
                )
                grab = np.where(
                    (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0
                ).astype(np.uint8)
                grab = cv2.bitwise_and(grab, mesh_prior)
                contours, _ = cv2.findContours(
                    grab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    outer = max(contours, key=cv2.contourArea)
                    area = float(cv2.contourArea(outer))
                    prior_a = float(np.count_nonzero(mesh_prior))
                    if area >= float(h * w) * 0.06 and area < prior_a * 0.98:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(mask, [outer], -1, 255, -1)
                        return CoverSurfaceEngine._manufacture_smooth_cover(
                            mask
                        )
            except Exception:
                pass

        # Otsu fallback — score by mesh overlap when available.
        _, dark = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        _, light = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        best = None
        best_score = -1.0
        frame = float(h * w)
        prior_area = float(np.count_nonzero(mesh_prior)) if mesh_prior is not None else 0.0
        for binary in (dark, light):
            closed = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=2,
            )
            if mesh_prior is not None:
                closed = cv2.bitwise_and(closed, mesh_prior)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            for outer in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                area = float(cv2.contourArea(outer))
                if area < frame * 0.06 or area > frame * 0.92:
                    continue
                x, y, bw, bh = cv2.boundingRect(outer)
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect < 1.15 or aspect > 3.4:
                    continue
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [outer], -1, 255, -1)
                score = area * min(aspect, 2.4)
                if prior_area > 0:
                    overlap = float(
                        np.count_nonzero((mask > 0) & (mesh_prior > 0))
                    )
                    iou = overlap / max(
                        area + prior_area - overlap, 1.0
                    )
                    # Prefer silhouettes that sit inside the user's cage.
                    score = overlap * (0.35 + iou) * min(aspect, 2.4)
                    if iou < 0.25:
                        continue
                if score > best_score:
                    best_score = score
                    best = mask
        if best is None and mesh_prior is not None:
            # Last resort: never return the raw oversized cage (white card).
            # Prefer photo pixels that are not near-white studio spill.
            device = CoverSurfaceEngine._device_pixels_from_photo(img)
            cand = cv2.bitwise_and(mesh_prior, device)
            if np.count_nonzero(cand) >= float(h * w) * 0.05:
                best = cand
            else:
                best = cv2.erode(
                    mesh_prior,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
        if best is None:
            return None
        # Final gate: strip pure studio white even if GrabCut leaked.
        best = cv2.bitwise_and(
            best, CoverSurfaceEngine._device_pixels_from_photo(img)
        )
        if np.count_nonzero(best) < float(h * w) * 0.04:
            return None
        smooth = CoverSurfaceEngine._manufacture_smooth_cover(best)
        if mesh_prior is not None:
            smooth = CoverSurfaceEngine.complete_phone_silhouette(
                smooth, mesh_prior, phone_bgr=phone_bgr
            )
        else:
            smooth = CoverSurfaceEngine.seal_phone_body(
                smooth, phone_bgr=phone_bgr
            )
        return smooth

    @staticmethod
    def _device_pixels_from_photo(
        img_bgr: np.ndarray, *, rim_edges: bool = True
    ) -> np.ndarray:
        """
        Binary mask of non-studio pixels (reject the backdrop card).

        Threshold is taken from the image-corner card, so light-grey studios
        are excluded while silver / white phones (not corner-connected) stay.
        Interior specular on the device is not border-connected to the card.

        ``rim_edges`` (default True) unions Canny edges so grow-to-rim can
        land on the outline. Wrap clipping must pass ``rim_edges=False`` —
        otherwise AA/studio edge pixels are treated as printable surface and
        artwork bleeds past the physical phone.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sat = np.max(img_bgr, axis=2).astype(np.float32) - np.min(
            img_bgr, axis=2
        ).astype(np.float32)
        h, w = gray.shape[:2]
        band = max(2, int(round(min(h, w) * 0.04)))
        corners = np.concatenate(
            [
                gray[:band, :band].reshape(-1),
                gray[:band, -band:].reshape(-1),
                gray[-band:, :band].reshape(-1),
                gray[-band:, -band:].reshape(-1),
            ]
        )
        studio_ref = float(np.median(corners))
        # Card = as bright as the corners (within ~5%) or near-white.
        thr = min(247.0, max(232.0, studio_ref - 12.0))
        card = ((gray >= thr) & (sat <= 12.0)) | (
            (gray >= 252.0) & (sat <= 8.0)
        )
        device = np.ones((h, w), dtype=np.uint8) * 255
        device[card] = 0
        if rim_edges:
            # Keep the true rim edge so grow-to-rim can land on the outline.
            edges = cv2.Canny(gray.astype(np.uint8), 20, 60)
            device = cv2.max(device, (edges > 0).astype(np.uint8) * 255)
        return device

    @staticmethod
    def _silhouette_matches_cage(
        mask: Optional[np.ndarray],
        cage: Optional[np.ndarray],
        *,
        min_fill: float = 0.78,
        min_side: float = 0.62,
    ) -> bool:
        """
        True when ``mask`` still covers the edit cage (not a half-phone bite).

        Used to decide whether wrap may be clipped to the silhouette.
        """
        if mask is None or cage is None:
            return False
        m = (mask > 127).astype(np.uint8)
        c = (cage > 127).astype(np.uint8)
        if m.shape[:2] != c.shape[:2]:
            m = cv2.resize(
                m, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        cage_a = float(np.count_nonzero(c))
        if cage_a < 64:
            return False
        fill = float(np.count_nonzero(m & c)) / cage_a
        if fill < min_fill:
            return False
        ys, xs = np.where(c > 0)
        x0, x1 = int(xs.min()), int(xs.max())
        bw = max(1, x1 - x0 + 1)
        strip = max(2, int(round(bw * 0.16)))
        left = c.copy()
        left[:, x0 + strip :] = 0
        right = c.copy()
        right[:, : max(0, x1 - strip + 1)] = 0
        for band in (left, right):
            band_a = float(np.count_nonzero(band))
            if band_a < 32:
                continue
            if float(np.count_nonzero(m & band)) / band_a < min_side:
                return False
        return True

    @staticmethod
    def seal_phone_body(
        mask: Optional[np.ndarray],
        phone_bgr: Optional[np.ndarray] = None,
        *,
        manufacture_smooth: bool = True,
        hull_gaps: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Force a solid wrap face — no open camera bites / diagonal bald patches.

        GrabCut often leaves a pac-man notch from the camera to the top edge.
        Fitting the mesh to that outline pulls the top edge around the camera
        and wipes print off the top-right of the phone.
        """
        if mask is None or getattr(mask, "size", 0) == 0:
            return mask
        m = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(m) < 64:
            return mask.astype(np.uint8)
        h, w = m.shape[:2]

        flood = m.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 128)
        m = cv2.bitwise_or(m, (flood == 0).astype(np.uint8) * 255)

        x, y, bw, bh = cv2.boundingRect(m)
        fill = float(np.count_nonzero(m)) / float(max(bw * bh, 1))
        if hull_gaps and fill < 0.93:
            close_px = max(9, int(round(min(bw, bh) * 0.045)))
            sealed = cv2.morphologyEx(
                m,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
                ),
                iterations=2,
            )
            contours, _ = cv2.findContours(
                sealed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                outer = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(outer)
                hull_mask = np.zeros((h, w), np.uint8)
                cv2.drawContours(hull_mask, [hull], -1, 255, -1)
                if phone_bgr is not None and getattr(phone_bgr, "size", 0):
                    from ..utils.helpers import to_bgr

                    img = to_bgr(phone_bgr)
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(
                            img, (w, h), interpolation=cv2.INTER_AREA
                        )
                    hull_mask = cv2.bitwise_and(
                        hull_mask,
                        CoverSurfaceEngine._device_pixels_from_photo(img),
                    )
                sealed = cv2.bitwise_or(sealed, hull_mask)
            m = sealed

        rim_px = max(1, int(round(min(h, w) * 0.004)))
        grown = cv2.dilate(
            m,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (rim_px * 2 + 1, rim_px * 2 + 1)
            ),
            iterations=1,
        )
        if phone_bgr is not None and getattr(phone_bgr, "size", 0):
            from ..utils.helpers import to_bgr

            img = to_bgr(phone_bgr)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            grown = cv2.bitwise_and(
                grown, CoverSurfaceEngine._device_pixels_from_photo(img)
            )
        if manufacture_smooth:
            return CoverSurfaceEngine._manufacture_smooth_cover(grown)
        contours, _ = cv2.findContours(
            grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            outer = max(contours, key=cv2.contourArea)
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, [outer], -1, 255, -1)
            if float(np.count_nonzero(filled)) <= float(h * w) * 0.82:
                return filled
        return grown

    @staticmethod
    def complete_phone_silhouette(
        mask: Optional[np.ndarray],
        cage: Optional[np.ndarray],
        phone_bgr: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Make a solid phone body for wrap — works on any model / colour.

        GrabCut on silver phones often bites out the camera side. Fill
        enclosed holes and optionally borrow cage pixels that still look
        like the device — never the white studio card outside the phone.
        """
        if mask is None or getattr(mask, "size", 0) == 0:
            return mask
        m = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(m) < 64:
            return mask.astype(np.uint8)
        h, w = m.shape[:2]

        # 1) Fill enclosed interior holes (lenses treated as background).
        flood = m.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 128)
        holes = (flood == 0).astype(np.uint8) * 255
        m = cv2.bitwise_or(m, holes)

        if cage is not None and np.count_nonzero(cage) > 64:
            c = (cage > 127).astype(np.uint8) * 255
            if c.shape[:2] != (h, w):
                c = cv2.resize(c, (w, h), interpolation=cv2.INTER_NEAREST)
                c = (c > 127).astype(np.uint8) * 255
            cage_a = float(np.count_nonzero(c))
            mask_a = float(np.count_nonzero(m))
            # Oversized edit box (white padding) must NOT inflate the phone.
            if cage_a <= mask_a * 1.12:
                core_pad = max(2, int(round(min(h, w) * 0.0035)))
                core = cv2.erode(
                    c,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (core_pad * 2 + 1, core_pad * 2 + 1)
                    ),
                    iterations=1,
                )
                # Reject studio-white / near-white pixels when photo is available.
                if phone_bgr is not None and getattr(phone_bgr, "size", 0):
                    from ..utils.helpers import to_bgr

                    img = to_bgr(phone_bgr)
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(
                            img, (w, h), interpolation=cv2.INTER_AREA
                        )
                    core = cv2.bitwise_and(
                        core, CoverSurfaceEngine._device_pixels_from_photo(img)
                    )
                m = cv2.bitwise_or(m, core)

        return CoverSurfaceEngine.seal_phone_body(m, phone_bgr=phone_bgr)

    @staticmethod
    def _soft_phone_gate(phone_mask: np.ndarray) -> np.ndarray:
        """Dilate + close the phone silhouette so cover edges stay smooth."""
        binary = (phone_mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            return phone_mask
        short = min(binary.shape[:2])
        pad = max(2, int(round(short * 0.004)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
        )
        gated = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        gated = cv2.dilate(gated, kernel, iterations=1)
        return gated

    @staticmethod
    def _manufacture_smooth_cover(mask: np.ndarray) -> np.ndarray:
        """
        Rebuild a manufactured-smooth cover silhouette from a noisy mask.

        Phone cases have smooth sides and constant corner radii. Contour
        smoothing + morph close removes residual zig-zags after perspective.
        """
        binary = (mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 64:
            return mask.astype(np.uint8)

        soft = max(3, (min(binary.shape[:2]) // 50) | 1)
        closed = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (soft, soft)),
            iterations=2,
        )
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return closed

        outer = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32
        )
        if outer.shape[0] >= 24:
            from .mesh import AdaptiveMeshBuilder

            window = max(9, min(31, outer.shape[0] // 30 * 2 + 1))
            outer = AdaptiveMeshBuilder._smooth_closed_polyline(
                outer, window=window
            )
            # Second lighter pass locks a product-smooth perimeter.
            outer = AdaptiveMeshBuilder._smooth_closed_polyline(
                outer, window=max(5, window // 2 * 2 + 1)
            )

        filled = np.zeros_like(binary)
        from .mesh import _fill_closed_polyline_aa

        cover = _fill_closed_polyline_aa(
            outer, binary.shape[:2], scale=4, expand_px=0.6
        )
        filled = (np.clip(cover, 0.0, 1.0) * 255.0).astype(np.uint8)
        # Allow a 1–2 px rebuild envelope so AA fill doesn't shrink the face.
        envelope = cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        filled = cv2.bitwise_and(filled, envelope)
        if np.count_nonzero(filled) < np.count_nonzero(binary) * 0.70:
            return closed
        return filled.astype(np.uint8)

    @staticmethod
    def _fit_rounded_cover(mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Fit a clean rounded rectangle to the cover face in rectified space.

        Manufactured covers have smooth sides and constant corner radii. Fitting
        that shape removes jagged silhouette noise while staying inside the
        detected face.
        """
        binary = (mask > 0).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            return mask, 0

        ys, xs = np.nonzero(binary)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        short = min(width, height)

        # Corner radius from how far the filled mask sits inside each bbox corner.
        radii = []
        for cx, cy, sx, sy in (
            (x0, y0, 1, 1),
            (x1, y0, -1, 1),
            (x1, y1, -1, -1),
            (x0, y1, 1, -1),
        ):
            radius = 1
            limit = max(2, int(short * 0.28))
            for candidate in range(2, limit):
                px = cx + sx * candidate
                py = cy + sy * candidate
                if not (x0 <= px <= x1 and y0 <= py <= y1):
                    break
                # Along the top/bottom toward the side, and along the side
                # toward the top/bottom, the cover should still be present once
                # we are past the rounded corner.
                along_x = binary[cy, cx + sx * candidate] if 0 <= cy < binary.shape[0] and 0 <= cx + sx * candidate < binary.shape[1] else 0
                along_y = binary[cy + sy * candidate, cx] if 0 <= cy + sy * candidate < binary.shape[0] and 0 <= cx < binary.shape[1] else 0
                if along_x and along_y:
                    radius = candidate
                    break
            radii.append(radius)

        corner = int(np.median(radii)) if radii else max(2, int(short * 0.08))
        corner = int(np.clip(corner, 2, short // 2))

        fitted = np.zeros_like(binary)
        CoverSurfaceEngine._fill_rounded_rect(
            fitted, x0, y0, x1, y1, corner
        )
        fitted = cv2.bitwise_and(fitted, binary) * 255
        if np.count_nonzero(fitted) < np.count_nonzero(binary) * 0.75:
            # Fall back to a lightly smoothed original when the geometric fit
            # would discard too much of a curved perspective silhouette.
            soft = max(3, (short // 40) | 1)
            smoothed = cv2.GaussianBlur(mask, (soft * 2 + 1, soft * 2 + 1), 0)
            _, fitted = cv2.threshold(smoothed, 96, 255, cv2.THRESH_BINARY)
            fitted = cv2.bitwise_and(fitted, binary * 255)
        return fitted.astype(np.uint8), corner

    @staticmethod
    def _fill_rounded_rect(
        mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, radius: int
    ) -> None:
        """Rasterise a filled axis-aligned rounded rectangle."""
        radius = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
        cv2.rectangle(mask, (x0 + radius, y0), (x1 - radius, y1), 1, -1)
        cv2.rectangle(mask, (x0, y0 + radius), (x1, y1 - radius), 1, -1)
        for center in (
            (x0 + radius, y0 + radius),
            (x1 - radius, y0 + radius),
            (x0 + radius, y1 - radius),
            (x1 - radius, y1 - radius),
        ):
            cv2.circle(mask, center, radius, 1, -1, cv2.LINE_AA)

    @staticmethod
    def _snap_cover_to_edges(
        image: np.ndarray, cover: np.ndarray, phone_mask: np.ndarray
    ) -> np.ndarray:
        """
        Pull cover sides onto strong local edges without leaving the phone.

        Classical Canny edges near the current perimeter correct small inward
        bias from rim peeling while rejecting background clutter.
        """
        if np.count_nonzero(cover) == 0:
            return cover

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 40, 40)
        edges = cv2.Canny(gray, 40, 120)
        # Only edges near the cover perimeter are allowed to influence the mask.
        band = max(2, int(round(min(cover.shape) * 0.018)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1)
        )
        dilate = cv2.dilate(cover, kernel, iterations=1)
        erode = cv2.erode(cover, kernel, iterations=1)
        ring = cv2.subtract(dilate, erode)
        local_edges = cv2.bitwise_and(edges, ring)

        if np.count_nonzero(local_edges) < 24:
            return cover

        # Grow only inside a thin dilated band of the current cover — never OR
        # raw edge pixels into an unbounded blob (that creates triangular spikes).
        max_expand = cv2.dilate(cover, kernel, iterations=1)
        grown = cv2.bitwise_or(cover, cv2.bitwise_and(local_edges, max_expand))
        grown = cv2.morphologyEx(
            grown, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band | 1, band | 1)),
            iterations=1,
        )
        grown = cv2.bitwise_and(grown, phone_mask)
        grown = cv2.bitwise_and(grown, max_expand)

        # Constrain growth to the rounded-rect envelope of the pre-snap cover so
        # Canny fragments along the side cannot invent a protruding lobe.
        envelope, _ = CoverSurfaceEngine._fit_rounded_cover(cover)
        if np.count_nonzero(envelope):
            envelope = cv2.dilate(envelope, kernel, iterations=1)
            grown = cv2.bitwise_and(grown, envelope)

        grown = PrintableRegionDetector._largest_component(grown)
        if np.count_nonzero(grown) < np.count_nonzero(cover) * 0.9:
            return cover
        return grown

    @staticmethod
    def _rim_width_pixels(image: np.ndarray, mask: np.ndarray) -> int:
        """
        Estimate how many pixels of phone frame sit outside the cover face.

        Samples concentric rings of the rectified silhouette. A sudden Lab
        colour shift near the outer edge indicates the metal/glass rim that
        belongs to the phone, not the printable cover.
        """
        height, width = mask.shape
        short = min(height, width)
        if short < 40 or np.count_nonzero(mask) == 0:
            return max(1, int(round(short * 0.012)))

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        distance = cv2.distanceTransform(
            (mask > 127).astype(np.uint8), cv2.DIST_L2, 5
        )
        max_dist = float(distance.max())
        if max_dist < 8:
            return max(1, int(round(short * 0.012)))

        # Interior reference: deep inside the cover face.
        interior = distance >= max_dist * 0.45
        if np.count_nonzero(interior) < 32:
            interior = distance >= max_dist * 0.25
        interior_colour = np.median(lab[interior], axis=0)

        max_rim = max(2, int(round(short * 0.055)))
        best_rim = max(1, int(round(short * 0.012)))
        best_score = -1.0

        for rim in range(1, max_rim + 1):
            band = (distance > 0) & (distance <= rim)
            if np.count_nonzero(band) < 16:
                continue
            band_colour = np.median(lab[band], axis=0)
            delta = float(np.linalg.norm(band_colour - interior_colour))
            # Prefer a rim that is chromatically distinct but not huge.
            score = delta - rim * 0.35
            if score > best_score and delta > 8.0:
                best_score = score
                best_rim = rim

        # Always keep a tiny production safety peel even on seamless photos.
        # Cap aggressively — over-peel was leaving a grey unprinted border
        # on clear cases / MagSafe shots where wrap should reach the rim.
        return int(np.clip(best_rim, 1, min(max_rim, max(1, int(round(short * 0.014))))))

    @staticmethod
    def _estimate_corner_radius(
        cover_mask: np.ndarray, cover_quad: np.ndarray
    ) -> float:
        """
        Corner roundness as percent of the cover's short edge.

        Measured from the photo silhouette — works for any phone / case shape
        (sharp old bars, modern soft corners, thick TPU, etc.). Not tied to a
        single model radius.
        """
        binary = (cover_mask > 0).astype(np.uint8)
        if np.count_nonzero(binary) < 64:
            return 6.0

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return 6.0
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 64:
            return 6.0

        rect = cv2.minAreaRect(contour)
        box = order_points(cv2.boxPoints(rect).astype(np.float32))
        short = float(max(min(rect[1]), 1.0))

        # Bisector inset at each sharp corner → local radius in pixels.
        radii_px: List[float] = []
        for i in range(4):
            corner = box[i]
            prev_pt = box[(i - 1) % 4]
            next_pt = box[(i + 1) % 4]
            to_prev = prev_pt - corner
            to_next = next_pt - corner
            n0 = float(np.linalg.norm(to_prev))
            n1 = float(np.linalg.norm(to_next))
            if n0 < 1e-3 or n1 < 1e-3:
                continue
            bisector = to_prev / n0 + to_next / n1
            bn = float(np.linalg.norm(bisector))
            if bn < 1e-3:
                continue
            direction = bisector / bn
            # Walk inward until we are clearly inside the filled cover.
            found = 0.0
            limit = int(short * 0.35)
            h, w = binary.shape[:2]
            for step in range(2, max(3, limit)):
                sample = corner + direction * float(step)
                x = int(round(sample[0]))
                y = int(round(sample[1]))
                if not (0 <= x < w and 0 <= y < h):
                    break
                if binary[y, x] > 0:
                    # Confirm a small neighbourhood is filled (past the arc).
                    x0, x1 = max(0, x - 1), min(w, x + 2)
                    y0, y1 = max(0, y - 1), min(h, y + 2)
                    if np.count_nonzero(binary[y0:y1, x0:x1]) >= 6:
                        found = float(step)
                        break
            if found > 0:
                radii_px.append(found)

        if radii_px:
            radius_px = float(np.median(radii_px))
            percent = 100.0 * radius_px / short
            return float(np.clip(percent, 2.5, 22.0))

        # Fallback: fill-ratio heuristic (still shape-agnostic).
        rect_area = max(float(rect[1][0] * rect[1][1]), 1.0)
        fill = float(np.clip(area / rect_area, 0.7, 1.0))
        missing = 1.0 - fill
        radius_px = missing * short * 1.8
        percent = 100.0 * radius_px / short
        return float(np.clip(percent, 2.5, 22.0))

    @staticmethod
    def wrap_target_mask(
        cover_mask: Optional[np.ndarray],
        phone_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Silhouette the print should fill — near the outer case/phone rim.

        If the estimated cover face is much smaller than the phone silhouette
        (common on clear cases), expand toward the phone edge so wrap does not
        leave a grey unprinted border.
        """
        if cover_mask is not None and np.count_nonzero(cover_mask):
            cover = CoverSurfaceEngine._manufacture_smooth_cover(cover_mask)
        else:
            cover = None
        if phone_mask is None or np.count_nonzero(phone_mask) == 0:
            return cover
        raw_phone = ((phone_mask > 0).astype(np.uint8)) * 255
        phone = CoverSurfaceEngine._manufacture_smooth_cover(phone_mask)
        if cover is not None and cover.shape[:2] != phone.shape[:2]:
            cover = cv2.resize(
                cover,
                (phone.shape[1], phone.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        short = min(phone.shape[:2])
        # Almost no inset — print should reach the visible outer rim.
        pad = max(0, int(round(short * 0.0008)))
        if pad > 0:
            phone_inner = cv2.erode(
                phone,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
                ),
                iterations=1,
            )
        else:
            phone_inner = phone.copy()
        # Never clip the smooth rim back onto the raw jagged silhouette —
        # that reintroduces stair-steps and leaves a white chalk border.
        # Generous envelope around the photo phone so full-bleed can reach the
        # visible edge while staying off the studio backdrop.
        env_pad = max(3, int(round(short * 0.008)))
        raw_env = cv2.dilate(
            raw_phone,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (env_pad * 2 + 1, env_pad * 2 + 1)
            ),
            iterations=1,
        )
        # Slight grow of the manufactured phone so wrap hugs the glass rim.
        rim_grow = max(2, int(round(short * 0.003)))
        phone_inner = cv2.dilate(
            phone_inner,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (rim_grow * 2 + 1, rim_grow * 2 + 1)
            ),
            iterations=1,
        )
        phone_inner = cv2.bitwise_and(phone_inner, raw_env)
        if cover is None or np.count_nonzero(cover) == 0:
            return phone_inner
        cover_area = float(np.count_nonzero(cover))
        phone_area = float(np.count_nonzero(phone_inner))
        if phone_area > 0 and cover_area < phone_area * 0.985:
            # Cover still inset — snap wrap to the phone rim (full-bleed).
            return phone_inner
        # Expand cover toward phone rim (never past phone_inner).
        grow = max(2, int(round(short * 0.018)))
        expanded = cv2.dilate(
            cover,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)
            ),
            iterations=1,
        )
        return cv2.bitwise_and(expanded, phone_inner)

    @staticmethod
    def symmetric_rim_gate(
        phone_mask: np.ndarray,
        quad: np.ndarray,
        corner_radius_percent: float,
        *,
        corner_radii: Optional[CornerRadii] = None,
        edge_inset_px: float = 0.0,
        silhouette_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Float outer-rim coverage from the existing silhouette contour.

        Supersampled fill of a lightly smoothed photo contour — same dimensions
        and corner radius (expand_px=0). No blur/feather reshape; sub-pixel AA
        comes from area-downsample only.
        """
        from .mesh import (
            AdaptiveMeshBuilder,
            _fill_closed_polyline_aa,
        )

        src = silhouette_mask if silhouette_mask is not None else phone_mask
        if float(np.max(src)) > 1.5:
            binary = (src > 127).astype(np.uint8) * 255
        else:
            binary = (src > 0.18).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 64:
            return None

        h, w = binary.shape[:2]
        expand = -float(abs(edge_inset_px))

        pts = AdaptiveMeshBuilder.outer_contour_polyline(binary, smooth=True)
        if pts is None or pts.shape[0] < 16:
            return None
        # Float AA of the existing contour — do not re-binarize into stairs.
        gate = _fill_closed_polyline_aa(
            pts,
            (h, w),
            scale=16,
            expand_px=expand,
        ).astype(np.float32)
        return np.clip(gate, 0.0, 1.0)

    @staticmethod
    def recommend_mesh_density(
        cover_mask: np.ndarray,
        corner_radius_percent: float = 8.0,
    ) -> Tuple[int, int]:
        """
        Pick mesh rows/cols from cover aspect + corner roundness.

        Tall phones get more rows; soft (large) corner radii get extra verts
        so Phase 5 adaptive spacing can approximate the arc cleanly.
        """
        from .mesh import adaptive_density_for_corners

        binary = cover_mask > 0
        if not np.count_nonzero(binary):
            return adaptive_density_for_corners(corner_radius_percent)
        ys, xs = np.nonzero(binary)
        height = float(max(ys.max() - ys.min(), 1))
        width = float(max(xs.max() - xs.min(), 1))
        aspect = height / width
        base_rows, base_cols = adaptive_density_for_corners(
            corner_radius_percent
        )
        if aspect >= 1.0:
            cols = base_cols
            rows = int(np.clip(round(cols * aspect * 0.72), 11, 21))
            rows = max(rows, base_rows)
        else:
            rows = base_rows
            cols = int(
                np.clip(round(rows / max(aspect, 0.35) * 0.72), 11, 17)
            )
            cols = max(cols, base_cols)
        return int(rows), int(cols)

    def _mesh_from_cover(
        self,
        cover_mask: np.ndarray,
        cover_quad: np.ndarray,
        rows: int,
        cols: int,
        corner_radius_percent: float,
        corner_radii: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[ControlMesh, np.ndarray, float]:
        """
        Build the printable mesh from the cover face, not the phone outline.

        Applies a print-safe margin and softens corners so artwork stays inside
        the rounded cover surface.
        """
        boundary = BoundaryEstimate(
            quad=cover_quad,
            contour=np.round(cover_quad).astype(np.int32).reshape(-1, 1, 2),
            mask=cover_mask,
            confidence=1.0,
        )
        mesh, printable_mask, margin_percent = (
            PrintableRegionDetector._mesh_from_boundary(boundary, rows, cols)
        )

        # Keep printable strictly inside the cover face — do not dilate outward
        # (dilation was letting the mesh sit past the real case edge).
        printable_mask = cv2.bitwise_and(printable_mask, cover_mask)
        if np.count_nonzero(printable_mask):
            # Geometric UV arcs keep edge winding stable; expand reaches the
            # photo rim without silhouette arc-length scramble.
            mesh = AdaptiveMeshBuilder.production_perimeter(
                mesh,
                cover_mask,
                corner_radius_percent=float(
                    np.clip(corner_radius_percent, 2.5, 22.0)
                ),
                max_move_fraction=0.04,
                corner_radii=corner_radii,
                preserve_corner_arcs=True,
            )
            # Printable must cover the settled mesh footprint (margin inset
            # alone leaves rim verts outside the printable gate).
            from .mesh import create_mesh_mask
            h, w = cover_mask.shape[:2]
            AdaptiveMeshBuilder._pull_boundary_inside(mesh, cover_mask)
            solid = (
                create_mesh_mask(
                    mesh,
                    (h, w),
                    feather_radius=0,
                    corner_radius_percent=float(
                        np.clip(corner_radius_percent, 2.5, 22.0)
                    ),
                    smooth_boundary=False,
                )
                * 255.0
            ).astype(np.uint8)
            pad = max(2, int(round(min(h, w) * 0.004)))
            solid = cv2.dilate(
                solid,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
                ),
                iterations=1,
            )
            printable_mask = cv2.bitwise_and(
                cv2.bitwise_or(printable_mask, solid), cover_mask
            )
        return mesh, printable_mask, margin_percent

    @staticmethod
    def _mask_quad(
        mask: np.ndarray, fallback: np.ndarray
    ) -> np.ndarray:
        """Stable perspective quad of a cover-face mask."""
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return order_points(fallback)
        contour = max(contours, key=cv2.contourArea)
        return PhoneBoundaryDetector._contour_quad(
            contour.astype(np.float32)
        )
