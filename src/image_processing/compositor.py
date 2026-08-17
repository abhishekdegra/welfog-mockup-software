"""
Compositing engine that prints a design onto a phone cover realistically.
"""

import copy
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import get_config
from ..utils.helpers import (
    add_grain, apply_vignette, clamp, order_points, to_bgr, to_bgra,
)
from .filters import ImageFilters
from .materials import (
    LIGHTING, MATERIALS, MaterialProfile, MaterialRenderingEngine,
    lighting_settings, material_settings,
)
from .mesh import (
    AdaptiveMeshBuilder, ControlMesh, MeshWarper, create_mesh_mask, mesh_aspect,
    DEFAULT_MESH_COLS, DEFAULT_MESH_ROWS, _sharp_quad_from_mesh,
)
from .cover_surface import CoverSurfaceEngine
from .smart_fit import SmartFitEstimator
from .template_cache import TemplateCache
from .device_template import CornerRadii, DeviceTemplateCatalog, estimate_corner_radii
from .curved_uv import (
    CurvedUVParams,
    DEFAULT_BEVEL_STRENGTH,
    DEFAULT_RIM_UV,
    estimate_rim_uv_from_margin,
)

logger = logging.getLogger("mockup.compositor")

# Outward grow for camera/lens openings. Side-bezel buttons need a fat punch,
# camera islands must hug the real hardware or the cutout looks oversized.
CAMERA_HOLE_EXPAND_PX = 2.25


DEFAULT_SETTINGS: Dict[str, float] = {
    # Placement
    'design_scale': 100.0,
    'offset_x': 0.0,
    'offset_y': 0.0,
    'rotation': 0.0,
    'region_inset': 0.0,
    'corner_radius': 11.0,
    # Phase 2 curved UV wrap (rim bevel foreshortening)
    'curved_uv': 1.0,          # 0 = off, 1 = on
    'rim_uv': 5.5,             # rim band % of cover short edge
    'bevel_strength': 92.0,    # wrap foreshortening strength %
    # Colour
    'exposure': 0.0,
    'brightness': 0.0,
    'contrast': 0.0,
    'highlights': 0.0,
    'shadows': 0.0,
    'gamma': 1.0,
    'temperature': 0.0,
    'tint': 0.0,
    'hue': 0.0,
    'saturation': 0.0,
    'vibrance': 0.0,
    # Detail
    'clarity': 0.0,
    'sharpness': 0.0,
    'blur': 0.0,
    'grain': 0.0,
    # Realism
    'opacity': 100.0,
    'edge_softness': 2.4,
    'texture_strength': 68.0,
    'reflection_strength': 42.0,
    'shadow_strength': 52.0,
    'tone_match': 22.0,
    'vignette': 0.0,
    # Lighting (scales reflections/highlights only)
    'lighting_reflection': 100.0,
    'lighting_highlight': 100.0,
    'lighting_softness': 50.0,
    'lighting_dir_x': 35.0,
    'lighting_dir_y': 22.5,
    # Phase 4 normal-based cover lighting
    'normal_lighting': 1.0,   # 0 = legacy lobes only
    'rim_bevel': 70.0,        # rim height amp %
    'ao_strength': 12.0,      # soft height AO %
    'micro_disp': 8.0,        # micro surface height %
}

# Look / material / lighting presets. Material and lighting entries drive the
# Material Rendering Engine; legacy looks remain for compatibility.
_LEGACY_LOOKS: Dict[str, Dict[str, float]] = {
    'Glossy Glass': {
        'texture_strength': 62.0, 'reflection_strength': 36.0,
        'shadow_strength': 35.0, 'contrast': 10.0, 'clarity': 8.0,
        'grain': 0.0, 'edge_softness': 1.2,
    },
    'Matte Silicone': {
        'texture_strength': 45.0, 'reflection_strength': 12.0,
        'shadow_strength': 40.0, 'contrast': -6.0, 'saturation': -8.0,
        'grain': 18.0, 'blur': 3.0, 'edge_softness': 1.8,
    },
    'Vivid Print': {
        'saturation': 22.0, 'vibrance': 18.0, 'contrast': 14.0,
        'clarity': 12.0, 'reflection_strength': 22.0,
        'texture_strength': 50.0,
    },
    'Soft Pastel': {
        'saturation': -18.0, 'brightness': 8.0, 'contrast': -10.0,
        'exposure': 6.0, 'texture_strength': 40.0,
        'reflection_strength': 18.0, 'grain': 8.0,
    },
    'Studio Product': {
        'texture_strength': 58.0, 'reflection_strength': 28.0,
        'shadow_strength': 40.0, 'clarity': 14.0, 'sharpness': 16.0,
        'tone_match': 22.0, 'vignette': 8.0,
    },
}

PRESETS: Dict[str, Dict[str, float]] = {
    'Default': {},
    **{name: material_settings(name) for name in MATERIALS},
    **{name: lighting_settings(name) for name in LIGHTING},
    **_LEGACY_LOOKS,
}

_DEFAULT_MATERIAL = 'Matte'
_DEFAULT_LIGHTING = 'Studio'
_LEGACY_MATERIAL_MAP = {
    'Glossy Glass': 'Glossy',
    'Matte Silicone': 'Silicon',
}


class Compositor:
    """
    Holds the phone image, the design, the cover region and all adjustments,
    and renders the composite at preview or full resolution.
    """

    PREVIEW_MAX = 1400

    def __init__(self, template_cache: Optional[TemplateCache] = None):
        cfg = get_config()
        self.settings: Dict[str, float] = dict(DEFAULT_SETTINGS)
        self.phone_image: Optional[np.ndarray] = None
        self.design_image: Optional[np.ndarray] = None
        self.control_mesh: Optional[ControlMesh] = None
        self.cover_points: Optional[np.ndarray] = None
        self.exclusion_mask: Optional[np.ndarray] = None
        self.printable_mask: Optional[np.ndarray] = None
        self.hardware_contours = []
        self.cutout_specs = []
        self.cutout_shape_tags: List[str] = []
        self.detection_confidence: float = 0.0
        self.automatic_margin: float = 0.0
        self.smart_fit_confidence: float = 0.0
        self.corner_radius_estimate: float = 6.0
        self._product_body_corner: Optional[float] = None
        self.corner_radii: CornerRadii = CornerRadii.uniform(6.0)
        self.from_template: bool = False
        self.model_id: str = ""
        self.curved_uv_params: CurvedUVParams = CurvedUVParams()
        self.fit_mode: str = 'fill'
        self.mirror: bool = False
        self.auto_detected: bool = False
        self.mesh_geometry_repaired: bool = False
        self.material_name: str = (
            cfg.default_material
            if cfg.default_material in MATERIALS
            else _DEFAULT_MATERIAL
        )
        self.lighting_name: str = (
            cfg.default_lighting
            if cfg.default_lighting in LIGHTING
            else _DEFAULT_LIGHTING
        )

        self.cover_engine = CoverSurfaceEngine(template_cache)
        self.material_engine = MaterialRenderingEngine()
        self._version = 0
        self._result_cache: "OrderedDict[Tuple[int, int], np.ndarray]" = OrderedDict()
        self._phone_wrap_mesh: Optional[ControlMesh] = None
        self._phone_wrap_mask: Optional[np.ndarray] = None
        self._phone_wrap_raw_mask: Optional[np.ndarray] = None
        self._phone_wrap_image_id: int = 0
        # Side volume/power ridges for wrap hug + relief shading (not punchouts).
        self._side_button_relief_mask: Optional[np.ndarray] = None
        # Per-button float coverage in the active composite space (wrap, not punch).
        self._side_button_wrap_cov: Optional[np.ndarray] = None
        self._side_button_validated_mask: Optional[np.ndarray] = None
        self._side_button_stem_bridge_mask: Optional[np.ndarray] = None
        self._scaled_phone_cache: "OrderedDict[int, Tuple[np.ndarray, float]]" = OrderedDict()

        # Seed default material + lighting floats onto settings.
        self.settings.update(material_settings(self.material_name))
        self.settings.update(lighting_settings(self.lighting_name))

    # ------------------------------------------------------------------ inputs

    def set_phone_image(self, image: np.ndarray) -> bool:
        """
        Set the phone image and detect its printable cover surface.

        The phone is only a geometric reference. Artwork targets the installed
        cover face. Previously corrected layouts are reused from the local
        template cache when the silhouette matches.
        """
        self.phone_image = to_bgr(image)
        self._invalidate_phone_wrap_cache()
        self.invalidate(clear_scaled=True)
        try:
            surface = self.cover_engine.analyze(self.phone_image)
            self._apply_cover_surface(surface, auto_detected=True)
            # Seed wrap assets from the photo so render hugs the device rim
            # even if the editable mesh is later dragged off-phone.
            self._ensure_phone_wrap_geometry(force=True)
            self._sync_printable_from_phone_wrap()
            # Align the editable cage to the wrap rim so Edit Mesh matches
            # what will actually render (still free to drag afterward).
            # Skip when a template correction was restored — user edits win.
            if self._phone_wrap_mesh is not None and not self.from_template:
                self.control_mesh = self._phone_wrap_mesh.copy()
                self.cover_points = self.control_mesh.corner_points()
                self.auto_detected = True
            # Volume buttons: wrap + raised relief (not punched). Fingerprint /
            # power holes stay manual (Erase) or Perfect Finish → Buttons.
        except Exception:
            logger.exception("Cover surface analysis failed")
            self.control_mesh = None
            self.cover_points = None
            self.exclusion_mask = None
            self.printable_mask = None
            self.auto_detected = False
            return False
        if self.design_image is not None:
            self.auto_fit_design()
        return self.control_mesh is not None

    def set_design_image(self, image: np.ndarray) -> None:
        """Set the design image, stored as BGRA so transparency survives."""
        self.design_image = to_bgra(image)
        if self.control_mesh is not None:
            self.auto_fit_design()
        self.invalidate()

    def clear(self) -> None:
        """Drop both images, the cover region and every adjustment."""
        self.phone_image = None
        self.design_image = None
        self.control_mesh = None
        self.cover_points = None
        self.exclusion_mask = None
        self.printable_mask = None
        self.hardware_contours = []
        self.cutout_specs = []
        self.cutout_shape_tags = []
        self.detection_confidence = 0.0
        self.automatic_margin = 0.0
        self.smart_fit_confidence = 0.0
        self.corner_radius_estimate = 6.0
        self._product_body_corner = None
        self.corner_radii = CornerRadii.uniform(6.0)
        self.from_template = False
        self.model_id = ""
        self.curved_uv_params = CurvedUVParams()
        self.auto_detected = False
        self.material_name = _DEFAULT_MATERIAL
        self.lighting_name = _DEFAULT_LIGHTING
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings.update(material_settings(self.material_name))
        self.settings.update(lighting_settings(self.lighting_name))
        self.fit_mode = 'fill'
        self.mirror = False
        self.invalidate(clear_scaled=True)

    def set_cover_points(self, points: np.ndarray) -> None:
        """
        Set legacy four-corner geometry and expand it into an editable mesh.

        This compatibility API keeps callers and old in-memory project data
        working while all new rendering uses piecewise-affine triangles.
        """
        if points is None:
            return

        self.cover_points = order_points(np.asarray(points, dtype=np.float32))
        # Legacy 4-corner API always expands at the current production density.
        self.control_mesh = ControlMesh.from_quad(
            self.cover_points,
            rows=DEFAULT_MESH_ROWS,
            cols=DEFAULT_MESH_COLS,
        )
        self.printable_mask = None
        self.auto_detected = False
        self.from_template = False
        self.invalidate()

    def set_control_mesh(self, mesh: ControlMesh) -> None:
        """Replace the editable mesh without disturbing exclusions/settings."""
        mesh = mesh.copy()
        corner = float(
            self.settings.get(
                "corner_radius", self.corner_radius_estimate or 8.0
            )
        )
        corner = float(np.clip(corner, 2.5, 22.0))

        # Upgrade accidental coarse cages (legacy 3×3 / 5×5) to production density.
        # Always keep the USER's 4 corners — never replace them with an old
        # detect-cover quad, and never auto-snap to the phone rim here (that
        # made dragged corners jump back on mouse-up). Full-bleed snap is only
        # for Perfect Finish → Edges.
        if mesh.rows < 7 or mesh.cols < 7:
            rows = max(DEFAULT_MESH_ROWS, mesh.rows)
            cols = max(DEFAULT_MESH_COLS, mesh.cols)
            mesh = ControlMesh.from_quad(mesh.corner_points(), rows, cols)
            mesh = AdaptiveMeshBuilder.force_rounded_perimeter(mesh, corner)

        # Do NOT auto production_perimeter / upright-AABB rebuild here.
        # Dragging one corner upward intentionally tilts the cage; "fixing"
        # that tilt snapped the handle back to the phone silhouette on release.
        # Load/analyze paths already rebuild upright cages; Perfect Finish →
        # Edges is the explicit snap-to-rim action.

        self.control_mesh = mesh
        self.cover_points = self.control_mesh.corner_points()
        # Manual mesh edits must NOT redefine wrap extent — warp always follows
        # the detected phone boundary (any phone / any design).
        self.auto_detected = False
        self.from_template = False
        self._sync_printable_from_phone_wrap()
        self._refresh_wrap_from_geometry()
        self._persist_manual_template()
        if self.design_image is not None:
            self.auto_fit_design(preserve_placement=True)
        else:
            self.invalidate()

    def _fullbleed_mesh_to_phone(
        self, mesh: ControlMesh, corner: float
    ) -> ControlMesh:
        """
        When the user's cage contains / overflows the phone, snap to the rim.

        Dragging corners far outside the device means "wrap the whole phone",
        not "leave a tilted sticker in the middle".
        """
        if (
            mesh is None
            or self.phone_image is None
            or mesh.rows < 2
            or mesh.cols < 2
        ):
            return mesh
        phone_mask = getattr(self.cover_engine, "last_phone_mask", None)
        if phone_mask is None or np.count_nonzero(phone_mask) == 0:
            return mesh

        from .cover_surface import CoverSurfaceEngine

        h, w = self.phone_image.shape[:2]
        pm = phone_mask
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
            pm = (pm > 127).astype(np.uint8) * 255

        solid = (
            create_mesh_mask(
                mesh,
                (h, w),
                feather_radius=0,
                corner_radius_percent=0.0,
                smooth_boundary=False,
            )
            * 255.0
        ).astype(np.uint8)
        phone_area = float(np.count_nonzero(pm))
        if phone_area < 64:
            return mesh
        covered = float(np.count_nonzero((pm > 0) & (solid > 0)))
        mesh_area = float(np.count_nonzero(solid))
        coverage = covered / phone_area
        # Frame swallows the phone, or spills past it while still covering most.
        should_snap = coverage >= 0.55 or (
            mesh_area >= phone_area * 0.95 and coverage >= 0.48
        )
        if not should_snap:
            return mesh

        gate = CoverSurfaceEngine.wrap_target_mask(None, pm)
        if gate is None or np.count_nonzero(gate) == 0:
            gate = pm
        try:
            measured = CoverSurfaceEngine._estimate_corner_radius(
                gate, mesh.corner_points()
            )
            if measured is not None:
                corner = float(np.clip(measured, 2.5, 22.0))
        except Exception:
            pass

        settled = AdaptiveMeshBuilder.production_perimeter(
            mesh,
            gate,
            corner_radius_percent=corner,
            max_move_fraction=0.35,
            corner_radii=self.corner_radii.as_tuple(),
            preserve_corner_arcs=True,
        )
        settled = settled.inset(-0.85)
        settled = AdaptiveMeshBuilder.force_rounded_perimeter(
            settled, corner, corner_radii=self.corner_radii.as_tuple(),
            adaptive=True,
        )
        rim_pad = max(2, int(round(min(h, w) * 0.0035)))
        rim = cv2.dilate(
            gate if gate is not None else pm,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (rim_pad * 2 + 1, rim_pad * 2 + 1)
            ),
            iterations=1,
        )
        # Mid-sides only — never remesh corners onto pixel stairs.
        AdaptiveMeshBuilder._snap_midsides_to_mask(
            settled, rim, smooth=True, max_move_fraction=0.18
        )
        AdaptiveMeshBuilder._expand_boundary_to_silhouette(
            settled, rim, corner_only=True
        )
        AdaptiveMeshBuilder._pull_boundary_inside(settled, rim)
        AdaptiveMeshBuilder._reinterpolate_interior(settled)
        self.corner_radius_estimate = corner
        self.settings["corner_radius"] = corner
        self.cover_engine.last_cover_mask = gate.copy()
        return settled

    def _phone_mask_looks_like_spilled_cage(self) -> bool:
        """True when mesh/mask is too large for a phone body in the frame."""
        if self.phone_image is None or self.control_mesh is None:
            return False
        h, w = self.phone_image.shape[:2]
        corners = self.control_mesh.corner_points()
        bw = float(corners[:, 0].max() - corners[:, 0].min())
        bh = float(corners[:, 1].max() - corners[:, 1].min())
        if bw > w * 0.70 or bh > h * 0.94 or float(corners[:, 1].min()) < -1.0:
            return True
        pm = getattr(self.cover_engine, "last_phone_mask", None)
        if pm is None or np.count_nonzero(pm) < 64:
            return True
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
        phone_a = float(np.count_nonzero(pm > 127))
        frame = float(h * w)
        # Phone body rarely fills > ~80% of a studio still.
        if phone_a > frame * 0.82:
            return True
        return False

    def _snap_mesh_to_phone_if_oversized(self) -> bool:
        """
        If the edit cage spills past the phone body, rebuild it on a tight AABB.

        Uses the cached phone mask only (no GrabCut) so sync/render stay fast.
        Perfect Finish refreshes that mask first.
        """
        if self.phone_image is None or self.control_mesh is None:
            return False
        from .mesh import AdaptiveMeshBuilder, ControlMesh

        h, w = self.phone_image.shape[:2]
        pm = getattr(self.cover_engine, "last_phone_mask", None)
        if pm is None or np.count_nonzero(pm) < 64:
            return False
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
            pm = (pm > 127).astype(np.uint8) * 255

        corner = float(
            np.clip(
                float(
                    self.settings.get(
                        "corner_radius", self.corner_radius_estimate or 11.0
                    )
                    or 11.0
                ),
                9.0,
                14.0,
            )
        )
        mesh_prior = (
            create_mesh_mask(
                self.control_mesh,
                (h, w),
                feather_radius=0,
                corner_radius_percent=corner,
                smooth_boundary=True,
                prefer_live_boundary=False,
            )
            * 255.0
        ).astype(np.uint8)
        mesh_a = float(np.count_nonzero(mesh_prior > 40))
        phone_a = float(np.count_nonzero(pm > 127))
        if mesh_a < 64 or phone_a < 64:
            return False
        overlap = float(
            np.count_nonzero((mesh_prior > 40) & (pm > 127))
        )
        spill = 1.0 - (overlap / mesh_a)
        if mesh_a <= phone_a * 1.08 and spill <= 0.10:
            return False

        quad = AdaptiveMeshBuilder._tight_aabb_quad_from_mask(pm)
        if quad is None:
            quad = AdaptiveMeshBuilder._aabb_quad_from_mask(pm)
        if quad is None:
            return False
        radii = (corner, corner, corner, corner)
        settled = ControlMesh.from_quad(
            order_points(np.asarray(quad, dtype=np.float32)),
            self.control_mesh.rows,
            self.control_mesh.cols,
            adaptive=True,
        )
        settled = AdaptiveMeshBuilder.force_rounded_perimeter(
            settled, corner, corner_radii=radii, adaptive=True
        )
        settled = AdaptiveMeshBuilder.densify_for_curvature(
            settled, corner, corner_radii=radii
        )
        AdaptiveMeshBuilder._pull_boundary_inside(settled, pm)
        AdaptiveMeshBuilder._reinterpolate_interior(settled)

        self.control_mesh = settled
        self.cover_points = settled.corner_points()
        self.corner_radii = CornerRadii.uniform(corner)
        self.corner_radius_estimate = corner
        self.settings["corner_radius"] = corner
        self.auto_detected = True
        self.mesh_geometry_repaired = True
        return True

    def _invalidate_phone_wrap_cache(self) -> None:
        self._phone_wrap_mesh = None
        self._phone_wrap_mask = None
        self._phone_wrap_raw_mask = None
        self._phone_wrap_image_id = 0
        self._side_button_relief_mask = None
        self._side_button_wrap_cov = None
        self._side_button_validated_mask = None
        self._side_button_stem_bridge_mask = None
        self._product_body_corner = None

    @staticmethod
    def _stem_bridge_between_tips_and_body(
        phone_mask: np.ndarray,
        tip_mask: np.ndarray,
        raw_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Photo-silhouette pixels between validated tips and the body wall.

        Reclaims the 1px orphan stems from wall smoothing — no invented geometry.
        """
        h, w = phone_mask.shape[:2]
        bridge = np.zeros((h, w), dtype=np.uint8)
        if raw_mask is None or np.count_nonzero(tip_mask) < 4:
            return bridge
        raw = raw_mask > 127
        body = phone_mask > 127
        tips = tip_mask > 127
        for y in np.unique(np.where(tips)[0]):
            txs = np.where(tips[y])[0]
            bxs = np.where(body[y])[0]
            if len(txs) == 0 or len(bxs) == 0:
                continue
            tl, tr = int(txs.min()), int(txs.max())
            bl, br = int(bxs.min()), int(bxs.max())
            if tr < bl:
                for x in range(tr + 1, bl):
                    if raw[y, x] and not body[y, x]:
                        bridge[y, x] = 255
            elif tl > br:
                for x in range(br + 1, tl):
                    if raw[y, x] and not body[y, x]:
                        bridge[y, x] = 255
        return bridge

    @staticmethod
    def _raw_protrusion_paint_for_components(
        phone_mask: np.ndarray,
        tip_mask: np.ndarray,
        raw_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        All photo-silhouette pixels for each validated button component.

        Covers partial tip rows (e.g. y=297 with only x=25) without inventing
        geometry — only ``raw & ~body`` inside each component's vertical band.
        """
        h, w = phone_mask.shape[:2]
        paint = (tip_mask > 127).copy()
        if raw_mask is None or np.count_nonzero(tip_mask) < 4:
            return paint.astype(np.uint8) * 255
        raw_out = (raw_mask > 127) & ~(phone_mask > 127)
        body = phone_mask > 127
        btn_u8 = (tip_mask > 127).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            btn_u8, connectivity=8
        )
        for label in range(1, num):
            y0 = int(stats[label, cv2.CC_STAT_TOP])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            x0 = int(stats[label, cv2.CC_STAT_LEFT])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            cx = x0 + 0.5 * float(bw)
            for y in range(y0, min(h, y0 + bh)):
                bxs = np.where(body[y])[0]
                if len(bxs) == 0:
                    continue
                xl, xr = int(bxs.min()), int(bxs.max())
                if cx < 0.5 * float(w):
                    xs_range = range(0, xl)
                else:
                    xs_range = range(xr + 1, w)
                for x in xs_range:
                    if raw_out[y, x]:
                        paint[y, x] = True
        return paint.astype(np.uint8) * 255

    def _side_button_paint_mask_for_size(
        self,
        hw: Tuple[int, int],
        phone_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Validated tips + full photo protrusion per button component."""
        vm = self._side_button_validated_mask
        if vm is None or np.count_nonzero(vm) < 4:
            return None
        ch, cw = int(hw[0]), int(hw[1])
        pm = phone_mask if phone_mask is not None else self._phone_wrap_mask
        if pm is None:
            if vm.shape[:2] != (ch, cw):
                return cv2.resize(
                    vm.astype(np.uint8),
                    (cw, ch),
                    interpolation=cv2.INTER_NEAREST,
                )
            return vm.copy()
        if pm.shape[:2] != (ch, cw):
            pm = cv2.resize(pm, (cw, ch), interpolation=cv2.INTER_NEAREST)
        tips_u8 = vm
        if vm.shape[:2] != (ch, cw):
            tips_u8 = cv2.resize(
                vm.astype(np.uint8),
                (cw, ch),
                interpolation=cv2.INTER_NEAREST,
            )
        raw = self._phone_wrap_raw_mask
        if raw is not None and raw.shape[:2] != (ch, cw):
            raw = cv2.resize(
                raw.astype(np.uint8),
                (cw, ch),
                interpolation=cv2.INTER_NEAREST,
            )
        stem = self._stem_bridge_between_tips_and_body(pm, tips_u8, raw)
        paint_u8 = self._raw_protrusion_paint_for_components(pm, tips_u8, raw)
        paint_u8 = np.maximum(paint_u8, stem)
        self._side_button_stem_bridge_mask = stem
        return paint_u8

    @staticmethod
    def _fill_1d_edge_profile(edge: np.ndarray) -> np.ndarray:
        """Fill NaN gaps in a per-row edge coordinate profile."""
        out = edge.astype(np.float32).copy()
        h = out.shape[0]
        valid = ~np.isnan(out)
        if not np.any(valid):
            return np.zeros(h, dtype=np.float32)
        first = int(np.flatnonzero(valid)[0])
        last = int(np.flatnonzero(valid)[-1])
        fill_val = float(out[valid][0])
        for y in range(first):
            out[y] = fill_val
        fill_val = float(out[valid][-1])
        for y in range(last + 1, h):
            out[y] = fill_val
        for y in range(first + 1, last + 1):
            if np.isnan(out[y]):
                out[y] = out[y - 1]
        return out

    @staticmethod
    def _straight_wall_reference(
        edge: np.ndarray,
        y_min: int,
        y_max: int,
        phone_h: float,
        *,
        side: str = "left",
    ) -> float:
        """Stable mid-body wall coordinate (ignores corner arcs and buttons)."""
        y_lo = y_min + int(round(phone_h * 0.18))
        y_hi = y_max - int(round(phone_h * 0.18))
        band = edge[y_lo : y_hi + 1]
        valid = ~np.isnan(band)
        if not np.any(valid):
            return float(np.nanmedian(edge))
        med = float(np.nanmedian(band[valid]))
        stable = valid & (np.abs(band - med) <= 1.25)
        if int(np.count_nonzero(stable)) >= 12:
            ref = band[stable]
        else:
            ref = band[valid]
        pct = 82.0 if side == "left" else 18.0
        return float(np.nanpercentile(ref, pct))

    def _midwall_aabb_quad_from_mask(
        self, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Axis-aligned product quad from mid-side walls (ignores tip outliers).
        """
        binary = (mask > 127).astype(np.uint8)
        if np.count_nonzero(binary) < 64:
            return None
        h, w = binary.shape[:2]
        ys, xs = np.where(binary)
        y_min = int(ys.min())
        y_max = int(ys.max())
        phone_h = max(float(y_max - y_min + 1), 1.0)
        edge_l = np.full(h, np.nan, dtype=np.float32)
        edge_r = np.full(h, np.nan, dtype=np.float32)
        for y in range(h):
            row = np.where(binary[y])[0]
            if len(row):
                edge_l[y] = float(row.min())
                edge_r[y] = float(row.max())
        el = self._fill_1d_edge_profile(edge_l)
        er = self._fill_1d_edge_profile(edge_r)
        wall_l = self._straight_wall_reference(
            el, y_min, y_max, phone_h, side="left"
        )
        wall_r = self._straight_wall_reference(
            er, y_min, y_max, phone_h, side="right"
        )
        if not np.isfinite(wall_l) or not np.isfinite(wall_r):
            return None
        if wall_r - wall_l < 8:
            return None
        return order_points(
            np.array(
                [
                    [wall_l, float(y_min)],
                    [wall_r, float(y_min)],
                    [wall_r, float(y_max)],
                    [wall_l, float(y_max)],
                ],
                dtype=np.float32,
            )
        )

    def _smooth_body_wall_from_raw(
        self,
        raw_mask: np.ndarray,
        quad: np.ndarray,
        button_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Photo body silhouette with validated tips removed.

        Mid-sides follow a light-smoothed photo edge (not raw pixel stairs,
        not a rectangle). Corners keep the same silhouette footprint and are
        only cleaned via contour smoothing — no enlarge/reshape.
        """
        from .mesh import AdaptiveMeshBuilder, _fill_closed_polyline_aa

        raw = (raw_mask > 127).astype(np.uint8) * 255
        h, w = raw.shape[:2]
        body = raw.copy()
        if button_mask is not None and np.count_nonzero(button_mask) >= 4:
            body = cv2.bitwise_and(
                body,
                cv2.bitwise_not((button_mask > 127).astype(np.uint8) * 255),
            )

        corners = order_points(np.asarray(quad, dtype=np.float32))
        y_min = int(np.floor(float(corners[:, 1].min())))
        y_max = int(np.ceil(float(corners[:, 1].max())))
        phone_h = max(float(y_max - y_min + 1), 1.0)

        edge_l = np.full(h, np.nan, dtype=np.float32)
        edge_r = np.full(h, np.nan, dtype=np.float32)
        for y in range(h):
            xs = np.where(body[y])[0]
            if len(xs):
                edge_l[y] = float(xs.min())
                edge_r[y] = float(xs.max())
        el = self._fill_1d_edge_profile(edge_l)
        er = self._fill_1d_edge_profile(edge_r)
        # Mid-sides: same stable wall used for tip detection (right edge is
        # already flat; left must not keep tip-strip / photo jitter).
        wall_l = self._straight_wall_reference(
            el, y_min, y_max, phone_h, side="left"
        )
        wall_r = self._straight_wall_reference(
            er, y_min, y_max, phone_h, side="right"
        )

        cleaned = np.zeros((h, w), dtype=np.uint8)
        edge_pts: list = []
        for y in range(h):
            xs = np.where(body[y])[0]
            if len(xs) == 0:
                continue
            xl_p = float(el[y]) if np.isfinite(el[y]) else float(xs.min())
            xr_p = float(er[y]) if np.isfinite(er[y]) else float(xs.max())
            # Continuous mix: photo corners ↔ straight mid-wall.
            # Smoothstep only removes the 1px join notch — it does not grow
            # the corner radius.
            v = (float(y) - float(y_min)) / phone_h
            near = min(max(v, 0.0), 1.0)
            near = min(near, 1.0 - near)
            t = float(np.clip((0.16 - near) / 0.08, 0.0, 1.0))
            t = t * t * (3.0 - 2.0 * t)
            xl = (1.0 - t) * float(wall_l) + t * xl_p
            xr = (1.0 - t) * float(wall_r) + t * xr_p
            xl_i = int(np.clip(np.round(xl), 0, w - 1))
            xr_i = int(np.clip(np.round(xr), xl_i, w - 1))
            cleaned[y, xl_i : xr_i + 1] = 255
            edge_pts.append((xl, float(y)))

        # Sub-pixel fill of the same 1D silhouette (no extra rounding).
        pts = None
        if len(edge_pts) >= 16:
            left = np.asarray(edge_pts, dtype=np.float32)
            right = []
            for y in range(h - 1, -1, -1):
                xs = np.where(cleaned[y])[0]
                if len(xs) == 0:
                    continue
                # Match the float right edge at this row when recorded.
                xr_p = float(er[y]) if np.isfinite(er[y]) else float(xs.max())
                v = (float(y) - float(y_min)) / phone_h
                near = min(max(v, 0.0), 1.0)
                near = min(near, 1.0 - near)
                t = float(np.clip((0.16 - near) / 0.08, 0.0, 1.0))
                t = t * t * (3.0 - 2.0 * t)
                xr = (1.0 - t) * float(wall_r) + t * xr_p
                right.append((xr, float(y)))
            pts = np.vstack(
                [left, np.asarray(right, dtype=np.float32)]
            )
        if pts is None or pts.shape[0] < 16:
            pts = AdaptiveMeshBuilder.outer_contour_polyline(
                cleaned, smooth=False
            )
        if pts is None or pts.shape[0] < 16:
            return cleaned
        cov = _fill_closed_polyline_aa(
            pts, (h, w), scale=16, expand_px=0.0
        )
        out = (cov >= 0.5).astype(np.uint8) * 255
        # Keep interior solid; never re-add tip pixels.
        core = cv2.erode(
            cleaned,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        out = cv2.bitwise_or(out, core)
        if button_mask is not None and np.count_nonzero(button_mask) >= 4:
            out = cv2.bitwise_and(
                out,
                cv2.bitwise_not((button_mask > 127).astype(np.uint8) * 255),
            )
        if np.count_nonzero(out) < 64:
            return cleaned
        return out

    @staticmethod
    def _fit_product_corner_radius(
        body_mask: np.ndarray,
        quad: np.ndarray,
        *,
        lo: float = 11.0,
        hi: float = 15.5,
    ) -> float:
        """
        Pick a corner radius (%) so geometric arcs match the photo rim.

        Bisector estimates often return tiny % values that produce chamfers.
        Scores AABB rounded-rects against the silhouette edge in corner bands.
        """
        from .mesh import (
            _fill_closed_polyline_aa,
            _sample_rounded_quad_perimeter,
        )

        binary = (body_mask > 127).astype(np.uint8)
        if np.count_nonzero(binary) < 64:
            return 12.0
        pq = order_points(np.asarray(quad, dtype=np.float32))
        h, w = binary.shape[:2]
        ys, xs = np.where(binary)
        y0, y1 = int(ys.min()), int(ys.max())
        ph = max(y1 - y0, 1)
        band = max(10, int(round(ph * 0.055)))

        def score(radius: float) -> float:
            cov = _fill_closed_polyline_aa(
                _sample_rounded_quad_perimeter(
                    pq, radius, samples_per_edge=96
                ),
                (h, w),
                scale=8,
                expand_px=0.0,
            )
            err = 0.0
            n = 0
            for y in list(range(y0, y0 + band)) + list(
                range(max(y0, y1 - band + 1), y1 + 1)
            ):
                bx = np.where(binary[y])[0]
                rx = np.where(cov[y] >= 0.5)[0]
                if len(bx) == 0 or len(rx) == 0:
                    continue
                err += abs(float(bx.min()) - float(rx.min()))
                err += abs(float(bx.max()) - float(rx.max()))
                n += 2
            return err / max(n, 1)

        best_r, best_e = 12.0, 1e9
        # Prefer slightly larger radii (smoother product look) when scores tie.
        for r in np.linspace(lo, hi, 22):
            e = score(float(r))
            # Soft prior toward ~12.5% — tiny radii score well but look chamfered.
            e = e + 0.12 * abs(float(r) - 12.5)
            if e < best_e:
                best_e, best_r = e, float(r)
        return float(np.clip(best_r, lo, hi))

    def _build_product_body_mask(
        self,
        raw_mask: np.ndarray,
        button_mask: Optional[np.ndarray],
        corner_radius: float,
        quad: np.ndarray,
    ) -> np.ndarray:
        """
        Product body from the photo silhouette.

        Mid-sides stay on the cleaned photo wall. Corners keep the real
        phone contour (contour-smoothed only) — never a generic rounded-rect.
        Button tips stay excluded from the body.
        """
        raw = (raw_mask > 127).astype(np.uint8) * 255
        body = self._smooth_body_wall_from_raw(raw, quad, button_mask)

        if button_mask is not None and np.count_nonzero(button_mask) >= 4:
            tips = (button_mask > 127) & (raw > 0)
            dist_in = cv2.distanceTransform(
                (body > 127).astype(np.uint8), cv2.DIST_L2, 5
            )
            tips = tips & (dist_in <= 3.0)
            body = body.copy()
            body[tips] = 0

        if np.count_nonzero(body) < 64:
            return raw
        return body

    def _derive_clean_body_and_button_masks(
        self,
        raw_mask: np.ndarray,
        phone_bgr: np.ndarray,
        *,
        exclusion_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Split the photo silhouette into a smooth body wall and localized buttons.

        Body mid-sides are clipped to a straight wall so zig-zag disappears.
        Corners keep the photo silhouette footprint (gate/mesh supply the
        perfect round arcs). Buttons are only the real outward tip pixels
        past that wall — never synthetic capsules.
        """
        from .mesh import AdaptiveMeshBuilder

        raw = (raw_mask > 127).astype(np.uint8) * 255
        h, w = raw.shape[:2]
        if np.count_nonzero(raw) < 64:
            return raw, None

        quad = AdaptiveMeshBuilder._aabb_quad_from_mask(raw)
        if quad is None:
            return raw, None

        corners = order_points(np.asarray(quad, dtype=np.float32))
        y_min = int(np.floor(float(corners[:, 1].min())))
        y_max = int(np.ceil(float(corners[:, 1].max())))
        phone_h = max(float(y_max - y_min + 1), 1.0)

        edge_l = np.full(h, np.nan, dtype=np.float32)
        edge_r = np.full(h, np.nan, dtype=np.float32)
        for y in range(h):
            xs = np.where(raw[y])[0]
            if len(xs):
                edge_l[y] = float(xs.min())
                edge_r[y] = float(xs.max())

        el = self._fill_1d_edge_profile(edge_l)
        er = self._fill_1d_edge_profile(edge_r)
        wall_l = self._straight_wall_reference(
            el, y_min, y_max, phone_h, side="left"
        )
        wall_r = self._straight_wall_reference(
            er, y_min, y_max, phone_h, side="right"
        )

        # Local prominence vs short smooth — catches 1–2 px real tips.
        smooth_l = cv2.GaussianBlur(el.reshape(-1, 1), (0, 0), 5.0).ravel()
        smooth_r = cv2.GaussianBlur(er.reshape(-1, 1), (0, 0), 5.0).ravel()
        y_lo = y_min + int(round(phone_h * 0.12))
        y_hi = y_max - int(round(phone_h * 0.08))
        min_span = max(4, int(round(phone_h * 0.008)))
        max_span = max(8, int(round(phone_h * 0.12)))

        buttons = np.zeros((h, w), dtype=np.uint8)
        found_any = False

        for side in ("left", "right"):
            if side == "left":
                outward = smooth_l - el
                wall = wall_l
                edge = el
            else:
                outward = er - smooth_r
                wall = wall_r
                edge = er

            prom = outward - cv2.GaussianBlur(
                outward.reshape(-1, 1), (0, 0), 8.0
            ).ravel()

            active = np.zeros(h, dtype=bool)
            for y in range(y_lo, y_hi + 1):
                # Real tip past the straight wall + local prominence.
                past_wall = (
                    (edge[y] < wall - 0.45)
                    if side == "left"
                    else (edge[y] > wall + 0.45)
                )
                active[y] = past_wall and outward[y] >= 0.45 and prom[y] >= 0.22

            i = 0
            while i < h:
                if not active[i]:
                    i += 1
                    continue
                j = i
                while j < h and active[j]:
                    j += 1
                span = j - i
                if min_span <= span <= max_span:
                    for y in range(max(0, i - 1), min(h, j + 1)):
                        if side == "left":
                            wx = int(np.ceil(wall))
                            xs = np.where(raw[y, :wx] > 0)[0]
                        else:
                            wx = int(np.floor(wall))
                            xs = np.where(raw[y, wx + 1 :] > 0)[0]
                            if len(xs):
                                xs = xs + wx + 1
                        if len(xs):
                            buttons[y, xs] = 255
                            found_any = True
                i = j

        if not found_any:
            corner = self._fit_product_corner_radius(
                self._smooth_body_wall_from_raw(raw, quad, None), quad
            )
            self._product_body_corner = float(corner)
            return self._build_product_body_mask(raw, None, corner, quad), None

        excl = (
            exclusion_mask
            if exclusion_mask is not None
            else self.exclusion_mask
        )
        cam_bin = self._side_button_camera_block(
            (h, w), excl, raw
        ) > 127
        validated = self._validate_side_button_mask(
            buttons, raw, quad, cam_bin
        )
        validated = cv2.bitwise_and(
            validated,
            cv2.bitwise_not(cam_bin.astype(np.uint8) * 255),
        )
        limited = self._limit_side_button_blobs(
            validated, quad, max_per_side=3
        )
        if limited is not None and np.count_nonzero(limited) >= 4:
            validated = limited

        if np.count_nonzero(validated) < 4:
            corner = self._fit_product_corner_radius(
                self._smooth_body_wall_from_raw(raw, quad, None), quad
            )
            self._product_body_corner = float(corner)
            return self._build_product_body_mask(raw, None, corner, quad), None

        # Photo silhouette (straight mid-sides, real corners). Tips stay
        # on a separate mask.
        straight = self._smooth_body_wall_from_raw(raw, quad, validated)
        corner = self._fit_product_corner_radius(straight, quad)
        self._product_body_corner = float(corner)
        body = self._build_product_body_mask(raw, validated, corner, quad)
        dist_in = cv2.distanceTransform(
            (body > 127).astype(np.uint8), cv2.DIST_L2, 5
        )
        tips = (validated > 127) & (raw > 0)
        # Wall smooth can orphan 1px stems between tip and body that are still
        # in the photo silhouette — reclaim those only (no invented geometry).
        orphan = (raw > 0) & (body == 0) & ~cam_bin
        near_tip = cv2.dilate(
            (validated > 127).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        ) > 0
        stem = orphan & near_tip & (dist_in <= 2.0)
        tips = (tips | stem) & (dist_in <= 2.5) & ~cam_bin
        # Claim each tip's full photo protrusion component (same raw blob),
        # so fragmented/short detections still wrap the real button height.
        tips = self._claim_full_raw_button_protrusions(
            tips, raw, body, cam_bin
        )
        # Silhouette nicks are location cues. Wrap the real device contour
        # of each key — never the studio-white overflow that caused speckles.
        snapped = self._snap_button_mask_to_device_surface(
            tips.astype(np.uint8) * 255, body, phone_bgr, raw
        )
        if np.count_nonzero(snapped) >= 4:
            validated = snapped
        else:
            validated = tips.astype(np.uint8) * 255
        plate = self._studio_plate_pixels(phone_bgr)
        if plate.shape[:2] == body.shape[:2]:
            before = int(np.count_nonzero(body))
            stripped = body.copy()
            stripped[plate] = 0
            if int(np.count_nonzero(stripped)) >= int(round(before * 0.97)):
                body = stripped
        return body, validated

    @staticmethod
    def _claim_full_raw_button_protrusions(
        tips: np.ndarray,
        raw: np.ndarray,
        body: np.ndarray,
        cam_bin: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Expand each validated tip to its full ``raw & ~body`` component.

        Device-agnostic: only claims photo silhouette pixels already outside
        the body wall. No synthetic capsules or coordinate tables.
        """
        tip_bool = tips.astype(bool) if tips.dtype != bool else tips
        if not np.any(tip_bool):
            return tip_bool
        h, w = tip_bool.shape[:2]
        raw_b = raw > 127 if raw.dtype != bool else raw
        body_b = body > 127 if body.dtype != bool else body
        raw_out = raw_b & ~body_b
        if cam_bin is not None:
            cam = cam_bin if cam_bin.dtype == bool else cam_bin > 0
            if cam.shape[:2] == (h, w):
                raw_out = raw_out & ~cam
        if not np.any(raw_out):
            return tip_bool
        n_raw, lab_raw, st_raw, _ = cv2.connectedComponentsWithStats(
            (raw_out.astype(np.uint8) * 255), connectivity=8
        )
        n_tip, lab_tip, _, _ = cv2.connectedComponentsWithStats(
            (tip_bool.astype(np.uint8) * 255), connectivity=8
        )
        out = tip_bool.copy()
        max_w = max(4, int(round(min(h, w) * 0.035)))
        max_h = max(8, int(round(h * 0.32)))
        for ti in range(1, n_tip):
            tip_m = lab_tip == ti
            if not np.any(tip_m):
                continue
            labs = np.unique(lab_raw[tip_m])
            labs = labs[labs > 0]
            for rl in labs:
                bw = int(st_raw[rl, cv2.CC_STAT_WIDTH])
                bh = int(st_raw[rl, cv2.CC_STAT_HEIGHT])
                area = int(st_raw[rl, cv2.CC_STAT_AREA])
                if area < 4 or bw > max_w or bh > max_h:
                    continue
                # Prefer thin side keys (taller than wide).
                if bw > 1 and bh < bw:
                    continue
                out[lab_raw == rl] = True
        return out

    def _snap_button_mask_to_device_surface(
        self,
        tip_mask: np.ndarray,
        body: np.ndarray,
        phone_bgr: np.ndarray,
        raw_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Map each detected button onto real device pixels.

        Silhouette nicks on the studio card mark WHERE a key sits; the wrap
        mask is the outer device contour in that span (true dark protrusions
        if present, otherwise the key's visible side-face). Studio-white
        pixels are never wrapped. Body wall geometry is unchanged.
        """
        if tip_mask is None or np.count_nonzero(tip_mask) < 4:
            return np.zeros(body.shape[:2], dtype=np.uint8)
        h, w = body.shape[:2]
        tips = tip_mask > 127
        if tips.shape[:2] != (h, w):
            tips = (
                cv2.resize(
                    (tip_mask > 127).astype(np.uint8) * 255,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 127
            )
        body_b = body > 127
        raw_b = body_b.copy()
        if raw_mask is not None and np.count_nonzero(raw_mask) >= 64:
            raw_u = raw_mask
            if raw_u.shape[:2] != (h, w):
                raw_u = cv2.resize(
                    raw_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            raw_b = raw_u > 127
        plate = self._studio_plate_pixels(phone_bgr)
        if plate.shape[:2] != (h, w):
            plate = (
                cv2.resize(
                    plate.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
        device = ~plate
        out = np.zeros((h, w), dtype=np.uint8)
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
            tips.astype(np.uint8) * 255, connectivity=8
        )
        mid_x = 0.5 * float(w)
        for lab in range(1, nlab):
            if int(stats[lab, cv2.CC_STAT_AREA]) < 4:
                continue
            y0 = int(stats[lab, cv2.CC_STAT_TOP])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            x0 = int(stats[lab, cv2.CC_STAT_LEFT])
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            left = (float(x0) + 0.5 * float(bw)) < mid_x
            y1 = min(h, y0 + bh)
            for y in range(y0, y1):
                owned = device[y] & (raw_b[y] | body_b[y])
                xs = np.where(owned)[0]
                if len(xs) == 0:
                    continue
                bxs = np.where(body_b[y] & device[y])[0]
                if left:
                    x_out = int(xs.min())
                    wall = int(bxs.min()) if len(bxs) else x_out
                    if x_out < wall:
                        out[y, x_out:wall] = 255
                    x_hi = min(w, wall + 2)
                    out[y, wall:x_hi] = np.where(
                        device[y, wall:x_hi] & body_b[y, wall:x_hi],
                        255,
                        out[y, wall:x_hi],
                    )
                else:
                    x_out = int(xs.max())
                    wall = int(bxs.max()) if len(bxs) else x_out
                    if x_out > wall:
                        out[y, wall + 1 : x_out + 1] = 255
                    x_lo = max(0, wall - 1)
                    out[y, x_lo : wall + 1] = np.where(
                        device[y, x_lo : wall + 1] & body_b[y, x_lo : wall + 1],
                        255,
                        out[y, x_lo : wall + 1],
                    )
        if int(np.count_nonzero(out)) < 4:
            kept = (tips & device).astype(np.uint8) * 255
            return kept if int(np.count_nonzero(kept)) >= 4 else out
        return out

    def _ensure_phone_wrap_geometry(
        self, *, force: bool = False
    ) -> Tuple[Optional[ControlMesh], Optional[np.ndarray]]:
        """
        Build the warp destination from the photo phone rim (dynamic).

        Edit-cage size is ignored: wrap is always exactly the device boundary —
        not smaller, not larger — for any phone model / studio shot.
        Side-button relief shading is optional and never invents silhouette bumps.
        """
        if self.phone_image is None:
            return None, None
        from .mesh import AdaptiveMeshBuilder, ControlMesh

        img_id = int(id(self.phone_image))
        if (
            not force
            and self._phone_wrap_mesh is not None
            and self._phone_wrap_mask is not None
            and self._phone_wrap_image_id == img_id
        ):
            return self._phone_wrap_mesh, self._phone_wrap_mask

        # Full photo rim for wrap clip — never manufacture-smooth (that insets
        # corners and leaves a silver gap). Oversized edit cages are ignored.
        pm = CoverSurfaceEngine.detect_phone_wrap_silhouette(
            self.phone_image, cover_quad=None
        )
        if pm is None or np.count_nonzero(pm) < 64:
            cage = (
                self.control_mesh.corner_points()
                if self.control_mesh is not None
                else None
            )
            pm = CoverSurfaceEngine.detect_phone_wrap_silhouette(
                self.phone_image, cover_quad=cage
            )
        if pm is None or np.count_nonzero(pm) < 64:
            # Last resort: legacy body detect (may be slightly inset).
            pm = CoverSurfaceEngine.detect_phone_body_mask(
                self.phone_image, cover_quad=None
            )
        if pm is None or np.count_nonzero(pm) < 64:
            return self._phone_wrap_mesh, self._phone_wrap_mask

        h, w = self.phone_image.shape[:2]
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
            pm = (pm > 127).astype(np.uint8) * 255

        # Heal pinholes; keep natural side-button tips from the photo.
        close_px = max(3, int(round(min(h, w) * 0.006)) | 1)
        body = cv2.morphologyEx(
            pm,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_px, close_px)
            ),
        )
        body = np.maximum(body, pm)

        # Split photo silhouette: body (no tips) vs real side-button tips.
        self._phone_wrap_raw_mask = body.copy()
        body_pm, btn_pm = self._derive_clean_body_and_button_masks(
            body, to_bgr(self.phone_image), exclusion_mask=self.exclusion_mask
        )
        if np.count_nonzero(body_pm) >= 64:
            body = body_pm
        if btn_pm is not None and np.count_nonzero(btn_pm) >= 4:
            self._side_button_validated_mask = btn_pm.copy()
            self._side_button_wrap_cov = None

        # Body already contour-cleaned in _derive. Do not subtract side-face
        # button pixels from the wall — that notched the silhouette.
        if self._side_button_validated_mask is not None:
            tips = self._side_button_validated_mask > 127
            plate = self._studio_plate_pixels(to_bgr(self.phone_image))
            if plate.shape[:2] == tips.shape[:2]:
                body[tips & plate] = 0

        # Drop silhouette pixels that sit on pure studio plate (bottom drip /
        # left speckles). Keeps 1px near real device content so corner AA of
        # the true phone rim is unchanged.
        body = self._strip_studio_overflow_mask(body, to_bgr(self.phone_image))

        # Relief shading only — body silhouette excludes button protrusions.
        wrap_pm, relief = self._volume_button_wrap_assets(body)
        if np.count_nonzero(wrap_pm) >= 64:
            body = wrap_pm

        # Topology quad from the photo body (ROI scale only — not the silhouette).
        quad = AdaptiveMeshBuilder._stable_quad_from_mask(body)
        if quad is None:
            quad = AdaptiveMeshBuilder._aabb_quad_from_mask(body)
        if quad is None:
            quad = AdaptiveMeshBuilder._tight_aabb_quad_from_mask(body)
        if quad is None:
            return self._phone_wrap_mesh, self._phone_wrap_mask

        # Corner % is derived from the photo silhouette for UV densify only.
        corner = self._fit_product_corner_radius(body, quad)
        try:
            cal_c, _ = AdaptiveMeshBuilder.calibrate_corner_radii_from_silhouette(
                body, quad, corner, (corner, corner, corner, corner)
            )
            corner = float(np.clip(0.55 * corner + 0.45 * float(cal_c), 6.0, 18.0))
        except Exception:
            corner = float(np.clip(corner, 6.0, 18.0))
        radii = (corner, corner, corner, corner)
        self.corner_radius_estimate = corner

        rows = (
            self.control_mesh.rows
            if self.control_mesh is not None
            else DEFAULT_MESH_ROWS
        )
        cols = (
            self.control_mesh.cols
            if self.control_mesh is not None
            else DEFAULT_MESH_COLS
        )
        # UV mesh: stable rounded topology snapped to the PHOTO body.
        # Coverage/gate use the silhouette mask — mesh must not redefine it.
        wrap = ControlMesh.from_quad(
            order_points(np.asarray(quad, dtype=np.float32)),
            rows,
            cols,
            adaptive=True,
        )
        wrap = AdaptiveMeshBuilder.force_rounded_perimeter(
            wrap, corner, corner_radii=radii, adaptive=True
        )
        AdaptiveMeshBuilder._snap_midsides_to_mask(
            wrap,
            body,
            smooth=True,
            max_move_fraction=0.06,
        )
        AdaptiveMeshBuilder._expand_boundary_to_silhouette(
            wrap, body, corner_only=True, corner_span=5
        )
        AdaptiveMeshBuilder._pull_boundary_inside(wrap, body)
        AdaptiveMeshBuilder._refine_corners(wrap, body, corner)
        AdaptiveMeshBuilder._straighten_sides(wrap, passes=1)
        AdaptiveMeshBuilder._reinterpolate_interior(wrap)

        wrap_pm = body
        self._phone_wrap_mesh = wrap
        self._phone_wrap_mask = wrap_pm.copy()
        self._phone_wrap_image_id = img_id
        self._side_button_relief_mask = (
            None if relief is None else relief.copy()
        )
        self.cover_engine.last_phone_mask = wrap_pm.copy()
        self.corner_radii = CornerRadii.uniform(corner)
        self.corner_radius_estimate = corner
        self.settings["corner_radius"] = corner
        self._debug_export_geometry_masks()
        # Edit cage tracks wrap mesh; silhouette mask remains photo truth.
        if force and wrap is not None and not self.from_template:
            self.control_mesh = wrap.copy()
            self.cover_points = self.control_mesh.corner_points()
        return self._phone_wrap_mesh, self._phone_wrap_mask

    def _volume_button_wrap_assets(
        self, phone_mask: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect volume rockers for subtle rim shading only.

        Body silhouette excludes side-button protrusions; buttons are separate.
        """
        body = (phone_mask > 127).astype(np.uint8) * 255
        if self.phone_image is None or np.count_nonzero(body) < 64:
            return body, None
        from .mesh import AdaptiveMeshBuilder
        from .region_detector import HardwareRegionDetector

        phone = to_bgr(self.phone_image)
        h, w = phone.shape[:2]
        if body.shape[:2] != (h, w):
            body = cv2.resize(body, (w, h), interpolation=cv2.INTER_NEAREST)
            body = (body > 127).astype(np.uint8) * 255
        quad = AdaptiveMeshBuilder._aabb_quad_from_mask(body)
        if quad is None:
            return body, None

        raw = HardwareRegionDetector.detect_verified_side_hardware(
            phone, quad, phone_mask=body
        )
        volume = HardwareRegionDetector.filter_volume_button_mask(
            raw, quad
        )
        if volume is None or np.count_nonzero(volume) < 24:
            volume = HardwareRegionDetector.filter_volume_button_mask(
                raw, quad, allow_compact=True
            )
        if volume is None or np.count_nonzero(volume) < 24:
            return body, None

        # Thin bezel strip for shading — button stays on top of the wrap.
        short = float(min(h, w))
        k = max(3, int(round(short * 0.005)) | 1)
        relief = cv2.dilate(
            volume,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
        rim_k = max(5, int(round(short * 0.022)) | 1)
        rim_band = cv2.subtract(
            body,
            cv2.erode(
                body,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_k, rim_k)),
                iterations=1,
            ),
        )
        relief = cv2.bitwise_and(relief, rim_band)
        if np.count_nonzero(relief) < 16:
            relief = cv2.bitwise_and(
                cv2.dilate(
                    volume,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                    iterations=1,
                ),
                body,
            )
        return body, relief

    def _side_button_relief_from_photo(
        self, phone_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compatibility: volume-button relief mask from photo detection."""
        _, relief = self._volume_button_wrap_assets(phone_mask)
        return relief

    def _camera_block_mask(
        self,
        shape_hw: Tuple[int, int],
        exclusion_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Binary mask of camera / non-button cutouts in ``shape_hw`` space.

        Side-button wrap must never enter these pixels. Built from the same
        exclusion / camera contours used by the final composite.
        """
        h, w = int(shape_hw[0]), int(shape_hw[1])
        out = np.zeros((h, w), dtype=np.uint8)
        # Prefer explicit camera-like contours (correct space after scale).
        cam_contours = []
        if self.hardware_contours:
            cam_contours = self._camera_like_contours(list(self.hardware_contours))
            if not cam_contours:
                cam_contours = self._upper_cutouts(list(self.hardware_contours))
        if cam_contours:
            from .region_detector import HardwareRegionDetector

            for contour in cam_contours:
                pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] < 3:
                    continue
                # Contours live in phone-native space; scale into shape_hw.
                if self.phone_image is not None:
                    ph, pw = self.phone_image.shape[:2]
                    if (ph, pw) != (h, w) and ph > 0 and pw > 0:
                        pts = pts.copy()
                        pts[:, 0] *= float(w) / float(pw)
                        pts[:, 1] *= float(h) / float(ph)
                HardwareRegionDetector.paint_cutout_mask(
                    out,
                    pts,
                    analytical=True,
                    expand_override=0.0,
                )
        elif exclusion_mask is not None and np.count_nonzero(exclusion_mask) >= 16:
            excl = exclusion_mask
            if excl.shape[:2] != (h, w):
                excl = cv2.resize(excl, (w, h), interpolation=cv2.INTER_LINEAR)
            # Without contour tags, treat strong exclusion cores as blocked.
            out = ((excl >= 200).astype(np.uint8) * 255)
        # Slight dilate so wrap cannot kiss the camera rim.
        if np.count_nonzero(out) >= 16:
            out = cv2.dilate(
                out,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
        return out

    def _side_button_camera_block(
        self,
        shape_hw: Tuple[int, int],
        exclusion_mask: Optional[np.ndarray],
        body_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Camera guard for side-button wrap.

        Blocks cutout interiors but keeps the outer side-wall rim where volume
        buttons sit beside the camera island.
        """
        cam = self._camera_block_mask(shape_hw, exclusion_mask) > 127
        if body_u8 is None:
            body_u8 = self._phone_wrap_mask
        if body_u8 is None or body_u8.shape[:2] != cam.shape[:2]:
            return cam.astype(np.uint8) * 255
        body = (body_u8 > 127).astype(np.uint8)
        if np.count_nonzero(body) < 64:
            return cam.astype(np.uint8) * 255
        edge_px = float(max(3.0, min(shape_hw[0], shape_hw[1]) * 0.014))
        dist_in = cv2.distanceTransform(body, cv2.DIST_L2, 5).astype(np.float32)
        side_rim = (body > 0) & (dist_in <= edge_px * 2.0)
        interior_cam = cam & ~side_rim
        return interior_cam.astype(np.uint8) * 255

    def _bezel_edge_zone(
        self,
        body_u8: np.ndarray,
        quad: np.ndarray,
    ) -> np.ndarray:
        """
        Thin L/R band hugging the phone silhouette edge (source-of-truth bezel).

        Side buttons must intersect this zone — face/camera interior is excluded.
        """
        h, w = body_u8.shape[:2]
        body = (body_u8 > 127).astype(np.uint8)
        if np.count_nonzero(body) < 64:
            return np.zeros((h, w), dtype=np.uint8)
        corners = order_points(np.asarray(quad, dtype=np.float32))
        x_min = int(np.floor(float(corners[:, 0].min())))
        x_max = int(np.ceil(float(corners[:, 0].max())))
        y_min = int(np.floor(float(corners[:, 1].min())))
        y_max = int(np.ceil(float(corners[:, 1].max())))
        bw = max(1, x_max - x_min + 1)
        bh = max(1, y_max - y_min + 1)
        edge_px = float(max(3.0, min(h, w) * 0.014))
        dist_in = cv2.distanceTransform(body, cv2.DIST_L2, 5).astype(np.float32)
        dist_out = cv2.distanceTransform(
            (1 - body).astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float32)
        on_rim = (body > 0) & (dist_in <= edge_px)
        outside_tip = (body == 0) & (dist_out > 0) & (dist_out <= edge_px * 1.6)
        edge = (on_rim | outside_tip).astype(np.uint8) * 255
        side_band = max(5, int(round(bw * 0.10)))
        band = np.zeros((h, w), dtype=np.uint8)
        band[y_min : y_max + 1, x_min : min(w, x_min + side_band)] = 255
        band[y_min : y_max + 1, max(0, x_max - side_band + 1) : x_max + 1] = 255
        band[: y_min + int(bh * 0.07), :] = 0
        band[y_max - int(bh * 0.06) :, :] = 0
        return cv2.bitwise_and(edge, band)

    def _validate_side_button_mask(
        self,
        candidates: np.ndarray,
        body_u8: np.ndarray,
        quad: np.ndarray,
        cam_bin: np.ndarray,
    ) -> np.ndarray:
        """
        Keep only components on the silhouette bezel that are not camera/face pads.

        Camera overlap is stripped first so edge-tip remnants can survive.
        """
        h, w = candidates.shape[:2]
        binary = ((candidates > 127) & ~cam_bin).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 4:
            return np.zeros((h, w), dtype=np.uint8)
        body = (body_u8 > 127).astype(np.uint8)
        bezel = self._bezel_edge_zone(body_u8, quad)
        dist_in = cv2.distanceTransform(body, cv2.DIST_L2, 5).astype(np.float32)
        edge_px = float(max(3.0, min(h, w) * 0.014))
        corners = order_points(np.asarray(quad, dtype=np.float32))
        phone_w = max(
            float(corners[:, 0].max()) - float(corners[:, 0].min()), 1.0
        )
        phone_h = max(
            float(corners[:, 1].max()) - float(corners[:, 1].min()), 1.0
        )
        out = np.zeros((h, w), dtype=np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 4:
                continue
            comp = labels == label
            bw_c = int(stats[label, cv2.CC_STAT_WIDTH])
            bh_c = int(stats[label, cv2.CC_STAT_HEIGHT])
            if bw_c > phone_w * 0.10 or bh_c > phone_h * 0.22:
                continue
            bezel_frac = float(np.count_nonzero(comp & (bezel > 127))) / float(
                area
            )
            outside_frac = float(np.count_nonzero(comp & (body == 0))) / float(
                area
            )
            if bezel_frac + outside_frac < 0.28:
                continue
            deep_face = comp & (body > 0) & (dist_in > edge_px * 2.2)
            if float(np.count_nonzero(deep_face)) / float(area) > 0.38:
                continue
            out[comp] = 255
        return out

    def _debug_log_side_button_mask(
        self,
        mask: np.ndarray,
        *,
        space: str = "native",
    ) -> None:
        """Temporary debug: log stats and write red overlay on the phone photo."""
        if mask is None or np.count_nonzero(mask) < 4:
            logger.info("side_button [%s]: no validated mask", space)
            return
        binary = (mask > 127).astype(np.uint8)
        num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes = []
        total = int(np.count_nonzero(binary))
        for i in range(1, num):
            boxes.append(
                (
                    int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_WIDTH]),
                    int(stats[i, cv2.CC_STAT_HEIGHT]),
                    int(stats[i, cv2.CC_STAT_AREA]),
                )
            )
        logger.info(
            "side_button [%s]: %d px, %d component(s), boxes=%s",
            space,
            total,
            num - 1,
            boxes,
        )
        if self.phone_image is None:
            return
        try:
            from pathlib import Path

            vis = to_bgr(self.phone_image).copy()
            sel = mask > 127
            if sel.shape[:2] != vis.shape[:2]:
                sel_u8 = (mask > 127).astype(np.uint8) * 255
                sel_u8 = cv2.resize(
                    sel_u8,
                    (vis.shape[1], vis.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                sel = sel_u8 > 127
            vis[sel] = (
                vis[sel].astype(np.float32) * 0.35
                + np.array([0.0, 0.0, 255.0], dtype=np.float32) * 0.65
            ).astype(np.uint8)
            out_path = Path("data/debug/side_button_mask.png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), vis)
            logger.info("side_button debug overlay -> %s", out_path.resolve())
        except Exception as exc:
            logger.debug("side_button debug overlay failed: %s", exc)

    def _debug_export_geometry_masks(self) -> None:
        """Write GREEN=body, RED=buttons, BLUE=camera debug overlay."""
        if self.phone_image is None:
            return
        try:
            from pathlib import Path

            phone_bgr = to_bgr(self.phone_image)
            h, w = phone_bgr.shape[:2]
            vis = phone_bgr.copy()
            body = self._phone_wrap_mask
            btn = self._side_button_validated_mask
            excl = self.exclusion_mask
            cam = (
                self._camera_block_mask((h, w), excl)
                if excl is not None and np.count_nonzero(excl) >= 16
                else None
            )
            if body is not None:
                bm = body > 127
                if bm.shape[:2] != (h, w):
                    bm_u8 = cv2.resize(
                        (body > 127).astype(np.uint8) * 255,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    bm = bm_u8 > 127
                vis[bm] = (
                    vis[bm].astype(np.float32) * 0.72
                    + np.array([0.0, 255.0, 0.0], dtype=np.float32) * 0.28
                ).astype(np.uint8)
                ctrs, _ = cv2.findContours(
                    bm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(vis, ctrs, -1, (0, 255, 0), 1)
            if btn is not None and np.count_nonzero(btn) >= 4:
                bm = btn > 127
                if bm.shape[:2] != (h, w):
                    bm_u8 = cv2.resize(
                        (btn > 127).astype(np.uint8) * 255,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    bm = bm_u8 > 127
                vis[bm] = (
                    vis[bm].astype(np.float32) * 0.30
                    + np.array([0.0, 0.0, 255.0], dtype=np.float32) * 0.70
                ).astype(np.uint8)
            if cam is not None and np.count_nonzero(cam) >= 4:
                cm = cam > 127
                vis[cm] = (
                    vis[cm].astype(np.float32) * 0.35
                    + np.array([255.0, 0.0, 0.0], dtype=np.float32) * 0.65
                ).astype(np.uint8)
            out_path = Path("data/debug/geometry_debug_overlay.png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), vis)
            logger.info("geometry debug overlay -> %s", out_path.resolve())
        except Exception as exc:
            logger.debug("geometry debug overlay failed: %s", exc)

    def _detect_side_buttons_from_wrap_protrusion(
        self,
        body_mask: np.ndarray,
        quad: np.ndarray,
        phone_bgr: np.ndarray,
        cam_bin: np.ndarray,
        *,
        raw_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Localized side-button pixels from raw-vs-body silhouette difference.

        When ``raw_mask`` is available, buttons are exactly ``raw & ~body``
        with depth validation. Otherwise falls back to outward bump scanning
        on the body edge profile.
        """
        from .region_detector import HardwareRegionDetector

        h, w = body_mask.shape[:2]
        body = (body_mask > 127).astype(np.uint8)
        if np.count_nonzero(body) < 64:
            return None
        corners = order_points(np.asarray(quad, dtype=np.float32))
        y_min = int(np.floor(float(corners[:, 1].min())))
        y_max = int(np.ceil(float(corners[:, 1].max())))
        phone_h = max(float(y_max - y_min + 1), 1.0)
        phone_w = max(
            float(corners[:, 0].max()) - float(corners[:, 0].min()), 1.0
        )
        edge_px = float(max(3.0, min(h, w) * 0.014))
        tip_px = max(2, int(round(phone_w * 0.012)))

        raw_hw: Optional[np.ndarray] = None
        try:
            raw = HardwareRegionDetector.detect_verified_side_hardware(
                phone_bgr, quad, phone_mask=body
            )
            if raw is not None and np.count_nonzero(raw) >= 8:
                raw_hw = (raw > 127).astype(np.uint8)
        except Exception:
            raw_hw = None

        out = np.zeros((h, w), dtype=np.uint8)
        raw_pm = (
            (raw_mask > 127).astype(np.uint8)
            if raw_mask is not None and np.count_nonzero(raw_mask) >= 64
            else None
        )

        if raw_pm is not None:
            _, split = self._derive_clean_body_and_button_masks(
                raw_mask,
                phone_bgr,
                exclusion_mask=self.exclusion_mask,
            )
            if split is not None and np.count_nonzero(split) >= 4:
                out = cv2.max(out, split)

        for side in ("left", "right"):
            edge_x = np.full(h, np.nan, dtype=np.float32)
            for y in range(h):
                row = np.where(body[y])[0]
                if len(row):
                    edge_x[y] = float(
                        row.min() if side == "left" else row.max()
                    )
            valid = ~np.isnan(edge_x)
            if not np.any(valid):
                continue
            fill = float(np.nanmedian(edge_x[valid]))
            for y in range(h):
                if np.isnan(edge_x[y]):
                    edge_x[y] = edge_x[y - 1] if y > 0 else fill
            smooth = cv2.GaussianBlur(
                edge_x.reshape(-1, 1), (0, 0), 6.0
            ).ravel()
            if side == "left":
                outward = smooth - edge_x
            else:
                outward = edge_x - smooth
            trend = cv2.GaussianBlur(
                outward.reshape(-1, 1), (0, 0), 12.0
            ).ravel()
            prominence = outward - trend

            i = 0
            while i < h:
                if (
                    prominence[i] < 0.35
                    or outward[i] < 0.62
                    or outward[i] > 4.5
                ):
                    i += 1
                    continue
                j = i
                while (
                    j < h
                    and prominence[j] >= 0.35
                    and 0.62 <= outward[j] <= 4.5
                ):
                    j += 1
                span = j - i
                if span >= 3:
                    y0, y1 = i, j - 1
                    t0 = (y0 - y_min) / phone_h
                    t1 = (y1 - y_min) / phone_h
                    if 0.08 <= t0 and t1 <= 0.92:
                        y_lo = max(0, y0 - 1)
                        y_hi = min(h - 1, y1 + 1)
                        for y in range(y_lo, y_hi + 1):
                            row = np.where(body[y])[0]
                            if len(row) == 0:
                                continue
                            if side == "left":
                                xe = int(row.min())
                                x_lo, x_hi = xe, min(w, xe + tip_px)
                            else:
                                xe = int(row.max())
                                x_lo, x_hi = max(0, xe - tip_px + 1), xe + 1
                            sel = np.zeros(w, dtype=bool)
                            sel[x_lo:x_hi] = True
                            hw_row = (
                                (raw_hw[y] > 0) & sel
                                if raw_hw is not None
                                else np.zeros(w, dtype=bool)
                            )
                            tip_row = np.zeros(w, dtype=bool)
                            if raw_pm is not None:
                                raw_xs = np.where(raw_pm[y] > 0)[0]
                                if len(raw_xs) and len(row):
                                    if side == "left":
                                        xl = int(raw_xs.min())
                                        xb = int(row.min())
                                        if xl < xb:
                                            tip_row[xl:xb] = True
                                    else:
                                        xr = int(raw_xs.max())
                                        xb = int(row.max())
                                        if xr > xb:
                                            tip_row[xb + 1 : xr + 1] = True
                            rim_row = tip_row & (~cam_bin[y])
                            use = hw_row | rim_row
                            if np.any(use):
                                out[y, use] = 255
                i = max(j, i + 1)

        if np.count_nonzero(out) < 12:
            return None
        return out

    def _side_button_detection_mask_native(
        self,
        exclusion_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Validated side-button pixels at phone-native resolution.

        Buttons are localized masks split from the raw photo silhouette; the
        main ``_phone_wrap_mask`` is the smooth body only.
        """
        if self.phone_image is None:
            return None
        from .mesh import AdaptiveMeshBuilder
        from .region_detector import HardwareRegionDetector

        phone = to_bgr(self.phone_image)
        h, w = phone.shape[:2]
        pm = self._phone_wrap_mask
        if pm is None or np.count_nonzero(pm) < 64:
            return None
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
            pm = (pm > 127).astype(np.uint8) * 255
        quad = AdaptiveMeshBuilder._aabb_quad_from_mask(pm)
        if quad is None:
            return None

        excl = exclusion_mask if exclusion_mask is not None else self.exclusion_mask
        cam_bin = self._side_button_camera_block(
            (h, w), excl, pm
        ) > 127

        cached = self._side_button_validated_mask
        candidates = np.zeros((h, w), dtype=np.uint8)
        if cached is not None and np.count_nonzero(cached) >= 4:
            if cached.shape[:2] != (h, w):
                cached = cv2.resize(
                    cached, (w, h), interpolation=cv2.INTER_NEAREST
                )
                cached = (cached > 127).astype(np.uint8) * 255
            cached = cv2.bitwise_and(
                cached, cv2.bitwise_not(cam_bin.astype(np.uint8) * 255)
            )
            if np.count_nonzero(cached) >= 4:
                candidates = cv2.max(candidates, cached)

        raw_pm = self._phone_wrap_raw_mask
        if raw_pm is not None and raw_pm.shape[:2] != (h, w):
            raw_pm = cv2.resize(raw_pm, (w, h), interpolation=cv2.INTER_NEAREST)
            raw_pm = (raw_pm > 127).astype(np.uint8) * 255

        protr = self._detect_side_buttons_from_wrap_protrusion(
            pm, quad, phone, cam_bin, raw_mask=raw_pm
        )
        if protr is not None and np.count_nonzero(protr) >= 12:
            candidates = cv2.max(candidates, protr)
        tips = self._restore_photo_side_button_tips(pm)
        if tips is not None and np.count_nonzero(tips) >= 8:
            candidates = cv2.max(candidates, tips)
        # Legacy silhouette seeds draw rounded rects — skip; protrusion is exact.
        if np.count_nonzero(candidates) < 12:
            raw = HardwareRegionDetector.detect_verified_side_hardware(
                phone, quad, phone_mask=pm
            )
            if raw is not None and np.count_nonzero(raw) >= 8:
                candidates = cv2.max(
                    candidates, (raw > 127).astype(np.uint8) * 255
                )

        validated = self._validate_side_button_mask(
            candidates, pm, quad, cam_bin
        )
        validated = cv2.bitwise_and(
            validated, cv2.bitwise_not(cam_bin.astype(np.uint8) * 255)
        )
        limited = self._limit_side_button_blobs(validated, quad, max_per_side=3)
        if limited is not None and np.count_nonzero(limited) >= 4:
            validated = limited
        if np.count_nonzero(validated) < 4:
            self._side_button_validated_mask = None
            self._debug_log_side_button_mask(validated, space="native(empty)")
            return None
        self._side_button_validated_mask = validated.copy()
        self._debug_log_side_button_mask(validated, space="native")
        return validated

    def _build_side_button_wrap_coverage(
        self,
        composite_hw: Tuple[int, int],
        phone_mask: Optional[np.ndarray],
        exclusion_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Float AA coverage from detected button pixels only (phone-native space).

        Uses the existing side-button detection mask as-is — no synthetic
        capsules, outward extrusion, or bezel rebuild. Each component gets
        sub-pixel AA on its actual contour; coverage never extends past the
        detected footprint.
        """
        # Remove unused chaikin import path — coverage uses validated contour only.
        from .mesh import _fill_closed_polyline_aa

        if self.phone_image is None:
            return None
        ph, pw = self.phone_image.shape[:2]
        # Prefer the already-validated mask from geometry split — do not
        # re-detect / overwrite (that broke the wrap connection).
        det = self._side_button_validated_mask
        if det is not None and np.count_nonzero(det) >= 4:
            if det.shape[:2] != (ph, pw):
                det = cv2.resize(
                    det.astype(np.uint8),
                    (pw, ph),
                    interpolation=cv2.INTER_NEAREST,
                )
                det = (det > 127).astype(np.uint8) * 255
        else:
            det = self._side_button_detection_mask_native(exclusion_mask)
        if det is None or np.count_nonzero(det) < 4:
            return None

        cam_bin = self._side_button_camera_block(
            (ph, pw), exclusion_mask, self._phone_wrap_mask
        ) > 127
        btn = ((det > 127) & ~cam_bin).astype(np.uint8) * 255
        if np.count_nonzero(btn) < 4:
            return None

        # Exact validated tips + 1px float AA (same composite space as body).
        cov = np.zeros((ph, pw), dtype=np.float32)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            btn, connectivity=8
        )
        for label in range(1, num):
            if int(stats[label, cv2.CC_STAT_AREA]) < 4:
                continue
            comp = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                cov = np.maximum(cov, comp.astype(np.float32) / 255.0)
                continue
            pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
                np.float32
            )
            patch = _fill_closed_polyline_aa(
                pts, (ph, pw), scale=8, expand_px=0.0
            )
            # Exact tip footprint only — no dilate halo (that leaked soft wrap
            # into long vertical edge streaks outside the buttons).
            patch = np.maximum(patch, comp.astype(np.float32) / 255.0)
            patch = np.where(comp > 0, patch, 0.0)
            cov = np.maximum(cov, patch)

        cov = np.clip(cov, 0.0, 1.0)
        cov[cam_bin] = 0.0
        if float(np.max(cov)) < 0.05:
            return None

        ch, cw = int(composite_hw[0]), int(composite_hw[1])
        if (ph, pw) != (ch, cw):
            # Nearest for mask fidelity; coverage clipped to resized tip mask.
            vm_src = det if det is not None else btn
            vm = (vm_src > 127).astype(np.uint8) * 255
            vm = cv2.resize(vm, (cw, ch), interpolation=cv2.INTER_NEAREST)
            cam_c = self._side_button_camera_block(
                (ch, cw), exclusion_mask, self._phone_wrap_mask
            ) > 127
            vm = cv2.bitwise_and(
                vm, cv2.bitwise_not(cam_c.astype(np.uint8) * 255)
            )
            self._side_button_validated_mask = (vm > 127).astype(np.uint8) * 255
            cov = cv2.resize(cov, (cw, ch), interpolation=cv2.INTER_LINEAR)
            cov = np.where(vm > 127, cov, 0.0)
            cov[cam_c] = 0.0
        else:
            self._side_button_validated_mask = (btn > 127).astype(np.uint8) * 255
            cov = np.where(btn > 127, cov, 0.0)
        cov = np.clip(cov, 0.0, 1.0)
        self._debug_log_side_button_mask(
            self._side_button_validated_mask,
            space="validated",
        )
        return cov

    def _apply_side_button_wrap(
        self,
        design: np.ndarray,
        mask: np.ndarray,
        alpha: np.ndarray,
        design_alpha: np.ndarray,
        phone_mask: Optional[np.ndarray],
        exclusion_mask: Optional[np.ndarray],
        *,
        opacity: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Body-first pass: reserve validated tip pixels for the later button layer.

        Does not paint wrap onto tips here (that swallowed keys into the body).
        Coverage is built for the isolated composite pass; tip alpha is cleared
        so the body blend cannot own button silhouette pixels.
        """
        del design_alpha, opacity  # reserved for the isolated tip layer
        h, w = mask.shape[:2]
        btn_cov = self._side_button_wrap_cov
        if (
            btn_cov is None
            or btn_cov.shape[:2] != (h, w)
            or float(np.max(btn_cov)) < 0.05
        ):
            btn_cov = self._build_side_button_wrap_coverage(
                (h, w), phone_mask, exclusion_mask
            )
            self._side_button_wrap_cov = btn_cov

        vm = self._side_button_validated_mask
        if vm is None or np.count_nonzero(vm) < 4:
            return design, mask, alpha
        valid = vm > 127
        if valid.shape[:2] != (h, w):
            valid = (
                cv2.resize(
                    vm.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 127
            )
        if not np.any(valid):
            return design, mask, alpha
        # Reserve only true outward protrusions (outside the body wall).
        # Side-face overlap keeps the body wrap so keys cannot flash the
        # original bezel if the isolated layer misses a pixel.
        body_on = np.zeros(valid.shape, dtype=bool)
        if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
            pm = phone_mask
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
            body_on = pm > 127
        protr = valid & ~body_on
        if np.any(protr):
            alpha = np.where(protr, 0.0, alpha)
            mask = np.where(protr, 0.0, mask)
        return design, mask, alpha

    def _limit_side_button_blobs(
        self,
        mask: np.ndarray,
        quad: np.ndarray,
        *,
        max_per_side: int = 3,
    ) -> Optional[np.ndarray]:
        """Keep the strongest few L/R pills; drop leftover ghost blobs."""
        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 4:
            return None
        h, w = binary.shape[:2]
        corners = order_points(np.asarray(quad, dtype=np.float32))
        mid_x = 0.5 * (
            float(corners[:, 0].min()) + float(corners[:, 0].max())
        )
        out = np.zeros((h, w), dtype=np.uint8)
        for side in ("left", "right"):
            band = np.zeros((h, w), dtype=np.uint8)
            if side == "left":
                band[:, : int(mid_x)] = binary[:, : int(mid_x)]
            else:
                band[:, int(mid_x) :] = binary[:, int(mid_x) :]
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                band, connectivity=8
            )
            scored: List[Tuple[float, int]] = []
            for label in range(1, num):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < 4:
                    continue
                bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                bw = int(stats[label, cv2.CC_STAT_WIDTH])
                # Prefer taller thin bezel strips over wide corner noise.
                score = float(bh) * 3.0 + float(area) - 0.75 * float(bw)
                if bh < 3:
                    score *= 0.35
                scored.append((score, label))
            scored.sort(reverse=True)
            for _, label in scored[: max(1, int(max_per_side))]:
                out[labels == label] = 255
        return out if np.count_nonzero(out) >= 4 else None

    @staticmethod
    def _phone_body_without_side_bumps(mask: np.ndarray) -> np.ndarray:
        """
        Morphologically open small L/R protrusions so the rounded cage sits on
        the main body; button hugs are applied afterward on mid-sides only.
        """
        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 64:
            return binary
        h, w = binary.shape[:2]
        open_px = max(3, int(round(min(h, w) * 0.007)))
        body = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (open_px * 2 + 1, open_px * 2 + 1)
            ),
            iterations=1,
        )
        if np.count_nonzero(body) < np.count_nonzero(binary) * 0.90:
            return binary
        # Keep corners/area — never shrink into a half-phone bite.
        if np.count_nonzero(body) < np.count_nonzero(binary) * 0.97:
            # Only strip pixels in thin L/R bands (true button tips).
            ys, xs = np.where(binary > 0)
            x0, x1 = int(xs.min()), int(xs.max())
            band = max(4, int(round((x1 - x0 + 1) * 0.045)))
            strip = binary.copy()
            strip[:, x0 + band : max(x0 + band, x1 - band + 1)] = 0
            tips = cv2.bitwise_and(strip, cv2.bitwise_not(body))
            body = cv2.bitwise_and(binary, cv2.bitwise_not(tips))
            if np.count_nonzero(body) < np.count_nonzero(binary) * 0.90:
                return binary
        return body

    def _detect_side_button_regions(
        self, phone_mask: np.ndarray, quad: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Locate volume / power / side-FP ridges for wrap hug (not exclusions).
        """
        if self.phone_image is None:
            return None
        from .region_detector import HardwareRegionDetector

        phone = to_bgr(self.phone_image)
        h, w = phone.shape[:2]
        peripheral = np.zeros((h, w), dtype=np.uint8)
        HardwareRegionDetector._detect_side_hardware_fullres(
            phone, peripheral, quad, relaxed=False
        )
        raw_mask = CoverSurfaceEngine.estimate_phone_mask_from_photo(
            self.phone_image, cover_quad=quad
        )
        for cand in (raw_mask, phone_mask):
            if cand is None or np.count_nonzero(cand) < 64:
                continue
            sil = cand
            if sil.shape[:2] != (h, w):
                sil = cv2.resize(sil, (w, h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(sil > 127)
            if len(xs) < 32:
                continue
            bw = max(1, int(xs.max() - xs.min() + 1))
            bh = max(1, int(ys.max() - ys.min() + 1))
            fill = float(np.count_nonzero(sil > 127)) / float(bw * bh)
            if fill > 0.965:
                continue
            peripheral = cv2.max(
                peripheral,
                HardwareRegionDetector.detect_buttons_from_silhouette(
                    sil, quad
                ),
            )
        if int(np.count_nonzero(peripheral)) < max(60, int(min(h, w) * 0.6)):
            relaxed = np.zeros((h, w), dtype=np.uint8)
            HardwareRegionDetector._detect_side_hardware_fullres(
                phone, relaxed, quad, relaxed=True
            )
            peripheral = cv2.max(peripheral, relaxed)
        seeds = self._seed_side_button_mask(phone, quad)
        if seeds is not None:
            peripheral = cv2.max(peripheral, seeds)
        if np.count_nonzero(peripheral) < 40:
            return None
        return self._filter_side_button_hug_mask(peripheral, quad)

    def _filter_side_button_hug_mask(
        self, mask: np.ndarray, quad: np.ndarray
    ) -> Optional[np.ndarray]:
        """Keep compact L/R bezel pills; drop face/camera junk."""
        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 40:
            return None
        h, w = binary.shape[:2]
        corners = order_points(np.asarray(quad, dtype=np.float32))
        x_min = float(corners[:, 0].min())
        x_max = float(corners[:, 0].max())
        y_min = float(corners[:, 1].min())
        y_max = float(corners[:, 1].max())
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)
        side_band = width * 0.14
        out = np.zeros((h, w), dtype=np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        kept = 0
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 24 or area > int(h * w * 0.035):
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            near_side = (cx - x_min) <= side_band or (x_max - cx) <= side_band
            if not near_side:
                continue
            if bw > width * 0.18 or bh > height * 0.34:
                continue
            if cy < y_min + height * 0.10 or cy > y_max - height * 0.10:
                continue
            # Prefer tall capsules (volume) or compact pills (power / FP).
            aspect = max(bw, bh) / max(min(bw, bh), 1e-3)
            if aspect < 1.15 and max(bw, bh) > width * 0.10:
                continue
            # Reject very long wall slabs (ghost wraps along the whole side).
            if bh > height * 0.22 and bw < width * 0.04:
                continue
            if bh > height * 0.28:
                continue
            out[labels == label] = 255
            kept += 1
        if kept == 0 or np.count_nonzero(out) < 40:
            return None
        return out

    def _merge_side_button_clusters(
        self, mask: np.ndarray, quad: np.ndarray
    ) -> np.ndarray:
        """
        Merge vertically adjacent same-bezel detections into one capsule.

        Seed zones often paint 2–3 stacked pills for one volume rocker; merging
        keeps a single smooth bump instead of stair-steps.
        """
        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 40:
            return binary
        h, w = binary.shape[:2]
        corners = order_points(np.asarray(quad, dtype=np.float32))
        x_min = float(corners[:, 0].min())
        x_max = float(corners[:, 0].max())
        mid_x = 0.5 * (x_min + x_max)
        gap_tol = max(
            8.0,
            (float(corners[:, 1].max()) - float(corners[:, 1].min())) * 0.025,
        )
        out = np.zeros((h, w), dtype=np.uint8)
        for side in ("left", "right"):
            band = np.zeros((h, w), dtype=np.uint8)
            if side == "left":
                band[:, : int(mid_x)] = binary[:, : int(mid_x)]
            else:
                band[:, int(mid_x) :] = binary[:, int(mid_x) :]
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                band, connectivity=8
            )
            boxes = []
            for label in range(1, num):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < 16:
                    continue
                x = int(stats[label, cv2.CC_STAT_LEFT])
                y = int(stats[label, cv2.CC_STAT_TOP])
                bw = int(stats[label, cv2.CC_STAT_WIDTH])
                bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                boxes.append([x, y, x + bw, y + bh, label])
            if not boxes:
                continue
            boxes.sort(key=lambda b: b[1])
            clusters: List[List[int]] = []
            cur = [0]
            for i in range(1, len(boxes)):
                prev = boxes[cur[-1]]
                nxt = boxes[i]
                gap = float(nxt[1] - prev[3])
                x_overlap = min(prev[2], nxt[2]) - max(prev[0], nxt[0])
                if gap <= gap_tol and x_overlap >= -3:
                    cur.append(i)
                else:
                    clusters.append(cur)
                    cur = [i]
            clusters.append(cur)
            for group in clusters:
                x1 = min(boxes[i][0] for i in group)
                y1 = min(boxes[i][1] for i in group)
                x2 = max(boxes[i][2] for i in group)
                y2 = max(boxes[i][3] for i in group)
                short = min(x2 - x1, y2 - y1)
                corner = max(2, short // 2)
                from .region_detector import HardwareRegionDetector

                HardwareRegionDetector._rounded_rectangle(
                    out, x1, y1, x2, y2, corner
                )
        return out if np.count_nonzero(out) else binary

    def _hug_side_buttons_into_wrap_mask(
        self, body: np.ndarray, raw_pm: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Restore real side-button tips onto a smooth wrap silhouette.

        ``_manufacture_smooth_cover`` erases volume/power protrusions, so the
        design stops on a flat wall while white button tips stick out. Pull
        those photo tips back with soft AA, and build a relief mask for raised
        shading (design stays on the buttons — no white punchouts).
        """
        from .mesh import AdaptiveMeshBuilder

        body_u8 = (body > 127).astype(np.uint8) * 255
        raw_u8 = (raw_pm > 127).astype(np.uint8) * 255
        wrap_mask = raw_u8.copy()
        h, w = body_u8.shape[:2]

        # Photo tips just outside the smoothed body (true hardware bumps).
        tips = self._restore_photo_side_button_tips(body_u8)
        if tips is not None and np.count_nonzero(tips) >= 24:
            wrap_mask = cv2.bitwise_or(wrap_mask, tips)

        quad = AdaptiveMeshBuilder._aabb_quad_from_mask(wrap_mask)
        if quad is None:
            return wrap_mask, None

        buttons = self._detect_side_button_regions(wrap_mask, quad)
        relief = np.zeros((h, w), dtype=np.uint8)
        if tips is not None:
            relief = cv2.bitwise_or(relief, tips)
        if buttons is not None:
            buttons = self._merge_side_button_clusters(buttons, quad)
            thick = cv2.dilate(
                buttons,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
            relief = cv2.bitwise_or(relief, thick)
            # Mild outward coverage so wrap reaches tip faces without stairs.
            depth = float(max(2.5, min(h, w) * 0.007))
            bumped = self._paint_outward_button_bumps(
                body_u8, buttons, quad, depth_px=depth
            )
            # Soften painted bumps — capsule edges, not chunky blocks.
            if np.count_nonzero(bumped):
                bumped_f = cv2.GaussianBlur(
                    bumped.astype(np.float32) / 255.0, (0, 0), sigmaX=2.4
                )
                bumped_f = np.where(bumped_f > 0.38, bumped_f, 0.0)
                wrap_f = np.maximum(
                    wrap_mask.astype(np.float32) / 255.0, bumped_f
                )
                wrap_mask = (np.clip(wrap_f * 255.0, 0, 255)).astype(np.uint8)

        relief = cv2.bitwise_and(relief, wrap_mask)
        # Float AA — hard thresholds made button tips look like pixel blocks.
        wrap_f = cv2.GaussianBlur(
            wrap_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=2.2
        )
        wrap_f = np.maximum(wrap_f, body_u8.astype(np.float32) / 255.0)
        wrap_f = np.where(wrap_f > 0.34, np.maximum(wrap_f, 0.88), wrap_f)
        wrap_mask = (np.clip(wrap_f * 255.0, 0, 255)).astype(np.uint8)
        if np.count_nonzero(relief) < 32:
            relief = None
        return wrap_mask, relief

    def _restore_photo_side_button_tips(
        self, smooth_body: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Recover volume/power tip pixels the smooth cover mask erased.
        """
        if self.phone_image is None:
            return None
        phone = to_bgr(self.phone_image)
        h, w = phone.shape[:2]
        body = (smooth_body > 127).astype(np.uint8) * 255
        if body.shape[:2] != (h, w):
            body = cv2.resize(body, (w, h), interpolation=cv2.INTER_NEAREST)
            body = (body > 127).astype(np.uint8) * 255
        if np.count_nonzero(body) < 64:
            return None

        gray = cv2.cvtColor(phone, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(phone, cv2.COLOR_BGR2LAB).astype(np.float32)
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
        device = ((dist >= 2.5) | (gray < 250)).astype(np.uint8) * 255

        # Search outside the smoothed body and on the inner bezel rim (buttons
        # often sit inside the wrap mask, not in the outer halo).
        pad = max(4, int(round(min(h, w) * 0.014)))
        halo = cv2.dilate(
            body,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
            ),
            iterations=1,
        )
        core = cv2.erode(
            body,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        tips_out = cv2.bitwise_and(
            device, cv2.bitwise_and(halo, cv2.bitwise_not(core))
        )
        tips_out = cv2.bitwise_and(tips_out, cv2.bitwise_not(body))
        edge_px = float(max(3.0, min(h, w) * 0.014))
        dist_in = cv2.distanceTransform(body, cv2.DIST_L2, 5).astype(np.float32)
        rim = (body > 0) & (dist_in <= edge_px * 2.4)
        tips_in = device & rim
        tips = cv2.bitwise_or(
            tips_out.astype(np.uint8) * 255,
            tips_in.astype(np.uint8) * 255,
        )

        # Keep L/R bezel bands only (drop top/bottom contact shadow).
        ys, xs = np.where(body > 0)
        if len(xs) < 32:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        band = max(5, int(round(bw * 0.07)))
        side = np.zeros((h, w), dtype=np.uint8)
        side[y0:y1 + 1, x0 : min(w, x0 + band)] = 255
        side[y0:y1 + 1, max(0, x1 - band + 1) : x1 + 1] = 255
        # Ignore extreme top/bottom (corners, not buttons).
        side[: y0 + int(bh * 0.08), :] = 0
        side[y1 - int(bh * 0.06) :, :] = 0
        tips = cv2.bitwise_and(tips, side)
        tips = cv2.morphologyEx(
            tips,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        tips = cv2.morphologyEx(
            tips,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        if np.count_nonzero(tips) < 24:
            return None
        # Drop huge slabs (failed halo fill).
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            tips, connectivity=8
        )
        out = np.zeros((h, w), dtype=np.uint8)
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            bw_c = int(stats[label, cv2.CC_STAT_WIDTH])
            bh_c = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < 12 or area > int(h * w * 0.02):
                continue
            if bw_c > bw * 0.12 or bh_c > bh * 0.36:
                continue
            out[labels == label] = 255
        if np.count_nonzero(out) < 24:
            return None
        return out

    @staticmethod
    def _paint_outward_button_bumps(
        body: np.ndarray,
        buttons: np.ndarray,
        quad: np.ndarray,
        *,
        depth_px: float,
    ) -> np.ndarray:
        """Paint stadium bumps that stick out past the body edge at each button."""
        from .region_detector import HardwareRegionDetector

        out = np.zeros_like(body)
        h, w = body.shape[:2]
        corners = order_points(np.asarray(quad, dtype=np.float32))
        centroid = corners.mean(axis=0)
        binary = (buttons > 127).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label in range(1, num):
            ys, xs = np.where(labels == label)
            if len(xs) < 8:
                continue
            cx = float(xs.mean())
            cy = float(ys.mean())
            y1 = int(max(0, ys.min() - 2))
            y2 = int(min(h - 1, ys.max() + 2))
            # Side: closer vertical edge.
            left_dist = abs(cx - float(corners[:, 0].min()))
            right_dist = abs(float(corners[:, 0].max()) - cx)
            side = "left" if left_dist <= right_dist else "right"
            if side == "left":
                a, b = corners[0], corners[3]
            else:
                a, b = corners[1], corners[2]
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length < 20:
                continue
            tangent = edge / length
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            mid = (a + b) * 0.5
            if float(np.dot(normal, mid - centroid)) < 0:
                normal = -normal
            # Body edge x at this band.
            edge_xs = []
            for y in range(y1, y2 + 1):
                row = body[y] > 127
                if not row.any():
                    continue
                edge_xs.append(
                    int(np.where(row)[0].min())
                    if side == "left"
                    else int(np.where(row)[0].max())
                )
            if not edge_xs:
                continue
            edge_x = float(np.median(edge_xs))
            # Match blob height tightly — no tall pillow inflation.
            half_h = max(3.5, 0.5 * float(ys.max() - ys.min() + 1) * 1.02)
            # Thin tip: mostly outward hairline, little inward face pad.
            inward = depth_px * 0.45
            outward = depth_px * 0.90
            if side == "left":
                x1 = edge_x - outward
                x2 = edge_x + inward
            else:
                x1 = edge_x - inward
                x2 = edge_x + outward
            yi1 = float(cy - half_h)
            yi2 = float(cy + half_h)
            short = min(x2 - x1, yi2 - yi1)
            # Soft capsule ends — still thin overall.
            corner = float(np.clip(short * 0.45, 1.5, max(2.0, short * 0.48)))
            HardwareRegionDetector._rounded_rectangle(
                out,
                int(np.clip(round(x1), 0, w - 1)),
                int(np.clip(round(yi1), 0, h - 1)),
                int(np.clip(round(x2), 0, w - 1)),
                int(np.clip(round(yi2), 0, h - 1)),
                max(1, int(round(corner))),
            )
        # Soften painted tips so wrap AA follows capsules, not pixel boxes.
        if np.count_nonzero(out) > 0:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=0.9)
            out = (out > 80).astype(np.uint8) * 255
        return out

    def _sync_printable_from_phone_wrap(self) -> None:
        """Printable face = detected phone rim (not the editable cage size)."""
        if self.phone_image is None:
            self.printable_mask = None
            return
        wrap, pm = self._ensure_phone_wrap_geometry()
        if wrap is None or pm is None:
            self.printable_mask = None
            return
        h, w = self.phone_image.shape[:2]
        corner = float(
            np.clip(
                float(self.settings.get("corner_radius", 11.0) or 11.0),
                6.5,
                16.0,
            )
        )
        radii = (corner, corner, corner, corner)
        mask_f = create_mesh_mask(
            wrap,
            (h, w),
            feather_radius=0,
            corner_radius_percent=corner,
            smooth_boundary=True,
            phone_silhouette=pm,
            corner_radii=radii,
            # Geometric rounded arcs — live perimeter inherits photo stairs.
            prefer_live_boundary=False,
        )
        mask_f = np.clip(mask_f, 0.0, 1.0)
        mask_f = np.where(mask_f > 0.50, np.maximum(mask_f, 0.97), mask_f)
        phone_f = (pm > 127).astype(np.float32)
        # Grow the rounded face just enough to erase bald corner tips, then
        # hard-clip to the phone body (perfect fit: full cover, zero studio).
        grow = max(4, int(round(min(h, w) * 0.010)))
        grown = cv2.dilate(
            (mask_f * 255.0).astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)
            ),
            iterations=1,
        ).astype(np.float32) / 255.0
        filled = np.maximum(grown, phone_f)
        solid = (np.minimum(filled, phone_f) * 255.0).astype(np.uint8)
        solid = cv2.bitwise_and(solid, pm)
        self.cover_engine.last_cover_mask = solid.copy()
        self.cover_engine.last_phone_mask = pm.copy()
        printable = solid
        if self.exclusion_mask is not None and np.count_nonzero(
            self.exclusion_mask
        ):
            hard = (self.exclusion_mask > 96).astype(np.uint8) * 255
            printable = cv2.bitwise_and(printable, cv2.bitwise_not(hard))
        self.printable_mask = printable

    def _sync_printable_from_mesh(self) -> None:
        """Compatibility: printable always tracks the phone wrap rim."""
        self._sync_printable_from_phone_wrap()

    def _refresh_wrap_from_geometry(self) -> None:
        """
        Derive wrap rim / margin from the live phone + printable masks.

        Replaces fixed bleed values: rim UV tracks how far the printable
        surface sits inside the detected phone outline.
        """
        phone_mask = getattr(self.cover_engine, "last_phone_mask", None)
        printable = self.printable_mask
        if (
            phone_mask is None
            or printable is None
            or np.count_nonzero(phone_mask) < 64
            or np.count_nonzero(printable) < 64
        ):
            return

        h, w = phone_mask.shape[:2]
        short = float(min(h, w))
        # Distance from phone exterior to printable interior ≈ print margin.
        phone_bin = (phone_mask > 127).astype(np.uint8)
        printable_bin = (printable > 127).astype(np.uint8)
        if phone_bin.shape[:2] != printable_bin.shape[:2]:
            printable_bin = cv2.resize(
                printable_bin, (w, h), interpolation=cv2.INTER_NEAREST
            )
        # Pixels that are on the phone but outside the printable face.
        rim_band = cv2.bitwise_and(phone_bin, cv2.bitwise_not(printable_bin))
        if np.count_nonzero(rim_band) < 16:
            margin_px = 0.0
        else:
            dist = cv2.distanceTransform(phone_bin, cv2.DIST_L2, 5)
            # Mean distance of printable edge samples to the phone exterior.
            edge = cv2.morphologyEx(
                printable_bin,
                cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            samples = dist[edge > 0]
            margin_px = float(np.median(samples)) if samples.size else 0.0

        margin_percent = 100.0 * margin_px / max(short, 1.0)
        self.automatic_margin = float(np.clip(margin_percent, 0.0, 8.0))
        rim = estimate_rim_uv_from_margin(self.automatic_margin)
        bevel = float(self.settings.get("bevel_strength", 92.0)) / 100.0
        self.curved_uv_params = CurvedUVParams(
            rim_uv=rim,
            bevel_strength=bevel,
            corner_radii=self.corner_radii,
            enabled=float(self.settings.get("curved_uv", 1.0)) >= 0.5,
        )
        # Keep settings rim readout in sync (percent), without forcing a
        # user-dragged rim override when it still sits on the default.
        if abs(float(self.settings.get("rim_uv", 5.5)) - 5.5) < 0.05:
            self.settings["rim_uv"] = round(rim * 100.0, 2)

    def set_mesh_points(
        self, points: np.ndarray, rows: Optional[int] = None,
        cols: Optional[int] = None,
    ) -> None:
        """Set mesh vertices in phone-image pixel coordinates."""
        if self.control_mesh is None and (rows is None or cols is None):
            raise ValueError("Mesh dimensions are required for a new mesh")
        mesh_rows = rows or self.control_mesh.rows
        mesh_cols = cols or self.control_mesh.cols
        self.set_control_mesh(ControlMesh(points, mesh_rows, mesh_cols))

    def set_hardware_exclusions(
        self,
        contours: List[np.ndarray],
        *,
        snap_geometry: bool = False,
        allow_clear: bool = False,
        cutout_specs: Optional[List] = None,
        shape_tags: Optional[List[str]] = None,
        corner_frac: float = 0.16,
        persist: bool = True,
        refit_design: bool = True,
    ) -> None:
        """
        Replace camera/flash exclusion cutouts from editable contours.

        Contours are in phone-image pixel coordinates. Rebuilds the exclusion
        mask while preserving the current printable mesh.

        snap_geometry=False (manual Edit Mesh drags): keep the user's polygon
        vertices so shapes stay adjustable. snap_geometry=True (Perfect Finish /
        auto detect): freeze CutoutSpecs and paint analytically / contour-true.

        Phase 3: optional ``cutout_specs`` skips re-freeze and paints from the
        authoritative list directly. ``shape_tags`` preserve editor tools
        (rectangle → mild-round hole, not stadium).

        ``persist`` / ``refit_design`` default True for Perfect Finish; live
        cutout edits should pass False so the UI does not hang on disk I/O
        and Smart Fit after every drag.
        """
        if self.phone_image is None:
            return

        height, width = self.phone_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cleaned: List[np.ndarray] = []
        from .region_detector import HardwareRegionDetector
        from .device_template import (
            CutoutSpec,
            build_cutout_specs,
        )

        gray = None
        if self.phone_image is not None:
            gray = cv2.cvtColor(to_bgr(self.phone_image), cv2.COLOR_BGR2GRAY)

        cover_quad = (
            self.control_mesh.corner_points()
            if self.control_mesh is not None
            else np.array(
                [[0, 0], [width, 0], [width, height], [0, height]],
                dtype=np.float32,
            )
        )

        specs: List[CutoutSpec]
        # Prefer explicit tags; otherwise keep previously locked editor shapes
        # so Perfect Finish / sync cannot silently morph capsule → rectangle.
        effective_tags = shape_tags
        if not effective_tags and self.cutout_shape_tags:
            effective_tags = list(self.cutout_shape_tags)
        if cutout_specs is not None:
            specs = list(cutout_specs)
        elif snap_geometry:
            specs = build_cutout_specs(
                contours,
                cover_quad,
                width,
                height,
                phone_gray=gray,
                authoritative=True,
                shape_tags=effective_tags,
                corner_frac=corner_frac,
            )
        else:
            # Manual drag — provisional specs, paint from live verts + tags.
            specs = build_cutout_specs(
                contours,
                cover_quad,
                width,
                height,
                authoritative=False,
                shape_tags=effective_tags,
                corner_frac=corner_frac,
            )

        if specs:
            for spec in specs:
                HardwareRegionDetector.paint_from_cutout_spec(
                    mask, spec, width, height
                )
                pts = spec.pixel_contour(width, height)
                if pts.shape[0] >= 3:
                    cleaned.append(pts.reshape(-1, 1, 2).astype(np.float32))
            self.cutout_specs = specs
            self.cutout_shape_tags = [
                str(getattr(s, "shape_tag", "") or "") for s in specs
            ]
        else:
            # Fallback: legacy path if specs empty but contours given.
            camera_ids = {
                id(c) for c in self._camera_like_contours(list(contours))
            }
            for contour in contours:
                poly = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
                if len(poly) < 3:
                    continue
                tight = (
                    CAMERA_HOLE_EXPAND_PX if id(contour) in camera_ids else None
                )
                HardwareRegionDetector.paint_cutout_mask(
                    mask, poly, analytical=True, expand_override=tight
                )
                cleaned.append(poly.reshape(-1, 1, 2).astype(np.float32))
            self.cutout_specs = []
            self.cutout_shape_tags = []

        if (
            not cleaned
            and self.hardware_contours
            and not allow_clear
        ):
            # Keep existing cutouts — empty replacement is almost always a bug.
            logger.warning(
                "Ignored empty cutout update that would wipe %d shapes",
                len(self.hardware_contours),
            )
            return

        # Soften the AA fringe only. Never binary-dilate hard cores — that
        # turns SDF circles/stadiums into stair-stepped octagons (flash/camera).
        if np.count_nonzero(mask):
            soft = max(3, int(round(min(height, width) * 0.0012)) | 1)
            blurred = cv2.GaussianBlur(mask.astype(np.float32), (soft, soft), 0)
            mask_f = np.maximum(mask.astype(np.float32), blurred)
            # Distance-soft expand (~1.2px) keeps curves round while clearing
            # bezel ridges — unlike MORPH_ELLIPSE dilate which facets arcs.
            core = (mask_f >= 232).astype(np.uint8)
            if np.count_nonzero(core):
                dist = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
                grow = np.clip(1.0 - dist / 1.35, 0.0, 1.0) * 255.0
                mask_f = np.maximum(mask_f, grow)
            mask = np.clip(mask_f, 0.0, 255.0).astype(np.uint8)

        self.exclusion_mask = mask
        self.hardware_contours = cleaned
        # Live cutout edits: punch holes into the existing solid gate — do not
        # rebuild the supersampled mesh mask (that froze the UI on every drag).
        if refit_design:
            self._sync_printable_from_mesh()
            self._refresh_wrap_from_geometry()
        else:
            self._sync_printable_from_exclusions()
        self.auto_detected = False
        self.from_template = False
        if persist:
            self._persist_manual_template()
        if refit_design and self.design_image is not None:
            self.auto_fit_design()
        else:
            self.invalidate()

    def _sync_printable_from_exclusions(self) -> None:
        """Punch exclusion holes into the cached solid cover gate (fast path)."""
        solid = getattr(self.cover_engine, "last_cover_mask", None)
        if solid is None or self.phone_image is None:
            self._sync_printable_from_mesh()
            return
        h, w = self.phone_image.shape[:2]
        if solid.shape[:2] != (h, w):
            self._sync_printable_from_mesh()
            return
        printable = solid
        if self.exclusion_mask is not None and np.count_nonzero(
            self.exclusion_mask
        ):
            hard = (self.exclusion_mask > 96).astype(np.uint8) * 255
            printable = cv2.bitwise_and(printable, cv2.bitwise_not(hard))
        else:
            printable = solid.copy()
        self.printable_mask = printable

    def scale_hardware_cutouts(self, factor: float) -> int:
        """
        Grow/shrink every cutout about its own centre — curves stay the same.

        factor > 1 enlarges (e.g. 1.08); factor < 1 shrinks. Shape is a uniform
        scale of the existing stadium/circle vertices.
        """
        if self.phone_image is None or not self.hardware_contours:
            return 0
        factor = float(np.clip(factor, 0.45, 1.80))
        if abs(factor - 1.0) < 1e-4:
            return 0
        scaled: List[np.ndarray] = []
        for contour in self.hardware_contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            center = pts.mean(axis=0)
            grown = center + (pts - center) * factor
            scaled.append(grown)
        if not scaled:
            return 0
        # Keep size exactly — do not re-expand via geometry snap.
        self.set_hardware_exclusions(scaled, snap_geometry=False)
        return len(scaled)

    def _solid_cover_gate(self) -> Optional[np.ndarray]:
        """Solid printable cover silhouette (no camera holes) for Final Fill."""
        if self.phone_image is None:
            return None
        h, w = self.phone_image.shape[:2]
        cover = getattr(self.cover_engine, "last_cover_mask", None)
        if cover is not None and cover.shape[:2] == (h, w):
            return cover
        phone = getattr(self.cover_engine, "last_phone_mask", None)
        if phone is not None and phone.shape[:2] == (h, w):
            return phone
        # Fallback: printable ∪ exclusion recovers the solid face.
        if self.printable_mask is not None and self.printable_mask.shape[:2] == (h, w):
            solid = self.printable_mask.copy()
            if self.exclusion_mask is not None:
                solid = np.maximum(solid, self.exclusion_mask)
            return solid
        return None

    @staticmethod
    def _stamp_soft_disk(
        mask: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        *,
        add: bool,
        gate: Optional[np.ndarray] = None,
        strength: float = 1.0,
    ) -> bool:
        """
        Soft circular stamp for Final Erase / Fill.

        ``add=True`` raises exclusion (erase wrap). ``add=False`` clears it
        (fill wrap back). Fill uses a hard core so wrap actually returns.
        """
        h, w = mask.shape[:2]
        r = max(1.5, float(radius))
        pad = int(np.ceil(r)) + 2
        x0 = max(0, int(np.floor(cx - pad)))
        y0 = max(0, int(np.floor(cy - pad)))
        x1 = min(w, int(np.ceil(cx + pad)) + 1)
        y1 = min(h, int(np.ceil(cy + pad)) + 1)
        if x1 <= x0 or y1 <= y0:
            return False
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt(
            (xx.astype(np.float32) - float(cx)) ** 2
            + (yy.astype(np.float32) - float(cy)) ** 2
        )
        # Exact brush radius — no inflated soft halo that over-erases.
        t = np.clip(1.0 - dist / r, 0.0, 1.0)
        soft = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        soft *= float(np.clip(strength, 0.05, 1.0))
        if gate is not None and gate.shape[:2] == (h, w):
            soft *= (gate[y0:y1, x0:x1].astype(np.float32) / 255.0)
        if float(np.max(soft)) < 1e-4:
            return False
        roi = mask[y0:y1, x0:x1].astype(np.float32)
        if add:
            mask[y0:y1, x0:x1] = np.clip(
                np.maximum(roi, soft * 255.0), 0.0, 255.0
            ).astype(np.uint8)
        else:
            # Hard wipe in the brush core so Fill actually restores wrap;
            # soft feather only on the rim.
            core = soft >= 0.35
            out = roi.copy()
            out[core] = 0.0
            feather = soft.copy()
            feather[core] = 0.0
            out *= 1.0 - feather
            mask[y0:y1, x0:x1] = np.clip(out, 0.0, 255.0).astype(np.uint8)
        return True

    def _commit_brush_exclusion_mask(
        self, mask: np.ndarray, *, expand: bool = True
    ) -> int:
        """
        Commit a brushed exclusion mask without snap / per-dab cutout spam.

        ``expand=True`` (Erase): light AA so holes look smooth.
        ``expand=False`` (Fill): keep cleared pixels cleared — blur must not
        grow exclusion back into corners / rim that the user just restored.
        """
        from .region_detector import HardwareRegionDetector

        if int(np.count_nonzero(mask > 96)) == 0:
            h, w = mask.shape[:2]
            self.exclusion_mask = np.zeros((h, w), dtype=np.uint8)
            self.hardware_contours = []
            self.cutout_specs = []
            self.cutout_shape_tags = []
            self._sync_printable_from_mesh()
            self.invalidate()
            return 0

        if expand:
            soft = max(3, int(round(min(mask.shape[:2]) * 0.0008)) | 1)
            blurred = cv2.GaussianBlur(mask.astype(np.float32), (soft, soft), 0)
            committed = np.clip(
                np.maximum(mask.astype(np.float32), blurred * 0.55), 0.0, 255.0
            ).astype(np.uint8)
        else:
            committed = mask.copy()
            # Kill soft dust so Fill doesn't leave ghost punches at corners.
            dust = (committed > 0) & (committed < 64)
            if np.any(dust):
                committed[dust] = 0

        contours = HardwareRegionDetector._smooth_exclusion_contours(committed)
        cleaned: List[np.ndarray] = []
        for contour in contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] >= 3:
                cleaned.append(pts.reshape(-1, 1, 2))

        self.exclusion_mask = committed
        self.hardware_contours = cleaned
        self.cutout_specs = []
        self.cutout_shape_tags = []
        # Mesh solid + remaining holes — inset cover gate alone was clipping
        # Fill at rounded corners.
        self._sync_printable_from_mesh()
        self.invalidate()
        return len(cleaned)

    def paint_exclusion_dabs(
        self, dabs: List[Tuple[float, float, float]]
    ) -> int:
        """
        Erase wrap by painting soft circular exclusion dabs (phone-pixel coords).

        Each dab is (x, y, radius). Rebuilds one clean contour set from the mask
        (never one cutout circle per dab). Mesh vertices are never moved.
        """
        if self.phone_image is None or not dabs:
            return 0

        h, w = self.phone_image.shape[:2]
        mask = (
            np.zeros((h, w), dtype=np.uint8)
            if self.exclusion_mask is None
            else self.exclusion_mask.copy()
        )
        painted = 0
        for x, y, radius in dabs:
            r = max(1.5, float(radius) * 0.85)
            if self._stamp_soft_disk(mask, float(x), float(y), r, add=True):
                painted += 1
        if painted == 0:
            return 0
        self._commit_brush_exclusion_mask(mask, expand=True)
        return painted

    def clear_exclusion_dabs(
        self, dabs: List[Tuple[float, float, float]]
    ) -> int:
        """
        Fill wrap back under the brush.

        Clears exclusion holes AND restores printable coverage on the phone
        (corner gaps often have no exclusion — they need coverage restore).
        """
        if self.phone_image is None or not dabs:
            return 0

        h, w = self.phone_image.shape[:2]
        cleared = 0
        has_excl = (
            self.exclusion_mask is not None
            and int(np.count_nonzero(self.exclusion_mask)) > 0
        )

        if has_excl:
            mask = self.exclusion_mask.copy()
            before_total = int(np.count_nonzero(mask))

            for x, y, radius in dabs:
                cx = float(x)
                cy = float(y)
                r = max(2.5, float(radius) * 1.35)
                if self._stamp_soft_disk(
                    mask, cx, cy, r, add=False, gate=None, strength=1.0
                ):
                    cleared += 1
                cv2.circle(
                    mask,
                    (int(round(cx)), int(round(cy))),
                    max(2, int(round(r * 0.92))),
                    0,
                    -1,
                    cv2.LINE_AA,
                )

            if cleared > 0 or int(np.count_nonzero(mask)) < before_total:
                stroke_union = np.zeros((h, w), dtype=np.uint8)
                for x, y, radius in dabs:
                    r = max(3, int(round(float(radius) * 1.45)))
                    cv2.circle(
                        stroke_union,
                        (int(round(float(x))), int(round(float(y)))),
                        r,
                        255,
                        -1,
                        cv2.LINE_AA,
                    )
                fringe = (stroke_union > 0) & (mask > 0) & (mask < 120)
                if np.any(fringe):
                    mask[fringe] = 0
                self._commit_brush_exclusion_mask(mask, expand=False)
                cleared = max(cleared, 1)

        # Corner / rim gaps: exclusion may be empty — paint printable back on
        # the phone body under the stroke so Fill actually works there.
        restored = self._restore_printable_under_dabs(dabs)
        return max(cleared, restored)

    def _restore_printable_under_dabs(
        self, dabs: List[Tuple[float, float, float]]
    ) -> int:
        """
        Grow printable / cover gate under Fill strokes (clipped to phone).

        Used when corner under-coverage is geometric, not an exclusion hole.
        """
        if self.phone_image is None or not dabs:
            return 0
        h, w = self.phone_image.shape[:2]
        phone = getattr(self.cover_engine, "last_phone_mask", None)
        if phone is None or np.count_nonzero(phone) < 64:
            return 0
        if phone.shape[:2] != (h, w):
            phone = cv2.resize(phone, (w, h), interpolation=cv2.INTER_NEAREST)
        phone_bin = (phone > 127).astype(np.uint8) * 255
        # Slight dilate so corner AA tips of the product are fillable.
        phone_bin = cv2.dilate(
            phone_bin,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        stroke = np.zeros((h, w), dtype=np.uint8)
        for x, y, radius in dabs:
            r = max(3, int(round(float(radius) * 1.5)))
            cv2.circle(
                stroke,
                (int(round(float(x))), int(round(float(y)))),
                r,
                255,
                -1,
                cv2.LINE_AA,
            )
        add = cv2.bitwise_and(stroke, phone_bin)
        if int(np.count_nonzero(add)) < 8:
            return 0

        solid = getattr(self.cover_engine, "last_cover_mask", None)
        if solid is None or solid.shape[:2] != (h, w):
            solid = (
                np.zeros((h, w), dtype=np.uint8)
                if self.printable_mask is None
                else self.printable_mask.copy()
            )
            if self.exclusion_mask is not None:
                solid = np.maximum(solid, self.exclusion_mask)
        before = int(np.count_nonzero(solid > 96))
        solid = cv2.max(solid, add)
        # Soft AA fringe on the stroke so Fill isn't a hard sticker.
        soft = cv2.GaussianBlur(add.astype(np.float32), (5, 5), 0)
        solid = np.clip(
            np.maximum(solid.astype(np.float32), soft * 0.85), 0, 255
        ).astype(np.uint8)
        solid = cv2.bitwise_and(solid, phone_bin)
        after = int(np.count_nonzero(solid > 96))
        if after <= before:
            return 0

        self.cover_engine.last_cover_mask = solid.copy()
        printable = solid
        if self.exclusion_mask is not None and np.count_nonzero(
            self.exclusion_mask
        ):
            hard = (self.exclusion_mask > 96).astype(np.uint8) * 255
            printable = cv2.bitwise_and(printable, cv2.bitwise_not(hard))
        self.printable_mask = printable

        # Nudge nearby mesh boundary verts outward toward fill so next sync
        # keeps the corner coverage.
        self._nudge_mesh_toward_fill_stroke(add)
        self.invalidate()
        return 1

    def _nudge_mesh_toward_fill_stroke(self, stroke_mask: np.ndarray) -> None:
        """Push nearby boundary verts slightly outward into restored coverage."""
        if self.control_mesh is None or stroke_mask is None:
            return
        if int(np.count_nonzero(stroke_mask)) < 8:
            return
        pts = self.control_mesh.points
        rows, cols = self.control_mesh.rows, self.control_mesh.cols
        h, w = stroke_mask.shape[:2]
        ys, xs = np.where(stroke_mask > 127)
        if len(xs) < 4:
            return
        stroke_c = np.array(
            [float(xs.mean()), float(ys.mean())], dtype=np.float32
        )
        # Boundary indices: outer ring.
        boundary = []
        for c in range(cols):
            boundary.append(c)
            boundary.append((rows - 1) * cols + c)
        for r in range(1, rows - 1):
            boundary.append(r * cols)
            boundary.append(r * cols + (cols - 1))
        mesh_c = pts.mean(axis=0)
        moved = False
        for idx in boundary:
            p = pts[idx].astype(np.float32)
            if float(np.linalg.norm(p - stroke_c)) > max(h, w) * 0.12:
                continue
            # Outward from mesh centre toward stroke.
            direction = stroke_c - mesh_c
            nrm = float(np.linalg.norm(direction))
            if nrm < 1e-3:
                continue
            direction /= nrm
            # Only nudge if stroke is outward from this vert relative to centre.
            if float(np.dot(p - mesh_c, direction)) < 0:
                continue
            step = min(4.0, max(1.5, nrm * 0.01))
            np_ = p + direction * step
            ix = int(np.clip(round(np_[0]), 0, w - 1))
            iy = int(np.clip(round(np_[1]), 0, h - 1))
            phone = getattr(self.cover_engine, "last_phone_mask", None)
            if phone is not None and phone.shape[:2] == (h, w):
                if phone[iy, ix] < 64:
                    continue
            pts[idx] = np_
            moved = True
        if moved:
            from .mesh import AdaptiveMeshBuilder

            AdaptiveMeshBuilder._reinterpolate_interior(self.control_mesh)
            self.cover_points = self.control_mesh.corner_points()

    def exclude_side_buttons(self) -> int:
        """
        Detect volume / power / mute / side-fingerprint openings and exclude.

        Does not move the control mesh. Unions new openings into existing
        camera cutouts so manual work is kept. Detection is photo-dynamic
        (ridge + silhouette) — no fixed-height seed zones.
        """
        if self.phone_image is None or self.control_mesh is None:
            return 0
        from .region_detector import HardwareRegionDetector

        mesh_before = self.control_mesh.points.copy()
        phone = to_bgr(self.phone_image)
        h, w = phone.shape[:2]
        quad = self.control_mesh.corner_points()
        phone_mask = getattr(self.cover_engine, "last_phone_mask", None)
        if phone_mask is None:
            phone_mask = self._phone_wrap_mask

        peripheral = HardwareRegionDetector.detect_verified_side_hardware(
            phone, quad, phone_mask=phone_mask
        )
        if np.count_nonzero(peripheral) == 0:
            return 0

        base = (
            np.zeros((h, w), dtype=np.uint8)
            if self.exclusion_mask is None
            else (self.exclusion_mask > 96).astype(np.uint8) * 255
        )
        before = int(np.count_nonzero(base))
        merged = cv2.max(base, peripheral)
        after = int(np.count_nonzero(merged))
        # Still polish when new pixels are few but contours appear (tiny FP).
        if after <= before and int(np.count_nonzero(peripheral)) < 40:
            return 0

        peripheral_contours = HardwareRegionDetector._smooth_exclusion_contours(
            peripheral
        )
        finished_new = HardwareRegionDetector.perfect_finish_contours(
            peripheral_contours, phone
        )
        if not finished_new:
            finished_new = peripheral_contours

        polished_buttons: List[np.ndarray] = []
        button_tags: List[str] = []
        for contour in finished_new:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            kind, params = HardwareRegionDetector._classify_cutout(pts)
            if kind == "circle" and params:
                circ = HardwareRegionDetector._sample_circle(
                    float(params[0]), float(params[1]), float(params[2]),
                    samples=64,
                )
                polished_buttons.append(
                    np.asarray(circ, dtype=np.float32).reshape(-1, 2)
                )
                button_tags.append("capsule")
                continue
            if kind in ("stadium", "rounded_rect") and len(params) >= 5:
                x1, y1, x2, y2, corner = (
                    float(params[0]), float(params[1]),
                    float(params[2]), float(params[3]), float(params[4]),
                )
                short = min(x2 - x1, y2 - y1)
                # Tight punch — fat grow erased side-wall wrap.
                aspect = max(x2 - x1, y2 - y1) / max(short, 1e-3)
                grow = 1.06 if aspect < 2.2 else 1.02
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                hw = 0.5 * (x2 - x1) * grow
                hh = 0.5 * (y2 - y1) * grow
                x1, x2 = cx - hw, cx + hw
                y1, y2 = cy - hh, cy + hh
                corner = float(np.clip(short * 0.48, 2.0, short * 0.5 - 0.5))
                stadium = HardwareRegionDetector._sample_rounded_rect(
                    x1, y1, x2, y2, corner, samples_per_corner=16
                )
                if stadium is not None:
                    polished_buttons.append(stadium.reshape(-1, 2))
                    button_tags.append("capsule")
                    continue
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())
            short = min(x2 - x1, y2 - y1)
            corner = float(
                np.clip(short * 0.48, 2.0, max(2.0, short * 0.5 - 0.5))
            )
            stadium = HardwareRegionDetector._sample_rounded_rect(
                x1, y1, x2, y2, corner, samples_per_corner=16
            )
            if stadium is not None:
                polished_buttons.append(stadium.reshape(-1, 2))
            else:
                polished_buttons.append(pts)
            button_tags.append("capsule")
        finished_new = polished_buttons

        existing = [
            np.asarray(c, dtype=np.float32).reshape(-1, 2)
            for c in (self.hardware_contours or [])
            if len(np.asarray(c).reshape(-1, 2)) >= 3
        ]
        existing_tags = list(self.cutout_shape_tags or [])
        kept_existing: List[np.ndarray] = []
        kept_tags: List[str] = []
        new_centers = [
            np.asarray(c, dtype=np.float32).reshape(-1, 2).mean(axis=0)
            for c in finished_new
        ]
        for idx, contour in enumerate(existing):
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            center = pts.mean(axis=0)
            tag = existing_tags[idx] if idx < len(existing_tags) else ""
            tag_l = (tag or "").lower()
            # Never drop camera / face cutouts when adding side pills.
            is_camera = tag_l in (
                "rounded_rect", "rectangle", "circle", "camera", ""
            ) and not self._contour_is_side_bezel(pts, quad)
            if is_camera:
                kept_existing.append(pts)
                kept_tags.append(tag or self._infer_heal_shape_tag(pts))
                continue
            span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            hit = any(
                float(np.linalg.norm(center - nc)) < max(14.0, span * 0.35)
                for nc in new_centers
            )
            if hit:
                continue
            kept_existing.append(pts)
            kept_tags.append(tag)

        # Merge only among new side pills — never collapse into the camera.
        merged_new = HardwareRegionDetector.merge_overlapping_contours(
            finished_new, overlap_ratio=0.55, center_frac=0.03
        )
        combined = kept_existing + [
            np.asarray(c, dtype=np.float32).reshape(-1, 2)
            for c in merged_new
        ]
        combined_tags = kept_tags + (["capsule"] * len(merged_new))
        if len(combined_tags) != len(combined):
            combined_tags = []
            for c in combined:
                pts = np.asarray(c, dtype=np.float32).reshape(-1, 2)
                if self._contour_is_side_bezel(pts, quad):
                    combined_tags.append("capsule")
                else:
                    combined_tags.append(self._infer_heal_shape_tag(pts))

        self.set_hardware_exclusions(
            [np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in combined],
            snap_geometry=True,
            shape_tags=combined_tags,
        )

        if not np.allclose(mesh_before, self.control_mesh.points, atol=0.01):
            self.control_mesh.points[:] = mesh_before
            self.cover_points = self.control_mesh.corner_points()
        return max(1, max(0, after - before), len(finished_new))

    def _contour_is_side_bezel(
        self, contour: np.ndarray, quad: np.ndarray
    ) -> bool:
        """True when a cutout sits on the L/R bezel (button / side FP)."""
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return False
        corners = order_points(np.asarray(quad, dtype=np.float32))
        x_min = float(corners[:, 0].min())
        x_max = float(corners[:, 0].max())
        y_min = float(corners[:, 1].min())
        y_max = float(corners[:, 1].max())
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        bw = float(pts[:, 0].max() - pts[:, 0].min())
        bh = float(pts[:, 1].max() - pts[:, 1].min())
        side_band = width * 0.14
        near_side = (cx - x_min) <= side_band or (x_max - cx) <= side_band
        if not near_side:
            return False
        t = (cy - y_min) / height
        if t < 0.10 or t > 0.88:
            return False
        return bw < width * 0.18 and bh < height * 0.28

    def _seed_side_button_mask(
        self, phone_bgr: np.ndarray, quad: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Deprecated helper — kept for compatibility. Prefer
        ``HardwareRegionDetector.detect_verified_side_hardware`` (no fixed
        height seeds). Always returns None.
        """
        return None

    def _auto_cut_side_hardware(self) -> int:
        """
        After phone load / wrap settle: punch verified side buttons + FP.

        Camera cutouts from analyze are kept. Mesh geometry is not moved.
        """
        if self.phone_image is None or self.control_mesh is None:
            return 0
        try:
            painted = int(self.exclude_side_buttons() or 0)
            if painted:
                self._merge_nearby_side_cutouts()
                self._retag_camera_cutouts()
                self._sync_printable_from_phone_wrap()
            return painted
        except Exception:
            logger.exception("Auto side-hardware cut failed")
            return 0

    def heal_realistic_wrap(self, *, include_hardware: bool = True) -> int:
        """
        Full-bleed realistic wrap for any phone — Perfect Finish → Everything.

        1. Solid sealed phone silhouette (no camera bites / diagonal bald)
        2. Upright rounded mesh snapped to the product rim (no tear)
        3. Camera island tightened to a rounded-rect (never a huge circle)
        4. Side buttons detected and punched as clean capsules
        5. Artwork re-fit to the healed printable face
        """
        if self.phone_image is None or self.control_mesh is None:
            return 0

        self._retain_plausible_cutouts()
        count = int(self.perfect_finish_cutouts(scope="edges") or 0)
        self._finalize_fullbleed_mesh()

        if include_hardware:
            if (
                self.hardware_contours
                or (
                    self.exclusion_mask is not None
                    and int(np.count_nonzero(self.exclusion_mask)) > 0
                )
            ):
                count = max(
                    count,
                    int(self.perfect_finish_cutouts(scope="camera") or 0),
                )
                self._tighten_camera_cutouts()
                self._retag_camera_cutouts()
                self._retain_plausible_cutouts()

            # Volume stays wrapped with relief. Intentional side punches live
            # under Perfect Finish → Buttons / Erase — not Everything.
            self._merge_nearby_side_cutouts()
            self._retag_camera_cutouts()

            if self.hardware_contours:
                tags = list(self.cutout_shape_tags or [])
                if len(tags) != len(self.hardware_contours):
                    tags = [
                        self._infer_heal_shape_tag(c)
                        for c in self.hardware_contours
                    ]
                self.set_hardware_exclusions(
                    [
                        np.asarray(c, dtype=np.float32).reshape(-1, 2)
                        for c in self.hardware_contours
                    ],
                    snap_geometry=True,
                    allow_clear=False,
                    shape_tags=tags,
                    persist=True,
                    refit_design=True,
                )

            # Keep full-bleed after cutout polish (never warp mesh to holes).
            # Do not snap mid-sides to jagged photo contour — that skewed the
            # phone area and made TR corners look torn.
            self._finalize_fullbleed_mesh()

        if self.design_image is not None:
            self.auto_fit_design(preserve_placement=True)
        else:
            self.invalidate()
        self.auto_detected = True
        return max(count, 1)

    def _finalize_fullbleed_mesh(self) -> None:
        """Align edit cage + wrap cache to the detected phone rim."""
        if self.phone_image is None or self.control_mesh is None:
            return
        wrap, pm = self._ensure_phone_wrap_geometry(force=True)
        if wrap is None or pm is None:
            return
        # Perfect Finish updates the visible edit cage to match wrap.
        self.control_mesh = wrap.copy()
        self.cover_points = self.control_mesh.corner_points()
        self.auto_detected = True
        self._sync_printable_from_phone_wrap()
        self._refresh_wrap_from_geometry()

    def _retag_camera_cutouts(self) -> None:
        """Large upper islands are rounded rects — never paint as huge circles."""
        if not self.hardware_contours or self.control_mesh is None:
            return
        tags = list(self.cutout_shape_tags or [])
        while len(tags) < len(self.hardware_contours):
            tags.append("")
        corners = self.control_mesh.corner_points()
        x_min = float(corners[:, 0].min())
        x_max = float(corners[:, 0].max())
        y_min = float(corners[:, 1].min())
        y_max = float(corners[:, 1].max())
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)
        upper_limit = y_min + height * 0.38
        side_band = width * 0.12
        cam = self._camera_like_contours(list(self.hardware_contours))
        if not cam:
            cam = self._upper_cutouts(list(self.hardware_contours))
        cam_ids = {id(c) for c in cam}
        for i, contour in enumerate(self.hardware_contours):
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            bw = float(pts[:, 0].max() - pts[:, 0].min())
            bh = float(pts[:, 1].max() - pts[:, 1].min())
            on_side = (cx - x_min) <= side_band or (x_max - cx) <= side_band
            # Bezel pills / fingerprint — always capsule, never camera rect.
            if on_side and max(bw, bh) < width * 0.28:
                tags[i] = "capsule"
                continue
            if cy > upper_limit or id(contour) not in cam_ids:
                continue
            if max(bw, bh) >= 36.0:
                tags[i] = "rounded_rect"
            elif (tags[i] or "").lower() in ("", "circle"):
                tags[i] = self._infer_heal_shape_tag(pts)
        self.cutout_shape_tags = tags

    def _infer_heal_shape_tag(self, contour: np.ndarray) -> str:
        """Pick a clean editor tool for a healed cutout (any phone)."""
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return "circle"
        bw = float(pts[:, 0].max() - pts[:, 0].min())
        bh = float(pts[:, 1].max() - pts[:, 1].min())
        short = max(min(bw, bh), 1e-3)
        aspect = max(bw, bh) / short
        # Camera modules are squarish but must NOT become filled circles.
        if max(bw, bh) >= 28.0 and aspect < 1.55:
            return "rounded_rect"
        if aspect < 1.25 and max(bw, bh) < 28.0:
            return "circle"
        if aspect >= 1.85:
            return "capsule"
        return "rectangle"

    def _retain_plausible_cutouts(self) -> None:
        """
        Keep camera-zone + side-bezel openings; drop session junk.

        Autosaves sometimes store a bottom-corner blob that punches a white
        hole in the wrap — that is not hardware on any phone.
        """
        source = list(self.hardware_contours or [])
        if not source or self.control_mesh is None:
            return
        cam = self._camera_like_contours(source)
        if not cam:
            cam = self._upper_cutouts(source)
        cam_ids = {id(c) for c in cam}
        corners = self.control_mesh.corner_points()
        x_min = float(corners[:, 0].min())
        x_max = float(corners[:, 0].max())
        y_min = float(corners[:, 1].min())
        y_max = float(corners[:, 1].max())
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)
        side_band = width * 0.13
        kept: List[np.ndarray] = []
        tags_in = list(self.cutout_shape_tags or [])
        tags_out: List[str] = []
        for idx, contour in enumerate(source):
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            tag = tags_in[idx] if idx < len(tags_in) else ""
            if id(contour) in cam_ids:
                kept.append(pts)
                tags_out.append(tag or self._infer_heal_shape_tag(pts))
                continue
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            bw = float(pts[:, 0].max() - pts[:, 0].min())
            bh = float(pts[:, 1].max() - pts[:, 1].min())
            near_side = (cx - x_min) <= side_band or (x_max - cx) <= side_band
            tag_l = (tag or "").lower()
            force_side = tag_l in ("capsule", "button", "pill_h", "pill_v")
            # Side buttons / FP: on the bezel, not a huge face blob.
            if (
                (near_side or force_side)
                and bw < width * 0.16
                and bh < height * 0.32
                and cy > y_min + height * 0.08
                and cy < y_max - height * 0.06
            ):
                kept.append(pts)
                tags_out.append(tag or "capsule")
        if not kept:
            return
        if len(kept) == len(source):
            # Still refresh tags for clean paint later.
            if tags_out and (
                len(self.cutout_shape_tags) != len(kept)
                or any(not t for t in self.cutout_shape_tags)
            ):
                self.cutout_shape_tags = tags_out
            return
        self.set_hardware_exclusions(
            [np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in kept],
            snap_geometry=True,
            allow_clear=True,
            shape_tags=tags_out,
            persist=False,
            refit_design=False,
        )

    def _merge_nearby_side_cutouts(self) -> None:
        """Merge only overlapping same-bezel pills — keep volume ≠ fingerprint."""
        source = list(self.hardware_contours or [])
        if len(source) < 2 or self.control_mesh is None:
            return
        corners = self.control_mesh.corner_points()
        x_min = float(corners[:, 0].min())
        x_max = float(corners[:, 0].max())
        y_min = float(corners[:, 1].min())
        y_max = float(corners[:, 1].max())
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)
        side_band = width * 0.13
        side_items: List[tuple] = []
        other: List[np.ndarray] = []
        other_tags: List[str] = []
        tags = list(self.cutout_shape_tags or [])
        for idx, contour in enumerate(source):
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            cx = float(pts[:, 0].mean())
            near_side = (cx - x_min) <= side_band or (x_max - cx) <= side_band
            bw = float(pts[:, 0].max() - pts[:, 0].min())
            bh = float(pts[:, 1].max() - pts[:, 1].min())
            tag = tags[idx] if idx < len(tags) else ""
            if near_side and max(bw, bh) < width * 0.18:
                side_items.append((pts, tag))
            else:
                other.append(pts)
                other_tags.append(tag)
        if len(side_items) < 2:
            return

        # Cluster by Y-overlap on the same bezel — never glue distant volume+FP.
        used = [False] * len(side_items)
        merged_side: List[np.ndarray] = []
        merged_tags: List[str] = []
        for i, (pts_i, tag_i) in enumerate(side_items):
            if used[i]:
                continue
            cluster = [pts_i]
            used[i] = True
            y1 = float(pts_i[:, 1].min())
            y2 = float(pts_i[:, 1].max())
            x1 = float(pts_i[:, 0].min())
            x2 = float(pts_i[:, 0].max())
            cx_i = 0.5 * (x1 + x2)
            changed = True
            while changed:
                changed = False
                for j, (pts_j, _tag_j) in enumerate(side_items):
                    if used[j]:
                        continue
                    jy1 = float(pts_j[:, 1].min())
                    jy2 = float(pts_j[:, 1].max())
                    jx1 = float(pts_j[:, 0].min())
                    jx2 = float(pts_j[:, 0].max())
                    cx_j = 0.5 * (jx1 + jx2)
                    if abs(cx_i - cx_j) > width * 0.08:
                        continue
                    # Require real Y overlap (or tiny gap).
                    gap = max(0.0, max(y1, jy1) - min(y2, jy2))
                    if gap > height * 0.025:
                        continue
                    overlap = min(y2, jy2) - max(y1, jy1)
                    if overlap < -height * 0.01 and gap > 4.0:
                        continue
                    cluster.append(pts_j)
                    used[j] = True
                    y1 = min(y1, jy1)
                    y2 = max(y2, jy2)
                    x1 = min(x1, jx1)
                    x2 = max(x2, jx2)
                    cx_i = 0.5 * (x1 + x2)
                    changed = True
            if len(cluster) == 1:
                merged_side.append(cluster[0])
                merged_tags.append(tag_i or "capsule")
            else:
                box = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
                merged_side.append(box)
                merged_tags.append("capsule")

        if len(merged_side) == len(side_items):
            return
        kept = other + merged_side
        out_tags = other_tags + merged_tags
        self.set_hardware_exclusions(
            [np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in kept],
            snap_geometry=True,
            allow_clear=True,
            shape_tags=out_tags,
            persist=False,
            refit_design=False,
        )

    def _find_camera_module_quad(self) -> Optional[np.ndarray]:
        """
        Locate the camera island as a rounded-rect quad on the phone face.

        Uses a top-left ROI contrast search so results stay stable regardless
        of mesh cage size (wide cages ballooned the hole; tight cages shrank
        it to a single lens).
        """
        if self.phone_image is None:
            return None
        phone = to_bgr(self.phone_image)
        h, w = phone.shape[:2]
        pm = getattr(self.cover_engine, "last_phone_mask", None)
        if pm is None or np.count_nonzero(pm) < 64:
            return None
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(pm > 127)
        if len(xs) < 32:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pw, ph = max(x1 - x0, 1), max(y1 - y0, 1)
        rx1 = min(w, x0 + int(pw * 0.58))
        ry1 = min(h, y0 + int(ph * 0.36))
        gray = cv2.cvtColor(phone, cv2.COLOR_BGR2GRAY)
        roi = gray[y0:ry1, x0:rx1]
        roi_m = (pm[y0:ry1, x0:rx1] > 127).astype(np.uint8)
        if np.count_nonzero(roi_m) < 64:
            return None
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        body_vals = blur[roi_m > 0]
        if body_vals.size < 64:
            return None
        med = float(np.median(body_vals))
        # Module glass / lenses are darker than the surrounding back glass.
        dark = (blur < med - 5.0).astype(np.uint8) * 255
        dark = cv2.bitwise_and(dark, roi_m * 255)
        k = max(9, int(min(pw, ph) * 0.04) | 1)
        seed = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=2,
        )
        seed = cv2.bitwise_and(seed, roi_m * 255)
        contours, _ = cv2.findContours(
            seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        roi_a = float(max(np.count_nonzero(roi_m), 1))
        best = None
        best_score = -1.0
        min_side = max(22, int(min(pw, ph) * 0.10))
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < roi_a * 0.05 or area > roi_a * 0.70:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 2.1:
                continue
            if min(bw, bh) < min_side:
                continue
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            left_bias = 1.0 + 0.45 * (1.0 - cx / max(rx1 - x0, 1))
            top_bias = 1.0 + 0.35 * (1.0 - cy / max(ry1 - y0, 1))
            fill = area / max(float(bw * bh), 1.0)
            score = area * left_bias * top_bias * (0.45 + 0.55 * fill)
            if score > best_score:
                best_score = score
                # Shrink the dark halo toward the real island (~10%).
                shrink = 0.10
                x1 = x + bw * shrink * 0.5
                y1 = y + bh * shrink * 0.5
                x2 = x + bw * (1.0 - shrink * 0.5)
                y2 = y + bh * (1.0 - shrink * 0.5)
                best = np.array(
                    [
                        [x0 + x1, y0 + y1],
                        [x0 + x2, y0 + y1],
                        [x0 + x2, y0 + y2],
                        [x0 + x1, y0 + y2],
                    ],
                    dtype=np.float32,
                )
        return best

    def _tighten_camera_cutouts(self) -> None:
        """
        Snap oversized camera AABBs to the photo module, then keep capsule/rect.

        Prefers a fresh photo detect when the live cutout is clearly larger than
        the real island (stops a huge rounded-rect from balding the top face).
        """
        source = list(self.hardware_contours or [])
        if not source or self.phone_image is None:
            return
        from .region_detector import HardwareRegionDetector

        cam = self._camera_like_contours(source)
        if not cam:
            cam = self._upper_cutouts(source)
        if not cam:
            return
        others = self._contours_not_in(source, cam)

        # Photo module via top-left ROI contrast (stable across mesh sizes).
        photo_cam: List[np.ndarray] = []
        module = self._find_camera_module_quad()
        if module is not None:
            photo_cam = [module]
        else:
            try:
                phone = to_bgr(self.phone_image)
                pm = getattr(self.cover_engine, "last_phone_mask", None)
                if pm is not None and np.count_nonzero(pm) > 64:
                    from .mesh import AdaptiveMeshBuilder

                    quad = AdaptiveMeshBuilder._stable_quad_from_mask(pm)
                else:
                    quad = (
                        self.control_mesh.corner_points()
                        if self.control_mesh is not None
                        else None
                    )
                _mask, detected, _conf = HardwareRegionDetector.detect(
                    phone, quad
                )
                photo_cam = self._camera_like_contours(list(detected or []))
                if not photo_cam:
                    photo_cam = self._upper_cutouts(list(detected or []))
            except Exception:
                photo_cam = []

        rebuilt = HardwareRegionDetector.rebuild_camera_cutouts(
            cam, self.phone_image
        )
        polished = [
            np.asarray(c, dtype=np.float32).reshape(-1, 2)
            for c in (rebuilt or cam)
        ]

        if photo_cam:
            # Keep module-sized photo islands (drop tiny lens disks).
            modules: List[Tuple[float, np.ndarray]] = []
            for p in photo_cam:
                pts = np.asarray(p, dtype=np.float32).reshape(-1, 2)
                bw = float(pts[:, 0].ptp())
                bh = float(pts[:, 1].ptp())
                if min(bw, bh) < 36.0 and max(bw, bh) < 55.0:
                    continue
                modules.append((bw * bh, pts))
            modules.sort(key=lambda t: t[0], reverse=True)

            if modules:
                tightened = []
                for a in polished:
                    ax1, ay1 = float(a[:, 0].min()), float(a[:, 1].min())
                    ax2, ay2 = float(a[:, 0].max()), float(a[:, 1].max())
                    a_area = max((ax2 - ax1) * (ay2 - ay1), 1.0)
                    # Default to largest photo module when live box is oversized.
                    p_area, p_pts = modules[0]
                    if a_area > p_area * 1.08:
                        # Build a clean rounded-rect from the photo island AABB.
                        px1 = float(p_pts[:, 0].min())
                        py1 = float(p_pts[:, 1].min())
                        px2 = float(p_pts[:, 0].max())
                        py2 = float(p_pts[:, 1].max())
                        pad = 1.5
                        tight = np.array(
                            [
                                [px1 - pad, py1 - pad],
                                [px2 + pad, py1 - pad],
                                [px2 + pad, py2 + pad],
                                [px1 - pad, py2 + pad],
                            ],
                            dtype=np.float32,
                        )
                        tightened.append(tight)
                    else:
                        tightened.append(a)
                polished = tightened
            if not polished and modules:
                polished = [modules[0][1]]

        # Prefer one module contour when rebuild returns a tight island.
        if len(polished) > 1:
            # Keep separate flash circles; merge only overlapping lens stacks.
            merged: List[np.ndarray] = []
            used = [False] * len(polished)
            for i, a in enumerate(polished):
                if used[i]:
                    continue
                group = [a]
                used[i] = True
                ax1, ay1 = float(a[:, 0].min()), float(a[:, 1].min())
                ax2, ay2 = float(a[:, 0].max()), float(a[:, 1].max())
                for j in range(i + 1, len(polished)):
                    if used[j]:
                        continue
                    b = polished[j]
                    bx1, by1 = float(b[:, 0].min()), float(b[:, 1].min())
                    bx2, by2 = float(b[:, 0].max()), float(b[:, 1].max())
                    # Flash-sized disk beside a tall module — keep separate.
                    bw, bh = bx2 - bx1, by2 - by1
                    if max(bw, bh) < min(ax2 - ax1, ay2 - ay1) * 0.45:
                        continue
                    overlap = not (
                        bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2
                    )
                    close = (
                        abs(0.5 * (bx1 + bx2) - 0.5 * (ax1 + ax2))
                        < max(ax2 - ax1, 1.0) * 0.85
                    )
                    if overlap or close:
                        group.append(b)
                        used[j] = True
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                if len(group) == 1:
                    merged.append(group[0])
                else:
                    merged.append(
                        np.array(
                            [
                                [ax1, ay1],
                                [ax2, ay1],
                                [ax2, ay2],
                                [ax1, ay2],
                            ],
                            dtype=np.float32,
                        )
                    )
            polished = merged
        combined = list(others) + list(polished)
        tags = [
            self._infer_heal_shape_tag(c) for c in combined
        ]
        self.set_hardware_exclusions(
            [np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in combined],
            snap_geometry=True,
            allow_clear=True,
            shape_tags=tags,
            persist=False,
            refit_design=False,
        )

    def perfect_finish_cutouts(self, scope: str = "all") -> int:
        """
        Selective production finish.

        scope:
          - "edges": only mesh perimeter / corners (expand wrap to rim)
          - "camera": only camera / flash cutout polish
          - "buttons": only side buttons / speakers exclusion
          - "all": everything (legacy behaviour)
        """
        if self.phone_image is None:
            return 0
        from .region_detector import HardwareRegionDetector
        from .mesh import AdaptiveMeshBuilder, create_mesh_mask
        from .cover_surface import CoverSurfaceEngine

        scope = str(scope or "all").lower().strip()
        if scope not in ("edges", "camera", "buttons", "all"):
            scope = "all"

        mesh_before = (
            None
            if self.control_mesh is None
            else self.control_mesh.points.copy()
        )
        count = 0

        # --- Edges & corners ---
        # Always snap to the photo wrap silhouette. Oversized edit cages are
        # ignored; volume stays wrapped (no auto punch on edges/all).
        if scope in ("edges", "all") and self.control_mesh is not None:
            self._invalidate_phone_wrap_cache()
            self._finalize_fullbleed_mesh()
            if self._phone_wrap_mask is not None:
                count = max(count, 1)

        # --- Side buttons: only when user explicitly chooses Buttons ---
        if scope == "buttons":
            painted = self.exclude_side_buttons()
            count = max(count, painted)

        # Flash / laser satellites — true circles (not on edges-only).
        if scope in ("camera", "all"):
            count = max(count, self._polish_flash_cutouts())

        # --- Camera / flash cutouts only ---
        if scope in ("camera", "all"):
            source = list(self.hardware_contours or [])
            if not source and self.exclusion_mask is not None:
                source = HardwareRegionDetector._smooth_exclusion_contours(
                    self.exclusion_mask
                )
            if source:
                cam = self._camera_like_contours(source)
                # Samsung / left-island cutouts were wrongly dropped as "buttons".
                if not cam:
                    cam = self._upper_cutouts(source)
                others = self._contours_not_in(source, cam)
                if cam:
                    # Locked editor tools (capsule / rectangle / …) must paint
                    # that geometry — never replace with a photo-traced blob.
                    locked_tools = {
                        "circle", "square", "rectangle", "rounded_square",
                        "rounded_rect", "oval", "pill_h", "pill_v", "capsule",
                        "button", "squircle", "superellipse", "triangle",
                    }
                    tags_all = list(self.cutout_shape_tags or [])
                    cam_tags: List[str] = []
                    other_tags: List[str] = []
                    tags_aligned = bool(
                        tags_all and len(tags_all) == len(source)
                    )
                    if tags_aligned:
                        cam_ids = {id(c) for c in cam}
                        for contour, tag in zip(source, tags_all):
                            if id(contour) in cam_ids:
                                cam_tags.append(str(tag or ""))
                            else:
                                other_tags.append(str(tag or ""))
                    user_locked = tags_aligned and any(
                        (t or "").lower().strip() in locked_tools
                        for t in cam_tags
                    )
                    if not tags_aligned:
                        # Fall back: any locked tag on the session keeps geometry.
                        user_locked = any(
                            (t or "").lower().strip() in locked_tools
                            for t in tags_all
                        )
                    if user_locked:
                        polished = [
                            np.asarray(c, dtype=np.float32).reshape(-1, 2)
                            for c in cam
                        ]
                    else:
                        rebuilt = HardwareRegionDetector.rebuild_camera_cutouts(
                            cam, self.phone_image
                        )
                        if rebuilt:
                            polished = [
                                np.asarray(c, dtype=np.float32).reshape(-1, 2)
                                for c in rebuilt
                            ]
                            cam_tags = [""] * len(polished)
                        else:
                            # Keep the user's selection — freeze snaps to photo.
                            # Do NOT force a stadium via _perfect_camera_module
                            # (that balloons Redmi / irregular islands).
                            polished = [
                                np.asarray(c, dtype=np.float32).reshape(-1, 2)
                                for c in cam
                            ]
                    if not polished:
                        polished = [
                            np.asarray(c, dtype=np.float32) for c in cam
                        ]
                    combined = list(others) + list(polished)
                    if not combined:
                        combined = list(source)
                    shape_tags = None
                    if tags_aligned:
                        shape_tags = list(other_tags) + list(cam_tags)
                        if len(shape_tags) != len(combined):
                            shape_tags = None
                    self.set_hardware_exclusions(
                        [np.asarray(c, dtype=np.float32).reshape(-1, 2)
                         for c in combined],
                        # Freeze geom from tags when locked; else photo snap.
                        snap_geometry=True,
                        allow_clear=False,
                        shape_tags=shape_tags,
                    )
                    count = max(count, len(combined))
                elif scope == "camera":
                    # Nothing classified as camera — keep every existing cutout.
                    count = max(count, len(source))

        # Absolute safety: only undo a truly exploded rebuild (corner check).
        if (
            mesh_before is not None
            and self.control_mesh is not None
            and scope in ("edges", "all")
        ):
            try:
                rows, cols = self.control_mesh.rows, self.control_mesh.cols
                if mesh_before.shape[0] == rows * cols:
                    before_c = np.array(
                        [
                            mesh_before[0],
                            mesh_before[cols - 1],
                            mesh_before[(rows - 1) * cols + (cols - 1)],
                            mesh_before[(rows - 1) * cols],
                        ],
                        dtype=np.float32,
                    )
                    after_c = self.control_mesh.corner_points()
                    span = float(
                        np.linalg.norm(
                            before_c.max(axis=0) - before_c.min(axis=0)
                        )
                    )
                    move = float(
                        np.max(np.linalg.norm(after_c - before_c, axis=1))
                    )
                    # Only roll back if corners flew off-canvas / collapsed.
                    if move > max(120.0, span * 0.85) or span < 8.0:
                        self.control_mesh.points[:] = mesh_before
                        self.cover_points = self.control_mesh.corner_points()
            except Exception:
                pass

        if count == 0 and mesh_before is not None:
            self._persist_manual_template()
            self.invalidate()
        elif count > 0:
            self._persist_manual_template()
            self.invalidate()
        return max(count, 1 if self.hardware_contours else 0)

    def _perfect_camera_module(
        self, cam: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Snap camera cutout(s) to one clean rounded rectangle.

        Merges nearby camera parts into a single module (iPhone island /
        Samsung stack), then fits a smooth rounded-rect to the phone photo
        while staying close to the user's edited border.
        """
        from .region_detector import HardwareRegionDetector

        parts = [
            np.asarray(c, dtype=np.float32).reshape(-1, 2)
            for c in cam
            if len(np.asarray(c).reshape(-1, 2)) >= 3
        ]
        if not parts:
            return []

        # Union bbox of all camera-like parts in the zone.
        all_pts = np.vstack(parts)
        ux1 = float(all_pts[:, 0].min())
        uy1 = float(all_pts[:, 1].min())
        ux2 = float(all_pts[:, 0].max())
        uy2 = float(all_pts[:, 1].max())
        # Merge only parts whose boxes actually touch. A flash/laser sitting
        # beside the lens stack must stay its own clean opening, otherwise the
        # module balloons far wider than the real hardware.
        merge = True
        if len(parts) >= 2:
            boxes = [
                (
                    float(p[:, 0].min()), float(p[:, 1].min()),
                    float(p[:, 0].max()), float(p[:, 1].max()),
                )
                for p in parts
            ]

            def _touching(a, b) -> bool:
                slack = 0.18 * min(
                    max(a[2] - a[0], a[3] - a[1]), max(b[2] - b[0], b[3] - b[1])
                )
                slack = max(slack, 2.0)
                return not (
                    a[2] + slack < b[0]
                    or b[2] + slack < a[0]
                    or a[3] + slack < b[1]
                    or b[3] + slack < a[1]
                )

            merge = all(
                _touching(boxes[i], boxes[j])
                for i in range(len(boxes))
                for j in range(i + 1, len(boxes))
            )

        gray = None
        if self.phone_image is not None and self.phone_image.size:
            phone = to_bgr(self.phone_image)
            gray = cv2.cvtColor(phone, cv2.COLOR_BGR2GRAY)

        def _one_clean(pts: np.ndarray, satellite: bool = False) -> np.ndarray:
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())
            user_w = max(x2 - x1, 1.0)
            user_h = max(y2 - y1, 1.0)
            if gray is not None and max(user_w, user_h) >= 14:
                sx1, sy1, sx2, sy2 = HardwareRegionDetector._snap_box_to_edges(
                    gray, x1, y1, x2, y2
                )
                fitted = HardwareRegionDetector._fit_hardware_box(
                    gray, sx1, sy1, sx2, sy2
                )
                if fitted is not None:
                    fx1, fy1, fx2, fy2 = fitted
                    # Stay close to the user's border — trust photo fit more.
                    x1 = float(np.clip(fx1, x1 - 0.05 * user_w, x1 + 0.05 * user_w))
                    y1 = float(np.clip(fy1, y1 - 0.05 * user_h, y1 + 0.05 * user_h))
                    x2 = float(np.clip(fx2, x2 - 0.05 * user_w, x2 + 0.05 * user_w))
                    y2 = float(np.clip(fy2, y2 - 0.05 * user_h, y2 + 0.05 * user_h))
            short = min(x2 - x1, y2 - y1)
            aspect = max(x2 - x1, y2 - y1) / max(short, 1.0)
            # Tiny pad so the ridge sits on the bezel, not inside glass.
            pad = max(0.4, short * 0.008)
            kind, _ = HardwareRegionDetector._classify_cutout(pts)
            round_hole = aspect <= 1.18 and (
                kind == "circle" or (satellite and aspect <= 1.30)
            )
            # Round satellite (flash / laser AF) → true circle, never a
            # rounded square, so the opening reads as a perfect lens hole.
            if round_hole:
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                radius = 0.25 * ((x2 - x1) + (y2 - y1))
                rcx, rcy, rr = HardwareRegionDetector._refine_circle(
                    gray, cx, cy, radius
                )
                if abs(rcx - cx) <= radius * 0.35 and abs(rcy - cy) <= radius * 0.35:
                    cx, cy = rcx, rcy
                    radius = float(np.clip(rr, radius * 0.8, radius * 1.2))
                circle = HardwareRegionDetector._sample_circle(
                    cx, cy, radius + pad, samples=64
                )
                if circle is not None:
                    return np.asarray(circle, dtype=np.float32).reshape(-1, 2)
            if aspect < 1.35:
                corner = float(np.clip(short * 0.32, 5.0, short * 0.44))
            else:
                corner = float(
                    np.clip(short * 0.48, 2.0, max(2.0, short * 0.5 - 0.5))
                )
            clean = HardwareRegionDetector._sample_rounded_rect(
                x1 - pad, y1 - pad, x2 + pad, y2 + pad, corner,
                samples_per_corner=16,
            )
            if clean is None:
                return pts
            return clean.reshape(-1, 2)

        if merge:
            return [_one_clean(all_pts)]
        areas = [
            float(
                (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min())
            )
            for p in parts
        ]
        biggest = max(areas) if areas else 0.0
        return [
            _one_clean(p, satellite=(biggest > 0 and a < biggest * 0.45))
            for p, a in zip(parts, areas)
        ]

    def _polish_flash_cutouts(self) -> int:
        """Re-freeze small round openings (flash / laser) as perfect circles."""
        if self.phone_image is None or not self.hardware_contours:
            return 0
        from .device_template import CutoutSpec, classify_cutout_kind
        from .region_detector import HardwareRegionDetector

        if self.control_mesh is None:
            return 0
        quad = self.control_mesh.corner_points()
        phone = to_bgr(self.phone_image)
        gray = cv2.cvtColor(phone, cv2.COLOR_BGR2GRAY)
        h, w = phone.shape[:2]
        # Largest top cutout ≈ camera island — flash is a small neighbour.
        cam_center = None
        cam_span = 0.0
        for contour in self.hardware_contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            bw = float(pts[:, 0].max() - pts[:, 0].min())
            bh = float(pts[:, 1].max() - pts[:, 1].min())
            span = max(bw, bh)
            cy = float(pts[:, 1].mean())
            if cy > h * 0.55:
                continue
            if span > cam_span:
                cam_span = span
                cam_center = pts.mean(axis=0)

        flash_specs: List = []
        other_contours: List[np.ndarray] = []
        other_specs: List = []
        for contour in self.hardware_contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            kind = classify_cutout_kind(pts, quad)
            bw = float(pts[:, 0].max() - pts[:, 0].min())
            bh = float(pts[:, 1].max() - pts[:, 1].min())
            short = max(min(bw, bh), 1.0)
            long = max(bw, bh)
            aspect = long / short
            area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
            circ = (4.0 * np.pi * area) / max(long * long, 1.0)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            near_cam = False
            if cam_center is not None:
                dist = float(np.linalg.norm(pts.mean(axis=0) - cam_center))
                near_cam = dist < max(cam_span * 1.8, min(w, h) * 0.18)
            smaller_than_island = (
                True if cam_span <= 0 else long < max(cam_span * 0.55, 8.0)
            )
            is_flash = kind == "flash" or (
                cy < h * 0.55
                and aspect < 1.65
                and long < min(w, h) * 0.11
                and short >= 3.5
                and (circ > 0.45 or near_cam)
                and smaller_than_island
            )
            # Never circle-ize the main camera island.
            if cam_span > 0 and long >= cam_span * 0.85:
                is_flash = False
            if not is_flash:
                other_contours.append(pts)
                continue
            cx, cy, radius = HardwareRegionDetector._fit_circle_least_squares(
                pts
            )
            if radius < 1.0:
                cx = 0.5 * (
                    float(pts[:, 0].min()) + float(pts[:, 0].max())
                )
                cy = 0.5 * (
                    float(pts[:, 1].min()) + float(pts[:, 1].max())
                )
                radius = 0.25 * (bw + bh)
            radius = float(np.clip(radius + 0.4, 1.5, min(w, h) * 0.08))
            circle = HardwareRegionDetector._sample_circle(
                float(cx), float(cy), radius, samples=72
            )
            circ_arr = np.asarray(circle, dtype=np.float32).reshape(-1, 2)
            norm = [
                [float(x / max(w, 1)), float(y / max(h, 1))]
                for x, y in circ_arr
            ]
            flash_specs.append(
                CutoutSpec(
                    kind="flash",
                    contour=norm,
                    geom="circle",
                    params=[float(cx), float(cy), float(radius)],
                    expand_px=1.25,
                    authoritative=True,
                )
            )
        if not flash_specs:
            return 0
        # Freeze others normally so we don't wipe camera islands.
        from .device_template import build_cutout_specs

        if other_contours:
            other_specs = build_cutout_specs(
                other_contours,
                quad,
                w,
                h,
                phone_gray=gray,
                authoritative=True,
            )
        combined_specs = list(other_specs) + flash_specs
        self.set_hardware_exclusions(
            [s.pixel_contour(w, h) for s in combined_specs],
            snap_geometry=True,
            allow_clear=False,
            cutout_specs=combined_specs,
        )
        return len(flash_specs)

    def _camera_like_contours(
        self, contours: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Keep cutouts in the upper camera zone (not thin side buttons)."""
        if self.control_mesh is None or not contours:
            return list(contours)
        corners = self.control_mesh.corner_points()
        xs = corners[:, 0]
        ys = corners[:, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)
        top_limit = y_min + height * 0.52
        # Only the extreme rim — camera islands sit inset from the bezel.
        side_band = width * 0.055
        kept: List[np.ndarray] = []
        for contour in contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            if cy > top_limit:
                continue
            bw = float(pts[:, 0].max() - pts[:, 0].min())
            bh = float(pts[:, 1].max() - pts[:, 1].min())
            aspect = max(bw, bh) / max(min(bw, bh), 1.0)
            near_side = (cx - x_min) <= side_band or (x_max - cx) <= side_band
            # Thin vertical capsules on the bezel = volume/power, not cameras.
            # Samsung camera islands are also left-sided but much wider.
            thin_button = (
                near_side
                and aspect >= 1.85
                and bw < width * 0.085
                and bh > height * 0.045
            )
            if thin_button:
                continue
            kept.append(contour)
        return kept

    def _cluster_camera_module(
        self, contours: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Group multi-part top camera hardware into one rounded module.

        This makes iPhone Pro / similar layouts export like real mockups on
        any model instead of trying to perfect-finish every lens/flash blob
        separately.
        """
        if self.control_mesh is None or len(contours) < 2:
            return contours
        from .region_detector import HardwareRegionDetector

        parts = [
            np.asarray(c, dtype=np.float32).reshape(-1, 2)
            for c in contours
            if len(np.asarray(c).reshape(-1, 2)) >= 3
        ]
        if len(parts) < 2:
            return contours

        all_pts = np.vstack(parts)
        corners = self.control_mesh.corner_points()
        x_min = float(corners[:, 0].min())
        x_max = float(corners[:, 0].max())
        y_min = float(corners[:, 1].min())
        y_max = float(corners[:, 1].max())
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)

        bx1 = float(all_pts[:, 0].min())
        by1 = float(all_pts[:, 1].min())
        bx2 = float(all_pts[:, 0].max())
        by2 = float(all_pts[:, 1].max())
        bw = bx2 - bx1
        bh = by2 - by1
        if bw < 20 or bh < 20:
            return contours
        if bw > width * 0.46 or bh > height * 0.34:
            return contours
        center_x = 0.5 * (bx1 + bx2)
        near_left = (center_x - x_min) <= width * 0.34
        near_right = (x_max - center_x) <= width * 0.34
        in_top = by2 <= y_min + height * 0.42
        if not in_top or not (near_left or near_right):
            return contours

        pad = max(4.0, min(bw, bh) * 0.12)
        x1 = bx1 - pad
        y1 = by1 - pad
        x2 = bx2 + pad
        y2 = by2 + pad
        short = min(x2 - x1, y2 - y1)
        corner = float(np.clip(short * 0.32, 8.0, short * 0.44))
        module = HardwareRegionDetector._sample_rounded_rect(
            x1, y1, x2, y2, corner, samples_per_corner=14
        )
        if module is None:
            return contours
        return [module.reshape(-1, 1, 2)]

    def _upper_cutouts(
        self, contours: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Fallback: any cutout in the upper half of the cover."""
        if self.control_mesh is None or not contours:
            return list(contours)
        corners = self.control_mesh.corner_points()
        y_min = float(corners[:, 1].min())
        y_max = float(corners[:, 1].max())
        mid = y_min + (y_max - y_min) * 0.50
        kept: List[np.ndarray] = []
        for contour in contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            if float(pts[:, 1].mean()) <= mid:
                kept.append(contour)
        return kept

    def _contours_not_in(
        self,
        contours: List[np.ndarray],
        exclude: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Contours whose centroid is not near any excluded cutout."""
        ex_centers = []
        for contour in exclude:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 3:
                ex_centers.append(pts.mean(axis=0))
        kept: List[np.ndarray] = []
        for contour in contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            c = pts.mean(axis=0)
            hit = any(
                float(np.linalg.norm(c - ec)) < 16.0 for ec in ex_centers
            )
            if not hit:
                kept.append(contour)
        return kept

    def _non_camera_contours(
        self, contours: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Opposite of _camera_like_contours."""
        return self._contours_not_in(
            contours, self._camera_like_contours(contours)
        )

    def redetect_cover(self) -> bool:
        """Re-run automatic cover-surface detection on the current phone image."""
        if self.phone_image is None:
            return False

        surface = self.cover_engine.analyze(
            self.phone_image, use_templates=False
        )
        self._apply_cover_surface(surface, auto_detected=True)
        if self.design_image is not None:
            self.auto_fit_design()
        self.invalidate()
        return self.control_mesh is not None

    def reset_cover_to_default(self) -> None:
        """Fall back to a centered phone-shaped cover estimate."""
        if self.phone_image is None:
            return

        surface = self.cover_engine.centered(self.phone_image)
        self._apply_cover_surface(surface, auto_detected=False)
        if self.design_image is not None:
            self.auto_fit_design()
        self.invalidate()

    def _apply_cover_surface(self, surface, auto_detected: bool) -> None:
        """Copy a CoverSurfaceResult into compositor state."""
        mesh = surface.mesh
        # Rebuild the wrap cage from the live phone/cover silhouette so
        # autosaves and minAreaRect seeds cannot leave a tilted or
        # unequal-margin sticker wrap on upright product shots.
        phone_mask = getattr(surface, "phone_mask", None)
        cover_mask = getattr(surface, "cover_mask", None)
        # Prefer a live upright silhouette over a skewed template mask so the
        # wrap cage stays axis-aligned on product shots.
        if self.phone_image is not None:
            try:
                live = CoverSurfaceEngine.estimate_phone_mask_from_photo(
                    self.phone_image
                )
                if live is not None and np.count_nonzero(live) > 64:
                    live_q = AdaptiveMeshBuilder._stable_quad_from_mask(live)
                    tpl_q = (
                        AdaptiveMeshBuilder._stable_quad_from_mask(phone_mask)
                        if phone_mask is not None
                        and np.count_nonzero(phone_mask) > 64
                        else None
                    )
                    live_tilt = (
                        AdaptiveMeshBuilder._quad_axis_deviation_deg(live_q)
                        if live_q is not None
                        else 99.0
                    )
                    tpl_tilt = (
                        AdaptiveMeshBuilder._quad_axis_deviation_deg(tpl_q)
                        if tpl_q is not None
                        else 99.0
                    )
                    if live_tilt + 0.5 < tpl_tilt or tpl_tilt > 2.0:
                        phone_mask = live
            except Exception:
                pass
        gate = None
        if phone_mask is not None and np.count_nonzero(phone_mask) > 64:
            gate = phone_mask
        elif cover_mask is not None and np.count_nonzero(cover_mask) > 64:
            gate = cover_mask
        if mesh is not None and gate is not None:
            corner = float(
                getattr(surface, "corner_radius_percent", None)
                or self.corner_radius_estimate
                or 8.0
            )
            radii = None
            if getattr(surface, "corner_radii", None) is not None:
                try:
                    radii = surface.corner_radii.as_tuple()
                except Exception:
                    radii = None
            mesh = AdaptiveMeshBuilder.production_perimeter(
                mesh,
                gate,
                corner_radius_percent=float(np.clip(corner, 2.5, 22.0)),
                max_move_fraction=0.18,
                corner_radii=radii,
                preserve_corner_arcs=True,
            )
        self.control_mesh = mesh
        self.cover_points = (
            None if mesh is None else mesh.corner_points()
        )
        self.exclusion_mask = surface.exclusion_mask
        self.printable_mask = surface.printable_mask
        self.hardware_contours = surface.hardware_contours
        # Phase 3: freeze typed cutouts from detected contours so export paint
        # uses the same authoritative holes as Perfect Finish.
        try:
            from .device_template import build_cutout_specs
            if self.phone_image is not None and surface.hardware_contours:
                h, w = self.phone_image.shape[:2]
                gray = cv2.cvtColor(to_bgr(self.phone_image), cv2.COLOR_BGR2GRAY)
                self.cutout_specs = build_cutout_specs(
                    surface.hardware_contours,
                    self.control_mesh.corner_points()
                    if self.control_mesh is not None
                    else surface.mesh.corner_points(),
                    w,
                    h,
                    phone_gray=gray,
                    authoritative=True,
                )
                # Re-paint exclusion from frozen specs (contour-true).
                if self.cutout_specs:
                    from .region_detector import HardwareRegionDetector
                    rebuilt = HardwareRegionDetector.paint_exclusion_from_specs(
                        self.cutout_specs, w, h
                    )
                    if np.count_nonzero(rebuilt):
                        self.exclusion_mask = rebuilt
            else:
                self.cutout_specs = []
        except Exception:
            self.cutout_specs = []
        self.detection_confidence = surface.confidence
        self.automatic_margin = surface.margin_percent
        self.corner_radii = surface.resolved_corner_radii()
        self.corner_radius_estimate = float(self.corner_radii.median())
        self.from_template = bool(surface.from_template)
        self.model_id = str(getattr(surface, "model_id", "") or "")
        self.auto_detected = auto_detected
        # Seed curved UV rim from detected print margin / settings.
        rim = estimate_rim_uv_from_margin(float(surface.margin_percent or 0.0))
        if float(self.settings.get("rim_uv", 5.5)) > 0:
            # Prefer explicit setting when user tuned it (stored as percent).
            rim_setting = float(self.settings.get("rim_uv", 5.5)) / 100.0
            if abs(rim_setting - DEFAULT_RIM_UV) > 0.008:
                rim = rim_setting
        self.curved_uv_params = CurvedUVParams(
            rim_uv=rim,
            bevel_strength=float(
                self.settings.get("bevel_strength", 92.0)
            ) / 100.0,
            corner_radii=self.corner_radii,
            enabled=float(self.settings.get("curved_uv", 1.0)) >= 0.5,
        )
        # Seed the corner slider from the detected cover roundness; smart-fit
        # may refine it when a design is present.
        self.settings['corner_radius'] = float(
            np.clip(self.corner_radius_estimate, 0.0, 30.0)
        )
        if phone_mask is not None and np.count_nonzero(phone_mask):
            self.cover_engine.last_phone_mask = phone_mask.copy()
        if cover_mask is not None and np.count_nonzero(cover_mask):
            self.cover_engine.last_cover_mask = cover_mask.copy()
        # Printable / wrap gate must follow the corrected mesh perimeter.
        self._sync_printable_from_mesh()
        # Template / GrabCut bites leave a bald camera strip — complete the
        # solid face against the live mesh cage (any phone colour / model).
        pm = getattr(self.cover_engine, "last_phone_mask", None)
        cover = getattr(self.cover_engine, "last_cover_mask", None)
        if (
            pm is not None
            and cover is not None
            and np.count_nonzero(pm)
            and np.count_nonzero(cover)
        ):
            healed = CoverSurfaceEngine.complete_phone_silhouette(
                pm, cover, phone_bgr=self.phone_image
            )
            if healed is not None and np.count_nonzero(healed) > 64:
                self.cover_engine.last_phone_mask = healed
                self._sync_printable_from_mesh()
        self._refresh_wrap_from_geometry()

    def _persist_manual_template(self) -> None:
        """Save the corrected cover mesh locally for this phone layout."""
        if self.phone_image is None or self.control_mesh is None:
            return
        try:
            slider = float(
                self.settings.get('corner_radius', self.corner_radius_estimate)
            )
            # Keep per-corner layout unless the global slider moved away.
            radii = self.corner_radii
            if abs(radii.median() - slider) > 0.35:
                radii = CornerRadii.uniform(slider)
                self.corner_radii = radii
            self.cover_engine.remember_correction(
                self.phone_image,
                self.control_mesh,
                self.exclusion_mask,
                margin_percent=self.automatic_margin,
                corner_radius_percent=slider,
                cover_mask=self.cover_engine.last_cover_mask,
                printable_mask=self.printable_mask,
                phone_mask=self.cover_engine.last_phone_mask,
                corner_radii=radii,
                hardware_contours=self.hardware_contours,
                cutouts=[s.to_dict() for s in self.cutout_specs],
                model_id=self.model_id,
            )
            if self.cover_engine.last_model_id:
                self.model_id = self.cover_engine.last_model_id
        except OSError as exc:
            # Template persistence must never break editing.
            logger.warning("Could not persist cover template: %s", exc)

    def save_device_template(
        self, model_id: str = "", display_name: str = ""
    ) -> Optional[str]:
        """
        Explicitly capture the current session as a named Phase 1 device model.

        Returns the saved ``model_id``, or None when capture is not possible.
        """
        if self.phone_image is None or self.control_mesh is None:
            return None
        catalog = DeviceTemplateCatalog()
        try:
            fp, _ = self.cover_engine.templates.fingerprint(
                self.phone_image,
                self.cover_engine.last_phone_mask,
            )
            template = catalog.capture_from_session(
                phone_image=self.phone_image,
                mesh=self.control_mesh,
                phone_mask=self.cover_engine.last_phone_mask,
                cover_mask=self.cover_engine.last_cover_mask,
                printable_mask=self.printable_mask,
                hardware_contours=self.hardware_contours,
                cutout_specs=self.cutout_specs,
                corner_radii=self.corner_radii,
                corner_radius_percent=float(
                    self.settings.get(
                        "corner_radius", self.corner_radius_estimate
                    )
                ),
                margin_percent=self.automatic_margin,
                fingerprint=fp,
                model_id=model_id or self.model_id or fp,
                display_name=display_name or model_id or self.model_id or fp[:8],
            )
            self.model_id = template.model_id
            self.cover_engine.last_model_id = template.model_id
            return template.model_id
        except Exception as exc:
            logger.warning("Could not save device template: %s", exc)
            return None

    def create_production_clone(self) -> "Compositor":
        """
        Clone phone geometry and look settings for an offline batch worker.

        Reuses the same local template manager so silhouette hits stay warm,
        but uses independent render caches so the interactive preview thread
        is never shared with batch rendering. Does not re-run detection.
        """
        if self.phone_image is None or self.control_mesh is None:
            raise ValueError('Phone geometry is required for batch production')

        clone = Compositor(template_cache=self.cover_engine.template_cache)
        # Share the same TemplateManager instance for cache reuse.
        clone.cover_engine.templates = self.cover_engine.templates
        clone.cover_engine.template_cache = self.cover_engine.template_cache

        clone.phone_image = self.phone_image.copy()
        clone.control_mesh = self.control_mesh.copy()
        clone.cover_points = (
            None if self.cover_points is None else self.cover_points.copy()
        )
        clone.exclusion_mask = (
            None if self.exclusion_mask is None
            else self.exclusion_mask.copy()
        )
        clone.printable_mask = (
            None if self.printable_mask is None
            else self.printable_mask.copy()
        )
        clone.hardware_contours = list(self.hardware_contours)
        clone.cutout_specs = [
            type(s).from_dict(s.to_dict()) for s in self.cutout_specs
        ]
        clone.detection_confidence = self.detection_confidence
        clone.automatic_margin = self.automatic_margin
        clone.smart_fit_confidence = self.smart_fit_confidence
        clone.corner_radius_estimate = self.corner_radius_estimate
        clone._product_body_corner = self._product_body_corner
        clone.corner_radii = CornerRadii(
            tl=self.corner_radii.tl,
            tr=self.corner_radii.tr,
            br=self.corner_radii.br,
            bl=self.corner_radii.bl,
        )
        clone.from_template = self.from_template
        clone.model_id = self.model_id
        clone.curved_uv_params = CurvedUVParams(
            rim_uv=self.curved_uv_params.rim_uv,
            bevel_strength=self.curved_uv_params.bevel_strength,
            corner_radii=clone.corner_radii,
            enabled=self.curved_uv_params.enabled,
        )
        clone.auto_detected = self.auto_detected
        clone.cover_engine.device_catalog = self.cover_engine.device_catalog
        clone.cover_engine.last_corner_radii = clone.corner_radii
        clone.cover_engine.last_model_id = self.model_id
        clone.fit_mode = self.fit_mode
        clone.mirror = self.mirror
        clone.material_name = self.material_name
        clone.lighting_name = self.lighting_name
        clone.settings = dict(self.settings)
        # Design is swapped per batch job.
        clone.design_image = None
        clone.invalidate(clear_scaled=True)
        return clone

    def set_fit_mode(self, mode: str) -> None:
        """Set how the design maps into the cover: fill, fit or stretch."""
        if mode in ('fill', 'fit', 'stretch') and mode != self.fit_mode:
            self.fit_mode = mode
            self.invalidate()

    def set_mirror(self, mirror: bool) -> None:
        """Flip the design horizontally."""
        if bool(mirror) != self.mirror:
            self.mirror = bool(mirror)
            self.invalidate()

    def auto_fit_design(self, *, preserve_placement: bool = False) -> Dict[str, float]:
        """Calculate cover placement on the live printable phone geometry."""
        if self.design_image is None or self.control_mesh is None:
            return {}
        # Fit to the phone-boundary wrap mesh (not a drifted edit cage).
        wrap_mesh, _pm = self._ensure_phone_wrap_geometry()
        fit_mesh = wrap_mesh if wrap_mesh is not None else self.control_mesh
        mode = self.fit_mode if self.fit_mode == "stretch" else "fill"
        if self.fit_mode == "fit":
            mode = "fit"
        kept = {}
        if preserve_placement:
            for key in ("offset_x", "offset_y", "rotation", "design_scale"):
                kept[key] = float(self.settings.get(key, DEFAULT_SETTINGS[key]))
        fit = SmartFitEstimator.estimate(
            self.design_image,
            fit_mesh,
            self.exclusion_mask,
            mode,
            margin_percent=self.automatic_margin,
            corner_radius_percent=self.corner_radius_estimate,
            printable_mask=self.printable_mask,
        )
        updates = fit.settings()
        if preserve_placement:
            updates.update(kept)
        else:
            # Hard-reset placement to geometry cover — never keep stale pan/tilt.
            updates["offset_x"] = 0.0
            updates["offset_y"] = 0.0
            updates["rotation"] = 0.0
            updates["region_inset"] = 0.0
            if mode == "fill":
                updates["design_scale"] = 100.0
        self.settings.update(updates)
        self.smart_fit_confidence = fit.confidence
        self.invalidate()
        return updates

    # ---------------------------------------------------------------- settings

    def update_settings(self, settings: Dict[str, float]) -> None:
        """Merge new adjustment values in and drop cached renders."""
        changed = False

        for key, value in settings.items():
            if self.settings.get(key) != value:
                self.settings[key] = value
                changed = True

        if changed:
            self.invalidate()

    def get_settings(self) -> Dict[str, float]:
        """Copy of the current settings."""
        return dict(self.settings)

    def reset(self) -> None:
        """Reset all adjustments to their defaults."""
        self.material_name = _DEFAULT_MATERIAL
        self.lighting_name = _DEFAULT_LIGHTING
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings.update(material_settings(self.material_name))
        self.settings.update(lighting_settings(self.lighting_name))
        self.mirror = False
        self.fit_mode = 'fill'
        self.invalidate()

    def apply_preset(self, name: str) -> Dict[str, float]:
        """
        Apply a named material, lighting, or legacy look preset.

        Material presets update surface response floats and texture kind.
        Lighting presets scale reflections/highlights only.
        Placement (scale/offset/rotation/inset/corners) is always preserved.

        Returns:
            The resulting settings so the UI can sync its sliders
        """
        if name not in PRESETS:
            return self.get_settings()

        placement = {
            key: self.settings.get(key, DEFAULT_SETTINGS[key])
            for key in (
                'design_scale', 'offset_x', 'offset_y', 'rotation',
                'region_inset', 'corner_radius',
            )
        }

        if name in MATERIALS:
            self.material_name = name
            self.settings.update(material_settings(name))
        elif name in LIGHTING:
            self.lighting_name = name
            self.settings.update(lighting_settings(name))
        elif name == 'Default':
            self.material_name = _DEFAULT_MATERIAL
            self.lighting_name = _DEFAULT_LIGHTING
            self.settings = dict(DEFAULT_SETTINGS)
            self.settings.update(material_settings(self.material_name))
            self.settings.update(lighting_settings(self.lighting_name))
        else:
            self.settings = dict(DEFAULT_SETTINGS)
            self.settings.update(copy.deepcopy(PRESETS[name]))
            mapped = _LEGACY_MATERIAL_MAP.get(name)
            if mapped is not None:
                self.material_name = mapped
            self.settings.update(lighting_settings(self.lighting_name))

        self.settings.update(placement)
        self.invalidate()
        return self.get_settings()

    def invalidate(self, clear_scaled: bool = False) -> None:
        """
        Drop cached renders (and optionally cached scaled inputs).

        The version counter is part of the cache key, so a render finishing on a
        worker thread after this call can never be served as a fresh result.
        """
        self._version += 1
        self._result_cache.clear()
        if clear_scaled:
            self._scaled_phone_cache.clear()
            self._invalidate_phone_wrap_cache()

    @property
    def is_ready(self) -> bool:
        """True when both images and a cover region are present."""
        return (self.phone_image is not None
                and self.design_image is not None
                and self.control_mesh is not None)

    # ----------------------------------------------------------------- render

    def render(self, max_size: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Render the composite.

        Args:
            max_size: Longest edge of the output; None renders at full size

        Returns:
            Composite as 8-bit BGR, or None when inputs are missing
        """
        if not self.is_ready:
            return None

        # Warp destination = detected phone rim. Edit cage size is ignored so
        # wrap is never under/over the device for any phone photo.
        wrap_mesh, wrap_mask = self._ensure_phone_wrap_geometry()
        if wrap_mesh is None:
            wrap_mesh = self.control_mesh
        if wrap_mask is not None:
            self.cover_engine.last_phone_mask = wrap_mask
        if self.printable_mask is None:
            self._sync_printable_from_phone_wrap()

        cache_key = (self._version, int(max_size or 0))
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            self._result_cache.move_to_end(cache_key)
            return cached

        phone, scale = self._get_scaled_phone(max_size)
        mesh = wrap_mesh.scaled(scale)

        inset = float(self.settings.get('region_inset', 0.0))
        if abs(inset) > 1e-6:
            mesh = mesh.inset(inset)

        exclusion = self._scaled_mask(self.exclusion_mask, phone.shape[:2])
        printable = self._scaled_mask(self.printable_mask, phone.shape[:2])
        # Scale phone wrap mask into printable if needed for clip fidelity.
        if printable is None and wrap_mask is not None:
            printable = self._scaled_mask(wrap_mask, phone.shape[:2])
        result = self._composite(phone, mesh, exclusion, printable)
        self._store_limited_cache(
            self._result_cache, cache_key, result,
            get_config().result_cache_size,
        )

        return result

    def get_preview(self, max_size: Optional[int] = None) -> Optional[np.ndarray]:
        """Render a preview sized composite."""
        preview_max = max_size or get_config().preview_max or self.PREVIEW_MAX
        return self.render(preview_max)

    def export(self, include_alpha: bool = False) -> Optional[np.ndarray]:
        """
        Render at full resolution for saving.

        Args:
            include_alpha: Return BGRA instead of BGR

        Returns:
            Final image, or None when inputs are missing
        """
        try:
            result = self.render(None)
        except MemoryError:
            logger.error("Export ran out of memory at full resolution")
            raise
        except Exception:
            logger.exception("Export render failed")
            raise

        if result is None:
            return None

        return to_bgra(result) if include_alpha else result

    def _get_scaled_phone(self, max_size: Optional[int]
                          ) -> Tuple[np.ndarray, float]:
        """Phone image scaled to the requested size, with the applied factor."""
        key = int(max_size or 0)
        cached = self._scaled_phone_cache.get(key)
        if cached is not None:
            self._scaled_phone_cache.move_to_end(key)
            return cached

        phone = self.phone_image
        scale = 1.0

        if max_size:
            h, w = phone.shape[:2]
            longest = max(h, w)
            if longest > max_size:
                scale = max_size / float(longest)
                phone = cv2.resize(phone, (max(1, int(round(w * scale))),
                                           max(1, int(round(h * scale)))),
                                   interpolation=cv2.INTER_AREA)

        self._store_limited_cache(
            self._scaled_phone_cache, key, (phone, scale),
            get_config().scaled_phone_cache_size,
        )
        return phone, scale

    @staticmethod
    def _store_limited_cache(cache: OrderedDict, key, value, limit: int) -> None:
        """Insert into an OrderedDict LRU with a hard entry cap."""
        cache[key] = value
        cache.move_to_end(key)
        max_entries = max(1, int(limit))
        while len(cache) > max_entries:
            cache.popitem(last=False)

    @staticmethod
    def _scaled_mask(
        mask: Optional[np.ndarray], shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Image-space mask resized to the current render resolution."""
        if mask is None:
            return None
        height, width = shape
        if mask.shape == (height, width):
            return mask
        return cv2.resize(
            mask, (width, height), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def _strip_studio_overflow_mask(
        mask: np.ndarray,
        phone_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Remove silhouette pixels on pure studio white far from device content.

        Fixes bottom-card drips and left-edge speckles without reshaping the
        true phone rim (1px halo around real content is kept for AA).
        """
        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 64 or phone_bgr is None:
            return binary
        phone = phone_bgr.astype(np.float32)
        if float(np.max(phone)) > 1.5:
            phone = phone / 255.0
        lum = phone.mean(axis=2)
        sat = phone.max(axis=2) - phone.min(axis=2)
        studio = (lum >= 0.92) & (sat <= 0.10)
        device = ~studio
        if not np.any(device):
            return binary
        near = cv2.dilate(
            device.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        overflow = (binary > 0) & studio & (near == 0)
        if np.any(overflow):
            binary = binary.copy()
            binary[overflow] = 0
        return binary

    @staticmethod
    def _studio_plate_pixels(phone_bgr: np.ndarray) -> np.ndarray:
        """
        Pixels that match the studio backdrop (not device content).

        Uses the median plate colour so white/light phones are not treated as
        backdrop — only samples that sit on the actual card.
        """
        h, w = phone_bgr.shape[:2]
        plate = phone_bgr.astype(np.float32)
        lum = plate.mean(axis=2)
        sat = plate.max(axis=2) - plate.min(axis=2)
        studio_px = (lum >= 245.0) & (sat <= 12.0)
        if int(np.count_nonzero(studio_px)) < 64:
            return np.zeros((h, w), dtype=bool)
        studio_rgb = np.median(plate[studio_px], axis=0)
        diff = np.abs(plate - studio_rgb.reshape(1, 1, 3)).mean(axis=2)
        return (diff <= 10.0) & (sat <= 16.0)

    @staticmethod
    def _clip_studio_plate_wrap(
        alpha: np.ndarray,
        mask: np.ndarray,
        phone_bgr: np.ndarray,
        *,
        protect: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Drop wrap that sits on pure studio white outside the real device.

        Detection overflow sometimes paints dark artwork onto the white card
        below the phone (bottom-center black patch). Corners/body on real
        device pixels are unchanged. Optional ``protect`` (button tips) is
        never cleared.
        """
        a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        m = np.clip(mask.astype(np.float32), 0.0, 1.0)
        phone = phone_bgr.astype(np.float32)
        if float(np.max(phone)) > 1.5:
            phone = phone / 255.0
        lum = phone.mean(axis=2)
        sat = phone.max(axis=2) - phone.min(axis=2)
        studio = (lum >= 0.92) & (sat <= 0.10)
        device = ~studio
        # Preserve 1px AA on real device content (side fringe, corners).
        near = cv2.dilate(
            device.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        overflow = studio & (near == 0) & ((a > 0.02) | (m > 0.02))
        # Bottom-center smudge: silhouette wrongly covers studio rows under the
        # phone. Those sit inside the 1px near-halo, so also clear wrap on
        # studio below the last real (non-studio) device row.
        content_ys = np.where(np.any(device, axis=1))[0]
        if content_ys.size > 0:
            y_bot = int(content_ys.max())
            below = np.arange(a.shape[0], dtype=np.int32)[:, None] > y_bot
            overflow = overflow | (
                studio & below & ((a > 0.02) | (m > 0.02))
            )
            # Mid-bottom only: opaque wrap on studio along the last content
            # rows (center smudge). Skip left/right corner bands so rounded
            # AA is unchanged.
            content_xs = np.where(np.any(device, axis=0))[0]
            if content_xs.size > 0:
                x0c, x1c = int(content_xs.min()), int(content_xs.max())
                margin = max(8, int(round(0.14 * (x1c - x0c + 1))))
                mid = np.zeros_like(studio, dtype=bool)
                mid[:, x0c + margin : max(x0c + margin, x1c - margin + 1)] = True
                on_bot = np.arange(a.shape[0], dtype=np.int32)[:, None] >= max(
                    0, y_bot - 1
                )
                overflow = overflow | (
                    studio & mid & on_bot & (a >= 0.45) & (m >= 0.45)
                )
        if protect is not None:
            prot = protect
            if prot.shape[:2] != a.shape[:2]:
                prot = cv2.resize(
                    prot.astype(np.uint8),
                    (a.shape[1], a.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            overflow = overflow & ~(prot.astype(bool))
        if np.any(overflow):
            a = np.where(overflow, 0.0, a)
            m = np.where(overflow, 0.0, m)
        return a, m

    @staticmethod
    def _spread_tip_wrap_texture(
        image: np.ndarray,
        tip_mask: np.ndarray,
        body_mask: np.ndarray,
        *,
        ink_lo: float,
        ink_hi: float,
    ) -> np.ndarray:
        """
        Spread inward body wrap onto each tip pixel (row-wise).

        Offsets the sample by how far the tip sits past the body wall so
        patterned artwork remains visible on thin side keys — not one flat
        column smeared across the whole button.
        """
        allowed = tip_mask.astype(bool)
        body = body_mask.astype(bool) & ~allowed
        if not np.any(allowed) or not np.any(body):
            return image
        h, w = allowed.shape[:2]
        out = image.copy()
        for y in range(h):
            txs = np.where(allowed[y])[0]
            if len(txs) == 0:
                continue
            bxs = np.where(body[y])[0]
            if len(bxs) == 0:
                continue
            row = out[y]
            lum = row.mean(axis=1) if row.ndim == 2 else row.astype(np.float32)
            tl, tr = int(txs.min()), int(txs.max())
            bl, br = int(bxs.min()), int(bxs.max())

            def _pick_sample(base_x: int) -> Optional[int]:
                target = 0.5 * (ink_lo + ink_hi)
                best_x: Optional[int] = None
                best_score = -1e9
                for probe in range(base_x, min(w, base_x + 18)):
                    lv = float(lum[probe])
                    if ink_lo <= lv < ink_hi:
                        score = -abs(lv - target)
                        if score > best_score:
                            best_score = score
                            best_x = probe
                            if abs(lv - target) < 0.02 * (ink_hi - ink_lo):
                                break
                if best_x is not None:
                    return best_x
                for probe in range(max(0, base_x - 4), base_x):
                    lv = float(lum[probe])
                    if ink_lo <= lv < ink_hi:
                        return probe
                return None

            if tr < bl:
                wall_x = bl
                for x in txs:
                    off = wall_x - int(x)
                    sx = _pick_sample(min(w - 1, wall_x + off + 8))
                    if sx is not None:
                        out[y, x] = row[sx]
            elif tl > br:
                wall_x = br
                for x in txs:
                    off = int(x) - wall_x
                    sx = _pick_sample(max(0, wall_x - off - 8))
                    if sx is not None:
                        out[y, x] = row[sx]
        return out

    def _paint_button_tips_from_body(
        self,
        design: np.ndarray,
        alpha: np.ndarray,
        mask: np.ndarray,
        tip_mask: np.ndarray,
        *,
        opacity: float,
        tip_cov: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Artwork on detected tip pixels only.

        Samples wrap RGB from the body on the same rows (inward of the tip) —
        exact detected tip contour, no enlarge into the body. Float tip_cov
        supplies the same soft AA the body gate uses (no hard rectangles).
        """
        allowed = tip_mask.astype(bool)
        if not np.any(allowed):
            return design, alpha, mask
        op = float(np.clip(opacity, 0.0, 1.0))
        h, w = allowed.shape[:2]
        # Gate falloff at the wall can sit below 0.80 — still valid wrap ink.
        body = (alpha > 0.42) & (mask > 0.42) & ~allowed
        if np.any(body):
            design = self._spread_tip_wrap_texture(
                design,
                allowed,
                body,
                ink_lo=0.06,
                ink_hi=0.88,
            )

        cov = tip_cov
        if cov is None:
            cov = allowed.astype(np.float32)
        else:
            cov = np.clip(cov.astype(np.float32), 0.0, 1.0)
            if cov.shape[:2] != (h, w):
                cov = cv2.resize(cov, (w, h), interpolation=cv2.INTER_LINEAR)
            # Solid coverage on validated tip pixels; soft cov only for AA.
            cov = np.where(
                allowed,
                np.maximum(cov, allowed.astype(np.float32) * 0.98),
                0.0,
            )
        mask = np.where(allowed, np.maximum(mask, cov), mask)
        alpha = np.where(allowed, np.maximum(alpha, cov * op), alpha)
        return design, alpha, mask

    def _composite_side_button_layer(
        self,
        output: np.ndarray,
        phone_bgr: np.ndarray,
        tip_mask: Optional[np.ndarray],
        phone_mask: Optional[np.ndarray] = None,
        tip_cov: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Isolated per-button wrap layer ABOVE the finished body composite.

        For each validated tip component (device-agnostic):
          1. Claim its full photo protrusion (raw & ~body) when available
          2. Build a localized side-face pad on the body wall (tip height only)
          3. Sample body artwork onto tip + face
          4. Apply reference-style top highlight / bottom shade / outer lip
          5. Composite above the body — never invent capsules or edge streaks
        """
        if (
            output is None
            or phone_bgr is None
            or tip_mask is None
            or np.count_nonzero(tip_mask) < 4
        ):
            return output
        h, w = output.shape[:2]
        tips = tip_mask > 127
        if tips.shape[:2] != (h, w):
            tips = (
                cv2.resize(
                    tip_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 127
            )
        if not np.any(tips):
            return output

        body = np.ones((h, w), dtype=bool)
        pm_u8 = None
        if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
            pm = phone_mask
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
            pm_u8 = (pm > 127).astype(np.uint8) * 255
            body = (pm > 127) & ~tips
        else:
            body = ~tips

        raw = self._phone_wrap_raw_mask
        if raw is not None and pm_u8 is not None:
            raw_r = raw
            if raw_r.shape[:2] != (h, w):
                raw_r = cv2.resize(
                    raw.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            tips = self._claim_full_raw_button_protrusions(
                tips, raw_r, pm_u8, None
            )
            plate = self._studio_plate_pixels(phone_bgr)
            if plate.shape[:2] != (h, w):
                plate = (
                    cv2.resize(
                        plate.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0
                )
            tips = tips & ~plate
            body = (pm_u8 > 127) & ~tips

        out_f = output.astype(np.float32)
        gray = out_f.mean(axis=2)
        body_px = body & (gray >= 14.0) & (gray < 170.0)
        if not np.any(body_px):
            body_px = body & (gray >= 10.0) & (gray < 210.0)
        if not np.any(body_px):
            return output

        plate = phone_bgr.astype(np.float32)
        if plate.shape[:2] != (h, w):
            plate = cv2.resize(plate, (w, h), interpolation=cv2.INTER_LINEAR)
        photo_g = plate.mean(axis=2)
        pl = photo_g
        ps = plate.max(axis=2) - plate.min(axis=2)
        studio_px = (pl >= 245.0) & (ps <= 12.0)
        if int(np.count_nonzero(studio_px)) >= 64:
            studio_rgb = np.median(plate[studio_px], axis=0)
        else:
            studio_rgb = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        tip_u8 = (tips.astype(np.uint8) * 255)
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
            tip_u8, connectivity=8
        )
        if nlab < 2:
            return output

        if tip_cov is not None and float(np.max(tip_cov)) > 0.05:
            cov_all = np.clip(tip_cov.astype(np.float32), 0.0, 1.0)
            if cov_all.shape[:2] != (h, w):
                cov_all = cv2.resize(
                    cov_all, (w, h), interpolation=cv2.INTER_LINEAR
                )
            cov_all = np.where(tips, cov_all, 0.0)
        else:
            cov_all = np.zeros((h, w), dtype=np.float32)

        result = out_f.copy()
        face_depth = max(1, min(3, int(round(min(h, w) * 0.007))))

        for lab in range(1, nlab):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < 4:
                continue
            tip_comp = labels == lab
            if not np.any(tip_comp):
                continue
            y0 = int(stats[lab, cv2.CC_STAT_TOP])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            y1 = min(h, y0 + bh)

            face_pad = np.zeros((h, w), dtype=bool)
            depth = max(1, min(face_depth, max(1, bw)))
            for y in range(y0, y1):
                if not np.any(tip_comp[y]):
                    continue
                txs = np.where(tip_comp[y])[0]
                bxs = np.where(body[y])[0]
                if len(txs) == 0 or len(bxs) == 0:
                    continue
                tmin, tmax = int(txs.min()), int(txs.max())
                bmin, bmax = int(bxs.min()), int(bxs.max())
                if tmax < bmin:
                    face_pad[y, bmin : min(w, bmin + depth)] = True
                elif tmin > bmax:
                    face_pad[y, max(0, bmax - depth + 1) : bmax + 1] = True
            face_pad = face_pad & body
            surface = tip_comp | face_pad

            cov = self._exact_coverage_aa(
                (tip_comp.astype(np.uint8) * 255), scale=8
            )
            cov = np.where(tip_comp, cov, 0.0)
            if np.any(cov_all > 0.05):
                cov = np.where(tip_comp, np.maximum(cov, cov_all), cov)
            if np.any(face_pad):
                cov = np.where(face_pad, np.maximum(cov, 0.97), cov)

            for y in range(y0, y1):
                txs = np.where(tip_comp[y])[0]
                if len(txs) == 0:
                    continue
                tmin, tmax = int(txs.min()), int(txs.max())
                bxs = np.where(body[y])[0]
                if len(bxs) == 0:
                    cov[y, txs] = np.maximum(cov[y, txs], 0.94)
                    continue
                bmin, bmax = int(bxs.min()), int(bxs.max())
                if tmax < bmin:
                    cov[y, tmax] = max(float(cov[y, tmax]), 0.99)
                    if tmax > tmin:
                        cov[y, tmin] = min(max(float(cov[y, tmin]), 0.72), 0.92)
                elif tmin > bmax:
                    cov[y, tmin] = max(float(cov[y, tmin]), 0.99)
                    if tmax > tmin:
                        cov[y, tmax] = min(max(float(cov[y, tmax]), 0.72), 0.92)

            painted = self._spread_tip_wrap_texture(
                result,
                surface,
                body_px & ~face_pad,
                ink_lo=14.0,
                ink_hi=170.0,
            )

            yy = np.arange(h, dtype=np.float32)
            t_vert = np.zeros(h, dtype=np.float32)
            if bh > 1:
                t_vert[y0:y1] = (yy[y0:y1] - float(y0)) / float(bh - 1)
            else:
                t_vert[y0:y1] = 0.5
            vert_k = (
                1.22
                - 0.18 * t_vert
                - 0.20 * np.clip((t_vert - 0.72) / 0.28, 0.0, 1.0)
            )
            vert_k = np.clip(vert_k, 0.78, 1.28).astype(np.float32)

            outer = np.zeros_like(surface)
            face = np.zeros_like(surface)
            for y in range(y0, y1):
                xs = np.where(surface[y])[0]
                if len(xs) == 0:
                    continue
                xmin, xmax = int(xs.min()), int(xs.max())
                bxs = np.where(body[y] | tip_comp[y])[0]
                if len(bxs) == 0:
                    outer[y, xs] = True
                    continue
                bmin, bmax = int(bxs.min()), int(bxs.max())
                if abs(xmin - bmin) <= abs(xmax - bmax):
                    outer[y, xmin] = True
                    face[y, xs[xs > xmin]] = True
                    if not np.any(face[y]):
                        face[y, xmax] = True
                else:
                    outer[y, xmax] = True
                    face[y, xs[xs < xmax]] = True
                    if not np.any(face[y]):
                        face[y, xmin] = True

            shaded = painted.copy()
            for y in range(y0, y1):
                if not np.any(surface[y]):
                    continue
                shaded[y, surface[y]] = np.clip(
                    shaded[y, surface[y]] * float(vert_k[y]), 0.0, 255.0
                )
            if np.any(face):
                shaded[face] = np.clip(shaded[face] * 1.08, 0.0, 255.0)
            if np.any(outer):
                shaded[outer] = np.clip(shaded[outer] * 0.78, 0.0, 255.0)
            if np.any(face_pad):
                junction = face_pad & (
                    cv2.dilate(
                        tip_comp.astype(np.uint8) * 255,
                        np.ones((3, 3), np.uint8),
                        iterations=1,
                    )
                    > 0
                )
                if np.any(junction):
                    shaded[junction] = np.clip(
                        shaded[junction] * 0.86, 0.0, 255.0
                    )

            blended = shaded * cov[:, :, np.newaxis] + studio_rgb.reshape(
                1, 1, 3
            ) * (1.0 - cov[:, :, np.newaxis])
            # Binary tip / side-face pixels get solid wrap — studio blend here
            # turned 1–2px keys into faint gray outlines.
            solid = surface & ((cov >= 0.50) | tip_comp | face_pad)
            solid = solid & ~studio_px
            soft = surface & ~solid & (cov > 0.08) & ~studio_px
            result = np.where(solid[:, :, np.newaxis], shaded, result)
            result = np.where(soft[:, :, np.newaxis], blended, result)

            if np.any(face_pad):
                groove = body & ~face_pad & (
                    cv2.dilate(
                        face_pad.astype(np.uint8) * 255,
                        np.ones((3, 3), np.uint8),
                        iterations=1,
                    )
                    > 0
                )
                row_band = np.zeros(h, dtype=bool)
                row_band[y0:y1] = True
                groove = groove & row_band[:, np.newaxis]
                if np.any(groove):
                    result[groove] = np.clip(result[groove] * 0.90, 0.0, 255.0)

        return np.clip(np.round(result), 0, 255).astype(np.uint8)

    @staticmethod
    def _kill_studio_print_fringe(
        alpha: np.ndarray,
        phone_bgr: np.ndarray,
        *,
        phone_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Strip only the soft translucent halo that sits *outside* the wrap
        core on a bright studio plate (halka red fringe).

        Never multiplies by a device/silver mask — that wiped light phones.
        """
        a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        core = (a >= 0.55).astype(np.uint8) * 255
        if np.count_nonzero(core) < 64:
            return a
        # Keep AA tip within ~2px of the opaque wrap.
        keep = cv2.dilate(
            core,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        phone = phone_bgr.astype(np.float32)
        if float(np.max(phone)) > 1.5:
            phone = phone / 255.0
        lum = phone.mean(axis=2)
        sat = phone.max(axis=2) - phone.min(axis=2)
        # Soft ink beyond the wrap, on near-white card only.
        fringe = (
            (keep == 0)
            & (a > 0.002)
            & (a < 0.60)
            & (lum >= 0.90)
            & (sat <= 0.12)
        )
        if phone_mask is not None and np.count_nonzero(phone_mask):
            pm = phone_mask
            if pm.shape[:2] != a.shape[:2]:
                pm = cv2.resize(
                    pm,
                    (a.shape[1], a.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            # Extra: soft ink clearly outside product footprint on white.
            fringe = fringe | (
                (pm < 40)
                & (keep == 0)
                & (a > 0.002)
                & (a < 0.60)
                & (lum >= 0.90)
            )
        return np.where(fringe, 0.0, a).astype(np.float32)

    @staticmethod
    def _smooth_silhouette_coverage(mask: np.ndarray) -> np.ndarray:
        """
        Float 0–1 coverage of a binary silhouette with production-smooth AA.

        Uses the outer contour + supersample fill so phone/cover gates do not
        reintroduce stair-steps onto the print rim.
        """
        from .mesh import _fill_closed_polyline_aa

        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 32:
            return np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
        outer = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
            np.float32
        )
        if outer.shape[0] >= 16:
            outer = AdaptiveMeshBuilder._smooth_closed_polyline(
                outer,
                window=max(5, min(11, (outer.shape[0] // 75) * 2 + 1)),
            )
        cover = _fill_closed_polyline_aa(
            outer, binary.shape[:2], scale=6, expand_px=0.35
        )
        cover = np.where(cover > 0.82, np.maximum(cover, 0.97), cover)
        return np.clip(cover, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _fill_silhouette_holes(binary: np.ndarray) -> np.ndarray:
        """
        Fill interior holes (camera/flash) so only the OUTER product outline
        is used for boundary AA. Does not grow or shrink the outer edge.
        """
        bin_u8 = (binary > 0).astype(np.uint8) * 255
        if np.count_nonzero(bin_u8) < 64:
            return (binary > 0).astype(np.uint8)
        h, w = bin_u8.shape[:2]
        inv = cv2.bitwise_not(bin_u8)
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(inv, flood_mask, (0, 0), 0)
        # Remaining non-zero in inv = closed interior holes.
        filled = cv2.bitwise_or(bin_u8, inv)
        return (filled > 0).astype(np.uint8)

    @staticmethod
    def _exact_coverage_aa(binary: np.ndarray, scale: int = 4) -> np.ndarray:
        """
        Sub-pixel coverage of an exact binary silhouette.

        Upsamples with NEAREST (geometry unchanged) and downsamples once with
        AREA — no blur, dilate, erode, or padding.
        """
        h, w = map(int, binary.shape[:2])
        bin_u8 = (binary > 0).astype(np.uint8) * 255
        ss = max(1, int(scale))
        if ss <= 1:
            return bin_u8.astype(np.float32) / 255.0
        big = cv2.resize(
            bin_u8, (w * ss, h * ss), interpolation=cv2.INTER_NEAREST
        )
        return (
            cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            / 255.0
        )

    @staticmethod
    def _rasterize_outer_boundary_aa(
        output: np.ndarray,
        phone_bgr: np.ndarray,
        coverage: np.ndarray,
        hole_w: Optional[np.ndarray],
        gate_f: Optional[np.ndarray] = None,
        corner_w: Optional[np.ndarray] = None,
        button_cov: Optional[np.ndarray] = None,
        tip_mask: Optional[np.ndarray] = None,
        phone_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Final outer-perimeter pass using continuous geometric gate coverage.

        Root cause of stairs: binary phone silhouette culls re-quantized the
        already-AA rounded path. Here the float gate is applied directly with
        no 0/255 threshold. Camera/inner pixels stay unchanged.
        """
        if output is None or phone_bgr is None:
            return output
        if gate_f is None or float(np.max(gate_f)) < 0.05:
            return output

        h, w = output.shape[:2]
        g = np.clip(gate_f.astype(np.float32), 0.0, 1.0)
        if g.shape[:2] != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
        g = np.clip(g, 0.0, 1.0)

        out_f = output.astype(np.float32)
        if phone_bgr is not None:
            plate = phone_bgr.astype(np.float32)
            pl = plate.mean(axis=2)
            ps = plate.max(axis=2) - plate.min(axis=2)
            studio_px = (pl >= 245.0) & (ps <= 12.0)
            if int(np.count_nonzero(studio_px)) >= 64:
                studio_rgb = np.median(plate[studio_px], axis=0)
            else:
                studio_rgb = np.array([255.0, 255.0, 255.0], dtype=np.float32)
            # Drop gate on studio below device content in the MID-BOTTOM only.
            # Corner bands keep float AA (same path as the clean top-right).
            content_ys = np.where(np.any(~studio_px, axis=1))[0]
            if content_ys.size > 0:
                below = (
                    np.arange(h, dtype=np.int32)[:, None] > int(content_ys.max())
                )
                mid = np.ones((h, w), dtype=bool)
                if corner_w is not None and float(np.max(corner_w)) > 0.05:
                    cw = corner_w
                    if cw.shape[:2] != (h, w):
                        cw = cv2.resize(
                            cw.astype(np.float32),
                            (w, h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    mid = np.clip(cw, 0.0, 1.0) < 0.28
                g = np.where(studio_px & below & mid, 0.0, g)
        else:
            studio_rgb = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        # Deep interior stays opaque. Mid-side walls stay straight. Corner
        # pockets keep the float geometric gate (forcing binary-on here was
        # what re-quantized rounded arcs back into stairs).
        body_on = np.zeros((h, w), dtype=bool)
        if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
            pm = phone_mask
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
            body_on = pm > 127
            dist_in = cv2.distanceTransform(
                body_on.astype(np.uint8), cv2.DIST_L2, 5
            )
            dist_out = cv2.distanceTransform(
                (1 - body_on.astype(np.uint8)), cv2.DIST_L2, 5
            ).astype(np.float32)
            mid_force = np.ones((h, w), dtype=bool)
            if corner_w is not None and float(np.max(corner_w)) > 0.05:
                cw0 = corner_w
                if cw0.shape[:2] != (h, w):
                    cw0 = cv2.resize(
                        cw0.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mid_force = np.clip(cw0, 0.0, 1.0) < 0.16
            g = np.where(mid_force & body_on & (dist_in >= 1.0), 1.0, g)
            # Clip stray wrap on straight walls only. Corner AA lives in the
            # 1px exterior ramp — zeroing all dist_out>1.05 re-quantized arcs.
            outside = mid_force & (~body_on) & (dist_out > 0.55)
            far = (~body_on) & (dist_out > 1.8)
            if tip_mask is not None and np.count_nonzero(tip_mask) >= 4:
                tm0 = tip_mask
                if tm0.shape[:2] != (h, w):
                    tm0 = cv2.resize(
                        tm0.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                outside = outside & ~(tm0 > 127)
                far = far & ~(tm0 > 127)
            g = np.where(outside | far, 0.0, g)
            g = np.clip(g, 0.0, 1.0)

        blended = out_f * g[:, :, np.newaxis] + studio_rgb.reshape(1, 1, 3) * (
            1.0 - g[:, :, np.newaxis]
        )

        apply = g < 0.985
        if hole_w is not None and float(np.max(hole_w)) > 0.05:
            hw = hole_w
            if hw.shape[:2] != (h, w):
                hw = cv2.resize(hw, (w, h), interpolation=cv2.INTER_LINEAR)
            apply = apply & (np.clip(hw, 0.0, 1.0) < 0.18)
            cut_g = Compositor._cutout_guard(hole_w, (h, w), margin_px=5.0)
            if cut_g is not None:
                apply = apply & ~cut_g
        if button_cov is not None and float(np.max(button_cov)) > 0.05:
            bf = np.clip(button_cov.astype(np.float32), 0.0, 1.0)
            if bf.shape[:2] != (h, w):
                bf = cv2.resize(bf, (w, h), interpolation=cv2.INTER_LINEAR)
            apply = apply & (bf < 0.30)
        if tip_mask is not None and np.count_nonzero(tip_mask) >= 4:
            tm = tip_mask
            if tm.shape[:2] != (h, w):
                tm = cv2.resize(
                    tm.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            apply = apply & ~(tm > 127)

        # Mid-side body columns must stay solid wrap. Soft-gate wash here
        # paints a light vertical "blur line" inside the silhouette (esp. right
        # wall). Soft AA stays on the true outer half-pixel / outside only.
        mid_side = np.ones((h, w), dtype=bool)
        if corner_w is not None and float(np.max(corner_w)) > 0.05:
            cw = corner_w
            if cw.shape[:2] != (h, w):
                cw = cv2.resize(
                    cw.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
            mid_side = np.clip(cw, 0.0, 1.0) < 0.16
        body_u8 = np.zeros((h, w), dtype=np.uint8)
        if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
            pm = phone_mask
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
            body_u8 = (pm > 127).astype(np.uint8) * 255
        elif coverage is not None:
            cov = np.clip(coverage.astype(np.float32), 0.0, 1.0)
            if cov.shape[:2] != (h, w):
                cov = cv2.resize(cov, (w, h), interpolation=cv2.INTER_LINEAR)
            body_u8 = (cov > 0.5).astype(np.uint8) * 255
        dist_in = np.zeros((h, w), dtype=np.float32)
        if np.count_nonzero(body_u8) >= 64:
            dist_in = cv2.distanceTransform(
                (body_u8 > 127).astype(np.uint8), cv2.DIST_L2, 5
            )
            solid_mid = mid_side & (body_u8 > 127) & (dist_in >= 1.0)
            apply = apply & ~solid_mid

        if not np.any(apply):
            result = np.clip(np.round(out_f), 0, 255).astype(np.uint8)
        else:
            result = np.where(apply[:, :, np.newaxis], blended, out_f)
            result = np.clip(np.round(result), 0, 255).astype(np.uint8)

        # Mid-side body must be solid wrap ink — never studio wash / light line.
        gray_chk = result.mean(axis=2)
        washed = np.zeros((h, w), dtype=bool)
        if np.count_nonzero(body_u8) >= 64:
            washed = (
                mid_side
                & (body_u8 > 127)
                & (dist_in >= 1.0)
                & (gray_chk > 40.0)
            )
            if tip_mask is not None and np.count_nonzero(tip_mask) >= 4:
                tm = tip_mask
                if tm.shape[:2] != (h, w):
                    tm = cv2.resize(
                        tip_mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                washed = washed & ~(tm > 127)
            if hole_w is not None and float(np.max(hole_w)) > 0.05:
                cut_g = Compositor._cutout_guard(hole_w, (h, w), margin_px=6.0)
                if cut_g is not None:
                    washed = washed & ~cut_g
        if np.any(washed):
            result = result.astype(np.float32)
            ys_w, xs_w = np.where(washed)
            for y, x in zip(ys_w, xs_w):
                src = None
                # Prefer inward (toward phone center).
                body_xs = np.where(body_u8[y] > 127)[0]
                if len(body_xs) == 0:
                    continue
                bmin, bmax = int(body_xs.min()), int(body_xs.max())
                # Left wall → sample right; right wall → sample left.
                if x <= bmin + 2:
                    search = range(x + 2, min(w, x + 16))
                elif x >= bmax - 2:
                    search = range(x - 2, max(-1, x - 16), -1)
                else:
                    search = list(range(x + 2, min(w, x + 12))) + list(
                        range(x - 2, max(-1, x - 12), -1)
                    )
                for cand in search:
                    if (
                        body_u8[y, cand] > 127
                        and 8.0 <= gray_chk[y, cand] < 45.0
                    ):
                        src = cand
                        break
                if src is not None:
                    result[y, x] = result[y, src]
            result = np.clip(np.round(result), 0, 255).astype(np.uint8)

        studio_bgr = np.clip(np.round(studio_rgb), 0, 255).astype(np.uint8)
        gray_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # Outside-ish soft gate: any leftover dark stair chips → re-blend or
        # studio. Does not touch opaque interior (g≈1) or tip masks.
        weak = (g < 0.55) & (gray_r < 100)
        if tip_mask is not None and np.count_nonzero(tip_mask) >= 4:
            tm = tip_mask
            if tm.shape[:2] != (h, w):
                tm = cv2.resize(
                    tm.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            weak = weak & ~(tm > 127)
        if np.any(weak):
            result = result.copy()
            # Prefer the already-supersampled blend; fall back to studio when
            # gate is nearly empty.
            near_empty = weak & (g < 0.15)
            mid_soft = weak & ~near_empty
            if np.any(mid_soft):
                result[mid_soft] = np.clip(
                    np.round(blended[mid_soft]), 0, 255
                ).astype(np.uint8)
            if np.any(near_empty):
                result[near_empty] = studio_bgr
        # Absolute outside: kill residual dark chips on near-zero gate.
        chip = (g < 0.06) & (gray_r < 140)
        if tip_mask is not None and np.count_nonzero(tip_mask) >= 4:
            tm = tip_mask
            if tm.shape[:2] != (h, w):
                tm = cv2.resize(
                    tm.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            chip = chip & ~(tm > 127)
        if np.any(chip):
            result = result.copy()
            result[chip] = studio_bgr
        return result

    @staticmethod
    def _trim_exterior_speckles(
        alpha: np.ndarray,
        phone_mask: Optional[np.ndarray],
        tip_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Drop ink outside the phone footprint only — never shrink interior fill.

        Cleans stray rim grains on the studio card without touching full wrap.
        Validated side-button tips sit past the body wall and must be kept.
        """
        if phone_mask is None or np.count_nonzero(phone_mask) < 64:
            return alpha
        pm = phone_mask
        if pm.shape[:2] != alpha.shape[:2]:
            pm = cv2.resize(
                pm,
                (alpha.shape[1], alpha.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        phone_bin = (pm > 127).astype(np.uint8)
        # 1–2 px AA halo is allowed; anything beyond is studio spill.
        halo = cv2.dilate(
            phone_bin,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        if tip_mask is not None and np.count_nonzero(tip_mask) >= 4:
            tip = tip_mask
            if tip.shape[:2] != alpha.shape[:2]:
                tip = cv2.resize(
                    tip.astype(np.uint8),
                    (alpha.shape[1], alpha.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            halo = np.maximum(halo, (tip > 127).astype(np.uint8) * 255)
        out = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        out = np.where(halo == 0, 0.0, out)
        return out

    @staticmethod
    def _cutout_guard(
        hole_w: Optional[np.ndarray],
        shape: Tuple[int, int],
        *,
        margin_px: float = 5.0,
    ) -> Optional[np.ndarray]:
        """Pixels near camera/flash holes — never apply outer-rim polish here."""
        if hole_w is None or float(np.max(hole_w)) < 0.05:
            return None
        h, w = map(int, shape)
        hw = hole_w
        if hw.shape[:2] != (h, w):
            hw = cv2.resize(hw, (w, h), interpolation=cv2.INTER_LINEAR)
        core = (np.clip(hw, 0.0, 1.0) > 0.42).astype(np.uint8)
        if np.count_nonzero(core) < 8:
            return (hw > 0.08)
        dist_out = cv2.distanceTransform(
            (1 - core).astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float32)
        return dist_out <= max(2.0, float(margin_px))

    @staticmethod
    def _outer_rim_band_maps(
        coverage: np.ndarray,
        shape: Tuple[int, int],
        corner_w: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Rim bands from the true outer product edge only — ignores cutout arcs.
        """
        from .materials import MaterialRenderingEngine

        h, w = map(int, shape)
        rim_px_side = max(2.5, float(min(h, w)) * 0.007)
        rim_px_corner = max(5.5, float(min(h, w)) * 0.014)
        if corner_w is not None and corner_w.shape[:2] == (h, w):
            cw = np.clip(corner_w.astype(np.float32), 0.0, 1.0)
            local_rim = rim_px_side + (rim_px_corner - rim_px_side) * cw
        else:
            local_rim = np.full((h, w), rim_px_side, dtype=np.float32)
        dist_in = MaterialRenderingEngine._outer_perimeter_distance(
            np.clip(coverage.astype(np.float32), 0.0, 1.0)
        )
        interior = dist_in > local_rim
        rim = (dist_in > 0.0) & ~interior
        return dist_in, interior, rim, local_rim

    @staticmethod
    def _rim_band_maps(
        phone_bin: np.ndarray,
        shape: Tuple[int, int],
        corner_w: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-pixel interior / rim bands — wider at rounded corners."""
        h, w = map(int, shape)
        rim_px_side = max(2.5, float(min(h, w)) * 0.007)
        rim_px_corner = max(5.5, float(min(h, w)) * 0.014)
        if corner_w is not None and corner_w.shape[:2] == (h, w):
            cw = np.clip(corner_w.astype(np.float32), 0.0, 1.0)
            local_rim = rim_px_side + (rim_px_corner - rim_px_side) * cw
        else:
            local_rim = np.full((h, w), rim_px_side, dtype=np.float32)
        dist_in = cv2.distanceTransform(phone_bin, cv2.DIST_L2, 5).astype(
            np.float32
        )
        interior = dist_in > local_rim
        rim = (dist_in > 0.0) & ~interior
        return dist_in, interior, rim, local_rim

    @staticmethod
    def _gate_wrap_envelope(
        rim: np.ndarray,
        phone_mask: np.ndarray,
        gate_f: Optional[np.ndarray],
        shape: Tuple[int, int],
        *,
        corner_w: Optional[np.ndarray] = None,
        coverage: Optional[np.ndarray] = None,
        hole_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Final wrap envelope from the geometric gate — never binary photo stairs.
        """
        h, w = map(int, shape)
        env = np.clip(rim.astype(np.float32), 0.0, 1.0)
        if gate_f is None or float(np.max(gate_f)) < 0.05:
            return env

        gate = np.clip(gate_f.astype(np.float32), 0.0, 1.0)
        if gate.shape[:2] != (h, w):
            gate = cv2.resize(gate, (w, h), interpolation=cv2.INTER_LINEAR)
        gate = np.clip(gate, 0.0, 1.0)

        pm = phone_mask
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
        phone_bin = (pm > 127).astype(np.uint8)
        dist_out = cv2.distanceTransform(
            (1 - phone_bin).astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float32)

        # Rim from the phone silhouette — NOT from mesh coverage. An oversized
        # edit-cage fill marks the real phone edge as "interior", so the float
        # gate never lands on the visible outer perimeter (stairs remain).
        _, interior, rim_band, _ = Compositor._rim_band_maps(
            phone_bin, (h, w), corner_w
        )
        guard = Compositor._cutout_guard(hole_w, (h, w))
        if guard is not None:
            interior = interior & ~guard
            rim_band = rim_band & ~guard
        # Soft fringe from the geometric gate — not binary mask stairs.
        fringe_px = max(2.5, float(min(h, w)) * 0.006)
        gate_soft = (gate > 0.02) & (gate < 0.985)
        # In corner pockets force the geometric arc even when the photo rim
        # band thinks those pixels are "interior" (stairs would remain).
        if corner_w is not None and corner_w.shape[:2] == (h, w):
            cw = np.clip(corner_w.astype(np.float32), 0.0, 1.0)
            edge = rim_band | gate_soft | (cw > 0.18)
        else:
            edge = rim_band | gate_soft

        env = np.where(interior & ~edge, np.maximum(env, 0.985), env)
        env = np.where(edge, gate, env)
        # Corner pockets only: force geometric arcs (kills photo stairs).
        # Mid-sides keep the straight body envelope — never carve white gaps.
        if corner_w is not None and corner_w.shape[:2] == (h, w):
            cw = np.clip(corner_w.astype(np.float32), 0.0, 1.0)
            env = np.where(cw > 0.18, np.minimum(env, gate), env)
        env = np.where((gate < 0.015) & (dist_out > fringe_px), 0.0, env)
        env = np.clip(env, 0.0, 1.0)
        if hole_w is not None:
            hw = hole_w
            if hw.shape[:2] != (h, w):
                hw = cv2.resize(hw, (w, h), interpolation=cv2.INTER_LINEAR)
            env = env * (1.0 - np.clip(hw, 0.0, 1.0))
        return env

    @staticmethod
    def _apply_rim_antialias(
        mask: np.ndarray,
        alpha: Optional[np.ndarray],
        phone_mask: np.ndarray,
        gate_f: Optional[np.ndarray],
        shape: Tuple[int, int],
        *,
        touch_mask: bool = True,
        touch_alpha: bool = True,
        corner_w: Optional[np.ndarray] = None,
        coverage: Optional[np.ndarray] = None,
        hole_w: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Outer-rim-only anti-alias: smooth geometric gate on the product edge;
        camera/flash cutout curves are never touched.
        """
        h, w = map(int, shape)
        pm = phone_mask
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
        phone_bin = (pm > 127).astype(np.uint8)
        if np.count_nonzero(phone_bin) < 64:
            m = np.clip(mask.astype(np.float32), 0.0, 1.0)
            if alpha is not None and touch_alpha:
                return m, np.clip(alpha.astype(np.float32), 0.0, 1.0)
            return m, alpha

        dist_out = cv2.distanceTransform(
            (1 - phone_bin).astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float32)

        m = np.clip(mask.astype(np.float32), 0.0, 1.0)
        gate = None
        if gate_f is not None and float(np.max(gate_f)) > 0.05:
            gate = np.clip(gate_f.astype(np.float32), 0.0, 1.0)
            if gate.shape[:2] != (h, w):
                gate = cv2.resize(gate, (w, h), interpolation=cv2.INTER_LINEAR)
            gate = np.clip(gate, 0.0, 1.0)

        if gate is not None:
            # Phone-silhouette rim (see _gate_wrap_envelope). Mesh-coverage rim
            # leaves the visible phone edge in the interior band.
            _, interior, rim, _local_rim = Compositor._rim_band_maps(
                phone_bin, (h, w), corner_w
            )
        else:
            cov = coverage if coverage is not None else mask
            _, interior, rim, _local_rim = Compositor._outer_rim_band_maps(
                cov, (h, w), corner_w
            )
        guard = Compositor._cutout_guard(hole_w, (h, w))
        if guard is not None:
            interior = interior & ~guard
            rim = rim & ~guard

        if touch_mask:
            if gate is not None:
                fringe_px = max(2.5, float(min(h, w)) * 0.006)
                gate_soft = (gate > 0.02) & (gate < 0.985)
                if corner_w is not None and corner_w.shape[:2] == (h, w):
                    cw = np.clip(corner_w.astype(np.float32), 0.0, 1.0)
                    edge = rim | gate_soft | (cw > 0.18)
                else:
                    edge = rim | gate_soft
                m = np.where(interior & ~edge, np.maximum(m, 0.985), m)
                m = np.where(edge, gate, m)
                if corner_w is not None and corner_w.shape[:2] == (h, w):
                    cw = np.clip(corner_w.astype(np.float32), 0.0, 1.0)
                    m = np.where(cw > 0.18, np.minimum(m, gate), m)
                m = np.where((gate < 0.015) & (dist_out > fringe_px), 0.0, m)
            else:
                m = np.where(interior, np.maximum(m, 0.985), m)
                ext = np.clip(
                    1.0 - np.maximum(dist_out - 0.5, 0.0) / 1.5, 0.0, 1.0
                )
                m = m * ext

        a_out = alpha
        if alpha is not None and touch_alpha:
            a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
            if gate is not None:
                fringe_px = max(2.5, float(min(h, w)) * 0.006)
                gate_soft = (gate > 0.02) & (gate < 0.985)
                if corner_w is not None and corner_w.shape[:2] == (h, w):
                    cw = np.clip(corner_w.astype(np.float32), 0.0, 1.0)
                    edge = rim | gate_soft | (cw > 0.18)
                else:
                    edge = rim | gate_soft
                a = np.where(interior & ~edge, np.maximum(a, 0.97), a)
                a = np.where(edge, gate, a)
                if corner_w is not None and corner_w.shape[:2] == (h, w):
                    cw = np.clip(corner_w.astype(np.float32), 0.0, 1.0)
                    a = np.where(cw > 0.18, np.minimum(a, gate), a)
                a = np.where((gate < 0.015) & (dist_out > fringe_px), 0.0, a)
            else:
                ext = np.clip(
                    1.0 - np.maximum(dist_out - 0.5, 0.0) / 1.5, 0.0, 1.0
                )
                safe = interior if guard is None else (interior & ~guard)
                a = np.where(safe, np.maximum(a, 0.97), a)
                a = a * ext
            if hole_w is not None:
                hw = hole_w
                if hw.shape[:2] != (h, w):
                    hw = cv2.resize(hw, (w, h), interpolation=cv2.INTER_LINEAR)
                a = a * (1.0 - np.clip(hw, 0.0, 1.0))
            a_out = np.clip(a, 0.0, 1.0)

        if hole_w is not None and touch_mask:
            hw = hole_w
            if hw.shape[:2] != (h, w):
                hw = cv2.resize(hw, (w, h), interpolation=cv2.INTER_LINEAR)
            m = m * (1.0 - np.clip(hw, 0.0, 1.0))

        return np.clip(m, 0.0, 1.0), a_out

    def _resolve_phone_boundary_mask(
        self,
        mesh: ControlMesh,
        shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Sealed phone silhouette for clipping wrap to the device.

        Prefers a cached mask; re-detects from the photo only when missing or
        when the cache clearly matches an oversized drifted cage.
        """
        h, w = map(int, shape)
        quad = mesh.corner_points()
        corner = float(
            max(
                10.0,
                float(
                    self.settings.get(
                        "corner_radius", self.corner_radius_estimate or 11.0
                    )
                    or 11.0
                ),
            )
        )
        mesh_prior = (
            create_mesh_mask(
                mesh,
                (h, w),
                feather_radius=0,
                corner_radius_percent=corner,
                smooth_boundary=True,
                corner_radii=(
                    self.corner_radii.as_tuple()
                    if getattr(self, "corner_radii", None) is not None
                    else None
                ),
                prefer_live_boundary=False,
                phone_silhouette=None,
            )
            * 255.0
        ).astype(np.uint8)
        mesh_a = float(np.count_nonzero(mesh_prior > 40))

        pm = getattr(self.cover_engine, "last_phone_mask", None)
        need_detect = pm is None or np.count_nonzero(pm) < 64
        if not need_detect and pm is not None and mesh_a > 64:
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
                pm = (pm > 127).astype(np.uint8) * 255
            phone_a = float(np.count_nonzero(pm > 127))
            # Cached mask looks like the oversized cage → redetect from photo.
            if phone_a > mesh_a * 0.92 or mesh_a > phone_a * 1.2:
                need_detect = True
            else:
                overlap = float(
                    np.count_nonzero((mesh_prior > 40) & (pm > 127))
                )
                if overlap / max(mesh_a, 1.0) < 0.35:
                    need_detect = True

        if need_detect:
            pm = CoverSurfaceEngine.detect_phone_body_mask(
                self.phone_image, cover_quad=quad
            )

        if pm is None or np.count_nonzero(pm) < 64:
            if np.count_nonzero(mesh_prior) > 64:
                self.cover_engine.last_phone_mask = mesh_prior.copy()
                return mesh_prior
            return None

        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
            pm = (pm > 127).astype(np.uint8) * 255

        # Never union with an oversized mesh_prior — that re-inflated the
        # phone mask into empty white card and left the device "elsewhere".
        phone_a = float(np.count_nonzero(pm > 127))
        if mesh_a > 64 and phone_a > 64 and mesh_a <= phone_a * 1.12:
            try:
                completed = CoverSurfaceEngine.complete_phone_silhouette(
                    pm, mesh_prior, phone_bgr=self.phone_image
                )
                if completed is not None and np.count_nonzero(completed) > 64:
                    pm = completed
            except Exception:
                pass

        sealed = CoverSurfaceEngine.seal_phone_body(
            pm, phone_bgr=self.phone_image
        )
        if sealed is not None and np.count_nonzero(sealed) > 64:
            pm = sealed

        self.cover_engine.last_phone_mask = pm.copy()
        return pm

    def _build_perfect_wrap_mask(
        self,
        mesh: ControlMesh,
        shape: Tuple[int, int],
        *,
        feather_radius: int = 0,
        phone_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Perfect Wrap mask from the PHOTO phone silhouette.

        Mesh supplies UV topology; coverage follows the detected body mask
        (true rounded corners), not a generic AABB rounded-rect.
        """
        h, w = map(int, shape)
        corner = float(
            np.clip(
                float(
                    self.settings.get(
                        "corner_radius", self.corner_radius_estimate or 11.0
                    )
                    or 11.0
                ),
                6.0,
                18.0,
            )
        )
        if getattr(self, "corner_radii", None) is not None:
            med = float(
                np.median(np.asarray(self.corner_radii.as_tuple(), dtype=np.float64))
            )
            corner = float(np.clip(max(corner, med), 6.0, 18.0))
        radii = (corner, corner, corner, corner)
        self.corner_radii = CornerRadii.uniform(corner)
        self.corner_radius_estimate = corner
        self.settings["corner_radius"] = corner

        if phone_mask is None or np.count_nonzero(phone_mask) < 64:
            phone_mask = self._resolve_phone_boundary_mask(mesh, (h, w))

        # Primary coverage = silhouette contour SDF (photo truth).
        gate_f = None
        mask = None
        if phone_mask is not None and np.count_nonzero(phone_mask) > 64:
            pm = phone_mask
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
                pm = (pm > 127).astype(np.uint8) * 255
            phone_mask = pm
            gate_f = self._product_rim_gate(mesh, pm, (h, w))
            if gate_f is not None:
                gate_f = np.clip(gate_f.astype(np.float32), 0.0, 1.0)
                mask = gate_f.copy()

        if mask is None:
            # Fallback: mesh fill clipped later if a silhouette appears.
            mask = create_mesh_mask(
                mesh,
                (h, w),
                feather_radius=max(0, int(feather_radius)),
                corner_radius_percent=corner,
                smooth_boundary=True,
                phone_silhouette=phone_mask,
                corner_radii=radii,
                prefer_live_boundary=True,
            )
            mask = np.clip(mask, 0.0, 1.0)

        # Opaque only deep interior — never crush the soft AA rim into stairs.
        if float(np.max(mask)) > 0.05:
            solid = (mask >= 0.5).astype(np.uint8)
            dist_in = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
            deep = dist_in > 2.0
            mask = np.where(deep, np.maximum(mask, 0.97), mask)
        return (
            np.clip(mask, 0.0, 1.0).astype(np.float32),
            phone_mask,
            gate_f,
        )

    def _product_rim_gate(
        self,
        mesh: ControlMesh,
        phone_mask: np.ndarray,
        shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Float outer gate from the photo body silhouette.

        Coverage follows the real phone contour (Chaikin + supersample AA).
        Mid-sides stay opaque so wrap cannot grain. Corners keep float
        coverage so each source corner stays its own shape. No generic
        rounded-rect overlay. Button pixels are not in this mask.
        """
        from .cover_surface import CoverSurfaceEngine
        from .mesh import (
            AdaptiveMeshBuilder,
            _corner_proximity_map,
            _fill_closed_polyline_aa,
            _sharp_quad_from_mesh,
        )

        h, w = map(int, shape)
        if phone_mask is None or np.count_nonzero(phone_mask) < 64:
            return None
        gate = phone_mask
        if gate.shape[:2] != (h, w):
            gate = cv2.resize(gate, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (gate > 127).astype(np.uint8) * 255
        on = binary > 0
        if int(np.count_nonzero(on)) < 64:
            return None

        # Body mask is already the photo silhouette. Displacement-capped
        # smoothing eases 1px kinks at the straight→arc join without growing
        # the corner radius. Then supersample-fill for sub-pixel AA.
        pts = AdaptiveMeshBuilder.outer_contour_polyline(binary, smooth=False)
        cov = None
        if pts is not None and pts.shape[0] >= 16:
            raw_pts = pts.astype(np.float32)
            sm = AdaptiveMeshBuilder._smooth_closed_polyline(raw_pts, window=5)
            delta = sm - raw_pts
            mag = np.linalg.norm(delta, axis=1, keepdims=True)
            cap = 0.40
            pts = raw_pts + delta * np.minimum(
                1.0, cap / np.maximum(mag, 1e-6)
            )
            cov = _fill_closed_polyline_aa(
                pts, (h, w), scale=16, expand_px=0.0
            )
        if cov is None or float(np.max(cov)) < 0.05:
            cov = CoverSurfaceEngine.symmetric_rim_gate(
                binary,
                _sharp_quad_from_mesh(mesh),
                float(
                    np.clip(
                        float(self.corner_radius_estimate or 11.0),
                        6.0,
                        18.0,
                    )
                ),
                corner_radii=self.corner_radii,
                silhouette_mask=binary,
            )
        if cov is None or float(np.max(cov)) < 0.05:
            cov = self._exact_coverage_aa(binary, scale=8)

        ys, xs = np.where(on)
        cw = _corner_proximity_map(
            (h, w),
            x0=float(xs.min()),
            y0=float(ys.min()),
            x1=float(xs.max()),
            y1=float(ys.max()),
            corner_frac=0.22,
        )
        dist_in = cv2.distanceTransform(on.astype(np.uint8), cv2.DIST_L2, 5)
        dist_out = cv2.distanceTransform(
            (1 - on.astype(np.uint8)), cv2.DIST_L2, 5
        ).astype(np.float32)
        mid = cw < 0.16
        # Deep interior stays opaque (Chaikin must not punch white holes).
        cov = np.where(on & (dist_in >= 8.0), 1.0, cov)
        # Straight walls: 1px-AA coverage of the same mid-side wall used for
        # the body. Isolated nubs past that wall are dropped — width unchanged.
        edge_l = np.full(h, np.nan, dtype=np.float32)
        edge_r = np.full(h, np.nan, dtype=np.float32)
        for y in range(h):
            row = np.where(on[y])[0]
            if len(row):
                edge_l[y] = float(row.min())
                edge_r[y] = float(row.max())
        el = self._fill_1d_edge_profile(edge_l)
        er = self._fill_1d_edge_profile(edge_r)
        y0i, y1i = int(ys.min()), int(ys.max())
        ph = float(max(y1i - y0i + 1, 1))
        wall_l = self._straight_wall_reference(
            el, y0i, y1i, ph, side="left"
        )
        wall_r = self._straight_wall_reference(
            er, y0i, y1i, ph, side="right"
        )
        xx = np.arange(w, dtype=np.float32)[None, :]
        wall_cov = np.minimum(
            np.clip(xx - (float(wall_l) - 0.5), 0.0, 1.0),
            np.clip((float(wall_r) + 0.5) - xx, 0.0, 1.0),
        )
        # True mid only (cw≈0). Do not mix the rectangle into corner arcs.
        wall_w = np.clip((0.12 - cw) / 0.08, 0.0, 1.0)
        wall_w = wall_w * wall_w * (3.0 - 2.0 * wall_w)
        cov = wall_w * wall_cov + (1.0 - wall_w) * cov
        cov = np.where((cw < 0.18) & (xx > float(wall_r) + 0.60), 0.0, cov)
        cov = np.where((cw < 0.18) & (xx < float(wall_l) - 0.60), 0.0, cov)
        cov = np.where(mid & on & (dist_in >= 1.0), 1.0, cov)
        cov = np.where(mid & (~on) & (dist_out > 0.55), 0.0, cov)
        # Far exterior speckles only — 1px corner AA band is kept.
        cov = np.where((~on) & (dist_out > 1.8), 0.0, cov)
        return np.clip(cov, 0.0, 1.0).astype(np.float32)

    def _composite(
        self, phone_bgr: np.ndarray, mesh: ControlMesh,
        exclusion_mask: Optional[np.ndarray],
        printable_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Run the full compositing pipeline for one resolution."""
        s = self.settings
        h, w = phone_bgr.shape[:2]

        phone = phone_bgr.astype(np.float32) / 255.0

        # Destination mesh is already geometry-fitted to the phone outline.
        # Do NOT ray-expand corners here — that reintroduced tilt/unequal
        # margins at render time while the edit cage stayed upright.

        # 1. Map the design through independent affine mesh triangles with
        # Phase 2 curved UV foreshortening on the bevel rim (flat back stays
        # identity). Editing one vertex still only affects adjacent cells.
        curved = CurvedUVParams(
            rim_uv=min(float(self.settings.get("rim_uv", 5.5)) / 100.0, 0.035),
            bevel_strength=min(
                float(self.settings.get("bevel_strength", 92.0)) / 100.0, 0.72
            ),
            corner_radii=self.corner_radii,
            enabled=float(self.settings.get("curved_uv", 1.0)) >= 0.5,
        )
        # Prefer live session params when margin-derived rim is fresher.
        if self.curved_uv_params is not None and curved.enabled:
            curved.rim_uv = float(
                np.clip(
                    0.65 * curved.rim_uv
                    + 0.35 * self.curved_uv_params.rim_uv,
                    0.03,
                    0.12,
                )
            )
            curved.corner_radii = self.corner_radii
        self.curved_uv_params = curved.clamped()

        source = MeshWarper.source_points(
            self.design_image.shape[:2],
            mesh.rows,
            mesh.cols,
            mesh_aspect(mesh),
            fit_mode=self.fit_mode,
            scale=float(s.get('design_scale', 100.0)) / 100.0,
            offset_x=float(s.get('offset_x', 0.0)) / 100.0,
            offset_y=float(s.get('offset_y', 0.0)) / 100.0,
            rotation=float(s.get('rotation', 0.0)),
            curved_uv=self.curved_uv_params,
        )

        warped = MeshWarper.warp(
            self.design_image,
            source,
            mesh,
            (h, w),
            mirror=self.mirror,
        )

        if warped is None:
            return phone_bgr.copy()

        design = warped[:, :, :3].astype(np.float32) / 255.0
        design_alpha = warped[:, :, 3].astype(np.float32) / 255.0

        # 2. Perfect Wrap: rounded product rim, clipped to phone boundary ONLY.
        feather_px = int(
            round(self._scaled_pixels(s.get('edge_softness', 0), w, h) * 0.05)
        )
        mask, phone_mask, gate_f = self._build_perfect_wrap_mask(
            mesh,
            (h, w),
            feather_radius=max(0, feather_px),
            phone_mask=(
                self._scaled_mask(self._phone_wrap_mask, (h, w))
                if self._phone_wrap_mask is not None
                else getattr(self.cover_engine, "last_phone_mask", None)
            ),
        )
        from .mesh import _corner_proximity_map, _sharp_quad_from_mesh

        try:
            # Corner weights from the phone footprint — mesh cage AABB is larger
            # and dilutes corner_w on the real bottom corners. Ignore studio
            # overflow in the mask so bottom corners match top-right weighting.
            if phone_mask is not None and np.count_nonzero(phone_mask) > 64:
                pm_cw = phone_mask
                if pm_cw.shape[:2] != (h, w):
                    pm_cw = cv2.resize(
                        pm_cw, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                pm_cw = self._strip_studio_overflow_mask(pm_cw, phone_bgr)
                ys_cw, xs_cw = np.where(pm_cw > 127)
                if ys_cw.size >= 16:
                    x0_cw = float(xs_cw.min())
                    y0_cw = float(ys_cw.min())
                    x1_cw = float(xs_cw.max())
                    y1_cw = float(ys_cw.max())
                else:
                    quad = _sharp_quad_from_mesh(mesh)
                    x0_cw = float(quad[:, 0].min())
                    y0_cw = float(quad[:, 1].min())
                    x1_cw = float(quad[:, 0].max())
                    y1_cw = float(quad[:, 1].max())
            else:
                quad = _sharp_quad_from_mesh(mesh)
                x0_cw = float(quad[:, 0].min())
                y0_cw = float(quad[:, 1].min())
                x1_cw = float(quad[:, 0].max())
                y1_cw = float(quad[:, 1].max())
            corner_w = _corner_proximity_map(
                (h, w),
                x0=x0_cw,
                y0=y0_cw,
                x1=x1_cw,
                y1=y1_cw,
                corner_frac=max(
                    0.18,
                    float(s.get("corner_radius", 8.0)) / 100.0 * 1.7,
                ),
            )
        except Exception:
            corner_w = np.zeros((h, w), dtype=np.float32)

        # Rim-only clamp: interior stays full; geometric gate smooths edge band.
        interior_core = None
        if phone_mask is not None and np.count_nonzero(phone_mask) > 64:
            pm = phone_mask
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
            mask, _ = self._apply_rim_antialias(
                mask,
                None,
                pm,
                gate_f,
                (h, w),
                touch_mask=True,
                touch_alpha=False,
                corner_w=corner_w,
            )
            phone_bin_fc = (pm > 127).astype(np.uint8)
            dist_fc = cv2.distanceTransform(phone_bin_fc, cv2.DIST_L2, 5)
            interior_core = dist_fc > max(3.0, float(min(h, w)) * 0.012)

        # Close thin transparent fringes at the rim so the cover reaches the
        # phone edge (large camera holes in the artwork stay open).
        if float(np.max(design_alpha)) > 0.05 and float(np.max(mask)) > 0.05:
            solid = (mask > 0.88).astype(np.uint8) * 255
            opaque = (design_alpha > 0.35).astype(np.uint8) * 255
            edge_px = max(3, int(round(min(h, w) * 0.010)))
            grown = cv2.dilate(
                opaque,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (edge_px * 2 + 1, edge_px * 2 + 1)
                ),
                iterations=1,
            )
            grown = cv2.bitwise_and(grown, solid)
            design_alpha = np.maximum(
                design_alpha, grown.astype(np.float32) / 255.0 * 0.98
            )
            # If wrap mask is solid but warp missed the rim, seed alpha from mask.
            seed = (solid > 0) & (opaque > 0)
            if np.any(seed):
                design_alpha = np.where(
                    (solid > 0) & (design_alpha < 0.90),
                    np.maximum(design_alpha, mask * 0.97),
                    design_alpha,
                )
            # Extend RGB into the fringe from nearby opaque design pixels.
            miss = (grown > 0) & (opaque == 0)
            if interior_core is not None:
                miss = miss & interior_core
            if np.any(miss):
                for c in range(3):
                    ch = design[:, :, c]
                    blur = cv2.GaussianBlur(ch, (0, 0), sigmaX=1.4)
                    ch = np.where(miss, blur, ch)
                    design[:, :, c] = ch
            miss2 = (solid > 0) & (opaque == 0) & (design_alpha > 0.5)
            if interior_core is not None:
                miss2 = miss2 & interior_core
            if np.any(miss2):
                for c in range(3):
                    ch = design[:, :, c]
                    blur = cv2.GaussianBlur(ch, (0, 0), sigmaX=1.6)
                    ch = np.where(miss2, blur, ch)
                    design[:, :, c] = ch

        # Hardware cutouts: full user exclusion (nothing inside the cutout).
        # Camera bump is a raised RIDGE on the border only — never fills inside.
        bump_module = None
        wrap_mask = np.clip(mask, 0.0, 1.0).copy()
        excl_f = None
        hole_w = None
        # Side-button coverage in THIS composite space (native detect → resize).
        btn_cov = self._build_side_button_wrap_coverage(
            (h, w), phone_mask, exclusion_mask
        )
        self._side_button_wrap_cov = btn_cov
        tip_vm = self._side_button_validated_mask
        if exclusion_mask is not None:
            excl = exclusion_mask
            if excl.shape[:2] != (h, w):
                excl = cv2.resize(excl, (w, h), interpolation=cv2.INTER_LINEAR)
            excl_f = np.clip(excl.astype(np.float32) / 255.0, 0.0, 1.0)
            bump_module, _, _ = self._camera_bump_exclusion_maps(excl, phone)
            excl_for_holes = excl_f
            if btn_cov is not None and float(np.max(btn_cov)) > 0.05:
                # Release only bezel button pixels from punch — never camera.
                cam = self._side_button_camera_block(
                    (h, w), exclusion_mask, phone_mask
                ) > 127
                release = np.clip(btn_cov, 0.0, 1.0) * (
                    (~cam).astype(np.float32)
                )
                excl_for_holes = excl_f * (1.0 - release)
            hole_w = self._hard_hole_weight(excl_for_holes)
            if btn_cov is not None and float(np.max(btn_cov)) > 0.05:
                # Camera hole AA rims must not shave the bezel button masks.
                hole_w = hole_w * (1.0 - np.clip(btn_cov, 0.0, 1.0))
            mask = mask * (1.0 - hole_w)

        material = self._resolve_material(s)
        opacity = float(s.get('opacity', 100.0)) / 100.0
        # Opaque covers fully hide phone-body graphics (MagSafe, logo) under
        # the print. Transparent materials keep natural show-through.
        if material.opacity >= 0.90 and opacity >= 0.90:
            content = np.clip(design_alpha, 0.0, 1.0)
            solid = np.clip(mask, 0.0, 1.0)
            solid = np.where(solid > 0.90, np.maximum(solid, 0.96), solid)
            boosted = np.clip(content * 1.12, 0.0, 1.0)
            # Opaque wrap: wherever the phone face is covered by wrap mask and
            # the design has any ink nearby, keep the cover opaque to the rim.
            alpha = np.clip(solid * boosted * opacity, 0.0, 1.0)
            alpha = np.where(
                (solid > 0.90) & (boosted > 0.12),
                np.maximum(alpha, solid * 0.96),
                alpha,
            )
        else:
            alpha = design_alpha * mask * opacity

        # Localized side-button wrap (bezel masks only; camera untouched).
        design, mask, alpha = self._apply_side_button_wrap(
            design,
            mask,
            alpha,
            design_alpha,
            phone_mask,
            exclusion_mask,
            opacity=opacity,
        )
        # Do NOT merge button AA into wrap_mask — that leaked soft tip cov into
        # the body stabilize path and speckled the left outer edge.

        # 3. Colour and detail adjustments on the design only.
        design = ImageFilters.apply_adjustments(design, s)

        # Kill affine warp streaks along cutout arcs and outer rounded corners
        # before material grain / gloss amplifies them into visible tearing.
        if excl_f is not None or float(np.max(wrap_mask)) > 0.05:
            design = MaterialRenderingEngine.stabilize_wrap_texture(
                design, wrap_mask, excl_f
            )

        tone_match = float(s.get('tone_match', 0.0)) / 100.0
        if tone_match > 0:
            # Cap tone-match so it cannot reintroduce MagSafe/logo as colour.
            design = ImageFilters.auto_match_tone(
                design, phone, mask, min(tone_match, 0.35)
            )

        # 4. Material Rendering Engine — surface texture, reflections,
        # highlights, body shadows, edge rim, and contact shadow map.
        lighting = LIGHTING.get(self.lighting_name)
        design, contact = self.material_engine.apply(
            design, phone, mask,
            material=material,
            lighting=lighting,
            settings=s,
            exclusion=excl_f,
        )

        # 4b. Camera bump ridge DISABLED — the raised lip looked blocky /
        # "inner messed up" on dark wraps. Keep a clean hard cutout only.
        if False and bump_module is not None:
            design, mask = MaterialRenderingEngine.apply_camera_bump(
                design,
                mask,
                phone,
                bump_module,
                np.zeros_like(mask),
                wrap_mask=wrap_mask,
                lighting=lighting,
            )
            if material.opacity >= 0.90 and opacity >= 0.90:
                content = np.clip(design_alpha, 0.0, 1.0)
                solid = np.clip(mask, 0.0, 1.0)
                # Soft AA tip on curves — avoid binary crush at the rim.
                solid = np.where(solid > 0.94, np.maximum(solid, 0.955), solid)
                boosted = np.clip(content * 1.08, 0.0, 1.0)
                alpha = np.clip(solid * boosted * opacity, 0.0, 1.0)
                alpha = np.where(
                    (solid > 0.94) & (boosted > 0.25),
                    np.maximum(alpha, 0.94),
                    alpha,
                )
            else:
                alpha = design_alpha * mask * opacity

        # Seal wrap right up to the cutout — kill light-leak fringe between
        # cover and camera island (inner polish). Never paints inside the hole.
        if hole_w is not None and float(np.max(hole_w)) > 0.05:
            hw = hole_w
            if hw.shape[:2] != (h, w):
                hw = cv2.resize(hw, (w, h), interpolation=cv2.INTER_LINEAR)
            hw = np.clip(hw.astype(np.float32), 0.0, 1.0)
            hole_bin = (hw > 0.45).astype(np.uint8)
            if np.count_nonzero(hole_bin) >= 16:
                dist_out = cv2.distanceTransform(
                    (1 - hole_bin).astype(np.uint8), cv2.DIST_L2, 5
                ).astype(np.float32)
                # 1–2px wrap band just outside the hole: full opacity.
                seal = (hw < 0.20) & (dist_out <= 2.0) & (dist_out > 0.0)
                if tip_vm is not None and np.count_nonzero(tip_vm) >= 4:
                    tclip = tip_vm > 127
                    if tclip.shape[:2] != (h, w):
                        tclip = (
                            cv2.resize(
                                tip_vm.astype(np.uint8),
                                (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            > 127
                        )
                    seal = seal & ~tclip
                if np.any(seal):
                    alpha = np.where(seal, np.maximum(alpha, 0.97), alpha)
                    mask = np.where(seal, np.maximum(mask, 0.97), mask)
                # Inside hole stays punched.
                alpha = alpha * (1.0 - hw)
                mask = mask * (1.0 - hw)

        # 4c. Side-button relief disabled — avoids synthetic 3D bumps on keys.
        # (Re-enable only when validated photo-edge masks are pixel-perfect.)

        grain = float(s.get('grain', 0.0)) / 100.0
        if grain > 0:
            design = add_grain(design, grain, mask=mask)

        # Soften AA only in corner pockets — any side mix blurs red onto the
        # white studio card (halka red fringe).
        alpha = np.clip(alpha, 0.0, 1.0)
        cutout_guard = self._cutout_guard(hole_w, (h, w))
        alpha_soft = cv2.GaussianBlur(alpha, (0, 0), 0.55)
        rim_band = (alpha > 0.01) & (alpha < 0.92) & (corner_w > 0.28)
        if cutout_guard is not None:
            rim_band = rim_band & ~cutout_guard
        mix = (0.28 * corner_w).astype(np.float32)
        alpha = np.where(
            rim_band,
            (1.0 - mix) * alpha + mix * alpha_soft,
            alpha,
        )
        # Re-assert solid coverage in corner cores — never on cutout curves.
        core_boost = (
            (mask > 0.92) & (corner_w > 0.35) & (alpha > 0.35)
        )
        if cutout_guard is not None:
            core_boost = core_boost & ~cutout_guard
        alpha = np.where(
            core_boost,
            np.maximum(alpha, mask * 0.96),
            alpha,
        )
        alpha = np.clip(alpha, 0.0, 1.0)

        # Drop studio-card speckles only — interior wrap stays full.
        # Keep validated side-button tips (they sit past the body wall).
        alpha = self._trim_exterior_speckles(
            alpha,
            phone_mask,
            tip_mask=tip_vm,
        )

        # Cutout openings stay punched through materials — do not re-fill here.
        # Final outer-rim AA runs later; holes are re-applied after that pass.

        # Soft contact shadow on the phone beneath the cover perimeter.
        # Never darken hardware cutouts — those must match the original phone.
        phone_blend = phone
        if contact is not None and float(np.max(contact)) > 1e-4:
            safe_contact = contact
            excl_for_shadow = exclusion_mask
            if excl_for_shadow is not None:
                excl = excl_for_shadow
                if excl.shape[:2] != (h, w):
                    excl = cv2.resize(
                        excl, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                excluded = np.clip(excl.astype(np.float32) / 255.0, 0.0, 1.0)
                safe_contact = contact * (1.0 - np.clip(excluded * 2.0, 0.0, 1.0))
            # Kill the charcoal halo on white / studio plates outside the phone.
            phone_lum = phone.mean(axis=2)
            plate_gate = np.clip((0.90 - phone_lum) / 0.28, 0.0, 1.0)
            safe_contact = safe_contact * plate_gate
            phone_blend = np.clip(
                phone * (1.0 - safe_contact[:, :, np.newaxis] * 0.38), 0.0, 1.0
            )

        # Smooth geometric envelope — never reintroduce binary photo stairs.
        rim = np.clip(mask.astype(np.float32), 0.0, 1.0)
        if (
            phone_mask is not None
            and np.count_nonzero(phone_mask) > 64
            and float(np.count_nonzero(rim > 0.28)) > 64
        ):
            pm = phone_mask
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
            rim = self._gate_wrap_envelope(
                rim,
                pm,
                gate_f,
                (h, w),
                corner_w=corner_w,
                coverage=mask,
                hole_w=hole_w,
            )
            rim, alpha = self._apply_rim_antialias(
                rim,
                alpha,
                pm,
                gate_f,
                (h, w),
                touch_mask=True,
                touch_alpha=True,
                corner_w=corner_w,
                coverage=mask,
                hole_w=hole_w,
            )
        alpha = np.minimum(alpha, rim)
        mask = np.minimum(mask, rim)
        # Visible outer edge = geometric float gate everywhere.
        # Binary body / mesh stairs must not paint past the product rim.
        # Button tips are restored AFTER rim_fix (below) so gate/soft AA
        # cannot wipe the bridged wrap beside real protrusions.
        btn_cov = self._side_button_wrap_cov
        bf_guard = None
        if btn_cov is not None and float(np.max(btn_cov)) > 0.05:
            bf_guard = np.clip(btn_cov.astype(np.float32), 0.0, 1.0)
            if bf_guard.shape[:2] != (h, w):
                bf_guard = cv2.resize(
                    bf_guard, (w, h), interpolation=cv2.INTER_LINEAR
                )
            # Never let soft coverage live outside the validated tip mask.
            if tip_vm is not None and np.count_nonzero(tip_vm) >= 4:
                tclip = tip_vm > 127
                if tclip.shape[:2] != (h, w):
                    tclip = (
                        cv2.resize(
                            tip_vm.astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        > 127
                    )
                bf_guard = np.where(tclip, bf_guard, 0.0)
        if gate_f is not None and float(np.max(gate_f)) > 0.05:
            g = np.clip(gate_f.astype(np.float32), 0.0, 1.0)
            if g.shape[:2] != (h, w):
                g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
            # Body gate only — do NOT merge button soft cov (that speckles the
            # left wall). True outward tips stay reserved for the tip layer.
            alpha = np.minimum(alpha, g)
            mask = np.minimum(mask, g)
            if tip_vm is not None and np.count_nonzero(tip_vm) >= 4:
                vmask = tip_vm > 127
                if vmask.shape[:2] != (h, w):
                    vmask = (
                        cv2.resize(
                            tip_vm.astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        > 127
                    )
                body_on = np.zeros((h, w), dtype=bool)
                if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
                    pm_t = phone_mask
                    if pm_t.shape[:2] != (h, w):
                        pm_t = cv2.resize(
                            pm_t, (w, h), interpolation=cv2.INTER_NEAREST
                        )
                    body_on = pm_t > 127
                protr = vmask & ~body_on
                if np.any(protr):
                    alpha = np.where(protr, 0.0, alpha)
                    mask = np.where(protr, 0.0, mask)
        if hole_w is not None:
            hw = hole_w
            if hw.shape[:2] != (h, w):
                hw = cv2.resize(hw, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = alpha * (1.0 - np.clip(hw, 0.0, 1.0))
            mask = mask * (1.0 - np.clip(hw, 0.0, 1.0))
        alpha = self._kill_studio_print_fringe(alpha, phone, phone_mask=None)

        # Outer perimeter AA: studio-fringe cleanup and short design warps can
        # leave soft gate pixels with alpha=0 (hard white|ink step). Re-apply
        # float gate coverage and pull wrap RGB into that fringe only.
        if gate_f is not None and float(np.max(gate_f)) > 0.05:
            g_edge = np.clip(gate_f.astype(np.float32), 0.0, 1.0)
            if g_edge.shape[:2] != (h, w):
                g_edge = cv2.resize(
                    g_edge, (w, h), interpolation=cv2.INTER_LINEAR
                )
            soft_edge = (g_edge > 0.05) & (g_edge < 0.95)
            # Corner pockets: rounded gate covers them but wrap alpha stopped on
            # photo stairs (gate≈1, alpha≈0). Fill those gaps so the float AA
            # fringe is attached to real wrap ink.
            corner_gap = np.zeros((h, w), dtype=bool)
            if corner_w is not None and corner_w.shape[:2] == (h, w):
                corner_gap = (
                    (corner_w > 0.32)
                    & (g_edge > 0.35)
                    & (alpha < 0.35)
                )
            if hole_w is not None:
                hw = hole_w
                if hw.shape[:2] != (h, w):
                    hw = cv2.resize(
                        hw, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                # Keep outer soft AA far from cutouts — hole fringe was
                # creating the light "inner" leak beside the camera island.
                hw_ok = np.clip(hw, 0.0, 1.0) < 0.18
                soft_edge = soft_edge & hw_ok
                corner_gap = corner_gap & hw_ok
            cut_guard = self._cutout_guard(hole_w, (h, w), margin_px=6.0)
            if cut_guard is not None:
                soft_edge = soft_edge & ~cut_guard
                corner_gap = corner_gap & ~cut_guard
            # Limit to the outer rim so interiors stay untouched.
            if phone_mask is not None and np.count_nonzero(phone_mask) > 64:
                pm_e = phone_mask
                if pm_e.shape[:2] != (h, w):
                    pm_e = cv2.resize(
                        pm_e, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                dist_in = cv2.distanceTransform(
                    (pm_e > 127).astype(np.uint8), cv2.DIST_L2, 5
                ).astype(np.float32)
                dist_out = cv2.distanceTransform(
                    (1 - (pm_e > 127).astype(np.uint8)), cv2.DIST_L2, 5
                ).astype(np.float32)
                # Mid-sides: only the true outer half-pixel may be soft —
                # deeper body columns washed by float gate create the vertical
                # blur/line artifact along L/R walls.
                mid_band = np.ones((h, w), dtype=bool)
                if corner_w is not None and corner_w.shape[:2] == (h, w):
                    mid_band = np.clip(corner_w, 0.0, 1.0) < 0.16
                soft_edge = soft_edge & (
                    (~mid_band & (dist_in < 4.0))
                    | (mid_band & (dist_in < 1.0))
                )
                # Allow at most ~1.5 px outside the photo rim for AA.
                corner_gap = corner_gap & (dist_out <= 1.5) & (dist_in < 6.0)
                # Mid-side body must stay fully opaque wrap (no soft alpha wash).
                solid_mid = (
                    mid_band & (pm_e > 127) & (dist_in >= 1.0)
                )
                if tip_vm is not None and np.count_nonzero(tip_vm) >= 4:
                    tclip = tip_vm > 127
                    if tclip.shape[:2] != (h, w):
                        tclip = (
                            cv2.resize(
                                tip_vm.astype(np.uint8),
                                (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            > 127
                        )
                    solid_mid = solid_mid & ~tclip
                # Exclude solid mid-body from soft rim rewrite.
                if np.any(solid_mid):
                    soft_edge = soft_edge & ~solid_mid
                    corner_gap = corner_gap & ~solid_mid
            rim_fix = soft_edge | corner_gap
            if bf_guard is not None:
                rim_fix = rim_fix & (bf_guard < 0.25)
            if np.any(rim_fix):
                # Replace binary stairs with float gate coverage on the rim.
                alpha = np.where(rim_fix, g_edge, alpha)
                mask = np.where(rim_fix, np.minimum(mask, g_edge), mask)
                ink = alpha > 0.85
                if np.any(ink):
                    for c in range(3):
                        ch = design[:, :, c]
                        seeded = np.where(ink, ch, 0.0).astype(np.float32)
                        blur = cv2.GaussianBlur(seeded, (0, 0), sigmaX=0.85)
                        weight = cv2.GaussianBlur(
                            ink.astype(np.float32), (0, 0), sigmaX=0.85
                        )
                        fill = blur / np.maximum(weight, 1e-4)
                        need = rim_fix & (alpha < 0.85)
                        design[:, :, c] = np.where(need, fill, ch)
            # Re-assert opaque mid-side body AFTER rim soft rewrite.
            if phone_mask is not None and np.count_nonzero(phone_mask) > 64:
                pm_e = phone_mask
                if pm_e.shape[:2] != (h, w):
                    pm_e = cv2.resize(
                        pm_e, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                dist_in = cv2.distanceTransform(
                    (pm_e > 127).astype(np.uint8), cv2.DIST_L2, 5
                ).astype(np.float32)
                mid_band = np.ones((h, w), dtype=bool)
                if corner_w is not None and corner_w.shape[:2] == (h, w):
                    mid_band = np.clip(corner_w, 0.0, 1.0) < 0.16
                solid_mid = mid_band & (pm_e > 127) & (dist_in >= 1.0)
                if tip_vm is not None and np.count_nonzero(tip_vm) >= 4:
                    tclip = tip_vm > 127
                    if tclip.shape[:2] != (h, w):
                        tclip = (
                            cv2.resize(
                                tip_vm.astype(np.uint8),
                                (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            > 127
                        )
                    solid_mid = solid_mid & ~tclip
                if np.any(solid_mid):
                    alpha = np.where(solid_mid, np.maximum(alpha, 0.97), alpha)
                    mask = np.where(solid_mid, np.maximum(mask, 0.97), mask)

        # True outward tips stay reserved — body rim AA must not own them.
        paint_src = tip_vm
        if paint_src is not None and np.count_nonzero(paint_src) >= 4:
            vmask = paint_src > 127
            if vmask.shape[:2] != (h, w):
                vmask = (
                    cv2.resize(
                        paint_src.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 127
                )
            body_on = np.zeros((h, w), dtype=bool)
            if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
                pm_t = phone_mask
                if pm_t.shape[:2] != (h, w):
                    pm_t = cv2.resize(
                        pm_t, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                body_on = pm_t > 127
            protr = vmask & ~body_on
            if np.any(protr):
                alpha = np.where(protr, 0.0, alpha)
                mask = np.where(protr, 0.0, mask)

        # Re-seal camera cutout AFTER rim soft rewrites so outer polish
        # cannot reopen a light-leak fringe beside the island.
        if hole_w is not None and float(np.max(hole_w)) > 0.05:
            hw = hole_w
            if hw.shape[:2] != (h, w):
                hw = cv2.resize(hw, (w, h), interpolation=cv2.INTER_LINEAR)
            hw = np.clip(hw.astype(np.float32), 0.0, 1.0)
            hole_bin = (hw > 0.45).astype(np.uint8)
            if np.count_nonzero(hole_bin) >= 16:
                dist_out = cv2.distanceTransform(
                    (1 - hole_bin).astype(np.uint8), cv2.DIST_L2, 5
                ).astype(np.float32)
                seal = (hw < 0.18) & (dist_out <= 2.25) & (dist_out > 0.0)
                if tip_vm is not None and np.count_nonzero(tip_vm) >= 4:
                    tclip = tip_vm > 127
                    if tclip.shape[:2] != (h, w):
                        tclip = (
                            cv2.resize(
                                tip_vm.astype(np.uint8),
                                (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            > 127
                        )
                    seal = seal & ~tclip
                if np.any(seal):
                    alpha = np.where(seal, np.maximum(alpha, 0.98), alpha)
                    mask = np.where(seal, np.maximum(mask, 0.98), mask)
            alpha = alpha * (1.0 - hw)
            mask = mask * (1.0 - hw)

        # Drop wrap painted onto pure studio white. Protect validated tips.
        tip_protect = None
        if paint_src is not None and np.count_nonzero(paint_src) >= 4:
            tip_protect = (paint_src > 127).astype(np.uint8) * 255
        alpha, mask = self._clip_studio_plate_wrap(
            alpha, mask, phone_bgr, protect=tip_protect
        )

        # 5. Blend and finish (float until the last step).
        alpha3 = alpha[:, :, np.newaxis]
        result = design * alpha3 + phone_blend * (1.0 - alpha3)

        vignette = float(s.get('vignette', 0.0)) / 100.0
        if vignette > 0:
            result = apply_vignette(result, vignette)

        output = np.clip(np.round(result * 255.0), 0, 255).astype(np.uint8)

        # Outer boundary raster AA — tip_mask protects validated buttons.
        output = self._rasterize_outer_boundary_aa(
            output,
            phone_bgr,
            mask,
            hole_w,
            gate_f=gate_f,
            corner_w=corner_w,
            button_cov=None,
            tip_mask=tip_vm,
            phone_mask=phone_mask,
        )

        # Clear wrap leaks outside body|tips. Do NOT paint wrap onto studio
        # fringe (that created the long vertical edge streaks).
        output = self._suppress_mid_side_edge_speckles(
            output,
            phone_bgr,
            phone_mask,
            tip_mask=tip_vm,
            corner_w=corner_w,
        )

        # Final rim polish: crush leftover black fringe on the soft gate band
        # without changing the geometric silhouette.
        if gate_f is not None and float(np.max(gate_f)) > 0.05:
            g_fin = np.clip(gate_f.astype(np.float32), 0.0, 1.0)
            if g_fin.shape[:2] != (h, w):
                g_fin = cv2.resize(g_fin, (w, h), interpolation=cv2.INTER_LINEAR)
            gray_f = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
            tip_bool = np.zeros((h, w), dtype=bool)
            if tip_vm is not None and np.count_nonzero(tip_vm) >= 4:
                tip_bool = tip_vm > 127
                if tip_bool.shape[:2] != (h, w):
                    tip_bool = (
                        cv2.resize(
                            tip_vm.astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        > 127
                    )
            # Soft rim band where AA should dominate — kill crushed-black chips.
            rim_band = (g_fin > 0.08) & (g_fin < 0.92) & ~tip_bool
            # Opaque outer body column + corner arc: crushed-black fringe
            # (gray≈3–6) that reads as edge chips when zoomed.
            pm_fin = phone_mask
            if pm_fin is not None and pm_fin.shape[:2] != (h, w):
                pm_fin = cv2.resize(
                    pm_fin, (w, h), interpolation=cv2.INTER_NEAREST
                )
            opaque_rim = np.zeros((h, w), dtype=bool)
            if pm_fin is not None and np.count_nonzero(pm_fin) >= 64:
                dist_in = cv2.distanceTransform(
                    (pm_fin > 127).astype(np.uint8), cv2.DIST_L2, 5
                )
                opaque_rim = (
                    (pm_fin > 127)
                    & (dist_in <= 2.25)
                    & (gray_f < 12.0)
                    & ~tip_bool
                )
            crush_mask = (rim_band & (gray_f < 12.0)) | opaque_rim
            if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
                from .mesh import _corner_proximity_map

                pm_cr = phone_mask
                if pm_cr.shape[:2] != (h, w):
                    pm_cr = cv2.resize(
                        pm_cr, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                ys_cr, xs_cr = np.where(pm_cr > 127)
                if xs_cr.size >= 16:
                    cw_f = _corner_proximity_map(
                        (h, w),
                        x0=float(xs_cr.min()),
                        y0=float(ys_cr.min()),
                        x1=float(xs_cr.max()),
                        y1=float(ys_cr.max()),
                        corner_frac=0.22,
                    )
                    crush_mask = crush_mask & (
                        np.clip(cw_f, 0.0, 1.0) < 0.30
                    )
            elif corner_w is not None and float(np.max(corner_w)) > 0.05:
                cw_f = corner_w
                if cw_f.shape[:2] != (h, w):
                    cw_f = cv2.resize(
                        cw_f.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                crush_mask = crush_mask & (np.clip(cw_f, 0.0, 1.0) < 0.30)
            # Never touch camera cutout neighborhood (inner must stay clean).
            if hole_w is not None and float(np.max(hole_w)) > 0.05:
                cut_g = self._cutout_guard(hole_w, (h, w), margin_px=6.0)
                if cut_g is not None:
                    crush_mask = crush_mask & ~cut_g
            if np.any(crush_mask):
                plate = phone_bgr.astype(np.float32)
                pl = plate.mean(axis=2)
                ps = plate.max(axis=2) - plate.min(axis=2)
                studio_px = (pl >= 245.0) & (ps <= 12.0)
                if int(np.count_nonzero(studio_px)) >= 64:
                    studio_rgb = np.median(plate[studio_px], axis=0)
                else:
                    studio_rgb = np.array(
                        [255.0, 255.0, 255.0], dtype=np.float32
                    )
                out_f = output.astype(np.float32)
                # Typical wrap ink on the body (excludes crushed rim + studio).
                body_ink = (
                    (pm_fin > 127)
                    & ~tip_bool
                    & (gray_f >= 14.0)
                    & (gray_f < 140.0)
                )
                if np.any(body_ink):
                    wrap_rgb = np.median(out_f[body_ink], axis=0)
                else:
                    wrap_rgb = studio_rgb
                ys_c, xs_c = np.where(crush_mask)
                for y, x in zip(ys_c, xs_c):
                    src_xy = None
                    # Prefer inward neighbors (horizontal then vertical) so
                    # corner arcs resample real wrap, not letterbox black.
                    for d in range(2, 16):
                        for cy, cx in (
                            (y, x + d),
                            (y, x - d),
                            (y + d, x),
                            (y - d, x),
                            (y + d, x + d),
                            (y + d, x - d),
                            (y - d, x + d),
                            (y - d, x - d),
                        ):
                            if (
                                0 <= cy < h
                                and 0 <= cx < w
                                and 14.0 <= gray_f[cy, cx] < 140.0
                                and (
                                    pm_fin is None
                                    or pm_fin[cy, cx] > 127
                                )
                            ):
                                src_xy = (cy, cx)
                                break
                        if src_xy is not None:
                            break
                    gg = float(g_fin[y, x])
                    if src_xy is None:
                        src_rgb = wrap_rgb
                    else:
                        src_rgb = out_f[src_xy[0], src_xy[1]]
                    if gg < 0.92:
                        out_f[y, x] = src_rgb * gg + studio_rgb * (1.0 - gg)
                    else:
                        out_f[y, x] = src_rgb
                output = np.clip(np.round(out_f), 0, 255).astype(np.uint8)

        # Button wrap as its own top layer (above body). Strict tip mask only.
        output = self._composite_side_button_layer(
            output,
            phone_bgr,
            tip_vm,
            phone_mask=phone_mask,
            tip_cov=bf_guard,
        )

        # Final perimeter chip kill AFTER tip layer — outside body|tips only.
        if phone_mask is not None and np.count_nonzero(phone_mask) >= 64:
            pm_c = phone_mask
            if pm_c.shape[:2] != (h, w):
                pm_c = cv2.resize(
                    pm_c, (w, h), interpolation=cv2.INTER_NEAREST
                )
            keep_c = pm_c > 127
            if tip_vm is not None and np.count_nonzero(tip_vm) >= 4:
                tm = tip_vm
                if tm.shape[:2] != (h, w):
                    tm = cv2.resize(
                        tip_vm.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                keep_c = keep_c | (tm > 127)
            keep_halo = (
                cv2.dilate(
                    keep_c.astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
                > 0
            )
            gray_c = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            plate = phone_bgr.astype(np.float32)
            pl = plate.mean(axis=2)
            ps = plate.max(axis=2) - plate.min(axis=2)
            studio_px = (pl >= 245.0) & (ps <= 12.0)
            if int(np.count_nonzero(studio_px)) >= 64:
                studio_bgr = np.clip(
                    np.round(np.median(plate[studio_px], axis=0)), 0, 255
                ).astype(np.uint8)
            else:
                studio_bgr = np.array([255, 255, 255], dtype=np.uint8)
            dist_out = cv2.distanceTransform(
                (1 - keep_c.astype(np.uint8)), cv2.DIST_L2, 5
            )
            chips = (~keep_halo) & (dist_out <= 2.5) & (gray_c < 160)
            if np.any(chips):
                output = output.copy()
                output[chips] = studio_bgr

        # Hard guarantee: opaque hole cores stay pixel-identical to the phone.
        # Threshold high so the soft SDF AA rim is not crushed into stairs.
        if exclusion_mask is not None:
            excl = exclusion_mask
            if excl.shape[:2] != (h, w):
                excl = cv2.resize(excl, (w, h), interpolation=cv2.INTER_LINEAR)
            hard_core = excl >= 238
            btn_cov = self._side_button_wrap_cov
            if btn_cov is not None and float(np.max(btn_cov)) > 0.05:
                bf = btn_cov
                if bf.shape[:2] != (h, w):
                    bf = cv2.resize(bf, (w, h), interpolation=cv2.INTER_LINEAR)
                hard_core = hard_core & (bf < 0.35)
            if np.any(hard_core):
                output[hard_core] = phone_bgr[hard_core]

        return output

    @staticmethod
    def _suppress_mid_side_edge_speckles(
        output: np.ndarray,
        phone_bgr: np.ndarray,
        phone_mask: Optional[np.ndarray],
        tip_mask: Optional[np.ndarray] = None,
        corner_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Remove jagged dark protrusions on mid-sides outside body|tips.

        Does not reshape the body silhouette or tip contours — only replaces
        stray dark pixels with the phone plate (studio/device).
        """
        if output is None or phone_bgr is None or phone_mask is None:
            return output
        if np.count_nonzero(phone_mask) < 64:
            return output
        h, w = output.shape[:2]
        pm = phone_mask
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
        keep = pm > 127
        if tip_mask is not None and np.count_nonzero(tip_mask) >= 4:
            tip = tip_mask
            if tip.shape[:2] != (h, w):
                tip = cv2.resize(
                    tip.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            keep = keep | (tip > 127)
        # Allow 1px body AA — only kill spills beyond that.
        keep_u8 = keep.astype(np.uint8) * 255
        keep_halo = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ) > 0
        from .mesh import _corner_proximity_map

        mid = np.ones((h, w), dtype=bool)
        phone_bin = (pm > 127).astype(np.uint8)
        ys_m, xs_m = np.where(phone_bin > 0)
        if xs_m.size >= 16:
            cw = _corner_proximity_map(
                (h, w),
                x0=float(xs_m.min()),
                y0=float(ys_m.min()),
                x1=float(xs_m.max()),
                y1=float(ys_m.max()),
                corner_frac=0.22,
            )
            mid = np.clip(cw, 0.0, 1.0) < 0.30
        elif corner_w is not None and float(np.max(corner_w)) > 0.05:
            cw = corner_w
            if cw.shape[:2] != (h, w):
                cw = cv2.resize(
                    cw.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
            mid = np.clip(cw, 0.0, 1.0) < 0.30
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        out = output.copy()
        dist_out = cv2.distanceTransform(
            (1 - phone_bin).astype(np.uint8), cv2.DIST_L2, 5
        )
        # Never touch pixels far from the silhouette — global row fixes painted
        # black letterbox / wrap ink across the whole canvas (horizontal bars).
        near_rim = dist_out <= 3.0
        plate = phone_bgr.astype(np.float32)
        pl = plate.mean(axis=2)
        ps = plate.max(axis=2) - plate.min(axis=2)
        studio_px = (pl >= 245.0) & (ps <= 12.0)
        if int(np.count_nonzero(studio_px)) >= 64:
            studio_rgb = np.median(plate[studio_px], axis=0)
        else:
            studio_rgb = np.array([255.0, 255.0, 255.0], dtype=np.float32)
        studio_bgr = np.clip(np.round(studio_rgb), 0, 255).astype(np.uint8)

        leak = mid & ~keep_halo & near_rim & (gray < 120)
        if np.any(leak):
            out[leak] = studio_bgr

        dist_in = cv2.distanceTransform(phone_bin, cv2.DIST_L2, 5)
        tip_bool = np.zeros((h, w), dtype=bool)
        if tip_mask is not None and np.count_nonzero(tip_mask) >= 4:
            tip_bool = tip_mask > 127
            if tip_bool.shape[:2] != (h, w):
                tip_bool = (
                    cv2.resize(
                        tip_mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 127
                )

        # Isolated right-wall nubs past the robust mid-side wall (compositing
        # only — body width / wrap geometry stay put).
        if xs_m.size >= 16:
            edge_r = np.full(h, np.nan, dtype=np.float32)
            for y in range(h):
                row = np.where(phone_bin[y] > 0)[0]
                if len(row):
                    edge_r[y] = float(row.max())
            er = Compositor._fill_1d_edge_profile(edge_r)
            y0i, y1i = int(ys_m.min()), int(ys_m.max())
            ph = float(max(y1i - y0i + 1, 1))
            wall_r = Compositor._straight_wall_reference(
                er, y0i, y1i, ph, side="right"
            )
            xx = np.arange(w, dtype=np.float32)[None, :]
            right_nub = (
                mid
                & (xx > float(wall_r) + 0.60)
                & near_rim
                & ~tip_bool
                & (gray < 220)
            )
            if np.any(right_nub):
                out[right_nub] = studio_bgr
        else:
            right_nub = np.zeros((h, w), dtype=bool)

        # Mid-side soft white fringe (gray≈240) reads as a blurry vertical line.
        # Snap near-white outside body|tips to pure studio; keep corner AA.
        soft_fringe = (
            mid
            & ~keep
            & near_rim
            & (gray >= 200)
            & (gray < 252)
            & ~tip_bool
        )
        if np.any(soft_fringe):
            out[soft_fringe] = studio_bgr

        rim_grain = (
            mid
            & (pm > 127)
            & (dist_in <= 2.5)
            & (dist_in >= 0.5)
            & (gray > 40)
            & ~tip_bool
        )
        if np.any(rim_grain):
            ys_g, xs_g = np.where(rim_grain)
            for y, x in zip(ys_g, xs_g):
                body_xs = np.where(pm[y] > 127)[0]
                if len(body_xs) == 0:
                    continue
                xl, xr = int(body_xs.min()), int(body_xs.max())
                if x <= xl + 2:
                    # Left wall — sample inward (right).
                    refs = range(xl + 2, min(w, xl + 14))
                elif x >= xr - 2:
                    # Right wall — sample inward (left).
                    refs = range(xr - 2, max(-1, xr - 14), -1)
                else:
                    continue
                ref = None
                for cand in refs:
                    gv = int(gray[y, cand])
                    if 10 <= gv < 40:
                        ref = cand
                        break
                if ref is not None:
                    out[y, x] = out[y, ref]
        # Outside body|tips: restore studio plate — never paint wrap ink onto
        # the studio fringe (that produced long vertical edge streaks).
        outside_spill = mid & ~keep & near_rim & (gray < 220)
        # Keep tip surfaces; only clear wrap/fringe leaks past the silhouette.
        if tip_bool.any():
            outside_spill = outside_spill & ~tip_bool
        spill_fixed = False
        if np.any(outside_spill):
            # Dark wrap leaks → studio; leave near-white studio AA alone.
            dark_spill = outside_spill & (gray < 80)
            if np.any(dark_spill):
                out[dark_spill] = studio_bgr
                spill_fixed = True
        # Corner bands: polish white rim grains INSIDE the body only.
        # Never rewrite the rounded-arc band — that copies wrap into the
        # studio-composited corner AA (horizontal gray bars at top-left).
        corner_sealed = False
        cw_seal = None
        if xs_m.size >= 16:
            cw_seal = cw
        elif corner_w is not None and float(np.max(corner_w)) > 0.05:
            cw_seal = corner_w
            if cw_seal.shape[:2] != (h, w):
                cw_seal = cv2.resize(
                    cw_seal.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
        if cw_seal is not None:
            corner_band = np.clip(cw_seal, 0.0, 1.0) >= 0.28
            arc_band = np.clip(cw_seal, 0.0, 1.0) > 0.50
            corner_rim = (
                corner_band
                & ~arc_band
                & (pm > 127)
                & (dist_in <= 2.5)
                & (dist_in >= 0.5)
                & (gray > 190)
                & (gray < 250)
                & ~tip_bool
            )
            if np.any(corner_rim):
                ys_c, xs_c = np.where(corner_rim)
                for y, x in zip(ys_c, xs_c):
                    body_xs = np.where(pm[y] > 127)[0]
                    if len(body_xs) == 0:
                        continue
                    xl, xr = int(body_xs.min()), int(body_xs.max())
                    if x <= xl + 2:
                        ref = min(w - 1, xl + 3)
                    elif x >= xr - 2:
                        ref = max(0, xr - 3)
                    else:
                        continue
                    if 18 <= int(gray[y, ref]) < 200:
                        out[y, x] = out[y, ref]
                        corner_sealed = True
        if (
            np.any(leak)
            or np.any(rim_grain)
            or spill_fixed
            or corner_sealed
            or np.any(right_nub)
        ):
            return out
        return output

    @staticmethod
    def _hard_hole_weight(excl_f: np.ndarray) -> np.ndarray:
        """
        Punch holes with a wider soft AA rim so circles/stadiums stay round.

        Soft exclusion ramps used to leave milky wrap inside openings; a hard
        binary core still clears the interior while the rim keeps product AA.
        """
        excl_f = np.clip(excl_f.astype(np.float32), 0.0, 1.0)
        # Clear the hole interior aggressively — a high core threshold left a
        # milky design tongue inside camera/button openings on light phones.
        rim = np.clip((excl_f - 0.05) / 0.55, 0.0, 1.0)
        rim = rim * rim * (3.0 - 2.0 * rim)
        core = (excl_f > 0.55).astype(np.float32)
        return np.clip(np.maximum(core, rim), 0.0, 1.0)

    def _camera_bump_exclusion_maps(
        self,
        exclusion_mask: np.ndarray,
        phone: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build cutout module masks for the border ridge (camera + buttons).

        Returns (module, None, None). Punch is always the full user exclusion —
        nothing from the cover may appear inside the cutout border. The bump
        ridge is shaded later along each module outline using wrap design.
        """
        from .region_detector import HardwareRegionDetector

        h, w = exclusion_mask.shape[:2]
        contours: List[np.ndarray] = []
        if self.hardware_contours:
            contours = list(self.hardware_contours)
        if not contours:
            contours = HardwareRegionDetector._smooth_exclusion_contours(
                exclusion_mask
            )

        camera_ids = {
            id(c) for c in self._camera_like_contours(list(contours))
        }
        module_u8 = np.zeros((h, w), np.uint8)
        for contour in contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())
            bw, bh = x2 - x1, y2 - y1
            if bw > w * 0.42 or bh > h * 0.38:
                continue
            near_side = x1 < w * 0.14 or x2 > w * 0.86
            if id(contour) in camera_ids:
                expand = CAMERA_HOLE_EXPAND_PX
            elif near_side:
                expand = 3.5
            else:
                expand = 2.2
            HardwareRegionDetector.paint_cutout_mask(
                module_u8, pts, analytical=True,
                expand_override=expand,
            )
        if int(np.count_nonzero(module_u8)) < 64:
            return None, None, None

        module = np.clip(module_u8.astype(np.float32) / 255.0, 0.0, 1.0)
        excl_f = np.clip(exclusion_mask.astype(np.float32) / 255.0, 0.0, 1.0)
        module = module * np.clip(excl_f * 1.05, 0.0, 1.0)
        if float(np.max(module)) < 0.05:
            return None, None, None
        return module, None, None

    def _apply_studio_background(
        self,
        output: np.ndarray,
        phone_bgr: np.ndarray,
        mesh: ControlMesh,
        print_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Optional solid-backdrop helper (disabled by default).

        Callers should prefer the original photo background so MagSafe mockups
        keep natural studio lighting / gradients from the source image.
        """
        return output

    def _resolve_material(self, settings: Dict[str, float]) -> MaterialProfile:
        """
        Build a live MaterialProfile from the active preset + slider values.

        Texture kind always comes from the named material so procedural
        surfaces stay consistent while sliders remain interactive.
        """
        base = MATERIALS.get(self.material_name)
        kind = base.texture_kind if base is not None else 'none'
        reflection = float(settings.get('reflection_strength', 28.0)) / 100.0
        reflection = float(np.clip(reflection, 0.0, 0.85))
        if base is not None and base.reflection > 1e-4:
            highlight = reflection * min(base.highlight / base.reflection, 1.15)
        else:
            highlight = reflection * 0.95
        return MaterialProfile(
            name=self.material_name,
            reflection=reflection,
            highlight=highlight,
            shadow_softness=float(settings.get('shadow_strength', 30.0)) / 100.0,
            surface_contrast=0.5 + float(settings.get('contrast', 0.0)) / 200.0,
            texture_strength=float(settings.get('texture_strength', 55.0)) / 100.0,
            opacity=float(settings.get('opacity', 100.0)) / 100.0,
            grain=float(settings.get('grain', 0.0)) / 100.0,
            micro_blur=float(settings.get('blur', 0.0)) / 12.0,
            edge_softness=float(settings.get('edge_softness', 4.0)) / 100.0,
            texture_kind=kind,
        )

    @staticmethod
    def _scaled_pixels(value: float, width: int, height: int) -> float:
        """
        Convert a resolution independent softness value into pixels.

        Keeping feathering proportional to the canvas means the preview and the
        exported image look the same.
        """
        reference = max(width, height) / 1000.0

        return clamp(float(value), 0, 100) * max(reference, 0.25)

    # ------------------------------------------------------------------- info

    def get_cover_points(self) -> Optional[np.ndarray]:
        """Legacy four corners derived from the editable mesh."""
        if self.control_mesh is None:
            return None
        return self.control_mesh.corner_points()

    def get_effective_cover_points(self) -> Optional[np.ndarray]:
        """Legacy four corners after the region inset is applied."""
        mesh = self.get_effective_mesh()
        if mesh is None:
            return None
        return mesh.corner_points()

    def get_control_mesh(self) -> Optional[ControlMesh]:
        """Independent copy of the base editable mesh."""
        return None if self.control_mesh is None else self.control_mesh.copy()

    def get_effective_mesh(self) -> Optional[ControlMesh]:
        """Editable mesh after applying the global region inset."""
        if self.control_mesh is None:
            return None
        inset = float(self.settings.get('region_inset', 0.0))
        if abs(inset) < 1e-6:
            return self.control_mesh.copy()
        return self.control_mesh.inset(inset)

    def get_info(self) -> Dict[str, Any]:
        """Summary used by the status bar."""
        info: Dict[str, Any] = {'ready': self.is_ready}

        if self.phone_image is not None:
            h, w = self.phone_image.shape[:2]
            info['phone_size'] = (w, h)

        if self.design_image is not None:
            h, w = self.design_image.shape[:2]
            info['design_size'] = (w, h)
        info['detection_confidence'] = self.detection_confidence
        info['smart_fit_confidence'] = self.smart_fit_confidence
        info['automatic_margin'] = self.automatic_margin
        info['from_template'] = self.from_template
        info['model_id'] = self.model_id
        info['corner_radii'] = self.corner_radii.to_dict()
        info['curved_uv'] = {
            "enabled": self.curved_uv_params.enabled,
            "rim_uv": self.curved_uv_params.rim_uv,
            "bevel_strength": self.curved_uv_params.bevel_strength,
        }

        return info
