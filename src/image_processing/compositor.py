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
        self._phone_wrap_image_id: int = 0
        # Side volume/power ridges for wrap hug + relief shading (not punchouts).
        self._side_button_relief_mask: Optional[np.ndarray] = None
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
            if self._phone_wrap_mesh is not None:
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
        self._phone_wrap_image_id = 0
        self._side_button_relief_mask = None

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

        # Relief shading only — wrap silhouette stays the photo rim (no fat paint).
        wrap_pm, relief = self._volume_button_wrap_assets(body)

        body_for_quad = wrap_pm
        quad = AdaptiveMeshBuilder._aabb_quad_from_mask(body_for_quad)
        if quad is None:
            quad = AdaptiveMeshBuilder._tight_aabb_quad_from_mask(wrap_pm)
        if quad is None:
            return self._phone_wrap_mesh, self._phone_wrap_mask

        # Corner arcs from the live silhouette — relative %, not model IDs.
        try:
            measured = estimate_corner_radii(body_for_quad, quad)
            corner = float(np.clip(measured.median(), 6.0, 18.0))
        except Exception:
            corner = float(
                np.clip(
                    self.settings.get("corner_radius", 11.0) or 11.0, 6.0, 18.0
                )
            )
        cal_c, _ = AdaptiveMeshBuilder.calibrate_corner_radii_from_silhouette(
            body_for_quad, quad, corner, (corner, corner, corner, corner)
        )
        corner = float(np.clip(max(float(cal_c), 6.0), 6.0, 18.0))
        radii = (corner, corner, corner, corner)

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
        wrap = ControlMesh.from_quad(
            order_points(np.asarray(quad, dtype=np.float32)),
            rows,
            cols,
            adaptive=True,
        )
        # Grow so rounded UV still reaches the product rim; hard-clip later.
        wrap = wrap.inset(-5.0)
        wrap = AdaptiveMeshBuilder.force_rounded_perimeter(
            wrap, corner, corner_radii=radii, adaptive=True
        )
        wrap = AdaptiveMeshBuilder.densify_for_curvature(
            wrap, corner, corner_radii=radii
        )
        # Hug the full phone outline (corners + natural button tips).
        rim_px = max(3, int(round(min(h, w) * 0.010)))
        body_rim = cv2.dilate(
            wrap_pm,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (rim_px * 2 + 1, rim_px * 2 + 1)
            ),
            iterations=1,
        )
        AdaptiveMeshBuilder._expand_boundary_to_silhouette(
            wrap, body_rim, corner_only=False
        )
        AdaptiveMeshBuilder._snap_midsides_to_mask(
            wrap,
            wrap_pm,
            smooth=True,
            max_move_fraction=0.035,
        )
        AdaptiveMeshBuilder._expand_boundary_to_silhouette(
            wrap, body_rim, corner_only=True, corner_span=6
        )
        AdaptiveMeshBuilder._reinterpolate_interior(wrap)

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
        return self._phone_wrap_mesh, self._phone_wrap_mask

    def _volume_button_wrap_assets(
        self, phone_mask: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect volume rockers for subtle rim shading only.

        Wrap extent = photo silhouette (including natural button tips). Never
        paint fat stadium bumps — those looked thicker than real keys.
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

    def _limit_side_button_blobs(
        self,
        mask: np.ndarray,
        quad: np.ndarray,
        *,
        max_per_side: int = 3,
    ) -> Optional[np.ndarray]:
        """Keep the strongest few L/R pills; drop leftover ghost blobs."""
        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 24:
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
                if area < 16:
                    continue
                bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                bw = int(stats[label, cv2.CC_STAT_WIDTH])
                # Prefer taller bezel pills (volume) over corner noise.
                score = float(area) + 2.5 * float(bh) - 0.5 * float(bw)
                scored.append((score, label))
            scored.sort(reverse=True)
            for _, label in scored[: max(1, int(max_per_side))]:
                out[labels == label] = 255
        return out if np.count_nonzero(out) >= 24 else None

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

        # Search only a thin halo outside the smoothed body.
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
        tips = cv2.bitwise_and(device, cv2.bitwise_and(halo, cv2.bitwise_not(core)))
        tips = cv2.bitwise_and(tips, cv2.bitwise_not(body))

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
            prefer_live_boundary=True,
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
    def _trim_exterior_speckles(
        alpha: np.ndarray,
        phone_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Drop ink outside the phone footprint only — never shrink interior fill.

        Cleans stray rim grains on the studio card without touching full wrap.
        """
        if phone_mask is None or np.count_nonzero(phone_mask) < 64:
            return alpha
        pm = phone_mask
        if pm.shape[:2] != alpha.shape[:2]:
            pm = cv2.resize(pm, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_NEAREST)
        phone_bin = (pm > 127).astype(np.uint8)
        # 1–2 px AA halo is allowed; anything beyond is studio spill.
        halo = cv2.dilate(
            phone_bin,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
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
        gate = cv2.GaussianBlur(gate, (0, 0), 0.32)
        gate = np.clip(gate, 0.0, 1.0)

        pm = phone_mask
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
        phone_bin = (pm > 127).astype(np.uint8)
        cov = coverage if coverage is not None else env
        _, interior, rim_band, _ = Compositor._outer_rim_band_maps(
            cov, (h, w), corner_w
        )
        guard = Compositor._cutout_guard(hole_w, (h, w))
        if guard is not None:
            interior = interior & ~guard
            rim_band = rim_band & ~guard

        env = np.where(interior, np.maximum(env, 0.985), env)
        env = np.where(
            rim_band,
            np.maximum(np.minimum(env, gate * 1.015), gate * 0.965),
            env,
        )
        env = np.where(phone_bin == 0, np.minimum(env, gate), env)
        env = np.where(phone_bin == 0, 0.0, np.clip(env, 0.0, 1.0))
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

        cov = coverage if coverage is not None else mask
        _, interior, rim, _local_rim = Compositor._outer_rim_band_maps(
            cov, (h, w), corner_w
        )
        guard = Compositor._cutout_guard(hole_w, (h, w))
        if guard is not None:
            interior = interior & ~guard
            rim = rim & ~guard

        m = np.clip(mask.astype(np.float32), 0.0, 1.0)
        gate = None
        if gate_f is not None and float(np.max(gate_f)) > 0.05:
            gate = np.clip(gate_f.astype(np.float32), 0.0, 1.0)
            if gate.shape[:2] != (h, w):
                gate = cv2.resize(gate, (w, h), interpolation=cv2.INTER_LINEAR)
            gate = cv2.GaussianBlur(gate, (0, 0), 0.32)
            gate = np.clip(gate, 0.0, 1.0)

        if touch_mask:
            if gate is not None:
                m = np.where(interior, np.maximum(m, 0.985), m)
                rim_m = np.maximum(
                    np.minimum(m, gate * 1.015), gate * 0.965
                )
                m = np.where(rim, rim_m, m)
                spike = rim & (gate < 0.38)
                m = np.where(spike, gate * np.clip(m, 0.0, 1.0), m)
                m = np.where(phone_bin == 0, np.minimum(m, gate), m)
                m = np.where(phone_bin == 0, 0.0, m)
            else:
                m = np.where(interior, np.maximum(m, 0.985), m)
                m = np.where(phone_bin == 0, 0.0, m)

        a_out = alpha
        if alpha is not None and touch_alpha:
            a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
            if gate is not None:
                a = np.where(interior, np.maximum(a, 0.97), a)
                rim_a = np.maximum(
                    np.minimum(a, gate * 1.025), gate * 0.935
                )
                a = np.where(rim, rim_a, a)
                spike = rim & (gate < 0.38)
                a = np.where(spike, gate * np.clip(a, 0.0, 1.0), a)
                a = np.where(phone_bin == 0, 0.0, a)
            else:
                safe = interior if guard is None else (interior & ~guard)
                a = np.where(safe, np.maximum(a, 0.97), a)
                a = np.where(phone_bin == 0, 0.0, a)
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
        Perfect Wrap mask: clean manufactured rounded-rect (all 4 corners equal).

        Never rebuilds from a jagged photo contour — that made uneven / sharp
        corners and wiped the cover. Phone mask only soft-clips studio spill.
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
                6.5,
                16.0,
            )
        )
        # Force identical arcs on TL/TR/BR/BL — per-corner drift made one
        # corner look sharp / cut while others stayed round.
        if getattr(self, "corner_radii", None) is not None:
            med = float(
                np.median(np.asarray(self.corner_radii.as_tuple(), dtype=np.float64))
            )
            corner = float(np.clip(max(corner, med), 6.5, 16.0))
        radii = (corner, corner, corner, corner)
        self.corner_radii = CornerRadii.uniform(corner)
        self.corner_radius_estimate = corner
        self.settings["corner_radius"] = corner

        mask = create_mesh_mask(
            mesh,
            (h, w),
            feather_radius=max(0, int(feather_radius)),
            corner_radius_percent=corner,
            smooth_boundary=True,
            phone_silhouette=None,
            corner_radii=radii,
            # Geometric rounded arcs — live mesh perimeter inherits photo stairs.
            prefer_live_boundary=False,
        )
        mask = np.clip(mask, 0.0, 1.0)
        # Opaque print face; keep a short smooth AA tip on the curve.
        mask = np.where(mask > 0.55, np.maximum(mask, 0.97), mask)
        base_mask = mask.copy()

        if phone_mask is None or np.count_nonzero(phone_mask) < 64:
            phone_mask = self._resolve_phone_boundary_mask(mesh, (h, w))

        gate_f = None
        if phone_mask is not None and np.count_nonzero(phone_mask) > 64:
            pm = phone_mask
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
                pm = (pm > 127).astype(np.uint8) * 255
            phone_mask = pm
            gate_f = self._product_rim_gate(mesh, pm, (h, w))
            if gate_f is not None:
                gate_f = np.clip(gate_f.astype(np.float32), 0.0, 1.0)
            tip = max(2, int(round(min(h, w) * 0.005)))
            grown = cv2.dilate(
                (base_mask * 255.0).astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (tip * 2 + 1, tip * 2 + 1)
                ),
                iterations=1,
            ).astype(np.float32) / 255.0
            filled = np.maximum(base_mask, grown * 0.98)
            mask, _ = self._apply_rim_antialias(
                filled,
                None,
                pm,
                gate_f,
                (h, w),
                touch_mask=True,
                touch_alpha=False,
            )

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
        """Symmetric rounded gate — same corner turn on all four corners."""
        h, w = map(int, shape)
        if phone_mask is None or np.count_nonzero(phone_mask) < 64:
            return None
        gate = phone_mask
        if gate.shape[:2] != (h, w):
            gate = cv2.resize(gate, (w, h), interpolation=cv2.INTER_LINEAR)
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
        sym = CoverSurfaceEngine.symmetric_rim_gate(
            gate,
            _sharp_quad_from_mesh(mesh),
            corner,
            corner_radii=self.corner_radii,
        )
        if sym is not None and float(np.max(sym)) > 0.05:
            return sym
        pad = max(1, int(round(min(h, w) * 0.0025)))
        dil = cv2.dilate(
            (gate > 127).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
            ),
            iterations=1,
        )
        return self._smooth_silhouette_coverage(dil)

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
            quad = _sharp_quad_from_mesh(mesh)
            corner_w = _corner_proximity_map(
                (h, w),
                x0=float(quad[:, 0].min()),
                y0=float(quad[:, 1].min()),
                x1=float(quad[:, 0].max()),
                y1=float(quad[:, 1].max()),
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
                    blur = cv2.GaussianBlur(ch, (0, 0), sigmaX=2.0)
                    ch = np.where(miss, blur, ch)
                    design[:, :, c] = ch
            miss2 = (solid > 0) & (opaque == 0) & (design_alpha > 0.5)
            if interior_core is not None:
                miss2 = miss2 & interior_core
            if np.any(miss2):
                for c in range(3):
                    ch = design[:, :, c]
                    blur = cv2.GaussianBlur(ch, (0, 0), sigmaX=2.4)
                    ch = np.where(miss2, blur, ch)
                    design[:, :, c] = ch

        # Hardware cutouts: full user exclusion (nothing inside the cutout).
        # Camera bump is a raised RIDGE on the border only — never fills inside.
        bump_module = None
        wrap_mask = np.clip(mask, 0.0, 1.0).copy()
        excl_f = None
        hole_w = None
        if exclusion_mask is not None:
            excl = exclusion_mask
            if excl.shape[:2] != (h, w):
                excl = cv2.resize(excl, (w, h), interpolation=cv2.INTER_LINEAR)
            excl_f = np.clip(excl.astype(np.float32) / 255.0, 0.0, 1.0)
            bump_module, _, _ = self._camera_bump_exclusion_maps(excl, phone)
            hole_w = self._hard_hole_weight(excl_f)
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

        # 4b. Camera bump — raised ridge on cutout BORDER only (cover design).
        if bump_module is not None:
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

        # 4c. Side volume/power — raised wrap relief (design stays on buttons).
        side_relief = self._scaled_mask(self._side_button_relief_mask, (h, w))
        if side_relief is not None and np.count_nonzero(side_relief) > 32:
            design = MaterialRenderingEngine.apply_side_button_relief(
                design,
                mask,
                side_relief.astype(np.float32) / 255.0,
                lighting=lighting,
            )

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
        alpha = self._trim_exterior_speckles(alpha, phone_mask)

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
        if hole_w is not None:
            hw = hole_w
            if hw.shape[:2] != (h, w):
                hw = cv2.resize(hw, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = alpha * (1.0 - np.clip(hw, 0.0, 1.0))
            mask = mask * (1.0 - np.clip(hw, 0.0, 1.0))
        alpha = self._kill_studio_print_fringe(alpha, phone, phone_mask=None)

        # 5. Blend and finish (float until the last step).
        alpha3 = alpha[:, :, np.newaxis]
        result = design * alpha3 + phone_blend * (1.0 - alpha3)

        vignette = float(s.get('vignette', 0.0)) / 100.0
        if vignette > 0:
            result = apply_vignette(result, vignette)

        output = np.clip(np.round(result * 255.0), 0, 255).astype(np.uint8)

        # Outer-corner polish only — never blend over camera/flash cutout arcs.
        outer_dist = MaterialRenderingEngine._outer_perimeter_distance(
            np.clip(mask.astype(np.float32), 0.0, 1.0)
        )
        gate_blend = gate_f
        if gate_blend is not None and float(np.max(gate_blend)) > 0.05:
            if gate_blend.shape[:2] != (h, w):
                gate_blend = cv2.resize(
                    gate_blend, (w, h), interpolation=cv2.INTER_LINEAR
                )
            gate_blend = np.clip(gate_blend.astype(np.float32), 0.0, 1.0)
        else:
            gate_blend = rim
        soft_band = (
            (outer_dist > 0.0)
            & (outer_dist <= max(4.0, float(min(h, w)) * 0.009))
            & (corner_w > 0.28)
        )
        if cutout_guard is not None:
            soft_band = soft_band & ~cutout_guard
        if np.any(soft_band):
            out_f = output.astype(np.float32)
            phone_f = phone_bgr.astype(np.float32)
            w3 = np.clip(gate_blend, 0.04, 0.92)[:, :, np.newaxis]
            blended = out_f * w3 + phone_f * (1.0 - w3)
            out_f = np.where(soft_band[:, :, np.newaxis], blended, out_f)
            output = np.clip(np.round(out_f), 0, 255).astype(np.uint8)

        # Hard guarantee: opaque hole cores stay pixel-identical to the phone.
        # Threshold high so the soft SDF AA rim is not crushed into stairs.
        if exclusion_mask is not None:
            excl = exclusion_mask
            if excl.shape[:2] != (h, w):
                excl = cv2.resize(excl, (w, h), interpolation=cv2.INTER_LINEAR)
            hard_core = excl >= 238
            if np.any(hard_core):
                output[hard_core] = phone_bgr[hard_core]

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
