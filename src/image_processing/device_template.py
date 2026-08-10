"""
Phase 1 device templates — phone-specific geometry for professional mockups.

A DeviceTemplate stores the manufactured cover layout once, so every later
photo of the same phone reuses:
  - outer phone / cover silhouette
  - per-corner radii (TL / TR / BR / BL)
  - labelled camera + button cutouts
  - printable UV bounds (cover-local quad)

Fingerprint CoverTemplate JSON under data/templates/ remains the opportunistic
photo cache. Named models live under data/models/ and are versioned separately.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .mesh import ControlMesh


DEVICE_TEMPLATE_VERSION = 1


def default_model_dir() -> Path:
    """Writable catalog of named phone / cover models."""
    try:
        from ..config import get_config
        return get_config().resolved_model_dir()
    except Exception:
        root = Path(__file__).resolve().parents[2]
        path = root / "data" / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class CornerRadii:
    """
    Per-corner roundness as percent of the cover short edge.

    Order matches mesh UV: TL, TR, BR, BL.
    """

    tl: float = 6.0
    tr: float = 6.0
    br: float = 6.0
    bl: float = 6.0

    def median(self) -> float:
        vals = [self.tl, self.tr, self.br, self.bl]
        return float(np.median(vals))

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (
            float(self.tl),
            float(self.tr),
            float(self.br),
            float(self.bl),
        )

    def clamped(self, lo: float = 2.0, hi: float = 28.0) -> "CornerRadii":
        return CornerRadii(
            tl=float(np.clip(self.tl, lo, hi)),
            tr=float(np.clip(self.tr, lo, hi)),
            br=float(np.clip(self.br, lo, hi)),
            bl=float(np.clip(self.bl, lo, hi)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "tl": float(self.tl),
            "tr": float(self.tr),
            "br": float(self.br),
            "bl": float(self.bl),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict], fallback: float = 6.0) -> "CornerRadii":
        if not data:
            f = float(fallback)
            return cls(tl=f, tr=f, br=f, bl=f)
        f = float(fallback)
        return cls(
            tl=float(data.get("tl", f)),
            tr=float(data.get("tr", f)),
            br=float(data.get("br", f)),
            bl=float(data.get("bl", f)),
        )

    @classmethod
    def uniform(cls, percent: float) -> "CornerRadii":
        p = float(percent)
        return cls(tl=p, tr=p, br=p, bl=p)


@dataclass
class CutoutSpec:
    """
    One hardware hole with stable semantic + geometric freeze (Phase 3).

    ``kind`` is semantic (camera / button / flash).
    ``geom`` is how to paint: circle | stadium | rounded_rect | contour.
    When ``authoritative`` is True the frozen ``params`` / ``contour`` win —
    paint must not re-AABB-classify and collapse photo-true holes.
    """

    kind: str  # camera | button | flash | speaker | other
    contour: List[List[float]]  # normalised [[x,y], ...] in image 0–1
    label: str = ""
    geom: str = ""  # circle | stadium | rounded_rect | rectangle | contour | ""
    params: List[float] = field(default_factory=list)
    expand_px: float = -1.0  # <0 → auto from kind
    authoritative: bool = False
    shape_tag: str = ""  # editor tool lock: capsule | rectangle | circle | …

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "contour": self.contour,
            "label": self.label,
            "geom": self.geom,
            "params": [float(x) for x in self.params],
            "expand_px": float(self.expand_px),
            "authoritative": bool(self.authoritative),
            "shape_tag": self.shape_tag,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CutoutSpec":
        return cls(
            kind=str(data.get("kind", "other")),
            contour=list(data.get("contour", [])),
            label=str(data.get("label", "")),
            geom=str(data.get("geom", "")),
            params=[float(x) for x in data.get("params", [])],
            expand_px=float(data.get("expand_px", -1.0)),
            authoritative=bool(data.get("authoritative", False)),
            shape_tag=str(data.get("shape_tag", "")),
        )

    def pixel_contour(self, width: int, height: int) -> np.ndarray:
        pts = np.asarray(self.contour, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return pts.reshape(0, 2)
        out = pts.copy()
        out[:, 0] *= float(max(width, 1))
        out[:, 1] *= float(max(height, 1))
        return out

    def resolved_expand(self) -> Optional[float]:
        if self.expand_px >= 0.0:
            return float(self.expand_px)
        if self.kind in ("camera", "flash"):
            return 2.25
        if self.kind == "button":
            # Tall volume rockers: tight punch so wrap hugs the ridge.
            # Compact side fingerprint / power pills: wider hole so the
            # sensor opening stays visible (not painted over).
            pts = np.asarray(self.contour, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] >= 3:
                bw = float(pts[:, 0].max() - pts[:, 0].min())
                bh = float(pts[:, 1].max() - pts[:, 1].min())
                aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
                return 1.25 if aspect >= 2.2 else 2.35
            return 2.0
        return None


@dataclass
class UVBounds:
    """
    Printable cover UV frame in normalised image coordinates.

    Quad order: TL, TR, BR, BL. When only an axis-aligned rect is known,
    store it as a degenerate quad from (x0,y0)-(x1,y1).

    Phase 2 adds rim / bevel params for curved source UV foreshortening.
    """

    quad: List[List[float]] = field(default_factory=list)  # 4×[x,y] in 0–1
    rim_uv: float = 0.055
    bevel_strength: float = 0.92

    def to_dict(self) -> dict:
        return {
            "quad": self.quad,
            "rim_uv": float(self.rim_uv),
            "bevel_strength": float(self.bevel_strength),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "UVBounds":
        if not data:
            return cls()
        return cls(
            quad=list(data.get("quad", [])),
            rim_uv=float(data.get("rim_uv", 0.055)),
            bevel_strength=float(data.get("bevel_strength", 0.92)),
        )

    @classmethod
    def from_mesh(
        cls,
        mesh: ControlMesh,
        width: int,
        height: int,
        *,
        rim_uv: float = 0.055,
        bevel_strength: float = 0.92,
    ) -> "UVBounds":
        corners = mesh.corner_points().astype(np.float32)
        w = max(float(width), 1.0)
        h = max(float(height), 1.0)
        quad = [
            [float(corners[i, 0] / w), float(corners[i, 1] / h)]
            for i in range(4)
        ]
        return cls(
            quad=quad,
            rim_uv=float(rim_uv),
            bevel_strength=float(bevel_strength),
        )

    def pixel_quad(self, width: int, height: int) -> Optional[np.ndarray]:
        if len(self.quad) < 4:
            return None
        pts = np.asarray(self.quad, dtype=np.float32).reshape(-1, 2)
        out = pts.copy()
        out[:, 0] *= float(width)
        out[:, 1] *= float(height)
        return out


@dataclass
class DeviceTemplate:
    """
    Authoritative phone / cover geometry for professional mockup reuse.

    Contours are image-normalised (0–1). Mesh UV uses CornerRadii percents of
    the cover short edge — same convention as settings['corner_radius'].
    """

    model_id: str
    display_name: str = ""
    aspect: float = 0.5
    corner_radii: CornerRadii = field(default_factory=CornerRadii)
    phone_contours: List[List[List[float]]] = field(default_factory=list)
    cover_contours: List[List[List[float]]] = field(default_factory=list)
    printable_contours: List[List[List[float]]] = field(default_factory=list)
    cutouts: List[CutoutSpec] = field(default_factory=list)
    uv_bounds: UVBounds = field(default_factory=UVBounds)
    rows: int = 13
    cols: int = 11
    mesh_points: List[List[float]] = field(default_factory=list)
    margin_percent: float = 0.0
    fingerprint: str = ""
    confidence: float = 0.9
    updated_at: float = 0.0
    version: int = DEVICE_TEMPLATE_VERSION
    notes: str = ""

    @property
    def corner_radius_percent(self) -> float:
        """Backward-compatible single slider value = median of four corners."""
        return self.corner_radii.median()

    def camera_cutouts(self) -> List[CutoutSpec]:
        return [c for c in self.cutouts if c.kind in ("camera", "flash")]

    def button_cutouts(self) -> List[CutoutSpec]:
        return [c for c in self.cutouts if c.kind == "button"]

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "aspect": self.aspect,
            "corner_radii": self.corner_radii.to_dict(),
            "corner_radius_percent": self.corner_radius_percent,
            "phone_contours": self.phone_contours,
            "cover_contours": self.cover_contours,
            "printable_contours": self.printable_contours,
            "cutouts": [c.to_dict() for c in self.cutouts],
            "uv_bounds": self.uv_bounds.to_dict(),
            "rows": self.rows,
            "cols": self.cols,
            "mesh_points": self.mesh_points,
            "margin_percent": self.margin_percent,
            "fingerprint": self.fingerprint,
            "confidence": self.confidence,
            "updated_at": self.updated_at,
            "version": self.version,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeviceTemplate":
        fallback = float(data.get("corner_radius_percent", 6.0))
        cutouts_raw = data.get("cutouts")
        cutouts: List[CutoutSpec] = []
        if cutouts_raw:
            cutouts = [CutoutSpec.from_dict(c) for c in cutouts_raw]
        else:
            # Legacy CoverTemplate-style exclusion_contours → unlabelled.
            for contour in data.get("exclusion_contours", []):
                cutouts.append(
                    CutoutSpec(kind="other", contour=list(contour))
                )
        return cls(
            model_id=str(data.get("model_id") or data.get("fingerprint") or ""),
            display_name=str(data.get("display_name", "")),
            aspect=float(data.get("aspect", 0.5)),
            corner_radii=CornerRadii.from_dict(
                data.get("corner_radii"), fallback=fallback
            ),
            phone_contours=list(data.get("phone_contours", [])),
            cover_contours=list(data.get("cover_contours", [])),
            printable_contours=list(data.get("printable_contours", [])),
            cutouts=cutouts,
            uv_bounds=UVBounds.from_dict(data.get("uv_bounds")),
            rows=int(data.get("rows", 13)),
            cols=int(data.get("cols", 11)),
            mesh_points=list(data.get("mesh_points", [])),
            margin_percent=float(data.get("margin_percent", 0.0)),
            fingerprint=str(data.get("fingerprint", "")),
            confidence=float(data.get("confidence", 0.9)),
            updated_at=float(data.get("updated_at", 0.0)),
            version=int(data.get("version", DEVICE_TEMPLATE_VERSION)),
            notes=str(data.get("notes", "")),
        )


def slugify_model_name(name: str) -> str:
    """Filesystem-safe model id from a display name."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip().lower())
    text = text.strip("-")
    return text or f"model-{uuid.uuid4().hex[:8]}"


def contours_from_mask(
    mask: Optional[np.ndarray], width: int, height: int
) -> List[List[List[float]]]:
    """Normalised outer polygons for JSON storage."""
    if mask is None or np.count_nonzero(mask) == 0:
        return []
    binary = (mask > 32).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    result: List[List[List[float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 8:
            continue
        approx = cv2.approxPolyDP(
            contour, max(1.0, 0.008 * cv2.arcLength(contour, True)), True
        )
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= max(width, 1)
        pts[:, 1] /= max(height, 1)
        result.append([[float(x), float(y)] for x, y in pts])
    return result


def mask_from_contours(
    contours: List[List[List[float]]], width: int, height: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for contour in contours:
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        scaled = pts.copy()
        scaled[:, 0] *= width
        scaled[:, 1] *= height
        cv2.fillPoly(
            mask,
            [np.round(scaled).astype(np.int32).reshape(-1, 1, 2)],
            255,
            cv2.LINE_AA,
        )
    return mask


def classify_cutout_kind(
    contour: np.ndarray,
    cover_quad: np.ndarray,
) -> str:
    """
    Label a hardware contour as camera / button / other from cover geometry.

    Same heuristics as compositor camera vs side-button separation — kept here
    so templates persist typed holes without depending on UI state.
    """
    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return "other"
    corners = np.asarray(cover_quad, dtype=np.float32).reshape(-1, 2)
    if corners.shape[0] < 4:
        return "other"
    x_min, x_max = float(corners[:, 0].min()), float(corners[:, 0].max())
    y_min, y_max = float(corners[:, 1].min()), float(corners[:, 1].max())
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)

    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    bw = float(pts[:, 0].max() - pts[:, 0].min())
    bh = float(pts[:, 1].max() - pts[:, 1].min())
    aspect = max(bw, bh) / max(min(bw, bh), 1.0)
    side_band = width * 0.12
    near_side = (cx - x_min) <= side_band or (x_max - cx) <= side_band
    top_half = cy <= y_min + height * 0.52

    thin_button = (
        near_side
        and aspect >= 1.85
        and bw < width * 0.12
        and bh > height * 0.04
    )
    if thin_button:
        return "button"
    # Compact side fingerprint / power pill (any phone).
    compact_fp = (
        near_side
        and bw < width * 0.14
        and height * 0.035 <= bh <= height * 0.22
        and 0.75 <= aspect <= 2.6
    )
    if compact_fp:
        return "button"
    if near_side and cy > y_min + height * 0.28:
        return "button"
    if top_half and not thin_button:
        # Small round satellite near camera island → flash.
        # Looser size so hi-res flash selections stay labeled flash (and
        # therefore forced to a perfect SDF circle on paint/freeze).
        if aspect < 1.55 and max(bw, bh) < min(width, height) * 0.16:
            return "flash"
        return "camera"
    return "other"


def build_cutout_specs(
    contours: Sequence[np.ndarray],
    cover_quad: np.ndarray,
    width: int,
    height: int,
    *,
    phone_gray: Optional[np.ndarray] = None,
    authoritative: bool = False,
    shape_tags: Optional[Sequence[str]] = None,
    corner_frac: float = 0.16,
) -> List[CutoutSpec]:
    """
    Normalise + label hardware contours for a DeviceTemplate.

    When ``authoritative`` is True (Perfect Finish / detect), geom+params are
    frozen from classification / photo snap so export paint never re-guesses.

    ``shape_tags`` (parallel to contours) preserve editor tools such as
    rectangle — without them paint reclassifies AABBs into heavy stadiums.
    """
    from .region_detector import HardwareRegionDetector

    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    specs: List[CutoutSpec] = []
    tags = list(shape_tags) if shape_tags is not None else []
    mild = float(np.clip(corner_frac if corner_frac > 0 else 0.16, 0.08, 0.28))
    for idx, contour in enumerate(contours):
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        kind = classify_cutout_kind(pts, cover_quad)
        tag = ""
        if idx < len(tags):
            tag = str(tags[idx] or "").lower().strip()

        def _box_params(frac: float) -> Tuple[str, Tuple[float, ...]]:
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())
            short = min(x2 - x1, y2 - y1)
            corner = float(np.clip(short * frac, 2.0, short * 0.28))
            return "rounded_rect", (x1, y1, x2, y2, corner)

        tagged_geom = None
        tagged_params: Tuple[float, ...] = ()
        if tag in ("rectangle", "rounded_rect", "rounded_square", "square"):
            frac = 0.08 if tag == "square" else (
                mild if tag in ("rounded_rect", "rounded_square") else 0.16
            )
            tagged_geom, tagged_params = _box_params(frac)
            if tag == "rectangle":
                tagged_geom = "rectangle"
        elif tag in ("pill_h", "pill_v", "capsule", "button"):
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())
            short = min(x2 - x1, y2 - y1)
            tagged_geom = "stadium"
            tagged_params = (
                x1,
                y1,
                x2,
                y2,
                float(np.clip(short * 0.48, 2.0, short * 0.5 - 0.5)),
            )
        elif tag in ("oval", "ellipse"):
            tagged_geom, tagged_params = _box_params(0.50)
            # Paint oval as stadium with full round ends on both axes.
            x1, y1, x2, y2, _c = tagged_params
            short = min(x2 - x1, y2 - y1)
            tagged_geom = "stadium"
            tagged_params = (x1, y1, x2, y2, short * 0.5 - 0.5)
        elif tag in ("squircle", "superellipse"):
            tagged_geom, tagged_params = _box_params(
                0.42 if tag == "squircle" else 0.35
            )
        elif tag == "circle":
            cx, cy, radius = HardwareRegionDetector._circle_params_from_pts(pts)
            if radius > 0.5:
                tagged_geom = "circle"
                tagged_params = (float(cx), float(cy), float(radius))

        if authoritative:
            if tagged_geom:
                expand = 2.25 if kind in ("camera", "flash") else (
                    1.25 if kind == "button" else -1.0
                )
                if kind == "button" and tagged_geom == "stadium":
                    bw = float(pts[:, 0].max() - pts[:, 0].min())
                    bh = float(pts[:, 1].max() - pts[:, 1].min())
                    aspect = max(bw, bh) / max(min(bw, bh), 1.0)
                    expand = 1.25 if aspect >= 2.2 else 2.35
                norm = [[float(x / w), float(y / h)] for x, y in pts]
                specs.append(
                    CutoutSpec(
                        kind=kind,
                        contour=norm,
                        geom=tagged_geom,
                        params=[float(p) for p in tagged_params],
                        expand_px=expand,
                        authoritative=True,
                        shape_tag=tag,
                    )
                )
                continue
            spec = HardwareRegionDetector.freeze_cutout_spec(
                pts,
                kind=kind,
                gray=phone_gray,
                width=int(width),
                height=int(height),
            )
            if tag:
                spec.shape_tag = tag
            specs.append(spec)
            continue

        if tagged_geom:
            geom, params = tagged_geom, tagged_params
        else:
            geom, params = HardwareRegionDetector._classify_cutout(pts)
            if kind == "flash" or HardwareRegionDetector._looks_like_true_disk(
                pts
            ):
                cx, cy, radius = HardwareRegionDetector._circle_params_from_pts(
                    pts
                )
                if radius > 0.5:
                    geom = "circle"
                    params = (float(cx), float(cy), float(radius))
        expand = 2.25 if kind in ("camera", "flash") else -1.0
        if kind == "button":
            bw = float(pts[:, 0].max() - pts[:, 0].min())
            bh = float(pts[:, 1].max() - pts[:, 1].min())
            aspect = max(bw, bh) / max(min(bw, bh), 1.0)
            expand = 1.25 if aspect >= 2.2 else 2.35
        norm = [[float(x / w), float(y / h)] for x, y in pts]
        specs.append(
            CutoutSpec(
                kind=kind,
                contour=norm,
                geom=geom if geom and geom != "free" else "contour",
                params=[float(p) for p in params] if params else [],
                expand_px=expand,
                authoritative=False,
                shape_tag=tag,
            )
        )
    return specs


def estimate_corner_radii(
    cover_mask: np.ndarray,
    cover_quad: Optional[np.ndarray] = None,
) -> CornerRadii:
    """
    Measure TL/TR/BR/BL roundness (%) from the cover silhouette.

    Uses the same bisector walk as CoverSurfaceEngine._estimate_corner_radius
    but keeps all four values instead of collapsing to a median.
    """
    from ..utils.helpers import order_points

    binary = (cover_mask > 0).astype(np.uint8)
    if np.count_nonzero(binary) < 64:
        return CornerRadii.uniform(6.0)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return CornerRadii.uniform(6.0)
    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) < 64:
        return CornerRadii.uniform(6.0)

    if cover_quad is not None and len(np.asarray(cover_quad).reshape(-1, 2)) >= 4:
        box = order_points(np.asarray(cover_quad, dtype=np.float32).reshape(-1, 2))
        # Short edge from ordered quad.
        edges = [
            float(np.linalg.norm(box[(i + 1) % 4] - box[i])) for i in range(4)
        ]
        short = float(max(min(edges), 1.0))
    else:
        rect = cv2.minAreaRect(contour)
        box = order_points(cv2.boxPoints(rect).astype(np.float32))
        short = float(max(min(rect[1]), 1.0))

    radii_px: List[float] = []
    h, w = binary.shape[:2]
    for i in range(4):
        corner = box[i]
        prev_pt = box[(i - 1) % 4]
        next_pt = box[(i + 1) % 4]
        to_prev = prev_pt - corner
        to_next = next_pt - corner
        n0 = float(np.linalg.norm(to_prev))
        n1 = float(np.linalg.norm(to_next))
        if n0 < 1e-3 or n1 < 1e-3:
            radii_px.append(short * 0.06)
            continue
        bisector = to_prev / n0 + to_next / n1
        bn = float(np.linalg.norm(bisector))
        if bn < 1e-3:
            radii_px.append(short * 0.06)
            continue
        direction = bisector / bn
        found = 0.0
        limit = int(short * 0.35)
        for step in range(2, max(3, limit)):
            sample = corner + direction * float(step)
            x = int(round(sample[0]))
            y = int(round(sample[1]))
            if not (0 <= x < w and 0 <= y < h):
                break
            if binary[y, x] > 0:
                x0, x1 = max(0, x - 1), min(w, x + 2)
                y0, y1 = max(0, y - 1), min(h, y + 2)
                if np.count_nonzero(binary[y0:y1, x0:x1]) >= 6:
                    found = float(step)
                    break
        radii_px.append(found if found > 0 else short * 0.06)

    percents = [
        float(np.clip(100.0 * r / short, 2.5, 22.0)) for r in radii_px
    ]
    # Conservative — smoothed gates often over-estimate vs real hardware.
    percents = [float(np.clip(p * 0.90, 2.5, 22.0)) for p in percents]
    # box order after order_points is TL, TR, BR, BL.
    return CornerRadii(
        tl=percents[0], tr=percents[1], br=percents[2], bl=percents[3]
    ).clamped(2.5, 22.0)


class DeviceTemplateCatalog:
    """
    Named device model store under data/models/{model_id}.json.

    Separate from the fingerprint photo cache so human-authored / captured
    models survive silhouette hash churn.
    """

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = Path(directory) if directory else default_model_dir()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, model_id: str) -> Path:
        safe = slugify_model_name(model_id)
        return self.directory / f"{safe}.json"

    def save(self, template: DeviceTemplate) -> Path:
        if not template.model_id:
            template.model_id = slugify_model_name(
                template.display_name or "device"
            )
        else:
            template.model_id = slugify_model_name(template.model_id)
        template.updated_at = time.time()
        template.version = max(int(template.version), DEVICE_TEMPLATE_VERSION)
        path = self.path_for(template.model_id)
        path.write_text(
            json.dumps(template.to_dict(), indent=2), encoding="utf-8"
        )
        return path

    def load(self, model_id: str) -> Optional[DeviceTemplate]:
        path = self.path_for(model_id)
        if not path.is_file():
            # Also accept raw filename without slug rewrite mismatch.
            alt = self.directory / f"{model_id}.json"
            path = alt if alt.is_file() else path
        if not path.is_file():
            return None
        try:
            return DeviceTemplate.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def delete(self, model_id: str) -> bool:
        path = self.path_for(model_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def list(self) -> List[DeviceTemplate]:
        items: List[DeviceTemplate] = []
        for path in self.directory.glob("*.json"):
            try:
                items.append(
                    DeviceTemplate.from_dict(
                        json.loads(path.read_text(encoding="utf-8"))
                    )
                )
            except (
                OSError, ValueError, KeyError, TypeError, json.JSONDecodeError
            ):
                continue
        items.sort(key=lambda t: t.updated_at, reverse=True)
        return items

    def find_by_fingerprint(
        self, fingerprint: str, aspect: float, aspect_tol: float = 0.08
    ) -> Optional[DeviceTemplate]:
        """Match a named model previously linked to a photo fingerprint."""
        if not fingerprint:
            return None
        best: Optional[DeviceTemplate] = None
        for template in self.list():
            if not template.fingerprint:
                continue
            if abs(template.aspect - aspect) > aspect_tol:
                continue
            if template.fingerprint == fingerprint:
                return template
            # Prefer exact; keep first aspect-compatible fingerprint carrier.
            if best is None:
                best = template
        return None if best and best.fingerprint != fingerprint else best

    def capture_from_session(
        self,
        *,
        phone_image: np.ndarray,
        mesh: ControlMesh,
        phone_mask: Optional[np.ndarray] = None,
        cover_mask: Optional[np.ndarray] = None,
        printable_mask: Optional[np.ndarray] = None,
        hardware_contours: Optional[Sequence[np.ndarray]] = None,
        cutout_specs: Optional[Sequence[CutoutSpec]] = None,
        corner_radii: Optional[CornerRadii] = None,
        corner_radius_percent: float = 6.0,
        margin_percent: float = 0.0,
        fingerprint: str = "",
        model_id: str = "",
        display_name: str = "",
        confidence: float = 0.95,
    ) -> DeviceTemplate:
        """
        Build + persist a DeviceTemplate from the live compositor session.

        This is the Phase 1 capture path: Perfect Finish / manual edits become
        a reusable phone-specific model. Phase 3 passes frozen cutout_specs.
        """
        height, width = phone_image.shape[:2]
        aspect = float(width) / max(float(height), 1.0)
        cover = cover_mask
        if cover is None or np.count_nonzero(cover) == 0:
            cover = phone_mask
        radii = corner_radii
        if radii is None:
            if cover is not None and np.count_nonzero(cover):
                radii = estimate_corner_radii(
                    cover, mesh.corner_points() if mesh is not None else None
                )
            else:
                radii = CornerRadii.uniform(corner_radius_percent)
        # Keep median aligned with slider if caller only had one value and
        # measurement failed soft.
        if abs(radii.median() - corner_radius_percent) > 8.0:
            # Prefer the explicit slider when measurement drifted wildly.
            radii = CornerRadii.uniform(corner_radius_percent)

        cutouts: List[CutoutSpec] = []
        if cutout_specs:
            cutouts = list(cutout_specs)
        elif hardware_contours and mesh is not None:
            gray = None
            try:
                gray = cv2.cvtColor(phone_image, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = None
            cutouts = build_cutout_specs(
                hardware_contours,
                mesh.corner_points(),
                width,
                height,
                phone_gray=gray,
                authoritative=True,
            )

        points = mesh.normalized_points(width, height) if mesh is not None else []
        mid = model_id or slugify_model_name(display_name or fingerprint or "device")
        template = DeviceTemplate(
            model_id=mid,
            display_name=display_name or mid,
            aspect=aspect,
            corner_radii=radii.clamped(2.5, 22.0),
            phone_contours=contours_from_mask(phone_mask, width, height),
            cover_contours=contours_from_mask(cover, width, height),
            printable_contours=contours_from_mask(printable_mask, width, height),
            cutouts=cutouts,
            uv_bounds=UVBounds.from_mesh(mesh, width, height) if mesh else UVBounds(),
            rows=mesh.rows if mesh is not None else 15,
            cols=mesh.cols if mesh is not None else 13,
            mesh_points=[[float(x), float(y)] for x, y in points],
            margin_percent=float(margin_percent),
            fingerprint=fingerprint or "",
            confidence=float(confidence),
        )
        self.save(template)
        return template
