"""Local offline JSON template cache and lightweight template manager."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .device_template import CornerRadii, UVBounds
from .mesh import ControlMesh
from .region_detector import PrintableRegion


def default_template_dir() -> Path:
    """App-local templates folder (never cloud)."""
    try:
        from ..config import get_config
        return get_config().resolved_template_dir()
    except Exception:
        root = Path(__file__).resolve().parents[2]
        path = root / "data" / "templates"
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class CoverTemplate:
    """
    Persisted printable-cover geometry for a recurring phone layout.

    Stored independently from project files as a single JSON document.

    Version 3 adds Phase 1 fields: per-corner radii, phone silhouette,
    labelled cutouts, and printable UV bounds. Older v1/v2 files still load.
    """

    fingerprint: str
    aspect: float
    rows: int
    cols: int
    mesh_points: List[List[float]]
    exclusion_contours: List[List[List[float]]] = field(default_factory=list)
    cover_contours: List[List[List[float]]] = field(default_factory=list)
    printable_contours: List[List[List[float]]] = field(default_factory=list)
    margin_percent: float = 0.0
    corner_radius_percent: float = 6.0
    confidence: float = 0.8
    updated_at: float = 0.0
    version: int = 3
    # --- Phase 1 ---
    corner_radii: Optional[Dict[str, float]] = None
    phone_contours: List[List[List[float]]] = field(default_factory=list)
    cutouts: List[dict] = field(default_factory=list)
    uv_bounds: Optional[dict] = None
    model_id: str = ""

    def radii(self) -> CornerRadii:
        """Resolved TL/TR/BR/BL radii (falls back to uniform slider value)."""
        return CornerRadii.from_dict(
            self.corner_radii, fallback=self.corner_radius_percent
        )

    def to_dict(self) -> dict:
        radii = self.radii()
        payload = {
            "fingerprint": self.fingerprint,
            "aspect": self.aspect,
            "rows": self.rows,
            "cols": self.cols,
            "mesh_points": self.mesh_points,
            "exclusion_contours": self.exclusion_contours,
            "cover_contours": self.cover_contours,
            "printable_contours": self.printable_contours,
            "margin_percent": self.margin_percent,
            "corner_radius_percent": float(radii.median()),
            "confidence": self.confidence,
            "updated_at": self.updated_at,
            "version": max(int(self.version), 3),
            "corner_radii": radii.to_dict(),
            "phone_contours": self.phone_contours,
            "cutouts": self.cutouts,
            "uv_bounds": self.uv_bounds or {},
            "model_id": self.model_id,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "CoverTemplate":
        fallback = float(data.get("corner_radius_percent", 6.0))
        radii_raw = data.get("corner_radii")
        return cls(
            fingerprint=str(data["fingerprint"]),
            aspect=float(data["aspect"]),
            rows=int(data["rows"]),
            cols=int(data["cols"]),
            mesh_points=list(data["mesh_points"]),
            exclusion_contours=list(data.get("exclusion_contours", [])),
            cover_contours=list(data.get("cover_contours", [])),
            printable_contours=list(data.get("printable_contours", [])),
            margin_percent=float(data.get("margin_percent", 0.0)),
            corner_radius_percent=fallback,
            confidence=float(data.get("confidence", 0.8)),
            updated_at=float(data.get("updated_at", 0.0)),
            version=int(data.get("version", 1)),
            corner_radii=CornerRadii.from_dict(
                radii_raw, fallback=fallback
            ).to_dict(),
            phone_contours=list(data.get("phone_contours", [])),
            cutouts=list(data.get("cutouts", [])),
            uv_bounds=dict(data["uv_bounds"]) if data.get("uv_bounds") else None,
            model_id=str(data.get("model_id", "")),
        )


class TemplateCache:
    """
    Low-level JSON persistence for cover geometry templates.

    Fingerprints are classical image hashes of the phone silhouette layout so
    the same phone model photographed similarly can skip re-detection.
    """

    HASH_SIZE = 16
    MATCH_DISTANCE = 12
    ASPECT_TOLERANCE = 0.08

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = Path(directory) if directory else default_template_dir()
        self.directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- fingerprint

    @classmethod
    def fingerprint(
        cls, image: np.ndarray, silhouette: Optional[np.ndarray] = None
    ) -> Tuple[str, float]:
        """Compact layout signature. Returns (hex fingerprint, aspect)."""
        height, width = image.shape[:2]
        aspect = float(width) / max(float(height), 1.0)

        if silhouette is not None and np.count_nonzero(silhouette) > 0:
            source = (silhouette > 0).astype(np.uint8) * 255
        else:
            if image.ndim == 2:
                source = image
            elif image.shape[2] == 4:
                source = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                source = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        tiny = cv2.resize(
            source,
            (cls.HASH_SIZE, cls.HASH_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        mean = float(tiny.mean())
        bits = (tiny.astype(np.float32) >= mean).astype(np.uint8).flatten()
        packed = np.packbits(bits)
        digest = hashlib.sha1(packed.tobytes()).hexdigest()[:24]
        return digest, aspect

    @staticmethod
    def hamming(a: str, b: str) -> int:
        """Approximate distance between two fingerprints via hex nibbles."""
        if len(a) != len(b):
            return 64
        distance = 0
        for left, right in zip(a, b):
            xor = int(left, 16) ^ int(right, 16)
            distance += bin(xor).count("1")
        return distance

    def path_for(self, fingerprint: str) -> Path:
        return self.directory / f"{fingerprint}.json"

    # ------------------------------------------------------------------- CRUD

    def save(
        self,
        image: np.ndarray,
        mesh: ControlMesh,
        exclusion_mask: Optional[np.ndarray],
        silhouette: Optional[np.ndarray] = None,
        cover_mask: Optional[np.ndarray] = None,
        printable_mask: Optional[np.ndarray] = None,
        margin_percent: float = 0.0,
        corner_radius_percent: float = 6.0,
        confidence: float = 0.9,
        fingerprint: Optional[str] = None,
        phone_mask: Optional[np.ndarray] = None,
        corner_radii: Optional[CornerRadii] = None,
        cutouts: Optional[List[dict]] = None,
        hardware_contours: Optional[List[np.ndarray]] = None,
        model_id: str = "",
    ) -> CoverTemplate:
        """Create or update the template for this phone layout."""
        fp, aspect = self.fingerprint(image, silhouette)
        if fingerprint is not None:
            fp = fingerprint
        height, width = image.shape[:2]
        points = mesh.normalized_points(width, height)

        radii = corner_radii
        if radii is None:
            radii = CornerRadii.uniform(corner_radius_percent)

        cutout_payload: List[dict] = list(cutouts or [])
        if not cutout_payload and hardware_contours:
            from .device_template import build_cutout_specs
            specs = build_cutout_specs(
                hardware_contours, mesh.corner_points(), width, height
            )
            cutout_payload = [s.to_dict() for s in specs]

        exclusion_contours = self._contours_from_mask(
            exclusion_mask, width, height
        )
        if cutout_payload and not exclusion_contours:
            exclusion_contours = [
                list(c.get("contour", [])) for c in cutout_payload if c.get("contour")
            ]

        uv = UVBounds.from_mesh(mesh, width, height)

        template = CoverTemplate(
            fingerprint=fp,
            aspect=aspect,
            rows=mesh.rows,
            cols=mesh.cols,
            mesh_points=[[float(x), float(y)] for x, y in points],
            exclusion_contours=exclusion_contours,
            cover_contours=self._contours_from_mask(
                cover_mask if cover_mask is not None else silhouette,
                width, height,
            ),
            printable_contours=self._contours_from_mask(
                printable_mask, width, height
            ),
            margin_percent=float(margin_percent),
            corner_radius_percent=float(radii.median()),
            confidence=float(confidence),
            updated_at=time.time(),
            version=3,
            corner_radii=radii.to_dict(),
            phone_contours=self._contours_from_mask(phone_mask, width, height),
            cutouts=cutout_payload,
            uv_bounds=uv.to_dict(),
            model_id=str(model_id or ""),
        )
        self.path_for(fp).write_text(
            json.dumps(template.to_dict(), indent=2), encoding="utf-8"
        )
        return template

    def load(self, fingerprint: str) -> Optional[CoverTemplate]:
        """Load one template by fingerprint."""
        path = self.path_for(fingerprint)
        if not path.is_file():
            return None
        try:
            return CoverTemplate.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def update(
        self, fingerprint: str, **fields
    ) -> Optional[CoverTemplate]:
        """Patch fields on an existing template and rewrite the JSON file."""
        template = self.load(fingerprint)
        if template is None:
            return None
        data = template.to_dict()
        data.update(fields)
        data["updated_at"] = time.time()
        data["version"] = max(int(data.get("version", 3)), 3)
        updated = CoverTemplate.from_dict(data)
        self.path_for(fingerprint).write_text(
            json.dumps(updated.to_dict(), indent=2), encoding="utf-8"
        )
        return updated

    def delete(self, fingerprint: str) -> bool:
        """Remove a template JSON file. Returns True when something was deleted."""
        path = self.path_for(fingerprint)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def list_templates(self) -> List[CoverTemplate]:
        """All readable templates, newest first."""
        templates: List[CoverTemplate] = []
        for path in self.directory.glob("*.json"):
            try:
                templates.append(
                    CoverTemplate.from_dict(
                        json.loads(path.read_text(encoding="utf-8"))
                    )
                )
            except (
                OSError, ValueError, KeyError, TypeError, json.JSONDecodeError
            ):
                continue
        templates.sort(key=lambda item: item.updated_at, reverse=True)
        return templates

    def find(
        self, image: np.ndarray, silhouette: Optional[np.ndarray] = None
    ) -> Optional[CoverTemplate]:
        """Return the closest matching local template, if any."""
        fingerprint, aspect = self.fingerprint(image, silhouette)
        best: Optional[Tuple[int, CoverTemplate]] = None

        for template in self.list_templates():
            if abs(template.aspect - aspect) > self.ASPECT_TOLERANCE:
                continue
            distance = self.hamming(fingerprint, template.fingerprint)
            if distance > self.MATCH_DISTANCE:
                continue
            if best is None or distance < best[0]:
                best = (distance, template)

        return None if best is None else best[1]

    def materialise(
        self, template: CoverTemplate, image_shape: Tuple[int, int]
    ) -> PrintableRegion:
        """Rebuild a PrintableRegion in the current image resolution — fast."""
        height, width = image_shape[:2]
        points = np.asarray(template.mesh_points, dtype=np.float32).copy()
        points[:, 0] *= width
        points[:, 1] *= height
        mesh = ControlMesh(points, template.rows, template.cols)

        cover = self._mask_from_contours(
            template.cover_contours, width, height
        )
        if np.count_nonzero(cover) == 0:
            cover = self._mask_from_mesh(mesh, width, height)

        phone = self._mask_from_contours(
            template.phone_contours, width, height
        )

        printable = self._mask_from_contours(
            template.printable_contours, width, height
        )
        if np.count_nonzero(printable) == 0:
            printable = self._mask_from_mesh(mesh, width, height)
        printable = cv2.bitwise_and(printable, cover) if np.count_nonzero(cover) else printable

        # Prefer Phase 3 CutoutSpecs (frozen geom) over raw exclusion polygons.
        exclusion = np.zeros((height, width), dtype=np.uint8)
        hardware_contours: List[np.ndarray] = []
        cutout_specs = []
        if template.cutouts:
            from .device_template import CutoutSpec
            from .region_detector import HardwareRegionDetector
            for raw in template.cutouts:
                if not raw.get("contour"):
                    continue
                spec = CutoutSpec.from_dict(raw)
                cutout_specs.append(spec)
                HardwareRegionDetector.paint_from_cutout_spec(
                    exclusion, spec, width, height
                )
                pts = spec.pixel_contour(width, height)
                if pts.shape[0] >= 3:
                    hardware_contours.append(
                        pts.reshape(-1, 1, 2).astype(np.float32)
                    )
        if not np.count_nonzero(exclusion):
            exclusion_source = template.exclusion_contours
            exclusion, hardware_contours = self._exclusion_from_contours(
                exclusion_source, width, height
            )
        if np.count_nonzero(exclusion):
            printable = cv2.bitwise_and(
                printable, cv2.bitwise_not((exclusion > 96).astype(np.uint8) * 255)
            )

        region = PrintableRegion(
            mesh=mesh,
            exclusion_mask=exclusion,
            hardware_contours=hardware_contours,
            confidence=max(template.confidence, 0.85),
            silhouette_mask=cover if np.count_nonzero(cover) else printable.copy(),
            printable_mask=printable,
            margin_percent=template.margin_percent,
        )
        # Stash Phase 1 extras for CoverSurfaceEngine without breaking the type.
        region.phone_mask = phone if np.count_nonzero(phone) else None  # type: ignore[attr-defined]
        region.corner_radii = template.radii()  # type: ignore[attr-defined]
        region.uv_bounds = UVBounds.from_dict(template.uv_bounds)  # type: ignore[attr-defined]
        region.model_id = str(template.model_id or "")  # type: ignore[attr-defined]
        region.cutout_specs = cutout_specs  # type: ignore[attr-defined]
        return region

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _mask_from_mesh(
        mesh: ControlMesh, width: int, height: int
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        boundary = np.round(mesh.boundary_points()).astype(np.int32)
        cv2.fillPoly(mask, [boundary.reshape(-1, 1, 2)], 255, cv2.LINE_AA)
        return mask

    @staticmethod
    def _mask_from_contours(
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

    @staticmethod
    def _exclusion_from_contours(
        contours: List[List[List[float]]], width: int, height: int
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        exclusion = np.zeros((height, width), dtype=np.uint8)
        hardware: List[np.ndarray] = []
        from .region_detector import HardwareRegionDetector
        for contour in contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 3:
                continue
            scaled = pts.copy()
            scaled[:, 0] *= width
            scaled[:, 1] *= height
            HardwareRegionDetector.paint_cutout_mask(
                exclusion, scaled, analytical=True
            )
            hardware.append(
                np.round(scaled).astype(np.float32).reshape(-1, 1, 2)
            )

        if np.count_nonzero(exclusion):
            soft = max(3, int(round(min(width, height) * 0.0012)) | 1)
            blurred = cv2.GaussianBlur(exclusion, (soft, soft), 0)
            exclusion = np.maximum(exclusion, blurred)
            exclusion = np.where(
                exclusion >= 250, 255, exclusion
            ).astype(np.uint8)
        return exclusion, hardware

    @staticmethod
    def _contours_from_mask(
        mask: Optional[np.ndarray], width: int, height: int
    ) -> List[List[List[float]]]:
        """Normalised polygons for compact JSON storage."""
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
                contour, max(1.0, 0.01 * cv2.arcLength(contour, True)), True
            )
            pts = approx.reshape(-1, 2).astype(np.float32)
            pts[:, 0] /= max(width, 1)
            pts[:, 1] /= max(height, 1)
            result.append([[float(x), float(y)] for x, y in pts])
        return result


class TemplateManager:
    """
    Lightweight offline template manager (load / save / update / delete).

    Templates are independent from project files and never leave the local
    `data/templates` directory.
    """

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.cache = TemplateCache(directory)

    @property
    def directory(self) -> Path:
        return self.cache.directory

    def list(self) -> List[CoverTemplate]:
        return self.cache.list_templates()

    def load(self, fingerprint: str) -> Optional[CoverTemplate]:
        return self.cache.load(fingerprint)

    def save(
        self,
        image: np.ndarray,
        mesh: ControlMesh,
        exclusion_mask: Optional[np.ndarray] = None,
        silhouette: Optional[np.ndarray] = None,
        cover_mask: Optional[np.ndarray] = None,
        printable_mask: Optional[np.ndarray] = None,
        margin_percent: float = 0.0,
        corner_radius_percent: float = 6.0,
        confidence: float = 0.9,
        phone_mask: Optional[np.ndarray] = None,
        corner_radii: Optional[CornerRadii] = None,
        cutouts: Optional[List[dict]] = None,
        hardware_contours: Optional[List[np.ndarray]] = None,
        model_id: str = "",
    ) -> CoverTemplate:
        """Save or overwrite the template matching this phone layout."""
        return self.cache.save(
            image,
            mesh,
            exclusion_mask,
            silhouette=silhouette,
            cover_mask=cover_mask,
            printable_mask=printable_mask,
            margin_percent=margin_percent,
            corner_radius_percent=corner_radius_percent,
            confidence=confidence,
            phone_mask=phone_mask,
            corner_radii=corner_radii,
            cutouts=cutouts,
            hardware_contours=hardware_contours,
            model_id=model_id,
        )

    def update(self, fingerprint: str, **fields) -> Optional[CoverTemplate]:
        return self.cache.update(fingerprint, **fields)

    def delete(self, fingerprint: str) -> bool:
        return self.cache.delete(fingerprint)

    def find(
        self, image: np.ndarray, silhouette: Optional[np.ndarray] = None
    ) -> Optional[CoverTemplate]:
        return self.cache.find(image, silhouette)

    def materialise(
        self, template: CoverTemplate, image_shape: Tuple[int, int]
    ) -> PrintableRegion:
        return self.cache.materialise(template, image_shape)

    def fingerprint(
        self, image: np.ndarray, silhouette: Optional[np.ndarray] = None
    ) -> Tuple[str, float]:
        return TemplateCache.fingerprint(image, silhouette)
