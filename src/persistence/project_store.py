"""
Offline project persistence — save / load / autosave mockup sessions.

Projects store image paths + settings + mesh metadata (not pixel buffers),
so files stay small and portable within the same machine paths.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    APP_VERSION, PROJECT_EXTENSION, PROJECT_FORMAT_VERSION, get_config,
)
from ..image_processing.compositor import Compositor
from ..image_processing.mesh import (
    ControlMesh,
    DEFAULT_MESH_COLS,
    DEFAULT_MESH_ROWS,
)
from ..utils.image_loader import ImageLoadError, ImageLoader

logger = logging.getLogger("mockup.project")


class ProjectError(Exception):
    """Raised when a project cannot be saved or loaded."""


@dataclass
class ProjectDocument:
    """Serializable mockup session."""

    format_version: int = PROJECT_FORMAT_VERSION
    app_version: str = APP_VERSION
    phone_path: Optional[str] = None
    design_path: Optional[str] = None
    settings: Dict[str, float] = field(default_factory=dict)
    material_name: str = "Glossy"
    lighting_name: str = "Studio"
    fit_mode: str = "fill"
    mirror: bool = False
    mesh_rows: int = DEFAULT_MESH_ROWS
    mesh_cols: int = DEFAULT_MESH_COLS
    mesh_points: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "app_version": self.app_version,
            "phone_path": self.phone_path,
            "design_path": self.design_path,
            "settings": dict(self.settings),
            "material_name": self.material_name,
            "lighting_name": self.lighting_name,
            "fit_mode": self.fit_mode,
            "mirror": self.mirror,
            "mesh_rows": self.mesh_rows,
            "mesh_cols": self.mesh_cols,
            "mesh_points": self.mesh_points,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ProjectDocument":
        if not isinstance(raw, dict):
            raise ProjectError("Project file is not a JSON object")
        version = int(raw.get("format_version", 1))
        if version > PROJECT_FORMAT_VERSION:
            raise ProjectError(
                f"Unsupported project version {version}; "
                f"this app supports up to {PROJECT_FORMAT_VERSION}"
            )
        settings_raw = raw.get("settings") or {}
        settings = {
            str(k): float(v)
            for k, v in settings_raw.items()
            if isinstance(v, (int, float))
        }
        mesh_points = raw.get("mesh_points")
        if mesh_points is not None and not isinstance(mesh_points, list):
            mesh_points = None
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        return cls(
            format_version=version,
            app_version=str(raw.get("app_version", APP_VERSION)),
            phone_path=raw.get("phone_path"),
            design_path=raw.get("design_path"),
            settings=settings,
            material_name=str(raw.get("material_name", "Glossy")),
            lighting_name=str(raw.get("lighting_name", "Studio")),
            fit_mode=str(raw.get("fit_mode", "fill")),
            mirror=bool(raw.get("mirror", False)),
            mesh_rows=int(raw.get("mesh_rows", DEFAULT_MESH_ROWS)),
            mesh_cols=int(raw.get("mesh_cols", DEFAULT_MESH_COLS)),
            mesh_points=mesh_points,
            metadata=dict(metadata),
        )


class ProjectStore:
    """Save and restore Compositor sessions as `.pcms` JSON projects."""

    @staticmethod
    def autosave_path() -> Path:
        return get_config().resolved_autosave_dir() / "last_session.pcms"

    @classmethod
    def capture(
        cls,
        compositor: Compositor,
        phone_path: Optional[Path] = None,
        design_path: Optional[Path] = None,
        *,
        name: str = "",
        existing: Optional[ProjectDocument] = None,
    ) -> ProjectDocument:
        """Snapshot the live session into a document."""
        mesh = compositor.control_mesh
        mesh_points = None
        rows, cols = DEFAULT_MESH_ROWS, DEFAULT_MESH_COLS
        if mesh is not None:
            rows, cols = mesh.rows, mesh.cols
            mesh_points = mesh.points.astype(float).tolist()

        now = datetime.now(timezone.utc).isoformat()
        metadata = dict(existing.metadata) if existing is not None else {}
        metadata.setdefault("created_at", now)
        metadata["modified_at"] = now
        if name:
            metadata["name"] = name
        elif phone_path is not None:
            metadata.setdefault("name", phone_path.stem)

        return ProjectDocument(
            format_version=PROJECT_FORMAT_VERSION,
            app_version=APP_VERSION,
            phone_path=str(phone_path) if phone_path else None,
            design_path=str(design_path) if design_path else None,
            settings=compositor.get_settings(),
            material_name=compositor.material_name,
            lighting_name=compositor.lighting_name,
            fit_mode=compositor.fit_mode,
            mirror=compositor.mirror,
            mesh_rows=rows,
            mesh_cols=cols,
            mesh_points=mesh_points,
            metadata=metadata,
        )

    @classmethod
    def save(
        cls,
        path: Path,
        compositor: Compositor,
        phone_path: Optional[Path] = None,
        design_path: Optional[Path] = None,
        *,
        existing: Optional[ProjectDocument] = None,
    ) -> ProjectDocument:
        """Write a project file to disk."""
        path = Path(path)
        if path.suffix.lower() != PROJECT_EXTENSION:
            path = path.with_suffix(PROJECT_EXTENSION)
        path.parent.mkdir(parents=True, exist_ok=True)

        document = cls.capture(
            compositor, phone_path, design_path, existing=existing
        )
        try:
            path.write_text(
                json.dumps(document.to_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to save project %s: %s", path, exc)
            raise ProjectError(f"Could not write project: {exc}") from exc

        logger.info("Saved project %s", path)
        return document

    @classmethod
    def autosave(
        cls,
        compositor: Compositor,
        phone_path: Optional[Path] = None,
        design_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Write the recovery autosave; returns path or None on failure."""
        if compositor.phone_image is None and compositor.design_image is None:
            return None
        target = cls.autosave_path()
        try:
            cls.save(target, compositor, phone_path, design_path)
            return target
        except ProjectError as exc:
            logger.warning("Autosave failed: %s", exc)
            return None

    @classmethod
    def load_document(cls, path: Path) -> ProjectDocument:
        """Parse a project file without applying it."""
        path = Path(path)
        if not path.exists():
            raise ProjectError(f"Project not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Corrupt project %s: %s", path, exc)
            raise ProjectError(f"Could not read project: {exc}") from exc
        return ProjectDocument.from_dict(raw)

    @classmethod
    def apply(
        cls,
        document: ProjectDocument,
        compositor: Compositor,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Restore a document onto a compositor.

        Returns the resolved (phone_path, design_path). Missing image files
        raise ProjectError after a clear message — never crash silently.
        """
        phone_path = Path(document.phone_path) if document.phone_path else None
        design_path = Path(document.design_path) if document.design_path else None

        if phone_path is not None:
            if not phone_path.exists():
                raise ProjectError(f"Phone image missing: {phone_path}")
            try:
                phone = ImageLoader.load_image(phone_path)
            except ImageLoadError as exc:
                raise ProjectError(f"Phone image unreadable: {exc}") from exc
            try:
                compositor.set_phone_image(phone)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Phone geometry failed while loading project")
                raise ProjectError(f"Cover detection failed: {exc}") from exc

        if design_path is not None:
            if not design_path.exists():
                raise ProjectError(f"Design image missing: {design_path}")
            try:
                design = ImageLoader.load_image(design_path)
            except ImageLoadError as exc:
                raise ProjectError(f"Design image unreadable: {exc}") from exc
            try:
                compositor.set_design_image(design)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Design apply failed while loading project")
                raise ProjectError(f"Could not apply design: {exc}") from exc

        # Restore look after images. Placement (scale/offset/rotation/inset) is
        # always recomputed from live geometry below — never restore stale
        # sticker offsets from an older skewed session.
        compositor.material_name = document.material_name
        compositor.lighting_name = document.lighting_name
        compositor.fit_mode = document.fit_mode
        compositor.mirror = bool(document.mirror)
        placement_keys = {
            "design_scale",
            "offset_x",
            "offset_y",
            "rotation",
            "region_inset",
        }
        if document.settings:
            for key, value in document.settings.items():
                if key not in placement_keys:
                    compositor.settings[key] = value

        if (
            document.mesh_points is not None
            and compositor.phone_image is not None
        ):
            try:
                points = np.asarray(document.mesh_points, dtype=np.float32)
                expected = document.mesh_rows * document.mesh_cols
                if points.shape == (expected, 2):
                    mesh = ControlMesh(
                        points, document.mesh_rows, document.mesh_cols
                    )
                    compositor.set_control_mesh(mesh)
                    # Re-apply non-placement look settings after mesh sync.
                    if document.settings:
                        for key, value in document.settings.items():
                            if key not in placement_keys:
                                compositor.settings[key] = value
                    # Autosaves often keep a loose cage + bitten template mask
                    # (half-wrap / sharp sticker). Sync rounded wrap mesh first
                    # so the edit UI blue outline matches what will render.
                    try:
                        compositor._finalize_fullbleed_mesh()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Wrap finalize after project load failed: %s", exc
                        )
                    try:
                        compositor.heal_realistic_wrap(include_hardware=False)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Wrap heal after project load failed: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not restore mesh from project: %s", exc)

        # CSS object-fit: cover from the current phone boundary + cutouts.
        if (
            compositor.design_image is not None
            and compositor.control_mesh is not None
        ):
            compositor.auto_fit_design()

        compositor.invalidate(clear_scaled=True)
        logger.info(
            "Applied project (phone=%s design=%s)",
            phone_path, design_path,
        )
        return phone_path, design_path

    @classmethod
    def load(
        cls, path: Path, compositor: Compositor
    ) -> Tuple[ProjectDocument, Optional[Path], Optional[Path]]:
        """Load and apply a project file."""
        document = cls.load_document(path)
        phone_path, design_path = cls.apply(document, compositor)
        return document, phone_path, design_path
