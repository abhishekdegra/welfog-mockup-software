"""
User preference store backed by Qt QSettings (offline, per-user).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QByteArray, QSettings

from ..config import ORG_NAME, APP_NAME, get_config


class UserSettings:
    """Thin typed wrapper around QSettings for recent projects and paths."""

    def __init__(self) -> None:
        self._settings = QSettings(ORG_NAME, APP_NAME)

    # ---------------------------------------------------------------- recent

    def recent_projects(self) -> List[Path]:
        raw = self._settings.value("recent_projects", [])
        if not isinstance(raw, list):
            return []
        paths: List[Path] = []
        for item in raw:
            try:
                path = Path(str(item))
            except (TypeError, ValueError):
                continue
            if path.suffix.lower() and path.exists():
                paths.append(path)
        return paths

    def add_recent_project(self, path: Path) -> None:
        cfg = get_config()
        path = Path(path).resolve()
        items = [p for p in self.recent_projects() if p.resolve() != path]
        items.insert(0, path)
        items = items[: max(1, int(cfg.max_recent_projects))]
        self._settings.setValue(
            "recent_projects", [str(p) for p in items]
        )
        self._settings.setValue("last_project", str(path))
        self._settings.sync()

    def last_project(self) -> Optional[Path]:
        raw = self._settings.value("last_project", "")
        if not raw:
            return None
        path = Path(str(raw))
        return path if path.exists() else None

    def clear_recent_projects(self) -> None:
        self._settings.setValue("recent_projects", [])
        self._settings.remove("last_project")
        self._settings.sync()

    # ------------------------------------------------------------- directories

    def last_dir(self, key: str, fallback: Optional[Path] = None) -> str:
        raw = self._settings.value(f"dirs/{key}", "")
        if raw:
            path = Path(str(raw))
            if path.is_dir():
                return str(path)
            if path.parent.is_dir():
                return str(path.parent)
        if fallback is not None and fallback.exists():
            return str(fallback if fallback.is_dir() else fallback.parent)
        return ""

    def set_last_dir(self, key: str, path: Path) -> None:
        path = Path(path)
        directory = path if path.is_dir() else path.parent
        if directory.exists():
            self._settings.setValue(f"dirs/{key}", str(directory))
            self._settings.sync()

    # -------------------------------------------------------------- window

    def window_geometry(self) -> Optional[QByteArray]:
        value = self._settings.value("window/geometry")
        return value if isinstance(value, QByteArray) else None

    def set_window_geometry(self, geometry: QByteArray) -> None:
        self._settings.setValue("window/geometry", geometry)
        self._settings.sync()

    def window_state(self) -> Optional[QByteArray]:
        value = self._settings.value("window/state")
        return value if isinstance(value, QByteArray) else None

    def set_window_state(self, state: QByteArray) -> None:
        self._settings.setValue("window/state", state)
        self._settings.sync()

    @property
    def reopen_last_on_startup(self) -> bool:
        default = get_config().reopen_last_project
        value = self._settings.value("reopen_last_project", default)
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes"}

    def set_reopen_last_on_startup(self, enabled: bool) -> None:
        self._settings.setValue("reopen_last_project", bool(enabled))
        self._settings.sync()
