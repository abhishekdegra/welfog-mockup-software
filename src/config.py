"""
Central application configuration for Phone Cover Mockup Studio.

All production knobs live here (or in the optional user JSON beside the
app data directory). Callers should prefer `get_config()` over hardcoding.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


APP_NAME = "Phone Cover Mockup Studio"
ORG_NAME = "MockupStudio"
APP_VERSION = "2.1.0"
PROJECT_EXTENSION = ".pcms"
PROJECT_FORMAT_VERSION = 1


def app_root() -> Path:
    """
    Project / install root.

    Frozen (PyInstaller) builds resolve next to the executable; source
    runs resolve to the repository root that contains `src/`.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Writable offline data directory (templates, logs, autosave, config)."""
    path = app_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class AppConfig:
    """Tunable production settings."""

    # Identity
    app_name: str = APP_NAME
    org_name: str = ORG_NAME
    app_version: str = APP_VERSION

    # Rendering / preview
    preview_max: int = 1400
    render_debounce_ms: int = 90
    export_quality: int = 96
    png_compression: int = 3
    default_material: str = "Glossy"
    default_lighting: str = "Studio"

    # Caches
    result_cache_size: int = 2
    scaled_phone_cache_size: int = 3
    template_dir: str = ""  # empty → data/templates
    model_dir: str = ""     # empty → data/models (Phase 1 device catalog)
    log_dir: str = ""       # empty → data/logs
    autosave_dir: str = ""  # empty → data/autosave
    max_recent_projects: int = 10
    autosave_interval_sec: int = 60
    reopen_last_project: bool = True

    # Export / batch
    default_export_dir: str = ""
    batch_overwrite_policy: str = "rename"  # rename | overwrite | skip
    export_confirm_overwrite: bool = True

    # Performance
    analysis_long_edge: int = 900
    clear_design_after_batch_job: bool = True

    # Theme / logging
    theme: str = "dark"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_max_bytes: int = 2_000_000
    log_backup_count: int = 3

    def resolved_template_dir(self) -> Path:
        if self.template_dir:
            path = Path(self.template_dir)
        else:
            path = data_dir() / "templates"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_model_dir(self) -> Path:
        """Named phone / cover device templates (Phase 1 catalog)."""
        if self.model_dir:
            path = Path(self.model_dir)
        else:
            path = data_dir() / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_log_dir(self) -> Path:
        if self.log_dir:
            path = Path(self.log_dir)
        else:
            path = data_dir() / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_autosave_dir(self) -> Path:
        if self.autosave_dir:
            path = Path(self.autosave_dir)
        else:
            path = data_dir() / "autosave"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_export_dir(self) -> Optional[Path]:
        if not self.default_export_dir:
            return None
        path = Path(self.default_export_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AppConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)


_CONFIG: Optional[AppConfig] = None


def config_path() -> Path:
    return data_dir() / "config.json"


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load config from disk, falling back to defaults."""
    global _CONFIG
    target = path or config_path()
    cfg = AppConfig()
    if target.exists():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg = AppConfig.from_dict(raw)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            cfg = AppConfig()
    _CONFIG = cfg
    return cfg


def save_config(cfg: Optional[AppConfig] = None, path: Optional[Path] = None) -> Path:
    """Persist the active (or given) config as JSON."""
    global _CONFIG
    active = cfg or _CONFIG or AppConfig()
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(active.to_dict(), indent=2),
        encoding="utf-8",
    )
    _CONFIG = active
    return target


def get_config() -> AppConfig:
    """Return the process-wide config, loading defaults on first use."""
    global _CONFIG
    if _CONFIG is None:
        return load_config()
    return _CONFIG


def set_config(cfg: AppConfig) -> None:
    """Replace the process-wide config (tests / advanced tooling)."""
    global _CONFIG
    _CONFIG = cfg
