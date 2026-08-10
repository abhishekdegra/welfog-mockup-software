"""Undo / redo snapshots for offline mockup editing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class EditSnapshot:
    """Geometry + look state that can be restored without reloading images."""

    label: str = ""
    coalesce_key: Optional[str] = None
    mesh_points: Optional[np.ndarray] = None
    mesh_rows: int = 0
    mesh_cols: int = 0
    exclusion_mask: Optional[np.ndarray] = None
    printable_mask: Optional[np.ndarray] = None
    hardware_contours: List[np.ndarray] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    material_name: str = "Glossy"
    lighting_name: str = "Studio"
    fit_mode: str = "fill"
    mirror: bool = False
    corner_radius_estimate: float = 6.0
    automatic_margin: float = 0.0
    auto_detected: bool = False
    from_template: bool = False


class EditHistory:
    """
    Linear undo/redo stack.

    `push` stores the state *before* a change. `undo` restores that state and
    parks the live state on the redo stack.
    """

    def __init__(self, limit: int = 80):
        self.limit = max(1, int(limit))
        self._undo: List[EditSnapshot] = []
        self._redo: List[EditSnapshot] = []

    def clear(self) -> None:
        """Drop both stacks (e.g. after loading a new phone)."""
        self._undo.clear()
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_label(self) -> str:
        return self._undo[-1].label if self._undo else ""

    def redo_label(self) -> str:
        return self._redo[-1].label if self._redo else ""

    def push(self, snapshot: EditSnapshot) -> None:
        """Record a pre-change snapshot; clears redo."""
        if (
            snapshot.coalesce_key
            and self._undo
            and self._undo[-1].coalesce_key == snapshot.coalesce_key
        ):
            # Continuous slider drag: keep the original pre-drag baseline.
            return
        self._undo.append(snapshot)
        while len(self._undo) > self.limit:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current: EditSnapshot) -> Optional[EditSnapshot]:
        """Pop undo, push current onto redo, return restored snapshot."""
        if not self._undo:
            return None
        previous = self._undo.pop()
        self._redo.append(current)
        return previous

    def redo(self, current: EditSnapshot) -> Optional[EditSnapshot]:
        """Pop redo, push current onto undo, return restored snapshot."""
        if not self._redo:
            return None
        nxt = self._redo.pop()
        self._undo.append(current)
        return nxt
