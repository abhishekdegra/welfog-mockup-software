"""
Custom widgets: the interactive preview canvas and the control panel pieces.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QFont, QImage, QPainter, QPainterPath, QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from .styles import Palette


SUPPORTED_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


def numpy_to_qpixmap(image: np.ndarray) -> Optional[QPixmap]:
    """
    Convert an OpenCV image to a QPixmap.

    The QImage is copied because Qt would otherwise reference the numpy buffer,
    which can be freed while the pixmap is still on screen.
    """
    if image is None or image.size == 0:
        return None

    array = np.ascontiguousarray(image)

    if array.ndim == 2:
        height, width = array.shape
        qimage = QImage(array.data, width, height, width,
                        QImage.Format_Grayscale8)
    elif array.shape[2] == 4:
        array = np.ascontiguousarray(cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA))
        height, width = array.shape[:2]
        qimage = QImage(array.data, width, height, width * 4,
                        QImage.Format_RGBA8888)
    else:
        height, width = array.shape[:2]
        qimage = QImage(array.data, width, height, width * 3,
                        QImage.Format_BGR888)

    return QPixmap.fromImage(qimage.copy())


class PreviewCanvas(QWidget):
    """
    Displays the composite with zoom, pan, drag and drop, and lets the user
    locally deform a multi-point cover mesh.
    """

    filesDropped = Signal(list)
    browseRequested = Signal()
    coverPointsChanged = Signal(object)
    meshPointsChanged = Signal(object, int, int)
    exclusionContoursChanged = Signal(object)
    exclusionBrushStroke = Signal(object)  # list of (nx, ny, radius_norm)
    viewChanged = Signal(float)
    # Drag deltas while "Move Design" is on (fraction of displayed image).
    designPanDelta = Signal(float, float)
    designPanFinished = Signal()
    designZoomDelta = Signal(float)  # multiplicative scale factor, e.g. 1.08

    HANDLE_RADIUS = 9
    HIT_RADIUS = 16
    EXCLUSION_HIT_RADIUS = 14
    MIN_ZOOM = 0.05
    MAX_ZOOM = 12.0

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumSize(460, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

        self._pixmap: Optional[QPixmap] = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._auto_fit = True

        self._mesh: Optional[np.ndarray] = None  # normalised 0-1 coordinates
        self._mesh_rows = 0
        self._mesh_cols = 0
        self._exclusion_contours: List[np.ndarray] = []
        self._exclusion_shapes: List[str] = []
        self._cutout_shape = "circle"
        self._cutout_corner_frac = 0.16
        self._cutout_rotation_deg = 0.0
        self._show_cover = False
        self._edit_cover = False
        self._erase_mode = False
        self._final_mode = False
        self._brush_kind: Optional[str] = None  # None | "erase" | "fill"
        self._brush_radius_px = 18.0
        self._brush_painting = False
        self._brush_stroke: List[tuple] = []  # (nx, ny, r_norm)
        self._brush_cursor: Optional[QPointF] = None
        self._hover_handle = -1
        self._drag_handle = -1
        self._dragging_quad = False
        self._hover_exclusion: tuple = (-1, -1)  # contour, vertex (-1 body, -2 delete)
        self._drag_exclusion: tuple = (-1, -1)
        self._drag_exclusion_body = -1
        self._drag_exclusion_role = -1
        self._exclusion_drag_origin: Optional[np.ndarray] = None
        self._exclusion_scale_center: Optional[np.ndarray] = None
        self._exclusion_scale_ref: float = 1.0
        self._panning = False
        self._design_panning = False
        self._design_pan_mode = False
        self._design_pan_last: Optional[QPointF] = None
        self._press_pos = QPoint()
        self._drag_origin: Optional[np.ndarray] = None
        self._moved = False
        self._drag_hint = False

        self.placeholder_title = "Drop your phone photo here"
        self.placeholder_body = ("or click to browse\n"
                                "PNG · JPG · WEBP · BMP · TIFF")

    # ------------------------------------------------------------- contents

    def set_image(self, image) -> None:
        """Set the displayed image from a numpy array or QPixmap."""
        pixmap = image if isinstance(image, QPixmap) else numpy_to_qpixmap(image)
        first_image = self._pixmap is None

        self._pixmap = pixmap

        if pixmap is not None and (first_image or self._auto_fit):
            self.fit_to_view()
        else:
            self.update()

    def clear_image(self) -> None:
        """Remove the displayed image and show the placeholder again."""
        self._pixmap = None
        self._mesh = None
        self._exclusion_contours = []
        self._exclusion_shapes = []
        self._hover_exclusion = (-1, -1)
        self._drag_exclusion = (-1, -1)
        self._drag_exclusion_body = -1
        self._drag_exclusion_role = -1
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def has_image(self) -> bool:
        """Whether an image is currently displayed."""
        return self._pixmap is not None

    def set_cover_points(self, points: Optional[np.ndarray]) -> None:
        """Compatibility setter for a legacy four-corner cover."""
        if points is None:
            self._mesh = None
            self._mesh_rows = 0
            self._mesh_cols = 0
        else:
            quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
            # Mesh storage is row-major TL/TR/BL/BR while the compatibility
            # API is the historical TL/TR/BR/BL order.
            self._mesh = quad[[0, 1, 3, 2]].copy()
            self._mesh_rows = 2
            self._mesh_cols = 2
        self.update()

    def cover_points(self) -> Optional[np.ndarray]:
        """Legacy four corners derived from the current mesh."""
        if self._mesh is None:
            return None
        if self._mesh_rows < 2 or self._mesh_cols < 2:
            return self._mesh.copy()
        return self._mesh[
            [
                0,
                self._mesh_cols - 1,
                self._mesh_rows * self._mesh_cols - 1,
                (self._mesh_rows - 1) * self._mesh_cols,
            ]
        ].copy()

    def _corner_indices(self) -> List[int]:
        """Only the four wrap corners — the sole editable mesh handles."""
        if self._mesh is None or self._mesh_rows < 2 or self._mesh_cols < 2:
            return []
        return [
            0,
            self._mesh_cols - 1,
            self._mesh_rows * self._mesh_cols - 1,
            (self._mesh_rows - 1) * self._mesh_cols,
        ]

    def _rebuild_mesh_from_corners(self) -> None:
        """Bilinear dense warp mesh from the four corner handles."""
        if self._mesh is None or self._mesh_rows < 2 or self._mesh_cols < 2:
            return
        from ..image_processing.mesh import ControlMesh

        corners = self.cover_points()
        if corners is None or len(corners) != 4:
            return
        rebuilt = ControlMesh.from_quad(
            corners, self._mesh_rows, self._mesh_cols
        )
        self._mesh = rebuilt.points.copy()
        self._mesh_rows = rebuilt.rows
        self._mesh_cols = rebuilt.cols

    def set_mesh_points(
        self, points: Optional[np.ndarray], rows: int = 0, cols: int = 0
    ) -> None:
        """Set all editable mesh vertices in normalised image coordinates."""
        if points is None:
            self._mesh = None
            self._mesh_rows = 0
            self._mesh_cols = 0
        else:
            array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            if len(array) != int(rows) * int(cols):
                raise ValueError("Mesh point count does not match rows × cols")
            self._mesh = array.copy()
            self._mesh_rows = int(rows)
            self._mesh_cols = int(cols)
        self.update()

    def mesh_points(self) -> Optional[np.ndarray]:
        """Current normalised mesh vertices."""
        return None if self._mesh is None else self._mesh.copy()

    def set_exclusion_contours(
        self,
        contours: List[np.ndarray],
        shapes: Optional[List[str]] = None,
    ) -> None:
        """Set normalised hardware outlines displayed above the mesh."""
        prev = list(getattr(self, "_exclusion_shapes", []))
        cleaned: List[np.ndarray] = []
        kept_shapes: List[str] = []
        forced = list(shapes) if shapes is not None else None
        for index, contour in enumerate(contours):
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2).copy()
            if len(pts) < 3:
                continue
            # Drop noise slivers that only confuse editing.
            span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            if span < 0.003:
                continue
            tag = ""
            if forced is not None and index < len(forced):
                tag = str(forced[index] or "").strip()
            if not tag and index < len(prev):
                tag = str(prev[index] or "").strip()
            if not tag:
                tag = self._infer_exclusion_shape_tag(pts)
            # Compositor sync already holds painted verts — rebuilding dense
            # stadiums here after every cutout save froze the UI.
            if forced is not None and tag:
                cleaned.append(np.clip(pts, -0.05, 1.05))
            else:
                cleaned.append(
                    self._normalize_exclusion_contour(pts, shape_tag=tag)
                )
            kept_shapes.append(tag)
        # Never blank cutouts while the user is mid mesh-drag — that made red
        # camera shapes vanish until edit mode was toggled again.
        if (
            not cleaned
            and self._exclusion_contours
            and (self._drag_handle >= 0 or self._dragging_quad)
        ):
            return
        self._exclusion_contours = cleaned
        self._exclusion_shapes = kept_shapes
        self._hover_exclusion = (-1, -1)
        self._drag_exclusion = (-1, -1)
        self._drag_exclusion_body = -1
        self._drag_exclusion_role = -1
        self.update()

    def _normalize_exclusion_contour(
        self,
        contour: np.ndarray,
        *,
        shape_tag: str = "",
        corner_frac: float = -1.0,
        rotation_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Store cutouts as sparse, editable geometry (Canva-style).

        When ``shape_tag`` is set (user-chosen tool), rebuild that shape from
        the AABB — never auto-morph square/triangle/pill into a circle.
        Untagged contours keep the previous classify path.
        """
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2).copy()
        if len(pts) < 3:
            return np.clip(pts, -0.05, 1.05)
        x1 = float(pts[:, 0].min())
        y1 = float(pts[:, 1].min())
        x2 = float(pts[:, 0].max())
        y2 = float(pts[:, 1].max())
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        width = max(1e-6, x2 - x1)
        height = max(1e-6, y2 - y1)
        short = min(width, height)
        # Geometric-mean size keeps AABB W×H stable across rebuilds:
        # half_w = size*sqrt(aspect), half_h = size/sqrt(aspect).
        size = 0.5 * float(np.sqrt(width * height))
        aspect = max(1e-3, width / height)

        try:
            from ..image_processing.region_detector import HardwareRegionDetector

            tag = (shape_tag or "").lower().strip()
            # Shapes the user explicitly chose — preserve across move/resize.
            preserve = {
                "circle", "square", "rectangle", "rounded_square", "rounded_rect",
                "oval", "pill_h", "pill_v", "squircle", "superellipse",
                "polygon", "custom_path", "triangle", "free", "capsule",
                "button",
            }
            if tag in preserve:
                # Only true circles stay locked 1:1. Squares stay square.
                # Everything else (rounded square, capsule, pills, rects…)
                # keeps independent width × height from the AABB.
                if tag == "circle":
                    aspect = 1.0
                    size = 0.5 * short
                elif tag == "square":
                    aspect = 1.0
                    size = 0.5 * short
                # Freeform tools: scale existing verts to the AABB — never
                # re-seed, or custom path / triangle edits get wiped.
                if tag in ("custom_path", "free", "triangle", "polygon"):
                    return np.clip(
                        self._scale_contour_to_box(pts, x1, y1, x2, y2),
                        -0.05,
                        1.05,
                    )
                poly = HardwareRegionDetector.make_shape_polygon(
                    tag,
                    (cx, cy),
                    size,
                    aspect=aspect,
                    corner_frac=(
                        float(corner_frac)
                        if corner_frac >= 0.0
                        else self._cutout_corner_frac
                    ),
                    rotation_deg=float(rotation_deg),
                ).reshape(-1, 2)
                return np.clip(poly, -0.05, 1.05)

            kind, params = HardwareRegionDetector._classify_cutout(pts)
            if kind == "circle":
                radius = 0.5 * short
                clean = HardwareRegionDetector._sample_circle(
                    cx, cy, radius, samples=72
                )
                return np.clip(clean.reshape(-1, 2), -0.05, 1.05)
            # Stadium / rounded_rect / free → sparse rounded rect (4 corner handles).
            aspect_box = max(width, height) / short
            if kind == "stadium" or aspect_box >= 1.35:
                corner = float(np.clip(short * 0.48, short * 0.2, short * 0.5 - 1e-4))
            else:
                corner = float(np.clip(short * 0.32, short * 0.12, short * 0.44))
            clean = HardwareRegionDetector._sample_rounded_rect(
                x1, y1, x2, y2, corner, samples_per_corner=3
            )
            if clean is not None and len(clean) >= 8:
                return np.clip(clean.reshape(-1, 2), -0.05, 1.05)
        except Exception:
            pass
        return np.clip(pts, -0.05, 1.05)

    def exclusion_contours(self) -> List[np.ndarray]:
        """Current normalised hardware exclusion outlines."""
        return [contour.copy() for contour in self._exclusion_contours]

    def exclusion_shapes(self) -> List[str]:
        """Shape tags parallel to ``exclusion_contours`` (rectangle, circle…)."""
        shapes = list(getattr(self, "_exclusion_shapes", []))
        while len(shapes) < len(self._exclusion_contours):
            shapes.append("")
        return shapes[: len(self._exclusion_contours)]

    def set_cutout_shape(self, shape: str) -> None:
        """Shape used when Shift+click adds a cutout."""
        allowed = {
            "circle", "square", "triangle", "free", "capsule", "button",
            "rounded_square", "rounded_rect", "rectangle", "oval",
            "pill_h", "pill_v", "squircle", "superellipse", "polygon",
            "custom_path",
        }
        self._cutout_shape = shape if shape in allowed else "circle"

    def cutout_shape(self) -> str:
        """Current cutout creation shape."""
        return self._cutout_shape

    def set_erase_mode(self, enabled: bool) -> None:
        """Paint-to-erase wrap mode (adds exclusion under the brush)."""
        self._erase_mode = bool(enabled)
        if self._erase_mode:
            self._final_mode = False
            self._brush_kind = "erase"
        elif not self._final_mode:
            self._brush_kind = None
            self._brush_painting = False
            self._brush_stroke = []
            self._brush_cursor = None
        self._refresh_brush_cursor()
        self.update()

    def set_final_brush_mode(self, mode: Optional[str]) -> None:
        """
        Final polish brush: ``\"erase\"``, ``\"fill\"``, or ``None`` to disable.

        Works without Edit Mesh so post-wrap cleanup stays clean.
        """
        if mode not in (None, "erase", "fill"):
            mode = None
        self._final_mode = mode is not None
        self._brush_kind = mode
        if self._final_mode:
            self._erase_mode = False
            # Smaller default tip for precise Final work.
            if self._brush_radius_px > 14.0:
                self._brush_radius_px = 11.0
        self._brush_painting = False
        self._brush_stroke = []
        self._brush_cursor = None
        self._refresh_brush_cursor()
        self.update()

    def final_brush_mode(self) -> Optional[str]:
        """Active Final brush kind, or None."""
        if self._final_mode and self._brush_kind in ("erase", "fill"):
            return self._brush_kind
        return None

    def brush_kind(self) -> Optional[str]:
        """Active paint kind for the current stroke (erase / fill)."""
        return self._brush_kind

    def _brush_active(self) -> bool:
        """True when a paint brush (Erase Wrap or Final) can draw."""
        if self._final_mode and self._brush_kind in ("erase", "fill"):
            return True
        return bool(self._erase_mode and self._edit_cover and self._brush_kind == "erase")

    def _refresh_brush_cursor(self) -> None:
        if self._brush_active():
            self.setCursor(Qt.BlankCursor)
        elif self._design_pan_mode and not self._edit_cover:
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def set_design_pan_mode(self, enabled: bool) -> None:
        """
        Drag on the preview to slide the print design (not the view).

        Disabled while Edit Mesh is active so cutout/mesh drags stay primary.
        """
        self._design_pan_mode = bool(enabled)
        if not self._design_pan_mode:
            self._design_panning = False
            self._design_pan_last = None
        if self._design_pan_mode and not self._edit_cover and not self._brush_active():
            self.setCursor(Qt.SizeAllCursor)
        elif not self._brush_active():
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def design_pan_mode(self) -> bool:
        """Whether drag pans the design artwork."""
        return self._design_pan_mode

    def erase_mode(self) -> bool:
        """Whether the erase-wrap brush is active."""
        return self._erase_mode

    def set_brush_radius_px(self, radius: float) -> None:
        """Brush radius in widget pixels (clamped)."""
        self._brush_radius_px = float(np.clip(radius, 4.0, 80.0))
        self.update()

    def brush_radius_px(self) -> float:
        """Current erase brush radius in widget pixels."""
        return self._brush_radius_px

    def set_show_cover(self, show: bool) -> None:
        """Show or hide the cover outline."""
        self._show_cover = bool(show)
        self.update()

    def set_edit_cover(self, editable: bool) -> None:
        """Enable dragging of the cover corners and camera cutouts."""
        self._edit_cover = bool(editable)
        if editable:
            self._show_cover = True
            self._design_panning = False
            self._design_pan_last = None
            # Edit Mesh and Final polish are exclusive.
            if self._final_mode:
                self._final_mode = False
                self._brush_kind = None
                self._brush_painting = False
                self._brush_stroke = []
                self._brush_cursor = None
        else:
            self._erase_mode = False
            if not self._final_mode:
                self._brush_kind = None
            self._brush_painting = False
            self._brush_stroke = []
        self._hover_handle = -1
        self._hover_exclusion = (-1, -1)
        self._refresh_brush_cursor()
        self.update()

    def is_edit_cover(self) -> bool:
        """Whether mesh editing is enabled."""
        return self._edit_cover

    # ----------------------------------------------------------------- view

    def zoom(self) -> float:
        """Current zoom factor."""
        return self._zoom

    def fit_zoom(self) -> float:
        """Zoom factor that fits the image inside the viewport."""
        if self._pixmap is None:
            return 1.0

        available_w = max(1, self.width() - 48)
        available_h = max(1, self.height() - 48)

        return min(available_w / max(1, self._pixmap.width()),
                   available_h / max(1, self._pixmap.height()))

    def fit_to_view(self) -> None:
        """Scale the image to fit and recenter it."""
        self._auto_fit = True
        self._zoom = self.fit_zoom()
        self._pan = QPointF(0.0, 0.0)
        self.viewChanged.emit(self._zoom)
        self.update()

    def reset_view(self) -> None:
        """Show the image at 100% and recenter it."""
        self._auto_fit = False
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.viewChanged.emit(self._zoom)
        self.update()

    def zoom_by(self, factor: float, anchor: Optional[QPointF] = None) -> None:
        """Multiply the zoom, keeping the point under `anchor` in place."""
        if self._pixmap is None:
            return

        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-6:
            return

        if anchor is not None:
            before = self._widget_to_image(anchor)
            self._zoom = new_zoom
            after = self._widget_to_image(anchor)
            delta = (after - before) * self._zoom
            self._pan += QPointF(delta.x(), delta.y())
        else:
            self._zoom = new_zoom

        self._auto_fit = False
        self.viewChanged.emit(self._zoom)
        self.update()

    def zoom_in(self) -> None:
        """Zoom in one step."""
        self.zoom_by(1.25)

    def zoom_out(self) -> None:
        """Zoom out one step."""
        self.zoom_by(1 / 1.25)

    def resizeEvent(self, event) -> None:
        """Keep the image fitted while the user has not manually zoomed."""
        super().resizeEvent(event)
        if self._auto_fit and self._pixmap is not None:
            self._zoom = self.fit_zoom()
            self.viewChanged.emit(self._zoom)

    # ------------------------------------------------------------- mapping

    def _image_rect(self) -> QRectF:
        """Rectangle the image occupies in widget coordinates."""
        if self._pixmap is None:
            return QRectF()

        width = self._pixmap.width() * self._zoom
        height = self._pixmap.height() * self._zoom
        left = (self.width() - width) / 2.0 + self._pan.x()
        top = (self.height() - height) / 2.0 + self._pan.y()

        return QRectF(left, top, width, height)

    def _widget_to_image(self, point: QPointF) -> QPointF:
        """Convert widget coordinates to image pixel coordinates."""
        rect = self._image_rect()
        if rect.isEmpty() or self._zoom <= 0:
            return QPointF()

        return QPointF((point.x() - rect.left()) / self._zoom,
                       (point.y() - rect.top()) / self._zoom)

    def _norm_to_widget(self, norm: np.ndarray) -> QPointF:
        """Convert a normalised cover point to widget coordinates."""
        rect = self._image_rect()

        return QPointF(rect.left() + float(norm[0]) * rect.width(),
                       rect.top() + float(norm[1]) * rect.height())

    def _widget_to_norm(self, point: QPointF) -> np.ndarray:
        """Convert widget coordinates to normalised image coordinates."""
        rect = self._image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        return np.array([(point.x() - rect.left()) / rect.width(),
                         (point.y() - rect.top()) / rect.height()],
                        dtype=np.float32)

    def _handle_at(self, point: QPointF) -> int:
        """Nearest wrap-corner handle within the hit radius, or -1."""
        if self._mesh is None or not self._edit_cover:
            return -1

        best_index = -1
        best_distance = (self.HIT_RADIUS * 1.35) ** 2

        for index in self._corner_indices():
            handle = self._norm_to_widget(self._mesh[index])
            distance = (handle.x() - point.x()) ** 2 + (handle.y() - point.y()) ** 2
            if distance <= best_distance:
                best_distance = distance
                best_index = index

        return best_index

    def _quad_contains(self, point: QPointF) -> bool:
        """Whether a widget point falls inside the 4-corner wrap quad."""
        corners = self._corner_indices()
        if self._mesh is None or len(corners) < 4:
            return False

        path = QPainterPath()
        path.moveTo(self._norm_to_widget(self._mesh[corners[0]]))
        for index in corners[1:]:
            path.lineTo(self._norm_to_widget(self._mesh[index]))
        path.closeSubpath()

        return path.contains(point)

    def _boundary_indices(self) -> List[int]:
        """Clockwise mesh perimeter indices."""
        if self._mesh is None or self._mesh_rows < 2 or self._mesh_cols < 2:
            return []
        top = list(range(self._mesh_cols))
        right = [
            row * self._mesh_cols + self._mesh_cols - 1
            for row in range(1, self._mesh_rows)
        ]
        bottom = [
            (self._mesh_rows - 1) * self._mesh_cols + col
            for col in range(self._mesh_cols - 2, -1, -1)
        ]
        left = [
            row * self._mesh_cols
            for row in range(self._mesh_rows - 2, 0, -1)
        ]
        return top + right + bottom + left

    # -------------------------------------------------------------- painting

    def paintEvent(self, event) -> None:
        """Draw the background, the image and the cover overlay."""
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing
                               | QPainter.SmoothPixmapTransform)

        if self._pixmap is None:
            painter.fillRect(self.rect(), QColor(Palette.CANVAS_BG))
            self._paint_grid(painter)
            self._paint_placeholder(painter)
            painter.end()
            return

        # Studio-style workspace behind the product preview (single clean colour).
        painter.fillRect(self.rect(), QColor("#F2F3F5"))

        rect = self._image_rect()

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 22))
        painter.drawRoundedRect(rect.adjusted(2, 3, 2, 4), 4, 4)

        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))

        if self._show_cover and self._mesh is not None:
            self._paint_cover(painter)

        if self._brush_active():
            self._paint_erase_brush(painter)

        self._paint_zoom_badge(painter)
        painter.end()

    def _paint_grid(self, painter: QPainter) -> None:
        """Faint dot grid so the canvas reads as a workspace."""
        painter.setPen(QPen(QColor(Palette.GRID), 1))
        step = 28

        for x in range(step // 2, self.width(), step):
            for y in range(step // 2, self.height(), step):
                painter.drawPoint(x, y)

    def _paint_placeholder(self, painter: QPainter) -> None:
        """Dashed drop target with instructions."""
        margin = 32
        rect = QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)

        accent = QColor(Palette.ACCENT) if self._drag_hint else QColor(Palette.BORDER_STRONG)
        pen = QPen(accent, 2, Qt.DashLine)
        pen.setDashPattern([6, 5])
        painter.setPen(pen)
        painter.setBrush(QColor(255, 255, 255, 6) if self._drag_hint
                         else Qt.NoBrush)
        painter.drawRoundedRect(rect, 16, 16)

        icon_font = QFont(self.font())
        icon_font.setPointSize(34)
        painter.setFont(icon_font)
        painter.setPen(QColor(Palette.ACCENT if self._drag_hint else Palette.TEXT_DIM))
        icon_rect = QRectF(rect.left(), rect.center().y() - 92, rect.width(), 60)
        painter.drawText(icon_rect, Qt.AlignCenter, "\u25A2")

        title_font = QFont(self.font())
        title_font.setPointSize(13)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor(Palette.TEXT))
        painter.drawText(QRectF(rect.left(), rect.center().y() - 28,
                                rect.width(), 34),
                         Qt.AlignCenter, self.placeholder_title)

        body_font = QFont(self.font())
        body_font.setPointSize(9)
        painter.setFont(body_font)
        painter.setPen(QColor(Palette.TEXT_DIM))
        painter.drawText(QRectF(rect.left(), rect.center().y() + 6,
                                rect.width(), 60),
                         Qt.AlignHCenter | Qt.AlignTop, self.placeholder_body)

    def _paint_cover(self, painter: QPainter) -> None:
        """4-corner handles + smooth wrap outline; cutouts stay visible on top."""
        corners = self._corner_indices()
        if self._mesh is None or len(corners) < 4:
            return
        points = [self._norm_to_widget(self._mesh[i]) for i in corners]

        accent = QColor(Palette.ACCENT)
        dragging_mesh = self._drag_handle >= 0 or self._dragging_quad

        # Soft rounded perimeter from the dense warp mesh (product silhouette).
        boundary = self._boundary_indices()
        if len(boundary) >= 8:
            peri = QPainterPath()
            peri.moveTo(self._norm_to_widget(self._mesh[boundary[0]]))
            for index in boundary[1:]:
                peri.lineTo(self._norm_to_widget(self._mesh[index]))
            peri.closeSubpath()
            fill = QColor(accent)
            # Slightly lighter while dragging so cutouts stay readable.
            fill.setAlpha(18 if dragging_mesh else (28 if self._edit_cover else 14))
            painter.setBrush(QBrush(fill))
            painter.setPen(
                QPen(accent, 1.8 if self._edit_cover else 1.3, Qt.SolidLine)
            )
            painter.drawPath(peri)
        else:
            path = QPainterPath()
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.closeSubpath()
            fill = QColor(accent)
            fill.setAlpha(20 if dragging_mesh else (32 if self._edit_cover else 16))
            painter.setBrush(QBrush(fill))
            painter.setPen(
                QPen(accent, 2.0 if self._edit_cover else 1.4, Qt.SolidLine)
            )
            painter.drawPath(path)

        # Light quad guide so the 4 drag corners stay obvious.
        if self._edit_cover:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(accent, 1.1, Qt.DashLine))
            quad = QPainterPath()
            quad.moveTo(points[0])
            for point in points[1:]:
                quad.lineTo(point)
            quad.closeSubpath()
            painter.drawPath(quad)

            for index, point in zip(corners, points):
                active = index in (self._hover_handle, self._drag_handle)
                radius = self.HANDLE_RADIUS + (3 if active else 0)
                painter.setPen(QPen(accent, 2.4))
                painter.setBrush(
                    QColor("#FFFFFF")
                    if active
                    else QColor(Palette.BG_ELEVATED)
                )
                painter.drawEllipse(point, radius, radius)

        # Cutouts: transparent fill + dashed outline (never opaque black).
        # Stay visible while mesh corners are dragged so editing stays clear.
        if self._exclusion_contours:
            danger = QColor(Palette.DANGER)
            exclusion_pen = QPen(
                danger, 2.2 if dragging_mesh else 1.8, Qt.DashLine
            )
            painter.setPen(exclusion_pen)
            tint = QColor(Palette.DANGER)
            tint.setAlpha(28 if dragging_mesh else 22)
            painter.setBrush(tint)
            for c_idx, contour in enumerate(self._exclusion_contours):
                hole = self._exclusion_overlay_path(
                    contour, shape_tag=self._exclusion_shape_tag(c_idx)
                )
                if hole is not None:
                    painter.drawPath(hole)

            if self._edit_cover:
                for c_idx, contour in enumerate(self._exclusion_contours):
                    hovered = self._hover_exclusion[0] == c_idx
                    handle_pts = self._exclusion_box_handle_norms(contour)
                    for role, vertex in enumerate(handle_pts):
                        point = self._norm_to_widget(vertex)
                        active = (
                            self._hover_exclusion == (c_idx, role)
                            or (
                                self._drag_exclusion[0] == c_idx
                                and self._drag_exclusion_role == role
                            )
                        )
                        radius = 5.5 + (2.0 if active else 0.0)
                        painter.setPen(QPen(danger, 1.6))
                        painter.setBrush(
                            QColor("#FFFFFF") if active
                            else QColor(Palette.BG_ELEVATED)
                        )
                        painter.drawEllipse(point, radius, radius)

                    badge = self._exclusion_delete_badge_pos(contour)
                    bp = self._norm_to_widget(badge)
                    painter.setPen(QPen(QColor("#FFFFFF"), 1.2))
                    painter.setBrush(QColor(Palette.DANGER))
                    painter.drawEllipse(bp, 7.0, 7.0)
                    painter.setPen(QPen(QColor("#FFFFFF"), 1.6))
                    painter.drawLine(
                        QPointF(bp.x() - 3.0, bp.y() - 3.0),
                        QPointF(bp.x() + 3.0, bp.y() + 3.0),
                    )
                    painter.drawLine(
                        QPointF(bp.x() + 3.0, bp.y() - 3.0),
                        QPointF(bp.x() - 3.0, bp.y() + 3.0),
                    )
                    if hovered:
                        center = contour.mean(axis=0)
                        cp = self._norm_to_widget(center)
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(danger)
                        painter.drawEllipse(cp, 2.5, 2.5)

    def _exclusion_overlay_path(
        self,
        contour: np.ndarray,
        *,
        shape_tag: str = "",
    ) -> Optional[QPainterPath]:
        """Widget-space path for a cutout (true circle / rounded rect / poly)."""
        if contour is None or len(contour) < 3:
            return None
        pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            return None
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        bw = float(maxs[0] - mins[0])
        bh = float(maxs[1] - mins[1])
        tag = (shape_tag or "").lower().strip()
        path = QPainterPath()

        # Prefer the locked editor tag so resize never "looks" like a morph.
        if tag == "circle" and bw > 1e-6 and bh > 1e-6:
            center = 0.5 * (mins + maxs)
            c = self._norm_to_widget(center.astype(np.float32))
            # Locked 1:1 — use short side so the stroke stays circular.
            side = 0.5 * min(bw, bh)
            tip = self._norm_to_widget(
                np.array([center[0] + side, center[1]], dtype=np.float32)
            )
            radius_w = float(max(2.0, abs(tip.x() - c.x())))
            path.addEllipse(c, radius_w, radius_w)
            return path
        if tag in (
            "oval", "squircle", "superellipse",
            "pill_h", "pill_v", "capsule", "button",
            "rounded_square", "rounded_rect", "square", "rectangle",
        ) and bw > 1e-5 and bh > 1e-5:
            tl = self._norm_to_widget(mins.astype(np.float32))
            br = self._norm_to_widget(maxs.astype(np.float32))
            rect = QRectF(tl, br).normalized()
            short = min(rect.width(), rect.height())
            if tag == "square":
                path.addRect(rect)
                return path
            if tag == "rectangle":
                # Mild "halka" round so the red overlay matches painted hole.
                rad = short * float(
                    np.clip(self._cutout_corner_frac, 0.08, 0.22)
                )
                path.addRoundedRect(rect, rad, rad)
                return path
            if tag == "oval":
                path.addEllipse(rect)
                return path
            if tag in ("pill_h", "pill_v", "capsule", "button"):
                rad = 0.5 * short
            elif tag == "rounded_square":
                rad = short * float(self._cutout_corner_frac)
            elif tag in ("squircle", "superellipse"):
                rad = short * 0.42
            else:
                rad = short * float(self._cutout_corner_frac)
            path.addRoundedRect(rect, rad, rad)
            return path

        center = pts.mean(axis=0)
        radii = np.linalg.norm(pts - center, axis=1)
        r_mean = float(radii.mean()) if len(radii) else 0.0
        r_std = float(radii.std()) if len(radii) else 0.0
        if r_mean > 1e-6 and r_std / r_mean < 0.045 and len(pts) >= 12:
            c = self._norm_to_widget(center.astype(np.float32))
            tip = self._norm_to_widget(
                np.array([center[0] + r_mean, center[1]], dtype=np.float32)
            )
            radius_w = float(max(2.0, abs(tip.x() - c.x())))
            path.addEllipse(c, radius_w, radius_w)
            return path
        if bw > 1e-5 and bh > 1e-5:
            rect_area = bw * bh
            area = 0.5 * abs(
                np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
                - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))
            )
            fill = float(area / max(rect_area, 1e-8))
            aspect = max(bw, bh) / max(min(bw, bh), 1e-8)
            if fill >= 0.72 and aspect < 2.8 and len(pts) >= 8:
                tl = self._norm_to_widget(mins.astype(np.float32))
                br = self._norm_to_widget(maxs.astype(np.float32))
                rect = QRectF(tl, br).normalized()
                rad = min(rect.width(), rect.height()) * (
                    0.30 if aspect < 1.35 else 0.45
                )
                path.addRoundedRect(rect, rad, rad)
                return path
        path.moveTo(self._norm_to_widget(pts[0].astype(np.float32)))
        for p in pts[1:]:
            path.lineTo(self._norm_to_widget(p.astype(np.float32)))
        path.closeSubpath()
        return path

    def _paint_zoom_badge(self, painter: QPainter) -> None:
        """Small zoom readout in the corner of the canvas."""
        text = f"{self._zoom * 100:.0f}%"
        font = QFont(self.font())
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 18
        rect = QRectF(14, self.height() - 34, width, 22)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(rect, 8, 8)

        painter.setPen(QColor(Palette.TEXT_MUTED))
        painter.drawText(rect, Qt.AlignCenter, text)

    def _paint_erase_brush(self, painter: QPainter) -> None:
        """Draw a thin selection stroke + live brush tip (Erase / Fill)."""
        is_fill = self._brush_kind == "fill"
        line = QColor(Palette.TEAL if is_fill else Palette.DANGER)
        tip = QColor(line)
        tip.setAlpha(220)

        # Thin path through dab centres — clear select line, not filled blobs.
        if len(self._brush_stroke) >= 1:
            path = QPainterPath()
            first = True
            for nx, ny, _r_norm in self._brush_stroke:
                pt = self._norm_to_widget(np.array([nx, ny], dtype=np.float32))
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
            painter.setBrush(Qt.NoBrush)
            glow = QColor(line)
            glow.setAlpha(90)
            painter.setPen(QPen(glow, 3.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)
            painter.setPen(QPen(line, 1.25, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)
            # Tiny node at the start so a single click still shows selection.
            start = self._norm_to_widget(
                np.array(
                    [self._brush_stroke[0][0], self._brush_stroke[0][1]],
                    dtype=np.float32,
                )
            )
            painter.setPen(QPen(line, 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(start, 2.5, 2.5)

        if self._brush_cursor is not None:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(tip, 1.4))
            painter.drawEllipse(
                self._brush_cursor, self._brush_radius_px, self._brush_radius_px
            )
            painter.setPen(QPen(QColor("#FFFFFF"), 1.0))
            painter.drawEllipse(
                self._brush_cursor,
                max(2.0, self._brush_radius_px - 1.5),
                max(2.0, self._brush_radius_px - 1.5),
            )
            # Crosshair centre for precise Final work.
            cx = self._brush_cursor.x()
            cy = self._brush_cursor.y()
            painter.setPen(QPen(tip, 1.0))
            painter.drawLine(QPointF(cx - 4, cy), QPointF(cx + 4, cy))
            painter.drawLine(QPointF(cx, cy - 4), QPointF(cx, cy + 4))

    def _norm_radius_to_widget(self, radius_norm: float) -> float:
        """Convert a normalised radius (fraction of image min side) to widgets."""
        rect = self._image_rect()
        return float(radius_norm) * float(min(rect.width(), rect.height()))

    def _widget_radius_to_norm(self, radius_px: float) -> float:
        """Convert widget-pixel brush radius to normalised image radius."""
        rect = self._image_rect()
        denom = max(float(min(rect.width(), rect.height())), 1.0)
        return float(radius_px) / denom

    def _append_brush_dab(self, position: QPointF) -> None:
        """Record one erase/fill dab at the widget position."""
        norm = self._widget_to_norm(position)
        r_norm = self._widget_radius_to_norm(self._brush_radius_px)
        nx, ny = float(norm[0]), float(norm[1])
        # Wider spacing → fewer overlapping stamps (was over-erasing).
        spacing = 0.55 if self._final_mode else 0.4
        if self._brush_stroke:
            lx, ly, lr = self._brush_stroke[-1]
            if ((nx - lx) ** 2 + (ny - ly) ** 2) ** 0.5 < max(lr, r_norm) * spacing:
                return
        self._brush_stroke.append((nx, ny, float(r_norm)))

    # ------------------------------------------------------------ mouse/keys

    def wheelEvent(self, event) -> None:
        """Zoom view, design (Move Design), or brush size (Alt)."""
        if self._pixmap is None:
            return

        steps = event.angleDelta().y() / 120.0
        if abs(steps) < 1e-3:
            return

        if self._brush_active() and (
            event.modifiers() & Qt.AltModifier
        ):
            self.set_brush_radius_px(self._brush_radius_px * (1.12 ** steps))
            return

        # Move Design: scroll zooms the print; Ctrl+scroll still zooms the view.
        if (
            self._design_pan_mode
            and not self._edit_cover
            and not (event.modifiers() & Qt.ControlModifier)
        ):
            factor = 1.08 ** steps
            self.designZoomDelta.emit(float(factor))
            return

        self.zoom_by(1.15 ** steps, QPointF(event.position()))

    def mousePressEvent(self, event) -> None:
        """Start dragging a handle, cutout, mesh, erase stroke, design, or view."""
        if event.button() == Qt.MiddleButton:
            self._begin_pan(event)
            return

        if self._pixmap is None:
            return

        position = QPointF(event.position())
        self._press_pos = event.position().toPoint()
        self._moved = False

        # Move Design: drag artwork left/right/up/down (not the view).
        if (
            event.button() == Qt.LeftButton
            and self._design_pan_mode
            and not self._edit_cover
            and not self._brush_active()
        ):
            self._design_panning = True
            self._design_pan_last = position
            self.setCursor(Qt.ClosedHandCursor)
            return

        # Final / Erase Wrap brush — paint under the cursor.
        if (
            event.button() == Qt.LeftButton
            and self._brush_active()
        ):
            self._brush_painting = True
            self._brush_stroke = []
            self._brush_cursor = position
            self._append_brush_dab(position)
            self.update()
            return

        # Right-click: only the × badge removes a cutout (never the body —
        # accidental right-clicks were wiping hard-edited shapes).
        if event.button() == Qt.RightButton and self._edit_cover:
            if self._exclusion_delete_badge_at(position) >= 0:
                self._remove_exclusion_at(position, badge_only=True)
            return

        if event.button() != Qt.LeftButton:
            return

        # Shift+click: add cutout in the selected shape.
        if self._edit_cover and event.modifiers() & Qt.ShiftModifier:
            self._add_shaped_exclusion(self._widget_to_norm(position))
            return

        # Ctrl+click on a cutout edge: insert a finishing dot (wrap stays 4-corner).
        if self._edit_cover and event.modifiers() & Qt.ControlModifier:
            if self._insert_exclusion_vertex_at(position):
                return
            return

        # × badge only (small hit target). Alt+click still removes intentionally.
        if self._edit_cover and self._exclusion_delete_badge_at(position) >= 0:
            self._remove_exclusion_at(position, badge_only=True)
            return
        if self._edit_cover and event.modifiers() & Qt.AltModifier:
            vertex = self._exclusion_vertex_at(position)
            if vertex[0] >= 0 and vertex[1] >= 0:
                self._remove_exclusion_vertex(vertex[0], vertex[1])
                return
            if self._remove_exclusion_at(position, badge_only=True):
                return
            # Alt+click on body: explicit delete.
            if self._remove_exclusion_at(position, badge_only=False):
                return

        # Camera cutouts take priority over mesh points near lenses.
        if self._edit_cover:
            handle = self._exclusion_handle_role_at(position)
            if handle[0] >= 0 and handle[1] >= 0:
                self._lock_exclusion_shape(handle[0])
                self._drag_exclusion = handle
                self._drag_exclusion_role = int(handle[1])
                self._exclusion_drag_origin = self._exclusion_contours[
                    handle[0]
                ].copy()
                self.setCursor(Qt.ClosedHandCursor)
                return
            body = self._exclusion_body_at(position)
            if body >= 0:
                self._lock_exclusion_shape(body)
                self._drag_exclusion_body = body
                self._exclusion_drag_origin = self._exclusion_contours[
                    body
                ].copy()
                self._hover_exclusion = (body, -1)
                self.setCursor(Qt.SizeAllCursor)
                return

        handle = self._handle_at(position)
        if handle >= 0:
            self._drag_handle = handle
            self._drag_origin = self._mesh.copy()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if self._edit_cover and self._quad_contains(position):
            self._dragging_quad = True
            self._drag_origin = self._mesh.copy()
            self.setCursor(Qt.SizeAllCursor)
            return

        self._begin_pan(event)

    def _begin_pan(self, event) -> None:
        """Start panning the view."""
        self._panning = True
        self._press_pos = event.position().toPoint()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        """Update handle drag, cutout drag, erase brush, pan, or hover."""
        position = QPointF(event.position())

        if (position.toPoint() - self._press_pos).manhattanLength() > 3:
            self._moved = True

        if self._design_panning and self._design_pan_last is not None:
            target = self._image_rect()
            tw = max(float(target.width()), 1.0)
            th = max(float(target.height()), 1.0)
            dx = (position.x() - self._design_pan_last.x()) / tw
            dy = (position.y() - self._design_pan_last.y()) / th
            self._design_pan_last = position
            # Emit even tiny moves — main window coalesces for smoothness.
            if abs(dx) > 1e-7 or abs(dy) > 1e-7:
                self.designPanDelta.emit(float(dx), float(dy))
            return

        if self._brush_active():
            self._brush_cursor = position
            if self._brush_painting:
                self._append_brush_dab(position)
            self.update()
            return

        if self._drag_handle >= 0 and self._mesh is not None:
            norm = self._widget_to_norm(position)
            self._mesh[self._drag_handle] = np.clip(norm, -0.5, 1.5)
            # Keep dense warp mesh in sync with the dragged corner.
            self._rebuild_mesh_from_corners()
            self.update()
            return

        if self._drag_exclusion[0] >= 0 and self._drag_exclusion_role >= 0:
            c_idx = self._drag_exclusion[0]
            role = int(self._drag_exclusion_role)
            if 0 <= c_idx < len(self._exclusion_contours):
                contour = self._exclusion_contours[c_idx]
                norm = np.clip(self._widget_to_norm(position), -0.05, 1.05)
                # Shift+drag a handle = uniform scale from centre (shrink/grow).
                if event.modifiers() & Qt.ShiftModifier and len(contour) >= 3:
                    center = contour.mean(axis=0)
                    handles = self._exclusion_box_handle_norms(contour)
                    old = handles[role] if role < len(handles) else contour.mean(axis=0)
                    old_r = float(np.linalg.norm(old - center))
                    new_r = float(np.linalg.norm(norm - center))
                    if old_r > 1e-5 and new_r > 1e-6:
                        factor = float(np.clip(new_r / old_r, 0.35, 2.8))
                        scaled = center + (contour - center) * factor
                        self._exclusion_contours[c_idx] = (
                            self._normalize_exclusion_contour(
                                scaled,
                                shape_tag=self._exclusion_shape_tag(c_idx),
                            )
                        )
                    self.update()
                    return
                self._exclusion_contours[c_idx] = self._resize_exclusion_box(
                    contour, role, norm,
                    shape_tag=self._exclusion_shape_tag(c_idx),
                )
                self._drag_exclusion = (c_idx, role)
                self.update()
            return

        if self._drag_exclusion_body >= 0 and self._exclusion_drag_origin is not None:
            start = self._widget_to_norm(QPointF(self._press_pos))
            cur = self._widget_to_norm(position)
            # Shift+drag body = scale from centre; plain drag = move.
            if event.modifiers() & Qt.ShiftModifier:
                origin = self._exclusion_drag_origin
                center = origin.mean(axis=0)
                ref = float(np.linalg.norm(start - center))
                now = float(np.linalg.norm(cur - center))
                if ref > 1e-5 and now > 1e-6:
                    factor = float(np.clip(now / ref, 0.35, 2.8))
                    scaled = center + (origin - center) * factor
                    self._exclusion_contours[self._drag_exclusion_body] = (
                        self._normalize_exclusion_contour(
                            scaled,
                            shape_tag=self._exclusion_shape_tag(
                                self._drag_exclusion_body
                            ),
                        )
                    )
                self.update()
                return
            delta = cur - start
            moved = np.clip(
                self._exclusion_drag_origin + delta, -0.05, 1.05
            )
            self._exclusion_contours[self._drag_exclusion_body] = (
                self._normalize_exclusion_contour(
                    moved,
                    shape_tag=self._exclusion_shape_tag(
                        self._drag_exclusion_body
                    ),
                )
            )
            self.update()
            return

        if self._dragging_quad and self._drag_origin is not None:
            start = self._widget_to_norm(QPointF(self._press_pos))
            delta = self._widget_to_norm(position) - start
            self._mesh = np.clip(
                self._drag_origin + delta, -0.5, 1.5
            )
            self.update()
            return

        if self._panning:
            delta = position - QPointF(self._press_pos)
            self._pan += delta
            self._press_pos = position.toPoint()
            self._auto_fit = False
            self.update()
            return

        hover = self._handle_at(position)
        excl_hover = (-1, -1)
        if self._edit_cover:
            badge = self._exclusion_delete_badge_at(position)
            if badge >= 0:
                excl_hover = (badge, -2)
            else:
                handle = self._exclusion_handle_role_at(position)
                if handle[0] >= 0:
                    excl_hover = handle
                else:
                    body = self._exclusion_body_at(position)
                    if body >= 0:
                        excl_hover = (body, -1)
            if excl_hover[0] >= 0:
                hover = -1
        if hover != self._hover_handle or excl_hover != self._hover_exclusion:
            self._hover_handle = hover
            self._hover_exclusion = excl_hover
            active = hover >= 0 or excl_hover[0] >= 0
            self.setCursor(
                Qt.PointingHandCursor if active
                else (Qt.OpenHandCursor if self._pixmap is not None
                      else Qt.PointingHandCursor)
            )
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        """Finish the current interaction and notify listeners."""
        if self._brush_painting and self._brush_stroke:
            stroke = list(self._brush_stroke)
            self._brush_painting = False
            self._brush_stroke = []
            self.exclusionBrushStroke.emit(stroke)
            self._refresh_brush_cursor()
            self.update()
            return

        was_design_pan = self._design_panning

        mesh_changed = self._drag_handle >= 0 or self._dragging_quad
        exclusion_changed = (
            self._drag_exclusion[0] >= 0 or self._drag_exclusion_body >= 0
        )
        drag_origin = (
            None if self._drag_origin is None else self._drag_origin.copy()
        )
        mesh_now = None if self._mesh is None else self._mesh.copy()

        self._drag_handle = -1
        self._dragging_quad = False
        self._drag_exclusion = (-1, -1)
        self._drag_exclusion_body = -1
        self._drag_exclusion_role = -1
        self._panning = False
        self._design_panning = False
        self._design_pan_last = None
        self._drag_origin = None
        self._exclusion_drag_origin = None
        self._exclusion_scale_center = None
        self._exclusion_scale_ref = 1.0
        if self._brush_active():
            self.setCursor(Qt.BlankCursor)
        elif self._design_pan_mode and not self._edit_cover:
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(
                Qt.OpenHandCursor if self._pixmap is not None
                else Qt.PointingHandCursor
            )

        if was_design_pan:
            self.designPanFinished.emit()
            self.update()
            return

        mesh_really_moved = self._moved
        if (
            mesh_changed
            and mesh_now is not None
            and drag_origin is not None
            and drag_origin.shape == mesh_now.shape
            and not np.allclose(mesh_now, drag_origin, atol=1e-4)
        ):
            mesh_really_moved = True

        if mesh_changed and mesh_now is not None and mesh_really_moved:
            self.meshPointsChanged.emit(
                mesh_now, self._mesh_rows, self._mesh_cols
            )
            self.coverPointsChanged.emit(self.cover_points())

        if exclusion_changed and self._moved:
            # Commit after a real drag — handles stay where the user left them.
            self.exclusionContoursChanged.emit(self.exclusion_contours())
        elif self._pixmap is None and not self._moved and event.button() == Qt.LeftButton:
            self.browseRequested.emit()

        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        """Fit the view — cutouts stay 4-corner editable (no extra dots)."""
        if self._pixmap is None or event.button() != Qt.LeftButton:
            return
        self.fit_to_view()

    def keyPressEvent(self, event) -> None:
        """Arrow keys nudge; Delete removes; [ ] resize erase/fill brush."""
        if self._brush_active() and event.key() in (
            Qt.Key_BracketLeft, Qt.Key_BracketRight
        ):
            factor = 0.85 if event.key() == Qt.Key_BracketLeft else 1.18
            self.set_brush_radius_px(self._brush_radius_px * factor)
            return

        if not self._edit_cover:
            super().keyPressEvent(event)
            return

        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            c_idx, v_idx = self._hover_exclusion
            if c_idx >= 0:
                # Whole cutout only — vertex delete reintroduced the dot mess.
                del self._exclusion_contours[c_idx]
                if c_idx < len(self._exclusion_shapes):
                    del self._exclusion_shapes[c_idx]
                self._hover_exclusion = (-1, -1)
                self.exclusionContoursChanged.emit(self.exclusion_contours())
                self.update()
                return
            # Do not remove wrap corners — always keep the 4-point frame.
            return

        if self._hover_exclusion[0] >= 0:
            c_idx, v_idx = self._hover_exclusion
            rect = self._image_rect()
            step = 1.0 / max(rect.width(), 1.0)
            offsets = {
                Qt.Key_Left: (-step, 0.0), Qt.Key_Right: (step, 0.0),
                Qt.Key_Up: (0.0, -step), Qt.Key_Down: (0.0, step),
            }
            offset = offsets.get(event.key())
            if offset is not None:
                multiplier = 10.0 if event.modifiers() & Qt.ShiftModifier else 1.0
                delta = np.array(offset, dtype=np.float32) * multiplier
                if v_idx >= 0:
                    self._exclusion_contours[c_idx][v_idx] += delta
                else:
                    self._exclusion_contours[c_idx] = (
                        self._exclusion_contours[c_idx] + delta
                    )
                self.exclusionContoursChanged.emit(self.exclusion_contours())
                self.update()
                return

        if self._mesh is None or self._hover_handle < 0:
            super().keyPressEvent(event)
            return

        rect = self._image_rect()
        step = 1.0 / max(rect.width(), 1.0)
        offsets = {
            Qt.Key_Left: (-step, 0.0), Qt.Key_Right: (step, 0.0),
            Qt.Key_Up: (0.0, -step), Qt.Key_Down: (0.0, step),
        }

        offset = offsets.get(event.key())
        if offset is None:
            super().keyPressEvent(event)
            return

        multiplier = 10.0 if event.modifiers() & Qt.ShiftModifier else 1.0
        self._mesh[self._hover_handle] += (
            np.array(offset, dtype=np.float32) * multiplier
        )
        self._rebuild_mesh_from_corners()
        self.meshPointsChanged.emit(
            self._mesh.copy(), self._mesh_rows, self._mesh_cols
        )
        self.coverPointsChanged.emit(self.cover_points())
        self.update()

    def _exclusion_handle_indices(self, count: int) -> List[int]:
        """Legacy fallback — prefer _exclusion_corner_handles."""
        if count <= 0:
            return []
        if count <= 4:
            return list(range(count))
        step = count / 4.0
        return sorted({int(i * step) % count for i in range(4)})

    def _exclusion_box_handle_norms(self, contour: np.ndarray) -> List[np.ndarray]:
        """Eight AABB handle positions in normalised space (corners + edge mids)."""
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(pts) == 0:
            return []
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        return [
            np.array([x1, y1], dtype=np.float32),
            np.array([x2, y1], dtype=np.float32),
            np.array([x2, y2], dtype=np.float32),
            np.array([x1, y2], dtype=np.float32),
            np.array([mx, y1], dtype=np.float32),
            np.array([x2, my], dtype=np.float32),
            np.array([mx, y2], dtype=np.float32),
            np.array([x1, my], dtype=np.float32),
        ]

    def _exclusion_corner_handles(self, contour: np.ndarray) -> List[int]:
        """
        Map the 8 box handles to nearest contour verts (for legacy callers).

        Prefer `_exclusion_box_handle_norms` + role indices for hit-test/drag.
        """
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(pts) == 0:
            return []
        targets = self._exclusion_box_handle_norms(contour)
        used = set()
        indices: List[int] = []
        for target in targets:
            tx, ty = float(target[0]), float(target[1])
            best_i = 0
            best_d = 1e18
            for i, p in enumerate(pts):
                if i in used and len(pts) >= len(targets):
                    continue
                d = (float(p[0]) - tx) ** 2 + (float(p[1]) - ty) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            used.add(best_i)
            indices.append(best_i)
        return indices

    def _exclusion_handle_role_at(self, position: QPointF) -> tuple:
        """Hit-test AABB box handles. Returns (contour_idx, role) or (-1, -1)."""
        best = (-1, -1)
        best_dist = self.EXCLUSION_HIT_RADIUS
        for c_idx, contour in enumerate(self._exclusion_contours):
            for role, target in enumerate(self._exclusion_box_handle_norms(contour)):
                point = self._norm_to_widget(target)
                dist = (
                    (point.x() - position.x()) ** 2
                    + (point.y() - position.y()) ** 2
                ) ** 0.5
                if dist <= best_dist:
                    best_dist = dist
                    best = (c_idx, role)
        return best

    def _scale_contour_to_box(
        self,
        contour: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> np.ndarray:
        """Scale/translate polygon verts so their AABB matches the target box."""
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2).copy()
        if len(pts) < 3:
            return pts
        ox1, oy1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        ox2, oy2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        ow = max(ox2 - ox1, 1e-6)
        oh = max(oy2 - oy1, 1e-6)
        nw = max(float(x2) - float(x1), 1e-6)
        nh = max(float(y2) - float(y1), 1e-6)
        out = np.empty_like(pts)
        out[:, 0] = float(x1) + (pts[:, 0] - ox1) / ow * nw
        out[:, 1] = float(y1) + (pts[:, 1] - oy1) / oh * nh
        return out

    def _resize_exclusion_box(
        self,
        contour: np.ndarray,
        corner_role: int,
        cursor: np.ndarray,
        *,
        shape_tag: str = "",
    ) -> np.ndarray:
        """
        Resize a cutout like Canva: drag one handle, opposite edge/corner fixed.

        Full independent width × height for all shapes except:
        - circle → locked 1:1
        - square → locked 1:1
        Edge mids (roles 4–7) change only one axis for easy wide/tall control.
        corner_role: 0=TL, 1=TR, 2=BR, 3=BL, 4=TM, 5=RM, 6=BM, 7=LM
        """
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        cx = float(np.clip(cursor[0], -0.05, 1.05))
        cy = float(np.clip(cursor[1], -0.05, 1.05))
        min_size = 0.012
        tag = (shape_tag or "").lower().strip()
        role = int(corner_role)

        if tag in ("circle", "square"):
            # Opposite corner fixed; side follows the dominant cursor delta.
            # Map edge roles to nearest corner for 1:1 shapes.
            corner = role if role < 4 else {4: 0, 5: 1, 6: 2, 7: 3}[role]
            if corner == 0:  # TL → BR fixed
                side = max(min_size, max(x2 - cx, y2 - cy))
                x1, y1 = x2 - side, y2 - side
            elif corner == 1:  # TR → BL fixed
                side = max(min_size, max(cx - x1, y2 - cy))
                x2, y1 = x1 + side, y2 - side
            elif corner == 2:  # BR → TL fixed
                side = max(min_size, max(cx - x1, cy - y1))
                x2, y2 = x1 + side, y1 + side
            else:  # BL → TR fixed
                side = max(min_size, max(x2 - cx, cy - y1))
                x1, y2 = x2 - side, y1 + side
        elif role == 4:  # top mid — height only
            y1 = min(cy, y2 - min_size)
        elif role == 5:  # right mid — width only
            x2 = max(cx, x1 + min_size)
        elif role == 6:  # bottom mid — height only
            y2 = max(cy, y1 + min_size)
        elif role == 7:  # left mid — width only
            x1 = min(cx, x2 - min_size)
        elif role == 0:  # TL
            x1, y1 = min(cx, x2 - min_size), min(cy, y2 - min_size)
        elif role == 1:  # TR
            x2, y1 = max(cx, x1 + min_size), min(cy, y2 - min_size)
        elif role == 2:  # BR
            x2, y2 = max(cx, x1 + min_size), max(cy, y1 + min_size)
        else:  # BL
            x1, y2 = min(cx, x2 - min_size), max(cy, y1 + min_size)

        if tag in ("custom_path", "free", "triangle", "polygon"):
            return np.clip(
                self._scale_contour_to_box(pts, x1, y1, x2, y2),
                -0.05,
                1.05,
            )
        box = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )
        return self._normalize_exclusion_contour(box, shape_tag=shape_tag)

    def _infer_exclusion_shape_tag(self, contour: np.ndarray) -> str:
        """
        Lock a stable editor shape for untagged cutouts (auto-detect / sync).

        Called once when the tag is empty so resize never re-classifies and
        morphs circle ↔ stadium ↔ rounded_rect mid-drag.
        """
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return "rounded_rect"
        width = float(pts[:, 0].max() - pts[:, 0].min())
        height = float(pts[:, 1].max() - pts[:, 1].min())
        aspect = width / max(height, 1e-6)
        center = pts.mean(axis=0)
        radii = np.linalg.norm(pts - center, axis=1)
        r_mean = float(radii.mean()) if len(radii) else 0.0
        r_std = float(radii.std()) if len(radii) else 0.0
        roundish = r_mean > 1e-6 and (r_std / r_mean) < 0.07

        kind = ""
        try:
            from ..image_processing.region_detector import HardwareRegionDetector

            kind, _params = HardwareRegionDetector._classify_cutout(pts)
        except Exception:
            kind = ""

        if kind == "circle" or (roundish and 0.85 <= aspect <= 1.15):
            return "circle"
        if kind == "stadium":
            # Preserve capsule/pill — never collapse to mild rounded_rect.
            return "pill_h" if width >= height else "pill_v"
        if kind == "rounded_rect":
            # Dense stadium polylines can classify as RR after mild-corner
            # changes — detect near-full end radius and keep as capsule.
            try:
                from ..image_processing.region_detector import HardwareRegionDetector

                _k, params = HardwareRegionDetector._classify_cutout(pts)
                if _k == "rounded_rect" and params and len(params) >= 5:
                    short = min(
                        float(params[2] - params[0]),
                        float(params[3] - params[1]),
                    )
                    if float(params[4]) >= short * 0.40:
                        return "pill_h" if width >= height else "pill_v"
            except Exception:
                pass
            return "rounded_rect"
        if aspect >= 1.35:
            return "pill_h"
        if aspect <= (1.0 / 1.35):
            return "pill_v"
        return "rounded_rect"

    def _exclusion_shape_tag(self, c_idx: int) -> str:
        """Return locked shape for a cutout; infer+store once if missing."""
        shapes = getattr(self, "_exclusion_shapes", [])
        while len(shapes) < len(self._exclusion_contours):
            shapes.append("")
        self._exclusion_shapes = shapes
        if not (0 <= c_idx < len(shapes)):
            return ""
        tag = str(shapes[c_idx] or "").strip()
        if tag:
            return tag
        if 0 <= c_idx < len(self._exclusion_contours):
            tag = self._infer_exclusion_shape_tag(self._exclusion_contours[c_idx])
            self._exclusion_shapes[c_idx] = tag
            return tag
        return ""

    def _lock_exclusion_shape(self, c_idx: int) -> str:
        """Ensure shape tag is frozen before a drag/resize starts."""
        return self._exclusion_shape_tag(c_idx)

    def _exclusion_vertex_at(self, position: QPointF) -> tuple:
        """Nearest actual contour vertex (for Alt-remove / insert), not box handles."""
        best = (-1, -1)
        best_dist = self.EXCLUSION_HIT_RADIUS
        for c_idx, contour in enumerate(self._exclusion_contours):
            for v_idx, vertex in enumerate(contour):
                point = self._norm_to_widget(vertex)
                dist = (
                    (point.x() - position.x()) ** 2
                    + (point.y() - position.y()) ** 2
                ) ** 0.5
                if dist <= best_dist:
                    best_dist = dist
                    best = (c_idx, v_idx)
        return best

    def _exclusion_delete_badge_pos(self, contour: np.ndarray) -> np.ndarray:
        """Normalised position of the remove (×) badge for a cutout."""
        mins = contour.min(axis=0)
        maxs = contour.max(axis=0)
        return np.array(
            [float(maxs[0]), float(mins[1])], dtype=np.float32
        )

    def _exclusion_delete_badge_at(self, position: QPointF) -> int:
        """Cutout index whose × badge is under the cursor, else -1."""
        for index in range(len(self._exclusion_contours) - 1, -1, -1):
            badge = self._exclusion_delete_badge_pos(
                self._exclusion_contours[index]
            )
            point = self._norm_to_widget(badge)
            dist = (
                (point.x() - position.x()) ** 2
                + (point.y() - position.y()) ** 2
            ) ** 0.5
            if dist <= 8.0:
                return index
        return -1

    def _remove_exclusion_at(
        self, position: QPointF, *, badge_only: bool = False
    ) -> bool:
        """Remove a cutout. Default: × badge only — body clicks never delete."""
        index = self._exclusion_delete_badge_at(position)
        if index < 0 and not badge_only:
            index = self._exclusion_body_at(position)
        if index < 0 and not badge_only:
            hit = self._exclusion_vertex_at(position)
            if hit[0] >= 0:
                index = hit[0]
        if index < 0:
            return False
        del self._exclusion_contours[index]
        if index < len(self._exclusion_shapes):
            del self._exclusion_shapes[index]
        self._hover_exclusion = (-1, -1)
        self._drag_exclusion = (-1, -1)
        self._drag_exclusion_body = -1
        self.exclusionContoursChanged.emit(self.exclusion_contours())
        self.update()
        return True

    def _remove_exclusion_vertex(self, c_idx: int, v_idx: int) -> bool:
        """Remove one finishing dot from a cutout (keeps at least 3)."""
        if not (0 <= c_idx < len(self._exclusion_contours)):
            return False
        contour = self._exclusion_contours[c_idx]
        if len(contour) <= 3 or not (0 <= v_idx < len(contour)):
            return False
        tag = self._exclusion_shape_tag(c_idx)
        trimmed = np.delete(contour, v_idx, axis=0)
        if tag in ("custom_path", "free", "triangle", "polygon"):
            # Keep freeform verts — do not re-seed the shape.
            self._exclusion_contours[c_idx] = np.clip(trimmed, -0.05, 1.05)
        else:
            self._exclusion_contours[c_idx] = self._normalize_exclusion_contour(
                trimmed,
                shape_tag=tag,
            )
        self._hover_exclusion = (c_idx, -1)
        self.exclusionContoursChanged.emit(self.exclusion_contours())
        self.update()
        return True

    def _insert_exclusion_vertex_at(self, position: QPointF) -> bool:
        """Insert a finishing dot on the nearest cutout edge."""
        if not self._exclusion_contours:
            return False
        best_c, best_i, best_dist = -1, -1, 22.0
        best_point = None
        for c_idx, contour in enumerate(self._exclusion_contours):
            n = len(contour)
            if n >= 24:
                continue
            for i in range(n):
                a = contour[i]
                b = contour[(i + 1) % n]
                proj, dist_img = self._point_to_segment_widget(position, a, b)
                if dist_img <= best_dist:
                    best_dist = dist_img
                    best_c, best_i = c_idx, i
                    best_point = proj
        if best_c < 0 or best_point is None:
            return False
        hit = self._exclusion_vertex_at(position)
        if hit[0] == best_c and hit[1] >= 0:
            return False
        contour = self._exclusion_contours[best_c]
        inserted = np.insert(
            contour, best_i + 1, np.clip(best_point, -0.05, 1.05), axis=0
        )
        tag = self._exclusion_shape_tag(best_c)
        if tag in ("custom_path", "free", "triangle", "polygon"):
            self._exclusion_contours[best_c] = np.clip(inserted, -0.05, 1.05)
        else:
            self._exclusion_contours[best_c] = self._normalize_exclusion_contour(
                inserted,
                shape_tag=tag,
            )
        self._hover_exclusion = (best_c, best_i + 1)
        self.exclusionContoursChanged.emit(self.exclusion_contours())
        self.update()
        return True

    def _point_to_segment_widget(
        self, position: QPointF, a_norm: np.ndarray, b_norm: np.ndarray
    ):
        """Project a widget point onto a normalised segment; return norm + dist."""
        a = self._norm_to_widget(a_norm)
        b = self._norm_to_widget(b_norm)
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        px, py = position.x(), position.y()
        abx, aby = bx - ax, by - ay
        length2 = abx * abx + aby * aby
        if length2 < 1e-6:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length2))
        qx, qy = ax + t * abx, ay + t * aby
        dist = ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5
        proj = a_norm * (1.0 - t) + b_norm * t
        return proj.astype(np.float32), dist

    def _exclusion_body_at(self, position: QPointF) -> int:
        """Index of an exclusion polygon containing the cursor, else -1."""
        norm = self._widget_to_norm(position)
        for index in range(len(self._exclusion_contours) - 1, -1, -1):
            contour = self._exclusion_contours[index]
            if len(contour) < 3:
                continue
            if cv2.pointPolygonTest(
                contour.astype(np.float32),
                (float(norm[0]), float(norm[1])),
                False,
            ) >= 0:
                return index
            center = contour.mean(axis=0)
            radius = float(np.median(np.linalg.norm(contour - center, axis=1)))
            if np.linalg.norm(norm - center) <= max(radius * 1.1, 0.01):
                return index
        return -1

    def _add_shaped_exclusion(self, center: np.ndarray) -> None:
        """Insert a cutout polygon using the selected shape tool."""
        from ..image_processing.region_detector import HardwareRegionDetector

        size = 0.018
        shape = self._cutout_shape
        # Default aspect for creation only — pills/capsules start elongated.
        aspect = 1.0
        if shape in ("pill_h", "pill-horizontal", "capsule_h"):
            aspect = 2.4
        elif shape in ("pill_v", "pill-vertical", "capsule_v"):
            aspect = 1.0 / 2.4
        elif shape in ("capsule", "button"):
            aspect = 1.0 / 3.0
        poly = HardwareRegionDetector.make_shape_polygon(
            shape,
            (float(center[0]), float(center[1])),
            size,
            aspect=aspect,
            corner_frac=self._cutout_corner_frac,
            rotation_deg=self._cutout_rotation_deg,
        ).reshape(-1, 2)
        self._exclusion_contours.append(
            self._normalize_exclusion_contour(poly, shape_tag=shape)
        )
        while len(self._exclusion_shapes) < len(self._exclusion_contours) - 1:
            self._exclusion_shapes.append("")
        self._exclusion_shapes.append(shape)
        self._hover_exclusion = (len(self._exclusion_contours) - 1, 0)
        self.exclusionContoursChanged.emit(self.exclusion_contours())
        self.update()

    def apply_cutout_style(
        self,
        *,
        corner_frac: float = 0.28,
        rotation_deg: float = 0.0,
    ) -> bool:
        """Rebuild hovered/last cutout with corner radius + rotation."""
        if not self._exclusion_contours:
            return False
        c_idx = int(self._hover_exclusion[0])
        if not (0 <= c_idx < len(self._exclusion_contours)):
            c_idx = len(self._exclusion_contours) - 1
        while len(self._exclusion_shapes) < len(self._exclusion_contours):
            self._exclusion_shapes.append(self._cutout_shape)
        shape = self._exclusion_shapes[c_idx] or self._cutout_shape
        self._cutout_corner_frac = float(np.clip(corner_frac, 0.0, 0.5))
        self._cutout_rotation_deg = float(rotation_deg)
        self._exclusion_contours[c_idx] = self._normalize_exclusion_contour(
            self._exclusion_contours[c_idx],
            shape_tag=shape,
            corner_frac=self._cutout_corner_frac,
            rotation_deg=self._cutout_rotation_deg,
        )
        self._exclusion_shapes[c_idx] = shape
        self.exclusionContoursChanged.emit(self.exclusion_contours())
        self.update()
        return True

    def _mesh_row_col(self, index: int) -> tuple:
        """Row/col for a flat mesh index."""
        if self._mesh_cols <= 0:
            return -1, -1
        return divmod(int(index), int(self._mesh_cols))

    def _insert_mesh_boundary_at(self, position: QPointF) -> bool:
        """Disabled — wrap editing is 4 corners only; dense mesh is internal."""
        return False

    def _remove_mesh_boundary_at(self, position: QPointF) -> bool:
        """Disabled — wrap corners cannot be removed."""
        return False

    def _remove_mesh_index(self, index: int) -> bool:
        """Disabled — wrap mesh density is managed internally."""
        return False

    # -------------------------------------------------------- drag and drop

    def dragEnterEvent(self, event) -> None:
        """Accept image files."""
        if self._extract_paths(event):
            event.acceptProposedAction()
            self._drag_hint = True
            self.update()

    def dragMoveEvent(self, event) -> None:
        """Keep accepting while the cursor moves over the canvas."""
        if self._extract_paths(event):
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        """Clear the drop highlight."""
        self._drag_hint = False
        self.update()

    def dropEvent(self, event) -> None:
        """Emit the dropped image paths."""
        paths = self._extract_paths(event)
        self._drag_hint = False
        self.update()

        if paths:
            event.acceptProposedAction()
            self.filesDropped.emit(paths)

    @staticmethod
    def _extract_paths(event) -> List[str]:
        """Supported image paths carried by a drag event."""
        mime = event.mimeData()
        if not mime.hasUrls():
            return []

        paths = []
        for url in mime.urls():
            local = url.toLocalFile()
            if local and Path(local).suffix.lower() in SUPPORTED_SUFFIXES:
                paths.append(local)

        return paths


class ClickableLabel(QLabel):
    """Label that emits a signal when clicked."""

    clicked = Signal()

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def mousePressEvent(self, event) -> None:
        """Emit `clicked` on left button press."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class SliderRow(QWidget):
    """A labelled slider with a live value badge that resets when clicked."""

    valueChanged = Signal(str, float)

    def __init__(self, key: str, title: str, minimum: float, maximum: float,
                 default: float, divisor: float = 1.0, decimals: int = 0,
                 suffix: str = "", tooltip: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.key = key
        self.divisor = float(divisor)
        self.decimals = decimals
        self.suffix = suffix
        self.default = float(default)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sliderLabel")
        header.addWidget(self.title_label)
        header.addStretch()

        self.value_label = ClickableLabel()
        self.value_label.setObjectName("sliderValue")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setToolTip("Click to reset")
        self.value_label.clicked.connect(self.reset)
        header.addWidget(self.value_label)

        layout.addLayout(header)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(round(minimum * self.divisor)),
                             int(round(maximum * self.divisor)))
        self.slider.setValue(int(round(default * self.divisor)))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, int(round((maximum - minimum) * self.divisor / 20))))
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        if tooltip:
            self.setToolTip(tooltip)
            self.slider.setToolTip(tooltip)

        self._update_label()

    def value(self) -> float:
        """Current value in real units."""
        return self.slider.value() / self.divisor

    def set_value(self, value: float, notify: bool = False) -> None:
        """Set the value, optionally emitting `valueChanged`."""
        raw = int(round(float(value) * self.divisor))

        if raw == self.slider.value():
            self._update_label()
            return

        if notify:
            self.slider.setValue(raw)
            return

        self.slider.blockSignals(True)
        self.slider.setValue(raw)
        self.slider.blockSignals(False)
        self._update_label()

    def reset(self) -> None:
        """Restore the default value and notify listeners."""
        self.set_value(self.default, notify=True)

    def is_modified(self) -> bool:
        """Whether the value differs from the default."""
        return abs(self.value() - self.default) > 1e-6

    def _on_slider_changed(self, _raw: int) -> None:
        """Update the badge and forward the new value."""
        self._update_label()
        self.valueChanged.emit(self.key, self.value())

    def _update_label(self) -> None:
        """Refresh the value badge text and highlight state."""
        value = self.value()
        text = f"{value:.{self.decimals}f}{self.suffix}"

        if self.decimals == 0 and value > 0 and self.default == 0:
            text = f"+{text}"

        self.value_label.setText(text)
        self.value_label.setObjectName(
            "sliderValue" if self.is_modified() else "badgeMuted")
        self.value_label.style().unpolish(self.value_label)
        self.value_label.style().polish(self.value_label)


class Card(QFrame):
    """Rounded surface with an optional title, used to group controls."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setObjectName("card")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 14)
        self._layout.setSpacing(8)

        self.header_row: Optional[QHBoxLayout] = None

        if title:
            self.header_row = QHBoxLayout()
            self.header_row.setContentsMargins(0, 0, 0, 0)
            self.header_row.setSpacing(8)

            label = QLabel(title)
            label.setObjectName("sectionTitle")
            label.setWordWrap(False)
            self.header_row.addWidget(label)
            self.header_row.addStretch()

            self._layout.addLayout(self.header_row)

    def body(self) -> QVBoxLayout:
        """Layout new content should be added to."""
        return self._layout

    def add_widget(self, widget: QWidget) -> QWidget:
        """Append a widget to the card."""
        self._layout.addWidget(widget)
        return widget

    def add_layout(self, layout) -> None:
        """Append a nested layout to the card."""
        self._layout.addLayout(layout)

    def add_header_widget(self, widget: QWidget) -> QWidget:
        """Place a widget on the right side of the card title."""
        if self.header_row is not None:
            self.header_row.addWidget(widget)
        else:
            self._layout.addWidget(widget)
        return widget


def horizontal_divider() -> QFrame:
    """Thin horizontal separator."""
    line = QFrame()
    line.setObjectName("divider")
    line.setFixedHeight(1)
    return line


def vertical_divider(height: int = 22) -> QFrame:
    """Thin vertical separator."""
    line = QFrame()
    line.setObjectName("vDivider")
    line.setFixedWidth(1)
    line.setFixedHeight(height)
    return line
