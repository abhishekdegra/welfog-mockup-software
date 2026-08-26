"""
Main application window for the Phone Cover Mockup Generator.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import (
    QMutex, QMutexLocker, Qt, QThread, QTimer, QWaitCondition, Signal,
)
from PySide6.QtGui import QAction, QFontMetrics, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..config import (
    APP_VERSION, PROJECT_EXTENSION, get_config,
)
from ..image_processing.compositor import DEFAULT_SETTINGS, PRESETS, Compositor
from ..persistence.edit_history import EditHistory, EditSnapshot
from ..persistence.project_store import ProjectError, ProjectStore
from ..persistence.user_settings import UserSettings
from ..utils.image_loader import ImageLoader, ImageLoadError
from .batch_dialog import BatchDialog
from .styles import DARK_THEME_STYLES, Palette
from .widgets import (
    Card, PreviewCanvas, SliderRow, horizontal_divider, numpy_to_qpixmap,
    vertical_divider,
)

logger = logging.getLogger("mockup.ui")



SLIDER_GROUPS: Dict[str, List[tuple]] = {
    'placement': [
        # key, title, min, max, default, divisor, decimals, suffix, tooltip
        ('design_scale', 'Zoom design', 25, 400, 100, 1, 0, '%',
         'Zoom the print art inside the phone wrap'),
        ('offset_x', 'Move left / right', -180, 180, 0, 1, 0, '',
         'Slide the design horizontally — drag with Move Design to place the print'),
        ('offset_y', 'Move up / down', -180, 180, 0, 1, 0, '',
         'Slide the design vertically — drag with Move Design to place the print'),
        ('rotation', 'Rotation', -180, 180, 0, 1, 0, '\u00b0', 'Rotate the design'),
        ('region_inset', 'Region Inset', -20, 40, 0, 1, 0, '%', 'Shrink or grow the printed area'),
        ('corner_radius', 'Corner Radius', 0, 50, 10, 1, 0, '%', 'Round the printed corners'),
    ],
    'colour': [
        ('exposure', 'Exposure', -100, 100, 0, 1, 0, '', 'Overall light in stops'),
        ('brightness', 'Brightness', -100, 100, 0, 1, 0, '', 'Lift or lower all tones'),
        ('contrast', 'Contrast', -100, 100, 0, 1, 0, '', 'Separate lights from darks'),
        ('highlights', 'Highlights', -100, 100, 0, 1, 0, '', 'Recover or bloom bright areas'),
        ('shadows', 'Shadows', -100, 100, 0, 1, 0, '', 'Open up or deepen dark areas'),
        ('gamma', 'Gamma', 0.1, 3.0, 1.0, 100, 2, '', 'Midtone response curve'),
        ('temperature', 'Temperature', -100, 100, 0, 1, 0, '', 'Cool (left) to warm (right)'),
        ('tint', 'Tint', -100, 100, 0, 1, 0, '', 'Green to magenta balance'),
        ('hue', 'Hue Shift', -180, 180, 0, 1, 0, '\u00b0', 'Rotate all colours'),
        ('saturation', 'Saturation', -100, 100, 0, 1, 0, '', 'Colour intensity'),
        ('vibrance', 'Vibrance', -100, 100, 0, 1, 0, '', 'Boost muted colours only'),
    ],
    'detail': [
        ('clarity', 'Clarity', -100, 100, 0, 1, 0, '', 'Local contrast and punch'),
        ('sharpness', 'Sharpness', -100, 100, 0, 1, 0, '', 'Edge definition of the print'),
        ('blur', 'Blur', 0, 100, 0, 1, 0, '', 'Soften the design'),
        ('grain', 'Grain', 0, 100, 0, 1, 0, '', 'Film grain, useful for matte finishes'),
    ],
    'realism': [
        ('opacity', 'Opacity', 0, 100, 100, 1, 0, '%', 'How opaque the print is'),
        ('edge_softness', 'Edge Softness', 0, 50, 3, 1, 0, '', 'Feather the print edges'),
        ('texture_strength', 'Material Blend', 0, 100, 52, 1, 0, '%',
         'Let the cover texture and shading show through the print'),
        ('reflection_strength', 'Reflections', 0, 100, 28, 1, 0, '%',
         'Soft sheen that keeps cover colour (no chalk white wash)'),
        ('shadow_strength', 'Shadows Depth', 0, 100, 34, 1, 0, '%',
         'Keep the shadows of the original photo'),
        ('tone_match', 'Tone Match', 0, 100, 0, 1, 0, '%',
         'Match the design brightness to the photo'),
        ('vignette', 'Vignette', 0, 100, 0, 1, 0, '%', 'Darken the frame corners'),
    ],
}


class RenderThread(QThread):
    """
    Long lived worker that renders the composite off the UI thread.

    Only the most recent request is kept, so dragging a slider never queues up
    a backlog of stale frames.
    """

    rendered = Signal(object, int, float)
    failed = Signal(str, int)

    def __init__(self, compositor: Compositor, parent=None):
        super().__init__(parent)

        self._compositor = compositor
        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._pending: Optional[tuple] = None
        self._abort = False

    def request(self, max_size: Optional[int], token: int) -> None:
        """Queue a render, replacing any request that has not started yet."""
        with QMutexLocker(self._mutex):
            self._pending = (max_size, token)
            self._condition.wakeAll()

    def stop(self) -> None:
        """Ask the thread to finish and wait for it."""
        with QMutexLocker(self._mutex):
            self._abort = True
            self._condition.wakeAll()

        self.wait(3000)

    def run(self) -> None:
        """Render queued requests until stopped."""
        while True:
            self._mutex.lock()
            while self._pending is None and not self._abort:
                self._condition.wait(self._mutex)

            if self._abort:
                self._mutex.unlock()
                return

            max_size, token = self._pending
            self._pending = None
            self._mutex.unlock()

            started = time.perf_counter()
            try:
                image = self._compositor.render(max_size)
                elapsed = (time.perf_counter() - started) * 1000.0
                self.rendered.emit(image, token, elapsed)
            except Exception as exc:  # keep the UI alive on unexpected failures
                logger.exception("Preview render failed")
                self.failed.emit(str(exc), token)


class ExportThread(QThread):
    """One-shot worker that renders at full resolution and writes the file."""

    done = Signal(bool, str, str)

    def __init__(self, compositor: Compositor, path: str, quality: int,
                 parent=None):
        super().__init__(parent)

        # Clone so preview rendering can continue without sharing caches.
        try:
            self._compositor = compositor.create_production_clone()
            if compositor.design_image is not None:
                self._compositor.design_image = compositor.design_image.copy()
                self._compositor.settings = dict(compositor.settings)
                self._compositor.material_name = compositor.material_name
                self._compositor.lighting_name = compositor.lighting_name
                self._compositor.fit_mode = compositor.fit_mode
                self._compositor.mirror = compositor.mirror
        except Exception:
            logger.exception("Could not clone compositor for export; using live session")
            self._compositor = compositor
        self._path = path
        self._quality = quality

    def run(self) -> None:
        """Render and save, reporting success through `done`."""
        try:
            image = self._compositor.export(
                include_alpha=Path(self._path).suffix.lower() == '.png')

            if image is None:
                self.done.emit(False, self._path, "Nothing to export yet.")
                return

            success, error = ImageLoader.save_image_ex(
                image, self._path, self._quality
            )
            if success:
                logger.info("Export written to %s", self._path)
            else:
                logger.error("Export write failed: %s", error)
            self.done.emit(success, self._path, error)
        except MemoryError:
            logger.error("Export out of memory")
            self.done.emit(False, self._path, "Not enough memory to export at full resolution.")
        except Exception as exc:
            logger.exception("Export failed")
            self.done.emit(False, self._path, str(exc))


class MainWindow(QMainWindow):
    """Application shell: header, preview canvas, control panel and status bar."""

    def __init__(self):
        super().__init__()

        cfg = get_config()
        self.PREVIEW_MAX = int(cfg.preview_max)
        self.RENDER_DEBOUNCE_MS = int(cfg.render_debounce_ms)

        self.compositor = Compositor()
        self.sliders: Dict[str, SliderRow] = {}
        self.phone_path: Optional[Path] = None
        self.design_path: Optional[Path] = None
        self.project_path: Optional[Path] = None
        self.current_preview: Optional[np.ndarray] = None
        self._render_token = 0
        self._export_thread: Optional[ExportThread] = None
        self._batch_dialog: Optional[BatchDialog] = None
        self._syncing = False
        self._dirty = False
        self._restoring_history = False
        self._design_pan_gesture = False
        self._design_pan_pending = (0.0, 0.0)
        self._edit_history = EditHistory(limit=50)
        self._mesh_baseline: Optional[EditSnapshot] = None
        self.user_settings = UserSettings()

        self.setWindowTitle(cfg.app_name)
        self.setMinimumSize(1180, 760)
        self.resize(1500, 940)
        self.setAcceptDrops(True)
        self.setStyleSheet(DARK_THEME_STYLES)

        self._build_ui()
        self._build_menus()
        self._connect_signals()
        self._restore_window_chrome()

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(self.RENDER_DEBOUNCE_MS)
        self.render_timer.timeout.connect(self._start_render)

        # Coalesce Move Design mouse moves into one smooth settings update.
        self.design_pan_timer = QTimer(self)
        self.design_pan_timer.setSingleShot(True)
        self.design_pan_timer.setInterval(16)
        self.design_pan_timer.timeout.connect(self._flush_design_pan)

        self.placement_idle_timer = QTimer(self)
        self.placement_idle_timer.setSingleShot(True)
        self.placement_idle_timer.setInterval(80)
        self.placement_idle_timer.timeout.connect(self._end_live_placement)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(max(15, int(cfg.autosave_interval_sec)) * 1000)
        self.autosave_timer.timeout.connect(self._autosave_session)
        self.autosave_timer.start()

        self.render_thread = RenderThread(self.compositor, self)
        self.render_thread.rendered.connect(self._on_rendered)
        self.render_thread.failed.connect(self._on_render_failed)
        self.render_thread.start()

        self._update_enabled_state()
        self.status_message("Load a phone photo and a design to begin")
        QTimer.singleShot(250, self._maybe_reopen_last_project)

    # ------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        """Assemble the window layout."""
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(14, 14, 14, 10)
        body_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self._build_preview_panel())
        self.splitter.addWidget(self._build_controls_panel())
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([960, 420])

        body_layout.addWidget(self.splitter)
        root.addWidget(body, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self._build_status_bar()

    def _build_header(self) -> QWidget:
        """Top bar with branding, presets and the primary actions."""
        header = QFrame()
        header.setObjectName("headerBar")
        header.setFixedHeight(64)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(12)

        logo = QLabel("\u25C8")
        logo.setStyleSheet(f"color: {Palette.ACCENT}; font-size: 22pt;")
        layout.addWidget(logo)

        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(0)

        title = QLabel("Phone Cover Mockup Studio")
        title.setObjectName("appTitle")
        titles.addWidget(title)

        subtitle = QLabel("Realistic cover mockups \u00b7 fully offline")
        subtitle.setObjectName("appSubtitle")
        titles.addWidget(subtitle)

        layout.addLayout(titles)
        layout.addSpacing(10)
        layout.addWidget(vertical_divider(30))
        layout.addStretch()

        preset_label = QLabel("Preset")
        preset_label.setObjectName("sliderLabel")
        layout.addWidget(preset_label)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.setFixedWidth(160)
        self.preset_combo.setToolTip("Apply a look, then fine tune the sliders")
        layout.addWidget(self.preset_combo)

        layout.addWidget(vertical_divider(30))

        self.load_phone_btn = QPushButton("Load Phone")
        self.load_phone_btn.setObjectName("ghostButton")
        self.load_phone_btn.setToolTip("Open a phone or cover photo (Ctrl+P)")
        layout.addWidget(self.load_phone_btn)

        self.load_design_btn = QPushButton("Load Design")
        self.load_design_btn.setObjectName("ghostButton")
        self.load_design_btn.setToolTip("Open the artwork to print (Ctrl+D)")
        layout.addWidget(self.load_design_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.setObjectName("successButton")
        self.export_btn.setToolTip("Save the mockup at full resolution (Ctrl+E)")
        layout.addWidget(self.export_btn)

        return header

    def _build_preview_panel(self) -> QWidget:
        """Canvas with its own toolbar and info strip."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        toolbar = QFrame()
        toolbar.setObjectName("floatingBar")
        bar = QVBoxLayout(toolbar)
        bar.setContentsMargins(10, 8, 10, 8)
        bar.setSpacing(6)

        # Row 1 — identity + zoom (badges elide so tools never get crushed)
        top = QHBoxLayout()
        top.setSpacing(8)

        preview_title = QLabel("Preview")
        preview_title.setObjectName("panelTitle")
        top.addWidget(preview_title)

        self.phone_badge = QLabel("no phone")
        self.phone_badge.setObjectName("badgeMuted")
        self.phone_badge.setMaximumWidth(150)
        self.phone_badge.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.phone_badge)

        self.design_badge = QLabel("no design")
        self.design_badge.setObjectName("badgeMuted")
        self.design_badge.setMaximumWidth(150)
        top.addWidget(self.design_badge)

        top.addStretch(1)

        self.undo_btn = QPushButton("\u21B6")
        self.undo_btn.setObjectName("toolButtonCompact")
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.undo_btn.setEnabled(False)
        top.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("\u21B7")
        self.redo_btn.setObjectName("toolButtonCompact")
        self.redo_btn.setToolTip("Redo (Ctrl+Y)")
        self.redo_btn.setEnabled(False)
        top.addWidget(self.redo_btn)

        top.addWidget(vertical_divider(22))

        self.zoom_out_btn = QPushButton("\u2212")
        self.zoom_out_btn.setObjectName("toolButtonCompact")
        self.zoom_out_btn.setToolTip("Zoom out (Ctrl+-)")
        top.addWidget(self.zoom_out_btn)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setObjectName("toolButtonCompact")
        self.zoom_in_btn.setToolTip("Zoom in (Ctrl++)")
        top.addWidget(self.zoom_in_btn)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setObjectName("toolButton")
        self.fit_btn.setToolTip("Fit to view (Ctrl+F)")
        self.fit_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        top.addWidget(self.fit_btn)

        self.actual_size_btn = QPushButton("1:1")
        self.actual_size_btn.setObjectName("toolButton")
        self.actual_size_btn.setToolTip("Show at 100% (Ctrl+1)")
        self.actual_size_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        top.addWidget(self.actual_size_btn)

        bar.addLayout(top)

        # Row 2 — full tool labels (never squeezed by long file names)
        tools = QHBoxLayout()
        tools.setSpacing(8)

        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setObjectName("toolButton")
        self.compare_btn.setCheckable(True)
        self.compare_btn.setToolTip("Show the original photo (C)")
        self.compare_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        tools.addWidget(self.compare_btn)

        self.region_btn = QPushButton("Region")
        self.region_btn.setObjectName("toolButton")
        self.region_btn.setCheckable(True)
        self.region_btn.setToolTip("Outline the detected cover area (R)")
        self.region_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        tools.addWidget(self.region_btn)

        self.edit_region_btn = QPushButton("Edit Mesh")
        self.edit_region_btn.setObjectName("toolButton")
        self.edit_region_btn.setCheckable(True)
        self.edit_region_btn.setToolTip(
            "Edit edges & cutouts: drag black dots · "
            "Ctrl/double-click edge add dot · Shift+click new shape · "
            "Erase Wrap · × remove"
        )
        self.edit_region_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        tools.addWidget(self.edit_region_btn)

        self.move_design_toolbar_btn = QPushButton("Move Design")
        self.move_design_toolbar_btn.setObjectName("toolButton")
        self.move_design_toolbar_btn.setCheckable(True)
        self.move_design_toolbar_btn.setToolTip(
            "Drag on the preview to move the print left/right/up/down"
        )
        self.move_design_toolbar_btn.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Fixed
        )
        tools.addWidget(self.move_design_toolbar_btn)

        self.erase_wrap_btn = QPushButton("Erase Wrap")
        self.erase_wrap_btn.setObjectName("toolButton")
        self.erase_wrap_btn.setCheckable(True)
        self.erase_wrap_btn.setToolTip(
            "Paint to remove cover wrap on buttons / holes. "
            "Size: [ ] keys or Alt+scroll"
        )
        self.erase_wrap_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        tools.addWidget(self.erase_wrap_btn)

        self.final_btn = QPushButton("Final")
        self.final_btn.setObjectName("toolButton")
        self.final_btn.setCheckable(True)
        self.final_btn.setToolTip(
            "Final polish after wrap: Erase to clear wrap, Fill to restore "
            "print on the real cover. Thin line shows your stroke."
        )
        self.final_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        tools.addWidget(self.final_btn)

        self.final_erase_btn = QPushButton("Erase")
        self.final_erase_btn.setObjectName("toolButtonCompact")
        self.final_erase_btn.setCheckable(True)
        self.final_erase_btn.setEnabled(False)
        self.final_erase_btn.setToolTip(
            "Final Erase — paint a thin stroke; wrap clears smoothly"
        )
        self.final_erase_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        tools.addWidget(self.final_erase_btn)

        self.final_fill_btn = QPushButton("Fill")
        self.final_fill_btn.setObjectName("toolButtonCompact")
        self.final_fill_btn.setCheckable(True)
        self.final_fill_btn.setEnabled(False)
        self.final_fill_btn.setToolTip(
            "Final Fill — restore wrap only on the real cover where it was removed"
        )
        self.final_fill_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        tools.addWidget(self.final_fill_btn)

        tools.addStretch(1)
        bar.addLayout(tools)

        layout.addWidget(toolbar)

        canvas_card = QFrame()
        canvas_card.setObjectName("canvasCard")
        canvas_layout = QVBoxLayout(canvas_card)
        canvas_layout.setContentsMargins(1, 1, 1, 1)

        self.canvas = PreviewCanvas()
        canvas_layout.addWidget(self.canvas)

        layout.addWidget(canvas_card, 1)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(6, 0, 6, 0)
        info_row.setSpacing(14)

        self.hint_label = QLabel(
            "Scroll to zoom \u00b7 drag to pan \u00b7 double click to fit"
        )
        self.hint_label.setObjectName("infoLabel")
        info_row.addWidget(self.hint_label)

        info_row.addStretch()

        self.size_label = QLabel("")
        self.size_label.setObjectName("infoLabel")
        info_row.addWidget(self.size_label)

        self.render_label = QLabel("")
        self.render_label.setObjectName("infoLabel")
        info_row.addWidget(self.render_label)

        layout.addLayout(info_row)

        return panel

    def _build_controls_panel(self) -> QWidget:
        """Right hand panel with the tabbed adjustment groups."""
        panel = QWidget()
        panel.setMinimumWidth(400)
        panel.setMaximumWidth(480)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_placement_tab(), "Placement")
        self.tabs.addTab(self._build_slider_tab('colour', 'Colour Grading'), "Colour")
        self.tabs.addTab(self._build_slider_tab('detail', 'Detail & Texture'), "Detail")
        self.tabs.addTab(self._build_slider_tab('realism', 'Material Realism'), "Realism")
        layout.addWidget(self.tabs, 1)

        actions = QVBoxLayout()
        actions.setSpacing(8)

        self.reset_btn = QPushButton("Reset Adjustments")
        self.reset_btn.setObjectName("dangerButton")
        self.reset_btn.setToolTip("Restore every slider to its default (Ctrl+R)")
        actions.addWidget(self.reset_btn)

        self.export_btn_2 = QPushButton("Export Mockup")
        self.export_btn_2.setObjectName("primaryButton")
        actions.addWidget(self.export_btn_2)

        layout.addLayout(actions)

        return panel

    def _build_placement_tab(self) -> QWidget:
        """Placement tab: source images, mapping options and geometry sliders."""
        container, body = self._scrollable_tab()

        sources = Card("Source Images")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        phone_caption = QLabel("Phone photo")
        phone_caption.setObjectName("sliderLabel")
        grid.addWidget(phone_caption, 0, 0)

        self.phone_name_label = QLabel("Not loaded")
        self.phone_name_label.setObjectName("infoLabel")
        self.phone_name_label.setWordWrap(True)
        self.phone_name_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self.phone_name_label, 0, 1)

        design_caption = QLabel("Design")
        design_caption.setObjectName("sliderLabel")
        grid.addWidget(design_caption, 1, 0)

        self.design_name_label = QLabel("Not loaded")
        self.design_name_label.setObjectName("infoLabel")
        self.design_name_label.setWordWrap(True)
        self.design_name_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self.design_name_label, 1, 1)

        grid.setColumnStretch(1, 1)
        sources.add_layout(grid)

        source_buttons = QHBoxLayout()
        source_buttons.setSpacing(8)

        self.swap_btn = QPushButton("Swap Images")
        self.swap_btn.setObjectName("ghostButton")
        self.swap_btn.setToolTip("Use the phone photo as the design and vice versa")
        source_buttons.addWidget(self.swap_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setObjectName("ghostButton")
        source_buttons.addWidget(self.clear_btn)

        sources.add_layout(source_buttons)
        body.addWidget(sources)

        mapping = Card("Print Area")
        options = QHBoxLayout()
        options.setSpacing(8)

        fit_label = QLabel("Fit")
        fit_label.setObjectName("sliderLabel")
        options.addWidget(fit_label)

        self.fit_mode_combo = QComboBox()
        self.fit_mode_combo.addItem("Fill (crop)", 'fill')
        self.fit_mode_combo.addItem("Contain (letterbox)", 'fit')
        self.fit_mode_combo.addItem("Stretch", 'stretch')
        options.addWidget(self.fit_mode_combo, 1)

        self.mirror_check = QCheckBox("Mirror")
        self.mirror_check.setToolTip("Flip the design horizontally")
        options.addWidget(self.mirror_check)

        mapping.add_layout(options)

        region_buttons = QHBoxLayout()
        region_buttons.setSpacing(8)

        self.detect_btn = QPushButton("Auto Detect")
        self.detect_btn.setObjectName("ghostButton")
        self.detect_btn.setToolTip("Find the cover region in the photo again")
        self.detect_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        region_buttons.addWidget(self.detect_btn)

        self.center_region_btn = QPushButton("Center Region")
        self.center_region_btn.setObjectName("ghostButton")
        self.center_region_btn.setToolTip("Use a centered phone-shaped area")
        self.center_region_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        region_buttons.addWidget(self.center_region_btn)

        mapping.add_layout(region_buttons)

        self.reset_mesh_btn = QPushButton("Reset Mesh")
        self.reset_mesh_btn.setObjectName("dangerButton")
        self.reset_mesh_btn.setToolTip(
            "Restore mesh + cutouts to how they were when the phone loaded "
            "(asks Yes/No first)"
        )
        mapping.add_widget(self.reset_mesh_btn)

        self.finish_scope_combo = QComboBox()
        self.finish_scope_combo.addItem("Everything", "all")
        self.finish_scope_combo.addItem("Edges & corners", "edges")
        self.finish_scope_combo.addItem("Camera cutouts", "camera")
        self.finish_scope_combo.addItem("Side buttons / fingerprint", "buttons")
        self.finish_scope_combo.setCurrentIndex(0)
        self.finish_scope_combo.setToolTip(
            "Everything = full realistic wrap (edges + corners + camera + "
            "side buttons / fingerprint) for any phone. Or pick one area so "
            "fixing it does not undo another."
        )
        mapping.add_widget(self.finish_scope_combo)

        self.perfect_finish_btn = QPushButton("Perfect Finish")
        self.perfect_finish_btn.setObjectName("ghostButton")
        self.perfect_finish_btn.setToolTip(
            "Apply finish only to the selected scope · Ctrl+Shift+F"
        )
        mapping.add_widget(self.perfect_finish_btn)

        final_label = QLabel("Final polish")
        final_label.setObjectName("sliderLabel")
        mapping.add_widget(final_label)
        final_hint = QLabel(
            "After wrap: use Final → Erase to clear print, or Fill to put wrap "
            "back on the real cover. A thin line shows your stroke while you paint."
        )
        final_hint.setObjectName("infoLabel")
        final_hint.setWordWrap(True)
        mapping.add_widget(final_hint)
        final_row = QHBoxLayout()
        final_row.setSpacing(8)
        self.final_panel_btn = QPushButton("Final")
        self.final_panel_btn.setObjectName("toolButton")
        self.final_panel_btn.setCheckable(True)
        self.final_panel_btn.setToolTip("Turn on Final polish brush")
        final_row.addWidget(self.final_panel_btn)
        self.final_panel_erase_btn = QPushButton("Erase")
        self.final_panel_erase_btn.setObjectName("ghostButton")
        self.final_panel_erase_btn.setCheckable(True)
        self.final_panel_erase_btn.setEnabled(False)
        final_row.addWidget(self.final_panel_erase_btn)
        self.final_panel_fill_btn = QPushButton("Fill")
        self.final_panel_fill_btn.setObjectName("ghostButton")
        self.final_panel_fill_btn.setCheckable(True)
        self.final_panel_fill_btn.setEnabled(False)
        final_row.addWidget(self.final_panel_fill_btn)
        mapping.add_layout(final_row)

        cutout_label = QLabel("Cutout shape")
        cutout_label.setObjectName("sliderLabel")
        mapping.add_widget(cutout_label)

        cutout_row = QHBoxLayout()
        cutout_row.setSpacing(8)
        self.cutout_shape_combo = QComboBox()
        self.cutout_shape_combo.addItem("Circle", "circle")
        self.cutout_shape_combo.addItem("Square", "square")
        self.cutout_shape_combo.addItem("Rounded Square", "rounded_square")
        self.cutout_shape_combo.addItem("Rectangle", "rectangle")
        self.cutout_shape_combo.addItem("Rounded Rectangle", "rounded_rect")
        self.cutout_shape_combo.addItem("Oval", "oval")
        self.cutout_shape_combo.addItem("Pill Horizontal", "pill_h")
        self.cutout_shape_combo.addItem("Pill Vertical", "pill_v")
        self.cutout_shape_combo.addItem("Capsule (button)", "capsule")
        self.cutout_shape_combo.addItem("Squircle (iPhone)", "squircle")
        self.cutout_shape_combo.addItem("Superellipse", "superellipse")
        self.cutout_shape_combo.addItem("Polygon", "polygon")
        self.cutout_shape_combo.addItem("Triangle", "triangle")
        self.cutout_shape_combo.addItem("Custom Path", "custom_path")
        self.cutout_shape_combo.addItem("Free (diamond)", "free")
        self.cutout_shape_combo.setToolTip(
            "Shape added with Shift+click in Edit Mesh mode. "
            "Never auto-switches — you choose the shape. "
            "Use Capsule / Pill for side buttons, or Erase Wrap to paint."
        )
        cutout_row.addWidget(self.cutout_shape_combo, 1)

        self.cutout_shrink_btn = QPushButton("\u2212")
        self.cutout_shrink_btn.setObjectName("toolButtonCompact")
        self.cutout_shrink_btn.setToolTip(
            "Shrink cutouts (−) · or Shift+drag a cutout handle/body to scale"
        )
        self.cutout_grow_btn = QPushButton("+")
        self.cutout_grow_btn.setObjectName("toolButtonCompact")
        self.cutout_grow_btn.setToolTip(
            "Grow cutouts (+) · or Shift+drag a cutout handle/body to scale"
        )
        cutout_row.addWidget(self.cutout_shrink_btn)
        cutout_row.addWidget(self.cutout_grow_btn)
        mapping.add_layout(cutout_row)

        cutout_edit_row = QHBoxLayout()
        cutout_edit_row.setSpacing(8)
        self.cutout_corner_spin = QDoubleSpinBox()
        self.cutout_corner_spin.setRange(0.0, 50.0)
        self.cutout_corner_spin.setSingleStep(1.0)
        self.cutout_corner_spin.setValue(16.0)
        self.cutout_corner_spin.setSuffix(" % r")
        self.cutout_corner_spin.setToolTip(
            "Corner roundness for rectangle / rounded cutouts "
            "(mild ~16% matches camera modules)."
        )
        cutout_edit_row.addWidget(self.cutout_corner_spin, 1)
        self.cutout_rot_spin = QDoubleSpinBox()
        self.cutout_rot_spin.setRange(-180.0, 180.0)
        self.cutout_rot_spin.setSingleStep(1.0)
        self.cutout_rot_spin.setValue(0.0)
        self.cutout_rot_spin.setSuffix(" °")
        self.cutout_rot_spin.setToolTip("Rotate the selected / last cutout")
        cutout_edit_row.addWidget(self.cutout_rot_spin, 1)
        self.cutout_apply_btn = QPushButton("Apply")
        self.cutout_apply_btn.setObjectName("toolButtonCompact")
        self.cutout_apply_btn.setToolTip(
            "Apply corner radius + rotation to the hovered or last cutout"
        )
        cutout_edit_row.addWidget(self.cutout_apply_btn)
        mapping.add_layout(cutout_edit_row)

        self.region_hint = QLabel(
            "Turn on Edit Mesh to adjust edges, curves, and local alignment."
        )
        self.region_hint.setObjectName("infoLabel")
        self.region_hint.setWordWrap(True)
        mapping.add_widget(self.region_hint)

        mapping.add_widget(horizontal_divider())

        # --- Adjust Design: pan / zoom where artwork prints ---
        adjust_title = QLabel("Adjust Design Position")
        adjust_title.setObjectName("sliderLabel")
        mapping.add_widget(adjust_title)

        adjust_hint = QLabel(
            "Move Design: drag to slide the print · scroll to zoom in/out · "
            "Ctrl+scroll zooms the view. Arrows / + − also work."
        )
        adjust_hint.setObjectName("infoLabel")
        adjust_hint.setWordWrap(True)
        mapping.add_widget(adjust_hint)

        nudge_row = QGridLayout()
        nudge_row.setContentsMargins(0, 4, 0, 4)
        nudge_row.setHorizontalSpacing(6)
        nudge_row.setVerticalSpacing(6)

        self.nudge_up_btn = QPushButton("▲")
        self.nudge_down_btn = QPushButton("▼")
        self.nudge_left_btn = QPushButton("◀")
        self.nudge_right_btn = QPushButton("▶")
        self.zoom_design_in_btn = QPushButton("+")
        self.zoom_design_out_btn = QPushButton("−")
        for btn, tip in (
            (self.nudge_up_btn, "Nudge design up"),
            (self.nudge_down_btn, "Nudge design down"),
            (self.nudge_left_btn, "Nudge design left"),
            (self.nudge_right_btn, "Nudge design right"),
            (self.zoom_design_in_btn, "Zoom design in"),
            (self.zoom_design_out_btn, "Zoom design out"),
        ):
            btn.setObjectName("toolButtonCompact")
            btn.setFixedSize(40, 32)
            btn.setToolTip(tip)

        nudge_row.addWidget(self.nudge_up_btn, 0, 1)
        nudge_row.addWidget(self.zoom_design_in_btn, 0, 2)
        nudge_row.addWidget(self.nudge_left_btn, 1, 0)
        nudge_row.addWidget(self.nudge_right_btn, 1, 2)
        nudge_row.addWidget(self.nudge_down_btn, 2, 1)
        nudge_row.addWidget(self.zoom_design_out_btn, 2, 2)
        mapping.add_layout(nudge_row)

        adjust_btns = QHBoxLayout()
        adjust_btns.setSpacing(8)
        self.move_design_btn = QPushButton("Move Design")
        self.move_design_btn.setObjectName("toolButton")
        self.move_design_btn.setCheckable(True)
        self.move_design_btn.setToolTip(
            "Drag on the preview to slide the design. "
            "Turn off Edit Mesh first. Middle-mouse still pans the view."
        )
        adjust_btns.addWidget(self.move_design_btn)
        self.center_design_btn = QPushButton("Center")
        self.center_design_btn.setObjectName("ghostButton")
        self.center_design_btn.setToolTip("Reset design to centre of the wrap")
        adjust_btns.addWidget(self.center_design_btn)
        self.reset_design_pos_btn = QPushButton("Reset Pos")
        self.reset_design_pos_btn.setObjectName("ghostButton")
        self.reset_design_pos_btn.setToolTip(
            "Reset move + zoom + rotation to defaults"
        )
        adjust_btns.addWidget(self.reset_design_pos_btn)
        mapping.add_layout(adjust_btns)

        mapping.add_widget(horizontal_divider())

        for spec in SLIDER_GROUPS['placement']:
            mapping.add_widget(self._make_slider(spec))

        body.addWidget(mapping)
        body.addStretch()

        return container

    def _build_slider_tab(self, group: str, title: str) -> QWidget:
        """Tab holding one card of sliders."""
        container, body = self._scrollable_tab()

        card = Card(title)

        reset_group = QPushButton("Reset")
        reset_group.setObjectName("linkButton")
        reset_group.clicked.connect(lambda: self._reset_group(group))
        card.add_header_widget(reset_group)

        for spec in SLIDER_GROUPS[group]:
            card.add_widget(self._make_slider(spec))

        body.addWidget(card)
        body.addStretch()

        return container

    @staticmethod
    def _scrollable_tab() -> tuple:
        """Scroll area plus the layout its content should be added to."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 6, 8, 6)
        layout.setSpacing(10)

        scroll.setWidget(content)

        return scroll, layout

    def _make_slider(self, spec: tuple) -> SliderRow:
        """Build a slider row from a specification tuple and register it."""
        key, title, minimum, maximum, default, divisor, decimals, suffix, tooltip = spec

        row = SliderRow(key, title, minimum, maximum, default,
                        divisor=divisor, decimals=decimals, suffix=suffix,
                        tooltip=tooltip)
        row.valueChanged.connect(self._on_slider_changed)
        self.sliders[key] = row

        return row

    def _build_status_bar(self) -> None:
        """Status bar with permanent readouts on the right."""
        bar = self.statusBar()

        self.zoom_status = QLabel("100%")
        self.zoom_status.setObjectName("infoLabel")
        bar.addPermanentWidget(self.zoom_status)

        self.region_status = QLabel("No region")
        self.region_status.setObjectName("infoLabel")
        bar.addPermanentWidget(self.region_status)

        bar.showMessage("Ready")

    def _build_menus(self) -> None:
        """Menu bar and shortcuts."""
        file_menu = self.menuBar().addMenu("&File")
        self._add_action(file_menu, "Load Phone Image…", "Ctrl+P", self.load_phone_image)
        self._add_action(file_menu, "Load Design Image…", "Ctrl+D", self.load_design_image)
        file_menu.addSeparator()
        self._add_action(file_menu, "New Project", "Ctrl+N", self.new_project)
        self._add_action(file_menu, "Open Project…", "Ctrl+O", self.open_project)
        self._add_action(file_menu, "Save Project", "Ctrl+S", self.save_project)
        self._add_action(file_menu, "Save Project As…", "Ctrl+Shift+S", self.save_project_as)
        self.recent_menu = file_menu.addMenu("Recent Projects")
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        self._add_action(file_menu, "Export Mockup…", "Ctrl+E", self.export_image)
        self._add_action(
            file_menu, "Batch Process Folder…", "Ctrl+B", self.open_batch_production
        )
        self._add_action(file_menu, "Copy Preview to Clipboard", "Ctrl+C",
                         self.copy_to_clipboard)
        file_menu.addSeparator()
        self._add_action(file_menu, "Clear All", "Ctrl+Shift+N", self.clear_all)
        self._add_action(file_menu, "Exit", "Ctrl+Q", self.close)

        edit_menu = self.menuBar().addMenu("&Edit")
        self._add_action(edit_menu, "Undo", "Ctrl+Z", self.undo_edit)
        self._add_action(edit_menu, "Redo", "Ctrl+Y", self.redo_edit)
        redo_alt = QAction("Redo", self)
        redo_alt.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        redo_alt.triggered.connect(self.redo_edit)
        self.addAction(redo_alt)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Reset Adjustments", "Ctrl+R", self.reset_adjustments)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Auto Detect Print Area", "Ctrl+Shift+D",
                         self.auto_detect_region)
        self._add_action(edit_menu, "Perfect Finish Geometry", "Ctrl+Shift+F",
                         self.perfect_finish_cutouts)
        self._add_action(edit_menu, "Center Print Area", "Ctrl+Shift+C",
                         self.center_region)
        self._add_action(
            edit_menu,
            "Reset Mesh to Start…",
            "Ctrl+Shift+R",
            self.reset_mesh_to_start,
        )
        self._add_action(edit_menu, "Toggle Mesh Editing", "E",
                         lambda: self.edit_region_btn.toggle())
        self._add_action(edit_menu, "Mirror Design", "M",
                         lambda: self.mirror_check.toggle())

        view_menu = self.menuBar().addMenu("&View")
        self._add_action(view_menu, "Fit to View", "Ctrl+F", self.canvas_fit)
        self._add_action(view_menu, "Actual Size", "Ctrl+1", self.canvas_actual_size)
        self._add_action(view_menu, "Zoom In", "Ctrl++", lambda: self.canvas.zoom_in())
        self._add_action(view_menu, "Zoom Out", "Ctrl+-", lambda: self.canvas.zoom_out())
        view_menu.addSeparator()
        self._add_action(view_menu, "Show Print Area", "R",
                         lambda: self.region_btn.toggle())
        self._add_action(view_menu, "Compare With Original", "C",
                         lambda: self.compare_btn.toggle())

        preset_menu = self.menuBar().addMenu("&Presets")
        for name in PRESETS:
            action = QAction(name, self)
            action.triggered.connect(lambda _=False, n=name: self.apply_preset(n))
            preset_menu.addAction(action)

        help_menu = self.menuBar().addMenu("&Help")
        self._add_action(help_menu, "Shortcuts", "F1", self.show_shortcuts)
        self._add_action(help_menu, "About", "", self.show_about)

    def _add_action(self, menu, text: str, shortcut: str, slot) -> QAction:
        """Create a menu action wired to a slot."""
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)

        return action

    def _connect_signals(self) -> None:
        """Wire every control to its handler."""
        self.load_phone_btn.clicked.connect(self.load_phone_image)
        self.load_design_btn.clicked.connect(self.load_design_image)
        self.export_btn.clicked.connect(self.export_image)
        self.export_btn_2.clicked.connect(self.export_image)
        self.reset_btn.clicked.connect(self.reset_adjustments)
        self.swap_btn.clicked.connect(self.swap_images)
        self.clear_btn.clicked.connect(self.clear_all)
        self.detect_btn.clicked.connect(self.auto_detect_region)
        self.center_region_btn.clicked.connect(self.center_region)
        self.reset_mesh_btn.clicked.connect(self.reset_mesh_to_start)
        self.perfect_finish_btn.clicked.connect(self.perfect_finish_cutouts)
        self.cutout_shape_combo.currentIndexChanged.connect(
            self._on_cutout_shape_changed
        )
        self.cutout_grow_btn.clicked.connect(
            lambda: self._scale_cutouts(1.08)
        )
        self.cutout_shrink_btn.clicked.connect(
            lambda: self._scale_cutouts(0.93)
        )
        self.cutout_apply_btn.clicked.connect(self._apply_cutout_edit)

        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        self.fit_mode_combo.currentIndexChanged.connect(self._on_fit_mode_changed)
        self.mirror_check.toggled.connect(self._on_mirror_toggled)

        self.compare_btn.toggled.connect(self._on_compare_toggled)
        self.region_btn.toggled.connect(self.canvas.set_show_cover)
        self.edit_region_btn.toggled.connect(self._on_edit_region_toggled)
        self.erase_wrap_btn.toggled.connect(self._on_erase_wrap_toggled)
        self.final_btn.toggled.connect(self._on_final_toggled)
        self.final_panel_btn.toggled.connect(self._on_final_toggled)
        self.final_erase_btn.toggled.connect(
            lambda checked: self._on_final_mode_toggled("erase", checked)
        )
        self.final_fill_btn.toggled.connect(
            lambda checked: self._on_final_mode_toggled("fill", checked)
        )
        self.final_panel_erase_btn.toggled.connect(
            lambda checked: self._on_final_mode_toggled("erase", checked)
        )
        self.final_panel_fill_btn.toggled.connect(
            lambda checked: self._on_final_mode_toggled("fill", checked)
        )
        self.move_design_btn.toggled.connect(self._on_move_design_toggled)
        self.move_design_toolbar_btn.toggled.connect(self._on_move_design_toggled)
        self.nudge_left_btn.clicked.connect(
            lambda: self._nudge_design(4.0, 0.0)
        )
        self.nudge_right_btn.clicked.connect(
            lambda: self._nudge_design(-4.0, 0.0)
        )
        self.nudge_up_btn.clicked.connect(
            lambda: self._nudge_design(0.0, 4.0)
        )
        self.nudge_down_btn.clicked.connect(
            lambda: self._nudge_design(0.0, -4.0)
        )
        self.zoom_design_in_btn.clicked.connect(
            lambda: self._zoom_design(1.08)
        )
        self.zoom_design_out_btn.clicked.connect(
            lambda: self._zoom_design(1 / 1.08)
        )
        self.center_design_btn.clicked.connect(self._center_design_position)
        self.reset_design_pos_btn.clicked.connect(self._reset_design_position)
        self.undo_btn.clicked.connect(self.undo_edit)
        self.redo_btn.clicked.connect(self.redo_edit)

        self.zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        self.zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        self.fit_btn.clicked.connect(self.canvas_fit)
        self.actual_size_btn.clicked.connect(self.canvas_actual_size)

        self.canvas.filesDropped.connect(self.handle_dropped_files)
        self.canvas.browseRequested.connect(self._browse_from_canvas)
        self.canvas.meshPointsChanged.connect(self._on_mesh_points_changed)
        self.canvas.exclusionContoursChanged.connect(
            self._on_exclusion_contours_changed
        )
        self.canvas.exclusionBrushStroke.connect(
            self._on_exclusion_brush_stroke
        )
        self.canvas.viewChanged.connect(self._on_view_changed)
        self.canvas.designPanDelta.connect(self._on_design_pan_delta)
        self.canvas.designPanFinished.connect(self._on_design_pan_finished)
        self.canvas.designZoomDelta.connect(self._on_design_zoom_delta)

    # -------------------------------------------------------------- loading

    def load_phone_image(self, file_path: Optional[str] = None) -> None:
        """Load the phone photo, then detect its cover region."""
        path = file_path or self._ask_open_path("Select Phone Photo")
        if not path:
            return

        try:
            image = ImageLoader.load_image(path)
        except ImageLoadError as exc:
            logger.warning("Phone load failed: %s", exc)
            self._error("Could not load phone image", str(exc))
            return

        self.progress_bar.setVisible(True)
        self.status_message("Detecting printable cover surface…")
        try:
            detected = self.compositor.set_phone_image(image)
        except Exception as exc:
            logger.exception("Phone setup failed")
            self.progress_bar.setVisible(False)
            self._error("Could not analyse phone image", str(exc))
            return
        finally:
            self.progress_bar.setVisible(False)

        self.phone_path = Path(path)
        self.user_settings.set_last_dir("phone", self.phone_path)
        self._edit_history.clear()
        self._update_history_buttons()
        self._mark_dirty()
        self._sync_sliders(self.compositor.get_settings())

        if self.compositor.phone_image is None:
            self._error("Could not analyse phone image", "No image data after load")
            return

        height, width = self.compositor.phone_image.shape[:2]
        self._set_badge(self.phone_badge, self.phone_path.name)
        self.phone_name_label.setText(f"{self.phone_path.name}  ·  {width}×{height}")

        self._sync_cover_to_canvas()
        self._update_enabled_state()

        if detected:
            confidence = int(round(self.compositor.detection_confidence * 100))
            if self.compositor.from_template:
                self.region_status.setText(f"Template {confidence}%")
                self.status_message(
                    f"Loaded {self.phone_path.name} · cover template reused"
                )
            else:
                self.region_status.setText(f"Cover surface {confidence}%")
                self.status_message(
                    f"Loaded {self.phone_path.name} · printable cover surface "
                    f"and {self.compositor.automatic_margin:.1f}% safety margin"
                )
        else:
            self.region_status.setText("Region: manual")
            self.status_message("Loaded phone photo · adjust the print area manually")

        if self.compositor.design_image is None:
            self.canvas.set_image(self.compositor.phone_image)
            self.canvas.set_show_cover(True)
            self.region_btn.setChecked(True)
            self._update_size_label(self.compositor.phone_image)
        else:
            self.request_render()

        self._store_mesh_baseline()

    def load_design_image(self, file_path: Optional[str] = None) -> None:
        """Load the artwork that gets printed on the cover."""
        path = file_path or self._ask_open_path("Select Design Image")
        if not path:
            return

        try:
            image = ImageLoader.load_image(path)
        except ImageLoadError as exc:
            logger.warning("Design load failed: %s", exc)
            self._error("Could not load design image", str(exc))
            return

        try:
            self.compositor.set_design_image(image)
        except Exception as exc:
            logger.exception("Design apply failed")
            self._error("Could not apply design", str(exc))
            return

        self.design_path = Path(path)
        self.user_settings.set_last_dir("design", self.design_path)
        self._mark_dirty()
        self._sync_sliders(self.compositor.get_settings())

        height, width = self.compositor.design_image.shape[:2]
        self._set_badge(self.design_badge, self.design_path.name)
        self.design_name_label.setText(f"{self.design_path.name}  ·  {width}×{height}")

        self._update_enabled_state()

        if self.compositor.phone_image is None:
            self.status_message("Design loaded · now load a phone photo")
            self.canvas.set_image(self.compositor.design_image)
            self._update_size_label(self.compositor.design_image)
            return

        fit = self.compositor.get_settings()
        self.status_message(
            f"Smart fit · scale {fit['design_scale']:.0f}% · "
            f"offset {fit['offset_x']:.0f}, {fit['offset_y']:.0f} · "
            f"rot {fit['rotation']:.1f}°"
        )
        self.request_render()

    def handle_dropped_files(self, paths: List[str]) -> None:
        """Route dropped files to the phone or design slot."""
        if not paths:
            return

        if len(paths) >= 2 and self.compositor.phone_image is None:
            self.load_phone_image(paths[0])
            self.load_design_image(paths[1])
            return

        path = paths[0]

        if self.compositor.phone_image is None:
            self.load_phone_image(path)
        elif self.compositor.design_image is None:
            self.load_design_image(path)
        else:
            box = QMessageBox(self)
            box.setWindowTitle("Replace image")
            box.setText(f"Where should <b>{Path(path).name}</b> go?")
            phone_button = box.addButton("Phone Photo", QMessageBox.ActionRole)
            design_button = box.addButton("Design", QMessageBox.ActionRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.exec()

            if box.clickedButton() == phone_button:
                self.load_phone_image(path)
            elif box.clickedButton() == design_button:
                self.load_design_image(path)

    def _browse_from_canvas(self) -> None:
        """Open the next missing image when the empty canvas is clicked."""
        if self.compositor.phone_image is None:
            self.load_phone_image()
        elif self.compositor.design_image is None:
            self.load_design_image()
        else:
            self.load_design_image()

    def swap_images(self) -> None:
        """Swap the phone photo and the design."""
        if self.compositor.phone_image is None or self.compositor.design_image is None:
            self.status_message("Load both images before swapping")
            return

        phone_path, design_path = self.phone_path, self.design_path
        if phone_path is None or design_path is None:
            return

        self.load_phone_image(str(design_path))
        self.load_design_image(str(phone_path))
        self.status_message("Images swapped")

    def clear_all(self) -> None:
        """Drop both images and start over."""
        if self._dirty:
            answer = QMessageBox.question(
                self, "Clear session?",
                "Clear the current phone, design and adjustments?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.compositor.clear()

        self.phone_path = None
        self.design_path = None
        self.project_path = None
        self.current_preview = None
        self._dirty = False
        self._edit_history.clear()
        self._mesh_baseline = None
        self._update_history_buttons()

        self.canvas.clear_image()
        self.canvas.set_mesh_points(None)
        self.canvas.set_exclusion_contours([])

        self._set_badge(self.phone_badge, "no phone", muted=True)
        self._set_badge(self.design_badge, "no design", muted=True)

        self.phone_name_label.setText("Not loaded")
        self.design_name_label.setText("Not loaded")
        self.size_label.setText("")
        self.render_label.setText("")
        self.region_status.setText("No region")

        self._syncing = True
        self._sync_sliders(self.compositor.get_settings())
        self.mirror_check.setChecked(False)
        self.fit_mode_combo.setCurrentIndex(0)
        self.preset_combo.setCurrentIndex(0)
        self.compare_btn.setChecked(False)
        self.region_btn.setChecked(False)
        self.edit_region_btn.setChecked(False)
        self._syncing = False

        self._update_enabled_state()
        self.status_message("Cleared")

    def _ask_open_path(self, title: str) -> Optional[str]:
        """Show an open dialog and return the chosen path."""
        key = "phone" if "Phone" in title else "design"
        start_dir = self.user_settings.last_dir(
            key,
            self.phone_path.parent if self.phone_path else None,
        )
        path, _ = QFileDialog.getOpenFileName(
            self, title, start_dir, ImageLoader.FILE_FILTER
        )
        return path or None

    # ------------------------------------------------------------ rendering

    def request_render(self) -> None:
        """Schedule a debounced re-render of the preview."""
        if not self.compositor.is_ready:
            if self.compositor.phone_image is not None:
                self.canvas.set_image(self.compositor.phone_image)
            return

        if not self._design_pan_gesture:
            self.progress_bar.setVisible(True)
        self.render_timer.start()

    def _begin_live_placement(self) -> None:
        """Zero debounce so pan/zoom samples the original artwork every move."""
        self._design_pan_gesture = True
        self.render_timer.setInterval(0)
        self.placement_idle_timer.stop()

    def _end_live_placement(self) -> None:
        """Restore normal debounce after the pointer stops moving."""
        if self.design_pan_timer.isActive():
            return
        self._design_pan_gesture = False
        self.render_timer.setInterval(self.RENDER_DEBOUNCE_MS)

    def _start_render(self) -> None:
        """Hand the current state to the render thread."""
        if not self.compositor.is_ready:
            self.progress_bar.setVisible(False)
            return

        self._render_token += 1
        edit_mesh = bool(self.edit_region_btn.isChecked())
        if edit_mesh:
            self.progress_bar.setVisible(False)
            if self.compositor.phone_image is not None:
                self.canvas.set_image(self.compositor.phone_image)
            return
        self.render_thread.request(self.PREVIEW_MAX, self._render_token)

    def _on_rendered(self, image, token: int, elapsed_ms: float) -> None:
        """Display a finished render, ignoring superseded ones."""
        if token != self._render_token:
            return

        self.progress_bar.setVisible(False)

        if image is None:
            return

        self.current_preview = image
        # While editing mesh/cutouts, keep the source phone on canvas so
        # handles land exactly where the user places them (composite warp
        # would make dots look like they "jump").
        if self.edit_region_btn.isChecked():
            if self.compositor.phone_image is not None:
                self.canvas.set_image(self.compositor.phone_image)
        elif not self.compare_btn.isChecked():
            if self.region_btn.isChecked():
                self.region_btn.setChecked(False)
            else:
                self.canvas.set_show_cover(False)
            self.canvas.set_image(image)

        self._update_size_label(self.compositor.phone_image)
        self.render_label.setText(f"rendered in {elapsed_ms:.0f} ms")

    def _on_render_failed(self, message: str, token: int) -> None:
        """Report a render failure without killing the session."""
        if token != self._render_token:
            return

        self.progress_bar.setVisible(False)
        logger.error("Render failed: %s", message)
        self.status_message(f"Render failed: {message}")

    def _update_size_label(self, image) -> None:
        """Show the source resolution in the info strip."""
        if image is None:
            self.size_label.setText("")
            return

        height, width = image.shape[:2]
        self.size_label.setText(f"output {width}×{height}")

    # -------------------------------------------------------------- controls

    def _on_slider_changed(self, key: str, value: float) -> None:
        """Push a slider value into the compositor and re-render."""
        if self._syncing:
            return

        self._push_history(f"adjust {key}", coalesce_key=f"slider:{key}")
        self.compositor.update_settings({key: value})
        self._mark_dirty()

        if key in ('region_inset',):
            self._sync_cover_to_canvas()

        self.request_render()

    def _reset_group(self, group: str) -> None:
        """Reset every slider in one tab."""
        updates = {}

        self._syncing = True
        for spec in SLIDER_GROUPS[group]:
            key = spec[0]
            row = self.sliders[key]
            row.set_value(row.default)
            updates[key] = row.default
        self._syncing = False

        self.compositor.update_settings(updates)
        self._sync_cover_to_canvas()
        self.request_render()
        self.status_message(f"{group.capitalize()} reset")

    def reset_adjustments(self) -> None:
        """Reset all sliders, the fit mode and the mirror flag."""
        self._push_history("reset adjustments")
        self.compositor.reset()

        self._syncing = True
        self._sync_sliders(self.compositor.get_settings())
        self.mirror_check.setChecked(False)
        self.fit_mode_combo.setCurrentIndex(0)
        self.preset_combo.setCurrentIndex(0)
        self._syncing = False

        self._sync_cover_to_canvas()
        self.request_render()
        self.status_message("All adjustments reset")

    def apply_preset(self, name: str) -> None:
        """Apply a named look and sync the sliders to it."""
        if self._syncing or name not in PRESETS:
            return

        self._push_history(f"preset {name}")
        settings = self.compositor.apply_preset(name)
        self._mark_dirty()

        self._syncing = True
        self._sync_sliders(settings)
        if self.preset_combo.currentText() != name:
            self.preset_combo.setCurrentText(name)
        self._syncing = False

        self.request_render()
        self.status_message(f"Preset applied: {name}")

    def _sync_sliders(self, settings: Dict[str, float]) -> None:
        """Update every slider widget from a settings dictionary."""
        was_syncing = self._syncing
        self._syncing = True

        for key, row in self.sliders.items():
            row.set_value(settings.get(key, DEFAULT_SETTINGS.get(key, row.default)))

        self._syncing = was_syncing

    def _on_fit_mode_changed(self, index: int) -> None:
        """Change how the design maps into the print area."""
        if self._syncing:
            return

        self._push_history("fit mode")
        self.compositor.set_fit_mode(self.fit_mode_combo.itemData(index))
        self._mark_dirty()
        self.request_render()

    def _on_mirror_toggled(self, checked: bool) -> None:
        """Mirror the design horizontally."""
        if self._syncing:
            return

        self._push_history("mirror")
        self.compositor.set_mirror(checked)
        self._mark_dirty()
        self.request_render()

    def _on_compare_toggled(self, checked: bool) -> None:
        """Swap between the original photo and the composite."""
        if checked:
            if self.compositor.phone_image is not None:
                self.canvas.set_image(self.compositor.phone_image)
                self.status_message("Showing the original photo")
        else:
            if self.current_preview is not None:
                self.canvas.set_image(self.current_preview)
            else:
                self.request_render()
            self.status_message("Showing the mockup")

    def _on_edit_region_toggled(self, checked: bool) -> None:
        """Enable independent editing of every print-mesh vertex."""
        if checked:
            if self.final_btn.isChecked() or self.final_panel_btn.isChecked():
                self._set_final_enabled(False)
            if self.move_design_btn.isChecked():
                self.move_design_btn.blockSignals(True)
                self.move_design_toolbar_btn.blockSignals(True)
                self.move_design_btn.setChecked(False)
                self.move_design_toolbar_btn.setChecked(False)
                self.move_design_btn.blockSignals(False)
                self.move_design_toolbar_btn.blockSignals(False)
                self.canvas.set_design_pan_mode(False)
        if not checked:
            # Commit any in-progress canvas geometry before leaving edit mode.
            self._commit_canvas_geometry()
            self.canvas.set_edit_cover(False)
            if self.erase_wrap_btn.isChecked():
                self.erase_wrap_btn.setChecked(False)
            self.region_hint.setText(
                "Turn on Edit Mesh to adjust edges · Move Design to slide art · "
                "Final (Erase/Fill) to polish the wrap after finish."
            )
            self.request_render()
            return

        self.canvas.set_edit_cover(True)
        self.region_btn.setChecked(True)
        # Edit against the real phone photo so dots match finger placement.
        if abs(float(self.compositor.settings.get("region_inset", 0.0))) > 1e-6:
            self._syncing = True
            self.sliders["region_inset"].set_value(0.0)
            self._syncing = False
            self.compositor.update_settings({"region_inset": 0.0})
        if self.compositor.phone_image is not None:
            self.canvas.set_image(self.compositor.phone_image)
        # Always re-push saved mesh + cutouts (never a blank / stale overlay).
        self._sync_cover_to_canvas()
        self.region_hint.setText(
            "Edit Mesh: drag corners — they stay put · "
            "Perfect Finish (Edges) = full phone wrap · × deletes cutouts."
        )
        self.status_message("Mesh editing on — 4 corners control the wrap")

    def _commit_canvas_geometry(self) -> None:
        """Flush canvas mesh/cutouts into the compositor if they differ."""
        if self.compositor.phone_image is None:
            return
        height, width = self.compositor.phone_image.shape[:2]
        mesh = self.canvas.mesh_points()
        if mesh is not None and self.canvas._mesh_rows >= 2:
            points = np.asarray(mesh, dtype=np.float32).copy()
            points[:, 0] *= width
            points[:, 1] *= height
            current = self.compositor.get_control_mesh()
            needs = (
                current is None
                or current.rows != self.canvas._mesh_rows
                or current.cols != self.canvas._mesh_cols
                or not np.allclose(
                    current.points.astype(np.float32), points, atol=0.35
                )
            )
            if needs:
                self.compositor.set_mesh_points(
                    points, self.canvas._mesh_rows, self.canvas._mesh_cols
                )

        contours = self.canvas.exclusion_contours()
        pixel = []
        for contour in contours:
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2).copy()
            pts[:, 0] *= width
            pts[:, 1] *= height
            pixel.append(pts)
        # Never wipe compositor cutouts with an empty canvas by accident.
        if pixel:
            # Pass per-cutout tool tags — without them paint reclassifies
            # camera AABBs into giant circles and ignores the user's shape.
            tags = list(self.canvas.exclusion_shapes())
            while len(tags) < len(pixel):
                tags.append(str(self.canvas.cutout_shape() or "rounded_rect"))
            self.compositor.set_hardware_exclusions(
                pixel,
                snap_geometry=False,
                allow_clear=False,
                shape_tags=tags[: len(pixel)],
                persist=False,
                refit_design=False,
            )
        elif not self.compositor.hardware_contours:
            self.compositor.set_hardware_exclusions(
                [], snap_geometry=False, allow_clear=True
            )

    def _on_cutout_shape_changed(self, _index: int = 0) -> None:
        """Apply the combo shape to the active cutout and keep it locked."""
        shape = str(self.cutout_shape_combo.currentData() or "rounded_rect")
        self.canvas.set_cutout_shape(shape)
        # Keep corner % in sync so Rounded Rect / Squircle rebuild matches UI.
        try:
            self.canvas._cutout_corner_frac = (
                float(self.cutout_corner_spin.value()) / 100.0
            )
            self.canvas._cutout_rotation_deg = float(self.cutout_rot_spin.value())
        except Exception:
            pass
        # Existing red selection must switch to this tool immediately —
        # otherwise only new Shift+clicks used the combo and render kept
        # the old (often mis-classified) hole.
        if self.canvas.apply_selected_cutout_shape(shape):
            self.status_message(
                f"Cutout shape: {self.cutout_shape_combo.currentText()}"
            )
        else:
            self.status_message(
                f"Cutout tool: {self.cutout_shape_combo.currentText()} "
                "(Shift+click to add)"
            )

    def _apply_cutout_edit(self) -> None:
        """Apply corner radius + rotation to the active cutout (non-destructive)."""
        if self.compositor.phone_image is None:
            self.status_message("Load a phone photo first")
            return
        corner = float(self.cutout_corner_spin.value()) / 100.0
        rotation = float(self.cutout_rot_spin.value())
        changed = self.canvas.apply_cutout_style(
            corner_frac=corner, rotation_deg=rotation
        )
        if not changed:
            self.status_message("Hover or select a cutout first")
            return
        # exclusionContoursChanged → compositor (exact user geometry preserved).
        self.status_message(
            f"Cutout updated · r={self.cutout_corner_spin.value():.0f}% "
            f"rot={rotation:.0f}°"
        )

    def _scale_cutouts(self, factor: float) -> None:
        """Grow/shrink Perfect Finish curves without changing their shape."""
        if self.compositor.phone_image is None:
            return
        if not self.compositor.hardware_contours:
            self.status_message("No cutouts to resize — add or Perfect Finish camera first")
            return
        self._push_history("cutout scale")
        count = self.compositor.scale_hardware_cutouts(factor)
        self._sync_cover_to_canvas()
        self._mark_dirty()
        self.request_render()
        pct = int(round((factor - 1.0) * 100))
        sign = "+" if pct >= 0 else ""
        self.region_status.setText(f"Cutouts scaled {sign}{pct}% · {count}")
        self.status_message(
            f"Cutout size {sign}{pct}% — curves unchanged"
        )

    def perfect_finish_cutouts(self) -> None:
        """Apply Perfect Finish only to the selected scope."""
        if self.compositor.phone_image is None:
            self.status_message("Load a phone photo first")
            return

        # Commit mesh only — never let an empty canvas wipe cutouts before finish.
        if self.edit_region_btn.isChecked():
            self._commit_canvas_geometry()
            # If canvas lost cutouts but compositor still has them, re-push.
            if (
                not self.canvas.exclusion_contours()
                and self.compositor.hardware_contours
            ):
                self._sync_cover_to_canvas()

        scope = str(self.finish_scope_combo.currentData() or "edges")
        label = self.finish_scope_combo.currentText()
        before_cutouts = [
            np.asarray(c, dtype=np.float32).copy()
            for c in (self.compositor.hardware_contours or [])
        ]
        before_mask = (
            None
            if self.compositor.exclusion_mask is None
            else self.compositor.exclusion_mask.copy()
        )
        self._push_history(f"perfect finish ({scope})")
        if scope == "all":
            count = self.compositor.heal_realistic_wrap(include_hardware=True)
        else:
            count = self.compositor.perfect_finish_cutouts(scope=scope)
        after_n = len(self.compositor.hardware_contours or [])
        restored = False
        if (
            scope in ("camera", "all")
            and before_cutouts
            and after_n < len(before_cutouts)
        ):
            self.compositor.hardware_contours = before_cutouts
            self.compositor.exclusion_mask = before_mask
            if before_mask is None and before_cutouts:
                self.compositor.set_hardware_exclusions(
                    [c.reshape(-1, 2) for c in before_cutouts],
                    snap_geometry=False,
                    allow_clear=False,
                )
            else:
                self.compositor._sync_printable_from_mesh()
            after_n = len(self.compositor.hardware_contours or [])
            restored = True
        self._sync_cover_to_canvas()
        self.region_btn.setChecked(True)
        self._mark_dirty()
        if self.edit_region_btn.isChecked() and self.compositor.phone_image is not None:
            self.canvas.set_image(self.compositor.phone_image)
        self.request_render()
        self.region_status.setText(
            f"Finish · {label} · {count} · cutouts {after_n}"
        )
        if restored:
            self.status_message(
                "Camera finish kept your cutouts (blocked an accidental wipe)"
            )
        else:
            self.status_message(f"Perfect Finish applied: {label}")

    def _on_erase_wrap_toggled(self, checked: bool) -> None:
        """Toggle paint-to-erase wrap brush (auto-enables Edit Mesh)."""
        if checked:
            self._set_final_enabled(False)
            if not self.edit_region_btn.isChecked():
                self.edit_region_btn.setChecked(True)
        self.canvas.set_erase_mode(checked)
        if checked:
            self.region_hint.setText(
                "Erase Wrap: paint over buttons / holes to remove print. "
                "Brush size [ ] or Alt+scroll. Shift+click Capsule also works."
            )
            self.status_message("Erase Wrap on — paint where wrap should go")
        else:
            self.status_message("Erase Wrap off")

    def _sync_final_buttons(self, *, master: bool, mode: Optional[str]) -> None:
        """Keep toolbar + panel Final toggles aligned without re-entrancy."""
        buttons = (
            (self.final_btn, master),
            (self.final_panel_btn, master),
            (self.final_erase_btn, master and mode == "erase"),
            (self.final_fill_btn, master and mode == "fill"),
            (self.final_panel_erase_btn, master and mode == "erase"),
            (self.final_panel_fill_btn, master and mode == "fill"),
        )
        for btn, state in buttons:
            btn.blockSignals(True)
            btn.setChecked(bool(state))
            btn.blockSignals(False)
        for btn in (
            self.final_erase_btn,
            self.final_fill_btn,
            self.final_panel_erase_btn,
            self.final_panel_fill_btn,
        ):
            btn.setEnabled(master)

    def _set_final_enabled(self, enabled: bool, mode: str = "erase") -> None:
        """Turn Final polish on/off and push brush mode to the canvas."""
        if enabled:
            if self.edit_region_btn.isChecked():
                self.edit_region_btn.setChecked(False)
            if self.erase_wrap_btn.isChecked():
                self.erase_wrap_btn.blockSignals(True)
                self.erase_wrap_btn.setChecked(False)
                self.erase_wrap_btn.blockSignals(False)
                self.canvas.set_erase_mode(False)
            if self.move_design_btn.isChecked():
                self.move_design_btn.setChecked(False)
            kind = mode if mode in ("erase", "fill") else "erase"
            self._sync_final_buttons(master=True, mode=kind)
            self.canvas.set_final_brush_mode(kind)
            self.region_btn.setChecked(True)
            if kind == "fill":
                self.region_hint.setText(
                    "Final Fill: paint where wrap should return. "
                    "Only the real cover is restored. Thin teal line = selection. "
                    "Brush [ ] / Alt+scroll."
                )
                self.status_message("Final Fill — paint to restore wrap on cover")
            else:
                self.region_hint.setText(
                    "Final Erase: paint to clear wrap smoothly. "
                    "Thin red line shows your stroke. Brush [ ] / Alt+scroll."
                )
                self.status_message("Final Erase — paint to clear wrap")
        else:
            self._sync_final_buttons(master=False, mode=None)
            self.canvas.set_final_brush_mode(None)
            self.status_message("Final polish off")

    def _on_final_toggled(self, checked: bool) -> None:
        """Master Final toggle (toolbar or Placement panel)."""
        if checked:
            current = self.canvas.final_brush_mode() or "erase"
            self._set_final_enabled(True, current)
        else:
            self._set_final_enabled(False)

    def _on_final_mode_toggled(self, mode: str, checked: bool) -> None:
        """Switch between Final Erase and Final Fill."""
        if not checked:
            # Keep the active mode selected while Final is on.
            if self.final_btn.isChecked() or self.final_panel_btn.isChecked():
                current = self.canvas.final_brush_mode() or mode
                self._sync_final_buttons(master=True, mode=current)
            return
        self._set_final_enabled(True, mode)

    def _on_move_design_toggled(self, checked: bool) -> None:
        """Toggle drag-to-pan the design artwork on the preview."""
        sender = self.sender()
        # Keep Placement + toolbar toggles in sync.
        for btn in (self.move_design_btn, self.move_design_toolbar_btn):
            if btn is sender:
                continue
            btn.blockSignals(True)
            btn.setChecked(checked)
            btn.blockSignals(False)

        if checked and self.edit_region_btn.isChecked():
            self.edit_region_btn.setChecked(False)
        if checked and (self.final_btn.isChecked() or self.final_panel_btn.isChecked()):
            self._set_final_enabled(False)

        self.canvas.set_design_pan_mode(checked)
        if checked:
            self.region_hint.setText(
                "Move Design: drag to slide · scroll to zoom print · "
                "+/− buttons · Ctrl+scroll zooms the view."
            )
            self.status_message("Move Design on — drag / scroll to place artwork")
            self.tabs.setCurrentIndex(0)
        else:
            self.status_message("Move Design off")

    def _nudge_design(self, dx: float, dy: float) -> None:
        """Nudge design offsets by slider units."""
        if self.compositor.design_image is None:
            self.status_message("Load a design first")
            return
        self._push_history("nudge design", coalesce_key="nudge-design")
        ox = float(self.compositor.settings.get("offset_x", 0.0)) + dx
        oy = float(self.compositor.settings.get("offset_y", 0.0)) + dy
        ox = float(np.clip(ox, -180.0, 180.0))
        oy = float(np.clip(oy, -180.0, 180.0))
        self.compositor.update_settings({"offset_x": ox, "offset_y": oy})
        self._syncing = True
        if "offset_x" in self.sliders:
            self.sliders["offset_x"].set_value(ox)
        if "offset_y" in self.sliders:
            self.sliders["offset_y"].set_value(oy)
        self._syncing = False
        self._mark_dirty()
        self.request_render()

    def _zoom_design(self, factor: float) -> None:
        """Multiply design_scale (percent) by factor and re-render."""
        if self.compositor.design_image is None:
            self.status_message("Load a design first")
            return
        was_live = self._design_pan_gesture
        self._begin_live_placement()
        if not was_live:
            self._push_history("zoom design", coalesce_key="design-zoom")
        scale = float(self.compositor.settings.get("design_scale", 100.0))
        scale = float(np.clip(scale * float(factor), 25.0, 400.0))
        self.compositor.update_settings({"design_scale": scale})
        self._syncing = True
        if "design_scale" in self.sliders:
            self.sliders["design_scale"].set_value(scale)
        self._syncing = False
        self._mark_dirty()
        self.request_render()
        self.placement_idle_timer.start()
        self.status_message(f"Design zoom {scale:.0f}%")

    def _on_design_zoom_delta(self, factor: float) -> None:
        """Scroll-wheel zoom while Move Design is active."""
        self._zoom_design(factor)

    def _on_design_pan_delta(self, dx: float, dy: float) -> None:
        """Accumulate drag deltas; flush on a short timer for smooth motion."""
        if self.compositor.design_image is None:
            return
        if not self._design_pan_gesture:
            self._push_history("move design", coalesce_key="design-pan")
            self._begin_live_placement()
            self._design_pan_pending = (0.0, 0.0)

        px, py = self._design_pan_pending
        self._design_pan_pending = (px + float(dx), py + float(dy))
        if not self.design_pan_timer.isActive():
            self.design_pan_timer.start()

    def _flush_design_pan(self) -> None:
        """Apply coalesced Move Design deltas in one settings update."""
        dx, dy = self._design_pan_pending
        self._design_pan_pending = (0.0, 0.0)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        # Grab-the-print; ×220 ≈ smooth 1:1 with a little extra travel.
        ox = float(self.compositor.settings.get("offset_x", 0.0)) - dx * 220.0
        oy = float(self.compositor.settings.get("offset_y", 0.0)) - dy * 220.0
        ox = float(np.clip(ox, -180.0, 180.0))
        oy = float(np.clip(oy, -180.0, 180.0))
        self.compositor.update_settings({"offset_x": ox, "offset_y": oy})
        self._mark_dirty()
        self.request_render()

    def _on_design_pan_finished(self) -> None:
        """Sync sliders after a pan drag without a second catch-up render."""
        if self.design_pan_timer.isActive():
            self.design_pan_timer.stop()
            self._flush_design_pan()
        self._end_live_placement()
        ox = float(self.compositor.settings.get("offset_x", 0.0))
        oy = float(self.compositor.settings.get("offset_y", 0.0))
        scale = float(self.compositor.settings.get("design_scale", 100.0))
        self._syncing = True
        if "offset_x" in self.sliders:
            self.sliders["offset_x"].set_value(ox)
        if "offset_y" in self.sliders:
            self.sliders["offset_y"].set_value(oy)
        if "design_scale" in self.sliders:
            self.sliders["design_scale"].set_value(scale)
        self._syncing = False
        self.status_message(
            f"Design · offset {ox:.0f}, {oy:.0f} · zoom {scale:.0f}%"
        )

    def _center_design_position(self) -> None:
        """Zero horizontal/vertical design offsets."""
        if self.compositor.design_image is None:
            self.status_message("Load a design first")
            return
        self._push_history("center design")
        self.compositor.update_settings({"offset_x": 0.0, "offset_y": 0.0})
        self._syncing = True
        if "offset_x" in self.sliders:
            self.sliders["offset_x"].set_value(0.0)
        if "offset_y" in self.sliders:
            self.sliders["offset_y"].set_value(0.0)
        self._syncing = False
        self._mark_dirty()
        self.request_render()
        self.status_message("Design centred")

    def _reset_design_position(self) -> None:
        """Reset zoom, pan, and rotation of the design."""
        if self.compositor.design_image is None:
            self.status_message("Load a design first")
            return
        self._push_history("reset design position")
        updates = {
            "design_scale": 100.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "rotation": 0.0,
        }
        self.compositor.update_settings(updates)
        self._syncing = True
        for key, value in updates.items():
            if key in self.sliders:
                self.sliders[key].set_value(value)
        self._syncing = False
        self._mark_dirty()
        self.request_render()
        self.status_message("Design position reset")

    def _on_exclusion_brush_stroke(self, stroke) -> None:
        """Apply a painted erase or fill stroke on the wrap."""
        if self.compositor.phone_image is None or not stroke:
            return
        height, width = self.compositor.phone_image.shape[:2]
        # Map brush size via displayed image scale (not min-side frac) so zoom
        # and letterboxing do not turn a small ring into a huge phone wipe.
        disp = self.canvas._image_rect()
        scale_x = float(width) / max(float(disp.width()), 1.0)
        scale_y = float(height) / max(float(disp.height()), 1.0)
        scale = 0.5 * (scale_x + scale_y)
        dabs = []
        for nx, ny, r_norm in stroke:
            # r_norm is fraction of displayed min-side; convert via same scale.
            brush_px = float(r_norm) * float(
                min(max(disp.width(), 1.0), max(disp.height(), 1.0))
            )
            phone_r = max(1.5, brush_px * scale)
            # Final Erase stays precise; Final Fill a bit larger for corners.
            mode = self.canvas.final_brush_mode()
            if mode == "erase":
                phone_r *= 0.9
            elif mode == "fill":
                phone_r *= 1.15
            dabs.append(
                (
                    float(nx) * width,
                    float(ny) * height,
                    phone_r,
                )
            )
        kind = self.canvas.brush_kind() or "erase"
        if kind == "fill":
            self._push_history("final fill", coalesce_key="final-fill")
            painted = self.compositor.clear_exclusion_dabs(dabs)
            action = "Filled wrap"
            detail = (
                "wrap restored"
                if painted
                else "paint on the corner / gap — Fill restores wrap on the phone"
            )
        else:
            label = (
                "final erase"
                if self.canvas.final_brush_mode() == "erase"
                else "erase wrap"
            )
            self._push_history(label, coalesce_key="final-erase")
            painted = self.compositor.paint_exclusion_dabs(dabs)
            action = "Erased wrap"
            detail = "original phone shows through"
        self._sync_cover_to_canvas()
        self._mark_dirty()
        self.request_render()
        self.region_status.setText(
            f"{action} · {painted} dabs · "
            f"cutouts {len(self.compositor.hardware_contours or [])}"
        )
        self.status_message(f"{action} — {detail}")

    def _on_mesh_points_changed(
        self, normalised, rows: int, cols: int
    ) -> None:
        """Store independently edited mesh vertices in source-image pixels."""
        if self.compositor.phone_image is None:
            return

        self._push_history("mesh edit")
        height, width = self.compositor.phone_image.shape[:2]
        points = np.asarray(normalised, dtype=np.float32).copy()
        points[:, 0] *= width
        points[:, 1] *= height

        # The user edited the effective print area on the canvas, so store those
        # points as the base region and clear the inset slider.
        if abs(float(self.compositor.settings.get('region_inset', 0.0))) > 1e-6:
            self._syncing = True
            self.sliders['region_inset'].set_value(0.0)
            self._syncing = False
            self.compositor.update_settings({'region_inset': 0.0})

        self.compositor.set_mesh_points(points, rows, cols)
        # Round-trip from control mesh so overlay == saved geometry.
        self._sync_cover_to_canvas()
        self._mark_dirty()
        self.region_status.setText(f"Mesh: {cols}×{rows} saved")
        self.request_render()
        self.status_message(
            "Wrap corners saved — stay where you drag them. "
            "Perfect Finish (Edges) snaps to the phone rim."
        )

    def _on_exclusion_contours_changed(self, normalised_contours) -> None:
        """Apply manually edited cutouts — keep exact user geometry (no snap-back)."""
        if self.compositor.phone_image is None:
            return

        height, width = self.compositor.phone_image.shape[:2]
        contours = []
        for contour in normalised_contours:
            points = np.asarray(contour, dtype=np.float32).reshape(-1, 2).copy()
            points[:, 0] *= width
            points[:, 1] *= height
            contours.append(points)

        had = len(self.compositor.hardware_contours or [])
        allow_clear = len(contours) == 0 and had > 0

        self._push_history("cutout edit")
        # Do NOT auto Perfect-Finish on drag release — that refit stadiums/circles
        # and made handles jump back. Smooth finish is only via Perfect Finish btn.
        tags = []
        try:
            tags = list(self.canvas.exclusion_shapes())
        except Exception:
            tags = []
        corner = 0.16
        try:
            corner = float(self.cutout_corner_spin.value()) / 100.0
        except Exception:
            pass
        self.compositor.set_hardware_exclusions(
            contours,
            snap_geometry=False,
            allow_clear=allow_clear,
            shape_tags=tags,
            corner_frac=corner,
            persist=False,
            refit_design=False,
        )
        self._sync_cover_to_canvas()
        self._mark_dirty()
        n = len(self.compositor.hardware_contours)
        self.region_status.setText(f"Cutouts: {n} saved")
        self.request_render()
        if allow_clear and n == 0:
            self.status_message("Cutout removed")
        else:
            self.status_message("Cutout saved — drag stays where you put it")

    def _sync_cover_to_canvas(self) -> None:
        """Push mesh vertices and hardware exclusions to the canvas."""
        # Always sync the editable control mesh (not inset-shifted), so dots
        # stay exactly where the user placed them.
        mesh = self.compositor.get_control_mesh()

        if mesh is None or self.compositor.phone_image is None:
            self.canvas.set_mesh_points(None)
            self.canvas.set_exclusion_contours([])
            return

        height, width = self.compositor.phone_image.shape[:2]
        self.canvas.set_mesh_points(
            mesh.normalized_points(width, height), mesh.rows, mesh.cols
        )

        contours = []
        for contour in self.compositor.hardware_contours:
            normalised = np.asarray(
                contour, dtype=np.float32
            ).reshape(-1, 2)
            normalised = normalised.copy()
            normalised[:, 0] /= max(width, 1)
            normalised[:, 1] /= max(height, 1)
            contours.append(normalised)

        # Mesh-only saves must not wipe cutout overlays. Recover from the
        # exclusion mask, then fall back to whatever the canvas already shows.
        if not contours and self.compositor.exclusion_mask is not None:
            from ..image_processing.region_detector import HardwareRegionDetector

            recovered = HardwareRegionDetector._smooth_exclusion_contours(
                self.compositor.exclusion_mask
            )
            for contour in recovered:
                normalised = np.asarray(
                    contour, dtype=np.float32
                ).reshape(-1, 2).copy()
                if len(normalised) < 3:
                    continue
                normalised[:, 0] /= max(width, 1)
                normalised[:, 1] /= max(height, 1)
                contours.append(normalised)
            if contours and not self.compositor.hardware_contours:
                # Restore editable polygons so Perfect Finish / delete keep working.
                pixel = []
                for contour in contours:
                    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2).copy()
                    pts[:, 0] *= width
                    pts[:, 1] *= height
                    pixel.append(pts)
                self.compositor.hardware_contours = [
                    p.reshape(-1, 1, 2) for p in pixel
                ]

        if not contours:
            keep = self.canvas.exclusion_contours()
            if keep:
                contours = keep
                self.canvas.set_exclusion_contours(
                    contours, shapes=self.canvas.exclusion_shapes()
                )
                return

        # Push locked editor shape tags from compositor so capsule/rectangle
        # never get re-inferred into a different tool after paint/sync.
        shapes = list(getattr(self.compositor, "cutout_shape_tags", []) or [])
        if not any(shapes) and self.compositor.cutout_specs:
            shapes = []
            for spec in self.compositor.cutout_specs:
                tag = str(getattr(spec, "shape_tag", "") or "").strip()
                if not tag:
                    geom = str(getattr(spec, "geom", "") or "").lower()
                    tag = {
                        "circle": "circle",
                        "stadium": "capsule",
                        "rounded_rect": "rounded_rect",
                        "rectangle": "rectangle",
                        "contour": "custom_path",
                    }.get(geom, "")
                shapes.append(tag)
        self.canvas.set_exclusion_contours(contours, shapes=shapes or None)

    def auto_detect_region(self) -> None:
        """Run cover detection again on the current phone photo."""
        if self.compositor.phone_image is None:
            self.status_message("Load a phone photo first")
            return

        self._push_history("auto detect")
        self.compositor.redetect_cover()
        self._sync_sliders(self.compositor.get_settings())
        self._sync_cover_to_canvas()
        self.region_btn.setChecked(True)
        mesh = self.compositor.get_control_mesh()
        self.region_status.setText(
            f"Mesh: {mesh.cols}×{mesh.rows} detected"
            if mesh is not None else "Mesh detected"
        )
        self._mark_dirty()
        self.request_render()
        self.status_message("Printable cover surface and hardware exclusions detected")

    def center_region(self) -> None:
        """Use a centered phone-shaped print area."""
        if self.compositor.phone_image is None:
            self.status_message("Load a phone photo first")
            return

        self._push_history("center region")
        self.compositor.reset_cover_to_default()
        self._sync_sliders(self.compositor.get_settings())
        self._sync_cover_to_canvas()
        self.region_btn.setChecked(True)
        mesh = self.compositor.get_control_mesh()
        self.region_status.setText(
            f"Mesh: {mesh.cols}×{mesh.rows} centered"
            if mesh is not None else "Mesh centered"
        )
        self.request_render()
        self._mark_dirty()
        self.status_message("Print area centered")

    def _store_mesh_baseline(self) -> None:
        """Remember mesh/cutouts right after phone load (or project open)."""
        if self.compositor.phone_image is None:
            self._mesh_baseline = None
            return
        self._mesh_baseline = self._capture_edit_snapshot("baseline")
        self._update_enabled_state()

    def reset_mesh_to_start(self) -> None:
        """Restore mesh + cutouts to the phone-load starting state (with confirm)."""
        if self.compositor.phone_image is None:
            self.status_message("Load a phone photo first")
            return
        if self._mesh_baseline is None:
            self.status_message("No starting mesh saved yet — reload the phone photo")
            return

        answer = QMessageBox.warning(
            self,
            "Reset Mesh?",
            "Mesh aur cutouts wapas usi starting state pe aa jayenge "
            "jo phone load hote hi tha.\n\n"
            "Abhi jo edits / Perfect Finish changes hain wo hat jayenge.\n\n"
            "Reset karna hai?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.status_message("Reset cancelled")
            return

        self._push_history("reset mesh to start")
        self._apply_edit_snapshot(self._mesh_baseline)
        self.region_btn.setChecked(True)
        self._mark_dirty()
        self.request_render()
        mesh = self.compositor.get_control_mesh()
        if mesh is not None:
            self.region_status.setText(
                f"Reset · mesh {mesh.cols}×{mesh.rows} (start)"
            )
        self.status_message("Mesh reset to starting state")

    def canvas_fit(self) -> None:
        """Fit the preview to the viewport."""
        self.canvas.fit_to_view()

    def canvas_actual_size(self) -> None:
        """Show the preview at 100%."""
        self.canvas.reset_view()

    def _on_view_changed(self, zoom: float) -> None:
        """Mirror the canvas zoom in the status bar."""
        self.zoom_status.setText(f"{zoom * 100:.0f}%")

    # ---------------------------------------------------------------- output

    def open_batch_production(self) -> None:
        """
        Open the batch production dialog.

        Requires a loaded phone (cover geometry). Designs are taken from a
        folder; the main window layout is unchanged.
        """
        if self.compositor.phone_image is None or self.compositor.control_mesh is None:
            self.status_message("Load a phone photo first to start a batch")
            QMessageBox.information(
                self, "Batch Production",
                "Load a phone image first so the cover template is ready.\n"
                "Then choose a folder of designs to process automatically.",
            )
            return

        if self._export_thread is not None and self._export_thread.isRunning():
            self.status_message("Finish the current export before starting a batch")
            return

        if self._batch_dialog is not None and self._batch_dialog.isVisible():
            self._batch_dialog.raise_()
            self._batch_dialog.activateWindow()
            return

        self._batch_dialog = BatchDialog(self.compositor, self)
        self._batch_dialog.finished.connect(self._on_batch_dialog_closed)
        self.status_message("Batch production — processing designs offline")
        self._batch_dialog.show()

    def _on_batch_dialog_closed(self, _result: int = 0) -> None:
        self._batch_dialog = None
        self.status_message("Ready")

    def export_image(self) -> None:
        """Render at full resolution and save to disk."""
        if not self.compositor.is_ready:
            self.status_message("Load both a phone photo and a design first")
            return

        if self._export_thread is not None and self._export_thread.isRunning():
            self.status_message("An export is already running")
            return

        cfg = get_config()
        export_dir = self.user_settings.last_dir(
            "export",
            cfg.resolved_export_dir() or (
                self.phone_path.parent if self.phone_path else None
            ),
        )
        suggested = "mockup.png"
        if self.phone_path is not None:
            suggested = f"{self.phone_path.stem}-mockup.png"
        start_path = str(Path(export_dir) / suggested) if export_dir else suggested

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Mockup", start_path,
            "PNG Image (*.png);;JPEG Image (*.jpg);;WebP Image (*.webp)")

        if not path:
            return

        if not Path(path).suffix:
            extension = '.jpg' if 'JPEG' in selected_filter else (
                '.webp' if 'WebP' in selected_filter else '.png')
            path += extension

        out = Path(path)
        if out.exists() and cfg.export_confirm_overwrite:
            answer = QMessageBox.question(
                self, "Overwrite file?",
                f"{out.name} already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                out = ImageLoader.unique_path(out)
                path = str(out)
                self.status_message(f"Exporting as {out.name}")

        self.user_settings.set_last_dir("export", out)
        self.progress_bar.setVisible(True)
        self.export_btn.setEnabled(False)
        self.export_btn_2.setEnabled(False)
        self.status_message("Exporting at full resolution…")

        self._export_thread = ExportThread(
            self.compositor, path, int(cfg.export_quality), self
        )
        self._export_thread.done.connect(self._on_export_done)
        self._export_thread.start()

    def _on_export_done(self, success: bool, path: str, error: str) -> None:
        """Report the export result."""
        self.progress_bar.setVisible(False)
        self._update_enabled_state()

        if success:
            self.status_message(f"Exported to {path}")
            QMessageBox.information(self, "Export complete",
                                    f"Mockup saved to:\n{path}")
        else:
            self._error("Export failed", error or "Unknown error")

    def copy_to_clipboard(self) -> None:
        """Copy the current preview to the clipboard."""
        if self.current_preview is None:
            self.status_message("Nothing to copy yet")
            return

        pixmap = numpy_to_qpixmap(self.current_preview)
        if pixmap is None:
            return

        QGuiApplication.clipboard().setPixmap(pixmap)
        self.status_message("Preview copied to clipboard")

    # -------------------------------------------------------- projects

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_window_title()

    def _capture_edit_snapshot(
        self, label: str = "", coalesce_key: Optional[str] = None
    ) -> EditSnapshot:
        """Freeze current editable compositor state for undo/redo."""
        mesh = self.compositor.control_mesh
        return EditSnapshot(
            label=label,
            coalesce_key=coalesce_key,
            mesh_points=(
                None if mesh is None else mesh.points.copy()
            ),
            mesh_rows=0 if mesh is None else mesh.rows,
            mesh_cols=0 if mesh is None else mesh.cols,
            exclusion_mask=(
                None if self.compositor.exclusion_mask is None
                else self.compositor.exclusion_mask.copy()
            ),
            printable_mask=(
                None if self.compositor.printable_mask is None
                else self.compositor.printable_mask.copy()
            ),
            hardware_contours=[
                np.asarray(c, dtype=np.float32).copy()
                for c in self.compositor.hardware_contours
            ],
            settings=dict(self.compositor.settings),
            material_name=self.compositor.material_name,
            lighting_name=self.compositor.lighting_name,
            fit_mode=self.compositor.fit_mode,
            mirror=bool(self.compositor.mirror),
            corner_radius_estimate=float(
                self.compositor.corner_radius_estimate
            ),
            automatic_margin=float(self.compositor.automatic_margin),
            auto_detected=bool(self.compositor.auto_detected),
            from_template=bool(self.compositor.from_template),
        )

    def _push_history(
        self, label: str, coalesce_key: Optional[str] = None
    ) -> None:
        """Store pre-change state unless we are restoring history."""
        if self._restoring_history or self._syncing:
            return
        if self.compositor.phone_image is None:
            return
        self._edit_history.push(
            self._capture_edit_snapshot(label, coalesce_key)
        )
        self._update_history_buttons()

    def _apply_edit_snapshot(self, snap: EditSnapshot) -> None:
        """Restore geometry and look from a snapshot."""
        from ..image_processing.mesh import ControlMesh

        self._restoring_history = True
        try:
            if (
                snap.mesh_points is not None
                and snap.mesh_rows > 0
                and snap.mesh_cols > 0
            ):
                self.compositor.control_mesh = ControlMesh(
                    snap.mesh_points.copy(),
                    snap.mesh_rows,
                    snap.mesh_cols,
                )
                self.compositor.cover_points = (
                    self.compositor.control_mesh.corner_points()
                )
            else:
                self.compositor.control_mesh = None
                self.compositor.cover_points = None

            self.compositor.exclusion_mask = (
                None if snap.exclusion_mask is None
                else snap.exclusion_mask.copy()
            )
            self.compositor.printable_mask = (
                None if snap.printable_mask is None
                else snap.printable_mask.copy()
            )
            self.compositor.hardware_contours = [
                np.asarray(c, dtype=np.float32).copy()
                for c in snap.hardware_contours
            ]
            # If a bad snapshot lost the mask but kept contours (or vice versa),
            # rebuild so Undo always brings cutouts back on screen.
            has_mask = (
                self.compositor.exclusion_mask is not None
                and np.count_nonzero(self.compositor.exclusion_mask) > 0
            )
            if self.compositor.hardware_contours and not has_mask:
                self.compositor.set_hardware_exclusions(
                    [
                        np.asarray(c, dtype=np.float32).reshape(-1, 2)
                        for c in self.compositor.hardware_contours
                    ],
                    snap_geometry=False,
                    allow_clear=False,
                )
            elif has_mask and not self.compositor.hardware_contours:
                from ..image_processing.region_detector import (
                    HardwareRegionDetector,
                )

                self.compositor.hardware_contours = (
                    HardwareRegionDetector._smooth_exclusion_contours(
                        self.compositor.exclusion_mask
                    )
                )
                self.compositor._sync_printable_from_mesh()
            else:
                self.compositor._sync_printable_from_mesh()

            self.compositor.settings = dict(snap.settings)
            self.compositor.material_name = snap.material_name
            self.compositor.lighting_name = snap.lighting_name
            self.compositor.fit_mode = snap.fit_mode
            self.compositor.mirror = bool(snap.mirror)
            self.compositor.corner_radius_estimate = float(
                snap.corner_radius_estimate
            )
            self.compositor.automatic_margin = float(snap.automatic_margin)
            self.compositor.auto_detected = bool(snap.auto_detected)
            self.compositor.from_template = bool(snap.from_template)
            self.compositor.invalidate()

            self._sync_sliders(self.compositor.get_settings())
            self._syncing = True
            idx = self.fit_mode_combo.findData(snap.fit_mode)
            if idx >= 0:
                self.fit_mode_combo.setCurrentIndex(idx)
            self.mirror_check.setChecked(bool(snap.mirror))
            preset = snap.material_name
            if self.preset_combo.findText(preset) >= 0:
                self.preset_combo.setCurrentText(preset)
            self._syncing = False

            self._sync_cover_to_canvas()
            if self.edit_region_btn.isChecked() and self.compositor.phone_image is not None:
                self.canvas.set_image(self.compositor.phone_image)
            self.request_render()
        finally:
            self._restoring_history = False
            self._update_history_buttons()

    def undo_edit(self) -> None:
        """Restore the previous editable state."""
        if not self._edit_history.can_undo():
            self.status_message("Nothing to undo")
            return
        label = self._edit_history.undo_label()
        current = self._capture_edit_snapshot("current")
        previous = self._edit_history.undo(current)
        if previous is None:
            return
        self._apply_edit_snapshot(previous)
        self._mark_dirty()
        self.status_message(f"Undo: {label or 'edit'}")

    def redo_edit(self) -> None:
        """Re-apply a previously undone state."""
        if not self._edit_history.can_redo():
            self.status_message("Nothing to redo")
            return
        label = self._edit_history.redo_label()
        current = self._capture_edit_snapshot("current")
        nxt = self._edit_history.redo(current)
        if nxt is None:
            return
        self._apply_edit_snapshot(nxt)
        self._mark_dirty()
        self.status_message(f"Redo: {label or 'edit'}")

    def _update_history_buttons(self) -> None:
        """Enable/disable undo/redo icons from stack state."""
        can_undo = self._edit_history.can_undo()
        can_redo = self._edit_history.can_redo()
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(can_undo)
            tip = "Undo (Ctrl+Z)"
            if can_undo and self._edit_history.undo_label():
                tip = f"Undo {self._edit_history.undo_label()} (Ctrl+Z)"
            self.undo_btn.setToolTip(tip)
        if hasattr(self, "redo_btn"):
            self.redo_btn.setEnabled(can_redo)
            tip = "Redo (Ctrl+Y)"
            if can_redo and self._edit_history.redo_label():
                tip = f"Redo {self._edit_history.redo_label()} (Ctrl+Y)"
            self.redo_btn.setToolTip(tip)

    def _update_window_title(self) -> None:
        cfg = get_config()
        name = self.project_path.name if self.project_path else "Untitled"
        dirty = " *" if self._dirty else ""
        self.setWindowTitle(f"{name}{dirty} — {cfg.app_name}")

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self.user_settings.recent_projects()
        if not recent:
            empty = QAction("(No recent projects)", self)
            empty.setEnabled(False)
            self.recent_menu.addAction(empty)
            return
        for path in recent:
            action = QAction(str(path), self)
            action.triggered.connect(
                lambda _=False, p=path: self.open_project(str(p))
            )
            self.recent_menu.addAction(action)
        self.recent_menu.addSeparator()
        clear_action = QAction("Clear Recent", self)
        clear_action.triggered.connect(self._clear_recent_projects)
        self.recent_menu.addAction(clear_action)

    def _clear_recent_projects(self) -> None:
        self.user_settings.clear_recent_projects()
        self._rebuild_recent_menu()

    def new_project(self) -> None:
        """Start a blank session."""
        self.clear_all()
        self.status_message("New project")

    def open_project(self, file_path: Optional[str] = None) -> None:
        """Open a `.pcms` project file."""
        path = file_path
        if not path:
            start = self.user_settings.last_dir("project")
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Project", start,
                f"Mockup Project (*{PROJECT_EXTENSION});;All Files (*)",
            )
        if not path:
            return
        try:
            document, phone_path, design_path = ProjectStore.load(
                Path(path), self.compositor
            )
        except ProjectError as exc:
            self._error("Could not open project", str(exc))
            return
        except Exception as exc:
            logger.exception("Unexpected project open failure")
            self._error("Could not open project", str(exc))
            return

        self.project_path = Path(path)
        self.phone_path = phone_path
        self.design_path = design_path
        self._dirty = False
        self.user_settings.add_recent_project(self.project_path)
        self.user_settings.set_last_dir("project", self.project_path)
        self._rebuild_recent_menu()
        self._sync_sliders(self.compositor.get_settings())
        self._refresh_loaded_labels()
        self._sync_cover_to_canvas()
        self._store_mesh_baseline()
        self._update_enabled_state()
        self._update_window_title()
        meta_name = document.metadata.get("name", self.project_path.stem)
        self.status_message(f"Opened project · {meta_name}")
        if self.compositor.is_ready:
            self.request_render()
        elif self.compositor.phone_image is not None:
            self.canvas.set_image(self.compositor.phone_image)

    def save_project(self) -> None:
        """Save to the current project path, or Save As when unset."""
        if self.project_path is None:
            self.save_project_as()
            return
        self._write_project(self.project_path)

    def save_project_as(self) -> None:
        """Choose a path and save the project."""
        start = self.user_settings.last_dir(
            "project",
            self.phone_path.parent if self.phone_path else None,
        )
        suggested = Path(start or ".") / (
            f"{(self.phone_path.stem if self.phone_path else 'mockup')}{PROJECT_EXTENSION}"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", str(suggested),
            f"Mockup Project (*{PROJECT_EXTENSION})",
        )
        if not path:
            return
        out = Path(path)
        if out.suffix.lower() != PROJECT_EXTENSION:
            out = out.with_suffix(PROJECT_EXTENSION)
        self._write_project(out)

    def _write_project(self, path: Path) -> None:
        try:
            ProjectStore.save(
                path, self.compositor, self.phone_path, self.design_path
            )
        except ProjectError as exc:
            self._error("Could not save project", str(exc))
            return
        self.project_path = path
        self._dirty = False
        self.user_settings.add_recent_project(path)
        self.user_settings.set_last_dir("project", path)
        self._rebuild_recent_menu()
        self._update_window_title()
        self.status_message(f"Project saved · {path.name}")

    def _autosave_session(self) -> None:
        """Periodic recovery snapshot."""
        if self.compositor.phone_image is None and self.compositor.design_image is None:
            return
        path = ProjectStore.autosave(
            self.compositor, self.phone_path, self.design_path
        )
        if path is not None:
            logger.debug("Autosaved session to %s", path)

    def _maybe_reopen_last_project(self) -> None:
        """Optionally restore the last project after the UI is shown."""
        if not self.user_settings.reopen_last_on_startup:
            return
        last = self.user_settings.last_project()
        if last is None:
            # Fall back to autosave recovery if present.
            autosave = ProjectStore.autosave_path()
            if autosave.exists():
                answer = QMessageBox.question(
                    self, "Restore autosave?",
                    "A recovery autosave was found. Restore it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if answer == QMessageBox.Yes:
                    self.open_project(str(autosave))
            return
        try:
            self.open_project(str(last))
        except Exception:
            logger.exception("Could not reopen last project")

    def _refresh_loaded_labels(self) -> None:
        """Sync badge/name labels after a project load."""
        if self.phone_path is not None and self.compositor.phone_image is not None:
            h, w = self.compositor.phone_image.shape[:2]
            self._set_badge(self.phone_badge, self.phone_path.name)
            self.phone_name_label.setText(f"{self.phone_path.name}  ·  {w}×{h}")
        if self.design_path is not None and self.compositor.design_image is not None:
            h, w = self.compositor.design_image.shape[:2]
            self._set_badge(self.design_badge, self.design_path.name)
            self.design_name_label.setText(f"{self.design_path.name}  ·  {w}×{h}")

    def _restore_window_chrome(self) -> None:
        geometry = self.user_settings.window_geometry()
        state = self.user_settings.window_state()
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)

    # ------------------------------------------------------------------ misc

    def _update_enabled_state(self) -> None:
        """Enable the actions that make sense for the current state."""
        has_phone = self.compositor.phone_image is not None
        has_design = self.compositor.design_image is not None
        ready = has_phone and has_design

        self.export_btn.setEnabled(ready)
        self.export_btn_2.setEnabled(ready)
        self.reset_btn.setEnabled(has_phone or has_design)
        self.swap_btn.setEnabled(ready)
        self.detect_btn.setEnabled(has_phone)
        self.center_region_btn.setEnabled(has_phone)
        self.reset_mesh_btn.setEnabled(
            has_phone and self._mesh_baseline is not None
        )
        self.perfect_finish_btn.setEnabled(has_phone)
        if hasattr(self, "finish_scope_combo"):
            self.finish_scope_combo.setEnabled(has_phone)
        if hasattr(self, "cutout_grow_btn"):
            self.cutout_grow_btn.setEnabled(has_phone)
            self.cutout_shrink_btn.setEnabled(has_phone)
        self.region_btn.setEnabled(has_phone)
        self.edit_region_btn.setEnabled(has_phone)
        self.erase_wrap_btn.setEnabled(has_phone)
        for btn_name in (
            "final_btn",
            "final_panel_btn",
        ):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(has_phone)
        final_on = bool(
            getattr(self, "final_btn", None) is not None
            and self.final_btn.isChecked()
        )
        for btn_name in (
            "final_erase_btn",
            "final_fill_btn",
            "final_panel_erase_btn",
            "final_panel_fill_btn",
        ):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(has_phone and final_on)
        has_design = self.compositor.design_image is not None
        for btn_name in (
            "move_design_btn",
            "move_design_toolbar_btn",
            "nudge_up_btn",
            "nudge_down_btn",
            "nudge_left_btn",
            "nudge_right_btn",
            "zoom_design_in_btn",
            "zoom_design_out_btn",
            "center_design_btn",
            "reset_design_pos_btn",
        ):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(has_phone and has_design)
        self.compare_btn.setEnabled(ready)
        self.load_design_btn.setEnabled(True)
        self._update_history_buttons()

        self.canvas.placeholder_title = ("Drop your design here"
                                        if has_phone and not has_design
                                        else "Drop your phone photo here")

    def status_message(self, message: str, timeout: int = 6000) -> None:
        """Show a transient message in the status bar."""
        self.statusBar().showMessage(message, timeout)

    def _error(self, title: str, message: str) -> None:
        """Show an error dialog and mirror it in the status bar."""
        QMessageBox.critical(self, title, message)
        self.status_message(f"{title}: {message}")

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        """Re-apply the stylesheet after an object name change."""
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _set_badge(
        self,
        badge: QLabel,
        text: str,
        *,
        muted: bool = False,
        max_width: int = 150,
    ) -> None:
        """Set badge text with elision so long file names never crush the toolbar."""
        badge.setMaximumWidth(max_width)
        badge.setToolTip(text)
        metrics = QFontMetrics(badge.font())
        # Leave room for badge padding.
        elided = metrics.elidedText(text, Qt.ElideMiddle, max_width - 16)
        badge.setText(elided)
        badge.setObjectName("badgeMuted" if muted else "badgeLabel")
        self._repolish(badge)

    def show_shortcuts(self) -> None:
        """List the keyboard shortcuts."""
        QMessageBox.information(
            self, "Keyboard shortcuts",
            "<table cellpadding='4'>"
            "<tr><td><b>Ctrl+P</b></td><td>Load phone photo</td></tr>"
            "<tr><td><b>Ctrl+D</b></td><td>Load design</td></tr>"
            "<tr><td><b>Ctrl+O</b></td><td>Open project</td></tr>"
            "<tr><td><b>Ctrl+S</b></td><td>Save project</td></tr>"
            "<tr><td><b>Ctrl+E</b></td><td>Export mockup</td></tr>"
            "<tr><td><b>Ctrl+B</b></td><td>Batch process folder</td></tr>"
            "<tr><td><b>Ctrl+C</b></td><td>Copy preview</td></tr>"
            "<tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>"
            "<tr><td><b>Ctrl+Y</b></td><td>Redo</td></tr>"
            "<tr><td><b>Ctrl+Shift+R</b></td><td>Reset mesh to start (with confirm)</td></tr>"
            "<tr><td><b>Ctrl+R</b></td><td>Reset adjustments</td></tr>"
            "<tr><td><b>Ctrl+Shift+F</b></td><td>Settle edges + polish existing cutouts</td></tr>"
            "<tr><td><b>Erase Wrap</b></td><td>Paint wrap off buttons / holes</td></tr>"
            "<tr><td><b>[ ] / Alt+scroll</b></td><td>Erase brush size</td></tr>"
            "<tr><td><b>Ctrl+F</b></td><td>Fit to view</td></tr>"
            "<tr><td><b>Ctrl+1</b></td><td>Actual size</td></tr>"
            "<tr><td><b>Ctrl+ +/-</b></td><td>Zoom in / out</td></tr>"
            "<tr><td><b>R</b></td><td>Show print area</td></tr>"
            "<tr><td><b>E</b></td><td>Edit deformation mesh</td></tr>"
            "<tr><td><b>C</b></td><td>Compare with original</td></tr>"
            "<tr><td><b>M</b></td><td>Mirror design</td></tr>"
            "</table>")

    def show_about(self) -> None:
        """Show the about dialog."""
        QMessageBox.about(
            self, "About Phone Cover Mockup Studio",
            f"<h3>Phone Cover Mockup Studio</h3>"
            f"<p>Version {APP_VERSION} · runs fully offline</p>"
            "<p>Prints any artwork onto a phone cover photo with smart fit, "
            "material rendering and batch production — all on-device.</p>"
            "<p style='color:#8792AC'>Built with PySide6, OpenCV and NumPy.</p>")

    # ------------------------------------------------------- window plumbing

    def dragEnterEvent(self, event) -> None:
        """Accept image drops anywhere in the window."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        """Forward window level drops to the same handler as the canvas."""
        paths = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if not local:
                continue
            path = Path(local)
            if path.suffix.lower() == PROJECT_EXTENSION:
                self.open_project(str(path))
                event.acceptProposedAction()
                return
            if ImageLoader.is_supported(path):
                paths.append(local)

        if paths:
            event.acceptProposedAction()
            self.handle_dropped_files(paths)

    def closeEvent(self, event) -> None:
        """Stop workers, persist chrome, and autosave before exit."""
        if self._dirty and (
            self.compositor.phone_image is not None
            or self.compositor.design_image is not None
        ):
            answer = QMessageBox.question(
                self, "Exit?",
                "You have unsaved project changes. Exit anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return

        self._autosave_session()
        self.user_settings.set_window_geometry(self.saveGeometry())
        self.user_settings.set_window_state(self.saveState())

        self.render_timer.stop()
        self.autosave_timer.stop()
        self.render_thread.stop()

        if self._batch_dialog is not None:
            self._batch_dialog.close()
            self._batch_dialog = None

        if self._export_thread is not None and self._export_thread.isRunning():
            self._export_thread.wait(3000)

        event.accept()
