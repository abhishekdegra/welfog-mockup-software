"""
Batch production dialog — progress, pause/resume/cancel/retry.

Does not alter the main window layout; opened from the File menu.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QVBoxLayout,
)

from ..image_processing.compositor import Compositor
from ..config import get_config
from ..production.batch_engine import (
    BatchProductionEngine, BatchProgress, BatchReport, BatchSessionState,
)


def _format_seconds(value: Optional[float]) -> str:
    if value is None or value < 0:
        return '—'
    total = int(round(value))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours:d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:d}:{seconds:02d}'


class BatchWorkerThread(QThread):
    """Runs BatchProductionEngine.start / retry_failed off the UI thread."""

    progress = Signal(object)
    finished_report = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        engine: BatchProductionEngine,
        mode: str = 'start',
        parent=None,
    ):
        super().__init__(parent)
        self._engine = engine
        self._mode = mode  # start | retry

    def run(self) -> None:
        try:
            self._engine.set_progress_callback(self._emit_progress)
            if self._mode == 'retry':
                report = self._engine.retry_failed()
            else:
                report = self._engine.start()
            self.finished_report.emit(report)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _emit_progress(self, progress: BatchProgress) -> None:
        self.progress.emit(progress)


class BatchDialog(QDialog):
    """
    Lightweight production panel.

    Workflow: choose design folder + output → Start → monitor progress.
    Requires a phone session already loaded in the main compositor.
    """

    def __init__(self, compositor: Compositor, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Batch Production')
        self.setMinimumWidth(520)
        self.setModal(True)

        self._compositor = compositor
        self._engine = BatchProductionEngine()
        self._worker: Optional[BatchWorkerThread] = None
        self._last_report: Optional[BatchReport] = None

        self._build_ui()
        self._set_running_ui(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(8)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText('Folder containing cover designs…')
        browse_folder = QPushButton('Browse…')
        browse_folder.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(browse_folder)
        form.addRow('Designs', folder_row)

        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText('Leave blank to auto-create beside designs')
        browse_output = QPushButton('Browse…')
        browse_output.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_edit)
        output_row.addWidget(browse_output)
        form.addRow('Output', output_row)

        self.format_combo = QComboBox()
        self.format_combo.addItems(['PNG', 'JPEG', 'WebP'])
        form.addRow('Export format', self.format_combo)

        layout.addLayout(form)

        self.summary_label = QLabel('Load a phone, then choose a design folder.')
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName('infoLabel')
        layout.addWidget(self.summary_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.stats_label = QLabel(self._idle_stats_text())
        self.stats_label.setObjectName('infoLabel')
        layout.addWidget(self.stats_label)

        controls = QHBoxLayout()
        self.start_btn = QPushButton('Start Batch')
        self.start_btn.setObjectName('primaryButton')
        self.start_btn.clicked.connect(self._start_batch)

        self.pause_btn = QPushButton('Pause')
        self.pause_btn.clicked.connect(self._toggle_pause)

        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setObjectName('dangerButton')
        self.cancel_btn.clicked.connect(self._cancel_batch)

        self.retry_btn = QPushButton('Retry Failed')
        self.retry_btn.clicked.connect(self._retry_failed)

        controls.addWidget(self.start_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.cancel_btn)
        controls.addWidget(self.retry_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)
        self._close_box = buttons

    @staticmethod
    def _idle_stats_text() -> str:
        return (
            'Total: 0    Completed: 0    Remaining: 0    Failed: 0\n'
            'Elapsed: —    ETA: —'
        )

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Select Design Folder')
        if path:
            self.folder_edit.setText(path)
            count = len(BatchProductionEngine.discover_designs(path))
            self.summary_label.setText(
                f'Found {count} supported design(s) in the selected folder.'
            )

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if path:
            self.output_edit.setText(path)

    def _set_running_ui(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.folder_edit.setEnabled(not running)
        self.output_edit.setEnabled(not running)
        self.format_combo.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.cancel_btn.setEnabled(running)
        # Retry only when idle after failures.
        failed = 0
        if self._last_report is not None:
            failed = self._last_report.failed
        self.retry_btn.setEnabled(not running and failed > 0)
        if running:
            self.pause_btn.setText('Pause')

    def _start_batch(self) -> None:
        if self._compositor.phone_image is None or self._compositor.control_mesh is None:
            QMessageBox.warning(
                self, 'Phone required',
                'Load a phone image first so the cover template is ready.',
            )
            return

        folder = self.folder_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, 'Design folder', 'Choose a design folder.')
            return

        output = self.output_edit.text().strip() or None
        fmt = self.format_combo.currentText().lower()
        if fmt == 'jpeg':
            fmt = 'jpg'

        try:
            jobs = self._engine.prepare(
                self._compositor,
                folder,
                output,
                export_format=fmt,
                quality=int(get_config().export_quality),
            )
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, 'Cannot start batch', str(exc))
            return

        self.summary_label.setText(
            f'Queued {len(jobs)} design(s) → {self._engine.output_dir}'
        )
        self.progress_bar.setValue(0)
        self._last_report = None
        self._set_running_ui(True)
        self._launch_worker('start')

    def _launch_worker(self, mode: str) -> None:
        self._worker = BatchWorkerThread(self._engine, mode=mode, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_report.connect(self._on_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _toggle_pause(self) -> None:
        if self._engine.get_progress().state == BatchSessionState.PAUSED:
            self._engine.resume()
            self.pause_btn.setText('Pause')
        else:
            self._engine.pause()
            self.pause_btn.setText('Resume')

    def _cancel_batch(self) -> None:
        self._engine.cancel()

    def _retry_failed(self) -> None:
        if self._engine.is_running:
            return
        self._set_running_ui(True)
        self.summary_label.setText('Retrying failed jobs…')
        self._launch_worker('retry')

    def _on_progress(self, progress: BatchProgress) -> None:
        total = max(progress.total, 1)
        done = progress.completed + progress.failed + progress.cancelled
        self.progress_bar.setMaximum(progress.total or 1)
        self.progress_bar.setValue(min(done, progress.total))

        current = progress.current_file or '—'
        state = progress.state.value
        self.stats_label.setText(
            f'Total: {progress.total}    Completed: {progress.completed}    '
            f'Remaining: {progress.remaining}    Failed: {progress.failed}\n'
            f'Elapsed: {_format_seconds(progress.elapsed_sec)}    '
            f'ETA: {_format_seconds(progress.eta_sec)}    '
            f'State: {state}    Current: {current}'
        )
        if progress.state == BatchSessionState.PAUSED:
            self.pause_btn.setText('Resume')
        elif progress.state == BatchSessionState.RUNNING:
            self.pause_btn.setText('Pause')

    def _on_finished(self, report: BatchReport) -> None:
        self._last_report = report
        self._set_running_ui(False)
        self.progress_bar.setValue(report.total)
        out = report.output_dir
        self.summary_label.setText(
            f'Done. Completed {report.completed}/{report.total}. '
            f'Failed {report.failed}. Output: {out}'
        )
        self.stats_label.setText(
            f'Total: {report.total}    Completed: {report.completed}    '
            f'Remaining: 0    Failed: {report.failed}\n'
            f'Elapsed: {_format_seconds(report.elapsed_sec)}    ETA: —'
        )
        if report.failed:
            self.retry_btn.setEnabled(True)
            QMessageBox.warning(
                self, 'Batch finished with errors',
                f'{report.failed} job(s) failed. See failed_list.txt in:\n{out}',
            )
        else:
            QMessageBox.information(
                self, 'Batch complete',
                f'Rendered {report.completed} mockup(s) to:\n{out}',
            )

    def _on_worker_failed(self, message: str) -> None:
        self._set_running_ui(False)
        QMessageBox.critical(self, 'Batch error', message)

    def reject(self) -> None:
        if self._engine.is_running:
            answer = QMessageBox.question(
                self, 'Cancel batch?',
                'A batch is still running. Cancel it and close?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            self._engine.cancel()
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(10000)
        super().reject()

    def closeEvent(self, event) -> None:
        if self._engine.is_running:
            self._engine.cancel()
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(10000)
        super().closeEvent(event)
