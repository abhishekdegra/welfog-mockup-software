"""
Batch Production Engine — folder-scale offline mockup generation.

Discovers designs, queues jobs, renders sequentially via the existing
Compositor API, exports results, and writes a production report. Pause,
resume, cancel, and retry are supported without touching geometry or
material internals.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from ..config import get_config
from ..image_processing.compositor import Compositor
from ..utils.image_loader import ImageLoadError, ImageLoader

logger = logging.getLogger("mockup.batch")


BATCH_INPUT_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
_EXPORT_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}


class BatchJobStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class BatchSessionState(str, Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    PAUSED = 'paused'
    CANCELLED = 'cancelled'
    FINISHED = 'finished'


@dataclass
class BatchJob:
    """One design file in the production queue."""

    source_path: Path
    output_path: Path
    status: BatchJobStatus = BatchJobStatus.PENDING
    error: str = ''
    attempts: int = 0
    duration_sec: float = 0.0


@dataclass
class BatchProgress:
    """Live counters for UI / logging."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    remaining: int = 0
    cancelled: int = 0
    elapsed_sec: float = 0.0
    eta_sec: Optional[float] = None
    current_file: str = ''
    state: BatchSessionState = BatchSessionState.IDLE
    last_error: str = ''


@dataclass
class BatchReport:
    """Final production report written beside the exports."""

    output_dir: Path
    total: int
    completed: int
    failed: int
    cancelled: int
    elapsed_sec: float
    jobs: List[BatchJob] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            'output_dir': str(self.output_dir),
            'total': self.total,
            'completed': self.completed,
            'failed': self.failed,
            'cancelled': self.cancelled,
            'elapsed_sec': round(self.elapsed_sec, 3),
            'completed_files': [
                {
                    'source': str(job.source_path),
                    'output': str(job.output_path),
                    'duration_sec': round(job.duration_sec, 3),
                }
                for job in self.jobs
                if job.status == BatchJobStatus.COMPLETED
            ],
            'failed_files': [
                {
                    'source': str(job.source_path),
                    'error': job.error,
                    'attempts': job.attempts,
                }
                for job in self.jobs
                if job.status == BatchJobStatus.FAILED
            ],
        }


ProgressCallback = Callable[[BatchProgress], None]


class BatchProductionEngine:
    """
    Sequential production queue over a folder of cover designs.

    The engine never owns geometry or shading logic — it only swaps designs on
    a production Compositor clone and calls `export()` + `ImageLoader.save_image`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pause = threading.Event()
        self._pause.set()  # set = not paused
        self._cancel = threading.Event()
        self._running = False

        self._compositor: Optional[Compositor] = None
        self._jobs: List[BatchJob] = []
        self._output_dir: Optional[Path] = None
        self._quality: int = 96
        self._export_ext: str = '.png'
        self._state = BatchSessionState.IDLE
        self._started_at: float = 0.0
        self._elapsed_offset: float = 0.0
        self._completed_durations: List[float] = []
        self._on_progress: Optional[ProgressCallback] = None

    # -------------------------------------------------------------- discovery

    @staticmethod
    def discover_designs(folder: Union[str, Path]) -> List[Path]:
        """
        List supported design images in a folder (non-recursive).

        Unsupported files are ignored. Order is case-insensitive by name.
        """
        root = Path(folder)
        if not root.is_dir():
            return []

        found: List[Path] = []
        for path in root.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() in BATCH_INPUT_EXTENSIONS:
                found.append(path)

        found.sort(key=lambda p: p.name.lower())
        return found

    # --------------------------------------------------------------- prepare

    def prepare(
        self,
        session: Compositor,
        design_folder: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        *,
        export_format: str = 'png',
        quality: int = 96,
        designs: Optional[Sequence[Union[str, Path]]] = None,
    ) -> List[BatchJob]:
        """
        Build the queue from a loaded phone session and a design folder.

        Args:
            session: Interactive compositor with phone geometry already ready
            design_folder: Folder containing cover designs
            output_dir: Destination; created automatically when omitted
            export_format: png | jpg | jpeg | webp
            quality: JPEG/WebP quality
            designs: Optional explicit file list (skips discovery)

        Returns:
            The prepared job list

        Raises:
            ValueError: When the phone session is incomplete or no designs found
        """
        with self._lock:
            if self._running:
                raise RuntimeError('A batch is already running')

            if session.phone_image is None or session.control_mesh is None:
                raise ValueError(
                    'Load a phone image (and detect the cover) before batching'
                )

            folder = Path(design_folder)
            if designs is None:
                sources = self.discover_designs(folder)
            else:
                sources = [
                    Path(p) for p in designs
                    if Path(p).suffix.lower() in BATCH_INPUT_EXTENSIONS
                ]

            if not sources:
                raise ValueError(
                    f'No supported designs found in: {folder}'
                )

            ext = self._normalise_export_ext(export_format)
            out = Path(output_dir) if output_dir else (
                folder / f'mockups_{time.strftime("%Y%m%d_%H%M%S")}'
            )
            out.mkdir(parents=True, exist_ok=True)

            policy = get_config().batch_overwrite_policy
            jobs = self._build_jobs(sources, out, ext, policy)
            if not jobs:
                raise ValueError(
                    'No jobs to process (all files skipped or unsupported)'
                )
            self._compositor = session.create_production_clone()
            self._jobs = jobs
            self._output_dir = out
            self._quality = int(max(1, min(100, quality if quality is not None else get_config().export_quality)))
            self._export_ext = ext
            self._state = BatchSessionState.IDLE
            self._cancel.clear()
            self._pause.set()
            self._elapsed_offset = 0.0
            self._completed_durations = []
            logger.info(
                "Prepared batch: %d jobs → %s (format=%s policy=%s)",
                len(jobs), out, ext, policy,
            )
            return list(jobs)

    # --------------------------------------------------------------- control

    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> None:
        """Register a listener invoked after each job and on state changes."""
        self._on_progress = callback

    def start(self) -> BatchReport:
        """
        Process the queue sequentially until finished, cancelled, or empty.

        Safe to call from a worker thread. Returns the final report and always
        writes summary files into the output folder.
        """
        with self._lock:
            if self._running:
                raise RuntimeError('Batch already running')
            if not self._jobs or self._compositor is None or self._output_dir is None:
                raise RuntimeError('Call prepare() before start()')
            self._running = True
            self._state = BatchSessionState.RUNNING
            self._started_at = time.monotonic()
            self._cancel.clear()
            self._pause.set()

        self._emit_progress()

        try:
            while True:
                if self._cancel.is_set():
                    self._mark_remaining_cancelled()
                    break

                # Pause gate — UI stays responsive; worker blocks here.
                while not self._pause.is_set():
                    if self._cancel.is_set():
                        break
                    time.sleep(0.05)

                if self._cancel.is_set():
                    self._mark_remaining_cancelled()
                    break

                job = self._next_pending()
                if job is None:
                    break

                self._run_job(job)

            with self._lock:
                if self._state != BatchSessionState.CANCELLED:
                    self._state = BatchSessionState.FINISHED
        finally:
            with self._lock:
                self._running = False
                if self._state == BatchSessionState.RUNNING:
                    self._state = BatchSessionState.FINISHED
                # Drop design bitmap; keep phone geometry for a possible retry.
                if self._compositor is not None:
                    self._compositor.design_image = None
                    self._compositor.invalidate(clear_scaled=True)

            report = self.build_report()
            self._write_reports(report)
            self._emit_progress()
            return report

    def pause(self) -> None:
        """Pause between jobs (current job finishes first)."""
        with self._lock:
            if not self._running or self._state != BatchSessionState.RUNNING:
                return
            self._state = BatchSessionState.PAUSED
            # Freeze elapsed so ETA stays honest across pause.
            self._elapsed_offset += time.monotonic() - self._started_at
        self._pause.clear()
        self._emit_progress()

    def resume(self) -> None:
        """Continue after pause."""
        with self._lock:
            if self._state != BatchSessionState.PAUSED:
                return
            self._state = BatchSessionState.RUNNING
            self._started_at = time.monotonic()
        self._pause.set()
        self._emit_progress()

    def cancel(self) -> None:
        """Stop after the current job; remaining pending jobs become cancelled."""
        with self._lock:
            if not self._running and self._state not in (
                BatchSessionState.RUNNING, BatchSessionState.PAUSED
            ):
                return
            self._state = BatchSessionState.CANCELLED
        self._cancel.set()
        self._pause.set()  # unblock pause wait
        self._emit_progress()

    def retry_failed(self) -> BatchReport:
        """
        Re-queue failed jobs and process them.

        Completed / cancelled jobs are left untouched. Requires a prior
        `prepare()` (or an unfinished session with a live compositor clone).
        """
        with self._lock:
            if self._running:
                raise RuntimeError('Cannot retry while a batch is running')
            if self._compositor is None or self._output_dir is None:
                raise RuntimeError('No batch session to retry')

            retried = 0
            for job in self._jobs:
                if job.status == BatchJobStatus.FAILED:
                    job.status = BatchJobStatus.PENDING
                    job.error = ''
                    retried += 1

            if retried == 0:
                return self.build_report()

            self._state = BatchSessionState.IDLE
            self._cancel.clear()
            self._pause.set()

        return self.start()

    # --------------------------------------------------------------- queries

    def get_progress(self) -> BatchProgress:
        """Snapshot of current counters."""
        with self._lock:
            completed = sum(
                1 for j in self._jobs if j.status == BatchJobStatus.COMPLETED
            )
            failed = sum(
                1 for j in self._jobs if j.status == BatchJobStatus.FAILED
            )
            cancelled = sum(
                1 for j in self._jobs if j.status == BatchJobStatus.CANCELLED
            )
            remaining = sum(
                1 for j in self._jobs
                if j.status in (BatchJobStatus.PENDING, BatchJobStatus.RUNNING)
            )
            current = next(
                (j.source_path.name for j in self._jobs
                 if j.status == BatchJobStatus.RUNNING),
                '',
            )
            last_error = next(
                (j.error for j in reversed(self._jobs)
                 if j.status == BatchJobStatus.FAILED and j.error),
                '',
            )
            elapsed = self._elapsed_offset
            if self._state == BatchSessionState.RUNNING:
                elapsed += time.monotonic() - self._started_at

            eta: Optional[float] = None
            if self._completed_durations and remaining > 0:
                avg = sum(self._completed_durations) / len(self._completed_durations)
                eta = avg * remaining

            return BatchProgress(
                total=len(self._jobs),
                completed=completed,
                failed=failed,
                remaining=remaining,
                cancelled=cancelled,
                elapsed_sec=elapsed,
                eta_sec=eta,
                current_file=current,
                state=self._state,
                last_error=last_error,
            )

    def get_jobs(self) -> List[BatchJob]:
        """Copy of the current queue."""
        with self._lock:
            return list(self._jobs)

    @property
    def output_dir(self) -> Optional[Path]:
        return self._output_dir

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def build_report(self) -> BatchReport:
        """Assemble a report from the current queue state."""
        progress = self.get_progress()
        with self._lock:
            return BatchReport(
                output_dir=self._output_dir or Path('.'),
                total=progress.total,
                completed=progress.completed,
                failed=progress.failed,
                cancelled=progress.cancelled,
                elapsed_sec=progress.elapsed_sec,
                jobs=list(self._jobs),
            )

    # ------------------------------------------------------------- internals

    def _run_job(self, job: BatchJob) -> None:
        assert self._compositor is not None
        compositor = self._compositor

        with self._lock:
            job.status = BatchJobStatus.RUNNING
            job.attempts += 1
            job.error = ''
        self._emit_progress()

        started = time.monotonic()
        try:
            image = ImageLoader.load_image(job.source_path)
            compositor.set_design_image(image)
            # Free loader reference early.
            del image

            rendered = compositor.export(
                include_alpha=job.output_path.suffix.lower() == '.png'
            )
            if rendered is None:
                raise RuntimeError('Compositor returned no image')

            ok, error = ImageLoader.save_image_ex(
                rendered, job.output_path, self._quality
            )
            del rendered
            if not ok:
                raise RuntimeError(error or 'Failed to write output file')

            with self._lock:
                job.status = BatchJobStatus.COMPLETED
                job.duration_sec = time.monotonic() - started
                self._completed_durations.append(job.duration_sec)
            logger.info("Batch job OK: %s", job.output_path.name)
        except (ImageLoadError, OSError, RuntimeError, ValueError) as exc:
            with self._lock:
                job.status = BatchJobStatus.FAILED
                job.error = str(exc)
                job.duration_sec = time.monotonic() - started
            logger.warning("Batch job failed (%s): %s", job.source_path.name, exc)
        except Exception as exc:  # noqa: BLE001 — isolate one bad file
            with self._lock:
                job.status = BatchJobStatus.FAILED
                job.error = f'Unexpected error: {exc}'
                job.duration_sec = time.monotonic() - started
            logger.exception("Unexpected batch failure for %s", job.source_path.name)
        finally:
            # Keep phone/mesh; drop only the design to limit memory growth.
            compositor.design_image = None
            compositor.invalidate(clear_scaled=False)
            self._emit_progress()

    def _next_pending(self) -> Optional[BatchJob]:
        with self._lock:
            for job in self._jobs:
                if job.status == BatchJobStatus.PENDING:
                    return job
            return None

    def _mark_remaining_cancelled(self) -> None:
        with self._lock:
            self._state = BatchSessionState.CANCELLED
            for job in self._jobs:
                if job.status == BatchJobStatus.PENDING:
                    job.status = BatchJobStatus.CANCELLED

    def _emit_progress(self) -> None:
        callback = self._on_progress
        if callback is None:
            return
        try:
            callback(self.get_progress())
        except Exception:
            pass

    def _write_reports(self, report: BatchReport) -> None:
        if self._output_dir is None:
            return
        try:
            summary_path = self._output_dir / 'batch_summary.json'
            summary_path.write_text(
                json.dumps(report.to_dict(), indent=2),
                encoding='utf-8',
            )

            failed = [
                job for job in report.jobs
                if job.status == BatchJobStatus.FAILED
            ]
            failed_path = self._output_dir / 'failed_list.txt'
            if failed:
                lines = [
                    f'{job.source_path.name}\t{job.error}'
                    for job in failed
                ]
                failed_path.write_text(
                    '\n'.join(lines) + '\n', encoding='utf-8'
                )
            elif failed_path.exists():
                failed_path.unlink()

            # Human-readable one-pager.
            text_path = self._output_dir / 'batch_summary.txt'
            text_path.write_text(
                '\n'.join([
                    'Phone Cover Mockup Studio — Batch Report',
                    f'Output: {report.output_dir}',
                    f'Total: {report.total}',
                    f'Completed: {report.completed}',
                    f'Failed: {report.failed}',
                    f'Cancelled: {report.cancelled}',
                    f'Elapsed (sec): {report.elapsed_sec:.2f}',
                    '',
                ]),
                encoding='utf-8',
            )
            logger.info(
                "Batch report written (%d completed, %d failed)",
                report.completed, report.failed,
            )
        except OSError as exc:
            logger.error("Could not write batch reports: %s", exc)

    @staticmethod
    def _normalise_export_ext(export_format: str) -> str:
        raw = export_format.strip().lower()
        if not raw.startswith('.'):
            raw = f'.{raw}'
        if raw == '.jpeg':
            raw = '.jpg'
        if raw not in _EXPORT_EXTENSIONS:
            raise ValueError(
                f'Unsupported export format {export_format!r}; '
                f'use png, jpg, jpeg, or webp'
            )
        return raw

    @staticmethod
    def _build_jobs(
        sources: Sequence[Path],
        output_dir: Path,
        ext: str,
        overwrite_policy: str = 'rename',
    ) -> List[BatchJob]:
        used: Dict[str, int] = {}
        jobs: List[BatchJob] = []
        policy = (overwrite_policy or 'rename').lower()
        for source in sources:
            stem = source.stem
            count = used.get(stem.lower(), 0)
            used[stem.lower()] = count + 1
            name = f'{stem}{ext}' if count == 0 else f'{stem}_{count}{ext}'
            output = output_dir / name

            if output.exists():
                if policy == 'skip':
                    logger.info("Skipping existing output %s", output.name)
                    continue
                if policy != 'overwrite':
                    output = ImageLoader.unique_path(output)

            jobs.append(
                BatchJob(
                    source_path=source,
                    output_path=output,
                )
            )
        return jobs
