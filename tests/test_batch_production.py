"""Regression tests for the Batch Production Engine."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.image_processing.compositor import Compositor
from src.image_processing.template_cache import TemplateCache
from src.production.batch_engine import (
    BATCH_INPUT_EXTENSIONS, BatchJobStatus, BatchProductionEngine,
    BatchSessionState,
)
from test_mesh_geometry import synthetic_phone


def _write_design(path: Path, color=(40, 90, 210)) -> None:
    img = np.full((120, 60, 3), color, np.uint8)
    ok, buf = cv2.imencode(path.suffix, img)
    assert ok
    path.write_bytes(buf.tobytes())


class BatchDiscoveryTests(unittest.TestCase):
    def test_discovers_supported_and_ignores_others(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_design(root / 'a.png')
            _write_design(root / 'b.jpg')
            _write_design(root / 'c.jpeg')
            _write_design(root / 'd.webp')
            (root / 'notes.txt').write_text('ignore', encoding='utf-8')
            (root / 'e.bmp').write_bytes(b'nope')

            found = BatchProductionEngine.discover_designs(root)
            names = {p.name for p in found}
            self.assertEqual(names, {'a.png', 'b.jpg', 'c.jpeg', 'd.webp'})
            self.assertTrue(BATCH_INPUT_EXTENSIONS >= {'.png', '.jpg', '.jpeg', '.webp'})


class BatchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = TemplateCache(self.root / 'templates')
        self.compositor = Compositor(template_cache=self.cache)
        phone = synthetic_phone()
        self.assertTrue(self.compositor.set_phone_image(phone))

        self.designs = self.root / 'designs'
        self.designs.mkdir()
        for name, color in (
            ('one.png', (20, 40, 200)),
            ('two.jpg', (30, 160, 80)),
            ('three.webp', (200, 40, 40)),
        ):
            _write_design(self.designs / name, color)

        self.output = self.root / 'out'

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_batch_exports_preserve_stems_and_writes_report(self) -> None:
        engine = BatchProductionEngine()
        jobs = engine.prepare(
            self.compositor, self.designs, self.output,
            export_format='png', quality=90,
        )
        self.assertEqual(len(jobs), 3)

        report = engine.start()
        self.assertEqual(report.completed, 3)
        self.assertEqual(report.failed, 0)
        self.assertTrue((self.output / 'one.png').exists())
        self.assertTrue((self.output / 'two.png').exists())
        self.assertTrue((self.output / 'three.png').exists())
        self.assertTrue((self.output / 'batch_summary.json').exists())
        self.assertTrue((self.output / 'batch_summary.txt').exists())
        self.assertFalse((self.output / 'failed_list.txt').exists())

        data = json.loads((self.output / 'batch_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(data['completed'], 3)

    def test_failed_job_does_not_stop_batch(self) -> None:
        bad = self.designs / 'broken.png'
        bad.write_bytes(b'not-an-image')

        engine = BatchProductionEngine()
        engine.prepare(self.compositor, self.designs, self.output, export_format='jpg')
        report = engine.start()

        self.assertGreaterEqual(report.completed, 3)
        self.assertGreaterEqual(report.failed, 1)
        self.assertTrue((self.output / 'failed_list.txt').exists())
        failed_text = (self.output / 'failed_list.txt').read_text(encoding='utf-8')
        self.assertIn('broken.png', failed_text)

    def test_pause_resume_and_cancel(self) -> None:
        # Many tiny designs so we can pause mid-queue.
        for i in range(8):
            _write_design(self.designs / f'extra_{i:02d}.png', (i * 20, 50, 100))

        engine = BatchProductionEngine()
        engine.prepare(self.compositor, self.designs, self.output)

        def runner() -> None:
            engine.start()

        thread = threading.Thread(target=runner)
        thread.start()

        # Wait until something completes, then pause.
        deadline = time.time() + 60
        while time.time() < deadline:
            p = engine.get_progress()
            if p.completed >= 1 or p.state == BatchSessionState.FINISHED:
                break
            time.sleep(0.05)

        engine.pause()
        time.sleep(0.15)
        state = engine.get_progress().state
        # May already have finished on a fast machine.
        if state == BatchSessionState.PAUSED:
            completed_while_paused = engine.get_progress().completed
            time.sleep(0.25)
            self.assertLessEqual(
                engine.get_progress().completed, completed_while_paused + 1
            )
            engine.resume()
            time.sleep(0.1)

        engine.cancel()
        thread.join(timeout=90)
        self.assertFalse(thread.is_alive())

        report = engine.build_report()
        self.assertEqual(
            report.cancelled + report.completed + report.failed, report.total
        )

    def test_retry_failed_reprocesses(self) -> None:
        bad = self.designs / 'later.png'
        bad.write_bytes(b'corrupt')

        engine = BatchProductionEngine()
        engine.prepare(self.compositor, self.designs, self.output)
        first = engine.start()
        self.assertGreaterEqual(first.failed, 1)

        # Fix the file, then retry.
        _write_design(bad, (90, 90, 90))
        second = engine.retry_failed()
        self.assertEqual(second.failed, 0)
        self.assertTrue((self.output / 'later.png').exists())

    def test_production_clone_reuses_phone_geometry(self) -> None:
        clone = self.compositor.create_production_clone()
        self.assertIsNotNone(clone.phone_image)
        self.assertIsNotNone(clone.control_mesh)
        self.assertIsNotNone(clone.exclusion_mask)
        self.assertIsNone(clone.design_image)
        # Independent object; mutating clone design must not touch UI session.
        design = np.full((80, 40, 3), 100, np.uint8)
        clone.set_design_image(design)
        self.assertIsNone(self.compositor.design_image)

    def test_prepare_requires_phone(self) -> None:
        empty = Compositor(template_cache=TemplateCache(self.root / 'empty-tpl'))
        engine = BatchProductionEngine()
        with self.assertRaises(ValueError):
            engine.prepare(empty, self.designs, self.output)


class BatchJobStatusTests(unittest.TestCase):
    def test_status_values(self) -> None:
        self.assertEqual(BatchJobStatus.PENDING.value, 'pending')
        self.assertEqual(BatchSessionState.FINISHED.value, 'finished')


if __name__ == '__main__':
    unittest.main()
