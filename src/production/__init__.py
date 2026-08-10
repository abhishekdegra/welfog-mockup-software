"""
Batch production pipeline — high-volume offline mockup generation.

Independent of geometry detection and material shading internals; drives the
existing Compositor public API only.
"""

from .batch_engine import (
    BATCH_INPUT_EXTENSIONS,
    BatchJob,
    BatchJobStatus,
    BatchProductionEngine,
    BatchProgress,
    BatchReport,
    BatchSessionState,
)

__all__ = [
    'BATCH_INPUT_EXTENSIONS',
    'BatchJob',
    'BatchJobStatus',
    'BatchProductionEngine',
    'BatchProgress',
    'BatchReport',
    'BatchSessionState',
]
