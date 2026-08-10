"""
Image processing package for Phone Cover Mockup Studio.
"""

from .compositor import Compositor, DEFAULT_SETTINGS, PRESETS
from .cover_surface import CoverSurfaceEngine, CoverSurfaceResult
from .filters import ImageFilters
from .materials import (
    LIGHTING, MATERIALS, LightingProfile, MaterialProfile,
    MaterialRenderingEngine, CoverNormalField,
)
from .mesh import AdaptiveMeshBuilder, ControlMesh, MeshWarper
from .region_detector import PrintableRegion, PrintableRegionDetector
from .smart_fit import SmartFitEstimator, SmartFitResult
from .template_cache import CoverTemplate, TemplateCache, TemplateManager
from .device_template import (
    CornerRadii,
    DeviceTemplate,
    DeviceTemplateCatalog,
)
from .curved_uv import CurvedUVParams
from .transform import PerspectiveTransform

__all__ = [
    'Compositor',
    'DEFAULT_SETTINGS',
    'PRESETS',
    'CoverSurfaceEngine',
    'CoverSurfaceResult',
    'ImageFilters',
    'LIGHTING',
    'MATERIALS',
    'LightingProfile',
    'MaterialProfile',
    'MaterialRenderingEngine',
    'CoverNormalField',
    'AdaptiveMeshBuilder',
    'ControlMesh',
    'MeshWarper',
    'PrintableRegion',
    'PrintableRegionDetector',
    'SmartFitEstimator',
    'SmartFitResult',
    'CoverTemplate',
    'TemplateCache',
    'TemplateManager',
    'CornerRadii',
    'DeviceTemplate',
    'DeviceTemplateCatalog',
    'CurvedUVParams',
    'PerspectiveTransform',
]
