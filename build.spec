# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for a portable Windows build.

Build from the project root (with the venv active):

    pip install pyinstaller
    pyinstaller build.spec

Output: dist/PhoneCoverMockupStudio/
"""

import os
import sys
from pathlib import Path

from PyInstaller.building.api import COLLECT, EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
ROOT = Path(SPECPATH).resolve()

hiddenimports = [
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'cv2',
    'numpy',
    'PIL',
    'PIL.Image',
    'src',
    'src.config',
    'src.main',
    'src.ui.main_window',
    'src.production.batch_engine',
    'src.persistence.project_store',
]

datas = []
resources = ROOT / 'src' / 'resources'
if resources.exists():
    datas.append((str(resources), 'src/resources'))

# Seed empty data folders so portable installs can write templates/logs.
for relative in ('data/templates', 'data/logs', 'data/autosave'):
    folder = ROOT / relative
    folder.mkdir(parents=True, exist_ok=True)
    keep = folder / '.keep'
    if not keep.exists():
        keep.write_text('', encoding='utf-8')
    datas.append((str(keep), relative.replace('\\', '/')))

try:
    datas += collect_data_files('cv2', includes=['**/*.xml', '**/*.json'])
except Exception:
    pass

binaries = []
app_name = 'PhoneCoverMockupStudio'
app_version = '2.1.0'
icon_path = ROOT / 'src' / 'resources' / 'app.ico'
icon = str(icon_path) if icon_path.exists() else None

a = Analysis(
    [str(ROOT / 'src' / 'main.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
    version=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)
