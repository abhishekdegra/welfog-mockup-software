"""
Launcher for Phone Cover Mockup Studio.

Run this from the project root:

    python main.py
"""

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(SRC / "main.py"), run_name="__main__")
