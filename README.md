# Phone Cover Mockup Studio

Offline desktop tool that prints artwork onto phone-cover photos with smart
auto-fit, material rendering, and batch production.

**Version:** 2.1.0 · Fully offline (no AI / cloud)

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Daily workflow

1. **Load Phone** — detects printable cover surface (templates reuse layouts)
2. **Load Design** — smart auto-fit onto the printable area
3. Tweak mesh / material / lighting presets if needed
4. **Export** (`Ctrl+E`) or **Batch Process Folder** (`Ctrl+B`)

## Projects

- Save / Open `.pcms` projects (`Ctrl+S` / `Ctrl+O`)
- Recent projects menu + optional reopen-last on startup
- Autosave recovery under `data/autosave/`

## Configuration

Editable offline settings live in `data/config.json` (created on first run):

- export quality, preview size, debounce
- default material / lighting
- cache sizes, overwrite policy
- log level, theme key

Logs: `data/logs/app.log`

## Windows portable build

```bash
scripts\build_windows.bat
```

Output folder: `dist/PhoneCoverMockupStudio/` — copy that folder to run without installing Python.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Stack

PySide6 · OpenCV · NumPy · Pillow
