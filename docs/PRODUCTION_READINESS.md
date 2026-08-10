# Production Readiness Report — v2.1.0

Date: 2026-08-01  
Phase: Chapter 6 — Production Optimization & Release Readiness

## Validation

| Check | Result |
|---|---|
| `compileall src` | Pass |
| Unit tests | Pass (all discovered tests) |
| Existing interactive workflow | Unchanged |
| Geometry / Smart Auto-Fit / Materials / Batch | Unchanged algorithms |
| Export API | Compatible (`Compositor.export` + `ImageLoader.save_image`) |

## Implemented

### Performance
- LRU caps on preview / scaled-phone caches (`result_cache_size`, `scaled_phone_cache_size`)
- Export uses a production compositor clone (no fight with preview thread)
- Batch reuses phone geometry + template cache; clears design bitmaps per job
- Configurable preview size and render debounce

### Reliability
- Stronger image validation (empty / corrupt / unsupported)
- Informative `save_image_ex` errors; batch/export log failures without crashing
- Cover analysis failures caught; UI survives bad phone files
- Batch overwrite policy: `rename` (default) / `overwrite` / `skip`
- Hardware cutout hard-restore retained from Chapter 4

### Project management
- `.pcms` projects (paths, settings, mesh, material/lighting, metadata, format version)
- Autosave → `data/autosave/last_session.pcms`
- Recent projects + reopen last (QSettings)
- Dirty-state confirmation on clear/exit

### UX
- Progress/status for load, export, batch, render failures
- Shortcuts updated (project + batch)
- Remembered last folders; window geometry restore
- Drag-drop `.pcms` projects

### Logging & config
- Rotating file log `data/logs/app.log` (info / warning / error)
- Central `data/config.json` via `src/config.py`

### Packaging
- Icons in `src/resources/`
- Updated `build.spec` (onedir portable)
- `scripts/build_windows.bat`
- README + requirements include PyInstaller

## Known limitations

1. Cover detection still runs on the UI thread (may freeze briefly on huge phones).
2. Full-resolution export can be memory-heavy on very large phone photos; OOM is caught and reported.
3. Projects store absolute image paths — moving source files breaks reopen until paths are updated.
4. PyInstaller build is prepared but not CI-verified on every machine (OpenCV DLL variance).
5. Theme is a config key only; single dark stylesheet ships today.
6. No automated leak detector; memory strategy is clone + per-job design release + LRU caches.

## Production readiness verdict

**Ready for daily offline production use** for the current single-user Windows workflow, with project save/autosave, batch folder processing, and portable packaging support. Address limitation (1) if multi-megapixel phones commonly stall the UI.
