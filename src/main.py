"""
Main entry point for the Phone Cover Mockup Studio.
"""

import os
import sys
from pathlib import Path


# Qt wants this set before the QApplication is constructed.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies() -> list:
    """Return labels of missing packages (empty when healthy)."""
    missing = []
    for package, label in (
        ("cv2", "OpenCV (opencv-python)"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("PySide6", "PySide6"),
    ):
        try:
            __import__(package)
        except ImportError:
            missing.append(label)
    return missing


def create_application():
    """Construct and configure the QApplication."""
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication

    from src.config import APP_NAME, APP_VERSION, ORG_NAME, load_config

    load_config()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")

    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    for candidate in (
        Path(__file__).parent / "resources" / "icon.ico",
        Path(__file__).parent / "resources" / "app.ico",
    ):
        if candidate.exists():
            app.setWindowIcon(QIcon(str(candidate)))
            break

    return app


def main() -> int:
    """Application entry point."""
    missing = check_dependencies()
    if missing:
        # Prefer a GUI message when Qt is available.
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(
                None,
                "Missing dependencies",
                "These packages are required but not installed:\n\n"
                + "\n".join(f"  · {name}" for name in missing)
                + "\n\nInstall them with:\n  pip install -r requirements.txt",
            )
        except Exception:
            print("Missing dependencies:", ", ".join(missing), file=sys.stderr)
        return 1

    from PySide6.QtWidgets import QMessageBox

    from src.config import APP_VERSION, load_config, save_config
    from src.utils.logging_setup import configure_logging

    load_config()
    # Ensure a writable default config exists for operators to tune.
    try:
        save_config()
    except OSError:
        pass

    logger = configure_logging()
    logger.info("Starting %s v%s", "Phone Cover Mockup Studio", APP_VERSION)

    app = create_application()

    try:
        from src.ui.main_window import MainWindow

        window = MainWindow()
        window.show()
        code = app.exec()
        logger.info("Application exited with code %s", code)
        return code
    except Exception as exc:
        logger = configure_logging()
        logger.exception("Failed to start application")
        QMessageBox.critical(
            None,
            "Failed to start",
            f"The application could not start:\n\n{exc}",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
