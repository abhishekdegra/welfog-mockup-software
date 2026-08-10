"""
Visual theme for the application: colour palette and Qt stylesheet.
"""


class Palette:
    """Colour tokens shared by the stylesheet and the custom painted widgets."""

    BG = "#0B0E14"
    BG_ELEVATED = "#111520"
    SURFACE = "#151A27"
    SURFACE_HOVER = "#1B2233"
    SURFACE_ACTIVE = "#222B40"
    CARD = "#131824"
    BORDER = "#232B3D"
    BORDER_STRONG = "#2E3852"

    TEXT = "#E8ECF7"
    TEXT_MUTED = "#8792AC"
    TEXT_DIM = "#5F6980"

    ACCENT = "#6C8CFF"
    ACCENT_HOVER = "#8AA2FF"
    ACCENT_PRESSED = "#5674E8"
    ACCENT_SOFT = "#2A3556"

    VIOLET = "#A78BFA"
    TEAL = "#2DD4BF"
    SUCCESS = "#34D399"
    WARNING = "#FBBF24"
    DANGER = "#F87171"
    DANGER_HOVER = "#FB8B8B"

    CANVAS_BG = "#080A10"
    GRID = "#141926"


DARK_THEME_STYLES = f"""
/* ---------------------------------------------------------------- base */
QWidget {{
    background-color: {Palette.BG};
    color: {Palette.TEXT};
    font-family: 'Segoe UI Variable', 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 9.5pt;
}}

QMainWindow {{
    background-color: {Palette.BG};
}}

QMainWindow::separator {{
    background: {Palette.BORDER};
    width: 1px;
    height: 1px;
}}

/* ------------------------------------------------------------- menu bar */
QMenuBar {{
    background-color: {Palette.BG_ELEVATED};
    color: {Palette.TEXT_MUTED};
    border-bottom: 1px solid {Palette.BORDER};
    padding: 3px 6px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    margin: 0 2px;
    border-radius: 6px;
}}

QMenuBar::item:selected {{
    background-color: {Palette.SURFACE_HOVER};
    color: {Palette.TEXT};
}}

QMenu {{
    background-color: {Palette.BG_ELEVATED};
    border: 1px solid {Palette.BORDER};
    border-radius: 10px;
    padding: 6px;
}}

QMenu::item {{
    padding: 7px 28px 7px 16px;
    border-radius: 6px;
    color: {Palette.TEXT};
}}

QMenu::item:selected {{
    background-color: {Palette.ACCENT_SOFT};
}}

QMenu::item:disabled {{
    color: {Palette.TEXT_DIM};
}}

QMenu::separator {{
    height: 1px;
    background: {Palette.BORDER};
    margin: 6px 10px;
}}

/* -------------------------------------------------------------- buttons */
QPushButton {{
    background-color: {Palette.SURFACE};
    border: 1px solid {Palette.BORDER};
    border-radius: 9px;
    padding: 8px 14px;
    color: {Palette.TEXT};
    font-weight: 600;
    min-height: 28px;
}}

QPushButton:hover {{
    background-color: {Palette.SURFACE_HOVER};
    border-color: {Palette.BORDER_STRONG};
}}

QPushButton:pressed {{
    background-color: {Palette.SURFACE_ACTIVE};
}}

QPushButton:disabled {{
    background-color: #0F131C;
    color: {Palette.TEXT_DIM};
    border-color: #1A1F2C;
}}

QPushButton#primaryButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {Palette.ACCENT}, stop:1 {Palette.VIOLET});
    border: none;
    color: #0A0D14;
    font-weight: 700;
}}

QPushButton#primaryButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {Palette.ACCENT_HOVER}, stop:1 #B79CFF);
}}

QPushButton#primaryButton:pressed {{
    background: {Palette.ACCENT_PRESSED};
}}

QPushButton#primaryButton:disabled {{
    background: #1A2036;
    color: {Palette.TEXT_DIM};
}}

QPushButton#successButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {Palette.TEAL}, stop:1 {Palette.SUCCESS});
    border: none;
    color: #06231D;
    font-weight: 700;
}}

QPushButton#successButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #4FE3D2, stop:1 #58E4AE);
}}

QPushButton#successButton:disabled {{
    background: #12241F;
    color: {Palette.TEXT_DIM};
}}

QPushButton#dangerButton {{
    background-color: transparent;
    border: 1px solid #4A2733;
    color: {Palette.DANGER};
    padding: 7px 12px;
    min-height: 28px;
    min-width: 0px;
}}

QPushButton#dangerButton:hover {{
    background-color: #2A1620;
    border-color: {Palette.DANGER};
    color: {Palette.DANGER_HOVER};
}}

QPushButton#ghostButton {{
    background-color: transparent;
    border: 1px solid {Palette.BORDER};
    color: {Palette.TEXT_MUTED};
    font-weight: 600;
    padding: 7px 12px;
    min-height: 28px;
    min-width: 0px;
}}

QPushButton#ghostButton:hover {{
    background-color: {Palette.SURFACE};
    color: {Palette.TEXT};
}}

QPushButton#ghostButton:checked {{
    background-color: {Palette.ACCENT_SOFT};
    border-color: {Palette.ACCENT};
    color: {Palette.ACCENT_HOVER};
}}

QPushButton#toolButton {{
    background-color: {Palette.SURFACE};
    border: 1px solid {Palette.BORDER};
    border-radius: 8px;
    padding: 6px 12px;
    color: {Palette.TEXT_MUTED};
    font-weight: 600;
    font-size: 9pt;
    min-height: 28px;
    min-width: 0px;
}}

QPushButton#toolButton:hover {{
    background-color: {Palette.SURFACE_HOVER};
    color: {Palette.TEXT};
}}

QPushButton#toolButton:checked {{
    background-color: {Palette.ACCENT_SOFT};
    border-color: {Palette.ACCENT};
    color: {Palette.ACCENT_HOVER};
}}

QPushButton#toolButtonCompact {{
    background-color: {Palette.SURFACE};
    border: 1px solid {Palette.BORDER};
    border-radius: 8px;
    padding: 6px 8px;
    color: {Palette.TEXT_MUTED};
    font-weight: 700;
    font-size: 10pt;
    min-width: 32px;
    max-width: 40px;
    min-height: 28px;
}}

QPushButton#toolButtonCompact:hover {{
    background-color: {Palette.SURFACE_HOVER};
    color: {Palette.TEXT};
}}

QPushButton#linkButton {{
    background: transparent;
    border: none;
    color: {Palette.ACCENT};
    font-weight: 600;
    padding: 2px 4px;
    text-align: right;
}}

QPushButton#linkButton:hover {{
    color: {Palette.ACCENT_HOVER};
}}

/* --------------------------------------------------------------- labels */
QLabel {{
    background: transparent;
    color: {Palette.TEXT};
}}

QLabel#appTitle {{
    font-size: 13pt;
    font-weight: 700;
    color: {Palette.TEXT};
}}

QLabel#appSubtitle {{
    font-size: 8pt;
    color: {Palette.TEXT_DIM};
}}

QLabel#panelTitle {{
    font-size: 11pt;
    font-weight: 700;
    color: {Palette.TEXT};
}}

QLabel#sectionTitle {{
    font-size: 9pt;
    font-weight: 700;
    color: {Palette.TEXT_MUTED};
    letter-spacing: 0px;
}}

QLabel#sliderLabel {{
    color: {Palette.TEXT_MUTED};
    font-size: 9pt;
}}

QLabel#sliderValue {{
    color: {Palette.ACCENT_HOVER};
    background-color: {Palette.SURFACE};
    border: 1px solid {Palette.BORDER};
    border-radius: 6px;
    font-size: 8.5pt;
    font-weight: 700;
    padding: 2px 6px;
}}

QLabel#infoLabel {{
    color: {Palette.TEXT_DIM};
    font-size: 8.5pt;
}}

QLabel#badgeMuted {{
    background-color: {Palette.SURFACE};
    color: {Palette.TEXT_DIM};
    border: 1px solid {Palette.BORDER};
    border-radius: 7px;
    padding: 3px 8px;
    font-size: 8pt;
    font-weight: 600;
}}

QLabel#badgeLabel {{
    background-color: {Palette.ACCENT_SOFT};
    color: {Palette.ACCENT_HOVER};
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 3px 8px;
    font-size: 8pt;
    font-weight: 700;
}}

/* ---------------------------------------------------------------- cards */
QFrame#card {{
    background-color: {Palette.CARD};
    border: 1px solid {Palette.BORDER};
    border-radius: 14px;
}}

QFrame#headerBar {{
    background-color: {Palette.BG_ELEVATED};
    border: none;
    border-bottom: 1px solid {Palette.BORDER};
}}

QFrame#canvasCard {{
    background-color: {Palette.CANVAS_BG};
    border: 1px solid {Palette.BORDER};
    border-radius: 14px;
}}

QFrame#floatingBar {{
    background-color: {Palette.BG_ELEVATED};
    border: 1px solid {Palette.BORDER};
    border-radius: 12px;
}}

QFrame#divider {{
    background-color: {Palette.BORDER};
    border: none;
    max-height: 1px;
}}

QFrame#vDivider {{
    background-color: {Palette.BORDER};
    border: none;
    max-width: 1px;
}}

/* ------------------------------------------------------------ group box */
QGroupBox {{
    background-color: {Palette.CARD};
    border: 1px solid {Palette.BORDER};
    border-radius: 12px;
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    font-weight: 700;
    color: {Palette.TEXT_MUTED};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 2px 8px;
    background-color: {Palette.BG};
    color: {Palette.TEXT_MUTED};
    font-size: 8.5pt;
    letter-spacing: 0px;
}}

/* --------------------------------------------------------------- slider */
QSlider {{
    background: transparent;
    min-height: 22px;
}}

QSlider::groove:horizontal {{
    height: 5px;
    background: #1C2233;
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {Palette.ACCENT}, stop:1 {Palette.VIOLET});
    border-radius: 3px;
}}

QSlider::add-page:horizontal {{
    background: #1C2233;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #F2F5FF;
    border: 3px solid {Palette.ACCENT};
    width: 8px;
    height: 8px;
    margin: -6px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    border-color: {Palette.ACCENT_HOVER};
    background: #FFFFFF;
}}

QSlider::handle:horizontal:pressed {{
    border-color: {Palette.VIOLET};
}}

QSlider:disabled::sub-page:horizontal {{
    background: #232838;
}}

QSlider:disabled::handle:horizontal {{
    background: #3A4358;
    border-color: #2A3244;
}}

/* ----------------------------------------------------------- scroll bar */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: #29324A;
    border-radius: 5px;
    min-height: 28px;
}}

QScrollBar::handle:vertical:hover {{
    background: #38446B;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    background: #29324A;
    border-radius: 5px;
    min-width: 28px;
}}

QScrollBar::handle:horizontal:hover {{
    background: #38446B;
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0px;
    height: 0px;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

/* ---------------------------------------------------------------- input */
QComboBox {{
    background-color: {Palette.SURFACE};
    border: 1px solid {Palette.BORDER};
    border-radius: 9px;
    padding: 7px 28px 7px 12px;
    color: {Palette.TEXT};
    min-height: 18px;
    min-width: 0px;
}}

QComboBox:hover {{
    border-color: {Palette.BORDER_STRONG};
}}

QComboBox:focus {{
    border-color: {Palette.ACCENT};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border: none;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {Palette.TEXT_MUTED};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {Palette.BG_ELEVATED};
    border: 1px solid {Palette.BORDER};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {Palette.ACCENT_SOFT};
    selection-color: {Palette.TEXT};
    outline: none;
}}

QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {Palette.SURFACE};
    border: 1px solid {Palette.BORDER};
    border-radius: 9px;
    padding: 7px 10px;
    color: {Palette.TEXT};
    selection-background-color: {Palette.ACCENT_SOFT};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {Palette.ACCENT};
}}

QCheckBox {{
    color: {Palette.TEXT_MUTED};
    spacing: 9px;
    padding: 3px 0;
}}

QCheckBox:hover {{
    color: {Palette.TEXT};
}}

QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border: 1px solid {Palette.BORDER_STRONG};
    border-radius: 5px;
    background-color: {Palette.SURFACE};
}}

QCheckBox::indicator:hover {{
    border-color: {Palette.ACCENT};
}}

QCheckBox::indicator:checked {{
    background-color: {Palette.ACCENT};
    border-color: {Palette.ACCENT};
    image: none;
}}

QRadioButton {{
    color: {Palette.TEXT_MUTED};
    spacing: 9px;
}}

QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {Palette.BORDER_STRONG};
    border-radius: 9px;
    background-color: {Palette.SURFACE};
}}

QRadioButton::indicator:checked {{
    background-color: {Palette.ACCENT};
    border-color: {Palette.ACCENT};
}}

/* ----------------------------------------------------------------- tabs */
QTabWidget::pane {{
    background-color: transparent;
    border: none;
    top: 6px;
}}

QTabBar {{
    qproperty-drawBase: 0;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {Palette.TEXT_DIM};
    padding: 8px 12px;
    margin-right: 4px;
    border-radius: 9px;
    font-size: 9pt;
    font-weight: 600;
    min-width: 72px;
}}

QTabBar::tab:hover {{
    color: {Palette.TEXT};
    background-color: {Palette.SURFACE};
}}

QTabBar::tab:selected {{
    background-color: {Palette.ACCENT_SOFT};
    color: {Palette.ACCENT_HOVER};
}}

/* --------------------------------------------------------- scroll areas */
QScrollArea {{
    background: transparent;
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background: transparent;
}}

QSplitter::handle {{
    background-color: transparent;
    width: 8px;
}}

QSplitter::handle:hover {{
    background-color: {Palette.ACCENT_SOFT};
}}

/* ----------------------------------------------------------- status bar */
QStatusBar {{
    background-color: {Palette.BG_ELEVATED};
    color: {Palette.TEXT_DIM};
    border-top: 1px solid {Palette.BORDER};
    padding: 3px 10px;
    font-size: 8.5pt;
}}

QStatusBar::item {{
    border: none;
}}

/* --------------------------------------------------------- progress bar */
QProgressBar {{
    background-color: {Palette.SURFACE};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {Palette.ACCENT}, stop:1 {Palette.TEAL});
    border-radius: 4px;
}}

/* -------------------------------------------------------------- tooltip */
QToolTip {{
    background-color: {Palette.BG_ELEVATED};
    color: {Palette.TEXT};
    border: 1px solid {Palette.BORDER_STRONG};
    border-radius: 8px;
    padding: 6px 9px;
    font-size: 8.5pt;
}}

/* --------------------------------------------------------------- dialog */
QDialog, QMessageBox {{
    background-color: {Palette.BG_ELEVATED};
}}

QMessageBox QLabel {{
    color: {Palette.TEXT};
}}

QDialog QPushButton, QMessageBox QPushButton {{
    min-width: 92px;
}}
"""
