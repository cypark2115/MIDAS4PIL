"""
Main application window (redesigned).

Two-tab layout (Calibration | Integration) with shared SyncBus for
synchronized 2theta cursor.  System QPalette applied at startup.
"""

import logging
import webbrowser
from pathlib import Path

from PySide6.QtWidgets import (QMainWindow, QTabWidget,
                                QFileDialog, QDialog, QVBoxLayout, QHBoxLayout,
                                QPlainTextEdit, QPushButton, QApplication,
                                QProgressBar, QMessageBox)
from PySide6.QtGui import QPalette, QColor, QAction, QActionGroup
from PySide6.QtCore import Qt

# Root of the source tree (midas4pil/gui/main_window.py → ../../..)
_PKG_ROOT  = Path(__file__).resolve().parents[2]
# midas4pil package directory — docs/ lives here in both dev and installed modes
_MIDAS_PKG = Path(__file__).resolve().parents[1]

from .crosshair import SyncBus
from .calib_tab import CalibrationTab

log = logging.getLogger(__name__)
from .integ_tab import IntegrationTab
from .help_text import USAGE_GUIDE, TECHNICAL_README


# Matplotlib colour sets per theme
_MPL_THEMES = {
    'dark': {
        'fig':       '#1a1a1a',
        'ax':        '#2d2d2d',
        'fg':        '#cccccc',
        'spine':     '#555555',
        'cmap':      'gray',       # low intensity = dark
        'pos_label': '#ffffff',    # white text on dark background
        'hist_bar':  '#888888',
        # Lineout curve colours (visible on dark axes)
        'line_I':    '#64b5f6',   # light blue
        'line_Isub': '#00e676',   # green
        'line_snip': '#ff9800',   # orange
        'ring':      '#ff9800',   # calibrant ring markers
    },
    'light': {
        'fig':       '#f5f5f5',
        'ax':        '#ffffff',
        'fg':        '#222222',
        'spine':     '#aaaaaa',
        'cmap':      'gray_r',     # low intensity = bright (like paper)
        'pos_label': '#111111',    # dark text on light background
        'hist_bar':  '#666666',
        # Lineout curve colours (visible on white axes)
        'line_I':    '#1565c0',   # dark blue
        'line_Isub': '#00695c',   # dark teal
        'line_snip': '#bf360c',   # dark orange-red
        'ring':      '#b71c1c',   # dark red for ring markers
    },
    'system': {
        'fig':       '#eeeeee',
        'ax':        '#f8f8f8',
        'fg':        '#333333',
        'spine':     '#999999',
        'cmap':      'gray_r',
        'pos_label': '#111111',
        'hist_bar':  '#666666',
        'line_I':    '#1565c0',
        'line_Isub': '#00695c',
        'line_snip': '#bf360c',
        'ring':      '#b71c1c',
    },
}


def _apply_dark_palette(app):
    """Apply a dark colour palette to the entire application."""
    p = QPalette()
    c = QColor
    CG = QPalette.ColorGroup
    CR = QPalette.ColorRole
    p.setColor(CR.Window,          c(42, 42, 42))
    p.setColor(CR.WindowText,       c(210, 210, 210))
    p.setColor(CR.Base,             c(28, 28, 28))
    p.setColor(CR.AlternateBase,    c(50, 50, 50))
    p.setColor(CR.Text,             c(210, 210, 210))
    p.setColor(CR.Button,           c(55, 55, 55))
    p.setColor(CR.ButtonText,       c(210, 210, 210))
    p.setColor(CR.BrightText,       c(255, 255, 255))
    p.setColor(CR.Highlight,        c(0, 120, 215))
    p.setColor(CR.HighlightedText,  c(255, 255, 255))
    p.setColor(CR.ToolTipBase,      c(50, 50, 50))
    p.setColor(CR.ToolTipText,      c(210, 210, 210))
    p.setColor(CR.PlaceholderText,  c(120, 120, 120))
    p.setColor(CR.Mid,              c(70, 70, 70))
    p.setColor(CR.Dark,             c(35, 35, 35))
    p.setColor(CR.Shadow,           c(20, 20, 20))
    p.setColor(CR.Link,             c(100, 180, 255))
    p.setColor(CR.LinkVisited,      c(170, 130, 255))
    # Disabled colour group — Fusion uses these for greyed-out controls
    p.setColor(CG.Disabled, CR.WindowText,  c(100, 100, 100))
    p.setColor(CG.Disabled, CR.Text,        c(100, 100, 100))
    p.setColor(CG.Disabled, CR.ButtonText,  c(100, 100, 100))
    p.setColor(CG.Disabled, CR.Base,        c(35,  35,  35))
    p.setColor(CG.Disabled, CR.Button,      c(42,  42,  42))
    p.setColor(CG.Disabled, CR.Highlight,   c(50,  80, 120))
    app.setPalette(p)
    app.setStyleSheet(
        "QMenuBar { background: #2d2d2d; color: #d2d2d2; }"
        "QMenuBar::item:selected { background: #0066cc; color: #ffffff; }"
        "QMenu { background: #2d2d2d; color: #d2d2d2; border: 1px solid #555; }"
        "QMenu::item { padding: 4px 24px 4px 12px; }"
        "QMenu::item:selected { background: #0066cc; color: #ffffff; }"
        "QMenu::separator { height: 1px; background: #555; margin: 2px 0px; }"
        "QScrollBar:vertical { width: 14px; }"
        "QScrollBar:horizontal { height: 14px; }"
        "QGroupBox { border: 1px solid #555; border-radius: 4px; "
        "            margin-top: 8px; padding-top: 4px; color: #d2d2d2; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        "QLabel { color: #d2d2d2; }"
        "QRadioButton { color: #d2d2d2; }"
        "QCheckBox { color: #d2d2d2; }"
        "QPushButton { border: 1px solid #666; border-radius: 3px; "
        "              padding: 3px 8px; background: #3a3a3a; color: #d2d2d2; }"
        "QPushButton:hover { background: #4a4a4a; color: #d2d2d2; }"
        "QPushButton:checked { background: #005a9e; border-color: #0078d4; "
        "                      color: #ffffff; }"
        "QPushButton:pressed { background: #2a2a2a; color: #d2d2d2; }"
        "QPushButton:disabled { background: #2a2a2a; color: #666666; "
        "                       border-color: #444; }"
        "QLineEdit { background: #2a2a2a; border: 1px solid #555; "
        "            border-radius: 2px; padding: 2px 4px; color: #d2d2d2; }"
        "QAbstractSpinBox { background: #2a2a2a; border: 1px solid #555; "
        "                   border-radius: 2px; padding: 2px 4px; color: #d2d2d2; }"
        "QAbstractSpinBox::up-button, QAbstractSpinBox::down-button "
        "  { background: #3a3a3a; border: none; width: 14px; }"
        "QComboBox { background: #3a3a3a; border: 1px solid #555; "
        "            border-radius: 2px; padding: 2px 4px; color: #d2d2d2; }"
        "QComboBox QAbstractItemView { background: #2d2d2d; color: #d2d2d2; "
        "                              selection-background-color: #0066cc; }"
        "QListWidget { background: #222; border: 1px solid #444; color: #d2d2d2; }"
        "QTabBar::tab { background: #3a3a3a; padding: 5px 12px; color: #d2d2d2; "
        "               border: 1px solid #555; border-bottom: none; }"
        "QTabBar::tab:selected { background: #1a1a1a; color: #ffffff; }"
        "QSplitter::handle:horizontal { background: #555; width: 4px; }"
        "QSplitter::handle:vertical   { background: #555; height: 4px; }"
        "QSplitter::handle:hover { background: #888; }"
    )


def _apply_light_palette(app):
    """Light theme — white/light-grey Qt palette."""
    p = QPalette()
    c = QColor
    CG = QPalette.ColorGroup
    CR = QPalette.ColorRole
    p.setColor(CR.Window,          c(240, 240, 240))
    p.setColor(CR.WindowText,       c(30,  30,  30))
    p.setColor(CR.Base,             c(255, 255, 255))
    p.setColor(CR.AlternateBase,    c(233, 233, 233))
    p.setColor(CR.Text,             c(30,  30,  30))
    p.setColor(CR.Button,           c(225, 225, 225))
    p.setColor(CR.ButtonText,       c(30,  30,  30))
    p.setColor(CR.BrightText,       c(0,   0,   0))
    p.setColor(CR.Highlight,        c(0,   120, 215))
    p.setColor(CR.HighlightedText,  c(255, 255, 255))
    p.setColor(CR.ToolTipBase,      c(255, 255, 220))
    p.setColor(CR.ToolTipText,      c(30,  30,  30))
    p.setColor(CR.PlaceholderText,  c(140, 140, 140))
    p.setColor(CR.Mid,              c(180, 180, 180))
    p.setColor(CR.Dark,             c(160, 160, 160))
    p.setColor(CR.Shadow,           c(100, 100, 100))
    p.setColor(CR.Link,             c(0,   90,  180))
    p.setColor(CR.LinkVisited,      c(100,  50, 150))
    # Disabled colour group
    p.setColor(CG.Disabled, CR.WindowText,  c(140, 140, 140))
    p.setColor(CG.Disabled, CR.Text,        c(140, 140, 140))
    p.setColor(CG.Disabled, CR.ButtonText,  c(140, 140, 140))
    p.setColor(CG.Disabled, CR.Base,        c(245, 245, 245))
    p.setColor(CG.Disabled, CR.Button,      c(210, 210, 210))
    app.setPalette(p)
    app.setStyleSheet(
        "QMenuBar { background: #e8e8e8; color: #1a1a1a; }"
        "QMenuBar::item:selected { background: #0078d4; color: #ffffff; }"
        "QMenu { background: #f5f5f5; color: #1a1a1a; border: 1px solid #bbb; }"
        "QMenu::item { padding: 4px 24px 4px 12px; }"
        "QMenu::item:selected { background: #0078d4; color: #ffffff; }"
        "QMenu::separator { height: 1px; background: #ccc; margin: 2px 0px; }"
        "QScrollBar:vertical { width: 14px; }"
        "QScrollBar:horizontal { height: 14px; }"
        "QGroupBox { border: 1px solid #bbb; border-radius: 4px; "
        "            margin-top: 8px; padding-top: 4px; color: #1a1a1a; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        "QLabel { color: #1a1a1a; }"
        "QRadioButton { color: #1a1a1a; }"
        "QCheckBox { color: #1a1a1a; }"
        "QPushButton { border: 1px solid #bbb; border-radius: 3px; "
        "              padding: 3px 8px; background: #e8e8e8; color: #1a1a1a; }"
        "QPushButton:hover { background: #d8d8d8; color: #1a1a1a; }"
        "QPushButton:checked { background: #0078d4; color: #ffffff; "
        "                      border-color: #005a9e; }"
        "QPushButton:pressed { background: #c0c0c0; color: #1a1a1a; }"
        "QPushButton:disabled { background: #d0d0d0; color: #999999; "
        "                       border-color: #ccc; }"
        "QLineEdit { background: #ffffff; border: 1px solid #bbb; "
        "            border-radius: 2px; padding: 2px 4px; color: #1a1a1a; }"
        "QAbstractSpinBox { background: #ffffff; border: 1px solid #bbb; "
        "                   border-radius: 2px; padding: 2px 4px; color: #1a1a1a; }"
        "QAbstractSpinBox::up-button, QAbstractSpinBox::down-button "
        "  { background: #e8e8e8; border: none; width: 14px; }"
        "QComboBox { background: #f0f0f0; border: 1px solid #bbb; "
        "            border-radius: 2px; padding: 2px 4px; color: #1a1a1a; }"
        "QComboBox QAbstractItemView { background: #ffffff; color: #1a1a1a; "
        "                              selection-background-color: #0078d4; }"
        "QListWidget { background: #ffffff; border: 1px solid #ccc; color: #1a1a1a; }"
        "QTabBar::tab { background: #e0e0e0; padding: 5px 12px; color: #1a1a1a; "
        "               border: 1px solid #bbb; border-bottom: none; }"
        "QTabBar::tab:selected { background: #f5f5f5; color: #1a1a1a; }"
        "QSplitter::handle:horizontal { background: #bbb; width: 4px; }"
        "QSplitter::handle:vertical   { background: #bbb; height: 4px; }"
        "QSplitter::handle:hover { background: #999; }"
    )


def _apply_system_palette(app):
    """Reset to the OS / system default theme."""
    app.setPalette(QPalette())
    app.setStyleSheet("")


class MainWindow(QMainWindow):
    """midas4pil main window: Calibration + Integration tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("midas4pil")

        # Apply system palette (default)
        self._current_theme = 'system'
        _apply_system_palette(QApplication.instance())

        # Size to 90% of available screen, capped at 1600×950, then centre.
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(1600, int(screen.width()  * 0.90))
        h = min(950,  int(screen.height() * 0.90))
        self.resize(w, h)
        self.setMinimumSize(700, 480)
        self.move(
            screen.x() + max(0, (screen.width()  - w) // 2),
            screen.y() + max(0, (screen.height() - h) // 2),
        )

        # ── Shared sync bus ──
        self._sync_bus = SyncBus(self)

        # ── Tabs ──
        self.tabs = QTabWidget()
        self.calib_tab = CalibrationTab()
        self.integ_tab = IntegrationTab(self._sync_bus)
        self.tabs.addTab(self.calib_tab, "Calibration")
        self.tabs.addTab(self.integ_tab, "Integration")
        self.setCentralWidget(self.tabs)

        # Apply initial mpl theme to match the Qt palette set above
        _init_colors = _MPL_THEMES[self._current_theme]
        self.calib_tab.apply_mpl_theme(_init_colors)
        self.integ_tab.apply_mpl_theme(_init_colors)

        # ── Cross-tab wiring ──
        self.calib_tab.geometry_ready.connect(self._on_geometry_from_calib)

        # ── Menus ──
        self._build_menus()

        # ── Status bar ──
        sb = self.statusBar()
        sb.setStyleSheet(
            "QStatusBar { font-size: 13px; min-height: 28px; padding-left: 4px; }")
        sb.showMessage("Ready")

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(220)
        self._progress_bar.setFixedHeight(18)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()
        sb.addPermanentWidget(self._progress_bar)

    # ── Progress bar API ──────────────────────────────────────────────────────

    def show_progress(self, value, maximum=100):
        """Show a determinate progress bar (e.g. calibration iterations)."""
        self._progress_bar.setRange(0, maximum)
        self._progress_bar.setValue(value)
        self._progress_bar.setFormat(f"%v / {maximum}")
        self._progress_bar.show()

    def show_busy(self):
        """Show an indeterminate (spinning) progress bar."""
        self._progress_bar.setRange(0, 0)
        self._progress_bar.show()

    def hide_progress(self):
        """Hide the progress bar and reset it."""
        self._progress_bar.hide()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")

    # ── Menus ─────────────────────────────────────────────────────────────────

    def _build_menus(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        act_load = QAction("Load &Geometry...", self)
        act_load.triggered.connect(self._on_load_geometry)
        file_menu.addAction(act_load)

        act_load_poni = QAction("Load &.poni...", self)
        act_load_poni.setToolTip("Load a .poni file as starting geometry")
        act_load_poni.triggered.connect(self._on_load_poni)
        file_menu.addAction(act_load_poni)

        act_load_midas = QAction("Load &MIDAS params...", self)
        act_load_midas.setToolTip("Load a MIDAS geometry_params.txt as starting geometry")
        act_load_midas.triggered.connect(self._on_load_midas)
        file_menu.addAction(act_load_midas)

        file_menu.addSeparator()

        act_exit = QAction("E&xit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # View menu — theme
        view_menu = menu_bar.addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)

        for label, key in (("&Dark",   'dark'),
                           ("&Light",  'light'),
                           ("&System", 'system')):
            act = QAction(label, self, checkable=True)
            act.setData(key)
            act.setChecked(key == self._current_theme)
            act.triggered.connect(
                lambda checked, k=key: self._on_theme(k))
            theme_group.addAction(act)
            theme_menu.addAction(act)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        act_usage = QAction("&Usage Guide", self)
        act_usage.triggered.connect(self._show_usage_guide)
        help_menu.addAction(act_usage)

        act_tech = QAction("&Technical README", self)
        act_tech.triggered.connect(self._show_technical_readme)
        help_menu.addAction(act_tech)

    def _on_theme(self, theme):
        if theme == self._current_theme:
            return
        self._current_theme = theme
        app = QApplication.instance()
        if theme == 'dark':
            _apply_dark_palette(app)
        elif theme == 'light':
            _apply_light_palette(app)
        else:
            _apply_system_palette(app)
        # Propagate to matplotlib canvases
        colors = _MPL_THEMES[theme]
        self.calib_tab.apply_mpl_theme(colors)
        self.integ_tab.apply_mpl_theme(colors)

    def _on_load_geometry(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load geometry file", "",
            "TOML files (*.toml);;All files (*)")
        if not path:
            return
        try:
            geom = self.calib_tab._load_toml(path)
            self.integ_tab.load_geometry(geom)
            n_ps = len(geom.get('panel_shifts', []))
            suffix = f" + {n_ps} panel shifts" if n_ps else ""
            self.statusBar().showMessage(
                f"Geometry loaded: {Path(path).name}{suffix}")
        except Exception as e:
            log.error("Load geometry failed: %s", e, exc_info=True)
            QMessageBox.warning(self, "Load Geometry Error",
                                f"Could not load geometry:\n{e}")
            self.statusBar().showMessage(f"Error loading geometry: {e}")

    def _on_load_poni(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load .poni file", "",
            "PONI files (*.poni);;All files (*)")
        if not path:
            return
        try:
            geom = self.calib_tab._load_poni(path)
            self.integ_tab.load_geometry(geom)
            self.statusBar().showMessage(
                f".poni loaded: {Path(path).name}"
                " (panel shifts / distortion unchanged)")
        except Exception as e:
            log.error("Load .poni failed: %s", e, exc_info=True)
            QMessageBox.warning(self, "Load .poni Error",
                                f"Could not load .poni file:\n{e}")
            self.statusBar().showMessage(f"Error loading .poni: {e}")

    def _on_load_midas(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load MIDAS params", "",
            "MIDAS params (*.txt);;All files (*)")
        if not path:
            return
        try:
            geom = self.calib_tab._load_midas_params(path)
            self.integ_tab.load_geometry(geom)
            n_ps = len(geom.get('panel_shifts', []))
            suffix = f" + {n_ps} panel shifts" if n_ps else ""
            self.statusBar().showMessage(
                f"MIDAS params loaded: {Path(path).name}{suffix}")
        except Exception as e:
            log.error("Load MIDAS params failed: %s", e, exc_info=True)
            QMessageBox.warning(self, "Load MIDAS Params",
                                f"Could not load MIDAS params:\n{e}")
            self.statusBar().showMessage(f"Error loading MIDAS params: {e}")

    def _on_geometry_from_calib(self, geom):
        self.integ_tab.load_geometry(geom)
        # Pass the calibration image folder so Browse starts there
        img_path = getattr(self.calib_tab, '_image_path', None)
        if img_path is not None:
            self.integ_tab.set_last_image_folder(Path(img_path).parent)
        # Transfer calibration mask to integration tab
        calib_mask = getattr(self.calib_tab, '_mask', None)
        if calib_mask is not None:
            self.integ_tab.load_mask(calib_mask)
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage("Geometry loaded from calibration")

    def _show_help_dialog(self, title, content, doc_path=None,
                          browser_label="Open in Browser"):
        """Show a plain-text help dialog.

        Parameters
        ----------
        doc_path : Path or None
            If provided and the file exists, a button labelled *browser_label*
            opens the file with the system default application (browser, editor,
            or markdown viewer depending on system file associations).
        browser_label : str
            Text for the browser button.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(820, 640)
        layout = QVBoxLayout(dlg)

        text = QPlainTextEdit(content)
        text.setReadOnly(True)
        layout.addWidget(text)

        btn_row = QHBoxLayout()
        if doc_path is not None and doc_path.exists():
            btn_browser = QPushButton(browser_label)
            btn_browser.setToolTip(f"Open {doc_path.name} with the system default application")
            _uri = doc_path.as_uri()
            btn_browser.clicked.connect(lambda: webbrowser.open(_uri))
            btn_row.addWidget(btn_browser)
        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dlg.exec()

    def _show_usage_guide(self):
        doc_path = _MIDAS_PKG / "docs" / "USAGE_GUIDE.md"
        self._show_help_dialog("Usage Guide", USAGE_GUIDE,
                               doc_path=doc_path,
                               browser_label="Open USAGE_GUIDE.md in Browser")

    def _show_technical_readme(self):
        doc_path = _MIDAS_PKG / "docs" / "README.md"
        self._show_help_dialog("Technical README", TECHNICAL_README,
                               doc_path=doc_path,
                               browser_label="Open README.md in Browser")
