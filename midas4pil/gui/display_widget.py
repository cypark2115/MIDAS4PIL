# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Cake + lineout display widget (redesigned).

Layout
------
Header bar  : [TIFF] [Cake] toggle  |  position label (2th, eta)
Splitter    : top = image area (image canvas + histogram side panel)
              bottom = lineout canvas

Image toggle
------------
TIFF view : raw detector image, aspect='equal' (true pixel ratio preserved).
            Green crosshair tracks mouse. LUT lookup gives (2th, eta).
Cake view : 2th vs eta axes, aspect='auto'. Green crosshair tracks mouse.

Contrast control
----------------
Narrow histogram canvas to the right of the image:
  upper axes : horizontal bar histogram of image pixel values
  lower axes : matplotlib RangeSlider → sets image vmin/vmax live

Synchronized cursor
-------------------
Mouse on this widget → emits sync_bus.tth_changed(tth_deg).
Receives sync_bus.tth_changed from stack widget → draws green vline on lineout
and (in cake view) a green vline on the cake image.
"""

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QPushButton, QSizePolicy,
                                QSplitter)
from PySide6.QtCore import Qt, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as _MplRect
import tifffile

from PySide6.QtWidgets import QDoubleSpinBox, QAbstractSpinBox

from .hist_clim import HistClimWidget

# Colour for crosshair / sync indicators — green works on all themes
_XH = '#00e676'   # material green A400

# Module-level defaults match the system/light theme so the initial render
# before apply_mpl_theme() is called looks correct with the default theme.
_FG     = '#333333'
_BG_FIG = '#eeeeee'
_BG_AX  = '#f8f8f8'

# Multi-lineout overlay colours — Material 600 palette, readable on both
# light and dark backgrounds.
_LINEOUT_COLORS = [
    '#1e88e5',  # blue 600
    '#43a047',  # green 600
    '#e53935',  # red 600
    '#8e24aa',  # purple 600
    '#00acc1',  # cyan 600
    '#d81b60',  # pink 600
    '#6d4c41',  # brown 600
    '#c0ca33',  # lime 600
]

# Default single-file line/bg colours (system theme; overridden by apply_mpl_theme)
_LINE_I    = '#1565c0'   # dark blue
_LINE_ISUB = '#00695c'   # dark teal
_LINE_SNIP = '#bf360c'   # dark orange-red
_LINE_RING = '#b71c1c'   # dark red


def _style_ax(ax, fg=_FG, bg=_BG_AX, spine='#999999'):
    """Apply theme colours to a matplotlib Axes."""
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(spine)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)


class _LineYLimWidget(QWidget):
    """Compact y-axis min/max control for the lineout panel.

    Two spinboxes (Max on top, Min on bottom) with an Auto reset button.
    Fixed width matches HistClimWidget so the lineout matplotlib canvas is the
    same physical width as the image canvas, keeping the 2θ x-axes aligned.
    """

    def __init__(self, callback, auto_callback=None, parent=None):
        super().__init__(parent)
        self.setFixedWidth(110)

        self._callback      = callback        # callable(vmin, vmax)
        self._auto_callback = auto_callback   # called when Auto button pressed
        self._auto_vmin = None
        self._auto_vmax = None
        self._updating  = False      # suppress recursive valueChanged signals

        vl = QVBoxLayout(self)
        vl.setContentsMargins(6, 8, 6, 8)
        vl.setSpacing(4)

        lbl_max = QLabel("Y Max")
        lbl_max.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(lbl_max)

        self._spn_max = QDoubleSpinBox()
        self._spn_max.setDecimals(2)
        self._spn_max.setRange(-1e15, 1e15)
        self._spn_max.setStepType(
            QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self._spn_max.setAlignment(Qt.AlignmentFlag.AlignRight)
        vl.addWidget(self._spn_max)

        vl.addStretch(1)

        lbl_min = QLabel("Y Min")
        lbl_min.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(lbl_min)

        self._spn_min = QDoubleSpinBox()
        self._spn_min.setDecimals(2)
        self._spn_min.setRange(-1e15, 1e15)
        self._spn_min.setStepType(
            QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self._spn_min.setAlignment(Qt.AlignmentFlag.AlignRight)
        vl.addWidget(self._spn_min)

        btn_auto = QPushButton("Auto")
        btn_auto.clicked.connect(self._on_auto)
        vl.addWidget(btn_auto)

        self._spn_max.valueChanged.connect(self._on_changed)
        self._spn_min.valueChanged.connect(self._on_changed)

    def set_limits(self, vmin, vmax):
        """Update spinboxes to match new auto-scaled limits."""
        self._auto_vmin = vmin
        self._auto_vmax = vmax
        self._updating = True
        self._spn_min.setValue(vmin)
        self._spn_max.setValue(vmax)
        self._updating = False

    def _on_changed(self):
        if self._updating:
            return
        vmin = self._spn_min.value()
        vmax = self._spn_max.value()
        if vmin < vmax and self._callback:
            self._callback(vmin, vmax)

    def _on_auto(self):
        if self._auto_callback:
            self._auto_callback()
            return
        if self._auto_vmin is not None and self._auto_vmax is not None:
            self.set_limits(self._auto_vmin, self._auto_vmax)
            if self._callback:
                self._callback(self._auto_vmin, self._auto_vmax)

    def clear(self):
        self._auto_vmin = None
        self._auto_vmax = None
        self._updating = True
        self._spn_min.setValue(0.0)
        self._spn_max.setValue(1.0)
        self._updating = False

    def apply_mpl_theme(self, colors):
        pass  # Qt spinboxes follow the application palette automatically


class _NavBar(NavigationToolbar2QT):
    """Toolbar with 'Subplots' removed — it adjusts subplot params which have
    no effect on axes created via fig.add_axes() with explicit coordinates."""
    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] not in ('Subplots', 'Customize')]


def _refresh_navbar_icons(toolbar):
    """Re-render NavigationToolbar2QT icons with the current palette foreground."""
    try:
        for (_text, _tip, image_file, _cb), action in zip(
                toolbar.toolitems, toolbar.actions()):
            if image_file:
                action.setIcon(toolbar._icon(image_file + '.png'))
    except Exception:
        pass


class CakeLineoutWidget(QWidget):
    """Dual-view display: TIFF or Cake (top) + Lineout (bottom)."""

    eta_range_changed      = Signal(float, float)  # emitted when eta lines are dragged
    tth_snip_range_changed = Signal(float, float)  # emitted when SNIP range vlines are dragged

    def __init__(self, sync_bus, parent=None):
        super().__init__(parent)
        self._sync_bus = sync_bus

        # ── State ──
        self._view = 'tiff'          # 'tiff' or 'cake'
        self._tif_data = None        # raw TIFF array (nrows, ncols)
        self._cake_data = None       # cake array (n_tth, n_eta)
        self._tth_cake = None        # 2theta bin centres
        self._eta_cake = None        # eta bin centres
        self._lineout = None         # (tth, I, bg, I_sub, sigma, px_cnt) tuple
        self._px_cnt      = None     # 1-D int array, px_cnt per tth bin
        self._px_cnt_cake = None     # 2-D int array, px_cnt per cake cell
        self._img_handle = None      # imshow AxesImage for main image
        self._hist_clim = None       # HistClimWidget instance
        self._cmap      = 'gray_r'   # default matches system theme

        # Current theme colours — updated by apply_mpl_theme(); used when
        # drawing lineout curves, legends, and ring markers.
        self._theme_colors = {
            'fig': _BG_FIG, 'ax': _BG_AX, 'fg': _FG, 'spine': '#999999',
            'cmap': 'gray_r', 'pos_label': '#111111', 'hist_bar': '#666666',
            'line_I': _LINE_I, 'line_Isub': _LINE_ISUB,
            'line_snip': _LINE_SNIP, 'ring': _LINE_RING,
        }

        # Lineout overlay store: base → (tth, I, bg, I_sub, sigma, px_cnt)
        # Empty store → fall back to self._lineout (single preview)
        self._lineout_store = {}
        self._lineout_title = ''

        # Curve visibility toggles (single-lineout view)
        self._lo_show       = {'I': True, 'I_sub': False, 'SNIP': False}
        self._lo_show_sigma = True

        # Mask overlay
        self._mask      = None    # bool array (True=bad) or None
        self._msk_handle = None   # RGBA imshow handle for overlay

        # Eta sector lines (cake view)
        self._eta_lo = -180.0
        self._eta_hi = 180.0
        self._eta_lo_line = None  # axhline handle
        self._eta_hi_line = None  # axhline handle
        self._drag_eta = None     # None, 'lo', or 'hi'

        # SNIP 2θ range vlines (lineout view)
        self._tth_snip_lo = 0.0
        self._tth_snip_hi = 90.0
        self._tth_lo_vline = None  # axvline handle
        self._tth_hi_vline = None  # axvline handle
        self._drag_tth = None      # None, 'lo', or 'hi'

        # LUTs for TIFF pixel → (2th, eta) lookup
        self.tth_lut = None
        self.eta_lut = None

        # ── Zoom state: image ──
        self._img_zoom_stack = []
        self._img_home_xlim  = None
        self._img_home_ylim  = None
        self._img_roi_start  = None
        self._img_dragging   = False
        self._img_roi_patch  = None

        # ── Zoom state: lineout ──
        self._line_zoom_stack = []
        self._line_home_xlim  = None
        self._line_home_ylim  = None
        self._line_roi_start  = None
        self._line_dragging   = False
        self._line_roi_patch  = None
        self._y_locked        = False   # True = keep current ylim until Auto pressed
        self._line_x_locked   = False   # True = user has explicitly zoomed the lineout x-axis

        self._build_ui()
        self._connect_sync()

    # ── UI construction ──

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        root.addWidget(self._build_header())

        # Main splitter: image area (top) + lineout (bottom)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_image_section())
        splitter.addWidget(self._build_lineout_section())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    def _build_header(self):
        header = QWidget()
        header.setFixedHeight(32)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(6)

        self._btn_tiff = QPushButton("Image")
        self._btn_tiff.setCheckable(True)
        self._btn_tiff.setChecked(True)
        self._btn_tiff.setFixedWidth(52)
        self._btn_tiff.clicked.connect(lambda: self._set_view('tiff'))

        self._btn_cake = QPushButton("Cake")
        self._btn_cake.setCheckable(True)
        self._btn_cake.setChecked(False)
        self._btn_cake.setFixedWidth(52)
        self._btn_cake.clicked.connect(lambda: self._set_view('cake'))

        self._pos_label = QLabel("2\u03b8 = ---   \u03b7 = ---")
        self._pos_label.setStyleSheet(
            "color: #ffffff; font-size: 13px; font-family: monospace;")

        hl.addWidget(self._btn_tiff)
        hl.addWidget(self._btn_cake)
        hl.addSpacing(12)
        hl.addWidget(self._pos_label)
        hl.addStretch()
        self._header_hl = hl
        return header

    def add_to_header(self, widget):
        """Insert *widget* into the header bar after the Cake button."""
        # Layout order: 0=TIFF, 1=Cake, 2=spacing(12), 3=pos_label, 4=stretch
        self._header_hl.insertWidget(2, widget)

    def _build_image_section(self):
        """Image canvas (left, stretch) + histogram side panel (right, fixed).

        Layout (vertical):
          top row  (stretch=1): [image canvas | histogram]  ← same height
          bottom row (fixed):    [toolbar spanning full width + 110-px spacer]

        Putting the toolbar in a separate bottom row makes the histogram exactly
        as tall as the image canvas — previously the toolbar lived inside
        canvas_col, making the histogram ~28 px taller than the canvas.
        """
        widget = QWidget()
        vl = QVBoxLayout(widget)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # ── Top row: image canvas + histogram (same height) ───────────────────
        top_row = QWidget()
        hl = QHBoxLayout(top_row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        self.img_fig = Figure(facecolor=_BG_FIG)
        self.img_ax = self.img_fig.add_axes([0.06, 0.10, 0.92, 0.86])
        _style_ax(self.img_ax)
        self.img_ax.set_title("", color=_FG, fontsize=9)

        # Crosshair lines on image — green, initially invisible
        self._img_vline = self.img_ax.axvline(
            0, color=_XH, lw=0.8, ls='--', visible=False, alpha=0.85)
        self._img_hline = self.img_ax.axhline(
            0, color=_XH, lw=0.8, ls='--', visible=False, alpha=0.85)

        self.img_canvas = FigureCanvasQTAgg(self.img_fig)
        self.img_canvas.mpl_connect('motion_notify_event',  self._on_img_motion)
        self.img_canvas.mpl_connect('axes_leave_event',     self._on_img_leave)
        self.img_canvas.mpl_connect('scroll_event',         self._on_img_scroll)
        self.img_canvas.mpl_connect('button_press_event',   self._on_img_press)
        self.img_canvas.mpl_connect('button_release_event', self._on_img_release)
        hl.addWidget(self.img_canvas, stretch=1)

        # Histogram + contrast side panel — same height as canvas
        self._hist_clim = HistClimWidget(width=110, parent=self)
        hl.addWidget(self._hist_clim, stretch=0)

        vl.addWidget(top_row, stretch=1)

        # ── Bottom row: toolbar + spacer (aligns with lineout section) ────────
        bottom_row = QWidget()
        hl2 = QHBoxLayout(bottom_row)
        hl2.setContentsMargins(0, 0, 0, 0)
        hl2.setSpacing(0)

        self._img_toolbar = _NavBar(self.img_canvas, bottom_row, coordinates=False)
        self._img_toolbar.setMaximumHeight(28)
        hl2.addWidget(self._img_toolbar, stretch=1)

        vl.addWidget(bottom_row, stretch=0)
        return widget

    def _build_lineout_section(self):
        """Lineout canvas (left, stretch) + 110-px ylim panel (right, fixed).

        Layout (vertical):
          top row  (stretch=1): [line_canvas (stretch) | _line_ylim (110px)]
          bottom row (fixed):    [toolbar spanning full width]

        The 110-px ylim panel matches the HistClimWidget width in the image
        section so both matplotlib canvases are identical physical width.
        Combined with identical axes left/width fractions the 2θ x-axes align.
        """
        widget = QWidget()
        vl_outer = QVBoxLayout(widget)
        vl_outer.setContentsMargins(0, 0, 0, 0)
        vl_outer.setSpacing(0)

        # Lineout header: curve toggle buttons
        lo_hdr = QWidget()
        lo_hdr.setFixedHeight(28)
        lh = QHBoxLayout(lo_hdr)
        lh.setContentsMargins(4, 2, 4, 2)
        lh.setSpacing(4)

        def _lo_btn(text, w):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFixedWidth(w)
            return b

        self._lo_btn_I     = _lo_btn("I",     34)
        self._lo_btn_Isub  = _lo_btn("I_sub", 50)
        self._lo_btn_snip  = _lo_btn("SNIP",  46)
        self._lo_btn_sigma = _lo_btn("±σ",    32)
        self._lo_btn_I.setChecked(True)
        self._lo_btn_sigma.setChecked(True)
        self._lo_btn_sigma.setToolTip("Show / hide ±1σ Poisson error band")
        self._lo_btn_I.clicked.connect(
            lambda checked: self._toggle_lo_curve('I', checked))
        self._lo_btn_Isub.clicked.connect(
            lambda checked: self._toggle_lo_curve('I_sub', checked))
        self._lo_btn_snip.clicked.connect(
            lambda checked: self._toggle_lo_curve('SNIP', checked))
        self._lo_btn_sigma.clicked.connect(
            lambda checked: self._toggle_lo_sigma(checked))
        lh.addWidget(self._lo_btn_I)
        lh.addWidget(self._lo_btn_Isub)
        lh.addWidget(self._lo_btn_snip)
        lh.addWidget(self._lo_btn_sigma)

        # SNIP 2θ range spinboxes — synchronized with orange vlines on lineout
        lh.addSpacing(8)
        lh.addWidget(QLabel("SNIP 2\u03b8:"))
        self._spin_snip_lo = QDoubleSpinBox()
        self._spin_snip_lo.setRange(0.0, 180.0)
        self._spin_snip_lo.setDecimals(2)
        self._spin_snip_lo.setSingleStep(0.1)
        self._spin_snip_lo.setFixedWidth(62)
        self._spin_snip_lo.setToolTip(
            "Lower 2\u03b8 bound for SNIP background.\n"
            "Drag the orange dashed line on the lineout to adjust.")
        lh.addWidget(self._spin_snip_lo)
        lh.addWidget(QLabel("\u2013"))
        self._spin_snip_hi = QDoubleSpinBox()
        self._spin_snip_hi.setRange(0.0, 180.0)
        self._spin_snip_hi.setDecimals(2)
        self._spin_snip_hi.setSingleStep(0.1)
        self._spin_snip_hi.setFixedWidth(62)
        self._spin_snip_hi.setToolTip(self._spin_snip_lo.toolTip())
        lh.addWidget(self._spin_snip_hi)
        self._spin_snip_lo.valueChanged.connect(self._on_snip_spin_changed)
        self._spin_snip_hi.valueChanged.connect(self._on_snip_spin_changed)

        lh.addSpacing(12)
        self._lo_pos_label = QLabel("  2θ = ---")
        self._lo_pos_label.setStyleSheet(
            "color: #cccccc; font-size: 12px; font-family: monospace;")
        lh.addWidget(self._lo_pos_label)
        lh.addStretch()
        vl_outer.addWidget(lo_hdr, stretch=0)

        # Top row: canvas + ylim side panel
        top = QWidget()
        hl = QHBoxLayout(top)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        self.line_fig = Figure(facecolor=_BG_FIG)
        # Same left/width as img_ax ([0.06, ..., 0.92]) so 2θ axes align.
        self.line_ax = self.line_fig.add_axes([0.06, 0.18, 0.92, 0.74])
        _style_ax(self.line_ax)
        self.line_ax.set_xlabel('2\u03b8 (deg)', fontsize=8)
        self.line_ax.set_ylabel('Intensity', fontsize=8)
        self.line_ax.set_title('Lineout', fontsize=9)

        # Sync cursor line on lineout — green, initially invisible
        self._line_vline = self.line_ax.axvline(
            0, color=_XH, lw=0.8, ls='--', visible=False, alpha=0.85)

        self.line_canvas = FigureCanvasQTAgg(self.line_fig)
        self.line_canvas.mpl_connect('motion_notify_event',  self._on_line_motion)
        self.line_canvas.mpl_connect('axes_leave_event',     self._on_line_leave)
        self.line_canvas.mpl_connect('scroll_event',         self._on_line_scroll)
        self.line_canvas.mpl_connect('button_press_event',   self._on_line_press)
        self.line_canvas.mpl_connect('button_release_event', self._on_line_release)
        hl.addWidget(self.line_canvas, stretch=1)

        # Y-axis limit control — same 110 px width as the image HistClimWidget
        # so both canvases are identical physical width and the 2θ axes align.
        self._line_ylim = _LineYLimWidget(
            self._on_line_ylim_changed,
            auto_callback=self._on_ylim_auto,
            parent=self)
        hl.addWidget(self._line_ylim, stretch=0)

        vl_outer.addWidget(top, stretch=1)

        # Bottom: toolbar spanning the full section width
        self._line_toolbar = _NavBar(self.line_canvas, widget, coordinates=False)
        self._line_toolbar.setMaximumHeight(28)
        vl_outer.addWidget(self._line_toolbar, stretch=0)

        return widget

    # ── Sync bus ──

    def _connect_sync(self):
        self._sync_bus.tth_changed.connect(self._on_external_tth)
        self._sync_bus.clear_hover.connect(self._on_external_clear)

    def _emit_tth(self, tth):
        """Emit to bus without re-triggering our own receiver."""
        self._sync_bus.tth_changed.disconnect(self._on_external_tth)
        self._sync_bus.tth_changed.emit(tth)
        self._sync_bus.tth_changed.connect(self._on_external_tth)

    def _on_external_tth(self, tth):
        """Received from stack widget — draw cursor on lineout (and cake if active)."""
        self._set_lineout_cursor(tth, visible=True)
        if self._view == 'cake':
            self._img_vline.set_xdata([tth, tth])
            self._img_vline.set_visible(True)
            self._img_hline.set_visible(False)
            self.img_canvas.draw_idle()

    def _on_external_clear(self):
        self._set_lineout_cursor(0, visible=False)
        if self._view == 'cake':
            self._img_vline.set_visible(False)
            self._img_hline.set_visible(False)
            self.img_canvas.draw_idle()

    # ── Public API ──

    def set_luts(self, tth_lut, eta_lut):
        """Provide LUTs for TIFF pixel → (2th, eta) lookup."""
        self.tth_lut = tth_lut
        self.eta_lut = eta_lut

    def show_results(self, result, title=''):
        """Display reduce_frame() output dict directly (in-memory)."""
        self._tif_data = result.get('image')   # may be None if not stored
        self._cake_data = result.get('cake_img')
        self._tth_cake = result.get('tth_cake')
        self._eta_cake = result.get('eta_cake')
        tth    = result.get('tth')
        I      = result.get('I')
        bg     = result.get('bg')
        I_sub  = result.get('I_sub')
        sigma  = result.get('sigma')
        px_cnt = result.get('px_cnt')
        self._px_cnt      = px_cnt
        self._px_cnt_cake = result.get('px_cnt_cake')
        self._lineout = (tth, I, bg, I_sub, sigma, px_cnt) if tth is not None else None
        if title != self._lineout_title:   # only reset zoom when switching to a different file
            self._line_x_locked = False
        self._lineout_title = title
        # If this file is already checked as an overlay, keep the overlay
        # store in sync so _refresh_lineout always uses the freshest data.
        if title and title in self._lineout_store and self._lineout is not None:
            self._lineout_store[title] = self._lineout

        self._btn_tiff.setEnabled(self._tif_data is not None)
        self._btn_cake.setEnabled(self._cake_data is not None)
        # If current view is not available, switch
        if self._view == 'tiff' and self._tif_data is None:
            self._set_view('cake')
        elif self._view == 'cake' and self._cake_data is None:
            self._set_view('tiff')

        self._refresh_image(title)
        self._refresh_lineout()

    def show_pair(self, base, data_folder):
        """Load and display saved results from lineouts/ and cakes/ subfolders.

        Also tries to load the raw TIFF from the root folder.
        """
        data_folder = Path(data_folder)
        _lo_candidates = [
            data_folder / 'lineouts'       / f'{base}.xye',
            data_folder / 'lineouts_nomask' / f'{base}.xye',
            data_folder / 'lineouts'       / f'{base}.xy',
        ]
        lineout_path = next((p for p in _lo_candidates if p.is_file()), _lo_candidates[0])
        cake_path    = data_folder / 'cakes'    / f'{base}.tif'
        tif_path     = data_folder / f'{base}.tif'

        # Raw TIFF
        if tif_path.is_file():
            try:
                self._tif_data = tifffile.imread(str(tif_path)).astype(np.float32)
            except Exception:
                self._tif_data = None
        else:
            self._tif_data = None

        # Cake TIFF (saved transposed) + axes sidecar
        if cake_path.is_file():
            try:
                cake_T = tifffile.imread(str(cake_path)).astype(np.float32)
                self._cake_data = cake_T.T   # restore to (n_tth, n_eta)
                axes_path = data_folder / 'cakes' / f'{base}_axes.npz'
                if axes_path.is_file():
                    axes = np.load(str(axes_path))
                    self._tth_cake = axes['tth']
                    self._eta_cake = axes['eta']
                else:
                    self._tth_cake = None
                    self._eta_cake = None
            except Exception:
                self._cake_data = None
                self._tth_cake  = None
                self._eta_cake  = None
        else:
            self._cake_data = None

        # Lineout
        self._px_cnt      = None
        self._px_cnt_cake = None   # not saved to disk
        if lineout_path.is_file():
            try:
                data = np.loadtxt(str(lineout_path))
                if data.ndim == 2 and data.shape[1] >= 6:
                    # New I_sub format: 2θ  I  SNIP_bg  I_sub  σ  px_cnt
                    _px = data[:, 5].astype(int)
                    self._lineout = (data[:, 0], data[:, 1],
                                     data[:, 2], data[:, 3], data[:, 4], _px)
                    self._px_cnt  = _px
                elif data.ndim == 2 and data.shape[1] == 5:
                    # Legacy I_sub (no px_cnt): 2θ  I  SNIP_bg  I_sub  σ
                    self._lineout = (data[:, 0], data[:, 1],
                                     data[:, 2], data[:, 3], data[:, 4], None)
                elif data.ndim == 2 and data.shape[1] == 4:
                    # New I-export: 2θ  I  σ  px_cnt
                    _px = data[:, 3].astype(int)
                    self._lineout = (data[:, 0], data[:, 1],
                                     None, None, data[:, 2], _px)
                    self._px_cnt  = _px
                elif data.ndim == 2 and data.shape[1] >= 3:
                    # Legacy I-export: 2θ  I  σ
                    self._lineout = (data[:, 0], data[:, 1],
                                     None, None, data[:, 2], None)
                else:
                    self._lineout = None
            except Exception:
                self._lineout = None
        else:
            self._lineout = None

        if base != self._lineout_title:   # only reset zoom when switching to a different file
            self._line_x_locked = False
        self._lineout_title = base

        self._btn_tiff.setEnabled(self._tif_data is not None)
        self._btn_cake.setEnabled(self._cake_data is not None)
        if self._view == 'tiff' and self._tif_data is None:
            self._set_view('cake')
        elif self._view == 'cake' and self._cake_data is None:
            self._set_view('tiff')

        self._refresh_image(base)
        self._refresh_lineout()

    def show_ring_overlay(self, ring_tth_list):
        """Draw vertical lines on the lineout at expected ring positions."""
        col = self._theme_colors.get('ring', _LINE_RING)
        for tth in ring_tth_list:
            self.line_ax.axvline(tth, color=col, alpha=0.6, lw=0.8, ls='--')
        self.line_canvas.draw_idle()

    def _resample_cake_uniform_tth(self, cake_img, ext):
        """Resample a varbin cake to a uniform 2θ grid before display.

        Varbin bins are uniform in R-space → non-uniform in 2θ-space (bin width
        ∝ cos²(2θ)).  Displaying with imshow, which maps all columns to equal
        display width, visually displaces ring positions by up to ~1° over a
        25° range.

        This performs nearest-neighbour resampling along the 2θ axis using the
        stored tth_centres as the source coordinate map.  For unibin (uniform
        2θ) cakes the result is identical to the input.  For varbin cakes each
        output column maps to the source bin whose centre is closest to the
        uniform target 2θ.

        Parameters
        ----------
        cake_img : (n_tth, n_eta) float32 array
        ext      : [tth_l, tth_r, eta_b, eta_t] from _cake_extent()

        Returns
        -------
        (n_tth, n_eta) array ready for imshow(..., extent=ext)
        """
        if ext is None or self._tth_cake is None or len(self._tth_cake) < 2:
            return cake_img
        n_tth = cake_img.shape[0]
        tth_l, tth_r = ext[0], ext[1]
        # Uniform 2θ centres for the n_tth display columns
        tth_uniform = np.linspace(
            tth_l + 0.5 * (tth_r - tth_l) / n_tth,
            tth_r - 0.5 * (tth_r - tth_l) / n_tth,
            n_tth)
        # For each uniform 2θ, find the nearest source column by interpolation
        frac = np.interp(tth_uniform, self._tth_cake, np.arange(n_tth))
        col_idx = np.clip(np.round(frac).astype(int), 0, n_tth - 1)
        return cake_img[col_idx, :]

    def _cake_extent(self):
        """Return [tth_left, tth_right, eta_bot, eta_top] using bin outer edges.

        matplotlib imshow extent defines the outer boundary of the image, not
        pixel centres.  Using bin centres as the extent shifts every displayed
        column by half a bin width — the hover coordinate would read the bin
        centre plus a half-bin offset.  Extrapolating half the first/last bin
        width outward gives the true outer edges so that hover coordinates and
        lineout x-positions align exactly.

        Returns None when tth_cake or eta_cake is not available.
        """
        tc = self._tth_cake
        ec = self._eta_cake
        if tc is None or ec is None or len(tc) == 0 or len(ec) == 0:
            return None
        tth_l = tc[0]  - 0.5 * (tc[1]  - tc[0])  if len(tc) > 1 else float(tc[0])
        tth_r = tc[-1] + 0.5 * (tc[-1] - tc[-2]) if len(tc) > 1 else float(tc[-1])
        eta_b = ec[0]  - 0.5 * (ec[1]  - ec[0])  if len(ec) > 1 else float(ec[0])
        eta_t = ec[-1] + 0.5 * (ec[-1] - ec[-2]) if len(ec) > 1 else float(ec[-1])
        return [float(tth_l), float(tth_r), float(eta_b), float(eta_t)]

    def _compute_mask_rgba_cake(self):
        """Project the pixel-space mask into 2θ-η cake coordinates.

        Returns (rgba, extent) where rgba is an (n_eta, n_tth, 4) float32
        array suitable for imshow with origin='lower', or (None, None) if
        the required data is not available.
        """
        if (self._mask is None or self.tth_lut is None or self.eta_lut is None
                or self._cake_data is None
                or self._tth_cake is None or self._eta_cake is None):
            return None, None

        rows, cols = np.where(self._mask)
        if len(rows) == 0:
            return None, None

        tth_vals = self.tth_lut[rows, cols]
        eta_vals = self.eta_lut[rows, cols]
        valid = np.isfinite(tth_vals) & np.isfinite(eta_vals)
        tth_vals = tth_vals[valid]
        eta_vals = eta_vals[valid]
        if len(tth_vals) == 0:
            return None, None

        n_tth = self._cake_data.shape[0]
        n_eta = self._cake_data.shape[1]

        # Use the same outer-edge extent as _cake_extent() so the overlay
        # and the main image align exactly.
        tc = self._tth_cake
        ec = self._eta_cake
        tth_l = float(tc[0]  - 0.5*(tc[1]  - tc[0])  if len(tc) > 1 else tc[0])
        tth_r = float(tc[-1] + 0.5*(tc[-1] - tc[-2]) if len(tc) > 1 else tc[-1])
        eta_b = float(ec[0]  - 0.5*(ec[1]  - ec[0])  if len(ec) > 1 else ec[0])
        eta_t = float(ec[-1] + 0.5*(ec[-1] - ec[-2]) if len(ec) > 1 else ec[-1])

        # Floor-based bin index matches how _jit_precompute_bins_2d assigns pixels.
        ti = np.clip(
            np.floor((tth_vals - tth_l) / (tth_r - tth_l) * n_tth
                     ).astype(int), 0, n_tth - 1)
        ei = np.clip(
            np.floor((eta_vals - eta_b) / (eta_t - eta_b) * n_eta
                     ).astype(int), 0, n_eta - 1)

        cake_mask = np.zeros((n_tth, n_eta), dtype=bool)
        cake_mask[ti, ei] = True

        # cake is displayed as cake.T with origin='lower' → shape (n_eta, n_tth)
        rgba = np.zeros((n_eta, n_tth, 4), dtype=np.float32)
        rgba[..., 0] = 1.0
        rgba[..., 3] = 0.65 * cake_mask.T.astype(np.float32)

        return rgba, [tth_l, tth_r, eta_b, eta_t]

    def show_mask_overlay(self, mask, visible=True):
        """Draw semi-transparent red overlay over masked pixels (True=bad).

        Parameters
        ----------
        mask    : bool array (nrows, ncols) or None
        visible : if False, remove any existing overlay
        """
        self._mask = mask
        # Remove old overlay handle (ax.clear() would also do this, but we
        # manage it explicitly so _refresh_image can restore it)
        if self._msk_handle is not None:
            try:
                self._msk_handle.remove()
            except Exception:
                pass
            self._msk_handle = None
        if mask is None or not visible or self._img_handle is None:
            self.img_canvas.draw_idle()
            return
        if self._view == 'tiff':
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[..., 0] = 1.0
            rgba[..., 3] = 0.65 * mask.astype(np.float32)
            self._msk_handle = self.img_ax.imshow(
                rgba, origin='upper', interpolation='nearest', zorder=3)
        elif self._view == 'cake':
            rgba, ext = self._compute_mask_rgba_cake()
            if rgba is not None:
                self._msk_handle = self.img_ax.imshow(
                    rgba, origin='lower', aspect='auto', extent=ext,
                    interpolation='nearest', zorder=3)
        self.img_canvas.draw_idle()

    def set_eta_lines(self, lo, hi):
        """Set the eta sector range and draw/update horizontal lines on the cake."""
        self._eta_lo = float(lo)
        self._eta_hi = float(hi)
        self._draw_eta_lines()

    def _draw_eta_lines(self):
        """Draw (or remove) eta sector lines on the cake/counts image."""
        for h in (self._eta_lo_line, self._eta_hi_line):
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
        self._eta_lo_line = None
        self._eta_hi_line = None
        if self._view != 'cake' or self._img_handle is None:
            self.img_canvas.draw_idle()
            return
        kw = dict(color='#ff8800', lw=1.2, ls='--', alpha=0.9, zorder=5)
        self._eta_lo_line = self.img_ax.axhline(self._eta_lo, **kw)
        self._eta_hi_line = self.img_ax.axhline(self._eta_hi, **kw)
        self.img_canvas.draw_idle()

    def _eta_drag_tol(self):
        """Return eta tolerance (data coords) for snapping to a line — ~10 px."""
        try:
            ylim = self.img_ax.get_ylim()
            h_px = self.img_ax.get_window_extent().height
            return max(1.0, 10.0 * abs(ylim[1] - ylim[0]) / max(h_px, 1))
        except Exception:
            return 5.0

    # ── SNIP 2θ range vlines ─────────────────────────────────────────────────

    @property
    def tth_snip_lo(self):
        return self._tth_snip_lo

    @property
    def tth_snip_hi(self):
        return self._tth_snip_hi

    def set_tth_snip_range(self, lo, hi):
        """Set SNIP 2θ range from outside (e.g. load_geometry). Updates spinboxes."""
        self._tth_snip_lo = float(lo)
        self._tth_snip_hi = float(hi)
        self._spin_snip_lo.blockSignals(True)
        self._spin_snip_hi.blockSignals(True)
        self._spin_snip_lo.setValue(self._tth_snip_lo)
        self._spin_snip_hi.setValue(self._tth_snip_hi)
        self._spin_snip_lo.blockSignals(False)
        self._spin_snip_hi.blockSignals(False)
        self._draw_tth_snip_lines()
        if self._lineout is not None or self._lineout_store:
            self.line_canvas.draw_idle()

    def _draw_tth_snip_lines(self):
        """Recreate orange SNIP-range vlines on the lineout after ax.clear()."""
        for h in (self._tth_lo_vline, self._tth_hi_vline):
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
        kw = dict(color='#ff8800', lw=1.2, ls='--', alpha=0.9, zorder=5)
        self._tth_lo_vline = self.line_ax.axvline(self._tth_snip_lo, **kw)
        self._tth_hi_vline = self.line_ax.axvline(self._tth_snip_hi, **kw)

    def _tth_drag_tol(self):
        """Return 2θ tolerance (data coords) for snapping to a vline — ~10 px."""
        try:
            xlim = self.line_ax.get_xlim()
            w_px = self.line_ax.get_window_extent().width
            return max(0.1, 10.0 * abs(xlim[1] - xlim[0]) / max(w_px, 1))
        except Exception:
            return 1.0

    def _on_snip_spin_changed(self):
        lo = self._spin_snip_lo.value()
        hi = self._spin_snip_hi.value()
        if lo >= hi:
            return
        self._tth_snip_lo = lo
        self._tth_snip_hi = hi
        self._draw_tth_snip_lines()
        self.line_canvas.draw_idle()
        self.tth_snip_range_changed.emit(lo, hi)

    def clear(self):
        self._tif_data = None
        self._cake_data = None
        self._lineout = None
        self._px_cnt = None
        self._px_cnt_cake = None
        self._lineout_title = ''
        self._lineout_store.clear()
        self._img_handle = None
        self._mask = None
        self._msk_handle = None
        self.img_ax.clear()
        _style_ax(self.img_ax)
        self.img_ax.set_title('Image', fontsize=9)
        self.img_canvas.draw_idle()
        self._clear_lineout_plot()
        self._line_ylim.clear()

    # ── View toggle ──

    def _set_view(self, view):
        self._view = view
        self._btn_tiff.setChecked(view == 'tiff')
        self._btn_cake.setChecked(view == 'cake')
        title = self.img_ax.get_title()
        self._refresh_image(title)
        # Sync lineout x-axis when switching to cake view
        if view == 'cake':
            self._refresh_lineout()  # re-clamps lineout xlim to cake extent

    # ── Rendering ──

    def _refresh_image(self, title=''):
        c = self._theme_colors
        self.img_ax.clear()
        _style_ax(self.img_ax,
                  fg=c.get('fg', _FG), bg=c.get('ax', _BG_AX),
                  spine=c.get('spine', '#999999'))

        # Recreate crosshair lines after ax.clear()
        self._img_vline = self.img_ax.axvline(
            0, color=_XH, lw=0.8, ls='--', visible=False, alpha=0.85)
        self._img_hline = self.img_ax.axhline(
            0, color=_XH, lw=0.8, ls='--', visible=False, alpha=0.85)

        if self._view == 'tiff' and self._tif_data is not None:
            self._img_handle = self.img_ax.imshow(
                self._tif_data, cmap=self._cmap, aspect='equal',
                origin='upper', interpolation='nearest')
            self.img_ax.set_xlabel('Column (px)', fontsize=8)
            self.img_ax.set_ylabel('Row (px)', fontsize=8)
            if title:
                self.img_ax.set_title(title, fontsize=9)
            self._update_histogram(self._tif_data)

        elif self._view == 'cake' and self._cake_data is not None:
            ext = self._cake_extent()   # outer bin edges, not centres
            cake_display = self._resample_cake_uniform_tth(self._cake_data, ext)
            self._img_handle = self.img_ax.imshow(
                cake_display.T, cmap=self._cmap, aspect='auto',
                origin='lower', extent=ext, interpolation='nearest')
            self.img_ax.set_xlabel('2\u03b8 (deg)', fontsize=8)
            self.img_ax.set_ylabel('\u03b7 (deg)', fontsize=8)
            if title:
                self.img_ax.set_title(title, fontsize=9)
            self._update_histogram(self._cake_data)  # histogram uses raw (unsampled) data

        # Restore eta sector lines after ax.clear() (cake view only)
        self._eta_lo_line = None
        self._eta_hi_line = None
        if self._view == 'cake' and self._img_handle is not None:
            kw = dict(color='#ff8800', lw=1.2, ls='--', alpha=0.9, zorder=5)
            self._eta_lo_line = self.img_ax.axhline(self._eta_lo, **kw)
            self._eta_hi_line = self.img_ax.axhline(self._eta_hi, **kw)

        # Show placeholder title when no image was drawn
        if self._img_handle is None:
            label = {'tiff': 'Image', 'cake': 'Cake'}.get(self._view, 'Image')
            self.img_ax.set_title(f'No {label} data', fontsize=9)

        # Restore mask overlay (ax.clear() removed all artists)
        self._msk_handle = None
        if self._mask is not None and self._img_handle is not None:
            if self._view == 'tiff' and self._tif_data is not None:
                rgba = np.zeros((*self._mask.shape, 4), dtype=np.float32)
                rgba[..., 0] = 1.0
                rgba[..., 3] = 0.65 * self._mask.astype(np.float32)
                self._msk_handle = self.img_ax.imshow(
                    rgba, origin='upper', interpolation='nearest', zorder=3)
            elif self._view == 'cake' and self._cake_data is not None:
                rgba, ext = self._compute_mask_rgba_cake()
                if rgba is not None:
                    self._msk_handle = self.img_ax.imshow(
                        rgba, origin='lower', aspect='auto', extent=ext,
                        interpolation='nearest', zorder=3)

        # Reset zoom state for the new image
        self._img_home_xlim = tuple(self.img_ax.get_xlim())
        self._img_home_ylim = tuple(self.img_ax.get_ylim())
        self._img_zoom_stack.clear()
        self.img_canvas.draw_idle()

    def _refresh_lineout(self):
        c = self._theme_colors
        col_ax    = c.get('ax',        _BG_AX)
        col_fg    = c.get('fg',        _FG)
        col_spine = c.get('spine',     '#999999')
        col_I     = c.get('line_I',    _LINE_I)
        col_Isub  = c.get('line_Isub', _LINE_ISUB)
        col_snip  = c.get('line_snip', _LINE_SNIP)
        col_pos   = c.get('pos_label', '#111111')
        self._lo_pos_label.setStyleSheet(
            f"color: {col_pos}; font-size: 12px; font-family: monospace;")

        prev_xlim = tuple(self.line_ax.get_xlim()) if (self._line_home_xlim is not None) else None
        prev_ylim = tuple(self.line_ax.get_ylim()) if self._y_locked else None
        self.line_ax.clear()
        _style_ax(self.line_ax, fg=col_fg, bg=col_ax, spine=col_spine)

        # Recreate sync cursor after clear
        self._line_vline = self.line_ax.axvline(
            0, color=_XH, lw=0.8, ls='--', visible=False, alpha=0.85)

        self.line_ax.set_xlabel('2\u03b8 (deg)', fontsize=8)
        self.line_ax.set_ylabel('Intensity', fontsize=8)

        if self._lineout_store:
            # ── Multi-lineout overlay ──────────────────────────────────────
            # All checked files: I only, cycling colours, thin.
            # Current file (cake view): I/I_sub/SNIP per toggle buttons,
            # theme colours, slightly thicker so it stands out.
            cur = self._lineout_title   # basename of file shown in cake/TIFF
            entries = list(self._lineout_store.items())
            n = len(entries)
            cyc = 0
            for base, (tth, I, bg, I_sub, sigma, *_) in entries:
                if base == cur:
                    continue   # drawn separately below with toggle control
                color = _LINEOUT_COLORS[cyc % len(_LINEOUT_COLORS)]
                cyc += 1
                self.line_ax.plot(tth, I, lw=0.8, color=color,
                                  label=base, alpha=0.85)

            # Current-file data: prefer overlay store, fall back to _lineout
            cur_data = self._lineout_store.get(cur)
            if cur_data is not None:
                ct, cI, cbg, cI_sub, csig, *_ = cur_data
            elif self._lineout is not None:
                ct, cI, cbg, cI_sub, csig, *_ = self._lineout
            else:
                ct = None

            if ct is not None:
                lbl = cur if cur else 'current'
                if self._lo_show.get('I', True):
                    self.line_ax.plot(ct, cI, lw=1.3, color=col_I,
                                      label=f"I  [{lbl}]")
                    if self._lo_show_sigma and csig is not None:
                        self.line_ax.fill_between(
                            ct, cI - csig, cI + csig,
                            alpha=0.15, color=col_I, linewidth=0)
                if self._lo_show.get('I_sub', False) and cI_sub is not None:
                    self.line_ax.plot(ct, cI_sub, lw=1.1, color=col_Isub,
                                      label=f"I_sub  [{lbl}]")
                    if self._lo_show_sigma and csig is not None:
                        self.line_ax.fill_between(
                            ct, cI_sub - csig, cI_sub + csig,
                            alpha=0.15, color=col_Isub, linewidth=0)
                if self._lo_show.get('SNIP', False) and cbg is not None:
                    self.line_ax.plot(ct, cbg, lw=0.9, color=col_snip,
                                      ls='--', label=f"SNIP  [{lbl}]")

            title_str = f'Lineout ({n} files)' if n > 1 else entries[0][0]
            self.line_ax.set_title(title_str, fontsize=9)
            self.line_ax.legend(fontsize=6, loc='upper right',
                                facecolor=col_ax, labelcolor=col_fg,
                                edgecolor=col_spine)

        elif self._lineout is not None:
            # ── Single-result preview (no files checked) ──────────────────
            tth, I, bg, I_sub, sigma, *_ = self._lineout
            if self._lo_show.get('I', True):
                self.line_ax.plot(tth, I, lw=0.8, color=col_I, label='I')
                if self._lo_show_sigma and sigma is not None:
                    self.line_ax.fill_between(
                        tth, I - sigma, I + sigma,
                        alpha=0.20, color=col_I, linewidth=0)
            if self._lo_show.get('I_sub', False) and I_sub is not None:
                self.line_ax.plot(tth, I_sub, lw=0.8, color=col_Isub,
                                  label='I_sub')
                if self._lo_show_sigma and sigma is not None:
                    self.line_ax.fill_between(
                        tth, I_sub - sigma, I_sub + sigma,
                        alpha=0.20, color=col_Isub, linewidth=0)
            if self._lo_show.get('SNIP', False) and bg is not None:
                self.line_ax.plot(tth, bg, lw=0.7, color=col_snip,
                                  ls='--', label='SNIP bg')
            handles, _ = self.line_ax.get_legend_handles_labels()
            if handles:
                self.line_ax.legend(fontsize=7, loc='upper right',
                                    facecolor=col_ax, labelcolor=col_fg,
                                    edgecolor=col_spine)
            self.line_ax.set_title(
                self._lineout_title or 'Lineout', fontsize=9)

        else:
            self.line_ax.set_title('Lineout', fontsize=9)
            self._line_home_xlim = None
            self._line_home_ylim = None
            self._line_zoom_stack.clear()
            self._line_ylim.clear()
            self.line_canvas.draw_idle()
            return

        # Compute natural home xlim from full cake extent.
        if self._view == 'cake' and self._tth_cake is not None:
            cake_ext = self._cake_extent()
            if cake_ext is not None:
                self._line_home_xlim = (cake_ext[0], cake_ext[1])
            else:
                self._line_home_xlim = tuple(self.line_ax.get_xlim())
        else:
            self._line_home_xlim = tuple(self.line_ax.get_xlim())

        self._line_home_ylim = tuple(self.line_ax.get_ylim())
        # Do not clear zoom stack — preserves right-click undo history across redraws.
        self._draw_tth_snip_lines()

        # Apply xlim — priority: user x-zoom > current cake xlim (may be zoomed) > home.
        if self._line_x_locked and prev_xlim is not None:
            self.line_ax.set_xlim(*prev_xlim)
        elif self._view == 'cake' and self._tth_cake is not None:
            self.line_ax.set_xlim(*self.img_ax.get_xlim())
        else:
            self.line_ax.set_xlim(*self._line_home_xlim)

        # Apply ylim — priority: user y-lock > auto-scaled home.
        if self._y_locked and prev_ylim is not None:
            self.line_ax.set_ylim(*prev_ylim)
            self._line_ylim.set_limits(*prev_ylim)
        else:
            self._line_ylim.set_limits(*self._line_home_ylim)

        self.line_canvas.draw_idle()

    def _toggle_lo_curve(self, name, checked):
        """Show/hide one of the lineout curves (I / I_sub / SNIP)."""
        self._lo_show[name] = checked
        self._refresh_lineout()

    def _toggle_lo_sigma(self, checked):
        """Show/hide the ±1σ Poisson error band on all active curves."""
        self._lo_show_sigma = checked
        self._refresh_lineout()

    def _on_line_ylim_changed(self, vmin, vmax):
        """Called by _line_ylim spinboxes when the user edits a limit."""
        if vmin >= vmax:
            return
        self._y_locked = True
        self.line_ax.set_ylim(vmin, vmax)
        self.line_canvas.draw_idle()

    def _on_ylim_auto(self):
        """Called by the Auto button — unlocks Y range and restores auto-scaled limits."""
        self._y_locked = False
        if self._line_home_ylim is not None:
            self._line_ylim.set_limits(*self._line_home_ylim)
            self.line_ax.set_ylim(*self._line_home_ylim)
            self.line_canvas.draw_idle()

    def _clear_lineout_plot(self):
        c = self._theme_colors
        self.line_ax.clear()
        _style_ax(self.line_ax,
                  fg=c.get('fg', _FG), bg=c.get('ax', _BG_AX),
                  spine=c.get('spine', '#999999'))
        self._line_vline = self.line_ax.axvline(
            0, color=_XH, lw=0.8, ls='--', visible=False, alpha=0.85)
        self.line_ax.set_title('Lineout', fontsize=9)
        self.line_ax.set_xlabel('2\u03b8 (deg)', fontsize=8)
        self.line_ax.set_ylabel('Intensity', fontsize=8)
        self.line_canvas.draw_idle()

    # ── Lineout overlay management ────────────────────────────────────────────

    def add_lineout(self, base, tth, I, bg=None, I_sub=None, sigma=None, px_cnt=None):
        """Add or update a lineout in the overlay store, then redraw."""
        self._lineout_store[base] = (tth, I, bg, I_sub, sigma, px_cnt)
        self._refresh_lineout()

    def remove_lineout(self, base):
        """Remove a lineout from the overlay store, then redraw."""
        self._lineout_store.pop(base, None)
        self._refresh_lineout()

    def clear_lineouts(self):
        """Remove all lineouts from the overlay store and clear the preview."""
        self._lineout_store.clear()
        self._lineout = None
        self._lineout_title = ''
        self._refresh_lineout()

    # ── x-axis sync between cake image and lineout ────────────────────────────

    def _sync_lineout_xlim(self):
        """Copy cake x-axis range to lineout (called after cake zoom)."""
        if self._view != 'cake' or self._tth_cake is None:
            return
        xlim = self.img_ax.get_xlim()
        self.line_ax.set_xlim(xlim)
        self.line_canvas.draw_idle()

    def _sync_cake_xlim(self):
        """Copy lineout x-axis range to cake image (called after lineout zoom)."""
        if self._view != 'cake' or self._tth_cake is None:
            return
        xlim = self.line_ax.get_xlim()
        self.img_ax.set_xlim(xlim)
        self.img_canvas.draw_idle()

    # ── Histogram + contrast ──

    def _update_histogram(self, data):
        def _apply(vmin, vmax):
            if self._img_handle is not None and vmin < vmax:
                self._img_handle.set_clim(vmin, vmax)
                self.img_canvas.draw_idle()

        self._hist_clim.set_data(data, callback=_apply)
        # Apply initial clim from widget
        vmin, vmax = self._hist_clim.get_clim()
        if self._img_handle is not None and vmin < vmax:
            self._img_handle.set_clim(vmin, vmax)
            self.img_canvas.draw_idle()

    # ── Mouse events ──

    def _on_img_motion(self, event):
        if event.inaxes != self.img_ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Eta line drag — move the grabbed line, suppress everything else
        if self._drag_eta is not None:
            if y is not None:
                eta = float(y)
                if self._drag_eta == 'lo':
                    self._eta_lo = eta
                    if self._eta_lo_line is not None:
                        self._eta_lo_line.set_ydata([eta, eta])
                else:
                    self._eta_hi = eta
                    if self._eta_hi_line is not None:
                        self._eta_hi_line.set_ydata([eta, eta])
                self.img_canvas.draw_idle()
            return

        # ROI drag — draw rubber-band rectangle, suppress crosshair
        if self._img_dragging and self._img_roi_start is not None:
            x0, y0 = self._img_roi_start
            dx, dy = x - x0, y - y0
            if self._img_roi_patch is not None:
                self._img_roi_patch.remove()
            self._img_roi_patch = _MplRect(
                (x0, y0), dx, dy,
                lw=1, edgecolor=_XH, facecolor='none', ls='--', alpha=0.85)
            self.img_ax.add_patch(self._img_roi_patch)
            self.img_canvas.draw_idle()
            return

        # Update crosshair on image
        self._img_vline.set_xdata([x, x])
        self._img_hline.set_ydata([y, y])
        self._img_vline.set_visible(True)
        self._img_hline.set_visible(True)
        self.img_canvas.draw_idle()

        # Compute (2th, eta) and update position label + sync
        if self._view == 'tiff' and self.tth_lut is not None:
            row_i = int(round(y))
            col_i = int(round(x))
            nr, nc = self.tth_lut.shape
            if 0 <= row_i < nr and 0 <= col_i < nc:
                tth = float(self.tth_lut[row_i, col_i])
                eta = float(self.eta_lut[row_i, col_i])
                I_val = float(self._tif_data[row_i, col_i]) if self._tif_data is not None else None
                I_str = f"   I = {I_val:.0f}" if I_val is not None else ""
                self._pos_label.setText(
                    f"2\u03b8 = {tth:.3f} deg   \u03b7 = {eta:.2f} deg{I_str}")
                self._set_lineout_cursor(tth, visible=True)
                self._emit_tth(tth)
        elif self._view == 'cake':
            tth, eta = x, y
            I_str = ""
            cnt_str = ""
            if self._cake_data is not None and self._tth_cake is not None and self._eta_cake is not None:
                ti = int(np.clip(np.searchsorted(self._tth_cake, tth), 0, self._cake_data.shape[0] - 1))
                ei = int(np.clip(np.searchsorted(self._eta_cake, eta), 0, self._cake_data.shape[1] - 1))
                I_val = float(self._cake_data[ti, ei])
                if np.isfinite(I_val):
                    I_str = f"   I = {I_val:.0f}"
                if self._px_cnt_cake is not None:
                    cnt_str = f"   px_cnt = {int(self._px_cnt_cake[ti, ei])}"
            self._pos_label.setText(
                f"2\u03b8 = {tth:.3f} deg   \u03b7 = {eta:.2f} deg{I_str}{cnt_str}")
            self._set_lineout_cursor(tth, visible=True)
            self._emit_tth(tth)
        else:
            I_str = ""
            if self._tif_data is not None:
                row_i, col_i = int(round(y)), int(round(x))
                nr, nc = self._tif_data.shape[:2]
                if 0 <= row_i < nr and 0 <= col_i < nc:
                    I_str = f"   I = {self._tif_data[row_i, col_i]:.0f}"
            self._pos_label.setText(f"col = {x:.0f}   row = {y:.0f}{I_str}")

    def _on_img_leave(self, event):
        self._img_vline.set_visible(False)
        self._img_hline.set_visible(False)
        self.img_canvas.draw_idle()
        self._set_lineout_cursor(0, visible=False)
        self._pos_label.setText("2\u03b8 = ---   \u03b7 = ---")
        self._sync_bus.clear_hover.emit()

    def _on_line_motion(self, event):
        if event.inaxes != self.line_ax or event.xdata is None:
            return
        # SNIP vline drag — move grabbed line, suppress everything else
        if self._drag_tth is not None:
            tth = float(event.xdata)
            if self._drag_tth == 'lo':
                self._tth_snip_lo = tth
                if self._tth_lo_vline is not None:
                    self._tth_lo_vline.set_xdata([tth, tth])
            else:
                self._tth_snip_hi = tth
                if self._tth_hi_vline is not None:
                    self._tth_hi_vline.set_xdata([tth, tth])
            self.line_canvas.draw_idle()
            return
        # ROI drag — draw rubber-band, suppress cursor sync
        if self._line_dragging and self._line_roi_start is not None:
            self._on_line_motion_drag(event)
            return
        tth = event.xdata
        self._set_lineout_cursor(tth, visible=True)

        # Look up intensity from the current (cake-displayed) lineout
        I_str = ""
        data = (self._lineout_store.get(self._lineout_title)
                if self._lineout_store else None)
        if data is None:
            data = self._lineout
        if data is not None:
            t_arr, I_arr, bg_arr, I_sub_arr, sig_arr, *rest = data
            px_cnt_arr = rest[0] if rest else None
            if t_arr is not None and len(t_arr) > 0:
                idx = int(np.clip(np.searchsorted(t_arr, tth), 0, len(t_arr) - 1))
                if idx > 0 and abs(t_arr[idx - 1] - tth) < abs(t_arr[idx] - tth):
                    idx -= 1
                sig_str = (f"  \u03c3={sig_arr[idx]:.2f}"
                           if sig_arr is not None and 0 <= idx < len(sig_arr) else "")
                if self._lo_show.get('I', True) and I_arr is not None:
                    I_str = f"   I = {I_arr[idx]:.2f}{sig_str}"
                elif self._lo_show.get('I_sub', False) and I_sub_arr is not None:
                    I_str = f"   I_sub = {I_sub_arr[idx]:.2f}{sig_str}"
                elif self._lo_show.get('SNIP', False) and bg_arr is not None:
                    I_str = f"   SNIP = {bg_arr[idx]:.2f}"
                if px_cnt_arr is not None and 0 <= idx < len(px_cnt_arr):
                    I_str += f"   px_cnt = {int(px_cnt_arr[idx])}"

        self._lo_pos_label.setText(f"  2\u03b8 = {tth:.3f} deg{I_str}")
        self._emit_tth(tth)

        # Update cake vline if in cake view
        if self._view == 'cake':
            self._img_vline.set_xdata([tth, tth])
            self._img_vline.set_visible(True)
            self._img_hline.set_visible(False)
            self.img_canvas.draw_idle()

    def _on_line_leave(self, event):
        self._set_lineout_cursor(0, visible=False)
        self._lo_pos_label.setText("  2θ = ---")
        if self._view == 'cake':
            self._img_vline.set_visible(False)
            self._img_hline.set_visible(False)
            self.img_canvas.draw_idle()
        self._sync_bus.clear_hover.emit()

    def _set_lineout_cursor(self, tth, visible):
        self._line_vline.set_xdata([tth, tth])
        self._line_vline.set_visible(visible)
        self.line_canvas.draw_idle()

    # ── Image zoom ────────────────────────────────────────────────────────────

    def _img_reset_zoom(self):
        if self._img_home_xlim is not None:
            self.img_ax.set_xlim(self._img_home_xlim)
            self.img_ax.set_ylim(self._img_home_ylim)
        self._img_zoom_stack.clear()
        self.img_canvas.draw_idle()
        self._sync_lineout_xlim()

    def _on_img_scroll(self, event):
        if event.inaxes != self.img_ax or event.xdata is None:
            return
        factor = 1.35
        scale  = 1.0 / factor if event.button == 'up' else factor
        xc, yc = event.xdata, event.ydata
        xlim = list(self.img_ax.get_xlim())
        ylim = list(self.img_ax.get_ylim())
        new_x = [xc + (v - xc) * scale for v in xlim]
        new_y = [yc + (v - yc) * scale for v in ylim]
        # Clamp to home extent when zooming out
        if event.button == 'down' and self._img_home_xlim is not None:
            home_span = abs(self._img_home_xlim[1] - self._img_home_xlim[0])
            if abs(new_x[1] - new_x[0]) >= home_span:
                self._img_reset_zoom()
                return
        self._img_zoom_stack.append((xlim, ylim))
        self.img_ax.set_xlim(new_x)
        self.img_ax.set_ylim(new_y)
        self.img_canvas.draw_idle()
        self._sync_lineout_xlim()

    def _on_img_press(self, event):
        if event.inaxes != self.img_ax:
            return
        if event.button == 3:
            if self._img_zoom_stack:
                xlim, ylim = self._img_zoom_stack.pop()
                self.img_ax.set_xlim(xlim)
                self.img_ax.set_ylim(ylim)
                self.img_canvas.draw_idle()
                self._sync_lineout_xlim()
            else:
                self._img_reset_zoom()
            return
        if getattr(self.img_canvas, 'toolbar', None) and self.img_canvas.toolbar.mode:
            return
        if event.button == 1 and event.xdata is not None:
            # Check for eta line drag (cake view only, before ROI zoom)
            if self._view == 'cake' and event.ydata is not None:
                tol = self._eta_drag_tol()
                if abs(event.ydata - self._eta_lo) < tol:
                    self._drag_eta = 'lo'
                    return
                if abs(event.ydata - self._eta_hi) < tol:
                    self._drag_eta = 'hi'
                    return
            self._img_dragging     = True
            self._img_roi_start    = (event.xdata, event.ydata)
            self._img_roi_start_px = (event.x, event.y)

    def _on_img_release(self, event):
        if self._drag_eta is not None:
            self._drag_eta = None
            # Clamp and normalise: lo <= hi
            lo = min(self._eta_lo, self._eta_hi)
            hi = max(self._eta_lo, self._eta_hi)
            self._eta_lo, self._eta_hi = lo, hi
            if self._eta_lo_line is not None:
                self._eta_lo_line.set_ydata([lo, lo])
            if self._eta_hi_line is not None:
                self._eta_hi_line.set_ydata([hi, hi])
            self.img_canvas.draw_idle()
            self.eta_range_changed.emit(lo, hi)
            return
        if not self._img_dragging or self._img_roi_start is None:
            self._img_dragging = False
            return
        self._img_dragging = False
        # Clean up patch regardless
        if self._img_roi_patch is not None:
            try:
                self._img_roi_patch.remove()
            except Exception:
                pass
            self._img_roi_patch = None

        if event.inaxes != self.img_ax or event.xdata is None:
            self._img_roi_start = None
            self.img_canvas.draw_idle()
            return

        x0, y0 = self._img_roi_start
        self._img_roi_start = None
        dx = event.xdata - x0
        dy = event.ydata - y0
        px_start = getattr(self, '_img_roi_start_px', None)
        if px_start is not None:
            px_ok = (abs(event.x - px_start[0]) >= 5 and
                     abs(event.y - px_start[1]) >= 5)
        else:
            px_ok = abs(dx) > 0 and abs(dy) > 0
        if not px_ok:
            self.img_canvas.draw_idle()
            return

        self._img_zoom_stack.append(
            (list(self.img_ax.get_xlim()), list(self.img_ax.get_ylim())))
        x0n, x1n = sorted([x0, x0 + dx])
        y0n, y1n = sorted([y0, y0 + dy])
        self.img_ax.set_xlim(x0n, x1n)
        cur_ylim = self.img_ax.get_ylim()
        if cur_ylim[0] > cur_ylim[1]:   # inverted y (origin='upper')
            self.img_ax.set_ylim(y1n, y0n)
        else:                            # normal y (cake, origin='lower')
            self.img_ax.set_ylim(y0n, y1n)
        self.img_canvas.draw_idle()
        self._sync_lineout_xlim()

    # ── Lineout zoom ──────────────────────────────────────────────────────────

    def _line_reset_zoom(self):
        self._line_x_locked = False
        if self._line_home_xlim is not None:
            self.line_ax.set_xlim(self._line_home_xlim)
            self.line_ax.set_ylim(self._line_home_ylim)
        self._line_zoom_stack.clear()
        self.line_canvas.draw_idle()
        self._sync_cake_xlim()

    def _on_line_scroll(self, event):
        if event.inaxes != self.line_ax or event.xdata is None:
            return
        factor = 1.35
        scale  = 1.0 / factor if event.button == 'up' else factor
        xc, yc = event.xdata, event.ydata
        xlim = list(self.line_ax.get_xlim())
        ylim = list(self.line_ax.get_ylim())
        new_x = [xc + (v - xc) * scale for v in xlim]
        new_y = [yc + (v - yc) * scale for v in ylim]
        if event.button == 'down' and self._line_home_xlim is not None:
            home_span = abs(self._line_home_xlim[1] - self._line_home_xlim[0])
            if abs(new_x[1] - new_x[0]) >= home_span:
                self._line_reset_zoom()
                return
        self._line_zoom_stack.append((xlim, ylim))
        self._line_x_locked = True
        self.line_ax.set_xlim(new_x)
        self.line_ax.set_ylim(new_y)
        self.line_canvas.draw_idle()
        self._sync_cake_xlim()

    def _on_line_press(self, event):
        if event.inaxes != self.line_ax:
            return
        if event.button == 3:
            if self._line_zoom_stack:
                xlim, ylim = self._line_zoom_stack.pop()
                self.line_ax.set_xlim(xlim)
                self.line_ax.set_ylim(ylim)
                self.line_canvas.draw_idle()
                self._sync_cake_xlim()
            else:
                self._line_reset_zoom()
            return
        if getattr(self.line_canvas, 'toolbar', None) and self.line_canvas.toolbar.mode:
            return
        if event.button == 1 and event.xdata is not None:
            # Check for SNIP vline drag before ROI zoom
            tol = self._tth_drag_tol()
            if abs(event.xdata - self._tth_snip_lo) < tol:
                self._drag_tth = 'lo'
                return
            if abs(event.xdata - self._tth_snip_hi) < tol:
                self._drag_tth = 'hi'
                return
            self._line_dragging     = True
            self._line_roi_start    = (event.xdata, event.ydata)
            self._line_roi_start_px = (event.x, event.y)

    def _on_line_release(self, event):
        if self._drag_tth is not None:
            self._drag_tth = None
            lo = min(self._tth_snip_lo, self._tth_snip_hi)
            hi = max(self._tth_snip_lo, self._tth_snip_hi)
            self._tth_snip_lo, self._tth_snip_hi = lo, hi
            if self._tth_lo_vline is not None:
                self._tth_lo_vline.set_xdata([lo, lo])
            if self._tth_hi_vline is not None:
                self._tth_hi_vline.set_xdata([hi, hi])
            self.line_canvas.draw_idle()
            self._spin_snip_lo.blockSignals(True)
            self._spin_snip_hi.blockSignals(True)
            self._spin_snip_lo.setValue(lo)
            self._spin_snip_hi.setValue(hi)
            self._spin_snip_lo.blockSignals(False)
            self._spin_snip_hi.blockSignals(False)
            self.tth_snip_range_changed.emit(lo, hi)
            return
        if not self._line_dragging or self._line_roi_start is None:
            self._line_dragging = False
            return
        self._line_dragging = False
        if self._line_roi_patch is not None:
            try:
                self._line_roi_patch.remove()
            except Exception:
                pass
            self._line_roi_patch = None

        if event.inaxes != self.line_ax or event.xdata is None:
            self._line_roi_start = None
            self.line_canvas.draw_idle()
            return

        x0, y0 = self._line_roi_start
        self._line_roi_start = None
        dx = event.xdata - x0
        dy = event.ydata - y0
        px_start = getattr(self, '_line_roi_start_px', None)
        if px_start is not None:
            if abs(event.x - px_start[0]) < 5:
                self.line_canvas.draw_idle()
                return
        elif abs(dx) < 1e-6:
            self.line_canvas.draw_idle()
            return

        self._line_zoom_stack.append(
            (list(self.line_ax.get_xlim()), list(self.line_ax.get_ylim())))
        self._line_x_locked = True
        x0n, x1n = sorted([x0, x0 + dx])
        if abs(dy) > 1e-6:
            y0n, y1n = sorted([y0, y0 + dy])
            self.line_ax.set_ylim(y0n, y1n)
        self.line_ax.set_xlim(x0n, x1n)
        self.line_canvas.draw_idle()
        self._sync_cake_xlim()

    def _on_line_motion_drag(self, event):
        """Called from _on_line_motion when dragging — draws ROI rectangle."""
        if event.inaxes != self.line_ax or event.xdata is None:
            return
        x0, y0 = self._line_roi_start
        dx, dy = event.xdata - x0, event.ydata - y0
        if self._line_roi_patch is not None:
            self._line_roi_patch.remove()
        self._line_roi_patch = _MplRect(
            (x0, y0), dx, dy,
            lw=1, edgecolor=_XH, facecolor='none', ls='--', alpha=0.85)
        self.line_ax.add_patch(self._line_roi_patch)
        self.line_canvas.draw_idle()

    def apply_mpl_theme(self, colors):
        # Store theme for use in _refresh_lineout / _refresh_image / ring overlay
        self._theme_colors = dict(colors)

        # Update cmap and live image
        self._cmap = colors.get('cmap', 'gray_r')
        if self._img_handle is not None:
            self._img_handle.set_cmap(self._cmap)

        # Update pos_label text color
        pos_color = colors.get('pos_label', '#111111')
        self._pos_label.setStyleSheet(
            f"color: {pos_color}; font-size: 13px; font-family: monospace;")

        # Update matplotlib figure/axis colours
        fg     = colors.get('fg',    _FG)
        bg_ax  = colors.get('ax',    _BG_AX)
        bg_fig = colors.get('fig',   _BG_FIG)
        spine  = colors.get('spine', '#999999')
        for fig in (self.img_fig, self.line_fig):
            fig.set_facecolor(bg_fig)
        for ax in (self.img_ax, self.line_ax):
            ax.set_facecolor(bg_ax)
            ax.tick_params(colors=fg)
            for sp in ax.spines.values():
                sp.set_color(spine)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)

        # Propagate to side panels
        self._hist_clim.apply_mpl_theme(colors)
        self._line_ylim.apply_mpl_theme(colors)

        # Redraw lineout with new line/legend colours
        if self._lineout is not None or self._lineout_store:
            self._refresh_lineout()
        else:
            self.line_canvas.draw_idle()

        self.img_canvas.draw_idle()
        _refresh_navbar_icons(self._img_toolbar)
        _refresh_navbar_icons(self._line_toolbar)
