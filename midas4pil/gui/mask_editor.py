"""
Mask editor dialog.

Provides a standalone QDialog with a matplotlib canvas for interactive
mask editing.  Shared between Calibration and Integration tabs.

Tools
-----
Circle  : press + drag → circular region
Rect    : press + drag → rectangular region
Polygon : sequential clicks + double-click (or click on first vertex) to close

Modes
-----
Mask   : set pixels to True (bad)
Unmask : set pixels to False (good)

Actions
-------
Load, Save, Auto, Despeckle, Threshold, Reset to Auto, Undo, Redo
"""

import logging
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox,
    QDoubleSpinBox, QLabel, QDialogButtonBox, QFileDialog,
    QSizePolicy,
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle as _MplCircle, Rectangle as _MplRect

from ..io import load_mask, save_mask, auto_mask
from .hist_clim import HistClimWidget

log = logging.getLogger(__name__)

# Module-level defaults — overridden by colors dict passed from the main window.
_BG_FIG = '#eeeeee'
_BG_AX  = '#f8f8f8'
_FG     = '#333333'

_DEFAULT_BRUSH_PX = 10    # initial circle-brush radius in image pixels


def _style_ax(ax, fg=_FG, bg=_BG_AX, spine='#999999'):
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(spine)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)


class MaskEditorDialog(QDialog):
    """Interactive mask editor.

    Parameters
    ----------
    image : 2-D numpy array
        Diffraction image displayed as the background.
    mask  : bool array (same shape as image) or None
        Initial mask (True = bad pixel).  None → all-False mask.
    colors : dict or None
        Matplotlib theme colors dict from _MPL_THEMES (keys: fig, ax, fg,
        spine, cmap, hist_bar).  None → module-level defaults (system theme).
    title : str
        Dialog window title.
    parent : QWidget or None
    """

    def __init__(self, image, mask=None, panel_map=None, colors=None,
                 title="Mask Editor", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 700)

        # Resolve theme colors (fall back to system-like defaults)
        c = colors or {}
        self._bg_fig = c.get('fig',      _BG_FIG)
        self._bg_ax  = c.get('ax',       _BG_AX)
        self._fg     = c.get('fg',       _FG)
        self._spine  = c.get('spine',    '#999999')
        self._cmap   = c.get('cmap',     'gray_r')
        self._hist_bar = c.get('hist_bar', '#666666')

        self._image = np.asarray(image, dtype=np.float64)
        self._panel_map = panel_map
        nrows, ncols = self._image.shape
        self._nrows = nrows
        self._ncols = ncols

        if mask is not None:
            self._mask = np.asarray(mask, dtype=bool).copy()
        else:
            self._mask = np.zeros((nrows, ncols), dtype=bool)

        # Undo/redo stacks (snapshots of _mask)
        self._undo_stack = []
        self._redo_stack = []

        # Drawing tool state
        self._tool = 'circle'     # 'circle' | 'rect' | 'polygon'
        self._mode = 'mask'       # 'mask' | 'unmask'
        self._brush_size = _DEFAULT_BRUSH_PX
        self._drag_start    = None   # (col, row) at button press
        self._drag_start_px = None   # (canvas x, canvas y)
        self._poly_verts    = []     # [(col, row), ...] in progress
        self._preview_artists = []   # temporary artists
        self._img_handle  = None
        self._msk_handle  = None
        self._accepted = False

        # Zoom state
        self._home_xlim = None
        self._home_ylim = None

        self._build_ui()
        self._draw_image()
        self._update_overlay()

    # ── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # Row 1: toolbar
        root.addLayout(self._build_toolbar())

        # Row 2: matplotlib canvas (stretch)
        self.fig = Figure(facecolor=self._bg_fig)
        self.ax  = self.fig.add_axes([0.04, 0.04, 0.94, 0.93])
        _style_ax(self.ax, fg=self._fg, bg=self._bg_ax, spine=self._spine)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.mpl_connect('scroll_event',         self._on_scroll)
        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        # Canvas + histogram side-panel (same layout as main display)
        self._hist_clim = HistClimWidget(width=100, parent=self)
        self._hist_clim.apply_mpl_theme({
            'fig': self._bg_fig, 'ax': self._bg_ax, 'fg': self._fg,
            'spine': self._spine, 'hist_bar': self._hist_bar,
        })
        img_row = QHBoxLayout()
        img_row.setContentsMargins(0, 0, 0, 0)
        img_row.setSpacing(0)
        img_row.addWidget(self.canvas, stretch=1)
        img_row.addWidget(self._hist_clim, stretch=0)
        root.addLayout(img_row, stretch=1)

        # Row 3: action buttons
        root.addLayout(self._build_actions())

        # Row 4: OK / Cancel
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    def _build_toolbar(self):
        hl = QHBoxLayout()
        hl.setSpacing(4)

        # Mode buttons
        self._btn_mask_mode   = QPushButton("Mask")
        self._btn_unmask_mode = QPushButton("Unmask")
        for b in (self._btn_mask_mode, self._btn_unmask_mode):
            b.setCheckable(True)
            b.setFixedWidth(68)
        self._btn_mask_mode.setChecked(True)
        self._btn_mask_mode.setToolTip(
            "Mask mode — drawing tools mark pixels as BAD (red overlay).\n"
            "Switch to Unmask mode to erase bad-pixel marks.")
        self._btn_unmask_mode.setToolTip(
            "Unmask mode — drawing tools clear bad-pixel marks.\n"
            "Switch to Mask mode to add new bad-pixel marks.")
        self._btn_mask_mode.clicked.connect(lambda: self._set_mode('mask'))
        self._btn_unmask_mode.clicked.connect(lambda: self._set_mode('unmask'))

        # Tool buttons
        self._btn_circle  = QPushButton("● Circle")
        self._btn_rect    = QPushButton("▭ Rect")
        self._btn_polygon = QPushButton("⬡ Poly")
        for b in (self._btn_circle, self._btn_rect, self._btn_polygon):
            b.setCheckable(True)
            b.setFixedWidth(72)
        self._btn_circle.setChecked(True)
        self._btn_circle.setToolTip(
            "Circle tool\n"
            "  Click          — mask/unmask a circle of radius = Size at that point\n"
            "  Drag           — draw an arbitrary-radius circle (center → edge)\n"
            "  Drag to edge   — circle is clipped at the image boundary\n"
            "  Click again    — deactivate Circle; left-drag now zooms (rubber-band)\n"
            "\n"
            "Zoom mode (no tool active):\n"
            "  Left-drag      — draw a zoom box and release to zoom in\n"
            "  Right-click    — reset to full view\n"
            "  Scroll wheel   — zoom in / out centred on cursor")
        self._btn_rect.setToolTip(
            "Rectangle tool\n"
            "  Click          — mask/unmask a single pixel at that point\n"
            "  Drag           — draw a rectangular region (corner to corner)\n"
            "  Drag to edge   — snaps to the image boundary if released outside\n"
            "  Click again    — deactivate Rect; left-drag now zooms (rubber-band)\n"
            "\n"
            "Zoom mode (no tool active):\n"
            "  Left-drag      — draw a zoom box and release to zoom in\n"
            "  Right-click    — reset to full view\n"
            "  Scroll wheel   — zoom in / out centred on cursor")
        self._btn_polygon.setToolTip(
            "Polygon tool\n"
            "  Click          — add a vertex\n"
            "  Double-click   — close the polygon and apply\n"
            "  Click on first vertex (within 8 px) — close and apply\n"
            "  Click outside image — vertex snaps to the nearest image border\n"
            "  Right-click    — cancel the polygon in progress\n"
            "\n"
            "Scroll wheel zooms; right-click resets zoom when no polygon is in progress.")
        # Circle and Rect toggle off when clicked while already active,
        # putting the editor in zoom-drag mode (no active draw tool).
        self._btn_circle.clicked.connect(lambda: self._toggle_tool('circle'))
        self._btn_rect.clicked.connect(lambda: self._toggle_tool('rect'))
        self._btn_polygon.clicked.connect(lambda: self._set_tool('polygon'))

        # Brush size
        self._spn_size = QSpinBox()
        self._spn_size.setRange(1, 500)
        self._spn_size.setValue(self._brush_size)
        self._spn_size.setFixedWidth(60)
        self._spn_size.setToolTip(
            "Brush size — radius in pixels used when the Circle tool is\n"
            "clicked (not dragged).  Has no effect on drag-drawn circles,\n"
            "rectangles, or polygons.")
        self._spn_size.valueChanged.connect(self._on_size_changed)

        # Undo / Redo
        self._btn_undo = QPushButton("← Undo")
        self._btn_redo = QPushButton("→ Redo")
        for b in (self._btn_undo, self._btn_redo):
            b.setFixedWidth(72)
        self._btn_undo.setToolTip("Undo last mask change  (up to 30 steps)")
        self._btn_redo.setToolTip("Redo last undone change")
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_redo.clicked.connect(self._on_redo)

        sep = QLabel("|")
        sep.setStyleSheet("color: #555555;")
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #555555;")
        sep3 = QLabel("|")
        sep3.setStyleSheet("color: #555555;")

        hl.addWidget(self._btn_mask_mode)
        hl.addWidget(self._btn_unmask_mode)
        hl.addWidget(sep)
        hl.addWidget(self._btn_circle)
        hl.addWidget(self._btn_rect)
        hl.addWidget(self._btn_polygon)
        hl.addWidget(sep2)
        hl.addWidget(QLabel("Size:"))
        hl.addWidget(self._spn_size)
        hl.addWidget(sep3)
        hl.addWidget(self._btn_undo)
        hl.addWidget(self._btn_redo)
        hl.addStretch()
        self._update_undo_buttons()
        return hl

    def _build_actions(self):
        hl = QHBoxLayout()
        hl.setSpacing(4)

        btn_load    = QPushButton("Load…")
        btn_save    = QPushButton("Save…")
        btn_auto    = QPushButton("Auto")
        btn_desp    = QPushButton("Despeckle")
        btn_reverse = QPushButton("Reverse")
        btn_reset   = QPushButton("Reset to Auto")
        btn_load.setToolTip(
            "Load a mask from a TIFF file (uint8: 1 = bad, 0 = good).\n"
            "The file must match the current image dimensions.")
        btn_save.setToolTip(
            "Save the current mask to a TIFF file (uint8: 1 = bad, 0 = good).")
        btn_auto.setToolTip(
            "Auto-generate mask from the image:\n"
            "  • Detector panel gaps (zero-count stripes)\n"
            "  • Saturated / overflow pixels\n"
            "  • Isolated hot pixels\n"
            "Replaces the current mask (undoable).")
        btn_desp.setToolTip(
            "Despeckle — remove isolated bad pixels that have no\n"
            "4-connected masked neighbours.  Useful after Auto mask\n"
            "to clean up spurious single-pixel flags.\n"
            "Only modifies if something actually changes (undoable).")
        btn_reverse.setToolTip(
            "Reverse — invert the mask (good ↔ bad for every pixel).\n"
            "Useful when a loaded mask has the opposite convention (undoable).")
        btn_reset.setToolTip(
            "Reset the mask to the auto-generated mask (same as clicking\n"
            "Auto from a clean state).  Discards all manual edits (undoable).")
        btn_load.clicked.connect(self._on_load)
        btn_save.clicked.connect(self._on_save)
        btn_auto.clicked.connect(self._on_auto)
        btn_desp.clicked.connect(self._on_despeckle)
        btn_reverse.clicked.connect(self._on_reverse)
        btn_reset.clicked.connect(self._on_reset_to_auto)

        sep = QLabel("|")
        sep.setStyleSheet("color: #555555;")

        self._spn_thresh = QDoubleSpinBox()
        self._spn_thresh.setRange(-1e12, 1e12)
        self._spn_thresh.setDecimals(1)
        self._spn_thresh.setValue(0.0)
        self._spn_thresh.setFixedWidth(90)
        self._spn_thresh.setToolTip("Threshold value used by Mask ≥ and Mask ≤")
        btn_above = QPushButton("Mask \u2265")
        btn_above.setToolTip(
            "Mask all pixels with intensity ≥ threshold value.\n"
            "Useful for masking saturated pixels or hot pixels with\n"
            "counts above a known overflow level.")
        btn_above.clicked.connect(self._on_threshold_above)
        btn_below = QPushButton("Mask \u2264")
        btn_below.setToolTip(
            "Mask all pixels with intensity ≤ threshold value.\n"
            "Useful for masking dead pixels (zero or negative counts)\n"
            "that were not caught by the Auto mask.")
        btn_below.clicked.connect(self._on_threshold_below)

        hl.addWidget(btn_load)
        hl.addWidget(btn_save)
        hl.addWidget(btn_auto)
        hl.addWidget(btn_desp)
        hl.addWidget(btn_reverse)
        hl.addWidget(sep)
        hl.addWidget(self._spn_thresh)
        hl.addWidget(btn_above)
        hl.addWidget(btn_below)
        hl.addStretch()
        hl.addWidget(btn_reset)
        return hl

    # ── Image + overlay ────────────────────────────────────────────────────

    def _draw_image(self):
        self.ax.clear()
        _style_ax(self.ax, fg=self._fg, bg=self._bg_ax, spine=self._spine)
        # Initial clim: use same histogram approach as main display
        self._img_handle = self.ax.imshow(
            self._image, cmap=self._cmap, origin='upper',
            interpolation='nearest', zorder=1)
        self._hist_clim.set_data(self._image, callback=self._on_contrast_changed)
        _vmin, _vmax = self._hist_clim.get_clim()
        if _vmin < _vmax:
            self._img_handle.set_clim(_vmin, _vmax)
        self._msk_handle = None
        self._home_xlim = tuple(self.ax.get_xlim())
        self._home_ylim = tuple(self.ax.get_ylim())

    def _update_overlay(self):
        """Redraw mask overlay (RGBA red, alpha=0.65 where masked)."""
        rgba = np.zeros((self._nrows, self._ncols, 4), dtype=np.float32)
        rgba[..., 0] = 1.0
        rgba[..., 3] = 0.65 * self._mask.astype(np.float32)

        if self._msk_handle is None:
            self._msk_handle = self.ax.imshow(
                rgba, origin='upper', interpolation='nearest', zorder=2)
        else:
            self._msk_handle.set_data(rgba)
        self.canvas.draw_idle()
        self._update_undo_buttons()

    def _clear_preview(self):
        for a in self._preview_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._preview_artists.clear()

    # ── Mode / tool setters ────────────────────────────────────────────────

    def _set_mode(self, mode):
        self._mode = mode
        self._btn_mask_mode.setChecked(mode == 'mask')
        self._btn_unmask_mode.setChecked(mode == 'unmask')
        # Cancel any in-progress polygon
        self._poly_verts.clear()
        self._clear_preview()
        self.canvas.draw_idle()

    def _set_tool(self, tool):
        """Set the active draw tool.  Pass None to enter zoom-drag mode."""
        self._tool = tool
        self._btn_circle.setChecked(tool == 'circle')
        self._btn_rect.setChecked(tool == 'rect')
        self._btn_polygon.setChecked(tool == 'polygon')
        # Cancel any in-progress polygon
        self._poly_verts.clear()
        self._clear_preview()
        self.canvas.draw_idle()

    def _toggle_tool(self, tool):
        """Toggle a draw tool on/off.  Turning off enters zoom-drag mode."""
        self._set_tool(None if self._tool == tool else tool)

    def _on_size_changed(self, value):
        self._brush_size = value

    # ── Undo / redo ────────────────────────────────────────────────────────

    def _push_undo(self):
        self._undo_stack.append(self._mask.copy())
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _on_undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._mask.copy())
        self._mask = self._undo_stack.pop()
        self._update_overlay()

    def _on_redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._mask.copy())
        self._mask = self._redo_stack.pop()
        self._update_overlay()

    def _update_undo_buttons(self):
        self._btn_undo.setEnabled(bool(self._undo_stack))
        self._btn_redo.setEnabled(bool(self._redo_stack))

    # ── Shape application ──────────────────────────────────────────────────

    def _apply_shape(self, kind, *args):
        self._push_undo()
        val = (self._mode == 'mask')
        nrows, ncols = self._nrows, self._ncols

        if kind == 'circle':
            cx, cy, r = args   # cx=col, cy=row
            Y, X = np.ogrid[:nrows, :ncols]
            self._mask[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = val

        elif kind == 'rect':
            c0, r0, c1, r1 = args
            c0, c1 = sorted([int(round(c0)), int(round(c1))])
            r0, r1 = sorted([int(round(r0)), int(round(r1))])
            c0 = max(0, c0); c1 = min(ncols - 1, c1)
            r0 = max(0, r0); r1 = min(nrows - 1, r1)
            self._mask[r0:r1 + 1, c0:c1 + 1] = val

        elif kind == 'polygon':
            verts = args[0]  # list of (col, row)
            if len(verts) < 3:
                return
            from matplotlib.path import Path
            mpath = Path([(c, r) for c, r in verts])
            cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
            pts = np.column_stack([cols.ravel(), rows.ravel()])
            inside = mpath.contains_points(pts).reshape(nrows, ncols)
            self._mask[inside] = val

        self._update_overlay()

    # ── Mouse event handlers ───────────────────────────────────────────────

    def _canvas_dist(self, x0, y0, x1, y1):
        """Euclidean distance in canvas (device) pixels."""
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def _on_press(self, event):
        # Handle clicks outside the axes
        if event.inaxes != self.ax:
            if event.button == 3:
                if self._poly_verts:
                    self._poly_verts.clear()
                    self._clear_preview()
                    self.canvas.draw_idle()
                else:
                    self._reset_zoom()
                return
            if self._tool == 'polygon' and event.button == 1 and not event.dblclick:
                # Snap vertex to image boundary
                xy = self.ax.transData.inverted().transform(
                    [[event.x, event.y]])[0]
                col = float(np.clip(xy[0], -0.5, self._ncols - 0.5))
                row = float(np.clip(xy[1], -0.5, self._nrows - 0.5))
                self._poly_verts.append((col, row))
                self._draw_poly_preview()
            return

        # Right-click inside axes: cancel polygon or reset zoom
        if event.button == 3:
            if self._poly_verts:
                self._poly_verts.clear()
                self._clear_preview()
                self.canvas.draw_idle()
            else:
                self._reset_zoom()
            return

        if event.button != 1:
            return

        col, row = event.xdata, event.ydata
        if col is None or row is None:
            return

        # Convert to canvas pixels for drag-vs-click detection
        xy_px = self.ax.transData.transform([[col, row]])[0]
        self._drag_start    = (col, row)
        self._drag_start_px = tuple(xy_px)

        if self._tool == 'polygon':
            # Check for double-click (close polygon)
            if event.dblclick and len(self._poly_verts) >= 2:
                self._poly_verts.append(self._poly_verts[0])  # close
                verts = self._poly_verts[:]
                self._poly_verts.clear()
                self._clear_preview()
                self._apply_shape('polygon', verts)
                return
            # Check if clicking near the first vertex (close polygon)
            if len(self._poly_verts) >= 3:
                fc, fr = self._poly_verts[0]
                fp = self.ax.transData.transform([[fc, fr]])[0]
                if self._canvas_dist(*fp, *xy_px) < 8:
                    verts = self._poly_verts[:]
                    self._poly_verts.clear()
                    self._clear_preview()
                    self._apply_shape('polygon', verts)
                    return
            self._poly_verts.append((col, row))
            self._draw_poly_preview()

    def _on_motion(self, event):
        if event.button != 1:
            return
        if self._drag_start is None:
            return

        if event.inaxes != self.ax:
            return
        if self._tool == 'polygon':
            return

        col, row = event.xdata, event.ydata
        if col is None or row is None:
            return

        # Check drag threshold (5 canvas pixels)
        xy_px = self.ax.transData.transform([[col, row]])[0]
        if self._canvas_dist(*self._drag_start_px, *xy_px) < 5:
            return

        self._clear_preview()

        if self._tool is None:
            # Zoom-drag mode: show rubber-band rectangle
            c0, r0 = self._drag_start
            patch = _MplRect(
                (c0, r0), col - c0, row - r0,
                lw=1, edgecolor='white', facecolor='none', ls='-',
                alpha=0.7, zorder=4)
            self.ax.add_patch(patch)
            self._preview_artists.append(patch)

        elif self._tool == 'circle':
            c0, r0 = self._drag_start
            radius = ((col - c0) ** 2 + (row - r0) ** 2) ** 0.5
            patch = _MplCircle(
                (c0, r0), radius,
                lw=1, edgecolor='#ff5252', facecolor='none', ls='--',
                alpha=0.85, zorder=4)
            self.ax.add_patch(patch)
            self._preview_artists.append(patch)

        elif self._tool == 'rect':
            c0, r0 = self._drag_start
            dc, dr = col - c0, row - r0
            patch = _MplRect(
                (c0, r0), dc, dr,
                lw=1, edgecolor='#ff5252', facecolor='none', ls='--',
                alpha=0.85, zorder=4)
            self.ax.add_patch(patch)
            self._preview_artists.append(patch)

        self.canvas.draw_idle()

    def _on_release(self, event):
        if event.button != 1:
            return
        if self._drag_start is None or self._tool == 'polygon':
            self._drag_start = None
            self._drag_start_px = None
            return

        col, row = event.xdata, event.ydata
        drag_dist = self._canvas_dist(*self._drag_start_px, event.x, event.y)

        # Zoom-drag mode: rubber-band rectangle zooms the view
        if self._tool is None:
            self._clear_preview()
            if drag_dist > 5 and col is not None and row is not None:
                c0, r0 = self._drag_start
                c_min, c_max = sorted([c0, col])
                r_min, r_max = sorted([r0, row])
                if c_max > c_min and r_max > r_min:
                    self.ax.set_xlim(c_min, c_max)
                    # imshow origin='upper': larger row = lower on screen,
                    # so ylim must be (r_max, r_min) to keep the inversion.
                    self.ax.set_ylim(r_max, r_min)
                    self.canvas.draw_idle()
            self._drag_start    = None
            self._drag_start_px = None
            return

        if col is None or row is None:
            # Snap release point to the image boundary
            xy = self.ax.transData.inverted().transform([[event.x, event.y]])[0]
            col = float(np.clip(xy[0], -0.5, self._ncols - 0.5))
            row = float(np.clip(xy[1], -0.5, self._nrows - 0.5))

        c0, r0 = self._drag_start

        self._clear_preview()

        if drag_dist < 5:
            # Treat as click: circle of brush_size at click point
            if self._tool == 'circle':
                self._apply_shape('circle', c0, r0, self._brush_size)
            # rect click: single pixel
            elif self._tool == 'rect':
                self._apply_shape('rect', c0, r0, c0, r0)
        else:
            if self._tool == 'circle':
                radius = ((col - c0) ** 2 + (row - r0) ** 2) ** 0.5
                self._apply_shape('circle', c0, r0, radius)
            elif self._tool == 'rect':
                self._apply_shape('rect', c0, r0, col, row)

        self._drag_start    = None
        self._drag_start_px = None
        self.canvas.draw_idle()

    def _draw_poly_preview(self):
        self._clear_preview()
        if not self._poly_verts:
            return
        cols = [v[0] for v in self._poly_verts]
        rows = [v[1] for v in self._poly_verts]
        lns, = self.ax.plot(cols, rows, color='#ff5252', lw=1, ls='--',
                            marker='o', markersize=4, zorder=4)
        self._preview_artists.append(lns)
        self.canvas.draw_idle()

    # ── Scroll wheel zoom ──────────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        factor = 1.35
        scale  = 1.0 / factor if event.button == 'up' else factor
        xc, yc = event.xdata, event.ydata
        xlim = list(self.ax.get_xlim())
        ylim = list(self.ax.get_ylim())
        new_x = [xc + (v - xc) * scale for v in xlim]
        new_y = [yc + (v - yc) * scale for v in ylim]
        self.ax.set_xlim(new_x)
        self.ax.set_ylim(new_y)
        self.canvas.draw_idle()

    def _reset_zoom(self):
        if self._home_xlim is not None:
            self.ax.set_xlim(self._home_xlim)
            self.ax.set_ylim(self._home_ylim)
            self.canvas.draw_idle()

    # ── Action handlers ────────────────────────────────────────────────────

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Mask", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
        if not path:
            return
        try:
            new_mask = load_mask(path)
            if new_mask.shape != self._mask.shape:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "Shape Mismatch",
                    f"Mask shape {new_mask.shape} != image shape {self._mask.shape}")
                return
            self._push_undo()
            self._mask = new_mask.copy()
            self._update_overlay()
        except Exception as e:
            log.error("Mask load failed: %s", e, exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load Error", str(e))

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", "mask.tif",
            "TIFF Files (*.tif *.tiff);;All Files (*)")
        if not path:
            return
        try:
            save_mask(self._mask, path)
        except Exception as e:
            log.error("Mask save failed: %s", e, exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_auto(self):
        new_mask = auto_mask(self._image, panel_map=self._panel_map)
        self._push_undo()
        self._mask = new_mask.copy()
        self._update_overlay()

    def _on_despeckle(self):
        """Remove isolated masked pixels with no 4-connected masked neighbours."""
        m = self._mask
        has_neighbour = np.zeros_like(m)
        has_neighbour[1:,  :] |= m[:-1, :]   # pixel above
        has_neighbour[:-1, :] |= m[1:,  :]   # pixel below
        has_neighbour[:,  1:] |= m[:, :-1]   # pixel left
        has_neighbour[:, :-1] |= m[:, 1: ]   # pixel right
        new_mask = self._mask & has_neighbour
        if not np.array_equal(new_mask, self._mask):
            self._push_undo()
            self._mask = new_mask
            self._update_overlay()

    def _on_reverse(self):
        """Invert the mask — good pixels become bad and vice versa."""
        self._push_undo()
        self._mask = ~self._mask
        self._update_overlay()

    def _on_threshold_above(self):
        val = self._spn_thresh.value()
        self._push_undo()
        self._mask |= (self._image >= val)
        self._update_overlay()

    def _on_threshold_below(self):
        val = self._spn_thresh.value()
        self._push_undo()
        self._mask |= (self._image <= val)
        self._update_overlay()

    def _on_contrast_changed(self, vmin, vmax):
        """Called by HistClimWidget when the user drags a contrast line."""
        if self._img_handle is not None and vmin < vmax:
            self._img_handle.set_clim(vmin, vmax)
            self.canvas.draw_idle()

    def _on_reset_to_auto(self):
        self._push_undo()
        self._mask = auto_mask(self._image, panel_map=self._panel_map).copy()
        self._update_overlay()

    # ── Accept / result ────────────────────────────────────────────────────

    def _on_accept(self):
        self._accepted = True
        self.accept()

    def result_mask(self):
        """Return the edited mask, or None if dialog was cancelled."""
        return self._mask if self._accepted else None
