# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Histogram contrast widget — horizontal profile, intensity on the Y-axis.

The Y-axis (tall/vertical) is intensity.  The X-axis (narrow/horizontal)
is the count histogram drawn as a filled step curve.  Two horizontal lines
mark vmin (green) and vmax (amber).  A shaded horizontal span between them
can be grabbed to shift both limits simultaneously.

Histogram bins are computed over the p0.1–p99.9 percentile range of the
data so that all bins fall inside the interesting intensity region, making
the profile fully visible regardless of detector bit depth.

The initial Y view is further auto-zoomed to the p1–p99 range.
Scroll wheel zooms the Y axis; right-click resets to the auto view.

Interaction
-----------
Scroll wheel (on histogram)     → zoom Y-axis in/out centred on cursor
Right-click (on histogram)      → reset Y-axis to auto view (p1–p99)
Click near green line (vmin)    → drag vmin up/down only
Click near amber line (vmax)    → drag vmax up/down only
Click between the two lines     → drag both (shift the window, span preserved)
Fires callback(vmin, vmax) on every update.
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib import ticker as _ticker
from PySide6.QtWidgets import QWidget, QVBoxLayout

_XH     = '#00e676'    # vmin line — green
_XH2    = '#ff9800'    # vmax line — amber
_FG     = '#cccccc'
_BG_FIG = '#1a1a1a'
_BG_AX  = '#2d2d2d'
_BAR    = '#888888'

_ZOOM_FACTOR = 1.35    # per scroll step
_N_BINS      = 512     # histogram resolution


def _style_ax(ax, fg=_FG, bg=_BG_AX, spine='#444444'):
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=7, length=3, width=0.6)
    for sp in ax.spines.values():
        sp.set_color(spine)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)


class HistClimWidget(QWidget):
    """
    Narrow side panel: intensity histogram with two draggable contrast lines.

    Layout: Y-axis = intensity (tall).
            X-axis = count (narrow, no numerical labels — relative only).
    Histogram is drawn as a filled step curve with an outline.
    vmin and vmax are horizontal lines that move up/down.
    The shaded span between them can be grabbed to shift both together.

    Parameters
    ----------
    width : int
        Fixed width of the widget in pixels.
    """

    def __init__(self, width=110, parent=None):
        super().__init__(parent)
        self.setFixedWidth(width)

        self._callback    = None
        self._vmin        = 0.0
        self._vmax        = 1.0
        self._dmin        = 0.0
        self._dmax        = 1.0
        self._view_lo     = 0.0    # auto-computed Y view limits (p1–p99)
        self._view_hi     = 1.0
        self._dragging    = None       # 'lo' | 'hi' | 'both' | None
        self._drag_anchor = 0.0        # event.ydata - vmin at drag start ('both')
        self._drag_span   = 0.0        # vmax - vmin at drag start ('both')

        self._line_lo      = None      # axhline for vmin
        self._line_hi      = None      # axhline for vmax
        self._span         = None      # axhspan between vmin and vmax
        self._profile_fill = None      # PolyCollection from fill_betweenx
        self._profile_line = None      # Line2D step outline
        self._bar_color    = _BAR

        # Stored theme colors — updated by apply_mpl_theme(); used in set_data/clear
        # so that re-draws after a theme switch use the correct palette.
        self._ax_fg    = _FG
        self._ax_bg    = _BG_AX
        self._ax_spine = '#444444'

        # Axes: wide left margin for intensity tick labels
        self._fig = Figure(facecolor=_BG_FIG)
        self._ax  = self._fig.add_axes([0.42, 0.06, 0.53, 0.90])
        _style_ax(self._ax)
        self._ax.set_xticks([])
        self._ax.yaxis.set_major_locator(_ticker.MaxNLocator(nbins=5, integer=True))
        self._ax.yaxis.set_major_formatter(_ticker.ScalarFormatter())
        self._ax.yaxis.get_major_formatter().set_scientific(False)
        self._ax.yaxis.get_major_formatter().set_useOffset(False)

        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.mpl_connect('scroll_event',         self._on_scroll)
        self._canvas.mpl_connect('button_press_event',   self._on_press)
        self._canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self._canvas.mpl_connect('button_release_event', self._on_release)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self._canvas)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_data(self, image, vmin=None, vmax=None, callback=None):
        """Rebuild histogram from *image* and draw.

        Parameters
        ----------
        image : ndarray
            Pixel array.  NaN and exact zeros are excluded.
        vmin, vmax : float, optional
            Initial contrast limits.  Defaults to p2 / p98 of the data.
        callback : callable(vmin, vmax), optional
            Called whenever either line is dragged.
        """
        if callback is not None:
            self._callback = callback

        finite = image[np.isfinite(image)]
        finite = finite[finite != 0]
        if len(finite) < 10:
            return

        self._dmin = float(finite.min())
        self._dmax = float(finite.max())
        if self._dmax <= self._dmin:
            return

        self._vmin = vmin if vmin is not None else float(np.percentile(finite, 2))
        self._vmax = vmax if vmax is not None else float(np.percentile(finite, 98))
        self._vmin = float(np.clip(self._vmin, self._dmin, self._dmax))
        self._vmax = float(np.clip(self._vmax, self._dmin, self._dmax))

        # Histogram range: p0.1–p99.9 so all bins land in the interesting
        # region regardless of detector bit depth or stray hot pixels.
        h_lo = float(np.percentile(finite, 0.1))
        h_hi = float(np.percentile(finite, 99.9))
        if h_hi <= h_lo:
            h_lo, h_hi = self._dmin, self._dmax
        counts, bins = np.histogram(finite, bins=_N_BINS, range=(h_lo, h_hi))

        # Auto Y-view: p1–p99 with a small margin so vmin/vmax are visible
        p1  = float(np.percentile(finite, 1))
        p99 = float(np.percentile(finite, 99))
        margin = (p99 - p1) * 0.05
        self._view_lo = max(self._dmin, p1 - margin)
        self._view_hi = min(self._dmax, p99 + margin)
        if self._view_hi <= self._view_lo:
            self._view_lo, self._view_hi = self._dmin, self._dmax

        self._ax.clear()
        _style_ax(self._ax, fg=self._ax_fg, bg=self._ax_bg, spine=self._ax_spine)
        self._ax.set_xticks([])
        self._ax.yaxis.set_major_locator(_ticker.MaxNLocator(nbins=5, integer=True))
        self._ax.yaxis.set_major_formatter(_ticker.ScalarFormatter())
        self._ax.yaxis.get_major_formatter().set_scientific(False)
        self._ax.yaxis.get_major_formatter().set_useOffset(False)

        # Build step-function arrays (Y = intensity bin edges, X = counts)
        y_step = np.repeat(bins, 2)[1:-1]   # [b0,b1, b1,b2, ..., b(n-1),bn]
        x_step = np.repeat(counts, 2)        # [c0,c0, c1,c1, ...]

        self._profile_fill = self._ax.fill_betweenx(
            y_step, 0, x_step,
            color=self._bar_color, alpha=0.55, linewidth=0)
        self._profile_line, = self._ax.plot(
            x_step, y_step,
            color=self._bar_color, lw=0.9)

        self._ax.set_xlim(0, counts.max() * 1.08)
        self._ax.set_ylim(self._view_lo, self._view_hi)

        # Draggable span
        self._span = self._ax.axhspan(
            self._vmin, self._vmax, color=_XH, alpha=0.12, zorder=2)

        # vmin / vmax lines
        self._line_lo = self._ax.axhline(
            self._vmin, color=_XH,  lw=1.8, ls='-', alpha=0.9, zorder=5)
        self._line_hi = self._ax.axhline(
            self._vmax, color=_XH2, lw=1.8, ls='-', alpha=0.9, zorder=5)

        self._canvas.draw_idle()

    def get_clim(self):
        """Return current (vmin, vmax)."""
        return self._vmin, self._vmax

    def clear(self):
        """Clear the histogram and reset contrast lines to default state."""
        self._ax.clear()
        _style_ax(self._ax, fg=self._ax_fg, bg=self._ax_bg, spine=self._ax_spine)
        self._ax.set_xticks([])
        self._line_lo      = None
        self._line_hi      = None
        self._span         = None
        self._profile_fill = None
        self._profile_line = None
        self._canvas.draw_idle()

    def apply_mpl_theme(self, colors):
        """Update histogram colours to match a new matplotlib theme dict."""
        self._ax_fg    = colors.get('fg',    self._ax_fg)
        self._ax_bg    = colors.get('ax',    self._ax_bg)
        self._ax_spine = colors.get('spine', self._ax_spine)
        self._fig.set_facecolor(colors['fig'])
        _style_ax(self._ax, fg=self._ax_fg, bg=self._ax_bg, spine=self._ax_spine)
        self._bar_color = colors.get('hist_bar', _BAR)
        if self._profile_fill is not None:
            self._profile_fill.set_facecolor(self._bar_color)
        if self._profile_line is not None:
            self._profile_line.set_color(self._bar_color)
        self._canvas.draw_idle()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tol(self):
        """Drag tolerance: 3% of the visible y-range."""
        lo, hi = self._ax.get_ylim()
        return abs(hi - lo) * 0.03

    def _which_line(self, y):
        """Return 'lo', 'hi', or None based on proximity to vmin/vmax."""
        if self._line_lo is None:
            return None
        tol  = self._tol()
        d_lo = abs(y - self._vmin)
        d_hi = abs(y - self._vmax)
        if d_lo < tol and d_lo <= d_hi:
            return 'lo'
        if d_hi < tol:
            return 'hi'
        return None

    def _update_artists(self):
        """Sync line positions and span to current vmin/vmax."""
        if self._line_lo is not None:
            self._line_lo.set_ydata([self._vmin, self._vmin])
        if self._line_hi is not None:
            self._line_hi.set_ydata([self._vmax, self._vmax])
        if self._span is not None:
            self._span.remove()
        self._span = self._ax.axhspan(
            self._vmin, self._vmax, color=_XH, alpha=0.12, zorder=2)

    # ── Scroll zoom (Y-axis) ──────────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.inaxes != self._ax:
            return
        lo, hi = self._ax.get_ylim()
        cy = event.ydata if event.ydata is not None else (lo + hi) / 2
        factor = _ZOOM_FACTOR if event.step < 0 else 1.0 / _ZOOM_FACTOR
        self._ax.set_ylim(cy + (lo - cy) * factor,
                          cy + (hi - cy) * factor)
        self._canvas.draw_idle()

    # ── Mouse interaction ─────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes != self._ax or event.ydata is None:
            return
        if event.button == 3:                        # right-click → reset view
            self._ax.set_ylim(self._view_lo, self._view_hi)
            self._canvas.draw_idle()
            return
        which = self._which_line(event.ydata)
        if which in ('lo', 'hi'):
            self._dragging = which
        elif self._vmin <= event.ydata <= self._vmax:
            self._dragging    = 'both'
            self._drag_anchor = event.ydata - self._vmin
            self._drag_span   = self._vmax - self._vmin

    def _on_motion(self, event):
        if self._dragging is None:
            return
        if event.inaxes != self._ax or event.ydata is None:
            return

        val = float(np.clip(event.ydata, self._dmin, self._dmax))

        if self._dragging == 'lo':
            self._vmin = min(val, self._vmax - 1e-9)

        elif self._dragging == 'hi':
            self._vmax = max(val, self._vmin + 1e-9)

        else:   # 'both' — shift window preserving span
            new_vmin = val - self._drag_anchor
            new_vmax = new_vmin + self._drag_span
            if new_vmin < self._dmin:
                new_vmin = self._dmin
                new_vmax = self._dmin + self._drag_span
            if new_vmax > self._dmax:
                new_vmax = self._dmax
                new_vmin = self._dmax - self._drag_span
            self._vmin = float(np.clip(new_vmin, self._dmin, self._dmax))
            self._vmax = float(np.clip(new_vmax, self._dmin, self._dmax))

        self._update_artists()
        self._canvas.draw_idle()
        if self._callback:
            self._callback(self._vmin, self._vmax)

    def _on_release(self, event):
        self._dragging = None
